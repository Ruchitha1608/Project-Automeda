"""
Imaging Module with GradCAM Explainability
Uses ResNet50 for breast cancer image classification
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


class SimpleGradCAM:
    """Simple GradCAM implementation that works reliably"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
    def _get_activations_hook(self, module, input, output):
        self.activations = output.detach()
        
    def _get_gradients_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class):
        """Generate GradCAM heatmap"""
        # Register hooks
        handle_forward = self.target_layer.register_forward_hook(self._get_activations_hook)
        handle_backward = self.target_layer.register_full_backward_hook(self._get_gradients_hook)
        
        try:
            # Forward pass
            self.model.zero_grad()
            output = self.model(input_tensor)
            
            # Backward pass for target class
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Calculate GradCAM
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            
            # Normalize
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.squeeze().cpu().numpy()
            
        finally:
            # Remove hooks
            handle_forward.remove()
            handle_backward.remove()


class BreastCancerClassifier:
    """ResNet50-based breast cancer image classifier with GradCAM"""
    
    def __init__(self):
        import os
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Trained model from train_imaging_model.py
        trained_model_path = os.path.join(base_dir, 'models', 'imaging_model_trained.pth')
        
        model_loaded = False
        
        # Load trained model with custom architecture
        if os.path.exists(trained_model_path):
            try:
                checkpoint = torch.load(trained_model_path, map_location=torch.device('cpu'))
                
                # Build model with custom classifier (matches training script)
                self.model = models.resnet50(weights=None)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, 2)
                )
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                model_loaded = True
                auc = checkpoint.get('val_auc', 'N/A')
                print(f"✅ Loaded trained imaging model (AUC: {auc})")
                
            except Exception as e:
                print(f"⚠️ Could not load trained model: {e}")
                model_loaded = False
        
        # Fallback: use pretrained ImageNet weights
        if not model_loaded:
            print("⚠️ No trained model found. Using pretrained ImageNet weights.")
            print("   Run: python train_imaging_model.py to train a proper model")
            self.model = models.resnet50(weights='IMAGENET1K_V1')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 2)
            self._simulate_training()
        
        # ImageNet normalization stats
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _simulate_training(self):
        """Simulate trained weights for demo purposes when no model is available"""
        # Adjust final layer weights to create demo predictions
        # NOTE: This is NOT a proper trained model - just for demo
        with torch.no_grad():
            # For simple Linear fc layer
            if isinstance(self.model.fc, nn.Linear):
                self.model.fc.weight[1] *= 1.5
                self.model.fc.bias[1] += 0.8
                self.model.fc.bias[0] -= 0.5
    
    def predict_image(self, img_pil):
        """
        Predict breast cancer from histopathology image
        
        Args:
            img_pil (PIL.Image): Input image
        
        Returns:
            tuple: (prediction_class, confidence, heatmap_overlay_pil)
                - prediction_class: str ("Benign" or "Malignant")
                - confidence: float (0-1)
                - heatmap_overlay_pil: PIL.Image with GradCAM overlay
        """
        # Store original image
        original_img = img_pil.copy()
        
        # Preprocess image
        img_tensor = self.transform(img_pil).unsqueeze(0)
        
        # Forward pass for prediction (no gradients needed)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        pred_class = "Benign" if predicted.item() == 0 else "Malignant"
        conf_score = confidence.item()
        
        # Generate GradCAM heatmap
        heatmap_overlay = self._generate_gradcam(img_tensor, original_img, predicted.item())
        
        return pred_class, conf_score, heatmap_overlay
    
    def _generate_gradcam(self, img_tensor, original_img, target_class):
        """Generate GradCAM heatmap overlay using simple implementation"""
        try:
            # Use our simple GradCAM implementation
            cam_generator = SimpleGradCAM(self.model, self.model.layer4)
            
            # Create tensor with gradients enabled
            img_tensor_grad = img_tensor.clone().detach().requires_grad_(True)
            
            # Generate heatmap
            with torch.enable_grad():
                cam = cam_generator.generate(img_tensor_grad, target_class)
            
            # Normalize to 0-255
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = (cam * 255).astype(np.uint8)
            
            # Resize to original image size
            original_size = original_img.size
            cam_resized = cv2.resize(cam, original_size)
            
            # Apply colormap (red heatmap)
            heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Convert original PIL to numpy
            original_np = np.array(original_img.convert('RGB'))
            
            # Overlay heatmap on original image (60% original, 40% heatmap)
            overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
            
            # Convert back to PIL
            overlay_pil = Image.fromarray(overlay)
            
            return overlay_pil
            
        except Exception as e:
            print(f"GradCAM error: {e}")
            # Return original image if GradCAM fails
            return original_img


# Global model instance
_model_instance = None


def get_model():
    """Get or create global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = BreastCancerClassifier()
    return _model_instance


def predict_image(img_pil):
    """
    Convenience function for image prediction
    
    Args:
        img_pil (PIL.Image): Input histopathology image
    
    Returns:
        tuple: (pred_class, confidence, heatmap_overlay_pil)
    """
    model = get_model()
    return model.predict_image(img_pil)


if __name__ == "__main__":
    # Test the module
    from PIL import Image
    import numpy as np
    
    # Create a dummy test image
    test_img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
    
    pred_class, confidence, heatmap = predict_image(test_img)
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Heatmap shape: {np.array(heatmap).shape}")
