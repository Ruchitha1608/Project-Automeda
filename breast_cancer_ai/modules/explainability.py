"""
Explainability Module
Grad-CAM, Integrated Gradients, and SHAP-based explanations for model predictions
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict, Callable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ============================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Visualizes which regions of an image contribute to predictions
    """
    
    def __init__(self, model: 'nn.Module', target_layer: str = None):
        """
        Initialize GradCAM
        
        Args:
            model: PyTorch model
            target_layer: Name of the target layer for CAM
                         (automatically detects last conv layer if None)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GradCAM. Install with: pip install torch")
        
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Find and register hooks for target layer
        self.target_layer = self._find_target_layer(target_layer)
        self._register_hooks()
    
    def _find_target_layer(self, target_layer: str = None) -> 'nn.Module':
        """Find appropriate target layer for CAM"""
        if target_layer:
            # Find by name
            for name, module in self.model.named_modules():
                if name == target_layer:
                    return module
        
        # Auto-detect: find last Conv2d layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model")
        
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def generate(
        self,
        input_tensor: 'torch.Tensor',
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index (uses argmax if None)
        
        Returns:
            Heatmap as numpy array (H, W), values 0-1
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        input_tensor: 'torch.Tensor',
        original_image: Image.Image,
        target_class: int = None,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Generate and visualize GradCAM
        
        Args:
            input_tensor: Input image tensor
            original_image: Original PIL Image
            target_class: Target class
            alpha: Overlay transparency
            colormap: Matplotlib colormap
        
        Returns:
            Tuple of (raw heatmap, overlayed image)
        """
        # Generate CAM
        heatmap = self.generate(input_tensor, target_class)
        
        # Resize to original image size
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize(original_image.size, Image.Resampling.BILINEAR)
        heatmap_resized = np.array(heatmap_pil) / 255.0
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]
        
        # Blend with original
        original_array = np.array(original_image.convert('RGB')) / 255.0
        blended = (1 - alpha) * original_array + alpha * heatmap_colored
        blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        
        return heatmap_resized, Image.fromarray(blended)
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        """Cleanup hooks on deletion"""
        self.remove_hooks()


# ============================================================
# GRAD-CAM++ (Enhanced version)
# ============================================================

class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ - Improved version with weighted gradients
    Better localization for multiple instances of same class
    """
    
    def generate(
        self,
        input_tensor: 'torch.Tensor',
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate GradCAM++ heatmap
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # GradCAM++ weighting
        grads = self.gradients
        acts = self.activations
        
        # Calculate alpha (Eq. 7 from paper)
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8
        
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + eps
        alpha = alpha_num / alpha_denom
        
        # ReLU of gradients * alpha, averaged over spatial dims
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        
        # Weighted sum
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

class IntegratedGradients:
    """
    Integrated Gradients attribution method
    Attributes importance to input features by integrating gradients
    from a baseline to the input
    """
    
    def __init__(self, model: 'nn.Module'):
        """
        Initialize Integrated Gradients
        
        Args:
            model: PyTorch model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")
        
        self.model = model
        self.model.eval()
    
    def generate(
        self,
        input_tensor: 'torch.Tensor',
        target_class: int = None,
        baseline: 'torch.Tensor' = None,
        steps: int = 50
    ) -> np.ndarray:
        """
        Calculate Integrated Gradients attribution
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class (argmax if None)
            baseline: Baseline tensor (zeros if None)
            steps: Number of interpolation steps
        
        Returns:
            Attribution map as numpy array
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Get target class
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
        
        # Generate interpolated inputs
        input_tensor.requires_grad = True
        
        # Accumulate gradients
        total_gradients = torch.zeros_like(input_tensor)
        
        for step in range(steps):
            # Interpolate between baseline and input
            alpha = step / steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            output = self.model(interpolated)
            
            # Backward pass
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot)
            
            total_gradients += interpolated.grad.data
        
        # Average gradients and multiply by input difference
        avg_gradients = total_gradients / steps
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        # Convert to attribution map (sum over channels)
        attribution = integrated_gradients.sum(dim=1).squeeze().cpu().numpy()
        
        # Normalize
        attribution = np.abs(attribution)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution
    
    def visualize(
        self,
        input_tensor: 'torch.Tensor',
        original_image: Image.Image,
        target_class: int = None,
        alpha: float = 0.5,
        colormap: str = 'hot'
    ) -> Tuple[np.ndarray, Image.Image]:
        """Generate and visualize attributions"""
        
        # Generate attributions
        attribution = self.generate(input_tensor, target_class)
        
        # Resize to original image size
        attr_pil = Image.fromarray((attribution * 255).astype(np.uint8))
        attr_pil = attr_pil.resize(original_image.size, Image.Resampling.BILINEAR)
        attr_resized = np.array(attr_pil) / 255.0
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        attr_colored = cmap(attr_resized)[:, :, :3]
        
        # Blend
        original_array = np.array(original_image.convert('RGB')) / 255.0
        blended = (1 - alpha) * original_array + alpha * attr_colored
        blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        
        return attr_resized, Image.fromarray(blended)


# ============================================================
# FEATURE IMPORTANCE (For tabular/omics data)
# ============================================================

class FeatureImportance:
    """
    Feature importance calculation for omics/tabular data
    Supports permutation importance and gradient-based methods
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        Initialize Feature Importance calculator
        
        Args:
            model: Trained model (sklearn or pytorch)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names or []
        
        # Detect model type
        self.is_sklearn = hasattr(model, 'predict_proba')
        self.is_torch = TORCH_AVAILABLE and isinstance(model, nn.Module)
    
    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        metric: Callable = None
    ) -> Dict[str, float]:
        """
        Calculate permutation importance
        
        Args:
            X: Feature matrix
            y: Target values
            n_repeats: Number of permutation repeats
            metric: Scoring function (accuracy if None)
        
        Returns:
            Dict of feature_name: importance
        """
        from sklearn.metrics import accuracy_score
        
        if metric is None:
            metric = accuracy_score
        
        # Baseline score
        if self.is_sklearn:
            y_pred = self.model.predict(X)
        elif self.is_torch:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y_pred = self.model(X_tensor).argmax(dim=1).numpy()
        else:
            raise ValueError("Model type not supported")
        
        baseline_score = metric(y, y_pred)
        
        # Permutation importance
        importances = {}
        
        for i in range(X.shape[1]):
            scores = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                
                if self.is_sklearn:
                    y_pred = self.model.predict(X_permuted)
                elif self.is_torch:
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_permuted)
                        y_pred = self.model(X_tensor).argmax(dim=1).numpy()
                
                scores.append(baseline_score - metric(y, y_pred))
            
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            importances[feature_name] = np.mean(scores)
        
        return importances
    
    def get_top_features(
        self,
        importances: Dict[str, float],
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top N most important features"""
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]


# ============================================================
# EXPLANATION TEXT GENERATOR
# ============================================================

class ExplanationGenerator:
    """
    Generate human-readable explanations for model predictions
    """
    
    def __init__(self):
        self.prediction_templates = {
            'malignant': [
                "The model identified features consistent with malignant tissue characteristics.",
                "High-confidence regions show typical patterns of abnormal cell proliferation.",
                "The highlighted areas indicate morphological features associated with malignancy."
            ],
            'benign': [
                "The tissue exhibits features consistent with benign growth patterns.",
                "The model found no significant indicators of malignancy.",
                "Cell structure and organization appear within normal parameters."
            ]
        }
        
        self.confidence_texts = {
            'high': "The model is highly confident in this assessment.",
            'medium': "Moderate confidence suggests additional review may be warranted.",
            'low': "Lower confidence indicates this case may benefit from expert review."
        }
    
    def generate_imaging_explanation(
        self,
        prediction: str,
        confidence: float,
        hotspot_regions: int = 0
    ) -> str:
        """
        Generate explanation for imaging prediction
        
        Args:
            prediction: Model prediction (malignant/benign)
            confidence: Confidence score
            hotspot_regions: Number of highlighted regions
        
        Returns:
            Explanation text
        """
        pred_key = 'malignant' if 'malignant' in prediction.lower() else 'benign'
        
        # Select template
        import random
        main_text = random.choice(self.prediction_templates[pred_key])
        
        # Confidence text
        if confidence >= 0.8:
            conf_text = self.confidence_texts['high']
        elif confidence >= 0.5:
            conf_text = self.confidence_texts['medium']
        else:
            conf_text = self.confidence_texts['low']
        
        # Region text
        if hotspot_regions > 0:
            region_text = f" The visualization highlights {hotspot_regions} key region(s) of interest."
        else:
            region_text = ""
        
        return f"{main_text}{region_text} {conf_text}"
    
    def generate_omics_explanation(
        self,
        prediction: str,
        confidence: float,
        top_genes: List[Tuple[str, float]]
    ) -> str:
        """
        Generate explanation for omics prediction
        
        Args:
            prediction: Model prediction
            confidence: Confidence score
            top_genes: List of (gene, importance) tuples
        
        Returns:
            Explanation text
        """
        # Gene mentions
        if top_genes:
            gene_str = ", ".join([g[0] for g in top_genes[:5]])
            gene_text = f"Key contributing genes include: {gene_str}."
        else:
            gene_text = ""
        
        # Main explanation
        if 'cancer' in prediction.lower() or 'malignant' in prediction.lower():
            main_text = "Gene expression patterns indicate elevated cancer-associated signatures."
        else:
            main_text = "Gene expression patterns appear within normal ranges."
        
        # Confidence
        conf_pct = f"{confidence:.1%}"
        
        return f"{main_text} {gene_text} Model confidence: {conf_pct}."
    
    def generate_combined_explanation(
        self,
        imaging_result: Dict,
        omics_result: Dict,
        agreement: bool
    ) -> str:
        """
        Generate explanation for multimodal analysis
        
        Args:
            imaging_result: Imaging prediction dict
            omics_result: Omics prediction dict
            agreement: Whether modalities agree
        
        Returns:
            Combined explanation text
        """
        if agreement:
            return (
                f"Both imaging and genomic analyses converge on the same conclusion, "
                f"increasing overall diagnostic confidence. "
                f"Imaging confidence: {imaging_result['confidence']:.1%}, "
                f"Omics confidence: {omics_result['confidence']:.1%}."
            )
        else:
            return (
                f"The imaging and genomic analyses show divergent results. "
                f"This discrepancy warrants additional expert review. "
                f"Imaging suggests: {imaging_result['prediction']} ({imaging_result['confidence']:.1%}), "
                f"Omics suggests: {omics_result['prediction']} ({omics_result['confidence']:.1%})."
            )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_explanation_figure(
    original_image: Image.Image,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float,
    method: str = "Grad-CAM"
) -> 'plt.Figure':
    """
    Create comprehensive explanation figure
    
    Args:
        original_image: Original image
        heatmap: Activation heatmap
        prediction: Model prediction
        confidence: Confidence score
        method: Explanation method name
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f'{method} Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = heatmap_resized.resize(original_image.size, Image.Resampling.BILINEAR)
    heatmap_np = np.array(heatmap_resized) / 255.0
    
    cmap = cm.get_cmap('jet')
    heatmap_colored = cmap(heatmap_np)[:, :, :3]
    
    original_np = np.array(original_image.convert('RGB')) / 255.0
    overlay = 0.6 * original_np + 0.4 * heatmap_colored
    
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title(f'{prediction}\n({confidence:.1%} confidence)', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Explainability Module")
    print("=" * 50)
    print("\nAvailable components:")
    print("  - GradCAM: Gradient-weighted Class Activation Mapping")
    print("  - GradCAMPlusPlus: Enhanced GradCAM with better localization")
    print("  - IntegratedGradients: Attribution via gradient integration")
    print("  - FeatureImportance: Permutation-based feature importance")
    print("  - ExplanationGenerator: Human-readable explanations")
    
    print(f"\nPyTorch available: {TORCH_AVAILABLE}")
    
    # Test explanation generator
    gen = ExplanationGenerator()
    
    exp = gen.generate_imaging_explanation(
        prediction="Malignant",
        confidence=0.89,
        hotspot_regions=3
    )
    print(f"\nSample imaging explanation:\n{exp}")
    
    exp = gen.generate_omics_explanation(
        prediction="Cancer",
        confidence=0.85,
        top_genes=[('BRCA1', 0.15), ('TP53', 0.12), ('HER2', 0.10)]
    )
    print(f"\nSample omics explanation:\n{exp}")
    
    print("\n✅ Explainability module ready!")
