"""
🔬 Train ResNet50 for Breast Cancer Histopathology Classification
Run this script to train a proper model on your BreakHis dataset

Usage: python train_imaging_model.py
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'data_dir': 'data/Breakhis-400x',
    'model_save_path': 'models/imaging_model_trained.pth',
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'val_split': 0.15,
    'test_split': 0.15,
    'image_size': 224,
    'patience': 5,
    'seed': 42
}

# Set seeds
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# ============================================================
# DATA LOADING
# ============================================================

def load_data(data_dir):
    """Load image paths and labels."""
    data_dir = Path(data_dir)
    
    image_paths = []
    labels = []
    
    # Benign (label=0)
    benign_dir = data_dir / 'benign'
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img_path in benign_dir.glob(ext):
            image_paths.append(str(img_path))
            labels.append(0)
    
    # Malignant (label=1)
    malignant_dir = data_dir / 'malignant'
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img_path in malignant_dir.glob(ext):
            image_paths.append(str(img_path))
            labels.append(1)
    
    return image_paths, labels

# ============================================================
# DATASET
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ============================================================
# MODEL
# ============================================================

def create_model(pretrained=True):
    """Create ResNet50 model with custom classifier for breast cancer detection."""
    model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    
    return model

# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    
    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_probs)
    return running_loss / total, correct / total, auc

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("🔬 Breast Cancer Image Classification Training")
    print("="*60 + "\n")
    
    # Load data
    print("📂 Loading data...")
    image_paths, labels = load_data(CONFIG['data_dir'])
    print(f"   Total: {len(image_paths)} images")
    print(f"   Benign: {labels.count(0)}, Malignant: {labels.count(1)}")
    
    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        image_paths, labels, test_size=CONFIG['test_split'],
        random_state=CONFIG['seed'], stratify=labels
    )
    
    val_ratio = CONFIG['val_split'] / (1 - CONFIG['test_split'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        random_state=CONFIG['seed'], stratify=y_trainval
    )
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = BreastCancerDataset(X_train, y_train, train_transform)
    val_dataset = BreastCancerDataset(X_val, y_val, val_transform)
    test_dataset = BreastCancerDataset(X_test, y_test, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_model(pretrained=True).to(device)
    
    # Class weights
    class_counts = [y_train.count(0), y_train.count(1)]
    weights = torch.tensor([1.0/c for c in class_counts], dtype=torch.float32)
    weights = (weights / weights.sum()).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    print("\n🚀 Starting training...")
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            
            os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'config': CONFIG
            }, CONFIG['model_save_path'])
            print(f"   ✅ Best model saved! (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
    
    # Test evaluation
    print("\n" + "="*60)
    print("📊 TEST SET EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(CONFIG['model_save_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\n🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"📈 Test AUC-ROC:  {test_auc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"   TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"   FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"   Model saved to: {CONFIG['model_save_path']}")
    print("="*60)

if __name__ == '__main__':
    main()
