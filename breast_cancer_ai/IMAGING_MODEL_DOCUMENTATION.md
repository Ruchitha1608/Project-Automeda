# 🔬 Breast Cancer Histopathology Classification
## Complete Model Documentation

**Report Generated:** February 6, 2026  
**Model Version:** 1.0  
**Dataset:** BreakHis 400x Magnification  

---

## 📋 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Dataset Information](#2-dataset-information)
3. [Data Split](#3-data-split)
4. [Model Architecture](#4-model-architecture)
5. [Training Configuration](#5-training-configuration)
6. [Training Process](#6-training-process)
7. [Performance Metrics](#7-performance-metrics)
8. [Confusion Matrix](#8-confusion-matrix)
9. [Clinical Interpretation](#9-clinical-interpretation)
10. [Files & Usage](#10-files--usage)

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 94.87% |
| **AUC-ROC** | 0.9872 |
| **Sensitivity** | 94.59% |
| **Specificity** | 95.45% |
| **F1-Score** | 0.9615 |

**Key Findings:**
- Model correctly identifies **94.6% of cancer cases** (Sensitivity)
- Model correctly identifies **95.5% of benign cases** (Specificity)
- Only **10 cancer cases missed** out of 185 malignant samples in test set
- Only **4 false alarms** (benign marked as malignant)

✅ **Recommendation:** Model suitable for clinical decision support (should be used alongside expert pathologist review)

---

## 2. Dataset Information

### BreakHis Dataset (400x Magnification)

| Category | Count | Percentage |
|----------|-------|------------|
| **Total Images** | 1,820 | 100% |
| Benign | 588 | 32.3% |
| Malignant | 1,232 | 67.7% |

**Class Imbalance Ratio:** 1:2.10 (Benign:Malignant)

### Dataset Source
- **Name:** Breast Cancer Histopathological Database (BreakHis)
- **Magnification:** 400x
- **Image Format:** PNG
- **Original Resolution:** Variable (resized to 224x224)

---

## 3. Data Split

### Stratified Split Configuration

| Split | Images | Percentage | Benign | Malignant |
|-------|--------|------------|--------|-----------|
| **Training** | 1,274 | 70.0% | 412 | 862 |
| **Validation** | 273 | 15.0% | 88 | 185 |
| **Test** | 273 | 15.0% | 88 | 185 |
| **TOTAL** | 1,820 | 100% | 588 | 1,232 |

### Split Strategy
- **Method:** Stratified sampling (preserves class proportions)
- **Random Seed:** 42 (for reproducibility)
- **Purpose:**
  - Training set: Model learning
  - Validation set: Hyperparameter tuning & early stopping
  - Test set: Final unbiased evaluation

---

## 4. Model Architecture

### Base Model: ResNet50

```
ResNet50 (Transfer Learning)
├── Pretrained: ImageNet V2 (1.2M images, 1000 classes)
├── 50 layers deep
└── Feature extractor layers (frozen during initial training)
```

### Custom Classifier Head

```
Layer 1: Dropout(p=0.5)
    ↓
Layer 2: Linear(2048 → 512) + ReLU activation
    ↓
Layer 3: BatchNorm1d(512)
    ↓
Layer 4: Dropout(p=0.3)
    ↓
Layer 5: Linear(512 → 2) → [Benign, Malignant]
```

### Parameter Count

| Type | Count |
|------|-------|
| Total Parameters | 24,559,170 |
| Trainable Parameters | 24,559,170 |
| Model Size | 94.0 MB |

---

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Image Size | 224 × 224 pixels |
| Batch Size | 32 |
| Learning Rate | 0.0001 |
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss (weighted) |
| Max Epochs | 20 |
| Early Stopping Patience | 5 epochs |
| Random Seed | 42 |

### Class Weights
Applied to handle imbalanced data (more malignant samples than benign).

---

## 6. Training Process

### Data Augmentation (Training Set Only)

| Augmentation | Parameter |
|--------------|-----------|
| Random Horizontal Flip | p=0.5 |
| Random Vertical Flip | p=0.5 |
| Random Rotation | ±15° |
| Color Jitter | brightness=0.2, contrast=0.2 |
| Normalization | ImageNet mean/std |

**ImageNet Normalization:**
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

### Learning Rate Schedule
- **Scheduler:** ReduceLROnPlateau
- **Factor:** 0.5 (reduce LR by half)
- **Patience:** 3 epochs

### Early Stopping
- **Monitor:** Validation AUC-ROC
- **Patience:** 5 epochs without improvement
- **Best Model:** Saved when validation AUC improves

### Training Results
| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 96.70% |
| Best Validation AUC-ROC | 0.9974 |

---

## 7. Performance Metrics

### Overall Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.87% |
| **AUC-ROC** | 0.9872 |
| **F1-Score** | 0.9615 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 89.36% | 95.45% | 92.31% | 88 |
| Malignant | 97.77% | 94.59% | 96.15% | 185 |
| **Weighted Avg** | 95.06% | 94.87% | 94.91% | 273 |

### Clinical Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **Sensitivity** | 94.59% | Cancer detection rate (True Positive Rate) |
| **Specificity** | 95.45% | Correct benign identification (True Negative Rate) |
| **PPV (Precision)** | 97.77% | When model predicts malignant, it's correct 97.77% of time |
| **NPV** | 89.36% | When model predicts benign, it's correct 89.36% of time |

---

## 8. Confusion Matrix

```
                    PREDICTED
                Benign    Malignant
            ┌──────────┬──────────┐
   Benign   │    84    │     4    │  (TN=84, FP=4)
 A ─────────├──────────┼──────────┤
 C Malignant│    10    │   175    │  (FN=10, TP=175)
 T          └──────────┴──────────┘
 U
 A              ↑            ↑
 L         Pred Benign  Pred Malignant
```

### Detailed Breakdown

| Outcome | Count | Description |
|---------|-------|-------------|
| **True Negatives (TN)** | 84 | Benign correctly identified as Benign |
| **True Positives (TP)** | 175 | Malignant correctly identified as Malignant |
| **False Positives (FP)** | 4 | Benign incorrectly flagged as Malignant |
| **False Negatives (FN)** | 10 | Malignant incorrectly classified as Benign |

### Confusion Matrix Percentages

| | Predicted Benign | Predicted Malignant | Total |
|---|---|---|---|
| **Actual Benign** | 95.45% (84/88) | 4.55% (4/88) | 88 |
| **Actual Malignant** | 5.41% (10/185) | 94.59% (175/185) | 185 |

---

## 9. Clinical Interpretation

### ⚠️ Critical Consideration: False Negatives

**False Negatives (FN = 10)** represent missed cancer cases. In clinical settings, these are the most dangerous errors as they could lead to:
- Delayed treatment
- Disease progression
- Worse patient outcomes

### Risk Assessment

| Error Type | Count | Risk Level | Clinical Impact |
|------------|-------|------------|-----------------|
| False Negatives | 10 | **HIGH** | Missed cancer diagnosis |
| False Positives | 4 | MODERATE | Unnecessary follow-up tests |

### Recommendations

1. **Use as Decision Support Tool**
   - Model predictions should complement, not replace, pathologist review
   - High-confidence predictions (>95%) can help prioritize cases

2. **Threshold Adjustment for High Sensitivity**
   - Current threshold: 0.5
   - For screening: Consider lowering threshold to catch more cancers
   - Trade-off: More false positives but fewer missed cancers

3. **Confidence-Based Routing**
   - High confidence (>90%): Automated pre-screening
   - Medium confidence (60-90%): Priority pathologist review
   - Low confidence (<60%): Immediate expert review

---

## 10. Files & Usage

### Project Structure

```
breast_cancer_ai/
├── models/
│   └── imaging_model_trained.pth    # Trained model (94.0 MB)
├── modules/
│   └── imaging.py                   # Inference module
├── notebooks/
│   └── Imaging_Model_Training.ipynb # Training notebook (Colab)
├── train_imaging_model.py           # Training script
├── generate_imaging_report.py       # This report generator
└── IMAGING_MODEL_DOCUMENTATION.md   # This documentation
```

### Usage

#### Training from Scratch
```bash
python train_imaging_model.py
```

#### Generate Report
```bash
python generate_imaging_report.py
```

#### Inference in Code
```python
from modules.imaging import predict_image
from PIL import Image

img = Image.open("path/to/histopathology_image.png")
prediction, confidence, heatmap = predict_image(img)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.1%}")
```

### Model Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': ...,  # Trained weights
    'val_auc': 0.9974,        # Best validation AUC
    'val_acc': 0.9670,        # Best validation accuracy
    'config': {
        'batch_size': 32,
        'learning_rate': 0.0001,
        'num_epochs': 20,
        ...
    }
}
```

---

## 📊 Summary Visualization

```
┌────────────────────────────────────────────────────────────────┐
│                 MODEL PERFORMANCE OVERVIEW                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Dataset: 1,820 images (588 Benign, 1,232 Malignant)         │
│   Test Set: 273 images (88 Benign, 185 Malignant)             │
│                                                                │
│   ┌─────────────────────────────────────────────────────┐     │
│   │  ACCURACY    ████████████████████          94.87%   │     │
│   │  AUC-ROC     █████████████████████         98.72%   │     │
│   │  SENSITIVITY ████████████████████          94.59%   │     │
│   │  SPECIFICITY ████████████████████          95.45%   │     │
│   └─────────────────────────────────────────────────────┘     │
│                                                                │
│   Errors: 10 missed cancers (FN) + 4 false alarms (FP)        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0  
**Last Updated:** February 6, 2026  
**Author:** Breast Cancer AI System  
