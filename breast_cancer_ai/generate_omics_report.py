#!/usr/bin/env python3
"""
Omics Model Report Generator

Generates a comprehensive report of the trained omics model performance.
Run this after training to produce formatted output of all metrics.

Usage:
    python generate_omics_report.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

def print_header(title):
    """Print formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)

def print_subheader(title):
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")

def print_metric(name, value, format_spec=".4f"):
    """Print a metric with formatting."""
    if isinstance(value, float):
        print(f"  {name}: {value:{format_spec}}")
    else:
        print(f"  {name}: {value}")

def print_confusion_matrix(tn, fp, fn, tp):
    """Print a visual confusion matrix."""
    print("\n  Confusion Matrix:")
    print("  ┌─────────────────────────────────────┐")
    print("  │           CONFUSION MATRIX          │")
    print("  ├─────────────┬───────────┬───────────┤")
    print("  │             │ Pred: Neg │ Pred: Pos │")
    print("  ├─────────────┼───────────┼───────────┤")
    print(f"  │ Actual: Neg │   {tn:^5}   │   {fp:^5}   │")
    print("  │ (Normal)    │   (TN)    │   (FP)    │")
    print("  ├─────────────┼───────────┼───────────┤")
    print(f"  │ Actual: Pos │   {fn:^5}   │   {tp:^5}   │")
    print("  │ (Cancer)    │   (FN)    │   (TP)    │")
    print("  └─────────────┴───────────┴───────────┘")

def generate_report():
    """Generate comprehensive model report."""
    
    # Find model file
    model_paths = [
        'models/omics_model_v3.pkl',
        'models/omics_model.pkl'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ Error: No trained omics model found!")
        print("   Run: python train_omics_model.py")
        sys.exit(1)
    
    # Load model
    print(f"\n📂 Loading model from: {model_path}")
    model_data = joblib.load(model_path)
    
    # Report header
    print_header("OMICS MODEL PERFORMANCE REPORT")
    print(f"\n  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model Version: {model_data.get('version', 'unknown')}")
    print(f"  Training Date: {model_data.get('training_date', 'unknown')}")
    
    # Model configuration
    print_header("MODEL CONFIGURATION")
    
    model = model_data['model']
    print_subheader("Algorithm: Random Forest")
    print_metric("Number of Trees", model.n_estimators)
    print_metric("Maximum Depth", model.max_depth)
    print_metric("Min Samples Split", model.min_samples_split)
    print_metric("Min Samples Leaf", model.min_samples_leaf)
    print_metric("Class Weight", model.class_weight)
    
    # Feature information
    print_header("FEATURE INFORMATION")
    
    feature_names = model_data.get('feature_names', [])
    selected_features = model_data.get('selected_features', [])
    
    print_metric("Total Genes", len(feature_names))
    print_metric("Selected Features", len(selected_features))
    
    print_subheader("Gene Panel")
    for i, gene in enumerate(feature_names, 1):
        marker = "✓" if gene in selected_features else " "
        print(f"  [{marker}] {i:2}. {gene}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print_subheader("Top 10 Biomarkers (by importance)")
        
        importances = model.feature_importances_
        if len(importances) == len(selected_features):
            importance_df = pd.DataFrame({
                'Gene': selected_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'Gene': [f'Feature_{i}' for i in range(len(importances))],
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        
        print("\n  Rank | Gene       | Importance")
        print("  " + "-" * 35)
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"   {i:2}  | {row['Gene']:<10} | {row['Importance']:.4f}")
    
    # Performance metrics
    print_header("PERFORMANCE METRICS")
    
    metrics = model_data.get('metrics', {})
    
    print_subheader("Training Performance")
    print_metric("Accuracy", metrics.get('train_accuracy', metrics.get('accuracy', 'N/A')))
    
    print_subheader("Validation Performance")
    print_metric("Accuracy", metrics.get('val_accuracy', 'N/A'))
    
    print_subheader("Test Performance")
    test_acc = metrics.get('test_accuracy', metrics.get('accuracy', 'N/A'))
    print_metric("Accuracy", test_acc)
    print_metric("AUC-ROC", metrics.get('auc_roc', 'N/A'))
    
    print_subheader("Clinical Metrics")
    sensitivity = metrics.get('sensitivity', metrics.get('recall', 'N/A'))
    specificity = metrics.get('specificity', 'N/A')
    
    if isinstance(sensitivity, float):
        print_metric("Sensitivity (Recall)", f"{sensitivity:.2%}")
    else:
        print_metric("Sensitivity (Recall)", sensitivity)
        
    if isinstance(specificity, float):
        print_metric("Specificity", f"{specificity:.2%}")
    else:
        print_metric("Specificity", specificity)
    
    # Confusion matrix from metrics
    cm = metrics.get('confusion_matrix', None)
    if cm is not None:
        try:
            if hasattr(cm, 'ravel'):
                flat = cm.ravel()
                tn, fp, fn, tp = flat
            elif isinstance(cm, (list, tuple)) and len(cm) == 2:
                # 2D list format: [[tn, fp], [fn, tp]]
                tn, fp = cm[0]
                fn, tp = cm[1]
            elif isinstance(cm, (list, tuple)) and len(cm) == 4:
                tn, fp, fn, tp = cm
            else:
                raise ValueError(f"Unexpected confusion matrix format: {cm}")
            
            print_confusion_matrix(tn, fp, fn, tp)
            
            # Derived metrics
            print_subheader("Derived Metrics")
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            print_metric("PPV (Precision)", f"{ppv:.2%}")
            print_metric("NPV", f"{npv:.2%}")
        except Exception as e:
            print(f"\n  Confusion Matrix: Error parsing ({e})")
    
    # Summary
    print_header("SUMMARY")
    
    auc = metrics.get('auc_roc', 0)
    
    print("\n  ╔════════════════════════════════════════╗")
    print("  ║        MODEL PERFORMANCE SUMMARY       ║")
    print("  ╠════════════════════════════════════════╣")
    if isinstance(test_acc, float):
        print(f"  ║  Test Accuracy:     {test_acc:>6.2%}            ║")
    if isinstance(auc, float):
        print(f"  ║  AUC-ROC:           {auc:>6.4f}            ║")
    if isinstance(sensitivity, float):
        print(f"  ║  Sensitivity:       {sensitivity:>6.2%}            ║")
    if isinstance(specificity, float):
        print(f"  ║  Specificity:       {specificity:>6.2%}            ║")
    print("  ╚════════════════════════════════════════╝")
    
    # Clinical interpretation
    print_header("CLINICAL INTERPRETATION")
    
    if isinstance(sensitivity, float) and sensitivity == 1.0:
        print("\n  ✅ 100% SENSITIVITY: No cancer cases missed")
        print("     - Suitable for screening applications")
        print("     - All positive cases correctly identified")
    
    if isinstance(specificity, float) and specificity < 0.7:
        print(f"\n  ⚠️  {specificity:.1%} SPECIFICITY: Some false positives")
        print("     - Follow-up testing recommended for positive results")
        print("     - Trade-off for high sensitivity")
    
    print("\n" + "=" * 60)
    print(" END OF REPORT ".center(60))
    print("=" * 60 + "\n")

if __name__ == "__main__":
    generate_report()
