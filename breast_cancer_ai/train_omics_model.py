"""
🧬 Train Omics Model for Breast Cancer Classification
Complete ML Pipeline with Proper Preprocessing and Feature Selection

This script:
1. Loads breast cancer gene expression data (TCGA-BRCA)
2. Preprocesses: handles missing values, normalizes, log-transforms
3. Performs stratified train/val/test split (NO DATA LEAKAGE)
4. Feature selection on training data ONLY
5. Trains Random Forest classifier
6. Evaluates with full metrics and confusion matrix
7. Saves model and generates report

Usage: python train_omics_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, 
    classification_report, precision_score, recall_score, 
    f1_score, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Data
    'data_path': 'data/tcga_brca_key_genes.csv',  # TCGA breast cancer data
    'label_column': 'label',
    
    # Train/Val/Test split
    'test_size': 0.15,
    'val_size': 0.15,
    
    # Preprocessing
    'log_transform': True,  # Apply log2 transformation
    'normalize': True,       # Z-score normalization
    
    # Feature Selection
    'n_top_features': 20,   # Number of features to select
    'feature_method': 'f_classif',  # 'f_classif' or 'mutual_info'
    
    # Model
    'model_type': 'random_forest',  # 'random_forest', 'gradient_boosting', 'svm', 'logistic'
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'class_weight': 'balanced',  # Handle imbalanced data
    
    # Cross-validation
    'cv_folds': 5,
    
    # Output
    'model_save_path': 'models/omics_model_v3.pkl',
    'report_save_path': 'models/omics_training_report.json',
    
    # Reproducibility
    'random_state': 42
}

# Set seed
np.random.seed(CONFIG['random_state'])

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_and_preprocess_data(config):
    """
    Load and preprocess omics data following FR-OM-2 requirements.
    
    Steps:
    1. Load CSV data
    2. Separate features and labels
    3. Handle missing values
    4. Apply log transformation (if required)
    5. Check for consistent numeric format
    """
    print("\n" + "="*60)
    print("📂 STEP 1: DATA LOADING & PREPROCESSING")
    print("="*60)
    
    # Load data
    df = pd.read_csv(config['data_path'])
    print(f"\n✓ Loaded: {config['data_path']}")
    print(f"  Shape: {df.shape[0]} samples × {df.shape[1]} features")
    
    # Separate features and labels
    label_col = config['label_column']
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    feature_names = list(X.columns)
    print(f"  Features: {len(feature_names)}")
    print(f"  Sample features: {feature_names[:5]}...")
    
    # Clean labels - map to binary
    label_mapping = {'Cancer': 1, 'Normal': 0, 'Tumor': 1, 'Healthy': 0}
    y_clean = y.map(lambda x: label_mapping.get(x, 1 if 'cancer' in str(x).lower() else 0))
    
    print(f"\n📊 Label Distribution:")
    print(f"  Cancer:  {(y_clean == 1).sum()} ({(y_clean == 1).sum()/len(y_clean)*100:.1f}%)")
    print(f"  Normal:  {(y_clean == 0).sum()} ({(y_clean == 0).sum()/len(y_clean)*100:.1f}%)")
    
    # Convert to numpy
    X = X.values.astype(np.float64)
    y = y_clean.values
    
    # FR-OM-2: Handle missing values
    print(f"\n🔧 Preprocessing:")
    missing_count = np.isnan(X).sum()
    if missing_count > 0:
        print(f"  • Handling {missing_count} missing values (median imputation)")
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    else:
        print(f"  • No missing values found ✓")
    
    # FR-OM-2: Log transformation (common for gene expression data)
    if config['log_transform']:
        # Add small constant to avoid log(0)
        X_min = X.min()
        if X_min <= 0:
            X = X - X_min + 1
        X = np.log2(X + 1)
        print(f"  • Applied log2(x+1) transformation ✓")
    
    # Check data quality
    print(f"\n📈 Data Statistics (after preprocessing):")
    print(f"  • Min: {X.min():.4f}")
    print(f"  • Max: {X.max():.4f}")
    print(f"  • Mean: {X.mean():.4f}")
    print(f"  • Std: {X.std():.4f}")
    
    return X, y, feature_names

# ============================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================

def split_data(X, y, config):
    """
    Stratified train/validation/test split.
    
    CRITICAL: Maintains class proportions in all splits
    """
    print("\n" + "="*60)
    print("📂 STEP 2: DATA SPLITTING (STRATIFIED)")
    print("="*60)
    
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=y
    )
    
    # Second split: separate validation from training
    val_ratio = config['val_size'] / (1 - config['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=config['random_state'],
        stratify=y_trainval
    )
    
    total = len(y)
    print(f"\n┌─────────────────────────────────────────────────────────┐")
    print(f"│                    DATA SPLIT                           │")
    print(f"├─────────────────────────────────────────────────────────┤")
    print(f"│  Split        Samples   %      Cancer   Normal         │")
    print(f"│  ─────────────────────────────────────────────────────  │")
    print(f"│  Training     {len(X_train):>6}   {len(X_train)/total*100:>5.1f}%   {(y_train==1).sum():>6}   {(y_train==0).sum():>6}          │")
    print(f"│  Validation   {len(X_val):>6}   {len(X_val)/total*100:>5.1f}%   {(y_val==1).sum():>6}   {(y_val==0).sum():>6}          │")
    print(f"│  Test         {len(X_test):>6}   {len(X_test)/total*100:>5.1f}%   {(y_test==1).sum():>6}   {(y_test==0).sum():>6}          │")
    print(f"│  ─────────────────────────────────────────────────────  │")
    print(f"│  TOTAL        {total:>6}   100.0%   {(y==1).sum():>6}   {(y==0).sum():>6}          │")
    print(f"└─────────────────────────────────────────────────────────┘")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================
# FEATURE SCALING & SELECTION (FR-OM-3)
# ============================================================

def scale_and_select_features(X_train, X_val, X_test, y_train, feature_names, config):
    """
    FR-OM-2 & FR-OM-3: Normalize and select features.
    
    CRITICAL: Fit scaler and selector on TRAINING data ONLY!
    This prevents data leakage.
    """
    print("\n" + "="*60)
    print("🔬 STEP 3: FEATURE SCALING & SELECTION")
    print("="*60)
    
    # FR-OM-2: Normalize expression levels (fit on train only)
    print(f"\n📊 Normalization (Z-score):")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # FIT on train
    X_val_scaled = scaler.transform(X_val)          # TRANSFORM only
    X_test_scaled = scaler.transform(X_test)        # TRANSFORM only
    print(f"  • Fitted on training data ({len(X_train)} samples)")
    print(f"  • Applied to validation and test sets")
    
    # FR-OM-3: Feature selection
    print(f"\n🎯 Feature Selection:")
    print(f"  • Method: {config['feature_method']}")
    print(f"  • Selecting top {config['n_top_features']} features")
    
    if config['feature_method'] == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=config['n_top_features'])
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=config['n_top_features'])
    
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)  # FIT on train
    X_val_selected = selector.transform(X_val_scaled)                    # TRANSFORM only
    X_test_selected = selector.transform(X_test_scaled)                  # TRANSFORM only
    
    # Get selected feature names and scores
    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    feature_scores = selector.scores_[selected_mask]
    
    # FR-OM-3: Rank features by importance
    ranked_features = sorted(zip(selected_features, feature_scores), 
                            key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top 10 Selected Biomarkers (ranked by F-score):")
    print(f"  ───────────────────────────────────────")
    for i, (gene, score) in enumerate(ranked_features[:10], 1):
        print(f"  {i:>2}. {gene:<12} Score: {score:>10.2f}")
    
    return (X_train_selected, X_val_selected, X_test_selected, 
            scaler, selector, selected_features, ranked_features)

# ============================================================
# MODEL TRAINING (FR-OM-4)
# ============================================================

def train_model(X_train, y_train, X_val, y_val, config):
    """
    FR-OM-4: Train ML model for cancer classification.
    """
    print("\n" + "="*60)
    print("🤖 STEP 4: MODEL TRAINING")
    print("="*60)
    
    # Create model based on config
    if config['model_type'] == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            class_weight=config['class_weight'],
            random_state=config['random_state'],
            n_jobs=-1
        )
        model_name = "Random Forest"
    elif config['model_type'] == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            random_state=config['random_state']
        )
        model_name = "Gradient Boosting"
    elif config['model_type'] == 'svm':
        model = SVC(kernel='rbf', probability=True, class_weight='balanced',
                   random_state=config['random_state'])
        model_name = "SVM (RBF)"
    else:
        model = LogisticRegression(max_iter=1000, class_weight='balanced',
                                  random_state=config['random_state'])
        model_name = "Logistic Regression"
    
    print(f"\n📋 Model Configuration:")
    print(f"  • Type: {model_name}")
    if config['model_type'] == 'random_forest':
        print(f"  • Trees: {config['n_estimators']}")
        print(f"  • Max Depth: {config['max_depth']}")
        print(f"  • Min Samples Split: {config['min_samples_split']}")
    print(f"  • Class Weight: {config['class_weight']}")
    
    # Cross-validation on training data
    print(f"\n🔄 Cross-Validation ({config['cv_folds']}-fold):")
    cv = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, 
                         random_state=config['random_state'])
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  • AUC-ROC scores: {cv_scores}")
    print(f"  • Mean AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train final model
    print(f"\n🚀 Training final model...")
    model.fit(X_train, y_train)
    
    # Validation performance
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)
    
    print(f"\n📊 Validation Results:")
    print(f"  • Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  • AUC-ROC:  {val_auc:.4f}")
    
    return model, model_name, cv_scores

# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, selected_features):
    """
    Comprehensive model evaluation on test set.
    """
    print("\n" + "="*60)
    print("📊 STEP 5: TEST SET EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n┌─────────────────────────────────────────────────────────┐")
    print(f"│                 PERFORMANCE METRICS                     │")
    print(f"├─────────────────────────────────────────────────────────┤")
    print(f"│  Accuracy:            {acc*100:>6.2f}%                          │")
    print(f"│  AUC-ROC:             {auc:>6.4f}                           │")
    print(f"│  Precision:           {precision*100:>6.2f}%                          │")
    print(f"│  Recall/Sensitivity:  {recall*100:>6.2f}%                          │")
    print(f"│  F1-Score:            {f1:>6.4f}                           │")
    print(f"└─────────────────────────────────────────────────────────┘")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    print(f"\n┌─────────────────────────────────────────────────────────┐")
    print(f"│                  CONFUSION MATRIX                       │")
    print(f"├─────────────────────────────────────────────────────────┤")
    print(f"│                       PREDICTED                         │")
    print(f"│                  Normal     Cancer                      │")
    print(f"│             ┌───────────┬───────────┐                   │")
    print(f"│    Normal   │   {tn:>5}   │   {fp:>5}   │  ← Actual Normal   │")
    print(f"│  A ─────────├───────────┼───────────┤                   │")
    print(f"│  C Cancer   │   {fn:>5}   │   {tp:>5}   │  ← Actual Cancer   │")
    print(f"│  T          └───────────┴───────────┘                   │")
    print(f"│                                                         │")
    print(f"│  TN={tn}: Normal correctly identified                     │")
    print(f"│  TP={tp}: Cancer correctly identified                   │")
    print(f"│  FP={fp}: Normal misclassified as Cancer                  │")
    print(f"│  FN={fn}: Cancer misclassified as Normal (CRITICAL!)      │")
    print(f"│                                                         │")
    print(f"│  Specificity: {specificity*100:.2f}%                                   │")
    print(f"└─────────────────────────────────────────────────────────┘")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Cancer'], digits=4))
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = sorted(zip(selected_features, importances),
                                   key=lambda x: x[1], reverse=True)
        
        print(f"\n🧬 Top Biomarkers by Model Importance:")
        print(f"  ───────────────────────────────────────")
        for i, (gene, imp) in enumerate(feature_importance[:10], 1):
            bar = '█' * int(imp * 50)
            print(f"  {i:>2}. {gene:<12} {imp:.4f} {bar}")
    
    metrics = {
        'accuracy': acc,
        'auc_roc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return metrics, y_pred, y_prob

# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, scaler, selector, selected_features, ranked_features, 
               metrics, config, feature_names):
    """
    Save trained model and all components for inference.
    """
    print("\n" + "="*60)
    print("💾 STEP 6: SAVING MODEL & REPORT")
    print("="*60)
    
    # Create models directory
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    
    # Package for inference
    model_package = {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features,
        'ranked_features': ranked_features,
        'feature_names': feature_names,
        'config': config,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat()
    }
    
    # Save model
    joblib.dump(model_package, config['model_save_path'])
    print(f"\n✓ Model saved: {config['model_save_path']}")
    print(f"  Size: {os.path.getsize(config['model_save_path'])/1024:.1f} KB")
    
    # Save report
    report = {
        'config': config,
        'metrics': metrics,
        'selected_features': selected_features,
        'ranked_features': [(f, float(s)) for f, s in ranked_features],
        'trained_at': datetime.now().isoformat()
    }
    
    with open(config['report_save_path'], 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report saved: {config['report_save_path']}")
    
    return model_package

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("🧬 BREAST CANCER OMICS CLASSIFICATION - TRAINING PIPELINE")
    print("="*70)
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load and preprocess
    X, y, feature_names = load_and_preprocess_data(CONFIG)
    
    # Step 2: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, CONFIG)
    
    # Step 3: Scale and select features (NO LEAKAGE - fit on train only)
    (X_train_sel, X_val_sel, X_test_sel, 
     scaler, selector, selected_features, ranked_features) = scale_and_select_features(
        X_train, X_val, X_test, y_train, feature_names, CONFIG
    )
    
    # Step 4: Train model
    model, model_name, cv_scores = train_model(X_train_sel, y_train, X_val_sel, y_val, CONFIG)
    
    # Step 5: Evaluate on test set
    metrics, y_pred, y_prob = evaluate_model(model, X_test_sel, y_test, selected_features)
    
    # Step 6: Save everything
    model_package = save_model(
        model, scaler, selector, selected_features, ranked_features,
        metrics, CONFIG, feature_names
    )
    
    # Final summary
    print("\n" + "="*70)
    print("📋 TRAINING SUMMARY")
    print("="*70)
    print(f"""
    ╔═════════════════════════════════════════════════════════════════╗
    ║              OMICS MODEL TRAINING COMPLETE                      ║
    ╠═════════════════════════════════════════════════════════════════╣
    ║                                                                 ║
    ║  Dataset: TCGA-BRCA ({len(X)} samples, {len(feature_names)} genes)             ║
    ║  Model: {model_name:<20}                              ║
    ║  Selected Features: {len(selected_features):<4}                                    ║
    ║                                                                 ║
    ║  ┌─────────────────────────────────────────────────────────┐    ║
    ║  │  TEST SET RESULTS:                                      │    ║
    ║  │  • Accuracy:    {metrics['accuracy']*100:.2f}%                               │    ║
    ║  │  • AUC-ROC:     {metrics['auc_roc']:.4f}                              │    ║
    ║  │  • Sensitivity: {metrics['recall']*100:.2f}% (Cancer detection)          │    ║
    ║  │  • Specificity: {metrics['specificity']*100:.2f}% (Normal detection)          │    ║
    ║  └─────────────────────────────────────────────────────────┘    ║
    ║                                                                 ║
    ║  ⚠️  False Negatives (missed cancers): {metrics['false_negatives']:<4}                   ║
    ║                                                                 ║
    ║  Top Biomarkers: {', '.join(selected_features[:5])}...
    ║                                                                 ║
    ╚═════════════════════════════════════════════════════════════════╝
    """)
    
    return model_package

if __name__ == '__main__':
    model_package = main()
