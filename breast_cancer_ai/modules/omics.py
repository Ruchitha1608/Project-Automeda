"""
Omics Analysis Module for Breast Cancer Detection

Analyzes genomic/transcriptomic data to identify cancer biomarkers and predict malignancy.
Follows ML best practices: Split → Scale → Select Features → Train (NO data leakage)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
from typing import Tuple, List, Dict, Optional, Any
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class OmicsAnalyzer:
    """
    Analyzes multi-omics data for cancer detection.
    
    Pipeline: Load → Clean Labels → Handle Missing → Split → Scale → Select → Train
    NO DATA LEAKAGE: Scaling and feature selection happen AFTER train/test split
    """
    
    def __init__(self, n_features: int = 50, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize OmicsAnalyzer.
        
        Args:
            n_features: Number of top features to select
            n_estimators: Number of trees in random forest
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Initialize components - will be fit on training data ONLY
        self.imputer = None  # Fit on train only
        self.scaler = None   # Fit on train only  
        self.selector = None # Fit on train only
        self.classifier = None
        self.label_encoder = LabelEncoder()
        
        self.feature_names = None
        self.selected_features = None
        self.is_trained = False
        
        # Training results storage
        self.training_results = {}
        
        # Known breast cancer biomarkers for interpretation
        self.known_biomarkers = [
            'BRCA1', 'BRCA2', 'TP53', 'HER2', 'ERBB2', 'ESR1', 
            'PGR', 'MKI67', 'AURKA', 'CCNB1', 'MYC', 'PIK3CA',
            'GATA3', 'FOXA1', 'CDH1', 'KRT19', 'KRT18', 'MUC1'
        ]
    
    def _clean_labels(self, y: pd.Series) -> np.ndarray:
        """
        Clean and standardize labels to binary Cancer/Normal.
        
        Handles various label formats: Cancer/Normal, 1/0, Tumor/Normal, etc.
        """
        # Convert to string and lowercase for comparison
        y_str = y.astype(str).str.lower().str.strip()
        
        # Map various labels - use exact match first, then partial
        cancer_exact = ['cancer', 'tumor', 'malignant', '1', 'positive', 'yes', 'true', 'm']
        normal_exact = ['normal', 'benign', 'healthy', '0', 'negative', 'no', 'false', 'b']
        
        y_clean = np.zeros(len(y), dtype=int)
        for i, label in enumerate(y_str):
            # Try exact match first
            if label in cancer_exact:
                y_clean[i] = 1
            elif label in normal_exact:
                y_clean[i] = 0
            # Then try partial match (checking if label starts with or equals)
            elif label.startswith('cancer') or label.startswith('tumor') or label.startswith('malignant'):
                y_clean[i] = 1
            elif label.startswith('normal') or label.startswith('benign') or label.startswith('healthy'):
                y_clean[i] = 0
            else:
                # Try numeric conversion
                try:
                    y_clean[i] = int(float(label))
                except:
                    y_clean[i] = 0  # Default to normal if unknown
        
        return y_clean
    
    def _validate_data(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Validate and clean input data."""
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].copy()
        
        # Remove constant columns
        non_constant = X_numeric.std() > 0
        X_numeric = X_numeric.loc[:, non_constant]
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_names = {0: 'Normal', 1: 'Cancer'}
        print(f"Class distribution: {dict((class_names.get(u, str(u)), c) for u, c in zip(unique, counts))}")
        
        if len(unique) < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        return X_numeric, y
    
    def train(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the omics classifier with proper ML pipeline (NO data leakage).
        
        Pipeline: Split → Impute → Scale → Select Features → Train
        
        Args:
            X: Feature DataFrame (samples × genes)
            y: Target labels (Cancer=1, Normal=0)
            test_size: Fraction for test set
            
        Returns:
            Dictionary with comprehensive training metrics
        """
        print("=" * 60)
        print("OMICS MODEL TRAINING - Proper ML Pipeline")
        print("=" * 60)
        
        # Clean labels
        y_clean = self._clean_labels(pd.Series(y))
        
        # Validate data
        X_clean, y_clean = self._validate_data(X, y_clean)
        self.feature_names = list(X_clean.columns)
        print(f"\nDataset: {X_clean.shape[0]} samples x {X_clean.shape[1]} genes")
        
        # ============================================================
        # STEP 1: SPLIT DATA FIRST (before any preprocessing)
        # ============================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_clean  # Maintain class balance
        )
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # ============================================================
        # STEP 2: IMPUTE MISSING VALUES (fit on train only)
        # ============================================================
        self.imputer = SimpleImputer(strategy='median')
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)  # Transform only!
        print(f"\n[OK] Missing values imputed with median (fit on train only)")
        
        # ============================================================
        # STEP 3: LOG2 TRANSFORM (for gene expression data)
        # ============================================================
        X_train_log = np.log2(np.abs(X_train_imputed) + 1)
        X_test_log = np.log2(np.abs(X_test_imputed) + 1)
        
        # ============================================================
        # STEP 4: SCALE DATA (fit on train only)
        # ============================================================
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_log)
        X_test_scaled = self.scaler.transform(X_test_log)  # Transform only!
        print(f"[OK] Data scaled with StandardScaler (fit on train only)")
        
        # ============================================================
        # STEP 5: FEATURE SELECTION (fit on train only)
        # ============================================================
        k = min(self.n_features, X_train_scaled.shape[1])
        self.selector = SelectKBest(f_classif, k=k)
        X_train_selected = self.selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.selector.transform(X_test_scaled)  # Transform only!
        
        selected_idx = self.selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_idx]
        print(f"[OK] Selected top {k} features (fit on train only)")
        
        # ============================================================
        # STEP 6: TRAIN CLASSIFIER
        # ============================================================
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.classifier.fit(X_train_selected, y_train)
        self.is_trained = True
        print(f"[OK] RandomForest classifier trained ({self.n_estimators} trees)")
        
        # ============================================================
        # STEP 7: EVALUATE ON TEST SET
        # ============================================================
        y_pred = self.classifier.predict(X_test_selected)
        y_proba = self.classifier.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        
        # Feature importance
        importances = self.classifier.feature_importances_
        feature_importance = list(zip(self.selected_features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Cross-validation for robust estimate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.classifier, X_train_selected, y_train, cv=cv, scoring='accuracy')
        
        # Store results
        self.training_results = {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, target_names=['Normal', 'Cancer']),
            'feature_importance': feature_importance,
            'top_biomarkers': feature_importance[:10],
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features_selected': k,
            'class_distribution': {
                'train': {'Normal': int(sum(y_train == 0)), 'Cancer': int(sum(y_train == 1))},
                'test': {'Normal': int(sum(y_test == 0)), 'Cancer': int(sum(y_test == 1))}
            }
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"\nTest Set Performance:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   AUC-ROC:  {auc:.4f}")
        print(f"\nCross-Validation (5-fold):")
        print(f"   Mean Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"\nConfusion Matrix:")
        print(f"   TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
        print(f"   FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
        print(f"\nTop 5 Biomarkers:")
        for i, (gene, imp) in enumerate(feature_importance[:5], 1):
            marker = "*" if gene in self.known_biomarkers else " "
            print(f"   {i}. {marker} {gene}: {imp:.4f}")
        
        return self.training_results
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (predictions, probabilities, per_sample_results)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Align columns with training data
        missing_cols = set(self.feature_names) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_names)
        
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        
        # Apply same preprocessing pipeline (transform only, no fit!)
        X_imputed = self.imputer.transform(X_aligned)
        X_log = np.log2(np.abs(X_imputed) + 1)
        X_scaled = self.scaler.transform(X_log)
        X_selected = self.selector.transform(X_scaled)
        
        # Predict
        predictions = self.classifier.predict(X_selected)
        probabilities = self.classifier.predict_proba(X_selected)[:, 1]
        
        # Per-sample results
        per_sample = []
        for i in range(len(predictions)):
            per_sample.append({
                'sample_idx': i,
                'prediction': 'Cancer' if predictions[i] == 1 else 'Normal',
                'confidence': float(probabilities[i]) if predictions[i] == 1 else float(1 - probabilities[i]),
                'cancer_probability': float(probabilities[i])
            })
        
        return predictions, probabilities, per_sample
    
    def get_top_biomarkers(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top biomarkers by importance."""
        if not self.is_trained:
            return []
        
        importances = self.classifier.feature_importances_
        feature_importance = list(zip(self.selected_features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:n]
    
    def save_model(self, path: str):
        """Save trained model and preprocessing pipeline."""
        model_data = {
            'imputer': self.imputer,
            'scaler': self.scaler,
            'selector': self.selector,
            'classifier': self.classifier,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'training_results': self.training_results,
            'n_features': self.n_features,
            'n_estimators': self.n_estimators
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and preprocessing pipeline."""
        model_data = joblib.load(path)
        self.imputer = model_data['imputer']
        self.scaler = model_data['scaler']
        self.selector = model_data['selector']
        self.classifier = model_data['classifier']
        self.feature_names = model_data['feature_names']
        self.selected_features = model_data['selected_features']
        self.training_results = model_data['training_results']
        self.n_features = model_data['n_features']
        self.n_estimators = model_data['n_estimators']
        self.is_trained = True
        print(f"Model loaded from {path}")


def analyze_omics(data_path: str, model_path: str = None) -> Tuple[str, float, List[str], Dict]:
    """
    Analyze omics data file for breast cancer prediction.
    
    Args:
        data_path: Path to CSV file with omics data
        model_path: Optional path to save/load model
        
    Returns:
        Tuple of (prediction, confidence, top_biomarkers, full_results)
    """
    try:
        # Load patient data
        data = pd.read_csv(data_path)
        print(f"\nLoaded data: {data.shape[0]} samples x {data.shape[1]} columns")
        
        # Detect label column
        label_cols = [c for c in data.columns if c.lower() in 
                     ['label', 'labels', 'class', 'target', 'diagnosis', 'sample_type', 'type']]
        
        if not label_cols:
            # No labels - prediction mode only
            return _predict_only(data, model_path)
        
        # Training/evaluation mode
        y_col = label_cols[0]
        X = data.drop(columns=[y_col])
        y = data[y_col]
        
        print(f"Label column: {y_col}")
        print(f"Features: {X.shape[1]} genes")
        
        # Initialize and train analyzer
        analyzer = OmicsAnalyzer(n_features=min(50, X.shape[1] - 1), n_estimators=100)
        results = analyzer.train(X, y)
        
        # Save model if path provided
        if model_path:
            analyzer.save_model(model_path)
        
        # Get overall prediction summary
        top_biomarkers = [b[0] for b in results['top_biomarkers'][:5]]
        
        # Create comprehensive results
        full_results = {
            'accuracy': results['accuracy'],
            'auc': results['auc'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'classification_report': results['classification_report'],
            'feature_importance': results['feature_importance'][:20],
            'top_biomarkers': top_biomarkers,
            'cv_mean_accuracy': results['cv_mean'],
            'cv_std': results['cv_std'],
            'class_distribution': results['class_distribution']
        }
        
        # Overall prediction (majority from test set implied by accuracy > 0.5)
        prediction = "Cancer" if results['accuracy'] > 0.5 else "Normal"
        confidence = results['accuracy']
        
        return prediction, confidence, top_biomarkers, full_results
        
    except Exception as e:
        print(f"Omics analysis error: {e}")
        import traceback
        traceback.print_exc()
        return "Unknown", 0.0, [], {'error': str(e)}


def _predict_only(data: pd.DataFrame, model_path: str = None) -> Tuple[str, float, List[str], Dict]:
    """Handle prediction when no labels are available."""
    
    # Check for pre-trained model
    if model_path and os.path.exists(model_path):
        analyzer = OmicsAnalyzer()
        analyzer.load_model(model_path)
        predictions, probabilities, per_sample = analyzer.predict(data)
        
        # Aggregate results
        avg_prob = probabilities.mean()
        prediction = "Cancer" if avg_prob > 0.5 else "Normal"
        confidence = avg_prob if avg_prob > 0.5 else (1 - avg_prob)
        top_biomarkers = [b[0] for b in analyzer.get_top_biomarkers(5)]
        
        return prediction, confidence, top_biomarkers, {
            'per_sample_predictions': per_sample,
            'average_cancer_probability': float(avg_prob),
            'top_biomarkers': top_biomarkers
        }
    
    # No model - return informative error
    return "Unknown", 0.0, [], {
        'error': 'No labels in data and no pre-trained model provided',
        'suggestion': 'Provide labeled data for training or a pre-trained model for prediction'
    }


def train_on_tcga(data_dir: str = 'data', save_path: str = 'models/omics_model.pkl') -> Dict:
    """
    Train omics model on TCGA-BRCA data.
    
    Args:
        data_dir: Directory containing TCGA data files
        save_path: Where to save trained model
        
    Returns:
        Training results dictionary
    """
    # Find TCGA data file
    possible_files = [
        os.path.join(data_dir, 'tcga_brca_key_genes.csv'),
        os.path.join(data_dir, 'tcga_brca_top500.csv'),
        os.path.join(data_dir, 'tcga_brca_data.csv')
    ]
    
    data_file = None
    for f in possible_files:
        if os.path.exists(f):
            data_file = f
            break
    
    if not data_file:
        raise FileNotFoundError(f"No TCGA data found in {data_dir}")
    
    print(f"Training on: {data_file}")
    
    # Run analysis (which includes training)
    prediction, confidence, biomarkers, results = analyze_omics(data_file, save_path)
    
    return results


# ============================================================
# BACKWARD-COMPATIBLE WRAPPER FOR STREAMLIT APP
# ============================================================

def analyze_omics_file(file_input) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Analyze omics data from file object or path (backward compatible with app).
    
    Args:
        file_input: Either a file path (str) or file-like object (from Streamlit uploader)
        
    Returns:
        Tuple of (prediction, confidence, biomarkers_with_importance)
        where biomarkers_with_importance is List[(gene_name, importance_score)]
    """
    import tempfile
    import io
    
    # Get the base directory for model path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try v3 model first (trained with train_omics_model.py), then v2
    model_paths = [
        os.path.join(base_dir, 'models', 'omics_model_v3.pkl'),
        os.path.join(base_dir, 'models', 'omics_model_v2.pkl'),
        os.path.join(base_dir, 'models', 'omics_model.pkl')
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    try:
        # Handle file object vs file path
        if isinstance(file_input, str):
            # It's a file path
            data_path = file_input
        else:
            # It's a file-like object (from Streamlit uploader)
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                # Read content from file object
                if hasattr(file_input, 'read'):
                    content = file_input.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    tmp.write(content)
                else:
                    tmp.write(str(file_input))
                data_path = tmp.name
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Check for label column
        label_cols = [c for c in data.columns if c.lower() in 
                     ['label', 'labels', 'class', 'target', 'diagnosis', 'sample_type', 'type']]
        
        if label_cols:
            X = data.drop(columns=[label_cols[0]])
        else:
            X = data
        
        # Check if we have trained model (v3 format from train_omics_model.py)
        if model_path and 'v3' in model_path:
            # Load the new model package
            model_package = joblib.load(model_path)
            
            model = model_package['model']
            scaler = model_package['scaler']
            selector = model_package['selector']
            ranked_features = model_package['ranked_features']
            
            # Preprocess: log transform
            X_np = X.values.astype(np.float64)
            X_min = X_np.min()
            if X_min <= 0:
                X_np = X_np - X_min + 1
            X_log = np.log2(X_np + 1)
            
            # Scale and select features
            X_scaled = scaler.transform(X_log)
            X_selected = selector.transform(X_scaled)
            
            # Predict
            predictions = model.predict(X_selected)
            probabilities = model.predict_proba(X_selected)[:, 1]
            
            # Aggregate results
            avg_prob = float(probabilities.mean())
            prediction = "Cancer" if avg_prob > 0.5 else "Normal"
            confidence = avg_prob if avg_prob > 0.5 else (1 - avg_prob)
            
            # Get biomarkers (already has importance from training)
            biomarkers = [(gene, float(score)) for gene, score in ranked_features[:10]]
            
            print(f"✅ Omics analysis: {prediction} ({confidence:.1%})")
            return prediction, confidence, biomarkers
        
        elif model_path:
            # Use older model format (v2 or v1)
            analyzer = OmicsAnalyzer()
            analyzer.load_model(model_path)
            
            # Make predictions
            predictions, probabilities, per_sample = analyzer.predict(X)
            
            # Aggregate results
            avg_prob = float(probabilities.mean())
            prediction = "Cancer" if avg_prob > 0.5 else "Normal"
            confidence = avg_prob if avg_prob > 0.5 else (1 - avg_prob)
            
            # Get biomarkers with importance scores
            biomarkers = analyzer.get_top_biomarkers(10)
            
            print(f"✅ Omics analysis (v2): {prediction} ({confidence:.1%})")
            return prediction, confidence, biomarkers
        
        else:
            # No model available
            print("⚠️ No trained omics model found. Run: python train_omics_model.py")
            return "Unknown", 0.5, [("No model", 0.0)]
            
    except Exception as e:
        print(f"Omics analysis error: {e}")
        import traceback
        traceback.print_exc()
        # Return default values on error
        return "Unknown", 0.5, [("Error", 0.0)]
    
    finally:
        # Cleanup temp file if created
        if not isinstance(file_input, str) and 'data_path' in locals():
            try:
                os.unlink(data_path)
            except:
                pass


if __name__ == "__main__":
    import sys
    
    # Default paths
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_dir, 'data', 'tcga_brca_key_genes.csv')
    model_path = os.path.join(data_dir, 'models', 'omics_model_v2.pkl')
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    print("=" * 70)
    print("BREAST CANCER OMICS ANALYSIS")
    print("Proper ML Pipeline: Split -> Scale -> Select -> Train (NO DATA LEAKAGE)")
    print("=" * 70)
    
    if os.path.exists(data_path):
        prediction, confidence, biomarkers, results = analyze_omics(data_path, model_path)
        
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"\nOverall Prediction: {prediction}")
        print(f"Model Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"AUC-ROC: {results.get('auc', 0):.4f}")
        print(f"\nTop Biomarkers: {', '.join(biomarkers)}")
    else:
        print(f"\nData file not found: {data_path}")
        print("Run with: python -m modules.omics <path_to_csv>")
