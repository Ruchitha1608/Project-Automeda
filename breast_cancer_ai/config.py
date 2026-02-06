"""
Configuration Module
Central settings management for Breast Cancer Diagnostic System
"""

import os
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
PAPERS_DIR = DATA_DIR / "papers"
IMAGES_DIR = DATA_DIR / "images"
DATABASE_DIR = PROJECT_ROOT / "database"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, PAPERS_DIR, IMAGES_DIR, DATABASE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATABASE SETTINGS
# ============================================================

DATABASE_PATH = DATABASE_DIR / "entities.db"

# ============================================================
# PUBMED / LITERATURE SETTINGS
# ============================================================

NCBI_API_KEY = "ae83b1da74148ccbacc801302448d41ae708"
NCBI_EMAIL = "breast_cancer_ai@research.edu"
MAX_PAPERS_DEFAULT = 10
PUBMED_RATE_LIMIT = 0.34  # seconds between requests (3/sec without key, 10/sec with key)

# ============================================================
# MODEL SETTINGS
# ============================================================

# Cancer Detection (ResNet50)
CANCER_MODEL_CONFIG = {
    "model_name": "resnet50",
    "num_classes": 2,
    "input_size": 224,
    "pretrained": True,
    "dropout": 0.5,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 50,
    "early_stopping_patience": 5,
}

# Image preprocessing
IMAGE_TRANSFORM_CONFIG = {
    "resize": (224, 224),
    "mean": [0.485, 0.456, 0.406],  # ImageNet
    "std": [0.229, 0.224, 0.225],   # ImageNet
}

# Omics Analysis
OMICS_CONFIG = {
    "n_features": 50,
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
}

# ============================================================
# NER / ENTITY EXTRACTION SETTINGS
# ============================================================

NER_CONFIG = {
    "model_name": "dmis-lab/biobert-v1.1",  # BioBERT for biomedical NER
    "max_length": 512,
    "batch_size": 16,
}

# Entity types to extract
ENTITY_TYPES = [
    "GENE",
    "PROTEIN", 
    "DRUG",
    "DISEASE",
    "CHEMICAL",
    "CELL_LINE",
    "MUTATION",
    "PATHWAY",
]

# Key breast cancer genes for pattern matching
KEY_GENES = [
    "BRCA1", "BRCA2", "TP53", "HER2", "ERBB2", "ESR1", "PGR",
    "PIK3CA", "CDH1", "PTEN", "RB1", "MYC", "CCND1", "EGFR",
    "AKT1", "MAP3K1", "GATA3", "FOXA1", "RUNX1", "CBFB",
]

# Common breast cancer drugs
KEY_DRUGS = [
    "Tamoxifen", "Trastuzumab", "Herceptin", "Letrozole", "Anastrozole",
    "Exemestane", "Palbociclib", "Ribociclib", "Abemaciclib",
    "Olaparib", "Talazoparib", "Pembrolizumab", "Atezolizumab",
    "Doxorubicin", "Cyclophosphamide", "Paclitaxel", "Docetaxel",
]

# Diseases/conditions
KEY_DISEASES = [
    "breast cancer", "TNBC", "triple-negative", "HER2-positive",
    "ER-positive", "PR-positive", "invasive ductal carcinoma",
    "invasive lobular carcinoma", "DCIS", "metastatic breast cancer",
]

# ============================================================
# KNOWLEDGE GRAPH SETTINGS
# ============================================================

KNOWLEDGE_GRAPH_CONFIG = {
    "node_colors": {
        "GENE": "#FF6B6B",      # Coral red
        "PROTEIN": "#4ECDC4",   # Teal
        "DRUG": "#45B7D1",      # Sky blue
        "DISEASE": "#96CEB4",   # Sage green
        "CHEMICAL": "#FFEAA7",  # Yellow
        "PATHWAY": "#DDA0DD",   # Plum
        "DEFAULT": "#95A5A6",   # Gray
    },
    "edge_colors": {
        "treats": "#27AE60",
        "causes": "#E74C3C",
        "associated_with": "#3498DB",
        "inhibits": "#9B59B6",
        "activates": "#F39C12",
        "DEFAULT": "#7F8C8D",
    },
    "layout": "force_atlas_2based",
    "physics": True,
    "height": "600px",
    "width": "100%",
}

# ============================================================
# EXPLAINABILITY SETTINGS
# ============================================================

EXPLAINABILITY_CONFIG = {
    "gradcam_target_layer": "layer4",  # For ResNet
    "heatmap_colormap": "jet",
    "overlay_alpha": 0.4,
    "integrated_gradients_steps": 50,
}

# ============================================================
# STREAMLIT UI SETTINGS
# ============================================================

UI_CONFIG = {
    "page_title": "Breast Cancer AI | Clinical Decision Support",
    "page_icon": "🧬",
    "layout": "wide",
    "theme": {
        "primary_color": "#0B3C5D",
        "secondary_color": "#1ABC9C",
        "background_color": "#F5F7FA",
        "text_color": "#2C3E50",
    },
}

# ============================================================
# LOGGING SETTINGS
# ============================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "app.log",
}

# ============================================================
# CLASS LABELS
# ============================================================

CANCER_CLASSES = {
    0: "Benign",
    1: "Malignant",
}

OMICS_CLASSES = {
    0: "Normal",
    1: "Cancer",
}

# ============================================================
# PREDEFINED PUBMED QUERIES
# ============================================================

PREDEFINED_QUERIES = {
    # BreakHis Dataset Queries (Imaging)
    "BreakHis Histopathology": '"BreakHis" AND "breast cancer" AND histopathology',
    "BreakHis Deep Learning": '"BreakHis" AND "deep learning" AND classification',
    "BreakHis CNN/ResNet": '"BreakHis dataset" AND ("ResNet" OR "CNN")',
    "Histopathology Classification": '"breast histopathological" AND "malignant" AND "benign"',
    
    # Gene Expression RNA-Seq Queries (Omics)
    "Gene Expression RNA-Seq": '"gene expression" AND "RNA-Seq" AND "breast cancer"',
    "TCGA-BRCA Dataset": '"breast cancer" AND "TCGA-BRCA" AND RNA-Seq',
    "BRCA1/TP53/HER2 Biomarkers": '"BRCA1" AND "TP53" AND "breast cancer" AND biomarkers',
    "Breast Cancer Biomarkers": '"breast cancer biomarkers" AND "gene expression"',
    
    # Multimodal/Combined Queries
    "Multimodal Imaging + Omics": '"multimodal" AND "breast cancer" AND imaging',
    "Histopathology + RNA-Seq": '"histopathology" AND "RNA-Seq" AND "breast cancer"',
    "Deep Learning Biomarkers": '"deep learning" AND "breast cancer" AND "biomarkers"',
    "Explainable AI Cancer": '"explainable" AND "breast cancer" AND "diagnosis"',
}


def get_config():
    """Return all configuration as a dictionary"""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "database_path": str(DATABASE_PATH),
        "cancer_model": CANCER_MODEL_CONFIG,
        "omics": OMICS_CONFIG,
        "ner": NER_CONFIG,
        "knowledge_graph": KNOWLEDGE_GRAPH_CONFIG,
        "explainability": EXPLAINABILITY_CONFIG,
        "ui": UI_CONFIG,
    }


if __name__ == "__main__":
    import json
    print("Configuration loaded successfully!")
    print(json.dumps(get_config(), indent=2, default=str))
