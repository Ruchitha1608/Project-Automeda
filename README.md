# 🧠 Explainable Multimodal AI for Breast Cancer Detection

A comprehensive Streamlit web application combining deep learning, genomics, and literature mining for breast cancer diagnosis with full explainability.

## 🎯 Features

- **🖼️ Imaging Analysis**: ResNet50-based histopathology classification with GradCAM heatmaps
- **🧬 Omics Analysis**: Gene expression profiling with Random Forest and biomarker identification
- **📚 Literature Mining**: Automated PubMed search with NER entity extraction (Rule-based + BioBERT)
- **🕸️ Knowledge Graph**: Visual entity relationship mapping
- **🗄️ Database**: SQLite storage for papers, entities, and predictions
- **🧠 Multimodal Integration**: Unified risk assessment combining all modalities

## 📁 Project Structure

```
breast_cancer_ai/
├── app.py                      # Main Streamlit application (5 tabs)
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── setup.py                    # Setup script
├── run.sh                      # Quick run script
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── modules/
│   ├── __init__.py
│   ├── literature.py           # PubMed integration + predefined queries
│   ├── literature_ner.py       # NER: Rule-based + BioBERT entity extraction
│   ├── knowledge_graph.py      # NetworkX/PyVis graph visualization
│   ├── imaging.py              # ResNet50 + GradCAM
│   ├── omics.py                # Random Forest gene expression ML
│   └── integration.py          # Multimodal risk calculation
├── database/
│   ├── __init__.py
│   ├── entities_db.py          # SQLite database for persistence
│   └── entities.db             # Database file
├── models/
│   ├── imaging_model_trained.pth  # Trained ResNet50 (94.87% accuracy)
│   └── omics_model_v3.pkl         # Trained Random Forest (96.72% accuracy)
├── data/
│   ├── Breakhis-400x/          # BreakHis histopathology images
│   ├── tcga_brca_key_genes.csv # TCGA gene expression data
│   ├── tcga_brca_top500.csv    # TCGA top 500 variable genes
│   ├── sample_breast.jpg       # Demo image
│   ├── sample_omics.csv        # Demo CSV
│   ├── generate_image.py       # Image generator
│   └── generate_data.py        # CSV generator
├── notebooks/
│   └── Imaging_Model_Training.ipynb
├── train_imaging_model.py      # Imaging model training script
├── train_omics_model.py        # Omics model training script
├── test_setup.py               # Dependency verification
├── test_integration.py         # Integration testing
├── IMAGING_MODEL_DOCUMENTATION.md
├── OMICS_MODEL_DOCUMENTATION.md
├── LITERATURE_MODEL_DOCUMENTATION.md
└── INTEGRATION_DOCUMENTATION.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Application

```bash
streamlit run app.py --server.port 8502
```

The application will open at `http://localhost:8502`

## 📊 Application Tabs

| Tab | Function |
|-----|----------|
| **1. Literature Mining** | PubMed search with NER entity extraction |
| **2. Imaging Analysis** | Upload and classify histopathology images |
| **3. Omics Analysis** | Upload gene expression CSV for classification |
| **4. Integrated Results** | Unified risk assessment |
| **5. Database History** | Browse stored papers, entities, predictions |

## 🔬 Technical Details

### Imaging Module
- **Model**: ResNet50 (trained on BreakHis dataset)
- **Performance**: 94.87% accuracy, AUC 0.9974
- **Classification**: Binary (Benign/Malignant)
- **Explainability**: GradCAM heatmaps on layer4
- **Preprocessing**: Resize(224), ImageNet normalization

### Omics Module
- **Model**: Random Forest (200 trees, max_depth=10)
- **Performance**: 96.72% accuracy, AUC 0.9993
- **Pipeline**: Impute → Log2 → Z-score → SelectKBest → RandomForest
- **Output**: Classification + top 10 biomarkers by importance

### Literature Module
- **API**: NCBI Entrez (Biopython)
- **NER**: Rule-based + optional BioBERT (dmis-lab/biobert-v1.1)
- **Entities**: GENE, DRUG, DISEASE, MUTATION, PATHWAY
- **Knowledge Graph**: NetworkX + PyVis visualization
- **12 Predefined Queries**: BreakHis, TCGA-BRCA, biomarkers

### Integration Module
- **Risk Calculation**: Weighted multimodal fusion
- **Risk Levels**: HIGH (>85%), MODERATE (65-85%), LOW (<65%)
- **Agreement Detection**: Concordant vs discordant findings

### Database Module
- **Storage**: SQLite (papers, entities, relations, predictions, biomarkers)
- **Export**: CSV export for entities and relations
- **History**: Query logging with timestamps

## 📦 Dependencies

```
# Core
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
torchcam>=0.4.0

# Machine Learning
scikit-learn>=1.3.0

# Literature/NER
biopython>=1.80
transformers>=4.30.0  # For BioBERT

# Visualization
matplotlib>=3.7.0
networkx>=3.0
pyvis>=0.3.0
wordcloud>=1.9.0

# Image Processing
Pillow>=10.0.0
opencv-python>=4.8.0
```

## 🧬 Input Data Format

### Histopathology Image
- **Format**: JPG or PNG
- **Type**: H&E stained tissue
- **Resolution**: Any (auto-resized to 224x224)

### Gene Expression CSV
```csv
BRCA1,TP53,HER2,Gene4,...,label
7.2,6.8,5.3,4.1,...,Cancer
5.1,5.4,4.9,4.2,...,Normal
```

## 🎓 Training Models

### Train Imaging Model
```bash
python train_imaging_model.py
```

### Train Omics Model
```bash
python train_omics_model.py
```

## 📖 Documentation

- [IMAGING_MODEL_DOCUMENTATION.md](IMAGING_MODEL_DOCUMENTATION.md) - Imaging pipeline details
- [OMICS_MODEL_DOCUMENTATION.md](OMICS_MODEL_DOCUMENTATION.md) - Omics pipeline details
- [LITERATURE_MODEL_DOCUMENTATION.md](LITERATURE_MODEL_DOCUMENTATION.md) - NER and literature mining
- [INTEGRATION_DOCUMENTATION.md](INTEGRATION_DOCUMENTATION.md) - Risk calculation algorithm

## 🔮 Future Enhancements

- [ ] Multi-class classification (IDC, DCIS, ILC)
- [ ] Survival prediction integration
- [ ] Treatment recommendation system
- [ ] PDF report export
- [ ] BioGPT integration for text generation
- [ ] Treatment recommendation system
- [ ] Export functionality (PDF reports)
- [ ] User authentication
- [ ] Database integration for case management

## 📄 Citation

```bibtex
@software{breast_cancer_multimodal_ai,
  title={Explainable Multimodal AI for Breast Cancer Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/breast_cancer_ai}
}
```

## ⚕️ Clinical Use Disclaimer

This AI system is designed for **research and clinical decision support only**. All diagnoses must be confirmed by licensed medical professionals. Do not use as sole basis for treatment decisions.

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

**Built with**: PyTorch • Scikit-learn • Biopython • Streamlit • GradCAM
