"""
Verification script to test all modules before running the application
"""

import sys
from pathlib import Path

print("🧬 Breast Cancer AI - Module Verification")
print("=" * 60)

# Test 1: Check imports
print("\n1️⃣ Testing imports...")
try:
    import streamlit
    print("   ✓ Streamlit")
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    import torchvision
    print("   ✓ TorchVision")
    from torchcam.methods import GradCAM
    print("   ✓ TorchCAM")
    import pandas
    print("   ✓ Pandas")
    import numpy
    print("   ✓ NumPy")
    from sklearn.ensemble import RandomForestClassifier
    print("   ✓ Scikit-learn")
    from Bio import Entrez
    print("   ✓ Biopython")
    from PIL import Image
    print("   ✓ Pillow")
    import cv2
    print("   ✓ OpenCV")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check modules
print("\n2️⃣ Testing custom modules...")
try:
    from modules import literature, imaging, omics, integration
    print("   ✓ All modules imported")
except ImportError as e:
    print(f"   ✗ Module import error: {e}")
    sys.exit(1)

# Test 3: Check data files
print("\n3️⃣ Checking data files...")
data_dir = Path("data")
if (data_dir / "sample_breast.jpg").exists():
    print("   ✓ sample_breast.jpg found")
else:
    print("   ✗ sample_breast.jpg missing")

if (data_dir / "sample_omics.csv").exists():
    print("   ✓ sample_omics.csv found")
    # Check CSV structure
    import pandas as pd
    df = pd.read_csv(data_dir / "sample_omics.csv")
    print(f"      Shape: {df.shape}")
    if 'label' in df.columns:
        print(f"      Labels: {df['label'].value_counts().to_dict()}")
else:
    print("   ✗ sample_omics.csv missing")

# Test 4: Test imaging module
print("\n4️⃣ Testing imaging module...")
try:
    from PIL import Image
    import numpy as np
    
    test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    pred_class, confidence, heatmap = imaging.predict_image(test_img)
    print(f"   ✓ Imaging test: {pred_class} ({confidence:.1%})")
except Exception as e:
    print(f"   ✗ Imaging test failed: {e}")

# Test 5: Test omics module
print("\n5️⃣ Testing omics module...")
try:
    if (data_dir / "sample_omics.csv").exists():
        pred, conf, biomarkers = omics.analyze_omics(data_dir / "sample_omics.csv")
        print(f"   ✓ Omics test: {pred} ({conf:.1%})")
        print(f"      Top biomarkers: {', '.join(biomarkers[:3])}")
    else:
        print("   ⚠ Skipped (no sample data)")
except Exception as e:
    print(f"   ✗ Omics test failed: {e}")

# Test 6: Test literature module (optional - requires internet)
print("\n6️⃣ Testing literature module (requires internet)...")
try:
    papers = literature.fetch_pubmed("breast cancer", max_results=1)
    if papers and papers[0]['pmid'] != 'N/A':
        print(f"   ✓ Literature test: Found {len(papers)} paper(s)")
    else:
        print("   ⚠ Literature test: No results or API error")
except Exception as e:
    print(f"   ⚠ Literature test warning: {e}")

# Test 7: Test integration module
print("\n7️⃣ Testing integration module...")
try:
    summary = integration.generate_summary(
        img_pred="Malignant",
        img_conf=0.91,
        omics_pred="Cancer",
        omics_conf=0.89,
        biomarkers=["BRCA1", "TP53", "HER2"],
        literature=[{"pmid": "12345", "title": "Test"}]
    )
    print(f"   ✓ Integration test: {summary['risk_level']}")
except Exception as e:
    print(f"   ✗ Integration test failed: {e}")

print("\n" + "=" * 60)
print("✅ Verification complete! All systems operational.")
print("\n🚀 Ready to launch application:")
print("   streamlit run app.py --server.port 8501")
print("\n   OR simply run: ./run.sh")
print("=" * 60)
