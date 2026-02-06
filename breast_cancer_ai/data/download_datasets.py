#!/usr/bin/env python3
"""
Download Real Datasets for Breast Cancer AI System

Datasets:
1. BreakHis - Breast Cancer Histopathological Image Database
2. TCGA-BRCA - The Cancer Genome Atlas Breast Cancer RNA-Seq

Note: Some datasets require registration or have size constraints.
This script provides multiple options.
"""

import os
import sys
import zipfile
import tarfile
import urllib.request
import shutil
from pathlib import Path

# Try importing optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.resolve()
IMAGES_DIR = BASE_DIR / "images"
OMICS_DIR = BASE_DIR / "omics"

# BreakHis Dataset Info
BREAKHIS_INFO = """
╔══════════════════════════════════════════════════════════════╗
║                    BreakHis Dataset                          ║
╠══════════════════════════════════════════════════════════════╣
║ Source: https://web.inf.ufpr.br/vri/databases/breast-cancer- ║
║         histopathological-database-breakhis/                 ║
║                                                              ║
║ Size: ~1.9 GB (7,909 images)                                 ║
║ Resolution: 700x460 pixels                                   ║
║ Magnification: 40X, 100X, 200X, 400X                        ║
║ Classes: Benign (4 types), Malignant (4 types)              ║
║                                                              ║
║ MANUAL DOWNLOAD REQUIRED:                                    ║
║ 1. Visit the website above                                   ║
║ 2. Fill the registration form                                ║
║ 3. Download BreaKHis_v1.tar.gz                              ║
║ 4. Place in: data/ folder                                    ║
║ 5. Run: python download_datasets.py --extract-breakhis      ║
╚══════════════════════════════════════════════════════════════╝
"""

# TCGA-BRCA Info
TCGA_INFO = """
╔══════════════════════════════════════════════════════════════╗
║                    TCGA-BRCA Dataset                         ║
╠══════════════════════════════════════════════════════════════╣
║ Source: https://portal.gdc.cancer.gov/                       ║
║         https://xenabrowser.net/datapages/                   ║
║                                                              ║
║ Contains: RNA-Seq gene expression for ~1,200 samples         ║
║ Includes: Tumor and normal tissue samples                    ║
║                                                              ║
║ Options:                                                     ║
║ A) Download from UCSC Xena (recommended, easier)             ║
║ B) Download from GDC Portal (requires account)               ║
╚══════════════════════════════════════════════════════════════╝
"""

# Alternative: Wisconsin Breast Cancer Dataset (UCI - small but real)
UCI_WDBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
UCI_WDBC_NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names"

# UCSC Xena TCGA-BRCA URLs
XENA_TCGA_BRCA_EXPRESSION = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz"
XENA_TCGA_BRCA_CLINICAL = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def download_file(url, destination, description="file"):
    """Download file with progress"""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {destination}")
    
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
            print()
        else:
            urllib.request.urlretrieve(url, destination)
        
        print(f"  ✓ Downloaded: {destination}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def extract_archive(archive_path, destination):
    """Extract tar.gz or zip archive"""
    print(f"Extracting: {archive_path}")
    
    try:
        if str(archive_path).endswith('.tar.gz') or str(archive_path).endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(destination)
        elif str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(destination)
        elif str(archive_path).endswith('.gz'):
            import gzip
            output_path = destination / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        print(f"  ✓ Extracted to: {destination}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


# ============================================================
# DATASET 1: BreakHis
# ============================================================

def setup_breakhis():
    """Setup BreakHis dataset structure after manual download"""
    print_header("Setting up BreakHis Dataset")
    print(BREAKHIS_INFO)
    
    # Look for downloaded archive
    archive_candidates = [
        BASE_DIR / "BreaKHis_v1.tar.gz",
        BASE_DIR / "breakhis.tar.gz",
        BASE_DIR.parent / "BreaKHis_v1.tar.gz",
    ]
    
    archive_path = None
    for candidate in archive_candidates:
        if candidate.exists():
            archive_path = candidate
            break
    
    if archive_path is None:
        print("\n⚠️  BreakHis archive not found!")
        print("Please download manually from the website above.")
        print(f"Place the archive in: {BASE_DIR}")
        return False
    
    print(f"\nFound archive: {archive_path}")
    
    # Create extraction directory
    extract_dir = BASE_DIR / "breakhis_raw"
    extract_dir.mkdir(exist_ok=True)
    
    # Extract
    if not extract_archive(archive_path, extract_dir):
        return False
    
    # Organize into benign/malignant structure
    return organize_breakhis(extract_dir)


def organize_breakhis(extract_dir):
    """Organize BreakHis into benign/malignant folders"""
    print("\nOrganizing images into benign/malignant folders...")
    
    benign_dir = IMAGES_DIR / "benign"
    malignant_dir = IMAGES_DIR / "malignant"
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)
    
    benign_count = 0
    malignant_count = 0
    
    # BreakHis structure: histology_slides/breast/[benign|malignant]/SOB/[type]/[patient]/[magnification]/
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.png'):
                src_path = Path(root) / file
                
                # Determine class from path
                path_str = str(root).lower()
                
                if 'benign' in path_str:
                    dest_path = benign_dir / file
                    benign_count += 1
                elif 'malignant' in path_str:
                    dest_path = malignant_dir / file
                    malignant_count += 1
                else:
                    continue
                
                # Copy file
                shutil.copy2(src_path, dest_path)
    
    print(f"\n✓ Organized {benign_count} benign images")
    print(f"✓ Organized {malignant_count} malignant images")
    print(f"✓ Total: {benign_count + malignant_count} images")
    
    return True


def download_sample_histopathology():
    """Download a few sample histopathology images for testing"""
    print_header("Downloading Sample Histopathology Images")
    
    # Using images from publicly available sources
    # Note: These are placeholder URLs - in reality you'd use proper sources
    
    benign_dir = IMAGES_DIR / "benign"
    malignant_dir = IMAGES_DIR / "malignant"
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n⚠️  For real histopathology images, please download BreakHis dataset")
    print("    or use images from your own institution.")
    print("\nGenerating high-quality synthetic samples for demo purposes...")
    
    # Generate better synthetic images that simulate histopathology
    try:
        from PIL import Image
        import numpy as np
        
        def create_synthetic_histopath(is_malignant=False, idx=0):
            """Create more realistic synthetic histopathology image"""
            np.random.seed(42 + idx)
            
            # Base pink/purple H&E staining colors
            if is_malignant:
                # Malignant: more purple (nuclear), irregular patterns
                base_color = np.array([180, 130, 180])  # More purple
                cell_density = 0.4
                cell_size_var = 0.5
            else:
                # Benign: more pink (cytoplasm), regular patterns
                base_color = np.array([230, 180, 190])  # More pink
                cell_density = 0.2
                cell_size_var = 0.2
            
            # Create base image
            img = np.zeros((460, 700, 3), dtype=np.uint8)
            img[:] = base_color
            
            # Add tissue texture
            noise = np.random.randint(-20, 20, (460, 700, 3))
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
            
            # Add cell-like structures
            n_cells = int(500 * cell_density)
            for _ in range(n_cells):
                cx = np.random.randint(20, 680)
                cy = np.random.randint(20, 440)
                radius = int(np.random.uniform(5, 15) * (1 + cell_size_var * np.random.randn()))
                radius = max(3, min(radius, 25))
                
                # Cell nucleus (darker purple)
                y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                mask = x*x + y*y <= radius*radius
                
                y_start = max(0, cy - radius)
                y_end = min(460, cy + radius + 1)
                x_start = max(0, cx - radius)
                x_end = min(700, cx + radius + 1)
                
                mask_y_start = max(0, radius - cy)
                mask_y_end = mask_y_start + (y_end - y_start)
                mask_x_start = max(0, radius - cx)
                mask_x_end = mask_x_start + (x_end - x_start)
                
                local_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                
                if is_malignant:
                    cell_color = np.array([100, 50, 120])  # Dark purple nuclei
                else:
                    cell_color = np.array([150, 100, 150])  # Lighter nuclei
                
                img[y_start:y_end, x_start:x_end][local_mask] = cell_color
            
            return Image.fromarray(img)
        
        # Generate samples
        n_samples = 10
        
        for i in range(n_samples):
            # Benign
            img = create_synthetic_histopath(is_malignant=False, idx=i)
            img.save(benign_dir / f"synthetic_benign_{i+1:03d}.png")
            
            # Malignant
            img = create_synthetic_histopath(is_malignant=True, idx=i+100)
            img.save(malignant_dir / f"synthetic_malignant_{i+1:03d}.png")
        
        print(f"\n✓ Generated {n_samples} benign samples in: {benign_dir}")
        print(f"✓ Generated {n_samples} malignant samples in: {malignant_dir}")
        print("\n⚠️  These are SYNTHETIC images for demo only!")
        print("    Download BreakHis for real histopathology data.")
        
        return True
        
    except Exception as e:
        print(f"Error generating samples: {e}")
        return False


# ============================================================
# DATASET 2: Gene Expression (Multiple Options)
# ============================================================

def download_uci_wdbc():
    """Download UCI Wisconsin Diagnostic Breast Cancer dataset"""
    print_header("Downloading UCI WDBC Dataset")
    
    print("""
This is the Wisconsin Diagnostic Breast Cancer dataset:
- 569 samples (357 benign, 212 malignant)
- 30 features computed from cell nuclei images
- Small but real dataset, good for testing
""")
    
    OMICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download data
    data_path = OMICS_DIR / "wdbc.data"
    if not download_file(UCI_WDBC_URL, data_path, "WDBC data"):
        return False
    
    # Process into proper CSV format
    if not PANDAS_AVAILABLE:
        print("pandas required to process data. Install with: pip install pandas")
        return False
    
    # Column names for WDBC
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    columns = ['id', 'diagnosis'] + feature_names
    
    df = pd.read_csv(data_path, header=None, names=columns)
    
    # Convert diagnosis to label
    df['label'] = df['diagnosis'].map({'M': 'Cancer', 'B': 'Normal'})
    
    # Drop ID and original diagnosis
    df = df.drop(['id', 'diagnosis'], axis=1)
    
    # Reorder columns to put label at end
    cols = [c for c in df.columns if c != 'label'] + ['label']
    df = df[cols]
    
    # Save
    output_path = BASE_DIR / "uci_wdbc.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Processed dataset saved to: {output_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Cancer: {sum(df['label'] == 'Cancer')}")
    print(f"  Normal: {sum(df['label'] == 'Normal')}")
    
    # Also copy as sample_omics.csv for compatibility
    shutil.copy(output_path, BASE_DIR / "sample_omics.csv")
    print(f"✓ Also copied to: {BASE_DIR / 'sample_omics.csv'}")
    
    return True


def download_tcga_brca():
    """Download TCGA-BRCA gene expression from UCSC Xena"""
    print_header("Downloading TCGA-BRCA from UCSC Xena")
    print(TCGA_INFO)
    
    OMICS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nDownloading gene expression data (~50MB compressed)...")
    print("This may take a few minutes...\n")
    
    # Download expression data
    expr_gz_path = OMICS_DIR / "TCGA_BRCA_expression.gz"
    if not download_file(XENA_TCGA_BRCA_EXPRESSION, expr_gz_path, "expression data"):
        return False
    
    # Download clinical data
    clinical_path = OMICS_DIR / "TCGA_BRCA_clinical.tsv"
    if not download_file(XENA_TCGA_BRCA_CLINICAL, clinical_path, "clinical data"):
        print("Warning: Clinical data download failed, continuing without it")
    
    # Extract and process
    print("\nProcessing data (this may take a minute)...")
    
    if not PANDAS_AVAILABLE:
        print("pandas required to process data. Install with: pip install pandas")
        return False
    
    try:
        import gzip
        
        # Read expression data
        expr_path = OMICS_DIR / "TCGA_BRCA_expression.tsv"
        with gzip.open(expr_gz_path, 'rb') as f_in:
            with open(expr_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Load expression matrix
        print("Loading expression matrix...")
        expr_df = pd.read_csv(expr_path, sep='\t', index_col=0)
        
        # Transpose (genes as columns, samples as rows)
        expr_df = expr_df.T
        
        print(f"Expression matrix: {expr_df.shape[0]} samples × {expr_df.shape[1]} genes")
        
        # Try to load clinical data for labels
        if clinical_path.exists():
            clinical_df = pd.read_csv(clinical_path, sep='\t', index_col=0)
            
            # Find sample type column
            sample_type_col = None
            for col in ['sample_type', 'sample_type_id', '_sample_type']:
                if col in clinical_df.columns:
                    sample_type_col = col
                    break
            
            if sample_type_col:
                # Merge clinical info
                common_samples = expr_df.index.intersection(clinical_df.index)
                expr_df = expr_df.loc[common_samples]
                
                # Add label based on sample type
                sample_types = clinical_df.loc[common_samples, sample_type_col]
                expr_df['label'] = sample_types.apply(
                    lambda x: 'Normal' if 'Normal' in str(x) else 'Cancer'
                )
            else:
                # Infer from sample ID (TCGA naming convention)
                # Sample type is in position 14-15 of barcode
                # 01-09 = Tumor, 10-19 = Normal
                def infer_label(sample_id):
                    try:
                        sample_type = int(sample_id.split('-')[3][:2])
                        return 'Normal' if sample_type >= 10 else 'Cancer'
                    except:
                        return 'Cancer'  # Default
                
                expr_df['label'] = expr_df.index.map(infer_label)
        else:
            # Infer labels from sample IDs
            def infer_label(sample_id):
                try:
                    sample_type = int(sample_id.split('-')[3][:2])
                    return 'Normal' if sample_type >= 10 else 'Cancer'
                except:
                    return 'Cancer'
            
            expr_df['label'] = expr_df.index.map(infer_label)
        
        # Select key breast cancer genes if available
        key_genes = [
            'BRCA1', 'BRCA2', 'TP53', 'ERBB2', 'ESR1', 'PGR', 'MYC', 'PIK3CA',
            'PTEN', 'EGFR', 'AKT1', 'CCND1', 'CDH1', 'RB1', 'CDKN2A', 'MDM2',
            'KRAS', 'BRAF', 'ATM', 'CHEK2', 'MKI67', 'BCL2', 'BAX', 'VEGFA'
        ]
        
        available_genes = [g for g in key_genes if g in expr_df.columns]
        
        # Create focused dataset with key genes
        if available_genes:
            focused_df = expr_df[available_genes + ['label']]
            focused_path = BASE_DIR / "tcga_brca_key_genes.csv"
            focused_df.to_csv(focused_path, index=False)
            print(f"\n✓ Key genes dataset: {focused_path}")
            print(f"  Genes: {len(available_genes)}")
        
        # Save full dataset (or subset of top variable genes)
        print("\nSelecting top variable genes...")
        gene_cols = [c for c in expr_df.columns if c != 'label']
        variances = expr_df[gene_cols].var()
        top_genes = variances.nlargest(500).index.tolist()
        
        subset_df = expr_df[top_genes + ['label']]
        subset_path = BASE_DIR / "tcga_brca_top500.csv"
        subset_df.to_csv(subset_path, index=False)
        
        print(f"\n✓ Top 500 variable genes: {subset_path}")
        print(f"  Samples: {len(subset_df)}")
        print(f"  Cancer: {sum(subset_df['label'] == 'Cancer')}")
        print(f"  Normal: {sum(subset_df['label'] == 'Normal')}")
        
        # Copy as default sample_omics.csv
        shutil.copy(subset_path, BASE_DIR / "sample_omics.csv")
        print(f"\n✓ Also copied to: {BASE_DIR / 'sample_omics.csv'}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download real datasets for Breast Cancer AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --list              # Show available datasets
  python download_datasets.py --uci               # Download small UCI dataset
  python download_datasets.py --tcga              # Download TCGA-BRCA (large)
  python download_datasets.py --extract-breakhis  # Extract BreakHis after manual download
  python download_datasets.py --all               # Download all available
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--uci', action='store_true', help='Download UCI WDBC dataset (small)')
    parser.add_argument('--tcga', action='store_true', help='Download TCGA-BRCA from Xena')
    parser.add_argument('--extract-breakhis', action='store_true', help='Extract BreakHis archive')
    parser.add_argument('--sample-images', action='store_true', help='Generate sample images for testing')
    parser.add_argument('--all', action='store_true', help='Download all available datasets')
    
    args = parser.parse_args()
    
    if args.list or len(sys.argv) == 1:
        print("""
╔══════════════════════════════════════════════════════════════╗
║           Available Datasets for Download                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  GENOMICS / OMICS DATA:                                      ║
║  ──────────────────────                                      ║
║  1. UCI WDBC (--uci)                                         ║
║     • 569 samples, 30 features                               ║
║     • Small, fast to download                                ║
║     • Good for testing                                       ║
║                                                              ║
║  2. TCGA-BRCA (--tcga)                                       ║
║     • ~1,200 samples, 20,000+ genes                          ║
║     • Real RNA-Seq expression data                           ║
║     • ~50MB download                                         ║
║                                                              ║
║  IMAGING DATA:                                               ║
║  ─────────────                                               ║
║  3. BreakHis (--extract-breakhis)                            ║
║     • 7,909 histopathology images                            ║
║     • Requires manual download first                         ║
║     • See instructions when running                          ║
║                                                              ║
║  4. Sample Images (--sample-images)                          ║
║     • Generated synthetic images for testing                 ║
║     • NOT real medical images                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Run with --uci or --tcga to download real data!
        """)
        return
    
    success = True
    
    if args.all or args.uci:
        if not download_uci_wdbc():
            success = False
    
    if args.all or args.tcga:
        if not download_tcga_brca():
            success = False
    
    if args.all or args.extract_breakhis:
        if not setup_breakhis():
            success = False
    
    if args.sample_images:
        if not download_sample_histopathology():
            success = False
    
    if success:
        print("\n" + "=" * 60)
        print(" ✓ Dataset download complete!")
        print("=" * 60)
        print(f"\nData saved to: {BASE_DIR}")
        print("\nFiles created:")
        for f in BASE_DIR.glob("*.csv"):
            print(f"  • {f.name}")
        
        if IMAGES_DIR.exists():
            benign_count = len(list((IMAGES_DIR / "benign").glob("*"))) if (IMAGES_DIR / "benign").exists() else 0
            malignant_count = len(list((IMAGES_DIR / "malignant").glob("*"))) if (IMAGES_DIR / "malignant").exists() else 0
            if benign_count + malignant_count > 0:
                print(f"  • images/benign: {benign_count} images")
                print(f"  • images/malignant: {malignant_count} images")
    else:
        print("\n⚠️  Some downloads failed. Check the output above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
