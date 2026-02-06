#!/usr/bin/env python3
"""
Setup Script for Breast Cancer AI Diagnostic System
Automated installation and environment configuration
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_NAME = "breast_cancer_ai"
PROJECT_VERSION = "1.0.0"

# Core dependencies (always required)
CORE_REQUIREMENTS = [
    "streamlit>=1.30.0",
    "biopython>=1.80",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pillow>=10.0.0",
    "matplotlib>=3.7.0",
    "scikit-learn>=1.3.0",
]

# PyTorch dependencies (for deep learning)
PYTORCH_REQUIREMENTS = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
]

# NLP/NER dependencies
NLP_REQUIREMENTS = [
    "transformers>=4.30.0",
    "spacy>=3.6.0",
]

# Visualization dependencies
VIZ_REQUIREMENTS = [
    "networkx>=3.0",
    "pyvis>=0.3.0",
    "wordcloud>=1.9.0",
]

# Development dependencies
DEV_REQUIREMENTS = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]

# Project directories to create
PROJECT_DIRS = [
    "data",
    "data/images",
    "data/images/benign",
    "data/images/malignant",
    "models",
    "logs",
    "database",
    "utils",
    "outputs",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_step(step: int, total: int, text: str):
    """Print step progress"""
    print(f"\n[{step}/{total}] {text}")


def run_command(cmd: list, check: bool = True) -> bool:
    """Run shell command"""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"  Error: {str(e)}")
        return False


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"Error: Python 3.9+ required. You have {version.major}.{version.minor}")
        return False
    print(f"  Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def check_pip():
    """Check pip availability"""
    try:
        import pip
        print(f"  pip available ✓")
        return True
    except ImportError:
        print("  Error: pip not available")
        return False


def install_packages(packages: list, label: str = "packages") -> bool:
    """Install Python packages"""
    print(f"  Installing {label}...")
    
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"  {label} installed ✓")
        return True
    else:
        print(f"  Warning: Some {label} may have failed")
        if result.stderr:
            # Print only warnings, not full error
            for line in result.stderr.split('\n')[:5]:
                if line.strip():
                    print(f"    {line}")
        return False


def create_directories(base_path: Path):
    """Create project directories"""
    for dir_path in PROJECT_DIRS:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    print("  Directories created ✓")


def write_requirements_file(base_path: Path, install_type: str):
    """Write requirements.txt based on install type"""
    requirements = CORE_REQUIREMENTS.copy()
    
    if install_type in ['full', 'pytorch']:
        requirements.extend(PYTORCH_REQUIREMENTS)
    
    if install_type in ['full', 'nlp']:
        requirements.extend(NLP_REQUIREMENTS)
    
    if install_type == 'full':
        requirements.extend(VIZ_REQUIREMENTS)
    
    req_path = base_path / "requirements_auto.txt"
    with open(req_path, 'w') as f:
        f.write(f"# Auto-generated requirements for {PROJECT_NAME}\n")
        f.write(f"# Install type: {install_type}\n\n")
        f.write("\n".join(requirements))
    
    print(f"  Requirements written to {req_path} ✓")
    return req_path


def verify_installation():
    """Verify key packages are installed"""
    checks = {
        'streamlit': False,
        'biopython': False,
        'numpy': False,
        'pandas': False,
        'PIL': False,
        'sklearn': False,
    }
    
    # Try core imports
    try:
        import streamlit
        checks['streamlit'] = True
    except ImportError:
        pass
    
    try:
        from Bio import Entrez
        checks['biopython'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        checks['numpy'] = True
    except ImportError:
        pass
    
    try:
        import pandas
        checks['pandas'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        checks['PIL'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        checks['sklearn'] = True
    except ImportError:
        pass
    
    # Optional
    optional = {}
    
    try:
        import torch
        optional['pytorch'] = True
    except ImportError:
        optional['pytorch'] = False
    
    try:
        import transformers
        optional['transformers'] = True
    except ImportError:
        optional['transformers'] = False
    
    try:
        import networkx
        optional['networkx'] = True
    except ImportError:
        optional['networkx'] = False
    
    return checks, optional


def fix_ssl_certificates():
    """Fix SSL certificates on macOS"""
    if sys.platform == 'darwin':
        print("  Attempting SSL certificate fix for macOS...")
        
        # Try to find and run Install Certificates.command
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        cert_paths = [
            f"/Applications/Python {python_version}/Install Certificates.command",
            f"/Applications/Python{python_version}/Install Certificates.command",
        ]
        
        for cert_path in cert_paths:
            if os.path.exists(cert_path):
                print(f"  Running: {cert_path}")
                try:
                    subprocess.run(['bash', cert_path], check=True, capture_output=True)
                    print("  SSL certificates installed ✓")
                    return True
                except:
                    pass
        
        # Fallback: install certifi
        try:
            import certifi
            print(f"  certifi available at: {certifi.where()}")
            return True
        except ImportError:
            install_packages(['certifi'], 'SSL certificates')
            return True
    
    return True


def create_env_file(base_path: Path):
    """Create .env file with configuration"""
    env_content = f"""# {PROJECT_NAME} Environment Configuration
# Generated by setup.py

# NCBI API Key (get from https://www.ncbi.nlm.nih.gov/account/settings/)
NCBI_API_KEY=ae83b1da74148ccbacc801302448d41ae708

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# Model Paths
IMAGING_MODEL_PATH=models/imaging_model.pth
OMICS_MODEL_PATH=models/omics_model.pkl

# Database
DATABASE_PATH=database/breast_cancer.db

# Logging
LOG_LEVEL=INFO
"""
    
    env_path = base_path / ".env"
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"  Environment file created: {env_path} ✓")


def create_run_script(base_path: Path):
    """Create convenient run script"""
    script_content = """#!/bin/bash
# Run the Breast Cancer AI Diagnostic System

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for product version
if [ "$1" == "product" ]; then
    echo "Running Product Version..."
    streamlit run app_product.py --server.port 8501
else
    echo "Running Standard Version..."
    streamlit run app.py --server.port 8501
fi
"""
    
    script_path = base_path / "run_app.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"  Run script created: {script_path} ✓")


# ============================================================
# MAIN SETUP FUNCTION
# ============================================================

def setup(
    install_type: str = 'core',
    skip_pytorch: bool = False,
    skip_nlp: bool = False,
    fix_ssl: bool = True,
    create_venv: bool = False,
    verbose: bool = False
):
    """
    Run complete setup
    
    Args:
        install_type: 'core', 'full', 'pytorch', 'nlp'
        skip_pytorch: Skip PyTorch installation
        skip_nlp: Skip NLP packages
        fix_ssl: Fix SSL certificates
        create_venv: Create virtual environment
        verbose: Verbose output
    """
    print_header(f"{PROJECT_NAME} Setup v{PROJECT_VERSION}")
    
    base_path = Path(__file__).parent.resolve()
    total_steps = 8
    
    # Step 1: Check Python
    print_step(1, total_steps, "Checking Python environment")
    if not check_python_version():
        return False
    if not check_pip():
        return False
    
    # Step 2: Create virtual environment (optional)
    if create_venv:
        print_step(2, total_steps, "Creating virtual environment")
        venv_path = base_path / "venv"
        if not venv_path.exists():
            run_command([sys.executable, "-m", "venv", str(venv_path)])
            print(f"  Virtual environment created at: {venv_path} ✓")
            print("  Activate with: source venv/bin/activate")
        else:
            print("  Virtual environment already exists ✓")
    else:
        print_step(2, total_steps, "Skipping virtual environment")
    
    # Step 3: Create directories
    print_step(3, total_steps, "Creating project directories")
    create_directories(base_path)
    
    # Step 4: Install core packages
    print_step(4, total_steps, "Installing core dependencies")
    install_packages(CORE_REQUIREMENTS, "core packages")
    
    # Step 5: Install optional packages
    print_step(5, total_steps, "Installing optional dependencies")
    
    if install_type == 'full' or (install_type == 'pytorch' and not skip_pytorch):
        print("  Installing PyTorch...")
        install_packages(PYTORCH_REQUIREMENTS, "PyTorch")
    
    if install_type == 'full' or (install_type == 'nlp' and not skip_nlp):
        print("  Installing NLP packages...")
        install_packages(NLP_REQUIREMENTS, "NLP packages")
    
    if install_type == 'full':
        print("  Installing visualization packages...")
        install_packages(VIZ_REQUIREMENTS, "visualization packages")
    
    # Step 6: Fix SSL (macOS)
    print_step(6, total_steps, "Configuring SSL certificates")
    if fix_ssl:
        fix_ssl_certificates()
    else:
        print("  Skipped")
    
    # Step 7: Create config files
    print_step(7, total_steps, "Creating configuration files")
    create_env_file(base_path)
    create_run_script(base_path)
    write_requirements_file(base_path, install_type)
    
    # Step 8: Verify installation
    print_step(8, total_steps, "Verifying installation")
    core_checks, optional_checks = verify_installation()
    
    print("\n  Core packages:")
    for pkg, status in core_checks.items():
        status_str = "✓" if status else "✗"
        print(f"    {pkg}: {status_str}")
    
    print("\n  Optional packages:")
    for pkg, status in optional_checks.items():
        status_str = "✓" if status else "not installed"
        print(f"    {pkg}: {status_str}")
    
    # Summary
    print_header("Setup Complete!")
    
    all_core_ok = all(core_checks.values())
    
    if all_core_ok:
        print("\n✅ All core dependencies installed successfully!")
        print("\nTo run the application:")
        print(f"  cd {base_path}")
        print("  streamlit run app.py")
        print("\nOr use the run script:")
        print("  ./run_app.sh")
        print("  ./run_app.sh product  # For product version")
    else:
        print("\n⚠️  Some core dependencies may be missing.")
        print("Try running: pip install -r requirements.txt")
    
    return all_core_ok


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description=f"Setup script for {PROJECT_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Install types:
  core     - Basic dependencies only (default)
  full     - All dependencies including PyTorch and NLP
  pytorch  - Core + PyTorch for deep learning
  nlp      - Core + NLP packages for entity extraction

Examples:
  python setup.py                    # Core installation
  python setup.py --install-type full    # Full installation
  python setup.py --create-venv      # Create virtual environment
  python setup.py --skip-pytorch     # Skip PyTorch
        """
    )
    
    parser.add_argument(
        '--install-type',
        choices=['core', 'full', 'pytorch', 'nlp'],
        default='core',
        help='Type of installation (default: core)'
    )
    parser.add_argument(
        '--skip-pytorch',
        action='store_true',
        help='Skip PyTorch installation'
    )
    parser.add_argument(
        '--skip-nlp',
        action='store_true',
        help='Skip NLP package installation'
    )
    parser.add_argument(
        '--no-ssl-fix',
        action='store_true',
        help='Skip SSL certificate fix'
    )
    parser.add_argument(
        '--create-venv',
        action='store_true',
        help='Create virtual environment'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    success = setup(
        install_type=args.install_type,
        skip_pytorch=args.skip_pytorch,
        skip_nlp=args.skip_nlp,
        fix_ssl=not args.no_ssl_fix,
        create_venv=args.create_venv,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
