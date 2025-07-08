#!/usr/bin/env python3
"""
Environment validation script for thesis setup
"""
import sys
import subprocess
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    packages = ['torch', 'numpy', 'open3d', 'matplotlib', 'pandas']
    print("=== Python Packages ===")
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")

def check_frameworks():
    """Check if frameworks are properly set up"""
    print("\n=== Frameworks ===")
    
    # Check TMC13
    tmc3_path = Path("frameworks/mpeg-pcc-tmc13/build/tmc3")
    if tmc3_path.exists():
        print("✓ TMC13 (G-PCC)")
    else:
        print("✗ TMC13 - Build required")
    
    # Check PccAI
    pccai_path = Path("frameworks/PccAI")
    if pccai_path.exists():
        print("✓ PccAI framework")
    else:
        print("✗ PccAI - Clone required")
    
    # Check ML repositories
    repos = [
        "frameworks/pcc_geo_cnn_v2",
        "frameworks/Point-cloud-compression-by-RNN",
        "frameworks/PCGCv1"
    ]
    
    for repo in repos:
        if Path(repo).exists():
            print(f"✓ {Path(repo).name}")
        else:
            print(f"✗ {Path(repo).name} - Clone required")

def check_datasets():
    """Check if datasets are available"""
    print("\n=== Datasets ===")
    
    dataset_files = [
        "datasets/8iVFB_small/longdress_vox10_1300.ply",
        "datasets/8iVFB_small/loot_vox10_1200.ply",
        "datasets/8iVFB_small/redandblack_vox10_1550.ply",
        "datasets/8iVFB_small/soldier_vox10_0690.ply"
    ]
    
    for dataset in dataset_files:
        if Path(dataset).exists():
            print(f"✓ {Path(dataset).name}")
        else:
            print(f"✗ {Path(dataset).name} - Download required")

def check_gpu():
    """Check GPU availability"""
    print("\n=== GPU Setup ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available - {torch.cuda.device_count()} device(s)")
            print(f"  Current device: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA not available - Using CPU only")
    except ImportError:
        print("✗ PyTorch not installed")

if __name__ == "__main__":
    print("Environment Validation for ML Geometry Compression Thesis")
    print("=" * 60)
    
    check_python_packages()
    check_frameworks()
    check_datasets()
    check_gpu()
    
    print("\n" + "=" * 60)
    print("Validation complete!")
