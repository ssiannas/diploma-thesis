#!/bin/bash

# Setup script for pcc_geo_cnn_v2 framework
# Quach et al., "Improved Deep Point Cloud Geometry Compression"
# Uses conda install for faster TensorFlow download

set -e

echo "=================================================="
echo "Setting up pcc_geo_cnn_v2 Framework"
echo "=================================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK_DIR="$PROJECT_ROOT/frameworks/pcc_geo_cnn_v2"

echo "Project root: $PROJECT_ROOT"
echo "Framework directory: $FRAMEWORK_DIR"
echo ""

# Create conda environment
ENV_NAME="pcc_geo_cnn"
echo "Creating conda environment: $ENV_NAME"
echo "Python 3.6.9, TensorFlow 1.15.0, CUDA 10.0"
echo ""
echo "NOTE: Using conda install for TensorFlow (faster than pip)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create environment
echo ""
echo "Step 1/4: Creating conda environment..."
conda create -n "$ENV_NAME" python=3.6.9 -y

# Activate and install
echo ""
echo "Step 2/4: Installing TensorFlow 1.15 via conda (faster)..."
echo ""

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install TensorFlow with conda (much faster than pip!)
conda install tensorflow-gpu=1.15.0 -c conda-forge -y

# Install tensorflow-compression
echo ""
echo "Step 3/4: Installing tensorflow-compression..."
pip install tensorflow-compression==1.3

# Install other requirements
echo ""
echo "Step 4/4: Installing other dependencies..."
pip install pandas matplotlib numpy plyfile pyyaml tqdm

echo ""
echo "=================================================="
echo "Environment setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download pretrained models:"
echo "   Link: https://drive.google.com/file/d/18uHmr0ZpgFLeL9Y5TUFTsQkRfz4XpQdJ/view"
echo "   Extract to: $FRAMEWORK_DIR/models/"
echo "   Should create: c1/, c2/, c3/, c4/, c5/, c6/ directories"
echo ""
echo "2. Activate environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "3. Test framework:"
echo "   cd $FRAMEWORK_DIR"
echo "   ls models/  # Verify models exist"
echo "   python test.py compress --help"
echo ""
echo "4. Return to main venv:"
echo "   conda deactivate"
echo "   source $PROJECT_ROOT/.venv/bin/activate"
echo ""
echo "See .docs/SETUP.md for detailed instructions."
echo ""
