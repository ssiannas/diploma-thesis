#!/bin/bash
# Install dependencies for pcc_geo_cnn_v2

set -e

echo "Installing pcc_geo_cnn_v2 dependencies..."
echo "Make sure internet is working!"
echo ""

# Source conda
source /home/ssiannas/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate pcc_geo_cnn

echo "Step 1/3: Installing TensorFlow 1.15 via conda..."
conda install tensorflow-gpu=1.15.0 -c conda-forge -y

echo ""
echo "Step 2/3: Installing tensorflow-compression..."
pip install tensorflow-compression==1.3

echo ""
echo "Step 3/3: Installing other dependencies..."
pip install pandas matplotlib numpy plyfile pyyaml tqdm

echo ""
echo "✅ Installation complete!"
echo ""
echo "Test with: python -c 'import tensorflow as tf; print(tf.__version__)'"
