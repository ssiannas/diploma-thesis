# Point Cloud Compression with Machine Learning

Benchmarking framework for evaluating ML-based point cloud compression methods.

**Diploma Thesis** | February 2026

---

## Quick Start

### 1. Setup Main Environment (PyTorch 2.0)

```bash
# Clone repository
git clone <repo-url>
cd diploma-thesis

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Run tests
make test
```

### 2. Test What's Working

```bash
# Activate venv
source .venv/bin/activate

# Test G-PCC (traditional codec)
python3 scripts/demo_gpcc.py

# Train simple baseline
python3 scripts/quick_train_simple.py --num-frames 5 --epochs 10
```

---

## Project Structure

```
diploma-thesis/
├── pcml/                      # Main package
│   ├── data/                  # Data loaders (PLY, etc.)
│   ├── metrics/               # PSNR, MSE, D1, D2
│   ├── models/                # Baseline models
│   ├── frameworks/            # Framework adapters
│   ├── training/              # Training utilities
│   └── visualization/         # Plotting tools
│
├── frameworks/                # External codecs (git submodules)
│   ├── mpeg-pcc-tmc13/       # G-PCC (C++)
│   ├── pcc_geo_cnn_v2/       # Learned compression (TF 1.15)
│   ├── learned-pcc/          # PointNet compression (PyTorch)
│   └── PccAI/                # MPEG framework (PyTorch)
│
├── scripts/                   # Training/testing scripts
├── datasets/                  # Point cloud data (gitignored)
├── models/                    # Checkpoints (gitignored)
├── results/                   # Outputs (gitignored)
│
├── .docs/                     # Documentation
├── tests/                     # Unit tests
├── requirements.txt           # Main dependencies
└── Makefile                   # Common commands
```

---

## Supported Methods

| Method | Type | Status | Environment |
|--------|------|--------|-------------|
| **G-PCC (TMC13)** | Traditional | ✅ Working | Main |
| **Simple Baseline** | MLP Autoencoder | ✅ Trained | Main |
| **pcc_geo_cnn_v2** | Learned (CNN) | 🔄 Setup | Conda (TF 1.15) |
| **learned-pcc** | PointNet + CompressAI | ⏳ Needs training | Conda (PyTorch 1.13) |
| **PccAI** | Framework | ⏳ Custom | Conda (PyTorch 1.8) |

---

## Usage

### Test G-PCC (Ready Now)

```bash
source .venv/bin/activate

# Single test
python3 scripts/demo_gpcc.py

# Multiple configs
python3 scripts/test_gpcc_configs.py

# Results in: results/gpcc_demo/
```

### Train Baseline Model

```bash
source .venv/bin/activate

# Quick test (5 frames, 10 epochs, ~5 min)
python3 scripts/quick_train_simple.py --num-frames 5 --epochs 10

# Full training (20 frames, 50 epochs, ~30 min)
python3 scripts/quick_train_simple.py

# Checkpoints in: models/baselines/simple_quick/
```

### Setup Additional Frameworks

For frameworks requiring separate environments (TensorFlow 1.x):

```bash
# pcc_geo_cnn_v2 (has pretrained models)
./scripts/setup_pcc_geo_cnn_v2.sh

# Then download models from Google Drive
# See: .docs/SETUP.md
```

---

## Datasets

### Recommended: 8iVFB Dataset

```bash
# Download from: http://plenodb.jpeg.org/
# Extract to: datasets/8iVFB/

# Small subset for testing (already included):
datasets/8iVFB_small/
├── longdress_vox10_1300.ply
├── longdress_vox10_1301.ply
└── ...
```

---

## Development

### Run Tests

```bash
make test              # Run all tests
make test-verbose      # Verbose output
make coverage          # Generate coverage report
```

### Code Quality

```bash
make lint              # Run flake8
make format            # Run black formatter
```

### Useful Commands

```bash
make help              # Show all commands
make clean             # Remove cache/artifacts
make train-baseline    # Train baseline model
```

---

## Environment Strategy

**Main Environment (.venv)**: PyTorch 2.0, Python 3.10
- Use for: baseline models, G-PCC, benchmarking
- **Default for all work**

**Conda Environments**: Only for legacy frameworks
- `pcc_geo_cnn` - TensorFlow 1.15, Python 3.6
- `learned_pcc` - PyTorch 1.13, Python 3.10
- `pccai` - PyTorch 1.8, Python 3.8

**Activate main env:**
```bash
source .venv/bin/activate
```

**Activate conda env (when needed):**
```bash
conda activate pcc_geo_cnn
```

---

## Results

Current performance (on longdress_vox10_1300.ply):

| Method | BPP | PSNR (dB) | MSE | Notes |
|--------|-----|-----------|-----|-------|
| G-PCC (lossless) | 14.0 | ∞ | 0.0 | Perfect reconstruction |
| Simple Baseline | TBD | 28.1 | 0.019 | Trained on 20 frames |

---

## Documentation

See `.docs/` for detailed documentation:

- **SETUP.md** - Detailed setup instructions
- **FRAMEWORKS.md** - Framework comparison & setup
- **CORE_PHILOSOPHY.md** - Development principles
- **PROJECT_TIMELINE.md** - Current progress & roadmap

---

## Citation

```bibtex
@mastersthesis{your_thesis_2026,
  title={Point Cloud Compression using Machine Learning},
  author={Your Name},
  year={2026},
  school={Your University}
}
```

---

## License

[Your license here]

---

## Conda Auto-Activation (Disabled)

We disabled conda base auto-activation to keep the main venv as default.

**To re-enable** (not recommended):
```bash
conda config --set auto_activate_base true
```

**Manual activation when needed:**
```bash
conda activate pcc_geo_cnn  # Or other env
```
