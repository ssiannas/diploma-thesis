# Point Cloud Compression with Machine Learning

Benchmarking framework for evaluating ML-based point cloud compression methods.

**Diploma Thesis** | February 2026

---

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Test what works
python3 scripts/demo_gpcc.py                                    # G-PCC codec
python3 scripts/quick_train_simple.py --num-frames 5 --epochs 10  # Train baseline

# Run tests
make test
```

---

## Current Status

### Working Methods
| Method | Type | Status | Performance |
|--------|------|--------|-------------|
| **G-PCC (TMC13)** | Traditional | ✅ Working | 14 BPP lossless, inf PSNR |
| **Simple Baseline** | MLP Autoencoder | ✅ Trained | 28.1 dB PSNR, 0.019 loss |
| **pcc_geo_cnn_v2** | Learned CNN | 🔧 Ready* | Pending models setup |

*Adapter implemented, TensorFlow 1.15 installed in conda env `pcc_geo_cnn`

### Infrastructure
- ✅ Data loaders (PLY format)
- ✅ Metrics (PSNR, MSE, D1, D2, Hausdorff)
- ✅ Visualization tools
- ✅ Testing framework
- ✅ Framework adapter pattern

---

## Project Structure

```
diploma-thesis/
├── pcml/                      # Main package
│   ├── data/                  # Point cloud loaders
│   ├── metrics/               # Quality & compression metrics
│   ├── models/                # Baseline models
│   ├── frameworks/            # External codec adapters
│   │   ├── base.py            # Base adapter interface
│   │   ├── gpcc.py            # G-PCC (working)
│   │   └── pcc_geo_cnn_v2.py  # Learned CNN (ready)
│   ├── training/              # Training utilities
│   └── visualization/         # Plotting tools
│
├── frameworks/                # External codecs (submodules)
│   ├── mpeg-pcc-tmc13/        # G-PCC (C++)
│   └── pcc_geo_cnn_v2/        # Learned compression (TF 1.15)
│
├── scripts/                   # Executable scripts
│   ├── demo_gpcc.py           # Test G-PCC
│   ├── test_gpcc_configs.py   # Multi-config G-PCC
│   ├── quick_train_simple.py  # Train baseline
│   └── test_pcc_geo_cnn_v2.py # Test learned CNN
│
├── datasets/                  # Point cloud data (gitignored)
│   └── 8iVFB_small/           # Test data (included)
│
├── models/                    # Checkpoints (gitignored)
├── results/                   # Outputs (gitignored)
└── tests/                     # Unit tests
```

---

## Usage

### Test G-PCC (Traditional Codec)
```bash
source .venv/bin/activate

# Single test
python3 scripts/demo_gpcc.py

# Multiple configs
python3 scripts/test_gpcc_configs.py
```

### Train Baseline Model
```bash
source .venv/bin/activate

# Quick test (5 frames, 10 epochs, ~5 min)
python3 scripts/quick_train_simple.py --num-frames 5 --epochs 10

# Full training (20 frames, 50 epochs, ~30 min)
python3 scripts/quick_train_simple.py
```

### Test Learned Compression (pcc_geo_cnn_v2)
```bash
# Requires: TensorFlow 1.15 (conda env), pretrained models
source .venv/bin/activate
python3 scripts/test_pcc_geo_cnn_v2.py
```

**Setup if needed**:
```bash
# Create conda environment
conda create -n pcc_geo_cnn python=3.6.9 -y
conda activate pcc_geo_cnn
conda install tensorflow-gpu=1.15.0 -c conda-forge -y
pip install tensorflow-compression==1.3 pandas matplotlib numpy plyfile pyyaml tqdm

# Models should be symlinked at: frameworks/pcc_geo_cnn_v2/models/
# Expected structure: c1/, c2/, c3/, c4/, c5/, c6/ (quality levels)
```

---

## Development

```bash
# Run tests
make test              # All tests
make test-verbose      # Verbose output
make coverage          # Coverage report

# Code quality
make lint              # flake8
make format            # black formatter

# Other commands
make help              # Show all targets
make clean             # Remove cache/artifacts
```

---

## Environments

**Main (.venv)**: Python 3.10, PyTorch 2.0
- Default for all work
- Baseline models, G-PCC, benchmarking
- Use: `source .venv/bin/activate`

**Conda (pcc_geo_cnn)**: Python 3.6, TensorFlow 1.15
- Only for pcc_geo_cnn_v2 framework
- Called via subprocess by main env
- Use: `conda activate pcc_geo_cnn`

---

## Metrics

### Training
- **Chamfer Distance**: Bidirectional mean distance for smooth gradients

### Evaluation (MPEG Standard)
- **MSE/PSNR**: Overall quality assessment
- **D1**: One-directional max distance
- **D2**: Bidirectional max distance (Hausdorff)
- **BPP**: Bits per point (compression rate)

See `METRICS_RATIONALE.md` for detailed explanation of why we use different metrics for training vs evaluation.

---

## Documentation

- `README.md` (this file) - Start here
- `METRICS_RATIONALE.md` - Why our metric choices
- `.docs/` - Additional documentation
  - `CORE_PHILOSOPHY.md` - Development principles
  - `CODE_STYLE.md` - Style guidelines
  - `PROJECT_TIMELINE.md` - Progress tracker
  - `SETUP.md` - Detailed setup
  - `FRAMEWORKS.md` - Framework comparison

---

## Results

Current performance (on longdress_vox10_1300.ply):

| Method | BPP | PSNR (dB) | Status |
|--------|-----|-----------|--------|
| G-PCC (lossless) | 14.0 | ∞ | ✅ Tested |
| Simple Baseline | TBD | 28.1 | ✅ Trained |
| pcc_geo_cnn_v2 | TBD | TBD | 🔧 Ready |

---

## Next Steps

1. Setup pcc_geo_cnn_v2 models (symlink or download)
2. Test pcc_geo_cnn_v2 compression
3. Create benchmark comparison script
4. Generate rate-distortion curves
5. Full evaluation on test set

---

## License

[Specify license]

---

## Citation

```bibtex
@mastersthesis{thesis_2026,
  title={Point Cloud Compression using Machine Learning},
  author={Your Name},
  year={2026},
  school={Your University}
}
```
