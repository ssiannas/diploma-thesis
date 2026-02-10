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
| **pcc_geo_cnn_v2** | Learned CNN | ✅ Working | 36 BPP, 53 dB PSNR (c1 model) |

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
│       └── models/            # → Symlink to Shared Drive
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

### External Storage (Shared Drive)

Large datasets and pretrained models are stored on an external drive:

**Location**: `/media/ssiannas/Shared Driv/` (always mounted)

**Contents**:
- `pcc_geo_cnn_v2/models/` - Pretrained CNN models (c1, c2, c3p variants, c4-ws)
- `pcc_geo_cnn_v2/` - Sample point clouds (longdress, loot, soldier, etc.)
- `jpeg-pleno/` - 8iVFB dataset (full JPEG Pleno sequences: longdress, loot, redandblack, soldier)
- Other pretrained models and datasets

**Note**: 8iVFB dataset = JPEG Pleno point cloud dataset. Small test subset in `datasets/8iVFB_small/`, full dataset on Shared Drive.

**Symlinks**:
- `frameworks/pcc_geo_cnn_v2/models` → `/media/ssiannas/Shared Driv/pcc_geo_cnn_v2/models`

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
# Uses TensorFlow 1.15 in conda env (called via subprocess)
source .venv/bin/activate
python3 scripts/test_pcc_geo_cnn_v2.py
```

**Results**: ~125s compression, 53 dB PSNR, 36 BPP (c1 model on 10K points)

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

Current benchmark results (longdress_vox10_1300.ply, 5K points):

| Method | Config | BPP | PSNR (dB) | Compression Time | Status |
|--------|--------|-----|-----------|------------------|--------|
| G-PCC | lossless | 14.0 | ∞ | <1s | ✅ Working |
| pcc_geo_cnn_v2 | c1 | 65.95 | 51.64 | 110s | ✅ Tested |
| pcc_geo_cnn_v2 | c2 | 31.97 | 52.34 | 121s | ✅ Tested |
| pcc_geo_cnn_v2 | c3p | 50.53 | 53.64 | 230s | ✅ Tested |
| Simple Baseline | trained | TBD | 28.1 | ~30min (training) | ✅ Trained |

**Best quality**: c3p (53.64 dB @ 50.5 BPP)
**Most efficient**: c2 (52.34 dB @ 32.0 BPP)

### Running Complete Benchmark

To generate full rate-distortion curves with all methods:

```bash
source .venv/bin/activate

# Complete benchmark (all methods, 10K points, ~1-2 hours)
python3 scripts/benchmark_all_methods.py \
  --num-points 10000 \
  --cnn-models c1 c2 c3 c3p \
  --output results/benchmark/full_comparison.json

# Generate visualizations
python3 scripts/plot_rd_curves.py \
  --input results/benchmark/full_comparison.json \
  --multi  # Creates multi-metric comparison plot
```

**Available models:**
- G-PCC: lossless (always included unless `--skip-gpcc`)
- pcc_geo_cnn_v2: c1, c2, c3, c3p (c3 not yet tested)
- Simple Baseline: not in benchmark script yet

**Note**: c4-ws exists in models directory but not in framework code (cannot use without modification).

---

## Next Steps

### Immediate (Ready to Run)
1. ✅ ~~Setup all frameworks~~
2. ✅ ~~Test pcc_geo_cnn_v2 models (c1, c2, c3p)~~
3. ✅ ~~Create benchmark comparison script~~
4. ✅ ~~Generate RD curve visualization~~
5. **Run complete benchmark** (c1, c2, c3, c3p + G-PCC on 10K points)

### Short-term
6. Test c3 model (only untested model)
7. Add Simple Baseline to benchmark script
8. Multi-frame evaluation (all 8iVFB sequences)
9. Full-scale testing (100K+ points)

### Long-term
10. Train new models on full 8iVFB dataset
11. Implement advanced architectures
12. Final thesis evaluation and comparison

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
