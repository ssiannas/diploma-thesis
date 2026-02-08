# PCML - Point Cloud Machine Learning Framework

A comprehensive benchmarking framework for point cloud compression using machine learning techniques. This project is part of a diploma thesis focused on evaluating and comparing various ML-based point cloud compression methods.

## Overview

PCML (Point Cloud Machine Learning) provides a unified framework for:
- Loading and preprocessing point cloud datasets (JPEG Pleno, 8iVFB, etc.)
- Implementing and benchmarking compression algorithms
- Computing quality metrics (geometry and color)
- Analyzing compression performance

## Project Structure

```
diploma-thesis/
├── pcml/                   # Main package source code
│   ├── __init__.py
│   ├── data/              # Data loaders and preprocessing
│   │   ├── __init__.py
│   │   ├── loaders.py     # Dataset loaders
│   │   └── types.py       # Point cloud data structures
│   └── metrics/           # Quality and compression metrics
│       ├── __init__.py
│       ├── compression.py # Compression metrics
│       └── quality.py     # Quality metrics (PSNR, Hausdorff, etc.)
├── tests/                 # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py        # Pytest fixtures
│   ├── test_data.py       # Data loader tests
│   └── test_metrics.py    # Metrics tests
├── scripts/               # Utility scripts
│   ├── validate_environment.py
│   └── build_frameworks.bash
├── notebooks/             # Jupyter notebooks for exploration
├── configs/               # Configuration files (YAML/Hydra)
├── datasets/              # Local datasets
├── models/                # Trained model checkpoints
├── results/               # Experiment outputs
├── docs/                  # Documentation
├── frameworks/            # ML compression frameworks (submodules)
├── pyproject.toml         # Project metadata and dependencies
├── setup.py               # Installation script
├── requirements.txt       # Core dependencies
├── requirements-dev.txt   # Development dependencies
├── Makefile               # Useful commands
└── README.md             # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full dependency list

## Installation

### Quick Setup

The easiest way to set up the development environment:

```bash
# Create virtual environment and install all dependencies
make setup

# Activate the virtual environment
source .venv/bin/activate
```

### Manual Setup

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install package
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,jupyter]"
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Installation Options

```bash
# Install core package only
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with Jupyter support
pip install -e ".[jupyter]"

# Install everything
pip install -e ".[all]"
```

## Usage

### Data Loading

```python
from pcml.data import PLYPointCloudLoader
from pcml.data.loaders import JPEGPleno8iVFBSequence, PointCloudDataset

# Load a single PLY file
loader = PLYPointCloudLoader(verbose=True)
point_cloud = loader.load("path/to/file.ply")

# Load a sequence from JPEG Pleno dataset
sequence = JPEGPleno8iVFBSequence(
    root_dir="/mnt/shared-dataset/jpeg-pleno",
    sequence_name="longdress"
)

# Create PyTorch dataset
dataset = PointCloudDataset(
    sequence=sequence,
    frame_indices=list(range(100)),
    normalize=True,
    num_points=100000
)
```

### Metrics Calculation

```python
from pcml.metrics import CompressionMetrics
from pcml.metrics.compression import CompressionCalculator
from pcml.metrics.quality import GeometryQualityCalculator, ColorQualityCalculator

# Compression metrics
metrics = CompressionCalculator.calculate_from_sizes(
    original_size_bytes=1000000,
    compressed_size_bytes=100000,
    num_points=100000
)
print(f"Compression ratio: {metrics.compression_ratio:.4f}")
print(f"Bits per point: {metrics.bits_per_point:.2f}")

# Geometry quality metrics
geo_metrics = GeometryQualityCalculator.calculate_all(
    original_points,
    reconstructed_points
)
print(f"PSNR: {geo_metrics.psnr:.2f} dB")
print(f"Hausdorff: {geo_metrics.hausdorff:.6f}")

# Color quality metrics
color_metrics = ColorQualityCalculator.calculate_all(
    original_colors,
    reconstructed_colors
)
print(f"Color PSNR: {color_metrics.psnr:.2f} dB")
```

## Development

### Running Tests

```bash
# Run all tests
make test

# Run tests with verbose output
make test-verbose

# Run tests with coverage
make test-coverage

# Run specific test file
pytest tests/test_data.py -v
```

### Code Quality

```bash
# Format code (black + isort)
make format

# Run linters (flake8 + pylint)
make lint

# Run type checker (mypy)
make type-check

# Run all checks
make check
```

### Available Make Commands

```bash
make help           # Show all available commands
make venv           # Create virtual environment
make install        # Install package in editable mode
make install-dev    # Install with dev dependencies
make setup          # Full setup from scratch
make test           # Run tests
make test-coverage  # Run tests with coverage
make format         # Format code
make lint           # Run linters
make type-check     # Run type checker
make clean          # Remove build artifacts
make docs           # Build documentation
```

## Datasets

### Local Datasets

Small test datasets are stored in `datasets/`:
- `8iVFB_small/`: Subset of 8iVFB dataset for quick testing

### Shared Datasets

Large datasets are stored on shared storage:
- JPEG Pleno dataset: `/mnt/shared-dataset/jpeg-pleno/`
  - Sequences: longdress, loot, redandblack, soldier

## Testing

The test suite includes:
- Unit tests for data loaders
- Unit tests for metrics calculation
- Integration tests for full workflows
- Fixtures for common test data (see `tests/conftest.py`)

Tests automatically skip if required datasets are not available.

## Project Status

This project is under active development as part of a diploma thesis.

### Completed
- Data loading infrastructure (PLY, JPEG Pleno)
- Compression metrics calculation
- Geometry and color quality metrics
- PyTorch dataset wrappers
- Testing framework

### In Progress
- ML compression model implementations
- Benchmarking pipeline
- Visualization tools

### Planned
- Model training scripts
- Automated benchmarking
- Results analysis and visualization
- Documentation website

## Contributing

As this is a thesis project, external contributions are not currently accepted. However, feedback and suggestions are welcome!

## License

MIT License - See LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{pcml2025,
  title={Point Cloud Compression using Machine Learning: A Benchmarking Framework},
  author={Student Name},
  year={2025},
  school={University Name}
}
```

## Contact

For questions or feedback, please contact: student@example.com

## Acknowledgments

- JPEG Pleno dataset providers
- 8iVFB dataset creators
- Open3D library maintainers
- PyTorch team
