#!/bin/bash
set -euo pipefail

# Setup conda environment for PCGCv2 post-processing pipeline.
# Auto-detects CUDA version, installs PyTorch + MinkowskiEngine.

ENV_NAME="pcgcv2"
PYTHON_VERSION="3.8"
ME_VERSION="0.5.4"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Creates the '$ENV_NAME' conda environment with PyTorch, MinkowskiEngine,"
    echo "and all dependencies for the post-processing pipeline."
    echo ""
    echo "Options:"
    echo "  --env-name NAME    Conda env name (default: $ENV_NAME)"
    echo "  --skip-me          Skip MinkowskiEngine build"
    echo "  --help             Show this help"
}

# Parse args
SKIP_ME=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --skip-me) SKIP_ME=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# -- Helpers --

log() { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

detect_cuda_version() {
    # Try nvcc first, fall back to nvidia-smi
    if command -v nvcc &>/dev/null; then
        nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+'
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+'
    else
        die "Cannot detect CUDA. Install CUDA toolkit or ensure nvcc/nvidia-smi are in PATH."
    fi
}

get_pytorch_install() {
    local cuda_ver="$1"
    local major minor
    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)

    # Map CUDA version to PyTorch install command
    # PyTorch 1.10 supports CUDA 11.3; PyTorch 1.12 supports CUDA 11.6
    # For CUDA 12.x, use PyTorch 2.x (but ME 0.5.4 may not support it)
    if [[ "$major" == "11" ]]; then
        case "$minor" in
            1|2|3)
                echo "pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch"
                ;;
            6|7)
                echo "pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch"
                ;;
            8)
                echo "pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.8 -c pytorch -c nvidia"
                ;;
            *)
                warn "CUDA 11.$minor not explicitly mapped. Trying CUDA 11.3 PyTorch build."
                echo "pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch"
                ;;
        esac
    elif [[ "$major" == "12" ]]; then
        case "$minor" in
            1|2|3|4|5|6)
                echo "pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia"
                ;;
            *)
                warn "CUDA 12.$minor not explicitly mapped. Trying CUDA 12.1 PyTorch build."
                echo "pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia"
                ;;
        esac
    else
        die "Unsupported CUDA major version: $major. Expected 11 or 12."
    fi
}

find_cuda_home() {
    # Common CUDA install locations
    for path in /usr/local/cuda /usr/local/cuda-* /opt/cuda; do
        if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
            echo "$path"
            return
        fi
    done
    die "Cannot find CUDA_HOME. Set it manually: export CUDA_HOME=/path/to/cuda"
}

# -- Main --

log "Detecting CUDA version..."
CUDA_VERSION=$(detect_cuda_version)
log "Found CUDA $CUDA_VERSION"

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

# Check GCC version (ME requires GCC <= 10 for CUDA 11.x)
if [[ "$CUDA_MAJOR" == "11" ]]; then
    if ! command -v gcc-10 &>/dev/null; then
        warn "gcc-10 not found. MinkowskiEngine build requires GCC <= 10 for CUDA 11.x."
        echo "  Install with: sudo apt install gcc-10 g++-10"
        echo "  Then re-run this script."
        if [[ "$SKIP_ME" == "false" ]]; then
            die "Cannot proceed without gcc-10."
        fi
    fi
fi

# Find conda
CONDA_SH=""
for candidate in ~/miniconda3/etc/profile.d/conda.sh ~/anaconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh; do
    if [[ -f "$candidate" ]]; then
        CONDA_SH="$candidate"
        break
    fi
done
[[ -z "$CONDA_SH" ]] && die "Cannot find conda.sh. Install miniconda first: https://docs.conda.io/en/latest/miniconda.html"

source "$CONDA_SH"

# Create env if it doesn't exist
if conda env list | grep -qw "$ENV_NAME"; then
    log "Conda env '$ENV_NAME' already exists. Activating."
else
    log "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
log "Active env: $(conda info --envs | grep '*')"

# Install PyTorch
if python -c "import torch; print(torch.__version__)" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    log "PyTorch $TORCH_VER already installed. Skipping."
else
    PYTORCH_CMD=$(get_pytorch_install "$CUDA_VERSION")
    log "Installing PyTorch for CUDA $CUDA_VERSION..."
    conda install $PYTORCH_CMD -y
fi

# Verify CUDA in PyTorch
if python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" &>/dev/null; then
    log "PyTorch CUDA verified."
else
    warn "PyTorch installed but torch.cuda.is_available() is False."
    warn "This may work if the GPU driver is available at runtime."
fi

# Install MinkowskiEngine
if [[ "$SKIP_ME" == "true" ]]; then
    log "Skipping MinkowskiEngine (--skip-me)."
elif python -c "import MinkowskiEngine" &>/dev/null; then
    log "MinkowskiEngine already installed. Skipping."
else
    log "Building MinkowskiEngine $ME_VERSION from source..."

    # Ensure openblas headers are available (required for --blas=openblas)
    if ! ldconfig -p | grep -q libopenblas && command -v apt-get &>/dev/null; then
        log "Installing libopenblas-dev..."
        apt-get install -y libopenblas-dev
    fi

    CUDA_HOME=$(find_cuda_home)
    log "Using CUDA_HOME=$CUDA_HOME"

    # Set compiler: CUDA 11.x needs gcc<=10; CUDA 12.x uses system gcc
    if [[ "$CUDA_MAJOR" == "11" ]]; then
        export CC=gcc-10
        export CXX=g++-10
        log "Using gcc-10/g++-10 for CUDA 11.x compatibility."
    fi
    export CUDA_HOME

    # Detect GPU compute capability for targeted build
    ARCH=$(python -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "")
    if [[ -n "$ARCH" ]]; then
        export TORCH_CUDA_ARCH_LIST="$ARCH"
        log "Targeting GPU compute capability: $ARCH"
    fi

    # Clone and build
    # CUDA 12.x: use main branch (has Thrust API fixes); CUDA 11.x: use tagged v0.5.4
    ME_DIR="/tmp/MinkowskiEngine_build"
    if [[ -d "$ME_DIR" ]]; then
        rm -rf "$ME_DIR"
    fi

    if [[ "$CUDA_MAJOR" == "12" ]]; then
        log "CUDA 12.x: cloning ME main branch (has Thrust compatibility fixes)..."
        git clone https://github.com/NVIDIA/MinkowskiEngine.git "$ME_DIR"
    else
        git clone --branch "v${ME_VERSION}" https://github.com/NVIDIA/MinkowskiEngine.git "$ME_DIR"
    fi
    cd "$ME_DIR"

    pip install ninja
    MAX_JOBS=4 python setup.py install --blas=openblas --force_cuda

    cd -
    rm -rf "$ME_DIR"

    # Verify
    if python -c "import MinkowskiEngine; print(f'ME {MinkowskiEngine.__version__}')"; then
        log "MinkowskiEngine installed successfully."
    else
        die "MinkowskiEngine import failed after build."
    fi
fi

# Install pip dependencies
log "Installing pip dependencies..."
pip install torchac numpy scipy tqdm plyfile open3d pyyaml

# Install pcml in editable mode
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

log "Installing pcml package from $PROJECT_ROOT..."
pip install -e "$PROJECT_ROOT"

# Make PCGCv2 executables executable
for exe in "$PROJECT_ROOT/frameworks/PCGCv2/tmc3" "$PROJECT_ROOT/frameworks/PCGCv2/pc_error_d"; do
    if [[ -f "$exe" ]]; then
        chmod +x "$exe"
        log "Made executable: $exe"
    fi
done

log "Done! Activate with: source $CONDA_SH && conda activate $ENV_NAME"
