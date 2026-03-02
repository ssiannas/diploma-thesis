#!/bin/bash
set -euo pipefail

# Create dataset symlinks pointing to external storage paths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASETS_DIR="$PROJECT_ROOT/datasets"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Creates symlinks in datasets/ pointing to external dataset locations."
    echo ""
    echo "Options:"
    echo "  --8ivfb PATH      Path to 8iVFB PLY dataset (-> datasets/8iVFB)"
    echo "  --decoded PATH    Path to PCGCv2 decoded training data (-> datasets/pcgcv2_decoded)"
    echo "  --modelnet PATH   Path to ModelNet40 dataset (-> datasets/ModelNet40)"
    echo "  --help             Show this help"
    echo ""
    echo "Example:"
    echo "  $0 --8ivfb /data/jpeg-pleno --decoded /data/pcgcv2_training_data"
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

# Parse args
PATH_8IVFB=""
PATH_DECODED=""
PATH_MODELNET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --8ivfb) PATH_8IVFB="$2"; shift 2 ;;
        --decoded) PATH_DECODED="$2"; shift 2 ;;
        --modelnet) PATH_MODELNET="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

log() { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

create_symlink() {
    local target="$1"
    local link_name="$2"
    local link_path="$DATASETS_DIR/$link_name"

    # Validate target exists
    if [[ ! -d "$target" ]]; then
        echo "ERROR: Target directory does not exist: $target" >&2
        return 1
    fi

    # Skip if symlink already exists and points to same target
    if [[ -L "$link_path" ]]; then
        existing=$(readlink -f "$link_path")
        resolved=$(readlink -f "$target")
        if [[ "$existing" == "$resolved" ]]; then
            log "$link_name -> $target (already exists, skipping)"
            return 0
        else
            warn "$link_name exists but points to $existing (expected $resolved). Skipping."
            return 0
        fi
    fi

    # Fail if non-symlink file/dir exists
    if [[ -e "$link_path" ]]; then
        echo "ERROR: $link_path exists and is not a symlink. Remove it manually." >&2
        return 1
    fi

    ln -s "$target" "$link_path"
    log "$link_name -> $target"
}

# Create datasets dir
mkdir -p "$DATASETS_DIR"

ERRORS=0

if [[ -n "$PATH_8IVFB" ]]; then
    create_symlink "$PATH_8IVFB" "8iVFB" || ((ERRORS++))
fi

if [[ -n "$PATH_DECODED" ]]; then
    create_symlink "$PATH_DECODED" "pcgcv2_decoded" || ((ERRORS++))
fi

if [[ -n "$PATH_MODELNET" ]]; then
    create_symlink "$PATH_MODELNET" "ModelNet40" || ((ERRORS++))
fi

if [[ $ERRORS -gt 0 ]]; then
    echo "Finished with $ERRORS error(s)."
    exit 1
fi

log "Dataset symlinks configured."
