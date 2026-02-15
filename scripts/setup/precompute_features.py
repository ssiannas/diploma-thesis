"""Precompute displacement and curvature features for MultiFrameDataset.

Reads manifest.json and for each (sequence, frame, rate) entry, computes:
  - displacement_{rate}.npy: NN displacement vectors (N, 3) float32
  - curvature_{rate}.npy: per-point PCA curvature (N,) float32

Uses multiprocessing for parallelism. Skips existing files (resume support).

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2
    python scripts/setup/precompute_features.py --data_root datasets/pcgcv2_decoded
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "postprocessing")
)
from dataset import compute_curvature


def process_entry(args_tuple):
    """Process a single (sequence, frame, rate) entry."""
    data_root, entry, curvature_k = args_tuple
    seq, frame, rate = entry["sequence"], entry["frame"], entry["rate"]
    frame_dir = Path(data_root) / seq / str(frame)

    disp_path = frame_dir / f"displacement_{rate}.npy"
    curv_path = frame_dir / f"curvature_{rate}.npy"

    # Skip if already computed
    if disp_path.exists() and curv_path.exists():
        return f"skip {seq}/{frame}/{rate}"

    decoded_path = frame_dir / f"pcgcv2_{rate}.npy"
    original_path = frame_dir / "original.npy"

    if not decoded_path.exists() or not original_path.exists():
        return f"missing {seq}/{frame}/{rate}"

    decoded = np.load(decoded_path).astype(np.float32)
    original = np.load(original_path).astype(np.float32)

    # NN displacement
    tree = cKDTree(original)
    _, nn_idx = tree.query(decoded)
    displacement = (original[nn_idx] - decoded).astype(np.float32)
    np.save(disp_path, displacement)

    # Curvature
    curvature = compute_curvature(decoded, k=curvature_k)
    np.save(curv_path, curvature)

    return f"done {seq}/{frame}/{rate} ({len(decoded)} pts)"


def main():
    parser = argparse.ArgumentParser(description="Precompute displacement & curvature")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--curvature_k", type=int, default=30)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    manifest_path = Path(args.data_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(
        f"Precomputing features for {len(manifest)} entries with {args.workers} workers"
    )

    work_items = [(args.data_root, entry, args.curvature_k) for entry in manifest]

    with Pool(args.workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_entry, work_items),
                total=len(work_items),
                desc="Precomputing",
            )
        )

    done = sum(1 for r in results if r.startswith("done"))
    skipped = sum(1 for r in results if r.startswith("skip"))
    missing = sum(1 for r in results if r.startswith("missing"))
    print(f"Done: {done}, Skipped (existing): {skipped}, Missing: {missing}")


if __name__ == "__main__":
    main()
