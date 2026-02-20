"""Precompute K-NN displacement targets for min-of-K loss training.

For each (sequence, frame, rate) entry, computes K nearest neighbors from
decoded to original and stores displacement_k{K}_{rate}.npy as (N, K, 3) float32.

Skips entries where the output file already exists (resume support).

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate pcgcv2
    python scripts/setup/precompute_knn_displacements.py \
        --data_root datasets/pcgcv2_decoded --k 5
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def process_entry(args_tuple):
    """Process a single (sequence, frame, rate) entry."""
    data_root, entry, k = args_tuple
    seq, frame, rate = entry["sequence"], entry["frame"], entry["rate"]
    frame_dir = Path(data_root) / seq / str(frame)

    out_path = frame_dir / f"displacement_k{k}_{rate}.npy"
    if out_path.exists():
        return "skip"

    decoded_path = frame_dir / f"pcgcv2_{rate}.npy"
    original_path = frame_dir / "original.npy"

    if not decoded_path.exists() or not original_path.exists():
        return "missing"

    decoded = np.load(decoded_path).astype(np.float32)
    original = np.load(original_path).astype(np.float32)

    tree = cKDTree(original)
    _, nn_idx = tree.query(decoded, k=k)  # (N, K)
    displacement = (original[nn_idx] - decoded[:, np.newaxis, :]).astype(
        np.float32
    )  # (N, K, 3)
    np.save(out_path, displacement)

    return f"done ({len(decoded)} pts)"


def main():
    parser = argparse.ArgumentParser(description="Precompute K-NN displacements")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument(
        "--rates", nargs="+", default=None, help="Filter to specific rates"
    )
    args = parser.parse_args()

    manifest_path = Path(args.data_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.rates:
        manifest = [e for e in manifest if e["rate"] in args.rates]

    print(
        f"Precomputing K={args.k} NN displacements for {len(manifest)} entries "
        f"with {args.workers} workers"
    )

    work_items = [(args.data_root, entry, args.k) for entry in manifest]

    with Pool(args.workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_entry, work_items),
                total=len(work_items),
                desc="Precomputing K-NN",
            )
        )

    done = sum(1 for r in results if r.startswith("done"))
    skipped = sum(1 for r in results if r.startswith("skip"))
    missing = sum(1 for r in results if r.startswith("missing"))
    print(f"Done: {done}, Skipped: {skipped}, Missing: {missing}")


if __name__ == "__main__":
    main()
