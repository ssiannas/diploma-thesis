"""Compute per-rate GT displacement statistics from training data.

Uses the same subset as training: sequences [longdress, loot, soldier],
frame_stride=12, rates [r2, r4, r6].

For each decoded cloud, computes per-point displacement magnitude via
nearest-neighbor assignment (decoded -> original), then aggregates:
  - Var(||Δ||)        : variance of displacement magnitude
  - E[||Δ||]          : mean displacement magnitude
  - p_nonzero         : fraction of points with ||Δ|| > 0.5 voxels

Outputs variance weights and fraction-nonzero weights, both normalized
to w_r2 = 1.0.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

DATA_ROOT = Path("datasets/pcgcv2_decoded")
MANIFEST = DATA_ROOT / "manifest.json"
TRAIN_SEQUENCES = ["longdress", "loot", "soldier"]
RATES = ["r2", "r4", "r6"]
FRAME_STRIDE = 12
NONZERO_THRESHOLD = 0.5  # voxels; below this is treated as "zero displacement"


def load_manifest():
    with open(MANIFEST) as f:
        entries = json.load(f)
    # Filter to training sequences only
    entries = [e for e in entries if e["sequence"] in TRAIN_SEQUENCES]
    # Apply frame_stride: keep one frame every FRAME_STRIDE within each sequence
    by_seq = defaultdict(list)
    for e in entries:
        by_seq[e["sequence"]].append(e)
    filtered = []
    for seq, seq_entries in by_seq.items():
        frames = sorted(set(e["frame"] for e in seq_entries))
        keep = set(frames[::FRAME_STRIDE])
        filtered.extend(e for e in seq_entries if e["frame"] in keep)
    return filtered


def compute_stats():
    entries = load_manifest()

    # Pre-group by (sequence, frame) so we load original once per frame
    by_frame = defaultdict(list)
    for e in entries:
        by_frame[(e["sequence"], e["frame"])].append(e["rate"])

    # Per-rate accumulators: list of all displacement magnitudes
    mags = defaultdict(list)  # rate -> flat list of ||Δ|| values

    frames = sorted(by_frame.keys())
    for seq, frame in tqdm(frames, desc="Frames"):
        frame_dir = DATA_ROOT / seq / str(frame)
        original_path = frame_dir / "original.npy"
        if not original_path.exists():
            continue
        original = np.load(original_path).astype(np.float32)
        tree = cKDTree(original)

        for rate in by_frame[(seq, frame)]:
            if rate not in RATES:
                continue
            decoded_path = frame_dir / f"pcgcv2_{rate}.npy"
            if not decoded_path.exists():
                continue
            decoded = np.load(decoded_path).astype(np.float32)
            _, nn_idx = tree.query(decoded, workers=-1)
            disp = original[nn_idx] - decoded  # (N, 3)
            mag = np.linalg.norm(disp, axis=1)  # (N,)
            mags[rate].append(mag)

    print()
    print(
        f"{'Rate':<6} {'N_points':>12} {'E[||Δ||]':>10} {'Var(||Δ||)':>12} {'p_nonzero':>10}"
    )
    print("-" * 54)

    var_weights = {}
    pnz_weights = {}

    for rate in RATES:
        all_mags = np.concatenate(mags[rate])
        n = len(all_mags)
        mean_ = all_mags.mean()
        var_ = all_mags.var()
        pnz = (all_mags > NONZERO_THRESHOLD).mean()
        var_weights[rate] = var_
        pnz_weights[rate] = pnz
        print(f"{rate:<6} {n:>12,} {mean_:>10.4f} {var_:>12.6f} {pnz:>10.4f}")

    print()
    # Normalize to w_r2 = 1.0
    var_w = [var_weights[r] / var_weights["r2"] for r in RATES]
    pnz_w = [pnz_weights[r] / pnz_weights["r2"] for r in RATES]

    print("Variance weights  (w_i = Var_i / Var_r2):")
    print(f"  {dict(zip(RATES, [round(w, 4) for w in var_w]))}")
    print(f"  YAML: rate_weights: {[round(w, 3) for w in var_w]}")

    print()
    print(
        f"Fraction-nonzero weights  (w_i = p_nz_i / p_nz_r2, threshold={NONZERO_THRESHOLD}):"
    )
    print(f"  {dict(zip(RATES, [round(w, 4) for w in pnz_w]))}")
    print(f"  YAML: rate_weights: {[round(w, 3) for w in pnz_w]}")

    print()
    print("Current static weights for reference:")
    print("  YAML: rate_weights: [1.0, 0.3, 0.05]")


if __name__ == "__main__":
    compute_stats()
