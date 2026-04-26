"""Step 0: Measure per-voxel displacement distribution before classification training.

Run on 10-20 training clouds per rate. Outputs:
  - Per-axis histograms (choose K)
  - p_zero_all (choose focal loss alpha)
  - Empirical collision rate from 1-NN matching

Usage:
    python scripts/analysis/displacement_histogram.py \
        --rates r2 r4 r6 --max_clouds 15
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DATA_ROOT = Path("datasets/pcgcv2_decoded")
SEQUENCES = ["longdress", "loot", "soldier"]  # exclude val (redandblack)


def greedy_collision_rate(decoded: np.ndarray, original: np.ndarray) -> float:
    """Fraction of decoded points that collide in 1-NN matching (before resolution)."""
    tree = cKDTree(original)
    _, idx = tree.query(decoded, k=1, workers=-1)
    # Count how many decoded points share the same nearest original
    counts = Counter(idx.tolist())
    collisions = sum(c - 1 for c in counts.values() if c > 1)
    return collisions / len(decoded)


def analyze_rate(rate: str, max_clouds: int):
    print(f"\n{'='*60}")
    print(f"Rate: {rate}")
    print(f"{'='*60}")

    all_deltas = []
    all_magnitudes = []
    collision_rates = []

    clouds_loaded = 0
    for seq in SEQUENCES:
        seq_dir = DATA_ROOT / seq
        if not seq_dir.exists():
            continue
        frame_dirs = sorted(d for d in seq_dir.iterdir() if d.is_dir())
        for frame_dir in frame_dirs:
            if clouds_loaded >= max_clouds:
                break
            orig_path = frame_dir / "original.npy"
            dec_path = frame_dir / f"pcgcv2_{rate}.npy"
            if not orig_path.exists() or not dec_path.exists():
                continue

            original = np.load(orig_path).astype(np.float32)
            decoded = np.load(dec_path).astype(np.float32)

            tree = cKDTree(original)
            _, idx = tree.query(decoded, k=1, workers=-1)
            nearest = original[idx]
            delta = np.round(nearest - decoded).astype(np.int32)
            all_deltas.append(delta)

            magnitudes = np.abs(delta).max(axis=1)
            all_magnitudes.append(magnitudes)

            cr = greedy_collision_rate(decoded, original)
            collision_rates.append(cr)
            clouds_loaded += 1

        if clouds_loaded >= max_clouds:
            break

    if not all_deltas:
        print("  No data found.")
        return

    all_deltas = np.concatenate(all_deltas, axis=0)
    all_magnitudes = np.concatenate(all_magnitudes, axis=0)
    N = len(all_deltas)

    # Per-axis histograms
    for axis, name in enumerate(["x", "y", "z"]):
        counts = Counter(all_deltas[:, axis].tolist())
        print(f"\n  {name}-axis displacement:")
        for k in sorted(counts.keys()):
            pct = counts[k] / N * 100
            bar = "#" * int(pct / 2)
            print(f"    delta={k:+3d}: {pct:5.1f}%  {bar}")

    # Zero-displacement stats
    zero_all = (all_deltas == 0).all(axis=1).mean() * 100
    zero_per_axis = (all_deltas == 0).mean(axis=0) * 100
    print(f"\n  Fully stationary (all axes zero): {zero_all:.1f}%")
    print(
        f"  Per-axis zero: x={zero_per_axis[0]:.1f}% y={zero_per_axis[1]:.1f}% z={zero_per_axis[2]:.1f}%"
    )

    # Chebyshev magnitude (max |delta| across axes)
    print(f"\n  Chebyshev magnitude |delta|_inf distribution:")
    for m in range(6):
        pct = (all_magnitudes == m).mean() * 100
        print(f"    |delta|={m}: {pct:.1f}%")
    pct_gt2 = (all_magnitudes > 2).mean() * 100
    pct_gt3 = (all_magnitudes > 3).mean() * 100
    print(f"    |delta|>2:  {pct_gt2:.1f}%  <- if >5%, use K=3 instead of K=2")
    print(f"    |delta|>3:  {pct_gt3:.1f}%")

    # Collision rate
    mean_cr = np.mean(collision_rates) * 100
    print(f"\n  Empirical collision rate (1-NN): {mean_cr:.1f}% of decoded points")

    # Recommendation
    print(f"\n  Recommendations:")
    if pct_gt2 > 5:
        print(f"    K=3 (21 output channels) — {pct_gt2:.1f}% of voxels need |delta|>2")
    else:
        print(
            f"    K=2 (15 output channels) — only {pct_gt2:.1f}% of voxels need |delta|>2"
        )

    alpha_zero = max(0.10, min(0.35, 1.0 - zero_all / 100.0))
    print(f"    focal alpha_zero ≈ {alpha_zero:.2f}  (p_zero={zero_all:.1f}%)")
    p_zero_bias = np.log(zero_all / (100.0 - zero_all + 1e-8))
    print(f"    cls_head bias for zero class ≈ {p_zero_bias:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rates", nargs="+", default=["r2", "r4", "r6"])
    p.add_argument("--max_clouds", type=int, default=15)
    p.add_argument("--data_root", default="datasets/pcgcv2_decoded")
    args = p.parse_args()

    global DATA_ROOT
    DATA_ROOT = Path(args.data_root)

    for rate in tqdm(args.rates, desc="Rates"):
        analyze_rate(rate, args.max_clouds)


if __name__ == "__main__":
    main()
