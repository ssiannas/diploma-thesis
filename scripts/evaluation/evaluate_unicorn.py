"""Evaluate Unicorn decoded PLYs with curvature-stratified PSNR.

Reads Unicorn output PLYs and BPP from its CSV results, runs them through
our CurvatureQualityCalculator, and prints a curvature-stratified comparison.

Usage:
    python scripts/evaluation/evaluate_unicorn.py \
        --unicorn_output /path/to/Unicorn-v1/output/8ivfb_test \
        --unicorn_results /path/to/Unicorn-v1/results/8ivfb_test \
        --original_root datasets/8iVFB_unicorn_test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pcml.metrics.curvature import CurvatureQualityCalculator

# Maps our sequence names to Unicorn output filename stems
SEQUENCES = {
    "longdress": "longdress_vox10_1300",
    "loot": "loot_vox10_1200",
    "redandblack": "redandblack_vox10_1550",
    "soldier": "soldier_vox10_0690",
}


def load_ply(path: Path) -> np.ndarray:
    data = PlyData.read(str(path))["vertex"]
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


def find_original(original_root: Path, stem: str) -> Path:
    for ext in [".ply", ".npy"]:
        p = original_root / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"Original not found for {stem} in {original_root}")


def load_cloud(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    return load_ply(path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--unicorn_output",
        required=True,
        help="Unicorn output dir with decoded PLYs (e.g. .../output/8ivfb_test)",
    )
    p.add_argument(
        "--unicorn_results",
        required=True,
        help="Unicorn results dir with CSV files (e.g. .../results/8ivfb_test)",
    )
    p.add_argument(
        "--original_root",
        required=True,
        help="Dir with original PLY/NPY files",
    )
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--peak", type=float, default=1023.0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.unicorn_output)
    res_dir = Path(args.unicorn_results)
    orig_root = Path(args.original_root)

    all_rows = []

    for seq_short, stem in tqdm(SEQUENCES.items(), desc="Sequences"):
        # Load original
        orig_path = find_original(orig_root, stem)
        original = load_cloud(orig_path)

        # Load curvature (compute if not cached)
        curv_cache = out_dir / f"{stem}_curvature_k{args.curvature_k}.npy"
        if curv_cache.exists():
            curvature = np.load(curv_cache)
        else:
            from scipy.spatial import cKDTree

            print(f"  Computing curvature for {stem}...")
            tree = cKDTree(original)
            k = args.curvature_k
            _, idx = tree.query(original, k=k + 1)
            curvature = np.zeros(len(original), dtype=np.float32)
            for i in range(len(original)):
                nb = original[idx[i, 1:]] - original[idx[i, 1:]].mean(axis=0)
                ev = np.linalg.eigvalsh(nb.T @ nb)
                ev = np.sort(np.abs(ev))
                s = ev.sum()
                curvature[i] = ev[0] / s if s > 1e-10 else 0.0
            np.save(curv_cache, curvature)

        # Load BPP from CSV (fall back to file-size computation if missing)
        csv_candidates = sorted(res_dir.glob(f"*_{stem}.csv"))
        df = pd.read_csv(csv_candidates[0]) if csv_candidates else None
        n_pts = len(original)

        # Find decoded PLYs for this stem
        ply_files = sorted(out_dir.glob(f"{stem}_R*.ply"))
        # Filter to final decoded PLYs (not _part*, _ref*)
        ply_files = [
            p for p in ply_files if "_part" not in p.name and "_ref" not in p.name
        ]

        calc = CurvatureQualityCalculator
        for ply_path in ply_files:
            rate_idx = int(ply_path.stem.split("_R")[-1])
            if df is not None and rate_idx < len(df):
                bpp = df.iloc[rate_idx]["bpp"]
                psnr_p2p = df.iloc[rate_idx].get("mseF,PSNR (p2point)", float("nan"))
            else:
                # Fall back to bin file size
                bin_path = out_dir / ply_path.name.replace(".ply", "_part0_dec.bin")
                bpp = (
                    (bin_path.stat().st_size * 8 / n_pts)
                    if bin_path.exists()
                    else float("nan")
                )
                psnr_p2p = float("nan")

            try:
                decoded = load_ply(ply_path)
            except Exception as e:
                print(f"  WARNING: could not load {ply_path.name}: {e}")
                continue

            m = calc.compute_stratified_psnr(
                original=original,
                reconstructed=decoded,
                curvature=curvature,
                peak=args.peak,
                direction="reverse",
                k=args.curvature_k,
            )
            all_rows.append(
                {
                    "sequence": seq_short,
                    "rate_idx": rate_idx,
                    "bpp": bpp,
                    "psnr_p2p": psnr_p2p,
                    "edge_psnr": m.edge_psnr,
                    "flat_psnr": m.flat_psnr,
                    "degradation": m.degradation,
                }
            )

    if not all_rows:
        print("No results collected.")
        return

    # Print table
    print("\n" + "=" * 85)
    print(
        f"{'Sequence':<14} {'R':>2} {'BPP':>7} {'P2P PSNR':>10} "
        f"{'Edge':>8} {'Flat':>8} {'Degr':>8}"
    )
    print("-" * 85)
    for r in sorted(all_rows, key=lambda x: (x["sequence"], x["rate_idx"])):
        print(
            f"{r['sequence']:<14} {r['rate_idx']:>2} {r['bpp']:>7.4f} "
            f"{r['psnr_p2p']:>10.2f} "
            f"{r['edge_psnr']:>8.2f} {r['flat_psnr']:>8.2f} {r['degradation']:>8.2f}"
        )
    print("=" * 85)

    # Save CSV
    out_csv = out_dir.parent / "unicorn_stratified_results.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
