"""Apply our post-processing model to Unicorn decoded PLYs and evaluate.

Tests cross-codec generalization: model trained on PCGCv2 outputs applied to Unicorn outputs.

Usage:
    python scripts/evaluation/evaluate_unicorn_postprocessed.py \
        --unicorn_output /path/to/Unicorn-v1/output/8ivfb_test \
        --original_root datasets/8iVFB_unicorn_test \
        --checkpoint models/postprocessing/good_candidates/film_v10_head_ep24.pt \
        [--rate_indices 4 5 6 7]
"""

import argparse
import sys
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "postprocessing")
)

from pcml.metrics.curvature import CurvatureQualityCalculator
from postprocessing.dataset import COORD_SCALE, bpp_to_log, compute_curvature
from postprocessing.model import FiLMHeadSparseUNetV2

SEQUENCES = {
    "longdress": "longdress_vox10_1300",
    "loot": "loot_vox10_1200",
    "redandblack": "redandblack_vox10_1550",
    "soldier": "soldier_vox10_0690",
}


def load_ply(path: Path) -> np.ndarray:
    data = PlyData.read(str(path))["vertex"]
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg.get("model", {})
    no_curv = not cfg.get("data", {}).get("use_curvature", True)
    in_ch = 3 if no_curv else 4
    model = FiLMHeadSparseUNetV2(
        in_channels=in_ch,
        max_displacement=model_cfg.get("max_displacement", 5.0),
        film_embed_dim=model_cfg.get("film_embed_dim", 64),
        rate_repr=model_cfg.get("film_rate_repr", "bpp"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, no_curv, ckpt["epoch"]


@torch.no_grad()
def apply_model(
    model,
    decoded: np.ndarray,
    bpp: float,
    no_curv: bool,
    curvature_k: int,
    device: torch.device,
) -> np.ndarray:
    if no_curv:
        features = (decoded / COORD_SCALE).astype(np.float32)
    else:
        curv = compute_curvature(decoded, k=curvature_k)
        features = np.column_stack([decoded / COORD_SCALE, curv[:, np.newaxis]]).astype(
            np.float32
        )

    coords_int = np.floor(decoded).astype(np.int32)
    sin = ME.SparseTensor(
        features=torch.from_numpy(features).float(),
        coordinates=torch.from_numpy(
            np.column_stack([np.zeros(len(coords_int), dtype=np.int32), coords_int])
        ).int(),
        device=device,
    )
    rate_tensor = torch.tensor([bpp_to_log(bpp)], dtype=torch.float32, device=device)
    pred = model(sin, rate_tensor)
    displacement = pred.F.cpu().numpy()
    return decoded + displacement


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--unicorn_output", required=True)
    p.add_argument("--original_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--rate_indices",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7],
        help="Unicorn R indices to evaluate (default: 4 5 6 7)",
    )
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--peak", type=float, default=1023.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.unicorn_output)
    orig_root = Path(args.original_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, no_curv, epoch = load_model(Path(args.checkpoint), device)
    print(f"Loaded model from epoch {epoch}, no_curv={no_curv}")

    # Load stratified CSV for BPP values
    csv_path = out_dir.parent / "unicorn_stratified_results.csv"
    bpp_df = pd.read_csv(csv_path) if csv_path.exists() else None

    rows = []
    for seq_short, stem in tqdm(SEQUENCES.items(), desc="Sequences"):
        orig_path = orig_root / f"{stem}.ply"
        if not orig_path.exists():
            print(f"  SKIP: original not found for {stem}")
            continue
        original = load_ply(orig_path)

        # Curvature for evaluation
        curv_cache = out_dir / f"{stem}_curvature_k{args.curvature_k}.npy"
        if curv_cache.exists():
            curvature = np.load(curv_cache)
        else:
            print(f"  Computing curvature for {stem}...")
            curvature = compute_curvature(original, k=args.curvature_k)
            np.save(curv_cache, curvature)

        for rate_idx in args.rate_indices:
            ply_path = out_dir / f"{stem}_R{rate_idx}.ply"
            if not ply_path.exists():
                continue

            # Get BPP from saved CSV
            bpp = float("nan")
            if bpp_df is not None:
                row = bpp_df[
                    (bpp_df["sequence"] == seq_short) & (bpp_df["rate_idx"] == rate_idx)
                ]
                if not row.empty:
                    bpp = float(row.iloc[0]["bpp"])

            try:
                decoded = load_ply(ply_path)
            except Exception as e:
                print(f"  WARNING: {ply_path.name}: {e}")
                continue

            # Baseline (no postprocessing)
            m_base = CurvatureQualityCalculator.compute_stratified_psnr(
                original=original,
                reconstructed=decoded,
                curvature=curvature,
                peak=args.peak,
                direction="reverse",
                k=args.curvature_k,
            )

            # D1 PSNR baseline: symmetric max of fwd/rev mean squared NN distances
            orig_tree = cKDTree(original)
            dec_tree = cKDTree(decoded)
            fwd_d, _ = orig_tree.query(decoded, workers=-1)
            rev_d, _ = dec_tree.query(original, workers=-1)
            peak_sq = args.peak**2
            d1_base = 10 * np.log10(peak_sq / max(np.mean(fwd_d**2), np.mean(rev_d**2)))

            # Apply model
            if np.isnan(bpp):
                print(f"  WARNING: no BPP for {seq_short} R{rate_idx}, skipping model")
                continue
            refined = apply_model(
                model, decoded, bpp, no_curv, args.curvature_k, device
            )

            m_ref = CurvatureQualityCalculator.compute_stratified_psnr(
                original=original,
                reconstructed=refined,
                curvature=curvature,
                peak=args.peak,
                direction="reverse",
                k=args.curvature_k,
            )

            # D1 PSNR refined
            ref_tree = cKDTree(refined)
            fwd_r, _ = orig_tree.query(refined, workers=-1)
            rev_r, _ = ref_tree.query(original, workers=-1)
            d1_ref = 10 * np.log10(peak_sq / max(np.mean(fwd_r**2), np.mean(rev_r**2)))

            rows.append(
                {
                    "sequence": seq_short,
                    "rate_idx": rate_idx,
                    "bpp": bpp,
                    "log_bpp": bpp_to_log(bpp),
                    "base_d1": d1_base,
                    "ref_d1": d1_ref,
                    "base_edge": m_base.edge_psnr,
                    "base_flat": m_base.flat_psnr,
                    "base_degr": m_base.degradation,
                    "ref_edge": m_ref.edge_psnr,
                    "ref_flat": m_ref.flat_psnr,
                    "ref_degr": m_ref.degradation,
                    "delta_d1": d1_ref - d1_base,
                    "delta_edge": m_ref.edge_psnr - m_base.edge_psnr,
                    "delta_flat": m_ref.flat_psnr - m_base.flat_psnr,
                }
            )

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print(
        f"{'Seq':<14} {'R':>2} {'BPP':>7} {'log(bpp)':>9} "
        f"{'Base E':>8} {'Ref E':>8} {'dE':>7} "
        f"{'Base F':>8} {'Ref F':>8} {'dF':>7} "
        f"{'Base D':>7} {'Ref D':>7}"
    )
    print("-" * 100)
    for _, r in df.sort_values(["sequence", "rate_idx"]).iterrows():
        print(
            f"{r['sequence']:<14} {int(r['rate_idx']):>2} {r['bpp']:>7.4f} {r['log_bpp']:>9.3f} "
            f"{r['base_edge']:>8.2f} {r['ref_edge']:>8.2f} {r['delta_edge']:>+7.2f} "
            f"{r['base_flat']:>8.2f} {r['ref_flat']:>8.2f} {r['delta_flat']:>+7.2f} "
            f"{r['base_degr']:>7.2f} {r['ref_degr']:>7.2f}"
        )
    print("=" * 100)

    print("\nAverages per rate:")
    avg = df.groupby("rate_idx")[
        ["bpp", "delta_d1", "delta_edge", "delta_flat", "base_degr", "ref_degr"]
    ].mean()
    print(avg.round(3).to_string())

    out_csv = out_dir.parent / "unicorn_postprocessed_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
