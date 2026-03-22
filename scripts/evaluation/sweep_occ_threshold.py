"""Sweep occupancy threshold on val sequence, then eval best on all sequences.

Usage:
    python scripts/evaluation/sweep_occ_threshold.py \
        --checkpoint models/postprocessing/occ_v1_r2/best_r2.pt \
        --rate r2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MinkowskiEngine as ME

from pcml.metrics.curvature import CurvatureQualityCalculator
from postprocessing.dataset import _OFFSETS_26, COORD_SCALE, RATE_BPP, bpp_to_log
from postprocessing.model import OccupancySparseUNet

DATA_ROOT = Path("results/oversmoothing/decoded_clouds")
SEQUENCES = [
    "longdress_vox10_1300",
    "loot_vox10_1200",
    "redandblack_vox10_1550",
    "soldier_vox10_0690",
]
VAL_SEQ = "redandblack_vox10_1550"


def build_candidates(decoded_npy: np.ndarray, radius: int = 1):
    """Dilate decoded cloud by `radius` hops and return (all_coords, all_feats, n_dec)."""
    coords_int = np.floor(decoded_npy).astype(np.int32)
    if radius == 1:
        offsets = _OFFSETS_26
    else:
        r = radius
        grid = np.array(
            [
                [dx, dy, dz]
                for dx in range(-r, r + 1)
                for dy in range(-r, r + 1)
                for dz in range(-r, r + 1)
                if dx != 0 or dy != 0 or dz != 0
            ],
            dtype=np.int32,
        )
        offsets = grid

    all_neighbors = (coords_int[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    occupied_set = {tuple(c) for c in coords_int.tolist()}
    empty = np.array(
        [c for c in all_neighbors.tolist() if tuple(c) not in occupied_set],
        dtype=np.int32,
    )
    if len(empty) > 0:
        empty = np.unique(empty, axis=0)
        all_coords = np.vstack([coords_int, empty])
    else:
        all_coords = coords_int

    n_dec = len(coords_int)
    is_occ = np.zeros(len(all_coords), dtype=np.float32)
    is_occ[:n_dec] = 1.0
    feats = np.column_stack([all_coords.astype(np.float32) / COORD_SCALE, is_occ])
    return all_coords, feats, n_dec


@torch.no_grad()
def get_probs(model, all_coords, all_feats, rate_tensor, device):
    sin = ME.SparseTensor(
        features=torch.from_numpy(all_feats).float(),
        coordinates=torch.from_numpy(
            np.column_stack([np.zeros(len(all_coords), dtype=np.int32), all_coords])
        ).int(),
        device=device,
    )
    logits = model(sin, rate_tensor)
    return (
        torch.sigmoid(logits.F.squeeze(-1)).cpu().numpy(),
        logits.C[:, 1:].cpu().numpy(),
    )


def eval_psnr(refined_coords, seq_dir: Path, rate: str = "r2", curvature_k: int = 30):
    decoded = np.load(seq_dir / f"pcgcv2_{rate}.npy").astype(np.float32)
    original = np.load(seq_dir / "original.npy").astype(np.float32)
    curvature = np.load(seq_dir / f"curvature_k{curvature_k}.npy")
    calc = CurvatureQualityCalculator()
    ref_m = calc.compute_stratified_psnr(
        original,
        refined_coords.astype(np.float32),
        curvature=curvature,
        k=curvature_k,
        direction="reverse",
    )
    dec_m = calc.compute_stratified_psnr(
        original, decoded, curvature=curvature, k=curvature_k, direction="reverse"
    )
    return {
        "edge_delta": ref_m.edge_psnr - dec_m.edge_psnr,
        "flat_delta": ref_m.flat_psnr - dec_m.flat_psnr,
        "edge_ref": ref_m.edge_psnr,
        "flat_ref": ref_m.flat_psnr,
        "n_points": len(refined_coords),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rate", default="r2")
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("config", {})
    film_embed_dim = ckpt_cfg.get("model", {}).get("film_embed_dim", 64)
    film_rate_repr = ckpt_cfg.get("model", {}).get("film_rate_repr", "bpp")

    model = OccupancySparseUNet(
        in_channels=4, film_embed_dim=film_embed_dim, rate_repr=film_rate_repr
    ).to(device)
    model.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
    model.eval()

    rate_scalar = bpp_to_log(RATE_BPP[args.rate])
    rate_tensor = torch.tensor([rate_scalar], dtype=torch.float32, device=device)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    # --- Phase 1: sweep on val sequence ---
    print(f"\n=== Threshold sweep on {VAL_SEQ} (radius={args.radius}) ===")
    seq_dir = DATA_ROOT / VAL_SEQ
    decoded = np.load(seq_dir / f"pcgcv2_{args.rate}.npy").astype(np.float32)
    all_coords, all_feats, n_dec = build_candidates(decoded, radius=args.radius)
    print(
        f"Candidates: {len(all_coords)} ({n_dec} decoded + {len(all_coords)-n_dec} empty)"
    )

    probs, coord_arr = get_probs(model, all_coords, all_feats, rate_tensor, device)

    # --- Phase 1a: threshold sweep ---
    print(f"\n{'Thresh':>8}  {'Kept':>8}  {'Edge dB':>8}  {'Flat dB':>8}")
    print("-" * 42)
    best_thresh = 0.5
    best_edge = -999.0
    for thresh in thresholds:
        keep = probs > thresh
        kept_coords = coord_arr[keep].astype(np.float32)
        stats = eval_psnr(kept_coords, seq_dir, rate=args.rate)
        marker = " <--" if stats["edge_delta"] > best_edge else ""
        if stats["edge_delta"] > best_edge:
            best_edge = stats["edge_delta"]
            best_thresh = thresh
        print(
            f"{thresh:>8.2f}  {keep.sum():>8}  {stats['edge_delta']:>+8.2f}  {stats['flat_delta']:>+8.2f}{marker}"
        )

    print(
        f"\nBest threshold by edge PSNR: {best_thresh:.2f} (edge delta={best_edge:+.2f} dB)"
    )

    # --- Phase 1b: top-K sweep (keep exactly n_dec points) ---
    print(f"\n=== Top-K sweep (K=decoded_count={n_dec}) on {VAL_SEQ} ===")
    topk_idx = np.argsort(probs)[::-1][:n_dec]
    topk_coords = coord_arr[topk_idx].astype(np.float32)
    stats_topk = eval_psnr(topk_coords, seq_dir, rate=args.rate)
    # also show a range around n_dec
    print(f"\n{'Scale':>8}  {'Kept':>8}  {'Edge dB':>8}  {'Flat dB':>8}")
    print("-" * 42)
    best_k_scale = 1.0
    best_k_edge = -999.0
    for scale in [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]:
        k = int(n_dec * scale)
        idx = np.argsort(probs)[::-1][:k]
        kept_coords = coord_arr[idx].astype(np.float32)
        stats = eval_psnr(kept_coords, seq_dir, rate=args.rate)
        marker = " <--" if stats["edge_delta"] > best_k_edge else ""
        if stats["edge_delta"] > best_k_edge:
            best_k_edge = stats["edge_delta"]
            best_k_scale = scale
        print(
            f"{scale:>8.2f}  {k:>8}  {stats['edge_delta']:>+8.2f}  {stats['flat_delta']:>+8.2f}{marker}"
        )

    print(
        f"\nBest top-K scale: {best_k_scale:.2f}x decoded count (edge delta={best_k_edge:+.2f} dB)"
    )

    # --- Phase 2: eval best threshold on all sequences ---
    print(f"\n=== Full eval at threshold={best_thresh:.2f}, radius={args.radius} ===")
    print(
        f"\n{'Sequence':<30}  {'Rate':>5}  {'Edge dB':>8}  {'Flat dB':>8}  {'Kept':>8}"
    )
    print("-" * 65)
    edge_deltas, flat_deltas = [], []
    for seq in SEQUENCES:
        seq_dir = DATA_ROOT / seq
        decoded = np.load(seq_dir / f"pcgcv2_{args.rate}.npy").astype(np.float32)
        all_coords, all_feats, n_dec = build_candidates(decoded, radius=args.radius)
        probs, coord_arr = get_probs(model, all_coords, all_feats, rate_tensor, device)
        keep = probs > best_thresh
        kept_coords = coord_arr[keep].astype(np.float32)
        stats = eval_psnr(kept_coords, seq_dir, rate=args.rate)
        seq_short = seq.split("_vox")[0]
        print(
            f"{seq_short:<30}  {args.rate:>5}  {stats['edge_delta']:>+8.2f}"
            f"  {stats['flat_delta']:>+8.2f}  {keep.sum():>8}"
        )
        edge_deltas.append(stats["edge_delta"])
        flat_deltas.append(stats["flat_delta"])

    print("-" * 65)
    print(
        f"{'AVERAGE':<30}  {args.rate:>5}  {np.mean(edge_deltas):>+8.2f}  {np.mean(flat_deltas):>+8.2f}"
    )

    # --- Phase 3: top-K full eval at best scale ---
    print(
        f"\n=== Full eval top-K (scale={best_k_scale:.2f}x decoded), radius={args.radius} ==="
    )
    print(
        f"\n{'Sequence':<30}  {'Rate':>5}  {'Edge dB':>8}  {'Flat dB':>8}  {'Kept':>8}"
    )
    print("-" * 65)
    edge_deltas_k, flat_deltas_k = [], []
    for seq in SEQUENCES:
        seq_dir = DATA_ROOT / seq
        decoded = np.load(seq_dir / f"pcgcv2_{args.rate}.npy").astype(np.float32)
        all_coords, all_feats, n_dec = build_candidates(decoded, radius=args.radius)
        probs, coord_arr = get_probs(model, all_coords, all_feats, rate_tensor, device)
        k = int(n_dec * best_k_scale)
        idx = np.argsort(probs)[::-1][:k]
        kept_coords = coord_arr[idx].astype(np.float32)
        stats = eval_psnr(kept_coords, seq_dir, rate=args.rate)
        seq_short = seq.split("_vox")[0]
        print(
            f"{seq_short:<30}  {args.rate:>5}  {stats['edge_delta']:>+8.2f}  {stats['flat_delta']:>+8.2f}  {k:>8}"
        )
        edge_deltas_k.append(stats["edge_delta"])
        flat_deltas_k.append(stats["flat_delta"])

    print("-" * 65)
    print(
        f"{'AVERAGE':<30}  {args.rate:>5}  {np.mean(edge_deltas_k):>+8.2f}  {np.mean(flat_deltas_k):>+8.2f}"
    )


if __name__ == "__main__":
    main()
