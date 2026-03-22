"""Full-cloud inference: apply trained post-processing model to a decoded cloud.

Runs in pcgcv2 conda env.

Usage:
    python postprocessing/inference.py \
        --input decoded_clouds/redandblack_vox10_1550/pcgcv2_r7.npy \
        --checkpoint models/postprocessing/best_r7.pt \
        --output decoded_clouds/redandblack_vox10_1550/refined_r7.npy
"""

import argparse
import logging

import MinkowskiEngine as ME
import numpy as np
import torch
from dataset import _OFFSETS_26, COORD_SCALE, RATE_BPP, bpp_to_log, compute_curvature
from logging_utils import setup_logging
from model import build_model

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Post-processing inference")
    p.add_argument("--input", type=str, required=True, help="Decoded .npy path")
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    p.add_argument("--output", type=str, required=True, help="Output refined .npy path")
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--rate",
        type=str,
        default=None,
        help="Rate string (e.g. r2) for FiLM models. "
        "Auto-detected from input filename if omitted.",
    )
    return p.parse_args()


@torch.no_grad()
def main():
    setup_logging()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load decoded cloud
    decoded = np.load(args.input).astype(np.float32)
    logger.info(f"Loaded decoded cloud: {decoded.shape}")

    # Load checkpoint to determine model config
    ckpt = torch.load(args.checkpoint, map_location=device)

    # New config-based checkpoints have "config" dict; old ones have "args"
    if "config" in ckpt:
        ckpt_cfg = ckpt["config"]
        no_curv = not ckpt_cfg.get("data", {}).get("use_curvature", True)
        max_disp = ckpt_cfg.get("model", {}).get("max_displacement", 5.0)
        model_type = ckpt_cfg.get("model", {}).get("model_type", "unet")
    else:
        model_args = ckpt["args"]
        no_curv = model_args.get("no_curvature", False)
        max_disp = model_args.get("max_displacement", 5.0)
        model_type = model_args.get("model_type", "unet")

    # Build features
    if no_curv:
        logger.info("No-curvature model: using xyz-only features")
        features = (decoded / COORD_SCALE).astype(np.float32)
    else:
        logger.info("Computing curvature...")
        curvature = compute_curvature(decoded, k=args.curvature_k)
        features = np.column_stack(
            [
                decoded / COORD_SCALE,
                curvature[:, np.newaxis],
            ]
        ).astype(np.float32)

    coords_int = np.floor(decoded).astype(np.int32)

    in_ch = 3 if no_curv else 4
    is_occupancy = model_type == "occupancy"
    is_film = model_type in (
        "film",
        "film_head",
        "film_head_v2",
        "film_head_v3",
        "occupancy",
    )
    model_cfg = ckpt_cfg.get("model", {}) if "config" in ckpt else {}
    film_rate_repr = model_cfg.get("film_rate_repr", "bpp")
    model = build_model(
        model_type,
        in_channels=in_ch,
        max_displacement=max_disp,
        film_embed_dim=model_cfg.get("film_embed_dim", 64),
        rate_repr=film_rate_repr,
        quantize_output=model_cfg.get("quantize_output", False),
    ).to(device)
    # Prefer EMA weights when available (used for all validation during training)
    state_dict = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    src = "ema" if "ema_state_dict" in ckpt else "model"
    logger.info(
        f"Loaded {model_type} checkpoint from epoch {ckpt['epoch']} ({src} weights)"
    )

    # Resolve rate for FiLM models
    if is_film:
        rate_str = args.rate
        if rate_str is None:
            # Auto-detect from input filename (e.g. pcgcv2_r2.npy)
            import re

            m = re.search(r"_r(\d+)", args.input)
            if m:
                rate_str = f"r{m.group(1)}"
            else:
                raise ValueError(
                    "FiLM model requires --rate (could not auto-detect from filename)"
                )
        if film_rate_repr == "bpp":
            rate_scalar = bpp_to_log(RATE_BPP[rate_str])
        else:
            rate_scalar = float(rate_str[1:]) / 7.0
        rate_tensor = torch.tensor([rate_scalar], dtype=torch.float32, device=device)
        logger.info(f"FiLM rate: {rate_str} -> {rate_scalar:.4f} ({film_rate_repr})")

    if is_occupancy:
        # Occupancy inference: dilate + predict + threshold
        logger.info("Building occupancy candidates (dilating decoded cloud)...")
        occupied_set = {tuple(c) for c in coords_int.tolist()}
        all_neighbors = (coords_int[:, None, :] + _OFFSETS_26[None, :, :]).reshape(
            -1, 3
        )
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
        all_feats = np.column_stack(
            [all_coords.astype(np.float32) / COORD_SCALE, is_occ]
        )
        logger.info(
            f"Candidates: {len(all_coords)} ({n_dec} decoded + {len(all_coords)-n_dec} empty)"
        )

        sin = ME.SparseTensor(
            features=torch.from_numpy(all_feats).float(),
            coordinates=torch.from_numpy(
                np.column_stack([np.zeros(len(all_coords), dtype=np.int32), all_coords])
            ).int(),
            device=device,
        )
        logits = model(sin, rate_tensor)
        probs = torch.sigmoid(logits.F.squeeze(-1)).cpu().numpy()
        keep = probs > 0.5
        # Map back to float coords (integer voxel centers)
        kept_coords = logits.C[:, 1:].cpu().numpy()[keep].astype(np.float32)
        logger.info(
            f"Occupancy threshold=0.5: kept {keep.sum()} / {len(keep)} candidates "
            f"(pos_rate={keep.mean()*100:.1f}%)"
        )
        np.save(args.output, kept_coords)
        logger.info(f"Saved refined cloud: {kept_coords.shape} -> {args.output}")
        return

    # Displacement-based forward pass
    sin = ME.SparseTensor(
        features=torch.from_numpy(features).float(),
        coordinates=torch.from_numpy(
            np.column_stack([np.zeros(len(coords_int), dtype=np.int32), coords_int])
        ).int(),
        device=device,
    )

    if is_film:
        pred = model(sin, rate_tensor)
    else:
        pred = model(sin)
    displacement = pred.F.cpu().numpy()

    # Apply displacement
    refined = decoded + displacement
    logger.info(
        f"Displacement stats: mean={np.abs(displacement).mean():.3f}, "
        f"max={np.abs(displacement).max():.3f}"
    )

    np.save(args.output, refined.astype(np.float32))
    logger.info(f"Saved refined cloud: {refined.shape} -> {args.output}")


if __name__ == "__main__":
    main()
