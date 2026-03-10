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
from dataset import COORD_SCALE, RATE_BPP, bpp_to_log, compute_curvature
from model import FiLMHeadSparseUNet, FiLMHeadSparseUNetV2, FiLMSparseUNet, SparseUNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
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
    is_film = model_type in ("film", "film_head", "film_head_v2")
    if is_film:
        model_cfg = ckpt_cfg.get("model", {}) if "config" in ckpt else {}
        film_embed_dim = model_cfg.get("film_embed_dim", 64)
        film_rate_repr = model_cfg.get("film_rate_repr", "scalar")
        if model_type == "film_head_v2":
            model = FiLMHeadSparseUNetV2(
                in_channels=in_ch,
                max_displacement=max_disp,
                film_embed_dim=film_embed_dim,
                rate_repr=film_rate_repr,
            ).to(device)
        elif model_type == "film_head":
            model = FiLMHeadSparseUNet(
                in_channels=in_ch,
                max_displacement=max_disp,
                film_embed_dim=film_embed_dim,
                rate_repr=film_rate_repr,
            ).to(device)
        else:
            model = FiLMSparseUNet(
                in_channels=in_ch,
                max_displacement=max_disp,
                film_embed_dim=film_embed_dim,
                rate_repr=film_rate_repr,
            ).to(device)
    else:
        model = SparseUNet(in_channels=in_ch, max_displacement=max_disp).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded {model_type} checkpoint from epoch {ckpt['epoch']}")

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

    # Forward pass on full cloud
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
