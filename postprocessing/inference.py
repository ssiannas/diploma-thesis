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
from dataset import COORD_SCALE, compute_curvature
from model import GatedSparseUNet, SparseUNet, ThresholdGatedUNet

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
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load decoded cloud
    decoded = np.load(args.input).astype(np.float32)
    logger.info(f"Loaded decoded cloud: {decoded.shape}")

    # Compute curvature (forward_self signal)
    logger.info("Computing curvature...")
    curvature = compute_curvature(decoded, k=args.curvature_k)

    # Build features
    features = np.column_stack(
        [
            decoded / COORD_SCALE,
            curvature[:, np.newaxis],
        ]
    ).astype(np.float32)

    coords_int = np.floor(decoded).astype(np.int32)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt["args"]
    max_disp = model_args.get("max_displacement", 5.0)
    model_type = model_args.get("model_type", "unet")
    is_gated = model_type in ("gated", "threshold")

    if model_type == "gated":
        model = GatedSparseUNet(in_channels=4, max_displacement=max_disp).to(device)
    elif model_type == "threshold":
        model = ThresholdGatedUNet(in_channels=4, max_displacement=max_disp).to(device)
    else:
        model = SparseUNet(in_channels=4, max_displacement=max_disp).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model_name = model_type
    logger.info(f"Loaded {model_name} checkpoint from epoch {ckpt['epoch']}")

    # Forward pass on full cloud
    sin = ME.SparseTensor(
        features=torch.from_numpy(features).float(),
        coordinates=torch.from_numpy(
            np.column_stack([np.zeros(len(coords_int), dtype=np.int32), coords_int])
        ).int(),
        device=device,
    )

    output = model(sin)
    if is_gated:
        pred, gate = output
        gate_vals = gate.F.squeeze(-1).cpu().numpy()
        displacement = pred.F.cpu().numpy() * gate_vals[:, np.newaxis]
        logger.info(
            f"Gate stats: mean={gate_vals.mean():.3f}, "
            f">0.5={100*(gate_vals > 0.5).mean():.1f}%"
        )
    else:
        pred = output
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
