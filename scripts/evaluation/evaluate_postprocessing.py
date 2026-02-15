"""Evaluate post-processing quality using curvature-stratified PSNR.

Runs in the MAIN Python env (not pcgcv2), uses pcml metrics.

Usage:
    python scripts/evaluate_postprocessing.py \
        --data_root results/oversmoothing/decoded_clouds \
        --sequence redandblack_vox10_1550 \
        --rate r7
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcml.metrics.curvature import CurvatureQualityCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DIRECTIONS = ["reverse", "forward_self"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate post-processing")
    p.add_argument(
        "--data_root",
        type=str,
        default="results/oversmoothing/decoded_clouds",
    )
    p.add_argument("--sequence", type=str, default="redandblack_vox10_1550")
    p.add_argument("--rate", type=str, default="r7")
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--peak", type=float, default=1023.0)
    return p.parse_args()


def compute_all_directions(original, cloud, curvature, k, peak):
    """Compute stratified PSNR for reverse and forward_self directions."""
    calc = CurvatureQualityCalculator
    results = {}
    for direction in DIRECTIONS:
        results[direction] = calc.compute_stratified_psnr(
            original=original,
            reconstructed=cloud,
            curvature=curvature,
            peak=peak,
            direction=direction,
            k=k,
        )
    return results


def main():
    args = parse_args()
    seq_dir = Path(args.data_root) / args.sequence

    # Load clouds
    original = np.load(seq_dir / "original.npy").astype(np.float32)
    decoded = np.load(seq_dir / f"pcgcv2_{args.rate}.npy").astype(np.float32)
    refined_path = seq_dir / f"refined_{args.rate}.npy"

    if not refined_path.exists():
        logger.error(f"Refined cloud not found: {refined_path}")
        logger.error("Run inference.py first to generate the refined cloud.")
        sys.exit(1)

    refined = np.load(refined_path).astype(np.float32)

    # Load precomputed curvature on original
    curvature = np.load(seq_dir / f"curvature_k{args.curvature_k}.npy")

    logger.info(f"Sequence: {args.sequence}")
    logger.info(f"Rate: {args.rate}")
    logger.info(f"Original: {original.shape[0]} pts")
    logger.info(f"Decoded:  {decoded.shape[0]} pts")
    logger.info(f"Refined:  {refined.shape[0]} pts")

    # Compute metrics
    logger.info("Computing metrics for decoded cloud (baseline)...")
    dec_metrics = compute_all_directions(
        original, decoded, curvature, args.curvature_k, args.peak
    )

    logger.info("Computing metrics for refined cloud...")
    ref_metrics = compute_all_directions(
        original, refined, curvature, args.curvature_k, args.peak
    )

    # Report
    print("\n" + "=" * 60)
    print(f"Post-Processing Evaluation: {args.sequence} @ {args.rate}")
    print("=" * 60)

    for label, metrics in [
        ("Decoded (baseline)", dec_metrics),
        ("Refined (post-proc)", ref_metrics),
    ]:
        print(f"\n  {label}:")
        for direction, m in metrics.items():
            print(
                f"    [{direction}] flat={m.flat_psnr:.2f} "
                f"edge={m.edge_psnr:.2f} degradation={m.degradation:.2f} dB"
            )

    # Compute deltas (reverse direction is the reliable metric)
    dec_rev = dec_metrics["reverse"]
    ref_rev = ref_metrics["reverse"]

    print(f"\n  Delta (refined - decoded), reverse direction:")
    print(f"    edge_psnr:    {ref_rev.edge_psnr - dec_rev.edge_psnr:+.2f} dB")
    print(f"    flat_psnr:    {ref_rev.flat_psnr - dec_rev.flat_psnr:+.2f} dB")
    print(f"    degradation:  {ref_rev.degradation - dec_rev.degradation:+.2f} dB")
    print(f"    (negative degradation delta = improvement)")
    print("=" * 60)


if __name__ == "__main__":
    main()
