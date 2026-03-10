"""Evaluate post-processing quality using curvature-stratified PSNR.

Single-pair mode:
    python scripts/evaluation/evaluate_postprocessing.py \
        --sequence redandblack_vox10_1550 --rate r2 --refined v6_r2

Batch mode (all sequences x rates, parallelized):
    python scripts/evaluation/evaluate_postprocessing.py \
        --rates r2 r4 r6 --refined_tag v6 --workers 8
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pcml.metrics.curvature import CurvatureQualityCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEQUENCES = [
    "longdress_vox10_1300",
    "loot_vox10_1200",
    "redandblack_vox10_1550",
    "soldier_vox10_0690",
]
DIRECTIONS = ["reverse", "forward_self"]
DEFAULT_WORKERS = 16


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate post-processing")
    p.add_argument(
        "--data_root",
        type=str,
        default="results/oversmoothing/decoded_clouds",
    )
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--peak", type=float, default=1023.0)

    # Single mode
    p.add_argument("--sequence", type=str, default=None)
    p.add_argument("--rate", type=str, default=None)
    p.add_argument(
        "--refined",
        type=str,
        default=None,
        help="Path or suffix for refined cloud (single mode)",
    )

    # Batch mode
    p.add_argument(
        "--rates",
        nargs="+",
        default=None,
        help="Rates to evaluate in batch mode (e.g. r2 r4 r6)",
    )
    p.add_argument(
        "--refined_tag",
        type=str,
        default=None,
        help="Tag for refined files in batch mode (loads refined_{tag}_{rate}.npy)",
    )
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return p.parse_args()


def evaluate_one(
    data_root: Path,
    seq: str,
    rate: str,
    refined_suffix: str,
    curvature_k: int,
    peak: float,
):
    """Evaluate one sequence/rate pair. Returns dict with all metrics."""
    seq_dir = data_root / seq
    original = np.load(seq_dir / "original.npy").astype(np.float32)
    decoded = np.load(seq_dir / f"pcgcv2_{rate}.npy").astype(np.float32)
    refined = np.load(seq_dir / f"refined_{refined_suffix}.npy").astype(np.float32)
    curvature = np.load(seq_dir / f"curvature_k{curvature_k}.npy")

    calc = CurvatureQualityCalculator
    result = {"sequence": seq, "rate": rate}

    for cloud, label in [(decoded, "dec"), (refined, "ref")]:
        for direction in DIRECTIONS:
            m = calc.compute_stratified_psnr(
                original=original,
                reconstructed=cloud,
                curvature=curvature,
                peak=peak,
                direction=direction,
                k=curvature_k,
            )
            result[f"{label}_{direction}_edge"] = m.edge_psnr
            result[f"{label}_{direction}_flat"] = m.flat_psnr
            result[f"{label}_{direction}_degradation"] = m.degradation

    return result


def print_batch_results(results):
    """Print formatted table of batch results."""
    results.sort(key=lambda r: (r["sequence"], r["rate"]))

    print("\n" + "=" * 90)
    print(
        f"{'Sequence':<28} {'Rate':>4}  "
        f"{'Edge dB':>8} {'Flat dB':>8} {'Degr dB':>8}  "
        f"{'Edge abs':>8} {'Flat abs':>8}"
    )
    print("-" * 90)

    rate_totals = {}
    for r in results:
        seq_short = r["sequence"].split("_vox")[0]
        rate = r["rate"]

        d_edge = r["ref_reverse_edge"] - r["dec_reverse_edge"]
        d_flat = r["ref_reverse_flat"] - r["dec_reverse_flat"]
        d_degr = r["ref_reverse_degradation"] - r["dec_reverse_degradation"]

        print(
            f"{seq_short:<28} {rate:>4}  "
            f"{d_edge:>+8.2f} {d_flat:>+8.2f} {d_degr:>+8.2f}  "
            f"{r['ref_reverse_edge']:>8.2f} {r['ref_reverse_flat']:>8.2f}"
        )

        if rate not in rate_totals:
            rate_totals[rate] = {"edge": [], "flat": [], "degr": []}
        rate_totals[rate]["edge"].append(d_edge)
        rate_totals[rate]["flat"].append(d_flat)
        rate_totals[rate]["degr"].append(d_degr)

    print("-" * 90)
    for rate in sorted(rate_totals):
        t = rate_totals[rate]
        avg_e = np.mean(t["edge"])
        avg_f = np.mean(t["flat"])
        avg_d = np.mean(t["degr"])
        print(
            f"{'AVERAGE':<28} {rate:>4}  "
            f"{avg_e:>+8.2f} {avg_f:>+8.2f} {avg_d:>+8.2f}"
        )
    print("=" * 90)


def run_single(args):
    """Original single sequence/rate evaluation."""
    data_root = Path(args.data_root)
    seq_dir = data_root / args.sequence

    original = np.load(seq_dir / "original.npy").astype(np.float32)
    decoded = np.load(seq_dir / f"pcgcv2_{args.rate}.npy").astype(np.float32)

    if args.refined is None:
        refined_path = seq_dir / f"refined_{args.rate}.npy"
    elif "/" in args.refined or args.refined.endswith(".npy"):
        refined_path = Path(args.refined)
    else:
        refined_path = seq_dir / f"refined_{args.refined}.npy"

    if not refined_path.exists():
        logger.error(f"Refined cloud not found: {refined_path}")
        sys.exit(1)

    refined = np.load(refined_path).astype(np.float32)
    curvature = np.load(seq_dir / f"curvature_k{args.curvature_k}.npy")

    logger.info(f"Sequence: {args.sequence} | Rate: {args.rate}")
    logger.info(
        f"Original: {original.shape[0]} | Decoded: {decoded.shape[0]} | "
        f"Refined: {refined.shape[0]}"
    )

    calc = CurvatureQualityCalculator
    for cloud, label in [
        (decoded, "Decoded (baseline)"),
        (refined, "Refined (post-proc)"),
    ]:
        print(f"\n  {label}:")
        for direction in DIRECTIONS:
            m = calc.compute_stratified_psnr(
                original=original,
                reconstructed=cloud,
                curvature=curvature,
                peak=args.peak,
                direction=direction,
                k=args.curvature_k,
            )
            print(
                f"    [{direction}] flat={m.flat_psnr:.2f} "
                f"edge={m.edge_psnr:.2f} degradation={m.degradation:.2f} dB"
            )

    # Delta
    dec_rev = calc.compute_stratified_psnr(
        original=original,
        reconstructed=decoded,
        curvature=curvature,
        peak=args.peak,
        direction="reverse",
        k=args.curvature_k,
    )
    ref_rev = calc.compute_stratified_psnr(
        original=original,
        reconstructed=refined,
        curvature=curvature,
        peak=args.peak,
        direction="reverse",
        k=args.curvature_k,
    )
    print(
        f"\n  Delta (reverse): edge={ref_rev.edge_psnr - dec_rev.edge_psnr:+.2f} "
        f"flat={ref_rev.flat_psnr - dec_rev.flat_psnr:+.2f} "
        f"degr={ref_rev.degradation - dec_rev.degradation:+.2f} dB"
    )


def run_batch(args):
    """Batch evaluation across all sequences and rates, parallelized."""
    data_root = Path(args.data_root)
    rates = args.rates
    tag = args.refined_tag

    jobs = [(seq, rate) for seq in SEQUENCES for rate in rates]
    n_workers = min(args.workers, len(jobs))

    logger.info(f"Batch evaluation: {len(jobs)} jobs with {n_workers} workers")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                evaluate_one,
                data_root,
                seq,
                rate,
                f"{tag}_{rate}",
                args.curvature_k,
                args.peak,
            ): (seq, rate)
            for seq, rate in jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Eval"):
            seq, rate = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Failed {seq}/{rate}: {e}")

    print_batch_results(results)


def main():
    args = parse_args()

    if args.rates and args.refined_tag:
        run_batch(args)
    elif args.sequence and args.rate:
        run_single(args)
    else:
        print("Usage:")
        print("  Single: --sequence SEQ --rate RATE [--refined SUFFIX]")
        print("  Batch:  --rates r2 r4 r6 --refined_tag v6 [--workers N]")
        sys.exit(1)


if __name__ == "__main__":
    main()
