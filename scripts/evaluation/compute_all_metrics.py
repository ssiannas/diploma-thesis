"""Compute D1 PSNR + curvature-stratified PSNR for all saved refined clouds.

Reads pre-saved .npy files (original, decoded, refined) — no model inference.
Exports a CSV with per-frame results for every sequence / rate / tag.

Usage (single run):
    python scripts/evaluation/compute_all_metrics.py \
        --tag nocurv_ep24 --rates r2 r4 r6 \
        --output results/metrics/nocurv_ep24_all_metrics.csv

Usage (parallel — one process per sequence, then merge):
    for seq in longdress loot redandblack soldier; do
        python scripts/evaluation/compute_all_metrics.py \\
            --tag nocurv_ep24 --rates r2 r4 r6 --sequences $seq \\
            --output results/metrics/nocurv_ep24_${seq}.csv &
    done
    wait
    python scripts/evaluation/compute_all_metrics.py --merge \\
        --inputs results/metrics/nocurv_ep24_*.csv \\
        --output results/metrics/nocurv_ep24_all_metrics.csv

Optimization: cKDTree(original) built once per frame; per rate, cKDTree(decoded)
and cKDTree(refined) built once each with backward distances reused for both
D1 PSNR and curvature-stratified PSNR (same computation, reverse direction).
"""

import argparse
import collections
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pcml.metrics.curvature import CurvatureQualityCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("datasets/pcgcv2_decoded")
SEQUENCES = ["longdress", "loot", "redandblack", "soldier"]
PEAK = 1023.0

FIELDNAMES = [
    "sequence",
    "frame",
    "rate",
    "n_original",
    "n_decoded",
    "n_refined",
    "dec_d1",
    "ref_d1",
    "d1_delta",
    "dec_edge",
    "ref_edge",
    "edge_delta",
    "dec_flat",
    "ref_flat",
    "flat_delta",
]


def _psnr(mse: float, peak: float = PEAK) -> float:
    return float("inf") if mse == 0 else 10.0 * np.log10(peak**2 / mse)


def compute_frame(original, curvature, decoded, refined, peak=PEAK, k=30):
    """Compute all metrics for one (original, decoded, refined) triple.

    Builds cKDTree(decoded) and cKDTree(refined) once each. The backward query
    (cloud -> original) is shared between D1 PSNR and stratified reverse PSNR.
    orig_tree is passed in (built once per frame, reused across rates).
    Uses workers=-1 for multi-core kNN queries.
    """
    orig_tree = _ORIG_TREE  # module-level, set before rate loop

    # --- decoded ---
    dec_tree = cKDTree(decoded)
    fwd_dec_sq = orig_tree.query(decoded, workers=-1)[0] ** 2  # decoded → original
    bwd_dec_sq = (
        dec_tree.query(original, workers=-1)[0] ** 2
    )  # original → decoded (shared!)

    dec_d1 = _psnr(max(float(np.mean(fwd_dec_sq)), float(np.mean(bwd_dec_sq))), peak)
    dec_m = CurvatureQualityCalculator.compute_stratified_psnr(
        original,
        decoded,
        curvature=curvature,
        k=k,
        direction="reverse",
        rev_sq_dists=bwd_dec_sq,  # skip tree rebuild + query
        rec_curvature=curvature,  # skip KL recompute (KL unused)
    )

    # --- refined ---
    ref_tree = cKDTree(refined)
    fwd_ref_sq = orig_tree.query(refined, workers=-1)[0] ** 2
    bwd_ref_sq = ref_tree.query(original, workers=-1)[0] ** 2

    ref_d1 = _psnr(max(float(np.mean(fwd_ref_sq)), float(np.mean(bwd_ref_sq))), peak)
    ref_m = CurvatureQualityCalculator.compute_stratified_psnr(
        original,
        refined,
        curvature=curvature,
        k=k,
        direction="reverse",
        rev_sq_dists=bwd_ref_sq,
        rec_curvature=curvature,
    )

    return dec_d1, ref_d1, dec_m, ref_m


# Module-level orig_tree — set per frame to avoid rebuilding inside rate loop
_ORIG_TREE = None


def print_aggregates(rows, sequences, rates):
    print(f"\n{'='*80}")
    print(
        f"{'Sequence':<15} {'Rate':>5} {'N':>5}  {'D1 Δ':>8}  {'Edge Δ':>8}  {'Flat Δ':>8}"
    )
    print(f"{'-'*80}")
    rate_agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        rate_agg[r["rate"]][r["sequence"]].append(r)
    for rate in rates:
        overall_d1, overall_edge, overall_flat = [], [], []
        for seq in sequences:
            seq_rows = rate_agg[rate].get(seq, [])
            if not seq_rows:
                continue
            d1 = np.mean([float(x["d1_delta"]) for x in seq_rows])
            edge = np.mean([float(x["edge_delta"]) for x in seq_rows])
            flat = np.mean([float(x["flat_delta"]) for x in seq_rows])
            print(
                f"{seq:<15} {rate:>5} {len(seq_rows):>5}  {d1:>+8.3f}  {edge:>+8.3f}  {flat:>+8.3f}"
            )
            overall_d1.append(d1)
            overall_edge.append(edge)
            overall_flat.append(flat)
        print(
            f"{'AVERAGE':<15} {rate:>5} {'':>5}  "
            f"{np.mean(overall_d1):>+8.3f}  "
            f"{np.mean(overall_edge):>+8.3f}  "
            f"{np.mean(overall_flat):>+8.3f}"
        )
        print()
    print(f"{'='*80}")


def run_eval(args):
    global _ORIG_TREE

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for seq in args.sequences:
        seq_dir = DATA_ROOT / seq
        if not seq_dir.exists():
            logger.warning("Missing sequence dir: %s", seq_dir)
            continue
        for frame_dir in sorted(seq_dir.iterdir()):
            if frame_dir.is_dir():
                all_frames.append((seq, frame_dir))

    logger.info(
        "Found %d frames | rates: %s | tag: %s", len(all_frames), args.rates, args.tag
    )

    rows = []
    skipped = 0

    for seq, frame_dir in tqdm(all_frames, desc="Frames"):
        original_path = frame_dir / "original.npy"
        curv_path = frame_dir / f"curvature_k{args.curvature_k}.npy"

        if not original_path.exists():
            skipped += 1
            continue

        original = np.load(original_path).astype(np.float32)

        if curv_path.exists():
            curvature = np.load(curv_path)
        else:
            curvature = CurvatureQualityCalculator.compute_curvature(
                original, k=args.curvature_k
            )
            np.save(curv_path, curvature)

        # Build orig_tree ONCE per frame — reused across all rates
        _ORIG_TREE = cKDTree(original)

        for rate in args.rates:
            decoded_path = frame_dir / f"pcgcv2_{rate}.npy"
            refined_path = frame_dir / f"refined_{args.tag}_{rate}.npy"

            if not decoded_path.exists() or not refined_path.exists():
                skipped += 1
                continue

            decoded = np.load(decoded_path).astype(np.float32)
            refined = np.load(refined_path).astype(np.float32)

            dec_d1, ref_d1, dec_m, ref_m = compute_frame(
                original, curvature, decoded, refined, k=args.curvature_k
            )

            rows.append(
                {
                    "sequence": seq,
                    "frame": frame_dir.name,
                    "rate": rate,
                    "n_original": len(original),
                    "n_decoded": len(decoded),
                    "n_refined": len(refined),
                    "dec_d1": round(dec_d1, 4),
                    "ref_d1": round(ref_d1, 4),
                    "d1_delta": round(ref_d1 - dec_d1, 4),
                    "dec_edge": round(dec_m.edge_psnr, 4),
                    "ref_edge": round(ref_m.edge_psnr, 4),
                    "edge_delta": round(ref_m.edge_psnr - dec_m.edge_psnr, 4),
                    "dec_flat": round(dec_m.flat_psnr, 4),
                    "ref_flat": round(ref_m.flat_psnr, 4),
                    "flat_delta": round(ref_m.flat_psnr - dec_m.flat_psnr, 4),
                }
            )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s (%d skipped)", len(rows), out_path, skipped)
    print_aggregates(rows, args.sequences, args.rates)


def run_merge(args):
    all_rows = []
    for csv_path in sorted(args.inputs):
        with open(csv_path, newline="") as f:
            all_rows.extend(list(csv.DictReader(f)))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(
        "Merged %d rows from %d files → %s", len(all_rows), len(args.inputs), out_path
    )

    # Aggregate view after merge
    rates = sorted({r["rate"] for r in all_rows})
    sequences = sorted({r["sequence"] for r in all_rows})
    print_aggregates(all_rows, sequences, rates)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # Default (no subcommand) = eval
    p.add_argument("--tag", default=None, help="Refined cloud tag, e.g. nocurv_ep24")
    p.add_argument("--rates", nargs="+", default=["r2", "r4", "r6"])
    p.add_argument("--curvature_k", type=int, default=30)
    p.add_argument("--output", default="results/metrics/all_metrics.csv")
    p.add_argument("--sequences", nargs="+", default=SEQUENCES)

    # Merge subcommand
    merge_p = sub.add_parser("merge", help="Merge per-sequence CSVs into one")
    merge_p.add_argument("--inputs", nargs="+", required=True)
    merge_p.add_argument("--output", required=True)

    args = p.parse_args()

    if args.cmd == "merge":
        run_merge(args)
    else:
        if args.tag is None:
            p.error("--tag is required")
        run_eval(args)


if __name__ == "__main__":
    main()
