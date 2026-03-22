"""
Compute BD-PSNR (Bjontegaard Delta PSNR) from nocurv_ep24_all.csv.

Uses piecewise linear (trapezoidal) integration over log(BPP) since we have
only 3 rate points (r2, r4, r6). Standard cubic Bjontegaard requires 4+.

BD-PSNR = (integral of PSNR_proposed - integral of PSNR_baseline) / log_bpp_range
Positive values mean the proposed method is better.
"""

import numpy as np
import pandas as pd

# Actual mean BPP values from pcgcv2 oversmoothing evaluation
BPP = {"r2": 0.0487, "r4": 0.1535, "r6": 0.3186}

METRICS = {
    "D1 PSNR": ("dec_d1", "ref_d1"),
    "Edge PSNR": ("dec_edge", "ref_edge"),
    "Flat PSNR": ("dec_flat", "ref_flat"),
}


def bd_psnr_trapz(bpps, psnr_a, psnr_b):
    """
    BD-PSNR via piecewise linear integration over log(BPP).
    psnr_a: baseline, psnr_b: proposed.
    Returns average PSNR improvement of b over a.
    """
    log_bpp = np.log(bpps)
    delta = np.array(psnr_b) - np.array(psnr_a)
    integral = np.trapz(delta, log_bpp)
    return integral / (log_bpp[-1] - log_bpp[0])


def compute(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    rates = ["r2", "r4", "r6"]
    bpps = [BPP[r] for r in rates]
    sequences = sorted(df["sequence"].unique())

    print(f"BD-PSNR (piecewise linear, 3 rate points: r2/r4/r6)")
    print(f"BPP operating points: {bpps}")
    print(f"Sequences: {sequences}")
    print()

    results = {}

    for metric_name, (col_base, col_prop) in METRICS.items():
        per_seq = {}
        for seq in sequences:
            seq_df = df[df["sequence"] == seq]
            base_psnr = []
            prop_psnr = []
            for r in rates:
                rate_df = seq_df[seq_df["rate"] == r]
                base_psnr.append(rate_df[col_base].mean())
                prop_psnr.append(rate_df[col_prop].mean())
            bd = bd_psnr_trapz(bpps, base_psnr, prop_psnr)
            per_seq[seq] = bd

        avg = np.mean(list(per_seq.values()))
        results[metric_name] = {"per_seq": per_seq, "avg": avg}

    # Print table
    header = (
        f"{'Metric':<14}" + "".join(f"{s:>14}" for s in sequences) + f"{'Average':>14}"
    )
    print(header)
    print("-" * len(header))
    for metric_name, data in results.items():
        row = f"{metric_name:<14}"
        for seq in sequences:
            row += f"{data['per_seq'][seq]:>+14.3f}"
        row += f"{data['avg']:>+14.3f}"
        print(row)

    print()
    print(
        "Note: 3 rate points only. Standard Bjontegaard uses cubic fit over 4+ points."
    )
    print("      Piecewise linear gives a conservative (lower) estimate of BD-PSNR.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="results/metrics/nocurv_ep24_all.csv",
        help="Path to metrics CSV",
    )
    args = parser.parse_args()
    compute(args.csv)
