"""Plot rate-distortion curves from evaluation CSV.

Usage:
    python scripts/plotting/plot_rd_curve.py \
        --csv results/metrics/nocurv_ep24_all.csv \
        --unicorn_csv /path/to/unicorn_postprocessed_results.csv \
        --output paper/figures/rd_curve.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BPP = {"r2": 0.05, "r4": 0.15, "r6": 0.30}
SEQUENCES = ["longdress", "loot", "redandblack", "soldier"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/metrics/nocurv_ep24_all.csv")
    p.add_argument("--unicorn_csv", default=None)
    p.add_argument("--output", default="paper/figures/rd_curve.png")
    p.add_argument("--per_sequence", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["bpp"] = df["rate"].map(BPP)

    per_seq = (
        df.groupby(["sequence", "rate", "bpp"])[
            ["dec_d1", "ref_d1", "dec_edge", "ref_edge", "dec_flat", "ref_flat"]
        ]
        .mean()
        .reset_index()
    )

    avg = (
        per_seq.groupby(["rate", "bpp"])[
            ["dec_d1", "ref_d1", "dec_edge", "ref_edge", "dec_flat", "ref_flat"]
        ]
        .mean()
        .reset_index()
        .sort_values("bpp")
    )

    bpp_vals = avg["bpp"].values

    # Load Unicorn data if provided
    u_avg = None
    if args.unicorn_csv:
        udf = pd.read_csv(args.unicorn_csv)
        u_avg = (
            udf.groupby("rate_idx")[
                [
                    "bpp",
                    "base_d1",
                    "ref_d1",
                    "base_edge",
                    "ref_edge",
                    "base_flat",
                    "ref_flat",
                ]
            ]
            .mean()
            .reset_index()
            .sort_values("bpp")
        )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    # (pcgcv2_dec, pcgcv2_ref, ylabel, unicorn_base, unicorn_ref)
    panels = [
        ("dec_d1", "ref_d1", "D1 PSNR (dB)", "base_d1", "ref_d1"),
        ("dec_edge", "ref_edge", "Edge PSNR (dB)", "base_edge", "ref_edge"),
        ("dec_flat", "ref_flat", "Flat PSNR (dB)", "base_flat", "ref_flat"),
    ]

    for ax, (dec_col, ref_col, ylabel, u_base, u_ref) in zip(axes, panels):
        if args.per_sequence:
            for seq in SEQUENCES:
                s = per_seq[per_seq["sequence"] == seq].sort_values("bpp")
                ax.plot(
                    s["bpp"],
                    s[dec_col],
                    color="#888888",
                    lw=0.8,
                    alpha=0.4,
                    linestyle="--",
                )
                ax.plot(s["bpp"], s[ref_col], color="#4488cc", lw=0.8, alpha=0.4)

        # PCGCv2 curves
        ax.plot(
            bpp_vals,
            avg[dec_col],
            color="#555555",
            lw=2.0,
            marker="o",
            markersize=5,
            label="PCGCv2 (baseline)",
            linestyle="--",
        )
        ax.plot(
            bpp_vals,
            avg[ref_col],
            color="#1a6faf",
            lw=2.0,
            marker="s",
            markersize=5,
            label="Proposed on PCGCv2",
        )

        for bpp, dec, ref in zip(bpp_vals, avg[dec_col], avg[ref_col]):
            delta = ref - dec
            ax.annotate(
                f"+{delta:.2f} dB",
                xy=(bpp, ref),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#1a6faf",
                clip_on=False,
            )

        # Unicorn curves (Edge and Flat panels only)
        if u_avg is not None and u_base is not None:
            u_bpp = u_avg["bpp"].values
            ax.plot(
                u_bpp,
                u_avg[u_base].values,
                color="#998833",
                lw=2.0,
                marker="^",
                markersize=5,
                label="Unicorn (baseline)",
                linestyle="--",
            )
            ax.plot(
                u_bpp,
                u_avg[u_ref].values,
                color="#b85010",
                lw=2.0,
                marker="D",
                markersize=5,
                label="Proposed on Unicorn (zero-shot)",
            )

            for bpp, dec, ref in zip(u_bpp, u_avg[u_base].values, u_avg[u_ref].values):
                delta = ref - dec
                sign = "+" if delta >= 0 else ""
                ax.annotate(
                    f"{sign}{delta:.2f}",
                    xy=(bpp, ref),
                    xytext=(0, -11),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.5,
                    color="#b85010",
                    clip_on=False,
                )

        ax.set_xlabel("Bitrate (bpp)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=7.5, loc="lower right")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - (ymax - ymin) * 0.06, ymax + (ymax - ymin) * 0.14)

    fig.suptitle(
        "Rate-Distortion: PCGCv2 and Unicorn vs. Proposed (avg. 4 sequences)",
        fontsize=11,
    )
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
