"""Visualization script for multi-frame benchmark results.

Generates comprehensive plots including:
- Rate-distortion curves with error bars
- Per-sequence comparisons
- Multiple quality metrics (PSNR, D1, D2)
- Compression time analysis
- Handles G-PCC (lossless) separately
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication-quality style
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    }
)


def plot_rd_curve_with_error_bars(stats_df, output_path, show_gpcc=True):
    """Plot rate-distortion curve with error bars (mean ± std).

    Args:
        stats_df: DataFrame with statistics (mean, std)
        output_path: Where to save the plot
        show_gpcc: Whether to include G-PCC (lossless point)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"G-PCC": "#2E86AB", "pcc_geo_cnn_v2": "#A23B72"}

    markers = {"lossless": "s", "c1": "o", "c2": "^", "c3p": "D"}

    # Plot learned methods
    learned_data = stats_df[stats_df["method"] == "pcc_geo_cnn_v2"]
    for config in ["c1", "c2", "c3p"]:
        subset = learned_data[learned_data["config"] == config]
        if len(subset) > 0:
            row = subset.iloc[0]
            ax.errorbar(
                row["bpp_mean"],
                row["psnr_mean"],
                xerr=row["bpp_std"],
                yerr=row["psnr_std"],
                marker=markers[config],
                markersize=10,
                linewidth=2,
                capsize=5,
                capthick=2,
                label=f"pcc_geo_cnn_v2 ({config})",
                color=colors["pcc_geo_cnn_v2"],
                alpha=0.8,
            )

    # Handle G-PCC separately (lossless = inf PSNR)
    if show_gpcc:
        gpcc_data = stats_df[stats_df["method"] == "G-PCC"]
        if len(gpcc_data) > 0:
            row = gpcc_data.iloc[0]
            # Place G-PCC at top of plot with annotation
            ylim_max = learned_data["psnr_mean"].max() + 3
            ax.plot(
                row["bpp_mean"],
                ylim_max - 1,
                marker=markers["lossless"],
                markersize=12,
                color=colors["G-PCC"],
                label="G-PCC (lossless)",
                zorder=10,
            )
            ax.annotate(
                f"Lossless\n{row['bpp_mean']:.2f} BPP",
                xy=(row["bpp_mean"], ylim_max - 1),
                xytext=(row["bpp_mean"] + 3, ylim_max - 2),
                fontsize=9,
                ha="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    ax.set_xlabel("Bits Per Point (BPP)", fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontweight="bold")
    ax.set_title(
        "Rate-Distortion Curve\n(Mean ± Std over multiple frames)", fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.95)

    # Set reasonable limits
    if len(learned_data) > 0:
        ax.set_xlim(left=0, right=learned_data["bpp_mean"].max() + 5)
        if show_gpcc:
            ax.set_ylim(bottom=learned_data["psnr_mean"].min() - 2, top=ylim_max)
        else:
            ax.set_ylim(
                bottom=learned_data["psnr_mean"].min() - 2,
                top=learned_data["psnr_mean"].max() + 2,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_multi_metric_comparison(stats_df, output_path):
    """Plot multiple quality metrics (PSNR, D1, D2) vs BPP.

    Args:
        stats_df: DataFrame with statistics
        output_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    learned_data = stats_df[stats_df["method"] == "pcc_geo_cnn_v2"]

    metrics = [
        ("bpp_mean", "psnr_mean", "psnr_std", "PSNR (dB)", "higher is better"),
        ("bpp_mean", "d1_mean", "d1_std", "D1 (Max Point-to-Point)", "lower is better"),
        (
            "bpp_mean",
            "d2_sym_mean",
            "d2_sym_std",
            "D2 Symmetric (Hausdorff)",
            "lower is better",
        ),
        (
            "compression_time_mean",
            "psnr_mean",
            "psnr_std",
            "PSNR vs Time",
            "compression time",
        ),
    ]

    for idx, (ax, (x_col, y_col, y_std_col, title, note)) in enumerate(
        zip(axes.flat, metrics)
    ):
        for config in ["c1", "c2", "c3p"]:
            subset = learned_data[learned_data["config"] == config]
            if len(subset) > 0 and y_col in subset.columns:
                row = subset.iloc[0]
                x_val = row[x_col]
                y_val = row[y_col]
                y_std = row[y_std_col] if y_std_col in subset.columns else 0

                ax.errorbar(
                    x_val,
                    y_val,
                    yerr=y_std if y_std_col else None,
                    marker="o" if config == "c1" else ("^" if config == "c2" else "D"),
                    markersize=8,
                    linewidth=2,
                    capsize=5,
                    label=config,
                )

        ax.set_xlabel(
            x_col.replace("_mean", "").replace("_", " ").upper(), fontweight="bold"
        )
        ax.set_ylabel(title.split("(")[0].strip(), fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        ax.text(
            0.05,
            0.95,
            note,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    plt.suptitle("Multi-Metric Quality Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_per_sequence_comparison(per_seq_df, output_path):
    """Plot performance across different sequences.

    Args:
        per_seq_df: DataFrame with per-sequence statistics
        output_path: Where to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    learned_data = per_seq_df[per_seq_df["method"] == "pcc_geo_cnn_v2"]

    # Plot 1: BPP by sequence
    ax = axes[0]
    sequences = learned_data["sequence"].unique()
    configs = ["c1", "c2", "c3p"]
    x = np.arange(len(sequences))
    width = 0.25

    for i, config in enumerate(configs):
        subset = learned_data[learned_data["config"] == config]
        means = [
            (
                subset[subset["sequence"] == seq]["bpp_mean"].values[0]
                if len(subset[subset["sequence"] == seq]) > 0
                else 0
            )
            for seq in sequences
        ]
        stds = [
            (
                subset[subset["sequence"] == seq]["bpp_std"].values[0]
                if len(subset[subset["sequence"] == seq]) > 0
                else 0
            )
            for seq in sequences
        ]

        ax.bar(
            x + i * width, means, width, yerr=stds, label=config, capsize=3, alpha=0.8
        )

    ax.set_xlabel("Sequence", fontweight="bold")
    ax.set_ylabel("Bits Per Point (BPP)", fontweight="bold")
    ax.set_title("Compression Efficiency by Sequence", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sequences, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: PSNR by sequence
    ax = axes[1]
    for i, config in enumerate(configs):
        subset = learned_data[learned_data["config"] == config]
        means = [
            (
                subset[subset["sequence"] == seq]["psnr_mean"].values[0]
                if len(subset[subset["sequence"] == seq]) > 0
                else 0
            )
            for seq in sequences
        ]
        stds = [
            (
                subset[subset["sequence"] == seq]["psnr_std"].values[0]
                if len(subset[subset["sequence"] == seq]) > 0
                else 0
            )
            for seq in sequences
        ]

        ax.bar(
            x + i * width, means, width, yerr=stds, label=config, capsize=3, alpha=0.8
        )

    ax.set_xlabel("Sequence", fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontweight="bold")
    ax.set_title("Quality by Sequence", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sequences, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_compression_time_comparison(stats_df, output_path):
    """Plot compression time comparison.

    Args:
        stats_df: DataFrame with statistics
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get all methods
    methods_configs = []
    times = []
    stds = []

    for _, row in stats_df.iterrows():
        label = (
            f"{row['method']}\n{row['config']}"
            if row["method"] != "G-PCC"
            else "G-PCC\nlossless"
        )
        methods_configs.append(label)
        times.append(row["time_mean"])
        stds.append(row.get("time_std", 0))

    x = np.arange(len(methods_configs))
    bars = ax.bar(x, times, yerr=stds, capsize=5, alpha=0.7)

    # Color G-PCC differently
    for i, bar in enumerate(bars):
        if "G-PCC" in methods_configs[i]:
            bar.set_color("#2E86AB")
        else:
            bar.set_color("#A23B72")

    ax.set_ylabel("Compression Time (seconds)", fontweight="bold")
    ax.set_title("Compression Time Comparison\n(Mean ± Std)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_configs)
    ax.grid(True, alpha=0.3, axis="y")

    # Add speedup annotations relative to slowest
    max_time = max(times)
    for i, (t, label) in enumerate(zip(times, methods_configs)):
        if t > 0:
            speedup = max_time / t
            if speedup > 1.5:
                ax.text(
                    i,
                    t + max(stds),
                    f"{speedup:.0f}×\nfaster",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(stats_df, output_path):
    """Create a formatted summary table.

    Args:
        stats_df: DataFrame with statistics
        output_path: Where to save the table image
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = []
    headers = ["Method", "Config", "Frames", "BPP", "PSNR (dB)", "D1", "D2", "Time (s)"]

    for _, row in stats_df.iterrows():
        # Handle infinity for G-PCC PSNR
        psnr_str = (
            "∞ (lossless)"
            if row["psnr_mean"] == float("inf")
            else f"{row['psnr_mean']:.2f} ± {row['psnr_std']:.2f}"
        )

        table_data.append(
            [
                row["method"],
                row["config"],
                f"{int(row['num_frames'])}",
                f"{row['bpp_mean']:.2f} ± {row['bpp_std']:.2f}",
                psnr_str,
                (
                    f"{row.get('d1_mean', 0):.3f} ± {row.get('d1_std', 0):.3f}"
                    if "d1_mean" in row
                    else "N/A"
                ),
                (
                    f"{row.get('d2_sym_mean', 0):.3f} ± {row.get('d2_sym_std', 0):.3f}"
                    if "d2_sym_mean" in row
                    else "N/A"
                ),
                f"{row['time_mean']:.1f} ± {row.get('time_std', 0):.1f}",
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["lightgray"] * len(headers),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color G-PCC row
    for i, row_data in enumerate(table_data):
        if "G-PCC" in row_data[0]:
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor("#E3F2FD")

    plt.title(
        "Comprehensive Results Summary\n(Mean ± Standard Deviation)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-frame benchmark results"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as input)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multi-Frame Results Visualization")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load statistics
    overall_stats_path = input_dir / "statistics_overall.csv"
    per_seq_stats_path = input_dir / "statistics_per_sequence.csv"

    if not overall_stats_path.exists():
        print(f"ERROR: {overall_stats_path} not found!")
        print("Run benchmark_multi_frame.py first.")
        return

    overall_df = pd.read_csv(overall_stats_path)
    print(f"✓ Loaded overall statistics: {len(overall_df)} entries")

    # Generate plots
    print("\nGenerating visualizations...")

    # 1. Rate-distortion curve with error bars
    plot_rd_curve_with_error_bars(
        overall_df, output_dir / "rd_curve_with_error_bars.png"
    )

    # 2. Multi-metric comparison
    # Add computed columns if missing
    if "d1_mean" not in overall_df.columns:
        # Load raw data to compute
        raw_path = input_dir / "results_all_frames.csv"
        if raw_path.exists():
            raw_df = pd.read_csv(raw_path)
            # Recompute statistics with all metrics
            stats = []
            for method in raw_df["method"].unique():
                for config in raw_df[raw_df["method"] == method]["config"].unique():
                    subset = raw_df[
                        (raw_df["method"] == method) & (raw_df["config"] == config)
                    ]
                    stats.append(
                        {
                            "method": method,
                            "config": config,
                            "num_frames": len(subset),
                            "bpp_mean": subset["bpp"].mean(),
                            "bpp_std": subset["bpp"].std(),
                            "psnr_mean": subset["psnr"].mean(),
                            "psnr_std": subset["psnr"].std(),
                            "d1_mean": subset["d1"].mean(),
                            "d1_std": subset["d1"].std(),
                            "d2_sym_mean": subset["d2_sym"].mean(),
                            "d2_sym_std": subset["d2_sym"].std(),
                            "time_mean": subset["compression_time"].mean(),
                            "time_std": subset["compression_time"].std(),
                        }
                    )
            overall_df = pd.DataFrame(stats)

    plot_multi_metric_comparison(overall_df, output_dir / "multi_metric_comparison.png")

    # 3. Per-sequence comparison
    if per_seq_stats_path.exists():
        per_seq_df = pd.read_csv(per_seq_stats_path)
        print(f"✓ Loaded per-sequence statistics: {len(per_seq_df)} entries")
        plot_per_sequence_comparison(
            per_seq_df, output_dir / "per_sequence_comparison.png"
        )

    # 4. Compression time comparison
    plot_compression_time_comparison(
        overall_df, output_dir / "compression_time_comparison.png"
    )

    # 5. Summary table
    create_summary_table(overall_df, output_dir / "summary_table.png")

    print()
    print("=" * 70)
    print("✅ All visualizations generated!")
    print("=" * 70)
    print(f"\nFiles saved to: {output_dir}/")
    print("  - rd_curve_with_error_bars.png")
    print("  - multi_metric_comparison.png")
    print("  - per_sequence_comparison.png")
    print("  - compression_time_comparison.png")
    print("  - summary_table.png")


if __name__ == "__main__":
    main()
