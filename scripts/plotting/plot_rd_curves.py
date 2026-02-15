"""Plot rate-distortion curves from benchmark results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_rd_curve(results_path, output_path=None, show=False):
    """Plot rate-distortion curves from benchmark results.

    Args:
        results_path: Path to benchmark_results.csv or .json
        output_path: Output path for plot (if None, auto-generated)
        show: Whether to display the plot interactively
    """
    results_path = Path(results_path)

    # Load results
    if results_path.suffix == ".csv":
        df = pd.read_csv(results_path)
    elif results_path.suffix == ".json":
        with open(results_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data["results"])
    else:
        raise ValueError(f"Unsupported file type: {results_path.suffix}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate lossless and lossy methods
    lossless_df = df[df["psnr"] == float("inf")]
    lossy_df = df[df["psnr"] != float("inf")]

    # Plot each lossy method
    methods = lossy_df["method"].unique()
    markers = {"G-PCC": "s", "pcc_geo_cnn_v2": "o", "Simple Baseline": "^"}
    colors = {"G-PCC": "#2E86AB", "pcc_geo_cnn_v2": "red", "Simple Baseline": "green"}

    for method in methods:
        method_df = lossy_df[lossy_df["method"] == method].sort_values("bpp")

        marker = markers.get(method, "o")
        color = colors.get(method, "black")

        # Plot line connecting points
        ax.plot(
            method_df["bpp"],
            method_df["psnr"],
            marker=marker,
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=method,
            color=color,
        )

        # Add config labels
        for _, row in method_df.iterrows():
            ax.annotate(
                row["config"],
                (row["bpp"], row["psnr"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.7,
            )

    # Plot lossless methods at top with special handling
    if len(lossless_df) > 0 and len(lossy_df) > 0:
        # Place lossless points above the highest lossy PSNR
        ylim_max = lossy_df["psnr"].max() + 3

        for _, row in lossless_df.iterrows():
            method = row["method"]
            marker = markers.get(method, "s")
            color = colors.get(method, "#2E86AB")

            # Plot point at top
            ax.plot(
                row["bpp"],
                ylim_max - 1,
                marker=marker,
                markersize=12,
                color=color,
                label=f"{method} (lossless)",
                zorder=10,
            )

            # Add annotation
            ax.annotate(
                f"Lossless\n{row['bpp']:.2f} BPP",
                xy=(row["bpp"], ylim_max - 1),
                xytext=(row["bpp"] + 2, ylim_max - 2),
                fontsize=9,
                ha="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    ax.set_xlabel("Bits Per Point (BPP)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Set reasonable axis limits
    ax.set_xlim(left=0)
    if df["psnr"].max() != float("inf"):
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = results_path.parent / "rd_curve.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

    return output_path


def plot_multiple_metrics(results_path, output_dir=None, show=False):
    """Plot multiple quality metrics comparison.

    Creates subplots for different metrics (PSNR, D1, D2, compression time).
    """
    results_path = Path(results_path)

    # Load results
    if results_path.suffix == ".csv":
        df = pd.read_csv(results_path)
    elif results_path.suffix == ".json":
        with open(results_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data["results"])
    else:
        raise ValueError(f"Unsupported file type: {results_path.suffix}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Compression Method Comparison", fontsize=16, fontweight="bold")

    methods = df["method"].unique()
    markers = {"G-PCC": "s", "pcc_geo_cnn_v2": "o", "Simple Baseline": "^"}
    colors = {"G-PCC": "blue", "pcc_geo_cnn_v2": "red", "Simple Baseline": "green"}

    # 1. Rate-Distortion (BPP vs PSNR)
    ax = axes[0, 0]
    for method in methods:
        method_df = df[df["method"] == method].sort_values("bpp")
        marker = markers.get(method, "o")
        color = colors.get(method, "black")
        ax.plot(
            method_df["bpp"],
            method_df["psnr"],
            marker=marker,
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=method,
            color=color,
        )
    ax.set_xlabel("BPP")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate-Distortion Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. BPP vs D1 (geometric error)
    ax = axes[0, 1]
    for method in methods:
        method_df = df[df["method"] == method].sort_values("bpp")
        marker = markers.get(method, "o")
        color = colors.get(method, "black")
        if method_df["d1"].notna().any():
            ax.plot(
                method_df["bpp"],
                method_df["d1"],
                marker=marker,
                linestyle="-",
                linewidth=2,
                markersize=8,
                label=method,
                color=color,
            )
    ax.set_xlabel("BPP")
    ax.set_ylabel("D1 (max one-directional distance)")
    ax.set_title("Geometric Error (D1)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. BPP vs D2 (Hausdorff)
    ax = axes[1, 0]
    for method in methods:
        method_df = df[df["method"] == method].sort_values("bpp")
        marker = markers.get(method, "o")
        color = colors.get(method, "black")
        if method_df["d2_sym"].notna().any():
            ax.plot(
                method_df["bpp"],
                method_df["d2_sym"],
                marker=marker,
                linestyle="-",
                linewidth=2,
                markersize=8,
                label=method,
                color=color,
            )
    ax.set_xlabel("BPP")
    ax.set_ylabel("D2 (Hausdorff distance)")
    ax.set_title("Geometric Error (D2)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Compression time comparison
    ax = axes[1, 1]
    for method in methods:
        method_df = df[df["method"] == method].sort_values("bpp")
        marker = markers.get(method, "o")
        color = colors.get(method, "black")
        ax.plot(
            method_df["bpp"],
            method_df["compression_time"],
            marker=marker,
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=method,
            color=color,
        )
    ax.set_xlabel("BPP")
    ax.set_ylabel("Compression Time (s)")
    ax.set_title("Compression Speed")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save figure
    if output_dir is None:
        output_dir = results_path.parent
    else:
        output_dir = Path(output_dir)

    output_path = output_dir / "metrics_comparison.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot rate-distortion curves")
    parser.add_argument(
        "--input",
        type=str,
        default="results/benchmark/benchmark_results.csv",
        help="Path to benchmark results (CSV or JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: same dir as input)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Create multi-metric comparison plot",
    )
    args = parser.parse_args()

    if args.multi:
        output_path = plot_multiple_metrics(args.input, args.output, args.show)
    else:
        output_path = plot_rd_curve(args.input, args.output, args.show)

    print(f"\n✓ Visualization complete: {output_path}")


if __name__ == "__main__":
    main()
