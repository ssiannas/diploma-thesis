"""k-Sensitivity analysis for curvature neighborhood sizes.

Loads cached decoded clouds (from analyze_oversmoothing.py) and recomputes
metrics at k=15, 30, 50 without re-running the codec.  This verifies that
our findings are not an artifact of the curvature estimation parameter.

Usage:
    python scripts/k_sensitivity.py \
        --cache-dir results/oversmoothing/decoded_clouds/longdress_vox10_1300 \
        --output-dir results/oversmoothing

    # Custom k values
    python scripts/k_sensitivity.py \
        --cache-dir results/oversmoothing/decoded_clouds/longdress_vox10_1300 \
        --k-values 10 20 30 50 \
        --output-dir results/oversmoothing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.metrics.curvature import CurvatureQualityCalculator

PEAK = 1023.0
DEFAULT_K_VALUES = [15, 30, 50]


def load_cached_clouds(cache_dir: Path, codec: str = "pcgcv2") -> dict:
    """Load original + all decoded clouds from cache directory.

    Returns:
        {
            "original": np.ndarray,
            "decoded": {rate_label: np.ndarray, ...},
            "meta": {rate_label: dict, ...},
        }
    """
    original_path = cache_dir / "original.npy"
    if not original_path.exists():
        raise FileNotFoundError(f"Original cloud not found: {original_path}")

    original = np.load(original_path)

    decoded = {}
    meta = {}
    for npy_path in sorted(cache_dir.glob(f"{codec}_*.npy")):
        if "_meta" in npy_path.stem:
            continue
        rate_label = npy_path.stem.replace(f"{codec}_", "")
        decoded[rate_label] = np.load(npy_path)

        meta_path = npy_path.with_name(f"{codec}_{rate_label}_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta[rate_label] = json.load(f)

    if not decoded:
        raise FileNotFoundError(
            f"No decoded clouds found for codec={codec} in {cache_dir}"
        )

    return {"original": original, "decoded": decoded, "meta": meta}


def run_k_sensitivity(
    cache_dir: Path,
    k_values: list[int],
    codec: str = "pcgcv2",
    n_strata: int = 3,
) -> list[dict]:
    """Run metrics at multiple k values on cached clouds.

    Returns:
        List of result dicts, one per (k, rate_label).
    """
    data = load_cached_clouds(cache_dir, codec)
    original = data["original"]
    seq_name = cache_dir.name

    print(f"\n{'='*60}")
    print(f"k-Sensitivity: {seq_name} [{codec}]")
    print(f"  Points: {len(original):,}")
    print(f"  Rates: {sorted(data['decoded'].keys())}")
    print(f"  k values: {k_values}")
    print(f"{'='*60}")

    results = []

    for k in k_values:
        # Compute / load curvature at this k
        curv_cache = cache_dir / f"curvature_k{k}.npy"
        if curv_cache.exists():
            print(f"\n  Loading cached curvature (k={k})...")
            curvature = np.load(curv_cache)
        else:
            print(f"\n  Computing curvature (k={k})...")
            t0 = time.time()
            curvature = CurvatureQualityCalculator.compute_curvature(original, k=k)
            np.save(curv_cache, curvature)
            print(f"  Done in {time.time() - t0:.1f}s")

        for rate_label in sorted(data["decoded"].keys()):
            decoded = data["decoded"][rate_label]
            meta = data["meta"].get(rate_label, {})

            num_input = len(original)
            compressed_bytes = meta.get("compressed_size_bytes", 0)
            bpp = (compressed_bytes * 8) / num_input if num_input > 0 else 0.0

            print(f"\n  k={k}, rate={rate_label} (BPP={bpp:.4f})")

            # Reverse stratified PSNR
            rev = CurvatureQualityCalculator.compute_stratified_psnr(
                original,
                decoded,
                curvature,
                peak=PEAK,
                n_strata=n_strata,
                direction="reverse",
                k=k,
            )

            # Forward-self
            fwd_self = CurvatureQualityCalculator.compute_stratified_psnr(
                original,
                decoded,
                curvature,
                peak=PEAK,
                n_strata=n_strata,
                direction="forward_self",
                k=k,
            )

            # Spearman correlation
            corr = CurvatureQualityCalculator.compute_curvature_error_correlation(
                original,
                decoded,
                curvature,
            )

            # NN multiplicity
            mult = CurvatureQualityCalculator.compute_nn_multiplicity(
                original,
                decoded,
                curvature,
                n_strata=n_strata,
            )

            # Continuous curve
            psnr_curve = CurvatureQualityCalculator.compute_psnr_vs_curvature_curve(
                original,
                decoded,
                curvature,
                peak=PEAK,
                n_bins=20,
                direction="reverse",
                k=k,
            )

            print(
                f"    Rev degradation: {rev.degradation:.2f} dB | "
                f"Spearman rho: {corr.rho:.4f}"
            )

            results.append(
                {
                    "curvature_k": k,
                    "codec": codec,
                    "sequence": seq_name,
                    "rate_label": rate_label,
                    "bpp": bpp,
                    "rev_flat_psnr": rev.flat_psnr,
                    "rev_medium_psnr": rev.medium_psnr,
                    "rev_edge_psnr": rev.edge_psnr,
                    "rev_degradation": rev.degradation,
                    "fwd_self_flat_psnr": fwd_self.flat_psnr,
                    "fwd_self_edge_psnr": fwd_self.edge_psnr,
                    "fwd_self_degradation": fwd_self.degradation,
                    "spearman_rho": corr.rho,
                    "spearman_pvalue": corr.p_value,
                    "mult_flat_mean": mult.per_stratum_mean.get("flat", 0.0),
                    "mult_edge_mean": mult.per_stratum_mean.get("edge", 0.0),
                    "mult_overall_mean": mult.overall_mean,
                    "psnr_curve_centers": psnr_curve.bin_centers.tolist(),
                    "psnr_curve_psnrs": psnr_curve.psnrs.tolist(),
                    "psnr_curve_counts": psnr_curve.counts.tolist(),
                }
            )

    return results


def generate_plots(
    results: list[dict],
    k_values: list[int],
    output_dir: Path,
) -> None:
    """Generate k-sensitivity comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_name = results[0]["sequence"]

    # Group by k
    by_k = {k: [r for r in results if r["curvature_k"] == k] for k in k_values}

    # Sort each group by BPP
    for k in k_values:
        by_k[k] = sorted(by_k[k], key=lambda r: r["bpp"])

    colors_k = {k_values[0]: "steelblue", k_values[-1]: "firebrick"}
    if len(k_values) == 3:
        colors_k[k_values[1]] = "darkorange"
    elif len(k_values) > 3:
        cmap = plt.cm.viridis
        for i, k in enumerate(k_values):
            colors_k[k] = cmap(i / (len(k_values) - 1))

    # 1. Reverse degradation vs BPP for each k
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in k_values:
        bpps = [r["bpp"] for r in by_k[k]]
        degs = [r["rev_degradation"] for r in by_k[k]]
        ax.plot(
            bpps,
            degs,
            "o-",
            color=colors_k[k],
            linewidth=2,
            markersize=7,
            label=f"k={k}",
        )
    ax.set_xlabel("Bits Per Point (BPP)")
    ax.set_ylabel("Rev Degradation (flat - edge PSNR) [dB]")
    ax.set_title(f"k-Sensitivity: Degradation — {seq_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    path = output_dir / "k_sensitivity_degradation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # 2. Spearman rho vs BPP for each k
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in k_values:
        bpps = [r["bpp"] for r in by_k[k]]
        rhos = [r["spearman_rho"] for r in by_k[k]]
        ax.plot(
            bpps,
            rhos,
            "o-",
            color=colors_k[k],
            linewidth=2,
            markersize=7,
            label=f"k={k}",
        )
    ax.set_xlabel("Bits Per Point (BPP)")
    ax.set_ylabel("Spearman rho (curvature vs error)")
    ax.set_title(f"k-Sensitivity: Correlation — {seq_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    path = output_dir / "k_sensitivity_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # 3. Continuous PSNR curves overlaid for a representative rate
    # Pick the median rate by BPP
    rate_labels = sorted(set(r["rate_label"] for r in results))
    mid_idx = len(rate_labels) // 2
    mid_rate = rate_labels[mid_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in k_values:
        r = [x for x in by_k[k] if x["rate_label"] == mid_rate]
        if r:
            r = r[0]
            ax.plot(
                r["psnr_curve_centers"],
                r["psnr_curve_psnrs"],
                "o-",
                color=colors_k[k],
                linewidth=2,
                markersize=5,
                label=f"k={k}",
            )
    ax.set_xlabel("Curvature (quantile-binned)")
    ax.set_ylabel("PSNR [dB]")
    ax.set_title(f"PSNR vs Curvature @ {mid_rate} — k-Sensitivity — {seq_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    path = output_dir / "k_sensitivity_psnr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # 4. Stability summary: for each rate, show relative variation across k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for rate_label in rate_labels:
        degs = []
        rhos = []
        for k in k_values:
            r = [x for x in by_k[k] if x["rate_label"] == rate_label]
            if r:
                degs.append(r[0]["rev_degradation"])
                rhos.append(r[0]["spearman_rho"])
        if len(degs) >= 2:
            mean_deg = np.mean(degs)
            range_deg = max(degs) - min(degs)
            rel_var_deg = range_deg / abs(mean_deg) if abs(mean_deg) > 0 else 0
            mean_rho = np.mean(rhos)
            range_rho = max(rhos) - min(rhos)
            rel_var_rho = range_rho / abs(mean_rho) if abs(mean_rho) > 0 else 0
            bpp = [x for x in by_k[k_values[0]] if x["rate_label"] == rate_label][0][
                "bpp"
            ]
            ax1.bar(rate_label, rel_var_deg * 100, color="steelblue", alpha=0.7)
            ax2.bar(rate_label, rel_var_rho * 100, color="firebrick", alpha=0.7)

    ax1.set_ylabel("Relative Variation (%)")
    ax1.set_title("Rev Degradation: Variation across k")
    ax1.axhline(y=15, color="red", linestyle="--", alpha=0.5, label="15% threshold")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    ax2.set_ylabel("Relative Variation (%)")
    ax2.set_title("Spearman rho: Variation across k")
    ax2.axhline(y=15, color="red", linestyle="--", alpha=0.5, label="15% threshold")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "k_sensitivity_stability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="k-Sensitivity analysis for curvature-stratified metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to decoded_clouds/<sequence> directory.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="pcgcv2",
        choices=["pcgcv2", "gpcc"],
        help="Codec whose cached clouds to analyze.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="Curvature neighborhood sizes to test (default: 15 30 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/oversmoothing",
        help="Output directory for results and plots.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    results = run_k_sensitivity(
        cache_dir=cache_dir,
        k_values=args.k_values,
        codec=args.codec,
    )

    if not results:
        print("No results generated.")
        sys.exit(1)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out variable-length arrays for JSON scalar output
    scalar_results = []
    for r in results:
        sr = {
            k: v
            for k, v in r.items()
            if k not in ("psnr_curve_centers", "psnr_curve_psnrs", "psnr_curve_counts")
        }
        scalar_results.append(sr)

    json_path = output_dir / "k_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(scalar_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Generate plots
    generate_plots(results, args.k_values, output_dir)

    # Print stability summary
    print(f"\n{'='*60}")
    print("STABILITY SUMMARY")
    print(f"{'='*60}")
    rate_labels = sorted(set(r["rate_label"] for r in results))
    by_k = {k: [r for r in results if r["curvature_k"] == k] for k in args.k_values}

    hdr = (
        f"{'Rate':<10} {'Deg range [dB]':<18} "
        f"{'Deg rel var %':<16} {'Rho range':<14} "
        f"{'Rho rel var %':<16}"
    )
    print(hdr)
    for rl in rate_labels:
        degs = []
        rhos = []
        for k in args.k_values:
            r = [x for x in by_k[k] if x["rate_label"] == rl]
            if r:
                degs.append(r[0]["rev_degradation"])
                rhos.append(r[0]["spearman_rho"])
        if len(degs) >= 2:
            deg_range = max(degs) - min(degs)
            deg_mean = np.mean(degs)
            deg_rv = (deg_range / abs(deg_mean) * 100) if abs(deg_mean) > 0 else 0
            rho_range = max(rhos) - min(rhos)
            rho_mean = np.mean(rhos)
            rho_rv = (rho_range / abs(rho_mean) * 100) if abs(rho_mean) > 0 else 0
            print(
                f"{rl:<10} {deg_range:<18.3f} "
                f"{deg_rv:<16.1f} {rho_range:<14.4f} "
                f"{rho_rv:<16.1f}"
            )


if __name__ == "__main__":
    main()
