"""Oversmoothing analysis for point cloud codecs across rate points.

Runs PCGCv2 or G-PCC at multiple rate points on one or more point clouds,
computes curvature-stratified quality metrics, and generates publication-quality
visualizations that reveal spatial patterns of oversmoothing.

Decoded clouds are cached as .npy files so expensive codec runs only happen
once.  Re-running the script reuses cached clouds and recomputes metrics.

Usage:
    # Quick single-frame test (PCGCv2, default)
    python scripts/analyze_oversmoothing.py \
        --input datasets/8iVFB_small/longdress_vox10_1300.ply \
        --rates r1 r3 r7 \
        --output-dir results/oversmoothing

    # All rates, all 4 8iVFB sequences
    python scripts/analyze_oversmoothing.py \
        --input-dir datasets/8iVFB_small/ \
        --rates r1 r2 r3 r4 r5 r6 r7 \
        --output-dir results/oversmoothing \
        --random-control

    # G-PCC baseline at multiple quantization scales
    python scripts/analyze_oversmoothing.py \
        --input-dir datasets/8iVFB_small/ \
        --codec gpcc \
        --output-dir results/oversmoothing
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import PLYPointCloudLoader
from pcml.frameworks.gpcc import GPCCAdapter
from pcml.frameworks.pcgcv2 import PCGCv2Adapter
from pcml.metrics.curvature import CurvatureQualityCalculator
from pcml.metrics.quality import GeometryQualityCalculator
from pcml.visualization.oversmoothing import (
    create_oversmoothing_report,
    plot_correlation_vs_rate,
    plot_degradation_vs_rate,
    plot_nn_multiplicity,
    plot_psnr_vs_curvature_curve,
    plot_stratified_psnr_bars,
)

logger = logging.getLogger(__name__)

ALL_RATES = ["r1", "r2", "r3", "r4", "r5", "r6", "r7"]

# Default G-PCC quantization scales (geometrically spaced, low→high bitrate)
GPCC_CODING_SCALES = [0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]

# 10-bit voxelized clouds use peak = 1023
PEAK = 1023.0


def _rate_label(codec: str, rate: str, coding_scale: float | None = None) -> str:
    """Canonical rate label for cache filenames and result keys."""
    if codec == "pcgcv2":
        return rate  # e.g. "r1"
    else:
        return f"qs{coding_scale}"  # e.g. "qs0.125"


def _cache_stem(codec: str, rate_label: str) -> str:
    """Filename stem for cached decoded cloud and metadata."""
    return f"{codec}_{rate_label}"


def _load_or_run_codec(
    codec: str,
    geometry: np.ndarray,
    rate_label: str,
    decoded_dir: Path,
    rate: str | None = None,
    coding_scale: float | None = None,
) -> tuple[np.ndarray, int, float]:
    """Load decoded cloud from cache, or run codec and cache the result.

    Returns:
        (decoded_geometry, compressed_size_bytes, codec_time_s)
    """
    stem = _cache_stem(codec, rate_label)
    cloud_path = decoded_dir / f"{stem}.npy"
    meta_path = decoded_dir / f"{stem}_meta.json"

    if cloud_path.exists() and meta_path.exists():
        decoded_geometry = np.load(cloud_path)
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  [cache hit] {cloud_path.name}")
        return decoded_geometry, meta["compressed_size_bytes"], meta["codec_time_s"]

    # Run codec
    if codec == "pcgcv2":
        adapter = PCGCv2Adapter(rate_point=rate)
        t0 = time.time()
        comp_result, decoded_geometry = adapter.compress_and_decompress(geometry)
        codec_time = time.time() - t0
    else:
        adapter = GPCCAdapter(coding_scale=coding_scale)
        t0 = time.time()
        comp_result, decoded_geometry = adapter.compress_and_decompress(geometry)
        codec_time = time.time() - t0

    # Save to cache
    decoded_dir.mkdir(parents=True, exist_ok=True)
    np.save(cloud_path, decoded_geometry)
    meta = {
        "compressed_size_bytes": comp_result.compressed_size_bytes,
        "codec_time_s": codec_time,
        "num_points_output": len(decoded_geometry),
        "codec": codec,
        "rate_label": rate_label,
    }
    if codec == "pcgcv2":
        meta["rate_point"] = rate
    else:
        meta["coding_scale"] = coding_scale
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [cached] {cloud_path.name}")
    return decoded_geometry, comp_result.compressed_size_bytes, codec_time


def analyze_single_file(
    input_path: Path,
    codec: str,
    rate_specs: list[dict],
    output_dir: Path,
    n_strata: int = 3,
    curvature_k: int = 30,
    random_control: bool = False,
) -> list[dict]:
    """Run oversmoothing analysis on a single point cloud file.

    Args:
        input_path: Path to input PLY file.
        codec: "pcgcv2" or "gpcc".
        rate_specs: List of dicts with rate info. Each has:
            - rate_label: str (e.g. "r1" or "qs0.125")
            - rate: str or None (PCGCv2 rate point name)
            - coding_scale: float or None (G-PCC quantization scale)
        output_dir: Directory for per-file outputs.
        n_strata: Number of curvature strata (3 or 5).
        curvature_k: Number of neighbors for PCA curvature.
        random_control: If True, run random stratification sanity check.

    Returns:
        List of result dicts (one per rate point).
    """
    loader = PLYPointCloudLoader()
    pc = loader.load(input_path)
    geometry = pc.geometry
    stem = input_path.stem

    file_output_dir = output_dir / stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Decoded clouds cache directory
    decoded_dir = output_dir / "decoded_clouds" / stem
    decoded_dir.mkdir(parents=True, exist_ok=True)

    # Cache original geometry
    orig_cache = decoded_dir / "original.npy"
    if not orig_cache.exists():
        np.save(orig_cache, geometry)

    rate_labels = [rs["rate_label"] for rs in rate_specs]
    print(f"\n{'='*60}")
    print(f"Analyzing: {stem} [{codec}]")
    print(f"  Points: {len(geometry):,}")
    print(f"  Rates: {', '.join(rate_labels)}")
    print(f"{'='*60}")

    # Compute curvature once (expensive, reuse across rates)
    curv_cache = decoded_dir / f"curvature_k{curvature_k}.npy"
    if curv_cache.exists():
        print(f"  Loading cached curvature (k={curvature_k})...")
        curvature = np.load(curv_cache)
    else:
        print(f"  Computing PCA curvature (k={curvature_k})...")
        t0 = time.time()
        curvature = CurvatureQualityCalculator.compute_curvature(
            geometry, k=curvature_k
        )
        curv_time = time.time() - t0
        np.save(curv_cache, curvature)
        print(f"  Curvature computed in {curv_time:.1f}s")

    print(
        f"  Curvature stats: mean={curvature.mean():.4f}, "
        f"std={curvature.std():.4f}, max={curvature.max():.4f}"
    )

    results = []

    for rs in rate_specs:
        rl = rs["rate_label"]
        print(f"\n  --- Rate: {rl} ---")

        # Compress + decompress (with caching)
        try:
            decoded_geometry, compressed_bytes, codec_time = _load_or_run_codec(
                codec=codec,
                geometry=geometry,
                rate_label=rl,
                decoded_dir=decoded_dir,
                rate=rs.get("rate"),
                coding_scale=rs.get("coding_scale"),
            )
        except Exception as e:
            print(f"  ERROR: {codec} failed at {rl}: {e}")
            continue

        num_points_input = len(geometry)
        num_points_output = len(decoded_geometry)
        bpp = (compressed_bytes * 8) / num_points_input

        print(
            f"  Codec: {codec_time:.1f}s | "
            f"In: {num_points_input:,} | Out: {num_points_output:,} | "
            f"BPP: {bpp:.4f}"
        )

        # Forward-direction stratified PSNR (original method)
        fwd_metrics = CurvatureQualityCalculator.compute_stratified_psnr(
            geometry,
            decoded_geometry,
            curvature,
            peak=PEAK,
            n_strata=n_strata,
            direction="forward",
            k=curvature_k,
        )

        # Reverse-direction stratified PSNR (fixes r1 anomaly)
        rev_metrics = CurvatureQualityCalculator.compute_stratified_psnr(
            geometry,
            decoded_geometry,
            curvature,
            peak=PEAK,
            n_strata=n_strata,
            direction="reverse",
            k=curvature_k,
        )

        # Forward-self: stratified by reconstructed curvature (no NN assignment)
        fwd_self_metrics = CurvatureQualityCalculator.compute_stratified_psnr(
            geometry,
            decoded_geometry,
            curvature,
            peak=PEAK,
            n_strata=n_strata,
            direction="forward_self",
            k=curvature_k,
        )

        # Spearman correlation (curvature vs D1 error)
        corr = CurvatureQualityCalculator.compute_curvature_error_correlation(
            geometry,
            decoded_geometry,
            curvature,
        )

        # NN multiplicity diagnostic
        mult = CurvatureQualityCalculator.compute_nn_multiplicity(
            geometry,
            decoded_geometry,
            curvature,
            n_strata=n_strata,
        )

        # Continuous PSNR-vs-curvature curve (reverse direction)
        psnr_curve = CurvatureQualityCalculator.compute_psnr_vs_curvature_curve(
            geometry,
            decoded_geometry,
            curvature,
            peak=PEAK,
            n_bins=20,
            direction="reverse",
            k=curvature_k,
        )

        # Overall PSNR (via cKDTree, same as stratified internally)
        dists, _ = CurvatureQualityCalculator.compute_per_point_error(
            geometry, decoded_geometry
        )
        overall_mse = float(np.mean(dists**2))
        overall_psnr = (
            10.0 * np.log10(PEAK**2 / overall_mse) if overall_mse > 0 else float("inf")
        )

        print(f"  Overall PSNR:      {overall_psnr:.2f} dB")
        print(f"  Fwd Flat PSNR:     {fwd_metrics.flat_psnr:.2f} dB")
        print(f"  Fwd Edge PSNR:     {fwd_metrics.edge_psnr:.2f} dB")
        print(f"  Fwd Degradation:   {fwd_metrics.degradation:.2f} dB")
        print(f"  Rev Flat PSNR:     {rev_metrics.flat_psnr:.2f} dB")
        print(f"  Rev Edge PSNR:     {rev_metrics.edge_psnr:.2f} dB")
        print(f"  Rev Degradation:   {rev_metrics.degradation:.2f} dB")
        print(f"  FwdSelf Flat PSNR: {fwd_self_metrics.flat_psnr:.2f} dB")
        print(f"  FwdSelf Edge PSNR: {fwd_self_metrics.edge_psnr:.2f} dB")
        print(f"  FwdSelf Degrad:    {fwd_self_metrics.degradation:.2f} dB")
        print(f"  Spearman rho:      {corr.rho:.4f} (p={corr.p_value:.2e})")
        print(
            f"  NN multiplicity:   flat={mult.per_stratum_mean.get('flat', 0):.2f}, "
            f"edge={mult.per_stratum_mean.get('edge', 0):.2f}"
        )

        result = {
            "codec": codec,
            "sequence": stem,
            "rate_label": rl,
            "coding_scale": rs.get("coding_scale"),
            "bpp": bpp,
            "overall_psnr": overall_psnr,
            # Forward direction (original method)
            "flat_psnr": fwd_metrics.flat_psnr,
            "medium_psnr": fwd_metrics.medium_psnr,
            "edge_psnr": fwd_metrics.edge_psnr,
            "degradation": fwd_metrics.degradation,
            "curvature_kl": fwd_metrics.curvature_kl,
            # Reverse direction (fixes r1)
            "rev_flat_psnr": rev_metrics.flat_psnr,
            "rev_medium_psnr": rev_metrics.medium_psnr,
            "rev_edge_psnr": rev_metrics.edge_psnr,
            "rev_degradation": rev_metrics.degradation,
            # Forward-self direction (stratified by rec curvature)
            "fwd_self_flat_psnr": fwd_self_metrics.flat_psnr,
            "fwd_self_medium_psnr": fwd_self_metrics.medium_psnr,
            "fwd_self_edge_psnr": fwd_self_metrics.edge_psnr,
            "fwd_self_degradation": fwd_self_metrics.degradation,
            # Spearman correlation
            "spearman_rho": corr.rho,
            "spearman_pvalue": corr.p_value,
            # NN multiplicity
            "mult_flat_mean": mult.per_stratum_mean.get("flat", 0.0),
            "mult_medium_mean": mult.per_stratum_mean.get("medium", 0.0),
            "mult_edge_mean": mult.per_stratum_mean.get("edge", 0.0),
            "mult_overall_mean": mult.overall_mean,
            # Continuous PSNR curve (store as lists for JSON)
            "psnr_curve_centers": psnr_curve.bin_centers.tolist(),
            "psnr_curve_psnrs": psnr_curve.psnrs.tolist(),
            "psnr_curve_counts": psnr_curve.counts.tolist(),
            # Metadata
            "num_points_input": num_points_input,
            "num_points_output": num_points_output,
            "compressed_bytes": compressed_bytes,
            "codec_time_s": codec_time,
            "curvature_k": curvature_k,
            "point_counts": fwd_metrics.point_counts,
        }

        # Assignment accuracy diagnostic (expensive — requires curvature on rec cloud)
        print(f"  Computing assignment accuracy...")
        t0 = time.time()
        accuracy = CurvatureQualityCalculator.compute_assignment_accuracy(
            geometry,
            decoded_geometry,
            curvature,
            k=curvature_k,
        )
        acc_time = time.time() - t0
        result["assignment_accuracy"] = accuracy
        print(f"  Assignment accuracy: {accuracy:.3f} ({acc_time:.1f}s)")

        # Random control sanity check
        if random_control:
            print(f"  Running random control (100 iterations)...")
            t0 = time.time()
            rand_result = CurvatureQualityCalculator.compute_random_stratified_psnr(
                geometry,
                decoded_geometry,
                curvature,
                peak=PEAK,
                n_strata=n_strata,
                direction="reverse",
            )
            rand_time = time.time() - t0
            result["random_mean_degradation"] = rand_result["mean_degradation"]
            result["random_std_degradation"] = rand_result["std_degradation"]
            print(
                f"  Random control: {rand_result['mean_degradation']:.4f} "
                f"+/- {rand_result['std_degradation']:.4f} dB ({rand_time:.1f}s)"
            )

        results.append(result)

        # Generate per-rate report figure
        report_path = file_output_dir / f"report_{codec}_{rl}.png"
        print(f"  Generating report: {report_path}")
        fig = create_oversmoothing_report(
            geometry,
            decoded_geometry,
            curvature,
            metrics=result,
            rate_label=f"{stem} @ {rl} [{codec}]",
            output_path=str(report_path),
        )
        plt.close(fig)

        # Continuous PSNR-vs-curvature curve
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_psnr_vs_curvature_curve(
            psnr_curve.bin_centers,
            psnr_curve.psnrs,
            psnr_curve.counts,
            ax=ax,
            title=f"PSNR vs Curvature (rev) — {stem} @ {rl} [{codec}]",
        )
        curve_path = file_output_dir / f"psnr_vs_curvature_{codec}_{rl}.png"
        fig.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # NN multiplicity bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_nn_multiplicity(
            mult.per_stratum_mean,
            ax=ax,
            title=f"NN Multiplicity — {stem} @ {rl} [{codec}]",
        )
        mult_path = file_output_dir / f"nn_multiplicity_{codec}_{rl}.png"
        fig.savefig(mult_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Generate cross-rate plots for this file
    if len(results) >= 2:
        _generate_cross_rate_plots(results, stem, codec, file_output_dir)

    return results


def _generate_cross_rate_plots(
    results: list[dict], stem: str, codec: str, output_dir: Path
) -> None:
    """Generate cross-rate summary plots for a single sequence."""
    rate_labels = [r["rate_label"] for r in results]
    flat_psnrs = [r["flat_psnr"] for r in results]
    medium_psnrs = [r["medium_psnr"] for r in results]
    edge_psnrs = [r["edge_psnr"] for r in results]
    degradations = [r["degradation"] for r in results]
    bpps = [r["bpp"] for r in results]

    # Stratified PSNR bar chart (forward direction)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_stratified_psnr_bars(
        rate_labels,
        flat_psnrs,
        medium_psnrs,
        edge_psnrs,
        ax=ax,
        title=f"Stratified PSNR (fwd) — {stem} [{codec}]",
    )
    bar_path = output_dir / f"stratified_psnr_bars_{codec}.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {bar_path}")

    # Reverse-direction stratified PSNR bar chart
    rev_flat = [r["rev_flat_psnr"] for r in results]
    rev_medium = [r["rev_medium_psnr"] for r in results]
    rev_edge = [r["rev_edge_psnr"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_stratified_psnr_bars(
        rate_labels,
        rev_flat,
        rev_medium,
        rev_edge,
        ax=ax,
        title=f"Stratified PSNR (rev) — {stem} [{codec}]",
    )
    bar_path = output_dir / f"stratified_psnr_bars_rev_{codec}.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {bar_path}")

    # Degradation vs rate — all three directions
    rev_degradations = [r["rev_degradation"] for r in results]
    fwd_self_degradations = [r["fwd_self_degradation"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        bpps,
        degradations,
        "o-",
        color="firebrick",
        linewidth=2,
        markersize=8,
        label="Forward (NN-assigned)",
    )
    ax.plot(
        bpps,
        rev_degradations,
        "s--",
        color="steelblue",
        linewidth=2,
        markersize=8,
        label="Reverse (orig curv)",
    )
    ax.plot(
        bpps,
        fwd_self_degradations,
        "D:",
        color="darkorange",
        linewidth=2,
        markersize=8,
        label="Forward-self (rec curv)",
    )
    for label, bpp, rdeg in zip(rate_labels, bpps, rev_degradations):
        ax.annotate(
            label,
            (bpp, rdeg),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=8,
            color="steelblue",
        )
    ax.set_xlabel("Bits Per Point (BPP)")
    ax.set_ylabel("Degradation (flat PSNR - edge PSNR) [dB]")
    ax.set_title(f"Oversmoothing Degradation — {stem} [{codec}]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    deg_path = output_dir / f"degradation_vs_rate_{codec}.png"
    fig.savefig(deg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {deg_path}")

    # NN multiplicity vs rate
    mult_flat = [r["mult_flat_mean"] for r in results]
    mult_edge = [r["mult_edge_mean"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        bpps,
        mult_flat,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=8,
        label="Flat",
    )
    ax.plot(
        bpps,
        mult_edge,
        "s-",
        color="firebrick",
        linewidth=2,
        markersize=8,
        label="Edge",
    )
    for label, bpp, mf, me in zip(rate_labels, bpps, mult_flat, mult_edge):
        ax.annotate(
            label, (bpp, me), textcoords="offset points", xytext=(5, 8), fontsize=8
        )
    ax.set_xlabel("Bits Per Point (BPP)")
    ax.set_ylabel("Mean NN Multiplicity")
    ax.set_title(f"NN Multiplicity vs Rate — {stem} [{codec}]")
    ax.legend()
    ax.grid(alpha=0.3)
    mult_path = output_dir / f"multiplicity_vs_rate_{codec}.png"
    fig.savefig(mult_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {mult_path}")

    # Spearman correlation vs rate
    rhos = [r["spearman_rho"] for r in results]
    pvals = [r["spearman_pvalue"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_correlation_vs_rate(
        rate_labels,
        rhos,
        pvals,
        bpps,
        ax=ax,
        title=f"Spearman Correlation — {stem} [{codec}]",
    )
    corr_path = output_dir / f"correlation_vs_rate_{codec}.png"
    fig.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {corr_path}")

    # Assignment accuracy vs rate
    if "assignment_accuracy" in results[0]:
        accuracies = [r["assignment_accuracy"] for r in results]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bpps, accuracies, "o-", color="forestgreen", linewidth=2, markersize=8)
        for label, bpp, acc in zip(rate_labels, bpps, accuracies):
            ax.annotate(
                f"{label}",
                (bpp, acc),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=9,
            )
        ax.set_xlabel("Bits Per Point (BPP)")
        ax.set_ylabel("Assignment Accuracy")
        ax.set_title(f"NN Assignment Accuracy — {stem} [{codec}]")
        ax.grid(alpha=0.3)
        ax.axhline(
            y=0.8, color="red", linestyle="--", alpha=0.5, label="Reliability threshold"
        )
        ax.set_ylim(0, 1.05)
        ax.legend()
        acc_path = output_dir / f"assignment_accuracy_{codec}.png"
        fig.savefig(acc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {acc_path}")


def _generate_cross_sequence_summary(all_results: list[dict], output_dir: Path) -> None:
    """Generate a cross-sequence comparison plot."""
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Group by (codec, sequence)
    codecs = sorted(set(r["codec"] for r in all_results))

    for codec in codecs:
        codec_results = [r for r in all_results if r["codec"] == codec]
        sequences = {}
        for r in codec_results:
            seq = r["sequence"]
            if seq not in sequences:
                sequences[seq] = []
            sequences[seq].append(r)

        if len(sequences) < 2:
            continue

        # Cross-sequence reverse-direction degradation
        fig, ax = plt.subplots(figsize=(10, 6))
        for seq_name, seq_results in sequences.items():
            seq_sorted = sorted(seq_results, key=lambda x: x["bpp"])
            bpps = [r["bpp"] for r in seq_sorted]
            rev_degs = [r["rev_degradation"] for r in seq_sorted]
            ax.plot(bpps, rev_degs, "o-", label=seq_name, linewidth=2, markersize=6)
        ax.set_xlabel("Bits Per Point (BPP)")
        ax.set_ylabel("Degradation (flat PSNR - edge PSNR) [dB]")
        ax.set_title(f"Oversmoothing Degradation (rev) — All Sequences [{codec}]")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        path = summary_dir / f"cross_sequence_summary_rev_{codec}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved reverse degradation summary: {path}")

        # Cross-sequence Spearman correlation
        fig, ax = plt.subplots(figsize=(10, 6))
        for seq_name, seq_results in sequences.items():
            seq_sorted = sorted(seq_results, key=lambda x: x["bpp"])
            bpps = [r["bpp"] for r in seq_sorted]
            rhos = [r["spearman_rho"] for r in seq_sorted]
            ax.plot(bpps, rhos, "o-", label=seq_name, linewidth=2, markersize=6)
        ax.set_xlabel("Bits Per Point (BPP)")
        ax.set_ylabel("Spearman rho (curvature vs error)")
        ax.set_title(f"Curvature-Error Correlation — All Sequences [{codec}]")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        path = summary_dir / f"cross_sequence_correlation_{codec}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved correlation summary: {path}")

        # Combined degradation vs rate (both directions)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for seq_name, seq_results in sequences.items():
            seq_sorted = sorted(seq_results, key=lambda x: x["bpp"])
            bpps = [r["bpp"] for r in seq_sorted]
            fwd_degs = [r["degradation"] for r in seq_sorted]
            rev_degs = [r["rev_degradation"] for r in seq_sorted]
            ax1.plot(bpps, fwd_degs, "o-", label=seq_name, linewidth=2, markersize=6)
            ax2.plot(bpps, rev_degs, "o-", label=seq_name, linewidth=2, markersize=6)
        for ax, title in [(ax1, "Forward (rec->orig)"), (ax2, "Reverse (orig->rec)")]:
            ax.set_xlabel("Bits Per Point (BPP)")
            ax.set_ylabel("Degradation [dB]")
            ax.set_title(f"Degradation — {title} [{codec}]")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = summary_dir / f"degradation_vs_rate_{codec}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined degradation plot: {path}")


def _save_aggregated_results(all_results: list[dict], output_dir: Path) -> None:
    """Save aggregated results as parquet, JSON, and CSV."""
    # Separate PSNR curves (variable-length) from scalar results
    psnr_curves = {}
    scalar_results = []
    for r in all_results:
        key = f"{r['codec']}_{r['sequence']}_{r['rate_label']}"
        psnr_curves[key] = {
            "bin_centers": r.get("psnr_curve_centers", []),
            "psnrs": r.get("psnr_curve_psnrs", []),
            "counts": r.get("psnr_curve_counts", []),
        }
        # Build scalar-only dict
        sr = {}
        skip_keys = {
            "psnr_curve_centers",
            "psnr_curve_psnrs",
            "psnr_curve_counts",
            "point_counts",
        }
        for k, v in r.items():
            if k in skip_keys:
                continue
            if isinstance(v, (np.integer, np.floating)):
                sr[k] = float(v)
            elif isinstance(v, dict):
                sr[k] = {
                    kk: int(vv) if isinstance(vv, np.integer) else vv
                    for kk, vv in v.items()
                }
            else:
                sr[k] = v
        scalar_results.append(sr)

    # Save PSNR curves separately
    curves_path = output_dir / "psnr_curves.json"
    with open(curves_path, "w") as f:
        json.dump(psnr_curves, f, indent=2)
    print(f"\nSaved PSNR curves: {curves_path}")

    # Save JSON
    json_path = output_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(scalar_results, f, indent=2)
    print(f"Saved JSON results: {json_path}")

    # Build DataFrame from scalar results
    df = pd.DataFrame(scalar_results)

    # Save parquet
    parquet_path = output_dir / "all_results.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet results: {parquet_path}")

    # Save CSV
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results: {csv_path}")

    return df


def _build_rate_specs(codec: str, rates: list[str]) -> list[dict]:
    """Build rate spec dicts from codec type and rate arguments."""
    if codec == "pcgcv2":
        return [{"rate_label": r, "rate": r, "coding_scale": None} for r in rates]
    else:
        return [
            {
                "rate_label": f"qs{qs}",
                "rate": None,
                "coding_scale": qs,
            }
            for qs in GPCC_CODING_SCALES
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Oversmoothing analysis for point cloud codecs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to a single PLY file.",
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory of PLY files (processes all .ply files found).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="pcgcv2",
        choices=["pcgcv2", "gpcc"],
        help="Codec to use (default: pcgcv2).",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        default=ALL_RATES,
        choices=ALL_RATES,
        help="PCGCv2 rate points (default: all r1-r7). Ignored for G-PCC.",
    )
    parser.add_argument(
        "--gpcc-scales",
        nargs="+",
        type=float,
        default=None,
        help="G-PCC coding scales (default: built-in 7 values). "
        "Only used with --codec gpcc.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/oversmoothing",
        help="Output directory for results and figures.",
    )
    parser.add_argument(
        "--n-strata",
        type=int,
        default=3,
        choices=[3, 5],
        help="Number of curvature strata (3=terciles, 5=quintiles).",
    )
    parser.add_argument(
        "--curvature-k",
        type=int,
        default=30,
        help="Number of neighbors for PCA curvature estimation.",
    )
    parser.add_argument(
        "--random-control",
        action="store_true",
        help="Run random stratification sanity check (100 iterations per rate).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override G-PCC scales if provided
    if args.gpcc_scales is not None and args.codec == "gpcc":
        global GPCC_CODING_SCALES
        GPCC_CODING_SCALES = sorted(args.gpcc_scales)

    # Collect input files
    if args.input:
        input_files = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        input_files = sorted(input_dir.glob("*.ply"))
        if not input_files:
            print(f"No .ply files found in {input_dir}")
            sys.exit(1)

    # Build rate specs
    rate_specs = _build_rate_specs(args.codec, args.rates)

    print(f"Codec:       {args.codec}")
    print(f"Input files: {len(input_files)}")
    print(f"Rate points: {[rs['rate_label'] for rs in rate_specs]}")
    print(f"Output dir:  {output_dir}")
    print(f"Strata:      {args.n_strata}")

    all_results = []

    for input_path in input_files:
        file_results = analyze_single_file(
            input_path=input_path,
            codec=args.codec,
            rate_specs=rate_specs,
            output_dir=output_dir,
            n_strata=args.n_strata,
            curvature_k=args.curvature_k,
            random_control=args.random_control,
        )
        all_results.extend(file_results)

    if not all_results:
        print("\nNo results generated (all rate points failed?).")
        sys.exit(1)

    # Save aggregated results
    df = _save_aggregated_results(all_results, output_dir)

    # Cross-sequence summary (if multiple files)
    if len(input_files) > 1:
        _generate_cross_sequence_summary(all_results, output_dir)

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    summary_cols = [
        "codec",
        "sequence",
        "rate_label",
        "bpp",
        "overall_psnr",
        "degradation",
        "rev_degradation",
        "fwd_self_degradation",
        "spearman_rho",
        "mult_flat_mean",
        "mult_edge_mean",
        "assignment_accuracy",
    ]
    available_cols = [c for c in summary_cols if c in df.columns]
    print(df[available_cols].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
