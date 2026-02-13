"""Oversmoothing analysis for point cloud codecs across rate points.

Runs PCGCv2 or G-PCC at multiple rate points on one or more point clouds,
computes curvature-stratified quality metrics, and generates publication-quality
visualizations that reveal spatial patterns of oversmoothing.

Decoded clouds are cached as .npy files so expensive codec runs only happen
once.  Re-running the script reuses cached clouds and recomputes metrics.

Execution is split into three phases:
  Phase 1 (sequential): cache decoded clouds  — GPU-bound codec runs
  Phase 2 (parallel):   compute all metrics   — CPU-bound, ProcessPoolExecutor
  Phase 3 (sequential): generate plots        — matplotlib is not process-safe

Usage:
    # Quick single-frame test (PCGCv2, default)
    python scripts/analyze_oversmoothing.py \
        --input datasets/8iVFB_small/longdress_vox10_1300.ply \
        --rates r1 r3 r7 \
        --output-dir results/oversmoothing

    # All rates, all 4 8iVFB sequences, 8 workers
    python scripts/analyze_oversmoothing.py \
        --input-dir datasets/8iVFB_small/ \
        --rates r1 r2 r3 r4 r5 r6 r7 \
        --output-dir results/oversmoothing \
        --random-control --workers 8

    # G-PCC baseline at multiple quantization scales
    python scripts/analyze_oversmoothing.py \
        --input-dir datasets/8iVFB_small/ \
        --codec gpcc \
        --output-dir results/oversmoothing --workers 8
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from pcml.visualization.oversmoothing import (
    create_oversmoothing_report,
    plot_correlation_vs_rate,
    plot_nn_multiplicity,
    plot_psnr_vs_curvature_curve,
    plot_stratified_psnr_bars,
)

logger = logging.getLogger(__name__)

ALL_RATES = ["r1", "r2", "r3", "r4", "r5", "r6", "r7"]

# Default G-PCC quantization scales (geometrically spaced)
GPCC_CODING_SCALES = [0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]

# 10-bit voxelized clouds use peak = 1023
PEAK = 1023.0


# ──────────────────────────────────────────────────────────────────────
# Phase 1 helpers: codec caching (sequential, GPU-bound)
# ──────────────────────────────────────────────────────────────────────


def _cache_stem(codec: str, rate_label: str) -> str:
    return f"{codec}_{rate_label}"


def _load_or_run_codec(
    codec: str,
    geometry: np.ndarray,
    rate_label: str,
    decoded_dir: Path,
    rate: str | None = None,
    coding_scale: float | None = None,
) -> tuple[np.ndarray, int, float]:
    """Load decoded cloud from cache, or run codec and cache it."""
    stem = _cache_stem(codec, rate_label)
    cloud_path = decoded_dir / f"{stem}.npy"
    meta_path = decoded_dir / f"{stem}_meta.json"

    if cloud_path.exists() and meta_path.exists():
        decoded_geometry = np.load(cloud_path)
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"    [cache hit] {cloud_path.name}")
        return (
            decoded_geometry,
            meta["compressed_size_bytes"],
            meta["codec_time_s"],
        )

    # Run codec
    if codec == "pcgcv2":
        adapter = PCGCv2Adapter(rate_point=rate)
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

    print(f"    [cached] {cloud_path.name}")
    return decoded_geometry, comp_result.compressed_size_bytes, codec_time


def cache_all_clouds(
    input_files: list[Path],
    codec: str,
    rate_specs: list[dict],
    output_dir: Path,
    curvature_k: int,
) -> None:
    """Phase 1: ensure all decoded clouds + curvature are cached."""
    print("\n" + "=" * 60)
    print("PHASE 1: Caching decoded clouds (sequential, GPU-bound)")
    print("=" * 60)
    loader = PLYPointCloudLoader()

    for input_path in input_files:
        stem = input_path.stem
        decoded_dir = output_dir / "decoded_clouds" / stem
        decoded_dir.mkdir(parents=True, exist_ok=True)

        pc = loader.load(input_path)
        geometry = pc.geometry

        # Cache original
        orig_cache = decoded_dir / "original.npy"
        if not orig_cache.exists():
            np.save(orig_cache, geometry)

        # Cache curvature
        curv_cache = decoded_dir / f"curvature_k{curvature_k}.npy"
        if not curv_cache.exists():
            print(f"  [{stem}] Computing curvature (k={curvature_k})...")
            t0 = time.time()
            curvature = CurvatureQualityCalculator.compute_curvature(
                geometry, k=curvature_k
            )
            np.save(curv_cache, curvature)
            print(f"  [{stem}] Curvature done in {time.time() - t0:.1f}s")
        else:
            print(f"  [{stem}] Curvature cached")

        # Cache all decoded clouds
        print(f"  [{stem}] Caching decoded clouds...")
        for rs in rate_specs:
            rl = rs["rate_label"]
            try:
                _load_or_run_codec(
                    codec=codec,
                    geometry=geometry,
                    rate_label=rl,
                    decoded_dir=decoded_dir,
                    rate=rs.get("rate"),
                    coding_scale=rs.get("coding_scale"),
                )
            except Exception as e:
                print(f"    ERROR [{stem}/{rl}]: {e}")

    print("Phase 1 complete.\n")


# ──────────────────────────────────────────────────────────────────────
# Phase 2 helpers: metric computation (parallel, CPU-bound)
# ──────────────────────────────────────────────────────────────────────


def _compute_metrics_for_rate(
    decoded_dir: Path,
    codec: str,
    rate_label: str,
    coding_scale: float | None,
    curvature_k: int,
    n_strata: int,
    random_control: bool,
) -> dict | None:
    """Compute all metrics for one (sequence, rate) pair.

    Loads everything from cached .npy files — safe for ProcessPoolExecutor.
    Returns a result dict, or None on failure.
    """
    stem = decoded_dir.name
    cloud_stem = _cache_stem(codec, rate_label)
    cloud_path = decoded_dir / f"{cloud_stem}.npy"
    meta_path = decoded_dir / f"{cloud_stem}_meta.json"
    orig_path = decoded_dir / "original.npy"
    curv_path = decoded_dir / f"curvature_k{curvature_k}.npy"

    if not cloud_path.exists() or not orig_path.exists():
        return None

    geometry = np.load(orig_path)
    decoded_geometry = np.load(cloud_path)
    curvature = np.load(curv_path)

    with open(meta_path) as f:
        meta = json.load(f)

    compressed_bytes = meta["compressed_size_bytes"]
    codec_time = meta["codec_time_s"]
    num_points_input = len(geometry)
    num_points_output = len(decoded_geometry)
    bpp = (compressed_bytes * 8) / num_points_input

    pid = os.getpid()
    tag = f"[{pid}] {stem}/{rate_label}"
    print(f"  {tag}: computing metrics (BPP={bpp:.4f})...")
    t0 = time.time()

    # Forward stratified PSNR
    fwd_metrics = CurvatureQualityCalculator.compute_stratified_psnr(
        geometry,
        decoded_geometry,
        curvature,
        peak=PEAK,
        n_strata=n_strata,
        direction="forward",
        k=curvature_k,
    )

    # Reverse stratified PSNR
    rev_metrics = CurvatureQualityCalculator.compute_stratified_psnr(
        geometry,
        decoded_geometry,
        curvature,
        peak=PEAK,
        n_strata=n_strata,
        direction="reverse",
        k=curvature_k,
    )

    # Forward-self
    fwd_self_metrics = CurvatureQualityCalculator.compute_stratified_psnr(
        geometry,
        decoded_geometry,
        curvature,
        peak=PEAK,
        n_strata=n_strata,
        direction="forward_self",
        k=curvature_k,
    )

    # Spearman correlation
    corr = CurvatureQualityCalculator.compute_curvature_error_correlation(
        geometry, decoded_geometry, curvature
    )

    # NN multiplicity
    mult = CurvatureQualityCalculator.compute_nn_multiplicity(
        geometry, decoded_geometry, curvature, n_strata=n_strata
    )

    # Continuous PSNR curve
    psnr_curve = CurvatureQualityCalculator.compute_psnr_vs_curvature_curve(
        geometry,
        decoded_geometry,
        curvature,
        peak=PEAK,
        n_bins=20,
        direction="reverse",
        k=curvature_k,
    )

    # Overall PSNR
    dists, _ = CurvatureQualityCalculator.compute_per_point_error(
        geometry, decoded_geometry
    )
    overall_mse = float(np.mean(dists**2))
    overall_psnr = (
        10.0 * np.log10(PEAK**2 / overall_mse) if overall_mse > 0 else float("inf")
    )

    result = {
        "codec": codec,
        "sequence": stem,
        "rate_label": rate_label,
        "coding_scale": coding_scale,
        "bpp": bpp,
        "overall_psnr": overall_psnr,
        "flat_psnr": fwd_metrics.flat_psnr,
        "medium_psnr": fwd_metrics.medium_psnr,
        "edge_psnr": fwd_metrics.edge_psnr,
        "degradation": fwd_metrics.degradation,
        "curvature_kl": fwd_metrics.curvature_kl,
        "rev_flat_psnr": rev_metrics.flat_psnr,
        "rev_medium_psnr": rev_metrics.medium_psnr,
        "rev_edge_psnr": rev_metrics.edge_psnr,
        "rev_degradation": rev_metrics.degradation,
        "fwd_self_flat_psnr": fwd_self_metrics.flat_psnr,
        "fwd_self_medium_psnr": fwd_self_metrics.medium_psnr,
        "fwd_self_edge_psnr": fwd_self_metrics.edge_psnr,
        "fwd_self_degradation": fwd_self_metrics.degradation,
        "spearman_rho": corr.rho,
        "spearman_pvalue": corr.p_value,
        "mult_flat_mean": mult.per_stratum_mean.get("flat", 0.0),
        "mult_medium_mean": mult.per_stratum_mean.get("medium", 0.0),
        "mult_edge_mean": mult.per_stratum_mean.get("edge", 0.0),
        "mult_overall_mean": mult.overall_mean,
        "psnr_curve_centers": psnr_curve.bin_centers.tolist(),
        "psnr_curve_psnrs": psnr_curve.psnrs.tolist(),
        "psnr_curve_counts": psnr_curve.counts.tolist(),
        "num_points_input": num_points_input,
        "num_points_output": num_points_output,
        "compressed_bytes": compressed_bytes,
        "codec_time_s": codec_time,
        "curvature_k": curvature_k,
        "point_counts": fwd_metrics.point_counts,
    }

    # Assignment accuracy
    accuracy = CurvatureQualityCalculator.compute_assignment_accuracy(
        geometry, decoded_geometry, curvature, k=curvature_k
    )
    result["assignment_accuracy"] = accuracy

    # Random control
    if random_control:
        rand_result = CurvatureQualityCalculator.compute_random_stratified_psnr(
            geometry,
            decoded_geometry,
            curvature,
            peak=PEAK,
            n_strata=n_strata,
            direction="reverse",
        )
        result["random_mean_degradation"] = rand_result["mean_degradation"]
        result["random_std_degradation"] = rand_result["std_degradation"]

    elapsed = time.time() - t0
    print(
        f"  {tag}: done in {elapsed:.1f}s "
        f"(rev_deg={rev_metrics.degradation:.2f} dB, "
        f"rho={corr.rho:.4f})"
    )
    return result


def compute_all_metrics_parallel(
    input_files: list[Path],
    codec: str,
    rate_specs: list[dict],
    output_dir: Path,
    curvature_k: int,
    n_strata: int,
    random_control: bool,
    max_workers: int,
) -> list[dict]:
    """Phase 2: compute metrics in parallel via ProcessPoolExecutor."""
    print("=" * 60)
    print(f"PHASE 2: Computing metrics " f"(parallel, {max_workers} workers)")
    print("=" * 60)

    # Build task list: (decoded_dir, rate_spec) for each combination
    tasks = []
    for input_path in input_files:
        stem = input_path.stem
        decoded_dir = output_dir / "decoded_clouds" / stem
        for rs in rate_specs:
            tasks.append((decoded_dir, rs))

    all_results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for decoded_dir, rs in tasks:
            future = executor.submit(
                _compute_metrics_for_rate,
                decoded_dir=decoded_dir,
                codec=codec,
                rate_label=rs["rate_label"],
                coding_scale=rs.get("coding_scale"),
                curvature_k=curvature_k,
                n_strata=n_strata,
                random_control=random_control,
            )
            futures[future] = (decoded_dir.name, rs["rate_label"])

        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
                else:
                    print(f"  WARN: {key[0]}/{key[1]} returned None")
            except Exception as e:
                print(f"  ERROR: {key[0]}/{key[1]}: {e}")

    # Sort by (sequence, bpp) for deterministic output
    all_results.sort(key=lambda r: (r["sequence"], r["bpp"]))

    elapsed = time.time() - t0
    print(f"Phase 2 complete: {len(all_results)} results in {elapsed:.1f}s\n")
    return all_results


# ──────────────────────────────────────────────────────────────────────
# Phase 3: plotting + aggregated output (sequential, main process)
# ──────────────────────────────────────────────────────────────────────


def generate_all_plots(
    all_results: list[dict],
    output_dir: Path,
    input_files: list[Path],
) -> None:
    """Phase 3: generate all plots from computed results."""
    print("=" * 60)
    print("PHASE 3: Generating plots")
    print("=" * 60)

    # Group results by (codec, sequence)
    by_seq: dict[str, list[dict]] = {}
    for r in all_results:
        seq = r["sequence"]
        if seq not in by_seq:
            by_seq[seq] = []
        by_seq[seq].append(r)

    # Per-sequence plots
    for stem, seq_results in by_seq.items():
        codec = seq_results[0]["codec"]
        file_output_dir = output_dir / stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        decoded_dir = output_dir / "decoded_clouds" / stem

        # Per-rate report plots
        orig_path = decoded_dir / "original.npy"
        curv_k = seq_results[0]["curvature_k"]
        curv_path = decoded_dir / f"curvature_k{curv_k}.npy"

        if orig_path.exists() and curv_path.exists():
            geometry = np.load(orig_path)
            curvature = np.load(curv_path)

            for r in seq_results:
                rl = r["rate_label"]
                cloud_stem = _cache_stem(codec, rl)
                dec_path = decoded_dir / f"{cloud_stem}.npy"
                if not dec_path.exists():
                    continue
                decoded_geometry = np.load(dec_path)

                # Full report
                report_path = file_output_dir / f"report_{codec}_{rl}.png"
                fig = create_oversmoothing_report(
                    geometry,
                    decoded_geometry,
                    curvature,
                    metrics=r,
                    rate_label=f"{stem} @ {rl} [{codec}]",
                    output_path=str(report_path),
                )
                plt.close(fig)

                # PSNR vs curvature curve
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_psnr_vs_curvature_curve(
                    r["psnr_curve_centers"],
                    r["psnr_curve_psnrs"],
                    r["psnr_curve_counts"],
                    ax=ax,
                    title=(f"PSNR vs Curvature (rev) " f"— {stem} @ {rl} [{codec}]"),
                )
                fig.savefig(
                    file_output_dir / f"psnr_vs_curvature_{codec}_{rl}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

                # NN multiplicity
                fig, ax = plt.subplots(figsize=(8, 5))
                mult_means = {
                    "flat": r["mult_flat_mean"],
                    "medium": r["mult_medium_mean"],
                    "edge": r["mult_edge_mean"],
                }
                plot_nn_multiplicity(
                    mult_means,
                    ax=ax,
                    title=(f"NN Multiplicity " f"— {stem} @ {rl} [{codec}]"),
                )
                fig.savefig(
                    file_output_dir / f"nn_multiplicity_{codec}_{rl}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

        # Cross-rate plots
        if len(seq_results) >= 2:
            _generate_cross_rate_plots(seq_results, stem, codec, file_output_dir)

    # Cross-sequence summary
    if len(by_seq) >= 2:
        _generate_cross_sequence_summary(all_results, output_dir)

    print("Phase 3 complete.\n")


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

    # Stratified PSNR bar chart (forward)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_stratified_psnr_bars(
        rate_labels,
        flat_psnrs,
        medium_psnrs,
        edge_psnrs,
        ax=ax,
        title=f"Stratified PSNR (fwd) — {stem} [{codec}]",
    )
    fig.savefig(
        output_dir / f"stratified_psnr_bars_{codec}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Stratified PSNR bar chart (reverse)
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
    fig.savefig(
        output_dir / f"stratified_psnr_bars_rev_{codec}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Degradation vs rate — three directions
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
    fig.savefig(
        output_dir / f"degradation_vs_rate_{codec}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

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
            label,
            (bpp, me),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=8,
        )
    ax.set_xlabel("Bits Per Point (BPP)")
    ax.set_ylabel("Mean NN Multiplicity")
    ax.set_title(f"NN Multiplicity vs Rate — {stem} [{codec}]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(
        output_dir / f"multiplicity_vs_rate_{codec}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

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
    fig.savefig(
        output_dir / f"correlation_vs_rate_{codec}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Assignment accuracy vs rate
    if "assignment_accuracy" in results[0]:
        accuracies = [r["assignment_accuracy"] for r in results]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            bpps,
            accuracies,
            "o-",
            color="forestgreen",
            linewidth=2,
            markersize=8,
        )
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
            y=0.8,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Reliability threshold",
        )
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.savefig(
            output_dir / f"assignment_accuracy_{codec}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def _generate_cross_sequence_summary(all_results: list[dict], output_dir: Path) -> None:
    """Generate cross-sequence comparison plots."""
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    codecs = sorted(set(r["codec"] for r in all_results))

    for codec in codecs:
        codec_results = [r for r in all_results if r["codec"] == codec]
        sequences: dict[str, list[dict]] = {}
        for r in codec_results:
            seq = r["sequence"]
            if seq not in sequences:
                sequences[seq] = []
            sequences[seq].append(r)

        if len(sequences) < 2:
            continue

        # Cross-sequence reverse degradation
        fig, ax = plt.subplots(figsize=(10, 6))
        for seq_name, seq_results in sequences.items():
            seq_sorted = sorted(seq_results, key=lambda x: x["bpp"])
            bpps = [r["bpp"] for r in seq_sorted]
            rev_degs = [r["rev_degradation"] for r in seq_sorted]
            ax.plot(
                bpps,
                rev_degs,
                "o-",
                label=seq_name,
                linewidth=2,
                markersize=6,
            )
        ax.set_xlabel("Bits Per Point (BPP)")
        ax.set_ylabel("Degradation (flat - edge PSNR) [dB]")
        ax.set_title(f"Oversmoothing (rev) — All Sequences [{codec}]")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        path = summary_dir / f"cross_sequence_summary_rev_{codec}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # Cross-sequence Spearman
        fig, ax = plt.subplots(figsize=(10, 6))
        for seq_name, seq_results in sequences.items():
            seq_sorted = sorted(seq_results, key=lambda x: x["bpp"])
            bpps = [r["bpp"] for r in seq_sorted]
            rhos = [r["spearman_rho"] for r in seq_sorted]
            ax.plot(
                bpps,
                rhos,
                "o-",
                label=seq_name,
                linewidth=2,
                markersize=6,
            )
        ax.set_xlabel("Bits Per Point (BPP)")
        ax.set_ylabel("Spearman rho (curvature vs error)")
        ax.set_title(f"Curvature-Error Correlation [{codec}]")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        path = summary_dir / f"cross_sequence_correlation_{codec}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # Combined degradation (both directions)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for seq_name, seq_results in sequences.items():
            seq_sorted = sorted(seq_results, key=lambda x: x["bpp"])
            bpps = [r["bpp"] for r in seq_sorted]
            fwd_d = [r["degradation"] for r in seq_sorted]
            rev_d = [r["rev_degradation"] for r in seq_sorted]
            ax1.plot(
                bpps,
                fwd_d,
                "o-",
                label=seq_name,
                linewidth=2,
                markersize=6,
            )
            ax2.plot(
                bpps,
                rev_d,
                "o-",
                label=seq_name,
                linewidth=2,
                markersize=6,
            )
        for ax, title in [
            (ax1, "Forward (rec->orig)"),
            (ax2, "Reverse (orig->rec)"),
        ]:
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
        print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# Aggregated output
# ──────────────────────────────────────────────────────────────────────


def _save_aggregated_results(all_results: list[dict], output_dir: Path) -> pd.DataFrame:
    """Save aggregated results as parquet, JSON, and CSV."""
    psnr_curves = {}
    scalar_results = []
    skip_keys = {
        "psnr_curve_centers",
        "psnr_curve_psnrs",
        "psnr_curve_counts",
        "point_counts",
    }
    for r in all_results:
        key = f"{r['codec']}_{r['sequence']}_{r['rate_label']}"
        psnr_curves[key] = {
            "bin_centers": r.get("psnr_curve_centers", []),
            "psnrs": r.get("psnr_curve_psnrs", []),
            "counts": r.get("psnr_curve_counts", []),
        }
        sr = {}
        for k, v in r.items():
            if k in skip_keys:
                continue
            if isinstance(v, (np.integer, np.floating)):
                sr[k] = float(v)
            elif isinstance(v, dict):
                sr[k] = {
                    kk: (int(vv) if isinstance(vv, np.integer) else vv)
                    for kk, vv in v.items()
                }
            else:
                sr[k] = v
        scalar_results.append(sr)

    # PSNR curves
    curves_path = output_dir / "psnr_curves.json"
    with open(curves_path, "w") as f:
        json.dump(psnr_curves, f, indent=2)
    print(f"Saved PSNR curves: {curves_path}")

    # JSON
    json_path = output_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(scalar_results, f, indent=2)
    print(f"Saved JSON results: {json_path}")

    # DataFrame
    df = pd.DataFrame(scalar_results)

    # Parquet
    parquet_path = output_dir / "all_results.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet results: {parquet_path}")

    # CSV
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results: {csv_path}")

    return df


# ──────────────────────────────────────────────────────────────────────
# Rate spec builder
# ──────────────────────────────────────────────────────────────────────


def _build_rate_specs(codec: str, rates: list[str]) -> list[dict]:
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


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


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
        help="Directory of PLY files.",
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
        help="PCGCv2 rate points. Ignored for G-PCC.",
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
    )
    parser.add_argument(
        "--n-strata",
        type=int,
        default=3,
        choices=[3, 5],
    )
    parser.add_argument(
        "--curvature-k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--random-control",
        action="store_true",
        help="Run random stratification sanity check.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for metric computation " "(default: 4).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    rate_specs = _build_rate_specs(args.codec, args.rates)
    n_tasks = len(input_files) * len(rate_specs)
    max_workers = min(args.workers, n_tasks)

    print(f"Codec:       {args.codec}")
    print(f"Input files: {len(input_files)}")
    print(f"Rate points: {[rs['rate_label'] for rs in rate_specs]}")
    print(f"Total tasks: {n_tasks}")
    print(f"Workers:     {max_workers}")
    print(f"Output dir:  {output_dir}")

    # Phase 1: cache decoded clouds (sequential)
    cache_all_clouds(input_files, args.codec, rate_specs, output_dir, args.curvature_k)

    # Phase 2: compute metrics (parallel)
    all_results = compute_all_metrics_parallel(
        input_files=input_files,
        codec=args.codec,
        rate_specs=rate_specs,
        output_dir=output_dir,
        curvature_k=args.curvature_k,
        n_strata=args.n_strata,
        random_control=args.random_control,
        max_workers=max_workers,
    )

    if not all_results:
        print("\nNo results generated.")
        sys.exit(1)

    # Save aggregated results
    df = _save_aggregated_results(all_results, output_dir)

    # Phase 3: generate plots (sequential)
    generate_all_plots(all_results, output_dir, input_files)

    # Summary table
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
    available = [c for c in summary_cols if c in df.columns]
    print(df[available].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
