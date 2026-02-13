"""Benchmark comparison of all compression methods.

Compares G-PCC, Simple Baseline, pcc_geo_cnn_v2, and PCGCv2 on the same
point clouds. Includes curvature-stratified quality metrics for
oversmoothing detection.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import PLYPointCloudLoader
from pcml.frameworks.gpcc import GPCCAdapter
from pcml.frameworks.pcc_geo_cnn_v2 import PCCGeoCNNv2Adapter
from pcml.frameworks.pcgcv2 import PCGCv2Adapter
from pcml.metrics.curvature import CurvatureQualityCalculator
from pcml.metrics.quality import GeometryQualityCalculator


def benchmark_gpcc(geometry, colors, config_name="lossless"):
    """Benchmark G-PCC compression."""
    print(f"  Testing G-PCC ({config_name})...")
    # G-PCC uses default lossless config (no config_path needed)
    adapter = GPCCAdapter()

    start = time.time()
    compressed, result = adapter.compress(geometry, colors)
    comp_time = time.time() - start

    recon_geom, recon_colors = adapter.decompress(compressed)
    metrics = GeometryQualityCalculator.calculate_all(geometry, recon_geom)

    bpp = (len(compressed) * 8) / len(geometry)

    return {
        "method": "G-PCC",
        "config": config_name,
        "input_points": len(geometry),
        "output_points": len(recon_geom),
        "compressed_bytes": len(compressed),
        "bpp": bpp,
        "psnr": metrics.psnr,
        "mse": metrics.mse,
        "rmse": metrics.rmse,
        "d1": metrics.d1,
        "d2_sym": metrics.d2_sym,
        "compression_time": comp_time,
    }


def benchmark_pcc_geo_cnn(geometry, colors, model_name="c1"):
    """Benchmark pcc_geo_cnn_v2 compression."""
    print(f"  Testing pcc_geo_cnn_v2 ({model_name})...")
    adapter = PCCGeoCNNv2Adapter(model_name=model_name)

    start = time.time()
    compressed, result = adapter.compress(geometry, colors)
    comp_time = time.time() - start

    recon_geom, recon_colors = adapter.decompress(compressed)
    metrics = GeometryQualityCalculator.calculate_all(geometry, recon_geom)

    bpp = (result.compressed_size_bytes * 8) / len(geometry)

    return {
        "method": "pcc_geo_cnn_v2",
        "config": model_name,
        "input_points": len(geometry),
        "output_points": len(recon_geom),
        "compressed_bytes": result.compressed_size_bytes,
        "bpp": bpp,
        "psnr": metrics.psnr,
        "mse": metrics.mse,
        "rmse": metrics.rmse,
        "d1": metrics.d1,
        "d2_sym": metrics.d2_sym,
        "compression_time": comp_time,
    }


def benchmark_pcgcv2(
    geometry, colors, rate_point="r3", resolution=1024, compute_curvature=True
):
    """Benchmark PCGCv2 compression with optional curvature-stratified metrics."""
    print(f"  Testing PCGCv2 ({rate_point})...")
    adapter = PCGCv2Adapter(rate_point=rate_point, resolution=resolution)

    start = time.time()
    compressed, result = adapter.compress(geometry, colors)
    recon_geom, _ = adapter.decompress(compressed)
    comp_time = time.time() - start

    # Standard MPEG metrics with peak = resolution - 1
    peak = resolution - 1
    metrics = GeometryQualityCalculator.calculate_all(geometry, recon_geom, peak=peak)
    bpp = (result.compressed_size_bytes * 8) / len(geometry)

    row = {
        "method": "PCGCv2",
        "config": rate_point,
        "input_points": len(geometry),
        "output_points": len(recon_geom),
        "compressed_bytes": result.compressed_size_bytes,
        "bpp": bpp,
        "psnr": metrics.psnr,
        "mse": metrics.mse,
        "rmse": metrics.rmse,
        "d1": metrics.d1,
        "d2_sym": metrics.d2_sym,
        "compression_time": comp_time,
    }

    # Curvature-stratified metrics for oversmoothing detection
    if compute_curvature:
        try:
            strat = CurvatureQualityCalculator.calculate_all(
                geometry, recon_geom, peak=peak
            )
            row.update(
                {
                    "flat_psnr": strat.flat_psnr,
                    "medium_psnr": strat.medium_psnr,
                    "edge_psnr": strat.edge_psnr,
                    "degradation": strat.degradation,
                    "curvature_kl": strat.curvature_kl,
                }
            )
        except Exception as e:
            print(f"    Warning: curvature metrics failed: {e}")

    return row


def main():
    parser = argparse.ArgumentParser(description="Benchmark all compression methods")
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/8iVFB_small/longdress_vox10_1300.ply",
        help="Input PLY file",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Number of points to test (subsamples if needed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpcc-configs",
        nargs="+",
        default=["lossless"],
        help="G-PCC configurations to test (currently only lossless)",
    )
    parser.add_argument(
        "--cnn-models",
        nargs="+",
        default=["c1"],
        help="pcc_geo_cnn_v2 models to test",
    )
    parser.add_argument(
        "--pcgcv2-rates",
        nargs="+",
        default=[],
        help="PCGCv2 rate points to test (r1-r7)",
    )
    parser.add_argument(
        "--pcgcv2-resolution",
        type=int,
        default=1024,
        help="Resolution for PCGCv2 (default: 1024 for 10-bit)",
    )
    parser.add_argument(
        "--skip-gpcc",
        action="store_true",
        help="Skip G-PCC tests",
    )
    parser.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Skip pcc_geo_cnn_v2 tests",
    )
    parser.add_argument(
        "--skip-pcgcv2",
        action="store_true",
        help="Skip PCGCv2 tests",
    )
    parser.add_argument(
        "--no-curvature",
        action="store_true",
        help="Skip curvature-stratified metrics (faster)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Benchmark: All Compression Methods")
    print("=" * 70)
    print()

    # Load point cloud
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    print(f"Loading: {input_path.name}")
    loader = PLYPointCloudLoader(verbose=False)
    pc = loader.load(str(input_path))

    # Subsample if needed
    if pc.num_points > args.num_points:
        print(f"Subsampling {pc.num_points:,} → {args.num_points:,} points")
        indices = np.random.choice(pc.num_points, size=args.num_points, replace=False)
        geometry = pc.geometry[indices]
        colors = pc.colors[indices] if pc.colors is not None else None
    else:
        geometry = pc.geometry
        colors = pc.colors

    print(f"Testing with {len(geometry):,} points")
    print()

    # Run benchmarks
    results = []

    # G-PCC
    if not args.skip_gpcc:
        print("Testing G-PCC configurations:")
        for config in tqdm(args.gpcc_configs, desc="G-PCC", unit="config"):
            try:
                result = benchmark_gpcc(geometry, colors, config)
                results.append(result)
                tqdm.write(
                    f"    ✓ {config}: {result['bpp']:.2f} BPP, {result['psnr']:.2f} dB"
                )
            except Exception as e:
                tqdm.write(f"    ✗ {config}: {e}")
        print()

    # pcc_geo_cnn_v2
    if not args.skip_cnn:
        print("Testing pcc_geo_cnn_v2 models:")
        for model in tqdm(args.cnn_models, desc="pcc_geo_cnn_v2", unit="model"):
            try:
                result = benchmark_pcc_geo_cnn(geometry, colors, model)
                results.append(result)
                tqdm.write(
                    f"    ✓ {model}: {result['bpp']:.2f} BPP, "
                    f"{result['psnr']:.2f} dB ({result['compression_time']:.1f}s)"
                )
            except Exception as e:
                tqdm.write(f"    ✗ {model}: {e}")
        print()

    # PCGCv2
    if not args.skip_pcgcv2 and args.pcgcv2_rates:
        print("Testing PCGCv2 rate points:")
        for rate in tqdm(args.pcgcv2_rates, desc="PCGCv2", unit="rate"):
            try:
                result = benchmark_pcgcv2(
                    geometry,
                    colors,
                    rate_point=rate,
                    resolution=args.pcgcv2_resolution,
                    compute_curvature=not args.no_curvature,
                )
                results.append(result)
                msg = (
                    f"    ✓ {rate}: {result['bpp']:.3f} BPP, "
                    f"{result['psnr']:.2f} dB ({result['compression_time']:.1f}s)"
                )
                if "degradation" in result:
                    msg += f" [degradation: {result['degradation']:.2f} dB]"
                tqdm.write(msg)
            except Exception as e:
                tqdm.write(f"    ✗ {rate}: {e}")
        print()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save as JSON
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "input_file": str(input_path),
                "num_points": len(geometry),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved JSON: {json_path}")

    # Print summary table
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(
        df[["method", "config", "bpp", "psnr", "compression_time"]].to_string(
            index=False
        )
    )
    print()


if __name__ == "__main__":
    main()
