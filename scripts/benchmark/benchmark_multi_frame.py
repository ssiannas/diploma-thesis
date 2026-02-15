"""Multi-frame benchmark for comprehensive evaluation.

Tests multiple random frames from each sequence to get statistically robust results.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.adapters.gpcc import GPCCAdapter
from pcml.adapters.pcc_geo_cnn_v2 import PCCGeoCNNv2Adapter
from pcml.data.loaders import PLYPointCloudLoader
from pcml.metrics.quality import GeometryQualityCalculator


def get_sequence_frames(sequence_dir, num_frames=10, seed=42):
    """Get random frame indices from a sequence.

    Args:
        sequence_dir: Path to sequence directory on external drive
        num_frames: Number of frames to randomly select
        seed: Random seed for reproducibility

    Returns:
        List of PLY file paths
    """
    sequence_path = Path(sequence_dir)
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    # Get all PLY files
    ply_files = sorted(list(sequence_path.glob("*.ply")))

    if not ply_files:
        raise ValueError(f"No PLY files found in {sequence_dir}")

    # Random sampling
    random.seed(seed)
    if len(ply_files) <= num_frames:
        selected = ply_files
    else:
        selected = random.sample(ply_files, num_frames)

    return sorted(selected)


def benchmark_frame(frame_path, num_points, methods, output_dir):
    """Benchmark a single frame with all methods.

    Args:
        frame_path: Path to PLY file
        num_points: Number of points to subsample
        methods: Dict of method configs {'gpcc': [], 'cnn': []}
        output_dir: Where to save individual results

    Returns:
        List of result dictionaries
    """
    results = []

    # Load point cloud
    loader = PLYPointCloudLoader(verbose=False)
    pc = loader.load(str(frame_path))

    # Subsample if needed
    if pc.num_points > num_points:
        indices = np.random.choice(pc.num_points, size=num_points, replace=False)
        geometry = pc.geometry[indices]
        colors = pc.colors[indices] if pc.colors is not None else None
    else:
        geometry = pc.geometry
        colors = pc.colors

    # Test G-PCC
    if "gpcc" in methods and methods["gpcc"]:
        for config in methods["gpcc"]:
            try:
                adapter = GPCCAdapter()
                start = time.time()
                compressed, result = adapter.compress(geometry, colors)
                comp_time = time.time() - start

                recon_geom, recon_colors = adapter.decompress(compressed)
                metrics = GeometryQualityCalculator.calculate_all(geometry, recon_geom)

                bpp = (len(compressed) * 8) / len(geometry)

                results.append(
                    {
                        "frame": frame_path.name,
                        "method": "G-PCC",
                        "config": config,
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
                )
            except Exception as e:
                tqdm.write(f"    ✗ G-PCC {config} on {frame_path.name}: {e}")

    # Test pcc_geo_cnn_v2
    if "cnn" in methods and methods["cnn"]:
        for model in methods["cnn"]:
            try:
                adapter = PCCGeoCNNv2Adapter(model_name=model)
                start = time.time()
                compressed, result = adapter.compress(geometry, colors)
                comp_time = time.time() - start

                recon_geom, recon_colors = adapter.decompress(compressed)
                metrics = GeometryQualityCalculator.calculate_all(geometry, recon_geom)

                bpp = (result.compressed_size_bytes * 8) / len(geometry)

                results.append(
                    {
                        "frame": frame_path.name,
                        "method": "pcc_geo_cnn_v2",
                        "config": model,
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
                )
            except Exception as e:
                tqdm.write(f"    ✗ {model} on {frame_path.name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-frame benchmark for comprehensive evaluation"
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["longdress", "loot", "redandblack", "soldier"],
        help="Sequences to test",
    )
    parser.add_argument(
        "--frames-per-sequence",
        type=int,
        default=10,
        help="Number of random frames to test per sequence",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50000,
        help="Number of points to test (subsamples if needed)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/media/ssiannas/Shared Driv/jpeg-pleno",
        help="Root directory containing sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark/multi_frame",
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpcc-configs",
        nargs="+",
        default=["lossless"],
        help="G-PCC configurations to test",
    )
    parser.add_argument(
        "--cnn-models",
        nargs="+",
        default=["c1", "c2", "c3p"],
        help="pcc_geo_cnn_v2 models to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame selection",
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
    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Frame Benchmark")
    print("=" * 80)
    print(f"Sequences: {', '.join(args.sequences)}")
    print(f"Frames per sequence: {args.frames_per_sequence}")
    print(f"Points per frame: {args.num_points:,}")
    print(f"Total frames: {len(args.sequences) * args.frames_per_sequence}")
    print()

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare methods
    methods = {}
    if not args.skip_gpcc:
        methods["gpcc"] = args.gpcc_configs
    if not args.skip_cnn:
        methods["cnn"] = args.cnn_models

    # Collect all frames to test
    all_frames = []
    for seq in args.sequences:
        seq_dir = Path(args.data_root) / seq / "Ply"
        try:
            frames = get_sequence_frames(seq_dir, args.frames_per_sequence, args.seed)
            all_frames.extend([(seq, f) for f in frames])
            print(f"✓ {seq}: {len(frames)} frames selected")
        except Exception as e:
            print(f"✗ {seq}: {e}")

    if not all_frames:
        print("ERROR: No frames to test!")
        return

    print()
    print(f"Total frames to test: {len(all_frames)}")
    num_methods = len(methods.get("gpcc", [])) + len(methods.get("cnn", []))
    print(f"Methods per frame: {num_methods}")
    print(f"Total tests: {len(all_frames) * num_methods}")
    print()

    # Estimate time
    avg_time_per_test = 2 + 130 * len(methods.get("cnn", []))  # G-PCC fast, CNN ~130s
    total_time = len(all_frames) * avg_time_per_test
    print(f"Estimated time: {total_time / 3600:.1f} hours")
    print()

    # Run benchmarks
    all_results = []

    for seq, frame_path in tqdm(all_frames, desc="Overall Progress", unit="frame"):
        tqdm.write(f"\nTesting {frame_path.name} from {seq}...")

        try:
            frame_results = benchmark_frame(
                frame_path, args.num_points, methods, output_dir
            )

            # Add sequence info
            for r in frame_results:
                r["sequence"] = seq

            all_results.extend(frame_results)

            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv(output_dir / "results_intermediate.csv", index=False)

        except Exception as e:
            tqdm.write(f"✗ Error on {frame_path.name}: {e}")

    # Save final results
    df = pd.DataFrame(all_results)

    # Save as CSV
    csv_path = output_dir / "results_all_frames.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")

    # Save as JSON
    json_path = output_dir / "results_all_frames.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": {
                    "sequences": args.sequences,
                    "frames_per_sequence": args.frames_per_sequence,
                    "num_points": args.num_points,
                    "seed": args.seed,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"✓ Saved JSON: {json_path}")

    # Compute and save statistics
    print("\n" + "=" * 80)
    print("Computing Statistics")
    print("=" * 80)

    stats = []
    for method in df["method"].unique():
        for config in df[df["method"] == method]["config"].unique():
            for seq in df["sequence"].unique():
                subset = df[
                    (df["method"] == method)
                    & (df["config"] == config)
                    & (df["sequence"] == seq)
                ]

                if len(subset) > 0:
                    stats.append(
                        {
                            "method": method,
                            "config": config,
                            "sequence": seq,
                            "num_frames": len(subset),
                            "bpp_mean": subset["bpp"].mean(),
                            "bpp_std": subset["bpp"].std(),
                            "bpp_min": subset["bpp"].min(),
                            "bpp_max": subset["bpp"].max(),
                            "psnr_mean": subset["psnr"].mean(),
                            "psnr_std": subset["psnr"].std(),
                            "psnr_min": subset["psnr"].min(),
                            "psnr_max": subset["psnr"].max(),
                            "time_mean": subset["compression_time"].mean(),
                            "time_std": subset["compression_time"].std(),
                        }
                    )

    stats_df = pd.DataFrame(stats)
    stats_path = output_dir / "statistics_per_sequence.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Saved statistics: {stats_path}")

    # Overall statistics (all sequences combined)
    overall_stats = []
    for method in df["method"].unique():
        for config in df[df["method"] == method]["config"].unique():
            subset = df[(df["method"] == method) & (df["config"] == config)]

            if len(subset) > 0:
                overall_stats.append(
                    {
                        "method": method,
                        "config": config,
                        "num_frames": len(subset),
                        "bpp_mean": subset["bpp"].mean(),
                        "bpp_std": subset["bpp"].std(),
                        "psnr_mean": subset["psnr"].mean(),
                        "psnr_std": subset["psnr"].std(),
                        "time_mean": subset["compression_time"].mean(),
                    }
                )

    overall_df = pd.DataFrame(overall_stats)
    overall_path = output_dir / "statistics_overall.csv"
    overall_df.to_csv(overall_path, index=False)
    print(f"✓ Saved overall statistics: {overall_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics (All Sequences)")
    print("=" * 80)
    print()
    print(overall_df.to_string(index=False))
    print()

    print("✅ Multi-frame benchmark complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
