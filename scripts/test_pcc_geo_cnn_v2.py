"""Test pcc_geo_cnn_v2 compression framework."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import PLYPointCloudLoader
from pcml.frameworks.pcc_geo_cnn_v2 import PCCGeoCNNv2Adapter
from pcml.metrics.quality import GeometryQualityCalculator


def main():
    print("=" * 60)
    print("Testing pcc_geo_cnn_v2 Compression")
    print("=" * 60)
    print()

    # Check if models exist
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "frameworks" / "pcc_geo_cnn_v2" / "models"

    if not models_dir.exists():
        print("ERROR: Models directory not found!")
        print(f"Expected: {models_dir}")
        print()
        print("Download models from:")
        print("https://drive.google.com/file/d/18uHmr0ZpgFLeL9Y5TUFTsQkRfz4XpQdJ/view")
        print()
        print("See: scripts/download_models_instructions.md")
        return

    available_models = sorted([d.name for d in models_dir.iterdir() if d.is_dir()])
    if not available_models:
        print("ERROR: No model subdirectories found!")
        print(f"Expected directories like: c1/, c2/, c3/, ..., c6/")
        return

    print(f"Available models: {', '.join(available_models)}")
    print()

    # Test configuration - use c1 (simpler architecture, no hyper prior)
    preferred_model = "c1"
    if preferred_model in available_models:
        model_name = preferred_model
    else:
        model_name = available_models[0] if available_models else "c1"
    print(f"Using model: {model_name}")

    # Load point cloud
    input_ply = project_root / "datasets" / "8iVFB_small" / "longdress_vox10_1300.ply"
    if not input_ply.exists():
        print(f"ERROR: Test point cloud not found: {input_ply}")
        return

    print(f"Loading point cloud: {input_ply.name}")
    loader = PLYPointCloudLoader(verbose=False)
    pc = loader.load(str(input_ply))

    # Subsample to reduce test time
    num_points = min(pc.num_points, 10000)
    if pc.num_points > num_points:
        indices = np.random.choice(pc.num_points, size=num_points, replace=False)
        geometry = pc.geometry[indices]
        colors = pc.colors[indices] if pc.colors is not None else None
        print(
            f"Subsampled {pc.num_points:,} → {num_points:,} points for faster testing"
        )
    else:
        geometry = pc.geometry
        colors = pc.colors

    print(f"Points: {len(geometry):,}")
    print()

    # Create adapter
    print(f"Creating adapter for model '{model_name}'...")
    try:
        adapter = PCCGeoCNNv2Adapter(
            model_name=model_name, resolution=1024, octree_level=3
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Compress
    print()
    print("Compressing...")
    try:
        compressed_data, result = adapter.compress(geometry, colors)
    except Exception as e:
        print(f"ERROR during compression: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"✓ Compressed: {result.compressed_size_bytes:,} bytes")
    print(f"  Time: {result.compression_time_seconds:.2f}s")
    print(f"  BPP: {(result.compressed_size_bytes * 8) / len(geometry):.2f}")

    # Decompress
    print()
    print("Decompressing...")
    try:
        recon_geometry, recon_colors = adapter.decompress(compressed_data)
    except Exception as e:
        print(f"ERROR during decompression: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"✓ Decompressed: {len(recon_geometry):,} points")

    # Calculate metrics
    print()
    print("Calculating metrics...")
    metrics = GeometryQualityCalculator.calculate_all(geometry, recon_geometry)

    bpp = (result.compressed_size_bytes * 8) / len(geometry)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Input points: {len(geometry):,}")
    print(f"Output points: {len(recon_geometry):,}")
    print(f"Compressed size: {result.compressed_size_bytes:,} bytes")
    print(f"BPP: {bpp:.2f}")
    print(f"PSNR: {metrics.psnr:.2f} dB")
    print(f"MSE: {metrics.mse:.6f}")
    print(f"RMSE: {metrics.rmse:.6f}")
    print(f"Compression time: {result.compression_time_seconds:.2f}s")
    print()

    # Save results
    output_dir = project_root / "results" / "pcc_geo_cnn_v2_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"metrics_{model_name}.txt", "w") as f:
        f.write(f"pcc_geo_cnn_v2 Test Results - Model {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {input_ply.name}\n")
        f.write(f"Points: {len(geometry):,}\n")
        f.write(f"Compressed size: {result.compressed_size_bytes:,} bytes\n")
        f.write(f"BPP: {bpp:.2f}\n")
        f.write(f"PSNR: {metrics.psnr:.2f} dB\n")
        f.write(f"MSE: {metrics.mse:.6f}\n")
        f.write(f"RMSE: {metrics.rmse:.6f}\n")
        f.write(f"Compression time: {result.compression_time_seconds:.2f}s\n")

    print(f"Results saved to: {output_dir}/")
    print()
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    main()
