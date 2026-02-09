"""Test G-PCC adapter."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import PLYPointCloudLoader
from pcml.frameworks.gpcc import GPCCAdapter
from pcml.metrics.quality import GeometryQualityCalculator


def main():
    print("Testing G-PCC adapter")

    loader = PLYPointCloudLoader(verbose=False)
    pc = loader.load("datasets/8iVFB_small/longdress_vox10_1300.ply")

    indices = np.random.choice(pc.num_points, size=10000, replace=False)
    geometry = pc.geometry[indices]
    colors = pc.colors[indices] if pc.colors is not None else None

    print(f"Input: {len(geometry)} points")

    adapter = GPCCAdapter()

    print("Compressing...")
    compressed_data, result = adapter.compress(geometry, colors)
    print(f"Compressed: {result.compressed_size_bytes} bytes")
    print(f"Time: {result.compression_time_seconds:.2f}s")

    print("Decompressing...")
    recon_geometry, recon_colors = adapter.decompress(compressed_data)
    print(f"Reconstructed: {len(recon_geometry)} points")

    metrics = GeometryQualityCalculator.calculate_all(geometry, recon_geometry)
    print(f"PSNR: {metrics.psnr:.2f} dB")
    print(f"MSE: {metrics.mse:.6f}")

    print("\nG-PCC adapter working!")


if __name__ == "__main__":
    main()
