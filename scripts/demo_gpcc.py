"""Demo G-PCC compression with visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import PLYPointCloudLoader
from pcml.frameworks.gpcc import GPCCAdapter
from pcml.metrics.quality import GeometryQualityCalculator
from pcml.visualization.point_clouds import plot_point_cloud_comparison


def main():
    output_dir = Path("results/gpcc_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading point cloud...")
    loader = PLYPointCloudLoader(verbose=False)
    pc = loader.load("datasets/8iVFB_small/longdress_vox10_1300.ply")

    indices = np.random.choice(pc.num_points, size=5000, replace=False)
    geometry = pc.geometry[indices]
    colors = pc.colors[indices] if pc.colors is not None else None

    print(f"Points: {len(geometry)}")

    print("\nCompressing with G-PCC...")
    adapter = GPCCAdapter()
    compressed_data, result = adapter.compress(geometry, colors)

    print(f"Compressed size: {result.compressed_size_bytes:,} bytes")
    print(f"Compression time: {result.compression_time_seconds:.3f}s")

    print("\nDecompressing...")
    recon_geometry, recon_colors = adapter.decompress(compressed_data)

    print("\nCalculating metrics...")
    metrics = GeometryQualityCalculator.calculate_all(geometry, recon_geometry)

    bpp = (result.compressed_size_bytes * 8) / len(geometry)

    print(f"PSNR: {metrics.psnr:.2f} dB")
    print(f"BPP: {bpp:.2f}")
    print(f"MSE: {metrics.mse:.6f}")

    print("\nGenerating visualization...")
    fig, (ax1, ax2) = plot_point_cloud_comparison(
        geometry,
        recon_geometry,
        colors,
        recon_colors,
        title=f"G-PCC Compression (PSNR: {metrics.psnr:.1f} dB, BPP: {bpp:.2f})",
    )

    output_path = output_dir / "gpcc_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.close()

    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"G-PCC Compression Results\n")
        f.write(f"========================\n\n")
        f.write(f"Points: {len(geometry)}\n")
        f.write(f"Compressed size: {result.compressed_size_bytes:,} bytes\n")
        f.write(f"BPP: {bpp:.2f}\n")
        f.write(f"PSNR: {metrics.psnr:.2f} dB\n")
        f.write(f"MSE: {metrics.mse:.6f}\n")
        f.write(f"Compression time: {result.compression_time_seconds:.3f}s\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
