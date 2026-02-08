"""Demo script for baseline compression model and visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import JPEGPleno8iVFBSequence, PointCloudDataset
from pcml.metrics.compression import CompressionCalculator
from pcml.metrics.quality import ColorQualityCalculator, GeometryQualityCalculator
from pcml.models.baseline import BaselineCompressionModel
from pcml.visualization.metrics import plot_compression_metrics, plot_quality_metrics
from pcml.visualization.point_clouds import plot_point_cloud_comparison


def main():
    print("=" * 80)
    print("PCML Baseline Model Demo")
    print("=" * 80)

    print("\n[1/6] Loading point cloud data...")
    from pcml.data.loaders import PLYPointCloudLoader

    loader = PLYPointCloudLoader(verbose=False)
    ply_path = "datasets/8iVFB_small/longdress_vox10_1300.ply"
    frame = loader.load(ply_path)

    original_points = frame.num_points
    indices = np.random.choice(original_points, size=2048, replace=False)
    frame.geometry = frame.geometry[indices]
    frame.colors = frame.colors[indices]

    min_vals = frame.geometry.min(axis=0)
    max_vals = frame.geometry.max(axis=0)
    frame.geometry = 2 * (frame.geometry - min_vals) / (max_vals - min_vals) - 1

    print(
        f"  [OK] Loaded frame with {frame.num_points} points "
        f"(sampled from {original_points})"
    )
    print(f"  [OK] Geometry shape: {frame.geometry.shape}")
    print(f"  [OK] Colors shape: {frame.colors.shape}")

    print("\n[2/6] Creating baseline autoencoder model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [OK] Using device: {device}")

    model = BaselineCompressionModel(
        num_points=2048,
        latent_dim=512,
        hidden_dims=[1024, 512],
        compress_colors=False,
        name="baseline_geometry_only",
    )
    model = model.to(device)
    model.eval()
    print(
        f"  [OK] Model created with "
        f"{sum(p.numel() for p in model.parameters())} parameters"
    )

    print("\n[3/6] Running compression...")
    geometry_tensor = torch.from_numpy(frame.geometry).float().to(device)

    with torch.no_grad():
        compressed, recon_geometry, _ = model(geometry_tensor)

    print(f"  [OK] Original points: {frame.geometry.shape}")
    print(f"  [OK] Compressed size: {compressed.shape} (latent vector)")
    print(f"  [OK] Reconstructed points: {recon_geometry.shape}")

    compression_ratio = (frame.geometry.nbytes) / (compressed.numel() * 4)
    bpp = (compressed.numel() * 4 * 8) / frame.num_points
    print(f"  [OK] Compression ratio: {compression_ratio:.2f}x")
    print(f"  [OK] Bits per point: {bpp:.2f}")

    print("\n[4/6] Calculating quality metrics...")
    recon_geometry_np = recon_geometry.squeeze(0).cpu().numpy()

    geom_metrics = GeometryQualityCalculator.calculate_all(
        original=frame.geometry, reconstructed=recon_geometry_np
    )
    print(f"  [OK] Geometry PSNR: {geom_metrics.psnr:.2f} dB")
    print(f"  [OK] Geometry MSE: {geom_metrics.mse:.6f}")
    print(f"  [OK] Geometry D1: {geom_metrics.d1:.6f}")
    print(f"  [OK] Hausdorff distance: {geom_metrics.hausdorff:.6f}")

    print("\n[5/6] Creating visualizations...")

    fig1, _ = plot_point_cloud_comparison(
        original_geometry=frame.geometry,
        reconstructed_geometry=recon_geometry_np,
        original_colors=frame.colors,
        reconstructed_colors=frame.colors,
        title=f"Baseline Model (untrained) - PSNR: {geom_metrics.psnr:.2f} dB",
        size=2,
        alpha=0.6,
    )
    plt.savefig("results/demo_point_cloud_comparison.png", dpi=150, bbox_inches="tight")
    print("  [OK] Saved: results/demo_point_cloud_comparison.png")

    metrics_data = {
        "Baseline (untrained)": {
            "compression_ratio": compression_ratio,
            "bits_per_point": bpp,
            "psnr": geom_metrics.psnr,
            "mse": geom_metrics.mse,
        }
    }

    fig2, _ = plot_compression_metrics(
        results=metrics_data,
        metrics=["compression_ratio", "bits_per_point"],
        figsize=(10, 5),
    )
    plt.savefig("results/demo_compression_metrics.png", dpi=150, bbox_inches="tight")
    print("  [OK] Saved: results/demo_compression_metrics.png")

    fig3, _ = plot_quality_metrics(
        results=metrics_data, metrics=["psnr", "mse"], figsize=(10, 5)
    )
    plt.savefig("results/demo_quality_metrics.png", dpi=150, bbox_inches="tight")
    print("  [OK] Saved: results/demo_quality_metrics.png")

    print("\n[6/6] Summary")
    print("  " + "=" * 76)
    print(f"  Model: {model.name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Input points: {frame.num_points}")
    print(f"  Latent dimension: 512")
    print(f"  Compression: {compression_ratio:.2f}x ({bpp:.2f} bpp)")
    print(f"  Quality (PSNR): {geom_metrics.psnr:.2f} dB")
    print("  " + "=" * 76)
    print("\n  Note: This model is UNTRAINED, so quality is poor.")
    print("  After training, expect PSNR > 30 dB with similar compression.")
    print("\n[OK] Demo complete!")
    print("=" * 80)

    plt.close("all")


if __name__ == "__main__":
    main()
