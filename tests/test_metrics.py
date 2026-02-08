#!/usr/bin/env python3
"""Tests for metrics calculation functionality."""

import numpy as np
import pytest

from pcml.metrics import CompressionMetrics
from pcml.metrics.compression import CompressionCalculator
from pcml.metrics.quality import (
    ColorMetrics,
    ColorQualityCalculator,
    GeometryMetrics,
    GeometryQualityCalculator,
)


def test_compression_metrics_basic() -> None:
    """Test compression metrics calculation."""
    # Create sample metrics
    metrics = CompressionMetrics(
        original_size_bytes=1000000,
        compressed_size_bytes=100000,
        compression_ratio=0.1,
        bits_per_point=32,
    )

    # Assertions
    assert metrics.original_size_bytes == 1000000
    assert metrics.compressed_size_bytes == 100000
    assert metrics.compression_ratio == 0.1
    assert metrics.bits_per_point == 32
    assert metrics.compression_gain == 10.0  # 1 / 0.1
    assert abs(metrics.compression_percentage - 90.0) < 0.01

    print(f"\nCompression Metrics:")
    print(f"  Original size: {metrics.original_size_bytes:,} bytes")
    print(f"  Compressed size: {metrics.compressed_size_bytes:,} bytes")
    print(f"  Compression ratio: {metrics.compression_ratio:.4f}")
    print(f"  Compression gain: {metrics.compression_gain:.2f}x")
    print(f"  Compression percentage: {metrics.compression_percentage:.1f}%")
    print(f"  Bits per point: {metrics.bits_per_point:.2f}")


def test_compression_calculator() -> None:
    """Test compression calculator functionality."""
    calc_metrics = CompressionCalculator.calculate_from_sizes(
        original_size_bytes=1000000,
        compressed_size_bytes=100000,
        num_points=100000,
        bpp_geometry=24.0,
        bpp_color=8.0,
    )

    # Assertions
    assert calc_metrics.compression_ratio == 0.1
    assert calc_metrics.bits_per_point == 8.0  # 100000 bytes * 8 / 100000 points
    assert calc_metrics.bpp_geometry == 24.0
    assert calc_metrics.bpp_color == 8.0

    print(f"\nCalculated Metrics:")
    print(f"  Compression ratio: {calc_metrics.compression_ratio:.4f}")
    print(f"  Bits per point: {calc_metrics.bits_per_point:.2f}")
    print(f"  BPP Geometry: {calc_metrics.bpp_geometry}")
    print(f"  BPP Color: {calc_metrics.bpp_color}")


def test_ply_size_estimation() -> None:
    """Test PLY file size estimation."""
    ply_size = CompressionCalculator.estimate_ply_file_size(
        num_points=1000000, has_colors=True, has_normals=False
    )

    # Assertions
    assert ply_size > 0, "PLY size should be positive"
    assert isinstance(ply_size, int), "PLY size should be an integer"

    print(f"\nEstimated PLY size for 1M points with colors: {ply_size:,} bytes")


def test_geometry_metrics_individual(
    sample_point_cloud: np.ndarray, sample_point_cloud_with_noise: np.ndarray
) -> None:
    """Test individual geometry quality metrics."""
    original = sample_point_cloud
    reconstructed = sample_point_cloud_with_noise

    # Calculate individual metrics
    mse = GeometryQualityCalculator.calculate_mse(original, reconstructed)
    rmse = GeometryQualityCalculator.calculate_rmse(original, reconstructed)
    psnr = GeometryQualityCalculator.calculate_psnr(original, reconstructed)
    d1 = GeometryQualityCalculator.calculate_d1(original, reconstructed)
    d2_sym = GeometryQualityCalculator.calculate_d2_symmetric(original, reconstructed)

    # Assertions
    assert mse >= 0, "MSE should be non-negative"
    assert rmse >= 0, "RMSE should be non-negative"
    assert np.isfinite(psnr), "PSNR should be finite"
    assert d1 >= 0, "D1 should be non-negative"
    assert d2_sym >= 0, "D2 symmetric should be non-negative"

    print(f"\nGeometry Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  D1: {d1:.6f}")
    print(f"  D2 (symmetric): {d2_sym:.6f}")


def test_geometry_metrics_all(
    sample_point_cloud: np.ndarray, sample_point_cloud_with_noise: np.ndarray
) -> None:
    """Test calculating all geometry quality metrics at once."""
    original = sample_point_cloud
    reconstructed = sample_point_cloud_with_noise

    # Test all metrics calculation
    all_metrics = GeometryQualityCalculator.calculate_all(original, reconstructed)

    # Assertions
    assert all_metrics.mse >= 0
    assert all_metrics.rmse >= 0
    assert np.isfinite(all_metrics.psnr)
    assert all_metrics.hausdorff >= 0

    print(f"\nAll geometry metrics at once:")
    print(f"  MSE: {all_metrics.mse:.6f}")
    print(f"  RMSE: {all_metrics.rmse:.6f}")
    print(f"  PSNR: {all_metrics.psnr:.2f} dB")
    print(f"  Hausdorff: {all_metrics.hausdorff:.6f}")


def test_color_metrics_individual(
    sample_colors: np.ndarray, sample_colors_with_noise: np.ndarray
) -> None:
    """Test individual color quality metrics."""
    original_colors = sample_colors
    reconstructed_colors = sample_colors_with_noise

    # Calculate individual metrics
    mse = ColorQualityCalculator.calculate_mse(original_colors, reconstructed_colors)
    psnr = ColorQualityCalculator.calculate_psnr(original_colors, reconstructed_colors)
    mae = ColorQualityCalculator.calculate_mae(original_colors, reconstructed_colors)
    max_error = ColorQualityCalculator.calculate_max_error(
        original_colors, reconstructed_colors
    )

    # Assertions
    assert mse >= 0, "MSE should be non-negative"
    assert np.isfinite(psnr), "PSNR should be finite"
    assert mae >= 0, "MAE should be non-negative"
    assert max_error >= 0, "Max error should be non-negative"

    print(f"\nColor Metrics:")
    print(f"  MSE: {mse:.2f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE: {mae:.2f}")
    print(f"  Max error: {max_error:.2f}")


def test_color_metrics_all(
    sample_colors: np.ndarray, sample_colors_with_noise: np.ndarray
) -> None:
    """Test calculating all color quality metrics at once."""
    original_colors = sample_colors
    reconstructed_colors = sample_colors_with_noise

    # Test all metrics calculation
    all_metrics = ColorQualityCalculator.calculate_all(
        original_colors, reconstructed_colors
    )

    # Assertions
    assert all_metrics.mse >= 0
    assert all_metrics.rmse >= 0
    assert np.isfinite(all_metrics.psnr)
    assert all_metrics.mae >= 0
    assert all_metrics.max_error >= 0

    print(f"\nAll color metrics:")
    print(f"  MSE: {all_metrics.mse:.2f}")
    print(f"  RMSE: {all_metrics.rmse:.2f}")
    print(f"  PSNR: {all_metrics.psnr:.2f} dB")
    print(f"  MAE: {all_metrics.mae:.2f}")
    print(f"  Max error: {all_metrics.max_error:.2f}")
