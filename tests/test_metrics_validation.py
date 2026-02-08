"""Validate that our metrics are mathematically correct.

This test suite ensures our metrics match expected values and reference
implementations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.metrics.compression import CompressionCalculator
from pcml.metrics.quality import ColorQualityCalculator, GeometryQualityCalculator
from pcml.models.losses import ChamferDistance


class TestMetricsValidation:
    """Test suite for metrics validation."""

    @pytest.fixture
    def identical_point_clouds(self):
        """Create two identical point clouds."""
        np.random.seed(42)
        pc = np.random.randn(1000, 3).astype(np.float32)
        return pc, pc.copy()

    @pytest.fixture
    def slightly_noisy_clouds(self):
        """Create original and slightly noisy version."""
        np.random.seed(42)
        original = np.random.randn(1000, 3).astype(np.float32)
        noise = np.random.randn(1000, 3).astype(np.float32) * 0.01
        noisy = original + noise
        return original, noisy

    def test_identical_clouds_zero_error(self, identical_point_clouds):
        """Identical point clouds should have near-zero error."""
        pc1, pc2 = identical_point_clouds

        metrics = GeometryQualityCalculator.calculate_all(pc1, pc2)

        assert metrics.mse < 1e-10, f"MSE should be ~0, got {metrics.mse}"
        assert metrics.rmse < 1e-5, f"RMSE should be ~0, got {metrics.rmse}"
        assert metrics.d1 < 1e-5, f"D1 should be ~0, got {metrics.d1}"
        assert metrics.psnr > 100, f"PSNR should be very high, got {metrics.psnr}"

        print(
            f"[OK] Identical clouds: MSE={metrics.mse:.2e}, PSNR={metrics.psnr:.1f} dB"
        )

    def test_psnr_calculation_correctness(self, slightly_noisy_clouds):
        """Verify PSNR calculation is correct."""
        original, noisy = slightly_noisy_clouds

        metrics = GeometryQualityCalculator.calculate_all(original, noisy)

        bbox_diagonal = np.linalg.norm(original.max(axis=0) - original.min(axis=0))
        expected_psnr = 10 * np.log10(bbox_diagonal**2 / metrics.mse)

        assert np.abs(metrics.psnr - expected_psnr) < 0.01, (
            f"PSNR mismatch: calculated={metrics.psnr:.2f}, "
            f"expected={expected_psnr:.2f}"
        )

        print(f"[OK] PSNR calculation correct: {metrics.psnr:.2f} dB")

    def test_chamfer_distance_symmetry(self):
        """Chamfer distance should be symmetric."""
        np.random.seed(42)
        pc1 = np.random.randn(500, 3).astype(np.float32)
        pc2 = np.random.randn(500, 3).astype(np.float32) * 0.5

        pc1_torch = torch.from_numpy(pc1).unsqueeze(0)
        pc2_torch = torch.from_numpy(pc2).unsqueeze(0)

        chamfer = ChamferDistance()

        dist_12 = chamfer(pc1_torch, pc2_torch)
        dist_21 = chamfer(pc2_torch, pc1_torch)

        assert torch.allclose(
            dist_12, dist_21, rtol=1e-5
        ), f"Chamfer not symmetric: {dist_12:.6f} vs {dist_21:.6f}"

        print(f"[OK] Chamfer symmetry: {dist_12:.6f} == {dist_21:.6f}")

    def test_chamfer_distance_components(self):
        """Test Chamfer distance forward and backward components."""
        np.random.seed(42)
        pc1 = np.random.randn(500, 3).astype(np.float32)
        pc2 = np.random.randn(500, 3).astype(np.float32)

        pc1_torch = torch.from_numpy(pc1).unsqueeze(0)
        pc2_torch = torch.from_numpy(pc2).unsqueeze(0)

        chamfer = ChamferDistance()
        total, forward, backward = chamfer(pc1_torch, pc2_torch, return_components=True)

        assert torch.allclose(
            total, forward + backward
        ), "Total should equal forward + backward"

        print(
            f"[OK] Chamfer components: total={total:.6f}, "
            f"forward={forward:.6f}, backward={backward:.6f}"
        )

    def test_color_metrics_identical(self):
        """Identical colors should have zero error."""
        np.random.seed(42)
        colors = np.random.randint(0, 256, size=(1000, 3), dtype=np.uint8)

        metrics = ColorQualityCalculator.calculate_all(colors, colors)

        assert metrics.mse < 1e-10, f"Color MSE should be ~0, got {metrics.mse}"
        assert metrics.psnr > 100, f"Color PSNR should be very high, got {metrics.psnr}"

        print(
            f"[OK] Identical colors: MSE={metrics.mse:.2e}, PSNR={metrics.psnr:.1f} dB"
        )

    def test_compression_ratio_calculation(self):
        """Verify compression ratio calculation."""
        original_size = 1000000
        compressed_size = 100000
        num_points = 10000

        metrics = CompressionCalculator.calculate_from_sizes(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            num_points=num_points,
        )

        expected_ratio = compressed_size / original_size
        expected_bpp = (compressed_size * 8) / num_points

        assert np.abs(metrics.compression_ratio - expected_ratio) < 1e-6
        assert np.abs(metrics.bits_per_point - expected_bpp) < 1e-6
        assert np.abs(metrics.compression_gain - 10.0) < 1e-6

        print(
            f"[OK] Compression: ratio={metrics.compression_ratio:.2f}, "
            f"bpp={metrics.bits_per_point:.2f}"
        )

    def test_d1_d2_relationship(self, slightly_noisy_clouds):
        """D2 should be >= D1 (both directions)."""
        original, noisy = slightly_noisy_clouds

        metrics = GeometryQualityCalculator.calculate_all(original, noisy)

        assert (
            metrics.d2_symmetric >= metrics.d1
        ), f"D2 ({metrics.d2_symmetric}) should be >= D1 ({metrics.d1})"

        print(f"[OK] D1={metrics.d1:.6f}, D2={metrics.d2_symmetric:.6f}")

    def test_hausdorff_equals_d2(self, slightly_noisy_clouds):
        """Hausdorff distance should equal D2_symmetric."""
        original, noisy = slightly_noisy_clouds

        metrics = GeometryQualityCalculator.calculate_all(original, noisy)

        assert np.abs(metrics.hausdorff - metrics.d2_symmetric) < 1e-6, (
            f"Hausdorff ({metrics.hausdorff}) should equal "
            f"D2 ({metrics.d2_symmetric})"
        )

        print(f"[OK] Hausdorff == D2: {metrics.hausdorff:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
