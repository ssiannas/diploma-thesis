"""Quality metrics for point cloud compression evaluation.

References:
    Geometric metrics (D1, D2): Tian et al., "Geometric distortion metrics for
        point cloud compression," ICIP 2017.
    PSNR for point clouds: Mekuria et al., "Design, Implementation, and Evaluation
        of a Point Cloud Codec for Tele-Immersive Video," TCSVT 2017.
    Chamfer distance: Barrow et al., "Parametric correspondence and chamfer matching,"
        IJCAI 1977.

See .docs/BIBLIOGRAPHY.md for full citations.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class GeometryMetrics:
    """Geometry quality metrics for point clouds."""

    mse: float  # Mean squared error
    rmse: float  # Root mean squared error
    psnr: Optional[float] = None  # Peak signal-to-noise ratio
    d1: Optional[float] = None  # D1 metric (max of one-directional distance)
    d2_sym: Optional[float] = None  # D2 symmetric metric (bidirectional distance)
    hausdorff: Optional[float] = None  # Hausdorff distance

    def __post_init__(self) -> None:
        """Validate metric consistency."""
        if self.mse < 0:
            raise ValueError(f"MSE must be non-negative, got {self.mse}")
        if self.rmse < 0:
            raise ValueError(f"RMSE must be non-negative, got {self.rmse}")
        if self.psnr is not None and self.psnr < 0:
            logger.warning(f"PSNR is negative: {self.psnr}")


@dataclass
class ColorMetrics:
    """Color quality metrics for point clouds with color attributes."""

    mse: float  # Mean squared error in RGB space
    rmse: float  # Root mean squared error
    psnr: Optional[float] = None  # Peak signal-to-noise ratio (using 255 as max)
    mae: Optional[float] = None  # Mean absolute error
    max_error: Optional[float] = None  # Maximum per-channel error

    def __post_init__(self) -> None:
        """Validate metric consistency."""
        if self.mse < 0:
            raise ValueError(f"MSE must be non-negative, got {self.mse}")
        if self.rmse < 0:
            raise ValueError(f"RMSE must be non-negative, got {self.rmse}")


@dataclass
class QualityMetrics:
    """Complete quality metrics for point cloud compression."""

    geometry: GeometryMetrics
    color: Optional[ColorMetrics] = None


class GeometryQualityCalculator:
    """Calculate geometry quality metrics for point clouds."""

    @staticmethod
    def calculate_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate mean squared error between two point clouds.

        Args:
            original: Original point cloud (N, 3).
            reconstructed: Reconstructed point cloud (M, 3).

        Returns:
            Mean squared error.
        """
        if original.shape[1] != 3 or reconstructed.shape[1] != 3:
            raise ValueError("Both point clouds must have shape (N, 3)")

        # Compute Euclidean distance from each reconstructed point to nearest original
        distances_sq = np.min(
            cdist(reconstructed, original, metric="euclidean") ** 2, axis=1
        )

        return np.mean(distances_sq)

    @staticmethod
    def calculate_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate root mean squared error.

        Args:
            original: Original point cloud (N, 3).
            reconstructed: Reconstructed point cloud (M, 3).

        Returns:
            Root mean squared error.
        """
        mse = GeometryQualityCalculator.calculate_mse(original, reconstructed)
        return np.sqrt(mse)

    @staticmethod
    def calculate_psnr(
        original: np.ndarray,
        reconstructed: np.ndarray,
        max_value: Optional[float] = None,
    ) -> float:
        """
        Calculate peak signal-to-noise ratio.

        For geometry, max_value is typically the bounding box diagonal.

        Args:
            original: Original point cloud (N, 3).
            reconstructed: Reconstructed point cloud (M, 3).
            max_value: Maximum value for PSNR calculation (bounding box diagonal).
                If None, computed from original point cloud.

        Returns:
            PSNR in dB.
        """
        if max_value is None:
            # Use bounding box diagonal as max value
            min_coords = original.min(axis=0)
            max_coords = original.max(axis=0)
            max_value = np.linalg.norm(max_coords - min_coords)

        if max_value == 0:
            return float("inf")

        mse = GeometryQualityCalculator.calculate_mse(original, reconstructed)

        if mse == 0:
            return float("inf")

        psnr = 20 * np.log10(max_value) - 10 * np.log10(mse)
        return psnr

    @staticmethod
    def calculate_d1(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate D1 metric (one-directional Chamfer distance).

        D1 is the maximum distance from reconstructed to original point cloud.

        Args:
            original: Original point cloud (N, 3).
            reconstructed: Reconstructed point cloud (M, 3).

        Returns:
            D1 metric value.
        """
        # Distance from reconstructed to original
        distances = np.min(cdist(reconstructed, original, metric="euclidean"), axis=1)
        return np.max(distances)

    @staticmethod
    def calculate_d2_symmetric(
        original: np.ndarray, reconstructed: np.ndarray
    ) -> float:
        """
        Calculate D2 symmetric metric (bidirectional Chamfer distance).

        D2_sym = max(d(reconstructed -> original), d(original -> reconstructed))

        Args:
            original: Original point cloud (N, 3).
            reconstructed: Reconstructed point cloud (M, 3).

        Returns:
            D2 symmetric metric value.
        """
        # Distance from reconstructed to original
        d1 = GeometryQualityCalculator.calculate_d1(original, reconstructed)

        # Distance from original to reconstructed
        d2 = GeometryQualityCalculator.calculate_d1(reconstructed, original)

        return max(d1, d2)

    @staticmethod
    def calculate_hausdorff(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate Hausdorff distance.

        This is equivalent to D2_sym metric.

        Args:
            original: Original point cloud (N, 3).
            reconstructed: Reconstructed point cloud (M, 3).

        Returns:
            Hausdorff distance.
        """
        return GeometryQualityCalculator.calculate_d2_symmetric(original, reconstructed)

    @staticmethod
    def calculate_all(
        original: np.ndarray, reconstructed: np.ndarray
    ) -> GeometryMetrics:
        """
        Calculate all geometry quality metrics.

        Args:
            original: Original point cloud (N, 3).
            reconstructed: Reconstructed point cloud (M, 3).

        Returns:
            GeometryMetrics object with all metrics computed.
        """
        mse = GeometryQualityCalculator.calculate_mse(original, reconstructed)
        rmse = np.sqrt(mse)
        psnr = GeometryQualityCalculator.calculate_psnr(original, reconstructed)
        d1 = GeometryQualityCalculator.calculate_d1(original, reconstructed)
        d2_sym = GeometryQualityCalculator.calculate_d2_symmetric(
            original, reconstructed
        )
        hausdorff = d2_sym  # Equivalent to D2_sym

        return GeometryMetrics(
            mse=mse,
            rmse=rmse,
            psnr=psnr,
            d1=d1,
            d2_sym=d2_sym,
            hausdorff=hausdorff,
        )


class ColorQualityCalculator:
    """Calculate color quality metrics for point clouds with color attributes."""

    @staticmethod
    def validate_colors(colors: np.ndarray) -> None:
        """
        Validate color array format.

        Args:
            colors: Color array (N, 3).

        Raises:
            ValueError: If color array is invalid.
        """
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"Colors must have shape (N, 3), got {colors.shape}")

    @staticmethod
    def calculate_mse(
        original_colors: np.ndarray, reconstructed_colors: np.ndarray
    ) -> float:
        """
        Calculate mean squared error in RGB color space.

        Args:
            original_colors: Original colors (N, 3) in range [0, 255] or [0, 1].
            reconstructed_colors: Reconstructed colors (N, 3) same range.

        Returns:
            MSE in RGB space.
        """
        ColorQualityCalculator.validate_colors(original_colors)
        ColorQualityCalculator.validate_colors(reconstructed_colors)

        if original_colors.shape[0] != reconstructed_colors.shape[0]:
            raise ValueError(
                f"Color arrays must have same number of points: "
                f"{original_colors.shape[0]} vs {reconstructed_colors.shape[0]}"
            )

        diff = original_colors.astype(np.float32) - reconstructed_colors.astype(
            np.float32
        )
        mse = np.mean(diff**2)

        return mse

    @staticmethod
    def calculate_psnr(
        original_colors: np.ndarray,
        reconstructed_colors: np.ndarray,
        max_value: float = 255.0,
    ) -> float:
        """
        Calculate peak signal-to-noise ratio for colors.

        Args:
            original_colors: Original colors (N, 3).
            reconstructed_colors: Reconstructed colors (N, 3).
            max_value: Maximum color value (255 for uint8, 1.0 for float).

        Returns:
            PSNR in dB.
        """
        mse = ColorQualityCalculator.calculate_mse(
            original_colors, reconstructed_colors
        )

        if mse == 0:
            return float("inf")

        psnr = 20 * np.log10(max_value) - 10 * np.log10(mse)
        return psnr

    @staticmethod
    def calculate_mae(
        original_colors: np.ndarray, reconstructed_colors: np.ndarray
    ) -> float:
        """
        Calculate mean absolute error in RGB color space.

        Args:
            original_colors: Original colors (N, 3).
            reconstructed_colors: Reconstructed colors (N, 3).

        Returns:
            MAE in RGB space.
        """
        ColorQualityCalculator.validate_colors(original_colors)
        ColorQualityCalculator.validate_colors(reconstructed_colors)

        diff = np.abs(
            original_colors.astype(np.float32) - reconstructed_colors.astype(np.float32)
        )
        mae = np.mean(diff)

        return mae

    @staticmethod
    def calculate_max_error(
        original_colors: np.ndarray, reconstructed_colors: np.ndarray
    ) -> float:
        """
        Calculate maximum per-channel error.

        Args:
            original_colors: Original colors (N, 3).
            reconstructed_colors: Reconstructed colors (N, 3).

        Returns:
            Maximum absolute error across all channels and points.
        """
        ColorQualityCalculator.validate_colors(original_colors)
        ColorQualityCalculator.validate_colors(reconstructed_colors)

        diff = np.abs(
            original_colors.astype(np.float32) - reconstructed_colors.astype(np.float32)
        )
        max_error = np.max(diff)

        return max_error

    @staticmethod
    def calculate_all(
        original_colors: np.ndarray,
        reconstructed_colors: np.ndarray,
        max_value: float = 255.0,
    ) -> ColorMetrics:
        """
        Calculate all color quality metrics.

        Args:
            original_colors: Original colors (N, 3).
            reconstructed_colors: Reconstructed colors (N, 3).
            max_value: Maximum color value (255 for uint8, 1.0 for float).

        Returns:
            ColorMetrics object with all metrics computed.
        """
        mse = ColorQualityCalculator.calculate_mse(
            original_colors, reconstructed_colors
        )
        rmse = np.sqrt(mse)
        psnr = ColorQualityCalculator.calculate_psnr(
            original_colors, reconstructed_colors, max_value
        )
        mae = ColorQualityCalculator.calculate_mae(
            original_colors, reconstructed_colors
        )
        max_error = ColorQualityCalculator.calculate_max_error(
            original_colors, reconstructed_colors
        )

        return ColorMetrics(mse=mse, rmse=rmse, psnr=psnr, mae=mae, max_error=max_error)
