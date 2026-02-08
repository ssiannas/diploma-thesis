"""Compression-related metrics for point cloud compression benchmarking."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Container for compression-related metrics."""

    original_size_bytes: int  # Original point cloud size in bytes
    compressed_size_bytes: int  # Compressed point cloud size in bytes
    compression_ratio: float  # Compressed size / Original size
    bits_per_point: float  # Total bits used per point
    bpp_geometry: Optional[float] = None  # Bits per point for geometry
    bpp_color: Optional[float] = None  # Bits per point for color
    compression_time_seconds: Optional[float] = None  # Time to compress
    decompression_time_seconds: Optional[float] = None  # Time to decompress

    def __post_init__(self) -> None:
        """Validate metrics consistency."""
        if self.compression_ratio <= 0 or self.compression_ratio > 1:
            logger.warning(
                f"Compression ratio out of typical range: {self.compression_ratio:.4f}"
            )

    @property
    def compression_gain(self) -> float:
        """Compression gain as ratio of original to compressed size."""
        if self.compressed_size_bytes == 0:
            return float("inf")
        return self.original_size_bytes / self.compressed_size_bytes

    @property
    def compression_percentage(self) -> float:
        """Percentage reduction from original size."""
        return (1 - self.compression_ratio) * 100


class CompressionCalculator:
    """Calculate compression metrics for point clouds."""

    @staticmethod
    def calculate_from_sizes(
        original_size_bytes: int,
        compressed_size_bytes: int,
        num_points: int,
        bpp_geometry: Optional[float] = None,
        bpp_color: Optional[float] = None,
        compression_time_seconds: Optional[float] = None,
        decompression_time_seconds: Optional[float] = None,
    ) -> CompressionMetrics:
        """
        Calculate compression metrics from file sizes.

        Args:
            original_size_bytes: Original point cloud size in bytes.
            compressed_size_bytes: Compressed data size in bytes.
            num_points: Number of points in the point cloud.
            bpp_geometry: Bits per point for geometry (optional).
            bpp_color: Bits per point for color (optional).
            compression_time_seconds: Time taken to compress (optional).
            decompression_time_seconds: Time taken to decompress (optional).

        Returns:
            CompressionMetrics object.
        """
        if original_size_bytes <= 0:
            raise ValueError(f"Invalid original size: {original_size_bytes}")

        if num_points <= 0:
            raise ValueError(f"Invalid number of points: {num_points}")

        compression_ratio = compressed_size_bytes / original_size_bytes
        bits_per_point = (compressed_size_bytes * 8) / num_points

        return CompressionMetrics(
            original_size_bytes=original_size_bytes,
            compressed_size_bytes=compressed_size_bytes,
            compression_ratio=compression_ratio,
            bits_per_point=bits_per_point,
            bpp_geometry=bpp_geometry,
            bpp_color=bpp_color,
            compression_time_seconds=compression_time_seconds,
            decompression_time_seconds=decompression_time_seconds,
        )

    @staticmethod
    def calculate_from_files(
        original_file: Union[str, Path],
        compressed_file: Union[str, Path],
        num_points: int,
        **kwargs,
    ) -> CompressionMetrics:
        """
        Calculate compression metrics from actual files.

        Args:
            original_file: Path to original point cloud file.
            compressed_file: Path to compressed point cloud file.
            num_points: Number of points in the point cloud.
            **kwargs: Additional arguments for calculate_from_sizes.

        Returns:
            CompressionMetrics object.

        Raises:
            FileNotFoundError: If either file doesn't exist.
        """
        original_path = Path(original_file)
        compressed_path = Path(compressed_file)

        if not original_path.exists():
            raise FileNotFoundError(f"Original file not found: {original_path}")

        if not compressed_path.exists():
            raise FileNotFoundError(f"Compressed file not found: {compressed_path}")

        original_size = original_path.stat().st_size
        compressed_size = compressed_path.stat().st_size

        return CompressionCalculator.calculate_from_sizes(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            num_points=num_points,
            **kwargs,
        )

    @staticmethod
    def estimate_ply_file_size(
        num_points: int,
        has_colors: bool = True,
        has_normals: bool = False,
        ascii_format: bool = True,
    ) -> int:
        """
        Estimate uncompressed PLY file size (ASCII format).

        This is a rough estimate based on typical PLY formatting.

        Args:
            num_points: Number of points.
            has_colors: If True, include RGB colors (3 uint8).
            has_normals: If True, include normal vectors (3 floats).
            ascii_format: If True, estimate ASCII format size.

        Returns:
            Estimated file size in bytes.
        """
        # PLY header size (approximate)
        header_size = 500  # Account for header with comments

        # Per-point size in ASCII format
        # Each coordinate: ~7 chars + 1 space = 8 chars per value
        # ASCII: 3 coords * 8 = 24 chars
        # Color: 3 values * 4 = 12 chars (uint8 is 1-3 chars)
        # Newline: 1 char

        if ascii_format:
            # ASCII format (approximate character counts)
            coords_chars = 24  # 3 floats * ~8 chars each
            color_chars = 12 if has_colors else 0  # 3 * uint8
            normal_chars = 24 if has_normals else 0  # 3 floats
            newline_chars = 1

            point_size = coords_chars + color_chars + normal_chars + newline_chars
        else:
            # Binary format
            coords_bytes = 12  # 3 * 4-byte floats
            color_bytes = 3 if has_colors else 0  # 3 * uint8
            normal_bytes = 12 if has_normals else 0  # 3 * 4-byte floats

            point_size = coords_bytes + color_bytes + normal_bytes

        return header_size + num_points * point_size
