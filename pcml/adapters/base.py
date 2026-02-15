"""Base adapter interface for compression frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CompressionResult:
    """Results from compression operation."""

    compressed_size_bytes: int
    compression_time_seconds: float
    decompression_time_seconds: float
    compressed_data: Optional[bytes] = None
    metadata: Optional[dict] = None


class BaseAdapter(ABC):
    """Base adapter for external compression frameworks."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compress(
        self, geometry: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> tuple[bytes, CompressionResult]:
        """Compress point cloud.

        Args:
            geometry: (N, 3) point coordinates
            colors: Optional (N, 3) RGB colors [0-255]

        Returns:
            (compressed_data, result)
        """
        pass

    @abstractmethod
    def decompress(
        self, compressed_data: bytes
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Decompress point cloud.

        Args:
            compressed_data: Compressed bytes

        Returns:
            (geometry, colors)
        """
        pass
