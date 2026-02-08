"""Base classes for point cloud compression models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CompressionOutput:
    """Output from compression model.

    Attributes:
        compressed_data: Compressed representation (latent code or bitstream)
        reconstructed_geometry: Reconstructed point cloud geometry (N, 3)
        reconstructed_colors: Optional reconstructed colors (N, 3)
        compression_time: Time taken for compression in seconds
        decompression_time: Time taken for decompression in seconds
        num_bytes: Estimated size of compressed data in bytes
    """

    compressed_data: torch.Tensor
    reconstructed_geometry: np.ndarray
    reconstructed_colors: Optional[np.ndarray] = None
    compression_time: float = 0.0
    decompression_time: float = 0.0
    num_bytes: int = 0

    def __post_init__(self) -> None:
        if self.reconstructed_geometry.shape[-1] != 3:
            raise ValueError(
                f"Geometry must have 3 coordinates, "
                f"got {self.reconstructed_geometry.shape[-1]}"
            )
        if (
            self.reconstructed_colors is not None
            and self.reconstructed_colors.shape[-1] != 3
        ):
            raise ValueError(
                f"Colors must have 3 channels, "
                f"got {self.reconstructed_colors.shape[-1]}"
            )


class BaseCompressionModel(nn.Module, ABC):
    """Abstract base class for point cloud compression models.

    All compression models should inherit from this class and implement
    the compress() and decompress() methods.
    """

    def __init__(self, name: str = "base_model") -> None:
        """Initialize the model.

        Args:
            name: Model name for identification
        """
        super().__init__()
        self.name = name

    @abstractmethod
    def compress(
        self,
        geometry: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compress point cloud to latent representation.

        Args:
            geometry: Point cloud coordinates (B, N, 3) or (N, 3)
            colors: Optional point colors (B, N, 3) or (N, 3)

        Returns:
            Compressed latent representation
        """
        pass

    @abstractmethod
    def decompress(
        self,
        compressed: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decompress latent representation to point cloud.

        Args:
            compressed: Compressed latent representation

        Returns:
            Tuple of (reconstructed_geometry, reconstructed_colors)
            reconstructed_geometry: (B, N, 3) or (N, 3)
            reconstructed_colors: (B, N, 3) or (N, 3) or None
        """
        pass

    def forward(
        self,
        geometry: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
    ) -> CompressionOutput:
        """Full compression-decompression cycle.

        Args:
            geometry: Point cloud coordinates (B, N, 3) or (N, 3)
            colors: Optional point colors (B, N, 3) or (N, 3)

        Returns:
            CompressionOutput containing reconstructed data and metadata
        """
        import time

        t0 = time.time()
        compressed = self.compress(geometry, colors)
        compression_time = time.time() - t0

        t0 = time.time()
        recon_geom, recon_colors = self.decompress(compressed)
        decompression_time = time.time() - t0

        num_bytes = compressed.numel() * 4

        recon_geom_np = recon_geom.detach().cpu().numpy()
        recon_colors_np = (
            recon_colors.detach().cpu().numpy() if recon_colors is not None else None
        )

        return CompressionOutput(
            compressed_data=compressed,
            reconstructed_geometry=recon_geom_np,
            reconstructed_colors=recon_colors_np,
            compression_time=compression_time,
            decompression_time=decompression_time,
            num_bytes=num_bytes,
        )

    def estimate_bpp(self, compressed: torch.Tensor, num_points: int) -> float:
        """Estimate bits per point.

        Args:
            compressed: Compressed representation
            num_points: Number of points in original cloud

        Returns:
            Bits per point
        """
        num_bytes = compressed.numel() * 4
        num_bits = num_bytes * 8
        return num_bits / num_points
