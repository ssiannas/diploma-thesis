"""Simple baseline autoencoder without normalization.

This is the TRUE baseline - a simple MLP autoencoder following the original
Hinton & Salakhutdinov (2006) architecture without any modern additions like
batch normalization, dropout, or skip connections.

References:
    Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality
    of data with neural networks," Science, 313(5786), 504-507.

See .docs/BIBLIOGRAPHY.md and .docs/CORE_PHILOSOPHY.md for rationale.
"""

from typing import Optional

import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    """Simple MLP autoencoder for point cloud compression - NO NORMALIZATION.

    This is the baseline reference point - the simplest autoencoder that could
    work. No BatchNorm, no LayerNorm, no dropout - just MLPs with ReLU.

    Following Hinton & Salakhutdinov (2006), this provides the true baseline
    against which we can measure the benefit of adding modern techniques.
    """

    def __init__(
        self,
        num_points: int = 2048,
        latent_dim: int = 512,
        hidden_dims: list[int] | None = None,
        compress_colors: bool = False,
    ) -> None:
        """Initialize the simple autoencoder.

        Args:
            num_points: Number of points in the point cloud
            latent_dim: Dimension of the latent representation
            hidden_dims: List of hidden layer dimensions. Default: [1024, 512]
            compress_colors: Whether to compress colors along with geometry
        """
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.compress_colors = compress_colors

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        # Input dimension: 3 for geometry, +3 if colors
        input_dim = num_points * 3
        if compress_colors:
            input_dim = num_points * 6

        # Encoder - simple MLP with ReLU
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder - symmetric to encoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to latent representation.

        Args:
            x: Flattened point cloud (B, N*C) where C is 3 or 6

        Returns:
            Latent representation (B, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to point cloud.

        Args:
            z: Latent representation (B, latent_dim)

        Returns:
            Reconstructed point cloud (B, N*C) where C is 3 or 6
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder.

        Args:
            x: Flattened point cloud (B, N*C) where C is 3 or 6

        Returns:
            Tuple of (latent, reconstruction)
        """
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon


class SimpleCompressionModel(nn.Module):
    """Wrapper around SimpleAutoencoder for compression interface.

    Provides compress/decompress methods compatible with the BaseCompressionModel
    interface, but using the simple autoencoder without normalization.
    """

    def __init__(
        self,
        num_points: int = 2048,
        latent_dim: int = 512,
        hidden_dims: list[int] | None = None,
        compress_colors: bool = False,
        name: str = "simple_autoencoder",
    ) -> None:
        """Initialize the simple compression model.

        Args:
            num_points: Number of points in the point cloud
            latent_dim: Dimension of the latent representation
            hidden_dims: List of hidden layer dimensions
            compress_colors: Whether to compress colors along with geometry
            name: Model name
        """
        super().__init__()
        self.name = name
        self.num_points = num_points
        self.compress_colors = compress_colors

        self.autoencoder = SimpleAutoencoder(
            num_points=num_points,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            compress_colors=compress_colors,
        )

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
            Compressed latent representation (B, latent_dim)
        """
        # Handle single sample (no batch dimension)
        if geometry.dim() == 2:
            geometry = geometry.unsqueeze(0)
            if colors is not None:
                colors = colors.unsqueeze(0)

        batch_size = geometry.shape[0]
        x = geometry.reshape(batch_size, -1)

        if self.compress_colors:
            if colors is None:
                raise ValueError("Colors required when compress_colors=True")
            colors_flat = colors.reshape(batch_size, -1)
            x = torch.cat([x, colors_flat], dim=1)

        return self.autoencoder.encode(x)

    def decompress(
        self,
        compressed: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decompress latent representation to point cloud.

        Args:
            compressed: Compressed latent representation (B, latent_dim)

        Returns:
            Tuple of (reconstructed_geometry, reconstructed_colors)
        """
        recon = self.autoencoder.decode(compressed)

        if self.compress_colors:
            geom_size = self.num_points * 3
            geom_flat = recon[:, :geom_size]
            colors_flat = recon[:, geom_size:]

            geometry = geom_flat.reshape(-1, self.num_points, 3)
            colors = colors_flat.reshape(-1, self.num_points, 3)
            return geometry, colors
        else:
            geometry = recon.reshape(-1, self.num_points, 3)
            return geometry, None

    def forward(
        self,
        geometry: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Full compression-decompression cycle.

        Args:
            geometry: Point cloud coordinates (B, N, 3) or (N, 3)
            colors: Optional point colors (B, N, 3) or (N, 3)

        Returns:
            Tuple of (compressed, recon_geometry, recon_colors)
        """
        compressed = self.compress(geometry, colors)
        recon_geometry, recon_colors = self.decompress(compressed)
        return compressed, recon_geometry, recon_colors
