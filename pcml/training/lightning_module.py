"""PyTorch Lightning module for point cloud compression training.

References:
    PyTorch Lightning: Falcon et al., "PyTorch Lightning," 2019.
        https://lightning.ai/docs/pytorch/

See .docs/BIBLIOGRAPHY.md for full citations.
"""

from typing import Any, Optional

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pcml.metrics.quality import GeometryQualityCalculator
from pcml.models.baseline import BaselineCompressionModel
from pcml.models.losses import ColorReconstructionLoss, PointCloudReconstructionLoss


class CompressionLightningModule(pl.LightningModule):
    """Lightning module for training point cloud compression models.

    This wraps a compression model with training/validation logic, metrics tracking,
    and automatic checkpointing.
    """

    def __init__(
        self,
        num_points: int = 2048,
        latent_dim: int = 512,
        hidden_dims: list[int] | None = None,
        compress_colors: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        chamfer_weight: float = 1.0,
        mse_weight: float = 0.0,
    ) -> None:
        """Initialize the Lightning module.

        Args:
            num_points: Number of points in point cloud
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            compress_colors: Whether to compress colors
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            chamfer_weight: Weight for Chamfer distance loss
            mse_weight: Weight for point-wise MSE loss
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = BaselineCompressionModel(
            num_points=num_points,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            compress_colors=compress_colors,
        )

        self.geometry_loss = PointCloudReconstructionLoss(
            chamfer_weight=chamfer_weight,
            mse_weight=mse_weight,
        )

        if compress_colors:
            self.color_loss = ColorReconstructionLoss()
        else:
            self.color_loss = None

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(
        self, geometry: torch.Tensor, colors: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            geometry: Input point cloud geometry (B, N, 3)
            colors: Optional input colors (B, N, 3)

        Returns:
            Tuple of (compressed, reconstructed_geometry, reconstructed_colors)
        """
        return self.model(geometry, colors)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary with 'geometry' and optionally 'colors'
            batch_idx: Batch index

        Returns:
            Training loss
        """
        geometry = batch["geometry"]
        colors = batch.get("colors", None)

        compressed, recon_geometry, recon_colors = self(geometry, colors)

        geom_loss = self.geometry_loss(recon_geometry, geometry)
        loss = geom_loss

        self.log(
            "train/geometry_loss", geom_loss, prog_bar=True, on_step=True, on_epoch=True
        )

        if colors is not None and recon_colors is not None:
            color_loss = self.color_loss(recon_colors, colors)
            loss = loss + color_loss
            self.log("train/color_loss", color_loss, on_step=True, on_epoch=True)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        bpp = (compressed.numel() * 4 * 8) / (geometry.shape[0] * geometry.shape[1])
        self.log("train/bpp", bpp, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, Any]:
        """Validation step.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index

        Returns:
            Dictionary of validation metrics
        """
        geometry = batch["geometry"]
        colors = batch.get("colors", None)

        compressed, recon_geometry, recon_colors = self(geometry, colors)

        geom_loss = self.geometry_loss(recon_geometry, geometry)
        loss = geom_loss

        self.log("val/geometry_loss", geom_loss, prog_bar=True, on_epoch=True)

        if colors is not None and recon_colors is not None:
            color_loss = self.color_loss(recon_colors, colors)
            loss = loss + color_loss
            self.log("val/color_loss", color_loss, on_epoch=True)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        bpp = (compressed.numel() * 4 * 8) / (geometry.shape[0] * geometry.shape[1])
        self.log("val/bpp", bpp, on_epoch=True)

        if batch_idx == 0:
            geom_np = geometry[0].cpu().numpy()
            recon_np = recon_geometry[0].detach().cpu().numpy()

            metrics = GeometryQualityCalculator.calculate_all(geom_np, recon_np)
            self.log("val/psnr", metrics.psnr, on_epoch=True)
            self.log("val/mse", metrics.mse, on_epoch=True)

        return {"val_loss": loss}

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
