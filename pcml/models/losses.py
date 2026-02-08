"""Loss functions for point cloud compression training.

References:
    Chamfer distance: Barrow et al., "Parametric correspondence and chamfer matching,"
        IJCAI 1977.
    Used widely in point cloud deep learning, e.g., Fan et al., "A Point Set Generation
        Network for 3D Object Reconstruction from a Single Image," CVPR 2017.

See .docs/BIBLIOGRAPHY.md for full citations.
"""

from typing import Optional

import torch
import torch.nn as nn


class ChamferDistance(nn.Module):
    """Chamfer Distance loss for point cloud reconstruction.

    Computes the average minimum distance between two point sets in both directions.
    This is the standard loss for point cloud autoencoder training.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize Chamfer distance loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, return_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Chamfer distance between predicted and target point clouds.

        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            return_components: If True, return (total, forward, backward)

        Returns:
            Chamfer distance loss, or tuple if return_components=True
        """
        batch_size = pred.shape[0]

        pred_expanded = pred.unsqueeze(2)
        target_expanded = target.unsqueeze(1)

        distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=-1)

        min_dist_pred_to_target = torch.min(distances, dim=2)[0]
        min_dist_target_to_pred = torch.min(distances, dim=1)[0]

        forward_loss = min_dist_pred_to_target.mean(dim=1)
        backward_loss = min_dist_target_to_pred.mean(dim=1)

        total_loss = forward_loss + backward_loss

        if self.reduction == "mean":
            total_loss = total_loss.mean()
            forward_loss = forward_loss.mean()
            backward_loss = backward_loss.mean()
        elif self.reduction == "sum":
            total_loss = total_loss.sum()
            forward_loss = forward_loss.sum()
            backward_loss = backward_loss.sum()

        if return_components:
            return total_loss, forward_loss, backward_loss

        return total_loss


class PointCloudReconstructionLoss(nn.Module):
    """Combined loss for point cloud reconstruction.

    Combines Chamfer distance with optional additional losses like point-wise MSE.
    """

    def __init__(
        self,
        chamfer_weight: float = 1.0,
        mse_weight: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize reconstruction loss.

        Args:
            chamfer_weight: Weight for Chamfer distance
            mse_weight: Weight for point-wise MSE (only if point counts match)
            reduction: Reduction method
        """
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.mse_weight = mse_weight
        self.reduction = reduction

        self.chamfer = ChamferDistance(reduction=reduction)
        if mse_weight > 0:
            self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined reconstruction loss.

        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)

        Returns:
            Combined loss
        """
        loss = self.chamfer_weight * self.chamfer(pred, target)

        if self.mse_weight > 0 and pred.shape == target.shape:
            loss = loss + self.mse_weight * self.mse(pred, target)

        return loss


class ColorReconstructionLoss(nn.Module):
    """Loss for color reconstruction (RGB).

    Simple MSE loss in RGB color space.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize color reconstruction loss.

        Args:
            reduction: Reduction method
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self, pred_colors: torch.Tensor, target_colors: torch.Tensor
    ) -> torch.Tensor:
        """Compute color reconstruction loss.

        Args:
            pred_colors: Predicted colors (B, N, 3)
            target_colors: Target colors (B, N, 3)

        Returns:
            Color MSE loss
        """
        return self.mse(pred_colors, target_colors)
