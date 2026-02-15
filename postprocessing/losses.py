"""Loss functions for displacement-based post-processing.

Curvature-weighted L1: up-weights high-curvature (edge) points so the model
cannot collapse to the trivial "predict zero" solution that minimizes unweighted
L1 on the dominant flat regions.
"""

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F


def curvature_weighted_l1_loss(
    pred: ME.SparseTensor,
    gt_displacement: torch.Tensor,
    curvature: torch.Tensor,
    alpha: float = 10.0,
) -> torch.Tensor:
    """L1 displacement loss with per-point curvature weighting.

    weight_i = 1 + alpha * curvature_i
    High-curvature points get up to (1+alpha)x more weight.
    """
    weights = 1.0 + alpha * curvature.unsqueeze(-1)  # (N, 1)
    per_point_l1 = torch.abs(pred.F - gt_displacement)  # (N, 3)
    return (weights * per_point_l1).mean()
