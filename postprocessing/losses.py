"""Loss functions for displacement-based post-processing.

Curvature-weighted L1 with magnitude weighting: up-weights high-curvature (edge)
points AND non-zero displacement targets so the model cannot collapse to the
trivial "predict zero" solution.
"""

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F


def curvature_weighted_l1_loss(
    pred: ME.SparseTensor,
    gt_displacement: torch.Tensor,
    curvature: torch.Tensor,
    alpha: float = 10.0,
    mag_floor: float = 0.1,
) -> torch.Tensor:
    """L1 displacement loss with curvature and magnitude weighting.

    Curvature weight: 1 + alpha * curvature_i
    Magnitude weight: 1.0 for non-zero targets, mag_floor for zero targets.
    Combined multiplicatively so non-zero edge points get maximum attention.
    """
    curv_w = 1.0 + alpha * curvature.unsqueeze(-1)  # (N, 1)

    # Magnitude weighting: 10x upweight for non-zero displacement targets
    gt_mag = gt_displacement.norm(dim=-1, keepdim=True)  # (N, 1)
    nonzero_mask = (gt_mag > 1e-6).float()
    mag_w = mag_floor + (1.0 - mag_floor) * nonzero_mask  # (N, 1)

    per_point_l1 = torch.abs(pred.F - gt_displacement)  # (N, 3)
    return (curv_w * mag_w * per_point_l1).mean()
