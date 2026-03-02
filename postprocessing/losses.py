"""Loss functions for displacement-based post-processing.

Chamfer loss: dynamic re-matching via NN search each forward pass.
Variants: focal-weighted, edge-gated, with kNN Laplacian, with voxel Laplacian.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F


class VoxelLaplacian(torch.nn.Module):
    """Fixed-weight sparse 3D Laplacian operator.

    Implements the discrete Laplacian as a non-learnable MinkowskiConvolution
    with kernel_size=3. Center weight = -6, 6 face-neighbors = +1.
    Gradients flow through the input features, not through the operator.
    """

    def __init__(self):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            bias=False,
            dimension=3,
        )
        # Freeze weights
        self.conv.kernel.requires_grad_(False)
        # Set Laplacian stencil: per-channel identity with spatial Laplacian
        # Kernel shape: (27, in_ch, out_ch) for 3x3x3
        with torch.no_grad():
            self.conv.kernel.zero_()
            # 3x3x3 lexicographic: face neighbors at indices 4,10,12,14,16,22
            # center at index 13
            face_idx = [4, 10, 12, 14, 16, 22]
            for c in range(3):
                self.conv.kernel[13, c, c] = -6.0
                for fi in face_idx:
                    self.conv.kernel[fi, c, c] = 1.0

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.conv(x)


class KendallUncertaintyWeights(torch.nn.Module):
    """Learned multi-task loss balancing via homoscedastic uncertainty.

    Kendall et al. (CVPR 2018): each task's weight is 1/(2*sigma^2), with
    a log-variance regularizer preventing sigma -> inf. Parameterized as
    log_var = log(sigma^2) for numerical stability.
    """

    def __init__(self, n_tasks: int = 2):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total

    def weights(self):
        """Return current effective weights for logging."""
        return [0.5 * torch.exp(-lv).item() for lv in self.log_vars]


COORD_SCALE = 1023.0


def dynamic_chamfer_loss(
    pred: ME.SparseTensor,
    original_patches: list,
    fwd_weight: float = 1.0,
    rev_weight: float = 1.0,
) -> tuple:
    """Chamfer distance between refined points and actual original cloud crops.

    No precomputed displacement targets. NN re-matched every forward pass.
    Uses squared distances (standard in PU-Net, Dis-PU).

    Args:
        pred: SparseTensor with predicted offsets as features (N_total, 3)
        original_patches: list of (M_i, 3) float tensors, one per batch element

    Returns:
        (loss, extras_dict) for consistency with other loss functions.
    """
    batch_idx = pred.C[:, 0]
    decoded = pred.C[:, 1:].float()  # integer voxel coords
    offsets = pred.F  # predicted displacement
    refined = decoded + offsets

    unique_batches = batch_idx.unique()
    n_patches = len(unique_batches)

    total_fwd = 0.0
    total_rev = 0.0

    for i, b in enumerate(unique_batches):
        mask = batch_idx == b
        refined_b = refined[mask]  # (N_b, 3)
        orig_b = original_patches[i].to(refined_b.device)  # (M_b, 3)

        if orig_b.shape[0] == 0:
            continue

        dists_sq = torch.cdist(refined_b, orig_b).pow(2)  # (N_b, M_b)
        fwd = dists_sq.min(dim=1).values.mean()  # each refined -> nearest orig
        rev = dists_sq.min(dim=0).values.mean()  # each orig -> nearest refined
        total_fwd += fwd
        total_rev += rev

    loss = (fwd_weight * total_fwd + rev_weight * total_rev) / max(n_patches, 1)
    extras = {
        "cd_fwd": float(total_fwd) / max(n_patches, 1) if n_patches > 0 else 0,
        "cd_rev": float(total_rev) / max(n_patches, 1) if n_patches > 0 else 0,
    }
    return loss, extras


def cd_vlap_loss(
    pred: ME.SparseTensor,
    original_patches: list,
    lap_op: VoxelLaplacian,
    kendall: KendallUncertaintyWeights,
    fwd_weight: float = 1.0,
    rev_weight: float = 1.0,
) -> tuple:
    """Two-stage loss: CD + voxel-graph Laplacian with Kendall balancing.

    CD provides global reconstruction. Voxel Laplacian provides edge sharpness
    via a fixed sparse convolution -- gradients flow through features directly,
    no dynamic graph or kNN needed.
    """
    batch_idx = pred.C[:, 0]
    decoded = pred.C[:, 1:].float()
    offsets = pred.F
    refined = decoded + offsets

    unique_batches = batch_idx.unique()
    n_patches = len(unique_batches)

    # --- CD term (same as dynamic_chamfer_loss) ---
    total_fwd = 0.0
    total_rev = 0.0
    # Collect NN-matched target positions for Laplacian
    target_positions = torch.empty_like(refined)

    for i, b in enumerate(unique_batches):
        mask = batch_idx == b
        refined_b = refined[mask]
        decoded_b = decoded[mask]
        orig_b = original_patches[i].to(refined_b.device)

        if orig_b.shape[0] == 0:
            continue

        dists_sq = torch.cdist(refined_b, orig_b).pow(2)
        total_fwd += dists_sq.min(dim=1).values.mean()
        total_rev += dists_sq.min(dim=0).values.mean()

        # NN of decoded in original -> target positions for Laplacian
        dec_orig_dists = torch.cdist(decoded_b, orig_b)
        nn_idx = dec_orig_dists.argmin(dim=1)
        target_positions[mask] = orig_b[nn_idx].detach()

    cd_loss = (fwd_weight * total_fwd + rev_weight * total_rev) / max(n_patches, 1)

    # --- Voxel Laplacian term ---
    # Build SparseTensors with position features on the decoded grid
    refined_st = ME.SparseTensor(
        features=refined,
        coordinate_map_key=pred.coordinate_map_key,
        coordinate_manager=pred.coordinate_manager,
    )
    target_st = ME.SparseTensor(
        features=target_positions,
        coordinate_map_key=pred.coordinate_map_key,
        coordinate_manager=pred.coordinate_manager,
    )

    lap_refined = lap_op(refined_st)
    lap_target = lap_op(target_st)
    vlap_loss = (lap_refined.F - lap_target.F).pow(2).mean()

    # Kendall auto-balancing
    total = kendall(cd_loss, vlap_loss)

    extras = {
        "cd_fwd": float(total_fwd) / max(n_patches, 1) if n_patches > 0 else 0,
        "cd_rev": float(total_rev) / max(n_patches, 1) if n_patches > 0 else 0,
        "vlap": vlap_loss.item(),
        "kw_cd": kendall.weights()[0],
        "kw_vlap": kendall.weights()[1],
    }
    return total, extras


def cd_laplacian_loss(
    pred: ME.SparseTensor,
    original_patches: list,
    curvature: torch.Tensor,
    fwd_weight: float = 1.0,
    rev_weight: float = 1.0,
    lambda_lap: float = 1.0,
    lap_k: int = 8,
    alpha: float = 3.0,
) -> tuple:
    """Composite loss: Chamfer distance + Laplacian preservation.

    CD provides stable global reconstruction. Laplacian explicitly targets
    oversmoothing by matching refined sharpness to original sharpness.
    NN mapping for Laplacian targets computed inline from original patches.
    """
    batch_idx = pred.C[:, 0]
    decoded = pred.C[:, 1:].float()
    offsets = pred.F
    refined = decoded + offsets

    unique_batches = batch_idx.unique()
    n_patches = len(unique_batches)

    total_fwd = 0.0
    total_rev = 0.0
    total_lap = 0.0

    for i, b in enumerate(unique_batches):
        mask = batch_idx == b
        decoded_b = decoded[mask]
        refined_b = refined[mask]
        orig_b = original_patches[i].to(refined_b.device)
        N_b = decoded_b.shape[0]

        if orig_b.shape[0] == 0:
            continue

        # --- CD term ---
        dists_sq = torch.cdist(refined_b, orig_b).pow(2)
        total_fwd += dists_sq.min(dim=1).values.mean()
        total_rev += dists_sq.min(dim=0).values.mean()

        # --- Laplacian term ---
        # NN of decoded in original -> target positions
        dec_orig_dists = torch.cdist(decoded_b, orig_b)
        nn_idx = dec_orig_dists.argmin(dim=1)
        target_b = orig_b[nn_idx].detach()

        # kNN graph from decoded positions
        k = min(lap_k, N_b - 1)
        dec_self_dists = torch.cdist(
            decoded_b.unsqueeze(0), decoded_b.unsqueeze(0)
        ).squeeze(0)
        _, knn_idx = dec_self_dists.topk(k + 1, dim=1, largest=False)
        knn_idx = knn_idx[:, 1:]  # exclude self

        # Laplacian = point - mean(neighbors)
        lap_ref = refined_b - refined_b[knn_idx].mean(dim=1)
        lap_tgt = target_b - target_b[knn_idx].mean(dim=1)

        curv_b = curvature[mask]
        curv_w = 1.0 + alpha * curv_b
        lap_diff = torch.abs(lap_ref - lap_tgt)
        total_lap += (curv_w.unsqueeze(-1) * lap_diff).mean()

    n = max(n_patches, 1)
    cd_loss = (fwd_weight * total_fwd + rev_weight * total_rev) / n
    lap_loss = total_lap / n
    loss = cd_loss + lambda_lap * lap_loss

    extras = {
        "cd_fwd": float(total_fwd) / n,
        "cd_rev": float(total_rev) / n,
        "lap": float(lap_loss),
    }
    return loss, extras


def focal_chamfer_loss(
    pred: ME.SparseTensor,
    original_patches: list,
    fwd_weight: float = 1.0,
    rev_weight: float = 1.0,
    focal_gamma: float = 2.0,
) -> tuple:
    """Chamfer distance with focal weighting -- hard examples get more gradient.

    Points with higher CD error (naturally edges) are upweighted by
    (error / max_error)^gamma. Weights are detached so they only modulate
    loss magnitude, not gradient direction.
    """
    batch_idx = pred.C[:, 0]
    decoded = pred.C[:, 1:].float()
    offsets = pred.F
    refined = decoded + offsets

    unique_batches = batch_idx.unique()
    n_patches = len(unique_batches)

    total_fwd = 0.0
    total_rev = 0.0

    for i, b in enumerate(unique_batches):
        mask = batch_idx == b
        refined_b = refined[mask]
        orig_b = original_patches[i].to(refined_b.device)

        if orig_b.shape[0] == 0:
            continue

        dists_sq = torch.cdist(refined_b, orig_b).pow(2)

        # Forward: each refined -> nearest orig
        fwd_min_sq = dists_sq.min(dim=1).values
        fwd_max = fwd_min_sq.detach().max().clamp(min=1e-8)
        fwd_w = (fwd_min_sq.detach() / fwd_max).pow(focal_gamma)
        total_fwd += (fwd_w * fwd_min_sq).mean()

        # Reverse: each orig -> nearest refined
        rev_min_sq = dists_sq.min(dim=0).values
        rev_max = rev_min_sq.detach().max().clamp(min=1e-8)
        rev_w = (rev_min_sq.detach() / rev_max).pow(focal_gamma)
        total_rev += (rev_w * rev_min_sq).mean()

    loss = (fwd_weight * total_fwd + rev_weight * total_rev) / max(n_patches, 1)
    extras = {
        "cd_fwd": float(total_fwd) / max(n_patches, 1) if n_patches > 0 else 0,
        "cd_rev": float(total_rev) / max(n_patches, 1) if n_patches > 0 else 0,
    }
    return loss, extras


def edge_gated_chamfer_loss(
    pred: ME.SparseTensor,
    original_patches: list,
    curvature: torch.Tensor,
    fwd_weight: float = 1.0,
    rev_weight: float = 1.0,
    lambda_flat: float = 1.0,
    edge_beta: float = 10.0,
    edge_tau: float = 0.33,
) -> tuple:
    """Chamfer distance with edge-gated forward/reverse and flat displacement penalty.

    Curvature is normalized per-patch to [0,1] so gating is consistent across
    clouds and rates. Forward CD weighted by edge gate. Reverse CD weighted by
    gate of the nearest refined point (focus coverage on edge originals).
    Flat penalty suppresses displacement where curvature is low.
    """
    batch_idx = pred.C[:, 0]
    decoded = pred.C[:, 1:].float()
    offsets = pred.F
    refined = decoded + offsets

    unique_batches = batch_idx.unique()
    n_patches = len(unique_batches)

    total_fwd = 0.0
    total_rev = 0.0
    # Collect per-patch gates for flat penalty
    all_w = torch.empty_like(curvature)

    for i, b in enumerate(unique_batches):
        mask = batch_idx == b
        refined_b = refined[mask]
        orig_b = original_patches[i].to(refined_b.device)
        curv_b = curvature[mask]

        # Per-patch curvature normalization to [0,1]
        c_min = curv_b.min()
        c_max = curv_b.max()
        c_range = (c_max - c_min).clamp(min=1e-8)
        curv_norm = (curv_b - c_min) / c_range

        w_b = torch.sigmoid(edge_beta * (curv_norm - edge_tau))
        all_w[mask] = w_b

        if orig_b.shape[0] == 0:
            continue

        dists_sq = torch.cdist(refined_b, orig_b).pow(2)

        # Forward: each refined -> nearest orig, edge-weighted
        fwd_min_sq = dists_sq.min(dim=1).values  # (N_b,)
        total_fwd += (w_b * fwd_min_sq).mean()

        # Reverse: unweighted -- all original points should be covered
        rev_min_sq = dists_sq.min(dim=0).values  # (M_b,)
        total_rev += rev_min_sq.mean()

    cd_loss = (fwd_weight * total_fwd + rev_weight * total_rev) / max(n_patches, 1)

    # Flat displacement penalty: penalize movement where curvature is low
    flat_penalty = ((1.0 - all_w) * offsets.pow(2).sum(dim=-1)).mean()

    loss = cd_loss + lambda_flat * flat_penalty

    edge_pct = (all_w > 0.5).float().mean().item() * 100.0
    extras = {
        "cd_fwd": float(total_fwd) / max(n_patches, 1) if n_patches > 0 else 0,
        "cd_rev": float(total_rev) / max(n_patches, 1) if n_patches > 0 else 0,
        "flat_penalty": flat_penalty.item(),
        "edge_pct": edge_pct,
    }
    return loss, extras


# ---------------------------------------------------------------------------
# Loss dispatch: typed context + registry of LossFunction objects
# ---------------------------------------------------------------------------


@dataclass
class LossContext:
    """Typed context built from batch data. Passed to every LossFunction."""

    curvature: torch.Tensor = None
    original_patches: Optional[List[torch.Tensor]] = None
    gt_displacement: Optional[torch.Tensor] = None
    input_sparse: Optional[ME.SparseTensor] = None
    kendall: Optional[KendallUncertaintyWeights] = None
    lap_op: Optional[VoxelLaplacian] = None


class LossFunction:
    """Base class for all loss functions. Subclass and override __call__."""

    needs_chamfer: bool = False
    needs_kendall: bool = False
    needs_lap_op: bool = False

    def __call__(
        self, pred: ME.SparseTensor, lc, ctx: LossContext
    ) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError


class DynamicChamferLoss(LossFunction):
    needs_chamfer = True

    def __call__(self, pred, lc, ctx):
        return dynamic_chamfer_loss(
            pred,
            ctx.original_patches,
            fwd_weight=lc.cd_fwd_weight,
            rev_weight=lc.cd_rev_weight,
        )


class FocalChamferLoss(LossFunction):
    needs_chamfer = True

    def __call__(self, pred, lc, ctx):
        return focal_chamfer_loss(
            pred,
            ctx.original_patches,
            fwd_weight=lc.cd_fwd_weight,
            rev_weight=lc.cd_rev_weight,
            focal_gamma=lc.focal_gamma,
        )


class EdgeGatedChamferLoss(LossFunction):
    needs_chamfer = True

    def __call__(self, pred, lc, ctx):
        return edge_gated_chamfer_loss(
            pred,
            ctx.original_patches,
            ctx.curvature,
            fwd_weight=lc.cd_fwd_weight,
            rev_weight=lc.cd_rev_weight,
            lambda_flat=lc.lambda_flat,
            edge_beta=lc.edge_beta,
            edge_tau=lc.edge_tau,
        )


class CDLaplacianLoss(LossFunction):
    needs_chamfer = True

    def __call__(self, pred, lc, ctx):
        return cd_laplacian_loss(
            pred,
            ctx.original_patches,
            ctx.curvature,
            fwd_weight=lc.cd_fwd_weight,
            rev_weight=lc.cd_rev_weight,
            lambda_lap=lc.lambda_lap,
            lap_k=lc.laplacian_k,
            alpha=lc.alpha,
        )


class CDVoxelLaplacianLoss(LossFunction):
    needs_chamfer = True
    needs_kendall = True
    needs_lap_op = True

    def __call__(self, pred, lc, ctx):
        return cd_vlap_loss(
            pred,
            ctx.original_patches,
            ctx.lap_op,
            ctx.kendall,
            fwd_weight=lc.cd_fwd_weight,
            rev_weight=lc.cd_rev_weight,
        )


_REGISTRY: Dict[str, LossFunction] = {
    "cd": DynamicChamferLoss(),
    "focal_cd": FocalChamferLoss(),
    "edge_cd": EdgeGatedChamferLoss(),
    "cd_lap": CDLaplacianLoss(),
    "cd_vlap": CDVoxelLaplacianLoss(),
}


def get_loss_fn(loss_type: str) -> LossFunction:
    """Look up a LossFunction by name. Raises ValueError for unknown types."""
    if loss_type not in _REGISTRY:
        raise ValueError(
            f"Unknown loss_type: {loss_type!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[loss_type]
