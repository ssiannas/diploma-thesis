"""Loss functions for displacement-based post-processing.

Curvature-weighted L1: direct displacement supervision with edge upweighting.

Chamfer loss: dynamic re-matching via NN search each forward pass.

Laplacian preservation loss: directly targets oversmoothing by matching the
discrete Laplacian (local curvature) of the refined cloud to the original.
Neighborhood graph is fixed from decoded positions, making it fully
differentiable without re-matching.

Stratified loss: soft sigmoid gating splits Laplacian (edges) from zero-displacement
penalty (flat) with Kendall learned uncertainty weighting for automatic balancing.
"""

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F


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


def curvature_weighted_l1_loss(
    pred: ME.SparseTensor,
    gt_displacement: torch.Tensor,
    curvature: torch.Tensor,
    alpha: float = 10.0,
    mag_floor: float = 0.1,
    smooth_l1_beta: float = 0.0,
    lambda_shrink: float = 0.0,
    shrink_gamma: float = 0.0,
    edge_threshold: float = 0.0,
) -> torch.Tensor:
    """Displacement loss with optional edge masking and curvature-gated shrinkage.

    When edge_threshold > 0: displacement loss only on edge points
    (curvature > threshold), shrinkage only on flat points. No conflicting
    gradients -- flat regions get "predict zero", edges get "match GT".
    """
    if smooth_l1_beta > 0:
        per_point = F.smooth_l1_loss(
            pred.F, gt_displacement, reduction="none", beta=smooth_l1_beta
        )
    else:
        per_point = torch.abs(pred.F - gt_displacement)

    if edge_threshold > 0:
        # Hard edge mask: displacement loss only on high-curvature points
        edge_mask = curvature > edge_threshold  # (N,)
        if edge_mask.any():
            loss = per_point[edge_mask].mean()
        else:
            loss = torch.tensor(0.0, device=pred.F.device)
    else:
        # Legacy: full weighted displacement loss
        curv_w = 1.0 + alpha * curvature.unsqueeze(-1)
        gt_mag = gt_displacement.norm(dim=-1, keepdim=True)
        nonzero_mask = (gt_mag > 1e-6).float()
        mag_w = mag_floor + (1.0 - mag_floor) * nonzero_mask
        loss = (curv_w * mag_w * per_point).mean()

    if lambda_shrink > 0:
        if shrink_gamma > 0:
            flat_w = torch.exp(-shrink_gamma * curvature)  # (N,)
            shrink = (flat_w * pred.F.pow(2).sum(dim=-1)).mean()
        else:
            gt_mag = gt_displacement.norm(dim=-1)
            zero_mask = gt_mag < 1e-6
            if zero_mask.any():
                shrink = pred.F[zero_mask].pow(2).mean()
            else:
                shrink = torch.tensor(0.0, device=pred.F.device)
        loss = loss + lambda_shrink * shrink

    return loss


def min_of_k_displacement_loss(
    pred: ME.SparseTensor,
    gt_displacement_k: torch.Tensor,
    curvature: torch.Tensor,
    alpha: float = 0.0,
    smooth_l1_beta: float = 1.0,
    lambda_shrink: float = 0.0,
) -> torch.Tensor:
    """Min-of-K displacement loss for ambiguous NN targets.

    For each point, computes loss against K candidate displacements and takes
    the minimum. At edges where NN jumps between sides of a sharp feature,
    the network only needs to match whichever side is geometrically consistent.

    Args:
        gt_displacement_k: (N, K, 3) K candidate displacements per point
    """
    K = gt_displacement_k.shape[1]
    pred_expanded = pred.F.unsqueeze(1).expand_as(gt_displacement_k)  # (N, K, 3)

    if smooth_l1_beta > 0:
        per_k = F.smooth_l1_loss(
            pred_expanded, gt_displacement_k, reduction="none", beta=smooth_l1_beta
        )  # (N, K, 3)
    else:
        per_k = torch.abs(pred_expanded - gt_displacement_k)

    per_k_loss = per_k.sum(dim=-1)  # (N, K) -- sum over xyz, keep K
    min_loss = per_k_loss.min(dim=1).values  # (N,) -- min over K candidates

    curv_w = 1.0 + alpha * curvature  # (N,)
    loss = (curv_w * min_loss).mean()

    if lambda_shrink > 0:
        # For shrinkage: a point has zero GT if ALL K candidates are zero
        all_zero = (gt_displacement_k.norm(dim=-1) < 1e-6).all(dim=1)  # (N,)
        if all_zero.any():
            loss = loss + lambda_shrink * pred.F[all_zero].pow(2).mean()

    return loss


COORD_SCALE = 1023.0


def _batched_knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """Compute kNN indices within each patch of a batched point cloud.

    Args:
        points: (B, M, 3) decoded coordinates
        k: number of neighbors (excluding self)

    Returns:
        knn_idx: (B, M, k) neighbor indices into the M dimension
    """
    dists = torch.cdist(points, points)  # (B, M, M)
    # topk smallest distances; index 0 is self (dist=0), so take k+1 and skip first
    _, idx = dists.topk(k + 1, dim=2, largest=False)
    return idx[:, :, 1:]  # (B, M, k)


def _gather_neighbors(points: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbor positions using kNN indices.

    Args:
        points: (B, M, 3)
        knn_idx: (B, M, k)

    Returns:
        neighbors: (B, M, k, 3)
    """
    B, M, k = knn_idx.shape
    flat_idx = knn_idx.reshape(B, M * k)  # (B, M*k)
    flat_idx_3d = flat_idx.unsqueeze(-1).expand(-1, -1, 3)  # (B, M*k, 3)
    return torch.gather(points, 1, flat_idx_3d).reshape(B, M, k, 3)


def laplacian_loss(
    pred: ME.SparseTensor,
    input_sparse: ME.SparseTensor,
    gt_displacement: torch.Tensor,
    curvature: torch.Tensor,
    alpha: float = 3.0,
    k: int = 8,
) -> torch.Tensor:
    """Laplacian preservation loss -- directly targets oversmoothing.

    Computes discrete Laplacian L(p_i) = p_i - mean(neighbors(p_i)) on a fixed
    neighborhood graph from decoded positions. Penalizes the difference between
    the refined cloud's Laplacian and the original cloud's Laplacian.

    Unlike displacement or Chamfer losses, this does not depend on point-to-point
    correspondence. It measures local geometric sharpness directly: oversmoothed
    regions have attenuated Laplacians, so minimizing |L_refined - L_original|
    drives the model to restore high-frequency surface detail.

    The neighborhood graph is built once from decoded positions and reused for
    both refined and original Laplacian computation, making the loss fully
    differentiable without dynamic re-matching.
    """
    decoded = input_sparse.F[:, :3] * COORD_SCALE  # (N, 3)
    refined = decoded + pred.F  # (N, 3), has grad
    original = (decoded + gt_displacement).detach()  # (N, 3), no grad

    batch_idx = pred.C[:, 0]
    curv_w = 1.0 + alpha * curvature  # (N,)

    unique_batches, counts = batch_idx.unique(return_counts=True)
    n_patches = unique_batches.shape[0]

    if n_patches == 0:
        return torch.tensor(0.0, device=pred.F.device)

    # Fast path: equal-size patches
    if counts.min() == counts.max():
        M = counts[0].item()
        k_actual = min(k, M - 1)

        dec_3d = decoded.view(n_patches, M, 3)
        ref_3d = refined.view(n_patches, M, 3)
        orig_3d = original.view(n_patches, M, 3)
        w_3d = curv_w.view(n_patches, M)

        # Fixed kNN graph from decoded positions
        knn_idx = _batched_knn(dec_3d, k_actual)  # (B, M, k)

        # Laplacian = point - mean(neighbors)
        ref_neighbors = _gather_neighbors(ref_3d, knn_idx)  # (B, M, k, 3)
        orig_neighbors = _gather_neighbors(orig_3d, knn_idx)  # (B, M, k, 3)

        lap_refined = ref_3d - ref_neighbors.mean(dim=2)  # (B, M, 3)
        lap_original = orig_3d - orig_neighbors.mean(dim=2)  # (B, M, 3)

        # Curvature-weighted L1 on Laplacian difference
        lap_diff = torch.abs(lap_refined - lap_original)  # (B, M, 3)
        return (w_3d.unsqueeze(-1) * lap_diff).mean()

    # Slow fallback: variable-size patches
    total_loss = torch.tensor(0.0, device=pred.F.device)
    for b in unique_batches:
        mask = batch_idx == b
        dec_b = decoded[mask].unsqueeze(0)  # (1, M, 3)
        ref_b = refined[mask].unsqueeze(0)
        orig_b = original[mask].unsqueeze(0)
        w_b = curv_w[mask]
        M_b = dec_b.shape[1]
        k_b = min(k, M_b - 1)

        knn_idx = _batched_knn(dec_b, k_b)
        ref_nb = _gather_neighbors(ref_b, knn_idx)
        orig_nb = _gather_neighbors(orig_b, knn_idx)

        lap_ref = ref_b.squeeze(0) - ref_nb.squeeze(0).mean(dim=1)
        lap_orig = orig_b.squeeze(0) - orig_nb.squeeze(0).mean(dim=1)

        lap_diff = torch.abs(lap_ref - lap_orig)
        total_loss = total_loss + (w_b.unsqueeze(-1) * lap_diff).mean()

    return total_loss / max(n_patches, 1)


def stratified_loss(
    pred: ME.SparseTensor,
    input_sparse: ME.SparseTensor,
    gt_displacement: torch.Tensor,
    curvature: torch.Tensor,
    kendall: KendallUncertaintyWeights,
    alpha: float = 3.0,
    k: int = 8,
    beta: float = 5.0,
    tau: float = 0.33,
) -> tuple:
    """Soft-gated stratified loss: Laplacian on edges, zero-penalty on flat.

    Uses sigmoid gating on curvature to smoothly split points into edge/flat:
        w_i = sigmoid(beta * (kappa_i - tau))
    Edge term: Laplacian preservation (restore sharpness)
    Flat term: ||displacement||^2 penalty (don't move)
    Balanced via Kendall learned uncertainty weighting.

    Returns (total_loss, edge_loss, flat_loss) for logging.
    """
    decoded = input_sparse.F[:, :3] * COORD_SCALE  # (N, 3)
    refined = decoded + pred.F
    original = (decoded + gt_displacement).detach()

    batch_idx = pred.C[:, 0]
    unique_batches, counts = batch_idx.unique(return_counts=True)
    n_patches = unique_batches.shape[0]

    # Soft gate: high w = edge, low w = flat
    w = torch.sigmoid(beta * (curvature - tau))  # (N,)

    # -- Flat term: zero-displacement penalty --
    disp_magnitude = pred.F.pow(2).sum(dim=-1)  # (N,)
    flat_loss = ((1.0 - w) * disp_magnitude).mean()

    # -- Edge term: Laplacian preservation --
    if n_patches == 0:
        edge_loss = torch.tensor(0.0, device=pred.F.device)
    elif counts.min() == counts.max():
        M = counts[0].item()
        k_actual = min(k, M - 1)

        dec_3d = decoded.view(n_patches, M, 3)
        ref_3d = refined.view(n_patches, M, 3)
        orig_3d = original.view(n_patches, M, 3)
        w_3d = w.view(n_patches, M)

        knn_idx = _batched_knn(dec_3d, k_actual)
        ref_neighbors = _gather_neighbors(ref_3d, knn_idx)
        orig_neighbors = _gather_neighbors(orig_3d, knn_idx)

        lap_refined = ref_3d - ref_neighbors.mean(dim=2)
        lap_original = orig_3d - orig_neighbors.mean(dim=2)

        lap_diff = torch.abs(lap_refined - lap_original)  # (B, M, 3)
        # Weight by soft gate (only edge points contribute)
        edge_loss = (w_3d.unsqueeze(-1) * lap_diff).mean()
    else:
        # Slow fallback
        edge_loss = torch.tensor(0.0, device=pred.F.device)
        for b in unique_batches:
            mask = batch_idx == b
            dec_b = decoded[mask].unsqueeze(0)
            ref_b = refined[mask].unsqueeze(0)
            orig_b = original[mask].unsqueeze(0)
            w_b = w[mask]
            M_b = dec_b.shape[1]
            k_b = min(k, M_b - 1)

            knn_idx = _batched_knn(dec_b, k_b)
            ref_nb = _gather_neighbors(ref_b, knn_idx)
            orig_nb = _gather_neighbors(orig_b, knn_idx)

            lap_ref = ref_b.squeeze(0) - ref_nb.squeeze(0).mean(dim=1)
            lap_orig = orig_b.squeeze(0) - orig_nb.squeeze(0).mean(dim=1)

            lap_diff = torch.abs(lap_ref - lap_orig)
            edge_loss = edge_loss + (w_b.unsqueeze(-1) * lap_diff).mean()
        edge_loss = edge_loss / max(n_patches, 1)

    total = kendall(edge_loss, flat_loss)
    return total, edge_loss.detach(), flat_loss.detach()


def stratified_displacement_loss(
    pred: ME.SparseTensor,
    gt_displacement: torch.Tensor,
    curvature: torch.Tensor,
    kendall: KendallUncertaintyWeights,
    smooth_l1_beta: float = 1.0,
    beta: float = 5.0,
    tau: float = 0.33,
) -> tuple:
    """Soft-gated stratified loss: displacement L1 on edges, zero-penalty on flat.

    Unlike stratified_loss (Laplacian edge term), the edge term here is smooth L1
    displacement which has a non-zero minimum and cannot collapse to zero.

    Returns (total_loss, edge_loss, flat_loss) for logging.
    """
    # Soft gate: high w = edge, low w = flat
    w = torch.sigmoid(beta * (curvature - tau))  # (N,)

    # Edge term: smooth L1 displacement (has non-zero optimum, won't collapse)
    if smooth_l1_beta > 0:
        per_point = F.smooth_l1_loss(
            pred.F, gt_displacement, reduction="none", beta=smooth_l1_beta
        )
    else:
        per_point = torch.abs(pred.F - gt_displacement)
    edge_loss = (w.unsqueeze(-1) * per_point).mean()

    # Flat term: zero-displacement penalty (don't move flat regions)
    disp_magnitude = pred.F.pow(2).sum(dim=-1)  # (N,)
    flat_loss = ((1.0 - w) * disp_magnitude).mean()

    total = kendall(edge_loss, flat_loss)
    return total, edge_loss.detach(), flat_loss.detach()


def gated_displacement_loss(
    pred: ME.SparseTensor,
    gate: ME.SparseTensor,
    gt_displacement: torch.Tensor,
    smooth_l1_beta: float = 0.1,
    lambda_gate: float = 1.0,
    gate_eps: float = 1e-3,
) -> tuple:
    """Gated displacement loss with per-stratum normalization.

    L_disp: Huber on ALL points (not gated), normalized per stratum
    (nonzero GT and zero GT separately, then averaged). This ensures
    the 40% nonzero points get equal gradient weight to the 60% zero points.

    L_gate: BCE supervising gate against indicator(||d_gt|| > gate_eps).

    Returns (total_loss, extras_dict) for logging.
    """
    gt_mag = gt_displacement.norm(dim=-1)  # (N,)
    nonzero_mask = gt_mag > gate_eps  # (N,)
    n_nonzero = nonzero_mask.sum().clamp(min=1)
    n_zero = (~nonzero_mask).sum().clamp(min=1)

    # Per-point displacement error
    if smooth_l1_beta > 0:
        per_point = F.smooth_l1_loss(
            pred.F, gt_displacement, reduction="none", beta=smooth_l1_beta
        ).mean(
            dim=-1
        )  # (N,) mean over xyz
    else:
        per_point = torch.abs(pred.F - gt_displacement).mean(dim=-1)

    # Per-stratum normalization: each stratum contributes equally
    loss_nonzero = per_point[nonzero_mask].sum() / n_nonzero
    loss_zero = per_point[~nonzero_mask].sum() / n_zero
    l_disp = 0.5 * (loss_nonzero + loss_zero)

    # Gate BCE loss
    gate_target = nonzero_mask.float()  # 1 = should move, 0 = should stay
    gate_pred = gate.F.squeeze(-1)  # (N,)
    l_gate = F.binary_cross_entropy(gate_pred, gate_target)

    total = l_disp + lambda_gate * l_gate

    extras = {
        "l_disp": l_disp.item(),
        "l_gate": l_gate.item(),
        "gate_mean": gate_pred.mean().item(),
        "gate_nonzero_mean": (
            gate_pred[nonzero_mask].mean().item() if nonzero_mask.any() else 0.0
        ),
        "gate_zero_mean": (
            gate_pred[~nonzero_mask].mean().item() if (~nonzero_mask).any() else 0.0
        ),
    }
    return total, extras


def chamfer_loss(
    pred: ME.SparseTensor,
    input_sparse: ME.SparseTensor,
    gt_displacement: torch.Tensor,
    curvature: torch.Tensor,
    alpha: float = 3.0,
) -> torch.Tensor:
    """Chamfer distance loss with dynamic re-matching per forward pass.

    Uses batched cdist when all patches are equal-sized (typical case).
    Falls back to per-patch loop for variable-size patches.
    """
    decoded = input_sparse.F[:, :3] * COORD_SCALE  # (N, 3)
    refined = decoded + pred.F  # (N, 3), has grad
    target = (decoded + gt_displacement).detach()  # (N, 3), no grad

    batch_idx = pred.C[:, 0]
    curv_w = 1.0 + alpha * curvature  # (N,)

    unique_batches, counts = batch_idx.unique(return_counts=True)
    n_patches = unique_batches.shape[0]

    # Fast path: all patches same size -> batched cdist
    if counts.min() == counts.max() and n_patches > 0:
        M = counts[0].item()
        ref_3d = refined.view(n_patches, M, 3)
        tgt_3d = target.view(n_patches, M, 3)
        w_3d = curv_w.view(n_patches, M)

        dists = torch.cdist(ref_3d, tgt_3d)  # (B, M, M)
        fwd_min = dists.min(dim=2).values  # (B, M)
        bwd_min = dists.min(dim=1).values  # (B, M)

        fwd_loss = (w_3d * fwd_min).mean(dim=1)  # (B,)
        bwd_loss = (w_3d * bwd_min).mean(dim=1)  # (B,)
        return (fwd_loss + bwd_loss).mean()

    # Slow fallback: variable-size patches
    total_loss = torch.tensor(0.0, device=pred.F.device)
    for b in unique_batches:
        mask = batch_idx == b
        ref_b = refined[mask]
        tgt_b = target[mask]
        w_b = curv_w[mask]
        dists = torch.cdist(ref_b, tgt_b)
        fwd_min = dists.min(dim=1).values
        bwd_min = dists.min(dim=0).values
        total_loss = total_loss + (w_b * fwd_min).mean() + (w_b * bwd_min).mean()
    return total_loss / max(n_patches, 1)
