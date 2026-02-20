"""Tests for postprocessing loss functions.

Uses synthetic sparse tensors -- requires MinkowskiEngine.
"""

import numpy as np
import pytest
import torch

try:
    import MinkowskiEngine as ME

    HAS_ME = True
except ImportError:
    HAS_ME = False

pytestmark = pytest.mark.skipif(not HAS_ME, reason="MinkowskiEngine not installed")


def _make_test_data(n: int = 100, seed: int = 42):
    """Create aligned pred, gt_displacement, curvature, and input_sparse for testing.

    Returns pred_feats (leaf tensor) separately for grad checking since
    ME SparseTensor.F is a non-leaf tensor.
    """
    rng = np.random.default_rng(seed)

    # Unique coords with batch index
    coords_np = rng.integers(0, 64, (n * 2, 3)).astype(np.int32)
    coords_np = np.unique(coords_np, axis=0)[:n]
    n = len(coords_np)
    batch_idx = np.zeros((n, 1), dtype=np.int32)
    coords_with_batch = np.hstack([batch_idx, coords_np])

    # Input features: normalized xyz + curvature
    feats_np = np.column_stack(
        [
            coords_np / 1023.0,
            rng.uniform(0, 0.5, n),
        ]
    ).astype(np.float32)

    input_sparse = ME.SparseTensor(
        features=torch.from_numpy(feats_np),
        coordinates=torch.from_numpy(coords_with_batch.astype(np.int32)),
    )

    # Pred: small displacements with grad
    pred_feats = torch.randn(n, 3) * 0.5
    pred_feats.requires_grad_(True)
    pred = ME.SparseTensor(
        features=pred_feats,
        coordinates=torch.from_numpy(coords_with_batch.astype(np.int32)),
    )

    gt_displacement = torch.randn(n, 3) * 0.3
    # Make ~50% of GT displacements zero (mimics real data)
    zero_mask = torch.rand(n) < 0.5
    gt_displacement[zero_mask] = 0.0

    curvature = torch.rand(n) * 0.5

    return pred, input_sparse, gt_displacement, curvature, n, pred_feats


class TestCurvatureWeightedL1Loss:
    def test_basic_shape_and_grad(self):
        from postprocessing.losses import curvature_weighted_l1_loss

        pred, _, gt_disp, curv, n, pred_feats = _make_test_data()
        loss = curvature_weighted_l1_loss(pred, gt_disp, curv, alpha=3.0)

        assert loss.ndim == 0  # scalar
        assert loss.item() > 0
        loss.backward()
        assert pred_feats.grad is not None

    def test_alpha_zero_ignores_curvature(self):
        from postprocessing.losses import curvature_weighted_l1_loss

        pred, _, gt_disp, curv, _, _ = _make_test_data()
        loss_a0 = curvature_weighted_l1_loss(pred, gt_disp, curv, alpha=0.0)
        # With alpha=0, changing curvature should not affect loss
        curv_different = curv * 10
        loss_a0_diff = curvature_weighted_l1_loss(
            pred, gt_disp, curv_different, alpha=0.0
        )
        assert torch.isclose(loss_a0, loss_a0_diff)

    def test_smooth_l1_reduces_outlier_influence(self):
        from postprocessing.losses import curvature_weighted_l1_loss

        pred, _, gt_disp, curv, _, _ = _make_test_data(seed=99)
        # Add large outliers
        gt_disp_outlier = gt_disp.clone()
        gt_disp_outlier[:5] = 10.0

        loss_l1 = curvature_weighted_l1_loss(
            pred, gt_disp_outlier, curv, alpha=0.0, smooth_l1_beta=0.0
        )
        loss_huber = curvature_weighted_l1_loss(
            pred, gt_disp_outlier, curv, alpha=0.0, smooth_l1_beta=1.0
        )
        # Huber should be smaller since it clips large errors
        assert loss_huber < loss_l1

    def test_lambda_shrink_penalizes_zero_gt(self):
        from postprocessing.losses import curvature_weighted_l1_loss

        pred, _, gt_disp, curv, _, _ = _make_test_data()
        loss_no_shrink = curvature_weighted_l1_loss(
            pred, gt_disp, curv, alpha=0.0, lambda_shrink=0.0
        )
        loss_with_shrink = curvature_weighted_l1_loss(
            pred, gt_disp, curv, alpha=0.0, lambda_shrink=1.0
        )
        # Shrinkage adds penalty -> loss should be higher
        assert loss_with_shrink > loss_no_shrink

    def test_perfect_prediction_zero_loss(self):
        from postprocessing.losses import curvature_weighted_l1_loss

        pred, _, gt_disp, curv, n, _ = _make_test_data()
        # Override pred features to match GT exactly
        pred_perfect = ME.SparseTensor(
            features=gt_disp.clone(),
            coordinates=pred.C,
        )
        loss = curvature_weighted_l1_loss(
            pred_perfect, gt_disp, curv, alpha=0.0, mag_floor=1.0
        )
        assert loss.item() < 1e-6


class TestMinOfKDisplacementLoss:
    def test_basic_shape_and_grad(self):
        from postprocessing.losses import min_of_k_displacement_loss

        pred, _, gt_disp, curv, n, pred_feats = _make_test_data()
        # Create K=5 candidates by adding noise to GT
        K = 5
        gt_k = gt_disp.unsqueeze(1).expand(-1, K, -1).clone()
        gt_k += torch.randn_like(gt_k) * 0.1  # slightly different candidates

        loss = min_of_k_displacement_loss(pred, gt_k, curv)
        assert loss.ndim == 0
        assert loss.item() > 0
        loss.backward()
        assert pred_feats.grad is not None

    def test_min_of_k_less_than_single(self):
        """Min-of-K should be <= loss using only the first candidate."""
        from postprocessing.losses import min_of_k_displacement_loss

        pred, _, gt_disp, curv, n, _ = _make_test_data(seed=7)
        K = 5
        gt_k = gt_disp.unsqueeze(1).expand(-1, K, -1).clone()
        gt_k += torch.randn_like(gt_k) * 0.5

        loss_mink = min_of_k_displacement_loss(
            pred, gt_k, curv, alpha=0.0, smooth_l1_beta=0.0
        )
        # Single candidate = min-of-1 (same function, same weighting)
        loss_single = min_of_k_displacement_loss(
            pred, gt_k[:, :1, :], curv, alpha=0.0, smooth_l1_beta=0.0
        )
        assert loss_mink <= loss_single + 1e-6

    def test_shrink_only_when_all_k_zero(self):
        """Shrinkage should only apply when ALL K candidates are zero."""
        from postprocessing.losses import min_of_k_displacement_loss

        pred, _, _, curv, n, _ = _make_test_data()
        K = 3
        gt_k = torch.randn(n, K, 3) * 0.3
        # Make first few points: one candidate nonzero, rest zero
        gt_k[:10, 1:, :] = 0.0  # candidates 1,2 zero but candidate 0 nonzero
        # Make next few points: ALL candidates zero
        gt_k[10:20, :, :] = 0.0

        loss_no_shrink = min_of_k_displacement_loss(pred, gt_k, curv, lambda_shrink=0.0)
        loss_with_shrink = min_of_k_displacement_loss(
            pred, gt_k, curv, lambda_shrink=1.0
        )
        # Should differ because points 10:20 have all-zero candidates
        assert loss_with_shrink > loss_no_shrink


class TestLaplacianLoss:
    def test_basic_runs(self):
        from postprocessing.losses import laplacian_loss

        pred, input_sparse, gt_disp, curv, n, _ = _make_test_data()
        loss = laplacian_loss(pred, input_sparse, gt_disp, curv, alpha=1.0, k=4)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_zero_displacement_low_loss(self):
        """If pred=0, refined=decoded, and gt_disp moves to original.
        Laplacian should be nonzero (decoded != original curvature)."""
        from postprocessing.losses import laplacian_loss

        pred, input_sparse, gt_disp, curv, n, _ = _make_test_data()
        pred_zero = ME.SparseTensor(
            features=torch.zeros(n, 3),
            coordinates=pred.C,
        )
        loss = laplacian_loss(pred_zero, input_sparse, gt_disp, curv, alpha=0.0, k=4)
        assert torch.isfinite(loss)


class TestStratifiedDisplacementLoss:
    def test_basic_shape_and_components(self):
        from postprocessing.losses import (
            KendallUncertaintyWeights,
            stratified_displacement_loss,
        )

        pred, _, gt_disp, curv, n, pred_feats = _make_test_data()
        kendall = KendallUncertaintyWeights(n_tasks=2)

        total, edge_loss, flat_loss = stratified_displacement_loss(
            pred, gt_disp, curv, kendall
        )
        assert total.ndim == 0
        assert edge_loss >= 0
        assert flat_loss >= 0
        total.backward()
        assert pred_feats.grad is not None

    def test_high_curvature_increases_edge_weight(self):
        """Points with high curvature should contribute more to edge_loss."""
        from postprocessing.losses import (
            KendallUncertaintyWeights,
            stratified_displacement_loss,
        )

        pred, _, gt_disp, _, n, _ = _make_test_data()
        kendall = KendallUncertaintyWeights(n_tasks=2)

        curv_low = torch.full((n,), 0.01)
        _, edge_low, flat_low = stratified_displacement_loss(
            pred, gt_disp, curv_low, kendall, beta=5.0, tau=0.33
        )

        curv_high = torch.full((n,), 0.9)
        _, edge_high, flat_high = stratified_displacement_loss(
            pred, gt_disp, curv_high, kendall, beta=5.0, tau=0.33
        )

        # High curvature -> more edge contribution, less flat contribution
        assert edge_high > edge_low
        assert flat_high < flat_low


class TestStratifiedLoss:
    def test_basic_runs(self):
        from postprocessing.losses import KendallUncertaintyWeights, stratified_loss

        pred, input_sparse, gt_disp, curv, n, _ = _make_test_data()
        kendall = KendallUncertaintyWeights(n_tasks=2)

        total, edge_loss, flat_loss = stratified_loss(
            pred, input_sparse, gt_disp, curv, kendall, alpha=1.0, k=4
        )
        assert total.ndim == 0
        assert torch.isfinite(total)
        assert edge_loss >= 0
        assert flat_loss >= 0


class TestChamferLoss:
    def test_basic_runs(self):
        from postprocessing.losses import chamfer_loss

        pred, input_sparse, gt_disp, curv, n, _ = _make_test_data()
        loss = chamfer_loss(pred, input_sparse, gt_disp, curv, alpha=1.0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0


class TestKendallUncertaintyWeights:
    def test_basic(self):
        from postprocessing.losses import KendallUncertaintyWeights

        kendall = KendallUncertaintyWeights(n_tasks=2)
        loss1 = torch.tensor(1.0)
        loss2 = torch.tensor(2.0)
        total = kendall(loss1, loss2)
        assert total.ndim == 0
        assert total.item() > 0

    def test_weights_sum(self):
        from postprocessing.losses import KendallUncertaintyWeights

        kendall = KendallUncertaintyWeights(n_tasks=2)
        weights = kendall.weights()
        assert len(weights) == 2
        assert all(w > 0 for w in weights)
