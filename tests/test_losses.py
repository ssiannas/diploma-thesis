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


class TestDynamicChamferLoss:
    def test_basic_runs(self):
        from postprocessing.losses import dynamic_chamfer_loss

        pred, _, _, _, n, pred_feats = _make_test_data()
        # Create a fake original patch
        orig = torch.randn(n, 3)
        loss, extras = dynamic_chamfer_loss(pred, [orig])
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0
        assert "cd_fwd" in extras
        assert "cd_rev" in extras
        loss.backward()
        assert pred_feats.grad is not None


class TestGetLossFn:
    def test_valid_types(self):
        from postprocessing.losses import get_loss_fn

        for name in ["cd", "focal_cd", "edge_cd", "cd_lap", "cd_vlap"]:
            fn = get_loss_fn(name)
            assert fn is not None

    def test_invalid_type(self):
        from postprocessing.losses import get_loss_fn

        with pytest.raises(ValueError):
            get_loss_fn("nonexistent")


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
