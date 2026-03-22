"""Tests for postprocessing model variants.

Covers: forward shapes, output bounds, gradient flow, rate representations.
Requires MinkowskiEngine -- skipped otherwise.
"""

import importlib.util

import numpy as np
import pytest
import torch

HAS_ME = importlib.util.find_spec("MinkowskiEngine") is not None
pytestmark = pytest.mark.skipif(not HAS_ME, reason="MinkowskiEngine not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse(n: int = 64, in_channels: int = 3, batch_size: int = 2, seed: int = 0):
    """Return (sparse_tensor, rates) for a small synthetic batch."""
    import MinkowskiEngine as ME

    rng = np.random.default_rng(seed)
    coords_list, feats_list = [], []
    for b in range(batch_size):
        pts = rng.integers(0, 32, (n, 3)).astype(np.int32)
        pts = np.unique(pts, axis=0)
        coords_list.append(torch.from_numpy(pts))
        feats_list.append(torch.randn(len(pts), in_channels))

    coords, feats = ME.utils.sparse_collate(coords_list, feats_list)
    x = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
    # bpp-style rates: log(bpp) values
    rates = torch.tensor([-2.5, -3.0][:batch_size])
    return x, rates


def _make_onehot_rates(batch_size: int = 2):
    """Return onehot-style rates (r2 and r4 as n/7.0 scalars)."""
    return torch.tensor([2 / 7.0, 4 / 7.0][:batch_size])


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_all_model_types_instantiate(self):
        from postprocessing.model import build_model

        for model_type in [
            "unet",
            "film",
            "film_head",
            "film_head_v2",
            "film_head_v3",
            "occupancy",
        ]:
            m = build_model(model_type, in_channels=4)
            assert m is not None

    def test_unknown_type_raises(self):
        from postprocessing.model import build_model

        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model("nonexistent", in_channels=4)

    def test_forward_after_build(self):
        from postprocessing.model import build_model

        model = build_model("film_head_v2", in_channels=3, max_displacement=5.0).eval()
        x, rates = _make_sparse(in_channels=3)
        with torch.no_grad():
            out = model(x, rates)
        assert out.F.shape[1] == 3


class TestRateUtils:
    def test_get_rate_in_dim_bpp(self):
        from postprocessing.model import _get_rate_in_dim

        assert _get_rate_in_dim("bpp") == 1

    def test_get_rate_in_dim_onehot(self):
        from postprocessing.model import MAX_RATE, _get_rate_in_dim

        assert _get_rate_in_dim("onehot") == MAX_RATE

    def test_get_rate_in_dim_fourier(self):
        from postprocessing.model import FOURIER_L, _get_rate_in_dim

        assert _get_rate_in_dim("fourier") == 2 * FOURIER_L

    def test_rate_to_input_bpp_shape(self):
        from postprocessing.model import _rate_to_input

        rates = torch.tensor([-2.5, -3.0])
        out = _rate_to_input(rates, "bpp")
        assert out.shape == (2, 1)

    def test_rate_to_input_onehot_shape(self):
        from postprocessing.model import MAX_RATE, _rate_to_input

        rates = torch.tensor([2 / 7.0, 4 / 7.0])
        out = _rate_to_input(rates, "onehot")
        assert out.shape == (2, MAX_RATE)
        # Each row is a valid one-hot
        assert (out.sum(dim=1) == 1).all()

    def test_rate_to_input_fourier_shape(self):
        from postprocessing.model import FOURIER_L, _rate_to_input

        rates = torch.tensor([-2.5, -3.0])
        out = _rate_to_input(rates, "fourier")
        assert out.shape == (2, 2 * FOURIER_L)

    def test_rate_to_input_scalar(self):
        """Scalar (0-dim) input should be handled gracefully."""
        from postprocessing.model import _rate_to_input

        out = _rate_to_input(torch.tensor(-2.5), "bpp")
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# SparseUNet (no rate conditioning)
# ---------------------------------------------------------------------------


class TestSparseUNet:
    def test_forward_shape(self):
        from postprocessing.model import SparseUNet

        model = SparseUNet(in_channels=3, max_displacement=5.0).eval()
        x, _ = _make_sparse(in_channels=3)
        out = model(x)
        assert out.F.shape == (x.F.shape[0], 3)

    def test_output_bounded(self):
        from postprocessing.model import SparseUNet

        model = SparseUNet(in_channels=3, max_displacement=5.0).eval()
        x, _ = _make_sparse(in_channels=3)
        with torch.no_grad():
            out = model(x)
        assert out.F.abs().max().item() <= 5.0 + 1e-5

    def test_gradient_flows(self):
        from postprocessing.model import SparseUNet

        model = SparseUNet(in_channels=3).train()
        x, _ = _make_sparse(in_channels=3)
        out = model(x)
        loss = out.F.pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_return_pre_head(self):
        from postprocessing.model import SparseUNet

        model = SparseUNet(in_channels=3).eval()
        x, _ = _make_sparse(in_channels=3)
        pre = model(x, return_pre_head=True)
        assert pre.F.shape[1] == 32  # 32-ch before head


# ---------------------------------------------------------------------------
# FiLMSparseUNet
# ---------------------------------------------------------------------------


class TestFiLMSparseUNet:
    def test_forward_shape_bpp(self):
        from postprocessing.model import FiLMSparseUNet

        model = FiLMSparseUNet(in_channels=3, rate_repr="bpp").eval()
        x, rates = _make_sparse(in_channels=3)
        out = model(x, rates)
        assert out.F.shape == (x.F.shape[0], 3)

    def test_forward_shape_onehot(self):
        from postprocessing.model import FiLMSparseUNet

        model = FiLMSparseUNet(in_channels=3, rate_repr="onehot").eval()
        x, _ = _make_sparse(in_channels=3)
        rates = _make_onehot_rates()
        out = model(x, rates)
        assert out.F.shape == (x.F.shape[0], 3)

    def test_forward_shape_fourier(self):
        from postprocessing.model import FiLMSparseUNet

        model = FiLMSparseUNet(in_channels=3, rate_repr="fourier").eval()
        x, rates = _make_sparse(in_channels=3)
        out = model(x, rates)
        assert out.F.shape == (x.F.shape[0], 3)

    def test_gradient_flows(self):
        from postprocessing.model import FiLMSparseUNet

        model = FiLMSparseUNet(in_channels=3, rate_repr="onehot").train()
        x, _ = _make_sparse(in_channels=3)
        rates = _make_onehot_rates()
        out = model(x, rates)
        out.F.pow(2).mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# FiLMHeadSparseUNetV2
# ---------------------------------------------------------------------------


class TestFiLMHeadSparseUNetV2:
    def test_forward_shape(self):
        from postprocessing.model import FiLMHeadSparseUNetV2

        model = FiLMHeadSparseUNetV2(in_channels=3).eval()
        x, rates = _make_sparse(in_channels=3)
        out = model(x, rates)
        assert out.F.shape == (x.F.shape[0], 3)

    def test_output_bounded(self):
        from postprocessing.model import FiLMHeadSparseUNetV2

        model = FiLMHeadSparseUNetV2(in_channels=3, max_displacement=5.0).eval()
        x, rates = _make_sparse(in_channels=3)
        with torch.no_grad():
            out = model(x, rates)
        # At init (zero-init FiLM), output ≈ tanh(backbone) * 5.0 -- bounded
        assert out.F.abs().max().item() < 10.0

    def test_gradient_flows(self):
        from postprocessing.model import FiLMHeadSparseUNetV2

        model = FiLMHeadSparseUNetV2(in_channels=3).train()
        x, rates = _make_sparse(in_channels=3)
        out = model(x, rates)
        out.F.pow(2).mean().backward()
        film_grads = [
            p.grad
            for name, p in model.named_parameters()
            if "film" in name and p.grad is not None
        ]
        assert len(film_grads) > 0, "FiLM head received no gradients"

    def test_different_rates_differ(self):
        """Different rates should produce different outputs (FiLM is active)."""
        from postprocessing.model import FiLMHeadSparseUNetV2

        # Need non-zero FiLM weights -- perturb from zero init
        model = FiLMHeadSparseUNetV2(in_channels=3).eval()
        with torch.no_grad():
            model.film_head.weight.normal_(0, 0.1)
            model.film_head.bias.normal_(0, 0.1)

        x, _ = _make_sparse(in_channels=3, batch_size=1)
        rates_a = torch.tensor([-1.0])
        rates_b = torch.tensor([-4.0])
        with torch.no_grad():
            out_a = model(x, rates_a)
            out_b = model(x, rates_b)
        assert not torch.allclose(out_a.F, out_b.F), "Different rates should differ"


# ---------------------------------------------------------------------------
# FiLMHeadSparseUNetV3
# ---------------------------------------------------------------------------


class TestFiLMHeadSparseUNetV3:
    def test_forward_shape_train(self):
        from postprocessing.model import FiLMHeadSparseUNetV3

        model = FiLMHeadSparseUNetV3(in_channels=3, max_displacement=5.0).train()
        x, rates = _make_sparse(in_channels=3)
        out = model(x, rates)
        assert out.F.shape == (x.F.shape[0], 3)

    def test_forward_shape_eval(self):
        from postprocessing.model import FiLMHeadSparseUNetV3

        model = FiLMHeadSparseUNetV3(in_channels=3, max_displacement=5.0).eval()
        x, rates = _make_sparse(in_channels=3)
        with torch.no_grad():
            out = model(x, rates)
        assert out.F.shape == (x.F.shape[0], 3)

    def test_return_logits(self):
        from postprocessing.model import FiLMHeadSparseUNetV3

        model = FiLMHeadSparseUNetV3(in_channels=3, max_displacement=5.0).eval()
        x, rates = _make_sparse(in_channels=3)
        with torch.no_grad():
            st, logits = model(x, rates, return_logits=True)
        n_pts = x.F.shape[0]
        n_classes = 2 * 5 + 1  # max_displacement=5 -> 11 classes
        assert logits.shape == (n_pts, 3, n_classes)

    def test_eval_output_integer(self):
        """Eval mode uses argmax -> displacements are exact integers."""
        from postprocessing.model import FiLMHeadSparseUNetV3

        model = FiLMHeadSparseUNetV3(in_channels=3, max_displacement=5.0).eval()
        x, rates = _make_sparse(in_channels=3)
        with torch.no_grad():
            out = model(x, rates)
        assert (out.F == out.F.round()).all(), "Eval displacements should be integers"

    def test_gradient_flows(self):
        from postprocessing.model import FiLMHeadSparseUNetV3

        model = FiLMHeadSparseUNetV3(in_channels=3).train()
        x, rates = _make_sparse(in_channels=3)
        out = model(x, rates)
        out.F.pow(2).mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# OccupancySparseUNet
# ---------------------------------------------------------------------------


class TestOccupancySparseUNet:
    def test_forward_shape(self):
        from postprocessing.model import OccupancySparseUNet

        model = OccupancySparseUNet(in_channels=4).eval()
        x, rates = _make_sparse(in_channels=4)
        out = model(x, rates)
        # Output: 1-ch logit per point
        assert out.F.shape == (x.F.shape[0], 1)

    def test_output_unbounded(self):
        """Logits should be unbounded (no tanh/sigmoid applied inside model)."""
        from postprocessing.model import OccupancySparseUNet

        model = OccupancySparseUNet(in_channels=4).eval()
        x, rates = _make_sparse(in_channels=4)
        with torch.no_grad():
            out = model(x, rates)
        # No hard bound; just check it runs and shape is correct
        assert torch.isfinite(out.F).all()

    def test_gradient_flows(self):
        from postprocessing.model import OccupancySparseUNet

        model = OccupancySparseUNet(in_channels=4).train()
        x, rates = _make_sparse(in_channels=4)
        out = model(x, rates)
        out.F.pow(2).mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
