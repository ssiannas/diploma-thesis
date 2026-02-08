"""Test basic model functionality.

This validates that the model can perform forward and backward passes
without errors.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.models.baseline import BaselineCompressionModel
from pcml.models.losses import ChamferDistance, PointCloudReconstructionLoss


class TestModelBasics:
    """Test suite for basic model functionality."""

    @pytest.fixture
    def device(self):
        """Get available device (CUDA if available)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        """Create baseline model."""
        model = BaselineCompressionModel(
            num_points=2048,
            latent_dim=512,
            hidden_dims=[1024, 512],
            compress_colors=False,
        )
        return model.to(device)

    @pytest.fixture
    def sample_batch(self, device):
        """Create sample batch of point clouds."""
        batch_size = 4
        num_points = 2048
        geometry = torch.randn(batch_size, num_points, 3, device=device)
        return geometry

    def test_model_forward_pass(self, model, sample_batch, device):
        """Test forward pass completes without errors."""
        model.eval()

        with torch.no_grad():
            compressed, recon_geom, recon_colors = model(sample_batch)

        assert compressed.shape == (
            4,
            512,
        ), f"Expected (4, 512), got {compressed.shape}"
        assert recon_geom.shape == (
            4,
            2048,
            3,
        ), f"Expected (4, 2048, 3), got {recon_geom.shape}"
        assert (
            recon_colors is None
        ), "Should not return colors when compress_colors=False"

        print(
            f"[OK] Forward pass: input={sample_batch.shape}, "
            f"compressed={compressed.shape}"
        )

    def test_model_backward_pass(self, model, sample_batch, device):
        """Test backward pass computes gradients."""
        model.train()

        compressed, recon_geom, _ = model(sample_batch)
        loss = torch.nn.functional.mse_loss(recon_geom, sample_batch)

        loss.backward()

        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "No gradients computed"

        gradient_sum = sum(
            p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
        )
        assert gradient_sum > 0, "All gradients are zero"

        print(
            f"[OK] Backward pass: loss={loss.item():.6f}, "
            f"gradient_sum={gradient_sum:.2e}"
        )

    def test_model_single_sample(self, model, device):
        """Test model works with single sample (no batch)."""
        model.eval()

        single_sample = torch.randn(2048, 3, device=device)

        with torch.no_grad():
            compressed, recon_geom, _ = model(single_sample)

        assert compressed.shape == (1, 512)
        assert recon_geom.shape == (1, 2048, 3)

        print("[OK] Single sample inference works")

    def test_chamfer_loss_forward(self, sample_batch):
        """Test Chamfer distance loss computation."""
        chamfer = ChamferDistance()

        pred = sample_batch + torch.randn_like(sample_batch) * 0.01
        loss = chamfer(pred, sample_batch)

        assert not torch.isnan(loss), "Chamfer loss is NaN"
        assert not torch.isinf(loss), "Chamfer loss is Inf"
        assert loss.item() > 0, "Chamfer loss should be positive"

        print(f"[OK] Chamfer loss: {loss.item():.6f}")

    def test_reconstruction_loss_forward(self, sample_batch):
        """Test reconstruction loss computation."""
        loss_fn = PointCloudReconstructionLoss(chamfer_weight=1.0, mse_weight=0.1)

        pred = sample_batch + torch.randn_like(sample_batch) * 0.01
        loss = loss_fn(pred, sample_batch)

        assert not torch.isnan(loss), "Reconstruction loss is NaN"
        assert not torch.isinf(loss), "Reconstruction loss is Inf"
        assert loss.item() > 0, "Reconstruction loss should be positive"

        print(f"[OK] Reconstruction loss: {loss.item():.6f}")

    def test_model_different_batch_sizes(self, model, device):
        """Test model handles different batch sizes."""
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16]

        for bs in batch_sizes:
            geometry = torch.randn(bs, 2048, 3, device=device)

            with torch.no_grad():
                compressed, recon_geom, _ = model(geometry)

            assert compressed.shape[0] == bs
            assert recon_geom.shape[0] == bs

        print(f"[OK] Model handles batch sizes: {batch_sizes}")

    def test_gpu_memory_single_forward(self, model, device):
        """Test GPU memory usage for single forward pass."""
        if device.type != "cuda":
            pytest.skip("CUDA not available")

        torch.cuda.reset_peak_memory_stats(device)

        geometry = torch.randn(16, 2048, 3, device=device)

        with torch.no_grad():
            compressed, recon_geom, _ = model(geometry)

        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        print(f"[OK] GPU memory (batch=16): {peak_memory_mb:.2f} MB")

        assert peak_memory_mb < 5000, f"Memory usage too high: {peak_memory_mb:.2f} MB"

    def test_model_gradients_flow(self, model, sample_batch):
        """Test gradients flow through entire model."""
        model.train()

        compressed, recon_geom, _ = model(sample_batch)
        loss = ((recon_geom - sample_batch) ** 2).sum()

        loss.backward()

        layers_with_gradients = 0
        layers_without_gradients = 0

        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                layers_with_gradients += 1
            else:
                layers_without_gradients += 1

        print(
            f"[OK] Gradient flow: {layers_with_gradients} layers have gradients, "
            f"{layers_without_gradients} don't"
        )

        assert layers_with_gradients > 0, "No layers received gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
