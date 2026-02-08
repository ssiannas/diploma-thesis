"""CRITICAL TEST: Overfitting on single sample.

This is the most important test. If the model cannot overfit to a single sample,
it will never work on the full dataset. This test MUST pass before any full training.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.models.baseline import BaselineCompressionModel
from pcml.models.losses import ChamferDistance


class TestOverfitting:
    """Test model can overfit to single sample - CRITICAL for training validation."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_single_sample_overfitting(self, device):
        """CRITICAL: Model must overfit to tiny dataset (1 sample).

        If this test fails, the model/training is broken and full training
        will NOT work. Debug this before proceeding.

        Note: Using eval mode to freeze BatchNorm for stable overfitting test.
        """
        print("\n" + "=" * 80)
        print("CRITICAL OVERFITTING TEST")
        print("=" * 80)

        model = BaselineCompressionModel(
            num_points=2048,
            latent_dim=512,
            hidden_dims=[1024, 512],
            compress_colors=False,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=3e-3)
        loss_fn = ChamferDistance()

        torch.manual_seed(42)
        single_sample = torch.randn(1, 2048, 3, device=device)

        model.eval()
        losses = []

        print("\nTraining on 1 sample for 1000 epochs (eval mode, higher LR)...")

        for epoch in range(1000):
            optimizer.zero_grad()

            _, recon, _ = model(single_sample)
            loss = loss_fn(recon, single_sample)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")

        initial_loss = losses[0]
        final_loss = losses[-1]
        reduction = (initial_loss - final_loss) / initial_loss

        print("\n" + "-" * 80)
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Final loss:   {final_loss:.6f}")
        print(f"Reduction:    {reduction*100:.1f}%")
        print("-" * 80)

        assert final_loss < initial_loss * 0.1, (
            f"CRITICAL FAILURE: Model did not overfit!\n"
            f"Initial: {initial_loss:.6f}, Final: {final_loss:.6f}\n"
            f"Expected >90% reduction, got {reduction*100:.1f}%\n"
            f"This means training will NOT work. Debug model/loss before proceeding!"
        )

        assert final_loss < 0.01, (
            f"CRITICAL FAILURE: Final loss too high!\n"
            f"Final loss: {final_loss:.6f}, Expected: <0.01\n"
            f"Model should memorize single sample completely."
        )

        print("\n[OK] OVERFITTING TEST PASSED")
        print("[OK] Model CAN learn - safe to proceed with full training!")
        print("=" * 80 + "\n")

    def test_overfitting_with_larger_batch(self, device):
        """Test overfitting on small batch of 4 samples."""
        print("\n" + "=" * 80)
        print("BATCH OVERFITTING TEST (4 samples)")
        print("=" * 80)

        model = BaselineCompressionModel(
            num_points=2048,
            latent_dim=512,
            hidden_dims=[1024, 512],
            compress_colors=False,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = ChamferDistance()

        torch.manual_seed(42)
        batch = torch.randn(4, 2048, 3, device=device)

        model.train()
        losses = []

        print("\nTraining on 4 samples for 500 epochs...")

        for epoch in range(500):
            optimizer.zero_grad()

            _, recon, _ = model(batch)
            loss = loss_fn(recon, batch)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:3d}: Loss = {loss.item():.6f}")

        initial_loss = losses[0]
        final_loss = losses[-1]
        reduction = (initial_loss - final_loss) / initial_loss

        print("\n" + "-" * 80)
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Final loss:   {final_loss:.6f}")
        print(f"Reduction:    {reduction*100:.1f}%")
        print("-" * 80)

        assert final_loss < initial_loss * 0.2, (
            f"Model did not overfit batch of 4 samples. "
            f"Reduction: {reduction*100:.1f}%"
        )

        print("\n[OK] Model can overfit small batch")
        print("=" * 80 + "\n")

    def test_loss_decreases_monotonically(self, device):
        """Test that loss generally decreases (allowing for small fluctuations)."""
        model = BaselineCompressionModel(
            num_points=2048,
            latent_dim=512,
            hidden_dims=[1024, 512],
            compress_colors=False,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = ChamferDistance()

        torch.manual_seed(42)
        single_sample = torch.randn(1, 2048, 3, device=device)

        model.train()
        losses = []

        for epoch in range(50):
            optimizer.zero_grad()
            _, recon, _ = model(single_sample)
            loss = loss_fn(recon, single_sample)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        window_size = 10
        first_window_avg = sum(losses[:window_size]) / window_size
        last_window_avg = sum(losses[-window_size:]) / window_size

        assert last_window_avg < first_window_avg, (
            f"Loss not decreasing: first 10 epochs avg={first_window_avg:.6f}, "
            f"last 10 epochs avg={last_window_avg:.6f}"
        )

        print(f"[OK] Loss decreases: {first_window_avg:.6f} -> {last_window_avg:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
