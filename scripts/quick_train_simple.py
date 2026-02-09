"""Quick training of simple baseline to validate pipeline."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import PLYPointCloudLoader
from pcml.metrics.quality import GeometryQualityCalculator
from pcml.models.losses import ChamferDistance
from pcml.models.simple import SimpleCompressionModel


class PointCloudDataset(Dataset):
    def __init__(self, ply_files: list[Path], num_points: int = 2048):
        self.ply_files = ply_files
        self.num_points = num_points
        self.loader = PLYPointCloudLoader(verbose=False)

    def __len__(self) -> int:
        return len(self.ply_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pc = self.loader.load(str(self.ply_files[idx]))

        if pc.num_points > self.num_points:
            indices = np.random.choice(
                pc.num_points, size=self.num_points, replace=False
            )
            geometry = pc.geometry[indices]
        else:
            pad_size = self.num_points - pc.num_points
            geometry = np.vstack([pc.geometry, np.zeros((pad_size, 3))])

        min_vals, max_vals = geometry.min(axis=0), geometry.max(axis=0)
        geometry = 2 * (geometry - min_vals) / (max_vals - min_vals + 1e-8) - 1

        return torch.from_numpy(geometry).float()


def train(
    data_dir: str = "datasets/8iVFB_small",
    num_frames: int = 20,
    num_epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-3,
    output_dir: str = "results/baseline_quick",
    checkpoint_dir: str = "models/baselines/simple_quick",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training simple baseline: {num_frames} frames, {num_epochs} epochs")
    print(f"Device: {device}")

    ply_files = sorted(Path(data_dir).glob("*.ply"))[:num_frames]
    dataset = PointCloudDataset(ply_files, num_points=2048)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    model = SimpleCompressionModel(
        num_points=2048, latent_dim=512, hidden_dims=[1024, 512]
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = ChamferDistance()

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for geometry in dataloader:
            geometry = geometry.to(device)
            optimizer.zero_grad()

            _, recon, _ = model(geometry)
            loss = loss_fn(recon, geometry)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: {avg_loss:.4f}")

    print(f"Loss: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        test_geom = dataset[0].unsqueeze(0).to(device)
        _, recon_geom, _ = model(test_geom)

        metrics = GeometryQualityCalculator.calculate_all(
            test_geom[0].cpu().numpy(), recon_geom[0].cpu().numpy()
        )
        print(f"PSNR: {metrics.psnr:.1f} dB")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.savefig(output_path / "loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "epoch": num_epochs,
        "loss": epoch_losses[-1],
        "psnr": metrics.psnr,
        "config": {
            "model_type": "SimpleCompressionModel",
            "num_points": 2048,
            "latent_dim": 512,
            "hidden_dims": [1024, 512],
            "num_frames": num_frames,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
    }
    torch.save(checkpoint_data, checkpoint_path / "checkpoint.pth")

    with open(checkpoint_path / "metrics.json", "w") as f:
        import json

        json.dump(
            {
                "final_loss": float(epoch_losses[-1]),
                "final_psnr": float(metrics.psnr),
                "epoch_losses": [float(x) for x in epoch_losses],
            },
            f,
            indent=2,
        )

    print(f"Results: {output_dir}/")
    print(f"Checkpoint: {checkpoint_dir}/")
    return epoch_losses[-1], metrics.psnr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/8iVFB_small")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="results/baseline_quick")
    parser.add_argument("--checkpoint-dir", default="models/baselines/simple_quick")

    args = parser.parse_args()
    train(
        args.data_dir,
        args.num_frames,
        args.epochs,
        args.batch_size,
        args.lr,
        args.output_dir,
        args.checkpoint_dir,
    )
