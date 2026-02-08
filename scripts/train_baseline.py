"""Training script for baseline compression model."""

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import PLYPointCloudLoader, PointCloudData
from pcml.training import CompressionLightningModule, TrainingConfig


class PointCloudDataModule(pl.LightningDataModule):
    """Lightning DataModule for point cloud datasets."""

    def __init__(
        self,
        data_dir: str,
        num_points: int = 2048,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            loader = PLYPointCloudLoader(verbose=False)
            ply_files = sorted(self.data_dir.glob("*.ply"))

            point_clouds = []
            for ply_file in ply_files[:10]:
                pc = loader.load(str(ply_file))

                import numpy as np

                if pc.num_points > self.num_points:
                    indices = np.random.choice(
                        pc.num_points, size=self.num_points, replace=False
                    )
                    pc.geometry = pc.geometry[indices]
                    if pc.colors is not None:
                        pc.colors = pc.colors[indices]

                min_vals = pc.geometry.min(axis=0)
                max_vals = pc.geometry.max(axis=0)
                pc.geometry = 2 * (pc.geometry - min_vals) / (max_vals - min_vals) - 1

                point_clouds.append(pc)

            dataset = PointCloudTensorDataset(point_clouds)

            train_size = int(len(dataset) * self.train_split)
            val_size = len(dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class PointCloudTensorDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for point clouds."""

    def __init__(self, point_clouds: list[PointCloudData]) -> None:
        self.point_clouds = point_clouds

    def __len__(self) -> int:
        return len(self.point_clouds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pc = self.point_clouds[idx]
        batch = {"geometry": torch.from_numpy(pc.geometry).float()}

        if pc.colors is not None:
            batch["colors"] = torch.from_numpy(pc.colors).float() / 255.0

        return batch


def main():
    print("=" * 80)
    print("Baseline Model Training")
    print("=" * 80)

    config = TrainingConfig(
        num_points=2048,
        latent_dim=512,
        hidden_dims=[1024, 512],
        batch_size=8,
        num_epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-5,
        chamfer_weight=1.0,
        mse_weight=0.0,
        num_workers=4,
        seed=42,
    )

    if config.seed is not None:
        pl.seed_everything(config.seed, workers=True)

    print("\n[1/5] Setting up data module...")
    data_module = PointCloudDataModule(
        data_dir="datasets/8iVFB_small",
        num_points=config.num_points,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split=0.8,
    )

    print("\n[2/5] Creating model...")
    model = CompressionLightningModule(
        num_points=config.num_points,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        compress_colors=False,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        chamfer_weight=config.chamfer_weight,
        mse_weight=config.mse_weight,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Model created with {total_params:,} parameters")

    print("\n[3/5] Setting up callbacks and logger...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="baseline-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=config.save_top_k,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss", patience=15, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger(save_dir=str(config.log_dir), name="baseline_model")

    print(f"  [OK] Checkpoints will be saved to: {config.checkpoint_dir}")
    print(f"  [OK] Logs will be saved to: {config.log_dir}")

    print("\n[4/5] Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        val_check_interval=config.val_check_interval,
        deterministic=True if config.seed is not None else False,
    )

    print(f"  [OK] Training for {config.num_epochs} epochs")
    print(f"  [OK] Using device: {trainer.accelerator}")

    print("\n[5/5] Starting training...")
    print("=" * 80)

    trainer.fit(model, datamodule=data_module)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
