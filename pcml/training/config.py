"""Training configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for point cloud compression model training."""

    num_points: int = 2048
    latent_dim: int = 512
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 512])

    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    chamfer_weight: float = 1.0
    mse_weight: float = 0.0

    checkpoint_dir: Path = field(default_factory=lambda: Path("models/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    val_check_interval: float = 1.0
    save_top_k: int = 3

    num_workers: int = 4
    prefetch_factor: int = 2

    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
