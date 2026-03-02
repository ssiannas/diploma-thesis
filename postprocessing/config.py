"""Typed configuration for training pipeline.

Replaces 36 CLI args with grouped dataclasses + YAML config files.
Usage: python postprocessing/train.py --config configs/cd_r2.yaml
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    data_root: str = "datasets/pcgcv2_decoded"
    rates: List[str] = field(default_factory=lambda: ["r2"])
    val_sequence: str = "redandblack"
    patch_size: int = 64
    stride: int = 32
    min_points: int = 100
    curvature_k: int = 30
    max_clouds: Optional[int] = None
    frame_stride: int = 1
    augment: bool = True
    use_curvature: bool = True
    overfit: bool = False


@dataclass
class ModelConfig:
    model_type: str = "unet"  # unet | film | film_head
    max_displacement: float = 5.0
    film_embed_dim: int = 64
    film_rate_repr: str = "bpp"  # bpp | onehot


@dataclass
class LossConfig:
    loss_type: str = "cd"  # cd | focal_cd | edge_cd | cd_lap | cd_vlap
    chamfer_padding: int = 0  # Phase 12 lesson: padding must be 0
    cd_fwd_weight: float = 1.0
    cd_rev_weight: float = 1.0
    alpha: float = 10.0
    laplacian_k: int = 8
    lambda_lap: float = 1.0
    lambda_flat: float = 1.0
    focal_gamma: float = 2.0
    edge_beta: float = 10.0
    edge_tau: float = 0.05


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    batch_size: int = 16
    lr: float = 1e-3
    warmup_epochs: int = 5
    epochs: int = 30
    num_workers: int = 8
    device: str = "cuda"
    save_dir: str = "models/postprocessing"
    log_file: Optional[str] = None
    resume: Optional[str] = None
    weight_decay: float = 1e-2
    rate_stratified: bool = False
    encoder_lr_scale: float = 1.0  # 0.1 for stage 2 (freeze encoder)


def _coerce_type(value: Any, field_type: type) -> Any:
    """Coerce a value to match the expected field type (handles YAML quirks)."""
    # PyYAML parses "1e-3" as str, not float -- coerce numeric strings
    if field_type is float and isinstance(value, str):
        return float(value)
    if field_type is int and isinstance(value, str):
        return int(value)
    if field_type is float and isinstance(value, int):
        return float(value)
    return value


def _merge_dict_into_dataclass(dc: Any, d: dict) -> None:
    """Recursively merge a dict into a dataclass instance in-place."""
    dc_fields = {f.name: f for f in fields(dc)}
    for key, value in d.items():
        if key not in dc_fields:
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dict_into_dataclass(current, value)
        else:
            value = _coerce_type(value, dc_fields[key].type)
            setattr(dc, key, value)


def load_config(path: str) -> TrainConfig:
    """Load a YAML config file into a TrainConfig dataclass."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = TrainConfig()
    _merge_dict_into_dataclass(cfg, raw)
    return cfg


def _parse_value(value_str: str) -> Any:
    """Parse a CLI override value string into a typed Python value."""
    # List: [r2,r3]
    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1].strip()
        if not inner:
            return []
        return [item.strip() for item in inner.split(",")]
    # Bool
    if value_str.lower() in ("true", "yes"):
        return True
    if value_str.lower() in ("false", "no"):
        return False
    # None
    if value_str.lower() == "none":
        return None
    # Number
    try:
        if "." in value_str or "e" in value_str.lower():
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str


def apply_overrides(cfg: TrainConfig, overrides: List[str]) -> None:
    """Apply key=value overrides to a TrainConfig. Supports dotted keys."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, value_str = override.split("=", 1)
        value = _parse_value(value_str)

        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise ValueError(f"Unknown config key: {key}")
            obj = getattr(obj, part)
        final_key = parts[-1]

        # Allow top-level shorthand for nested keys (e.g. lr=1e-4)
        if not hasattr(obj, final_key):
            # Try nested groups
            for group_name in ("data", "model", "loss"):
                group = getattr(cfg, group_name)
                if hasattr(group, final_key):
                    setattr(group, final_key, value)
                    break
            else:
                raise ValueError(f"Unknown config key: {key}")
        else:
            setattr(obj, final_key, value)


def config_to_dict(cfg: TrainConfig) -> dict:
    """Serialize a TrainConfig to a plain dict (for checkpoint saving)."""
    return asdict(cfg)
