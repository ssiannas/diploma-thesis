"""Typed configuration for training pipeline.

Replaces 36 CLI args with grouped dataclasses + YAML config files.
Usage: python postprocessing/train.py --config configs/cd_r2.yaml
"""

import typing
from dataclasses import asdict, dataclass, field, fields
from typing import Any, List, Optional

import yaml

_VALID_MODEL_TYPES = frozenset(
    ["unet", "film", "film_head", "film_head_v2", "film_head_v3", "occupancy"]
)
_VALID_LOSS_TYPES = frozenset(
    ["cd", "sw_cd", "focal_cd", "edge_cd", "cd_lap", "cd_vlap", "cls_focal"]
)


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
    rate_jitter: float = 0.0  # std of Gaussian noise on log(bpp) during training


@dataclass
class ModelConfig:
    model_type: str = "unet"  # unet | film | film_head
    max_displacement: float = 5.0
    film_embed_dim: int = 64
    film_rate_repr: str = "bpp"  # bpp | onehot
    quantize_output: bool = False


@dataclass
class LossConfig:
    loss_type: str = "cd"  # cd | sw_cd | focal_cd | edge_cd | cd_lap | cd_vlap
    sw_mode: str = "linear"  # linear | sqrt | log (only for sw_cd)
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
    # Per-rate loss weighting: order matches data.rates (empty = uniform)
    rate_weights: List[float] = field(default_factory=list)
    # Kendall learned rate weighting: replaces static rate_weights when True
    use_kendall_rates: bool = False
    # Dynamic detached rate weighting: weights = detach(L_i) / mean, clamped [0.1, 10]
    use_dynamic_rates: bool = False
    # DWA epoch-level weighting (Liu et al., CVPR 2019): w_i ∝ exp(L_i(t-1)/L_i(t-2)/T)
    use_dwa: bool = False
    dwa_temp: float = 2.0  # softmax temperature; higher = smoother adaptation
    dwa_floor: List[float] = field(
        default_factory=list
    )  # per-rate min weights (empty = no floor)
    # CE auxiliary loss weight for V3 classification head (0 = disabled)
    ce_weight: float = 0.0
    # Focal gamma for CE loss (0 = standard CE, 2 = focal)
    ce_focal_gamma: float = 0.0
    # Global scale applied to CD loss (use < 1 when CE is dominant)
    cd_loss_scale: float = 1.0
    # GradNorm rate weighting (Chen et al., ICML 2018): weights adapted via FiLM activation grads
    use_film_gradnorm: bool = False
    film_gradnorm_alpha: float = (
        1.5  # task asymmetry strength (higher = more aggressive balancing)
    )
    film_gradnorm_lr: float = 0.01  # lr for weight updates (separate from model lr)
    # Classification model (cls_focal): prior probability of zero displacement
    cls_p_zero: float = 0.70  # set from histogram Step 0 output
    # Gaussian label smoothing sigma (0.0 = hard one-hot labels, recommended)
    cls_sigma: float = 0.0


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    batch_size: int = 16
    lr: float = 1e-3
    warmup_epochs: int = 5
    epochs: int = 30
    num_workers: int = -1  # -1 = auto (os.cpu_count())
    device: str = "cuda"
    save_dir: str = "models/postprocessing"
    log_file: Optional[str] = None
    resume: Optional[str] = None
    weight_decay: float = 1e-2
    rate_stratified: bool = False
    encoder_lr_scale: float = 1.0  # 0.1 for stage 2 (freeze encoder)
    pretrained_backbone: Optional[str] = (
        None  # path to SparseUNet checkpoint for two-stage
    )


def _unwrap_optional(tp: type) -> type:
    """Return T if tp is Optional[T] (i.e. Union[T, None]), else tp unchanged."""
    if hasattr(tp, "__origin__") and tp.__origin__ is typing.Union:
        non_none = [a for a in tp.__args__ if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _coerce_type(value: Any, field_type: type) -> Any:
    """Coerce a value to match the expected field type (handles YAML quirks).

    Unwraps Optional[T] before checking so that Optional[float] fields accept
    integer YAML values and quoted numeric strings.
    """
    ft = _unwrap_optional(field_type)
    # PyYAML parses "1e-3" as str, not float -- coerce numeric strings
    if ft is float and isinstance(value, str):
        return float(value)
    if ft is int and isinstance(value, str):
        return int(value)
    if ft is float and isinstance(value, int):
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


def validate_config(cfg: TrainConfig) -> None:
    """Raise ValueError if cfg contains invalid or inconsistent values."""
    errors = []
    if cfg.model.model_type not in _VALID_MODEL_TYPES:
        errors.append(
            f"model.model_type={cfg.model.model_type!r} not in {sorted(_VALID_MODEL_TYPES)}"
        )
    if cfg.loss.loss_type not in _VALID_LOSS_TYPES:
        errors.append(
            f"loss.loss_type={cfg.loss.loss_type!r} not in {sorted(_VALID_LOSS_TYPES)}"
        )
    if cfg.batch_size <= 0:
        errors.append(f"batch_size={cfg.batch_size} must be > 0")
    if cfg.lr <= 0:
        errors.append(f"lr={cfg.lr} must be > 0")
    if cfg.epochs <= 0:
        errors.append(f"epochs={cfg.epochs} must be > 0")
    if cfg.loss.rate_weights and len(cfg.loss.rate_weights) != len(cfg.data.rates):
        errors.append(
            f"loss.rate_weights has {len(cfg.loss.rate_weights)} entries "
            f"but data.rates has {len(cfg.data.rates)}"
        )
    if errors:
        raise ValueError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def load_config(path: str) -> TrainConfig:
    """Load a YAML config file into a validated TrainConfig dataclass."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = TrainConfig()
    _merge_dict_into_dataclass(cfg, raw)
    validate_config(cfg)
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
