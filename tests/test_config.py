"""Tests for postprocessing config: loading, overrides, serialization."""

import pytest
import yaml

from postprocessing.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
    apply_overrides,
    config_to_dict,
    load_config,
    validate_config,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_train_config_defaults(self):
        cfg = TrainConfig()
        assert cfg.batch_size == 16
        assert cfg.lr == 1e-3
        assert cfg.epochs == 30
        assert cfg.device == "cuda"

    def test_data_config_defaults(self):
        cfg = DataConfig()
        assert cfg.rates == ["r2"]
        assert cfg.patch_size == 64
        assert cfg.use_curvature is True

    def test_nested_defaults_accessible(self):
        cfg = TrainConfig()
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.loss, LossConfig)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_minimal_yaml(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("batch_size: 8\n")
        cfg = load_config(str(config_file))
        assert cfg.batch_size == 8
        # Unspecified fields keep defaults
        assert cfg.lr == 1e-3

    def test_load_nested_yaml(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "data:\n  rates: [r2, r4]\n  patch_size: 32\nmodel:\n  model_type: film_head_v2\n"
        )
        cfg = load_config(str(config_file))
        assert cfg.data.rates == ["r2", "r4"]
        assert cfg.data.patch_size == 32
        assert cfg.model.model_type == "film_head_v2"

    def test_load_numeric_string_coercion(self, tmp_path):
        """PyYAML parses '1e-3' as string; config should coerce to float."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("lr: '1e-3'\n")
        cfg = load_config(str(config_file))
        assert cfg.lr == pytest.approx(1e-3)

    def test_load_empty_yaml(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("")
        cfg = load_config(str(config_file))
        # All defaults
        assert cfg.batch_size == 16

    def test_load_unknown_key_raises(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("nonexistent_key: 42\n")
        with pytest.raises(ValueError, match="Unknown config key"):
            load_config(str(config_file))

    def test_load_rate_weights(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "data:\n  rates: [r2, r4, r6]\nloss:\n  rate_weights: [1.0, 0.3, 0.05]\n"
        )
        cfg = load_config(str(config_file))
        assert cfg.loss.rate_weights == [1.0, 0.3, 0.05]


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_top_level_override(self):
        cfg = TrainConfig()
        apply_overrides(cfg, ["batch_size=32"])
        assert cfg.batch_size == 32

    def test_float_override(self):
        cfg = TrainConfig()
        apply_overrides(cfg, ["lr=5e-4"])
        assert cfg.lr == pytest.approx(5e-4)

    def test_bool_override_true(self):
        cfg = TrainConfig()
        apply_overrides(cfg, ["data.augment=false"])
        assert cfg.data.augment is False

    def test_bool_override_yes(self):
        cfg = TrainConfig()
        apply_overrides(cfg, ["data.augment=yes"])
        assert cfg.data.augment is True

    def test_dotted_nested_override(self):
        cfg = TrainConfig()
        apply_overrides(cfg, ["data.patch_size=128"])
        assert cfg.data.patch_size == 128

    def test_list_override(self):
        cfg = TrainConfig()
        apply_overrides(cfg, ["data.rates=[r2,r4,r6]"])
        assert cfg.data.rates == ["r2", "r4", "r6"]

    def test_none_override(self):
        cfg = TrainConfig()
        cfg.resume = "some/path.pt"
        apply_overrides(cfg, ["resume=none"])
        assert cfg.resume is None

    def test_shorthand_nested_key(self):
        """Unqualified keys that exist in a subgroup are resolved automatically."""
        cfg = TrainConfig()
        apply_overrides(cfg, ["patch_size=32"])
        assert cfg.data.patch_size == 32

    def test_unknown_key_raises(self):
        cfg = TrainConfig()
        with pytest.raises(ValueError, match="Unknown config key"):
            apply_overrides(cfg, ["nonexistent=1"])

    def test_missing_equals_raises(self):
        cfg = TrainConfig()
        with pytest.raises(ValueError, match="key=value"):
            apply_overrides(cfg, ["batch_size"])

    def test_multiple_overrides(self):
        cfg = TrainConfig()
        apply_overrides(cfg, ["batch_size=8", "lr=1e-4", "data.patch_size=32"])
        assert cfg.batch_size == 8
        assert cfg.lr == pytest.approx(1e-4)
        assert cfg.data.patch_size == 32


# ---------------------------------------------------------------------------
# config_to_dict / roundtrip
# ---------------------------------------------------------------------------


class TestConfigToDict:
    def test_returns_dict(self):
        cfg = TrainConfig()
        d = config_to_dict(cfg)
        assert isinstance(d, dict)

    def test_nested_structure_preserved(self):
        cfg = TrainConfig()
        d = config_to_dict(cfg)
        assert "data" in d
        assert "model" in d
        assert "loss" in d
        assert d["data"]["patch_size"] == cfg.data.patch_size

    def test_roundtrip_via_yaml(self, tmp_path):
        """Config -> dict -> YAML -> load_config -> same values."""
        cfg = TrainConfig()
        cfg.batch_size = 8
        cfg.data.rates = ["r2", "r4"]
        cfg.model.model_type = "film_head_v2"

        config_file = tmp_path / "roundtrip.yaml"
        config_file.write_text(yaml.dump(config_to_dict(cfg)))

        cfg2 = load_config(str(config_file))
        assert cfg2.batch_size == cfg.batch_size
        assert cfg2.data.rates == cfg.data.rates
        assert cfg2.model.model_type == cfg.model.model_type


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_defaults_pass(self):
        validate_config(TrainConfig())  # should not raise

    def test_invalid_model_type(self):
        cfg = TrainConfig()
        cfg.model.model_type = "nonexistent"
        with pytest.raises(ValueError, match="model.model_type"):
            validate_config(cfg)

    def test_invalid_loss_type(self):
        cfg = TrainConfig()
        cfg.loss.loss_type = "bad_loss"
        with pytest.raises(ValueError, match="loss.loss_type"):
            validate_config(cfg)

    def test_batch_size_zero(self):
        cfg = TrainConfig()
        cfg.batch_size = 0
        with pytest.raises(ValueError, match="batch_size"):
            validate_config(cfg)

    def test_lr_zero(self):
        cfg = TrainConfig()
        cfg.lr = 0.0
        with pytest.raises(ValueError, match="lr"):
            validate_config(cfg)

    def test_epochs_zero(self):
        cfg = TrainConfig()
        cfg.epochs = 0
        with pytest.raises(ValueError, match="epochs"):
            validate_config(cfg)

    def test_rate_weights_length_mismatch(self):
        cfg = TrainConfig()
        cfg.data.rates = ["r2", "r4"]
        cfg.loss.rate_weights = [1.0, 0.5, 0.1]  # 3 weights for 2 rates
        with pytest.raises(ValueError, match="rate_weights"):
            validate_config(cfg)

    def test_rate_weights_matching_length_passes(self):
        cfg = TrainConfig()
        cfg.data.rates = ["r2", "r4", "r6"]
        cfg.loss.rate_weights = [1.0, 0.3, 0.05]
        validate_config(cfg)  # should not raise

    def test_empty_rate_weights_passes(self):
        """Empty rate_weights (uniform) is always valid regardless of rates."""
        cfg = TrainConfig()
        cfg.data.rates = ["r2", "r4"]
        cfg.loss.rate_weights = []
        validate_config(cfg)  # should not raise

    def test_multiple_errors_reported_together(self):
        cfg = TrainConfig()
        cfg.model.model_type = "bad"
        cfg.loss.loss_type = "bad"
        cfg.batch_size = -1
        with pytest.raises(ValueError) as exc_info:
            validate_config(cfg)
        msg = str(exc_info.value)
        assert "model.model_type" in msg
        assert "loss.loss_type" in msg
        assert "batch_size" in msg

    def test_load_config_validates(self, tmp_path):
        """load_config must reject invalid model_type at load time."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("model:\n  model_type: nonexistent\n")
        with pytest.raises(ValueError, match="model.model_type"):
            load_config(str(config_file))


# ---------------------------------------------------------------------------
# logging_utils
# ---------------------------------------------------------------------------


class TestLoggingUtils:
    def test_setup_logging_idempotent(self):
        """Calling setup_logging twice should not add duplicate handlers."""
        import logging

        from postprocessing.logging_utils import setup_logging

        setup_logging()
        n_after_first = len(logging.getLogger().handlers)
        setup_logging()
        n_after_second = len(logging.getLogger().handlers)
        assert n_after_first == n_after_second

    def test_setup_logging_file(self, tmp_path):
        """File handler is added when log_file is provided."""
        import logging

        from postprocessing.logging_utils import setup_logging

        log_path = str(tmp_path / "test.log")
        setup_logging(log_path)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == log_path
