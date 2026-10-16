"""Configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from streamguard.config import AppConfig, ConfigError, default_config, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_default_config_is_valid() -> None:
    config = default_config()
    config.validate()
    assert config.simulator.n_reference > 0
    assert config.detector.features == config.drift.features


def test_rejects_invalid_values() -> None:
    with pytest.raises(ConfigError):
        AppConfig(simulator=_simulator(ar1_phi=1.5)).validate()
    with pytest.raises(ConfigError):
        AppConfig(detector=_detector(contamination=0.9)).validate()
    with pytest.raises(ConfigError):
        AppConfig(drift=_drift(correction="fdr")).validate()
    with pytest.raises(ConfigError):
        AppConfig(drift=_drift(window_size=50, min_window_size=100)).validate()
    with pytest.raises(ConfigError):
        AppConfig(simulator=_simulator(anomaly_kinds=("nonsense",))).validate()
    with pytest.raises(ConfigError):
        AppConfig(simulator=_simulator(multiplicative=("not_a_feature",))).validate()


def _simulator(**overrides: object):
    from dataclasses import replace

    from streamguard.config import SimulatorConfig

    return replace(SimulatorConfig(), **overrides)


def _detector(**overrides: object):
    from dataclasses import replace

    from streamguard.config import DetectorConfig

    return replace(DetectorConfig(), **overrides)


def _drift(**overrides: object):
    from dataclasses import replace

    from streamguard.config import DriftConfig

    return replace(DriftConfig(), **overrides)


def test_dict_round_trip_preserves_values() -> None:
    config = default_config()
    config.simulator.seed = 12345
    config.drift.correction = "bonferroni"
    restored = AppConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert restored.simulator.shift == config.simulator.shift


def test_from_dict_restores_shift_and_tuples() -> None:
    restored = AppConfig.from_dict(
        {
            "simulator": {"seed": 7, "shift": {"enabled": False, "latency_mult": 1.30}},
            "detector": {"features": ["latency_ms"]},
            "drift": {"features": ["latency_ms"], "correction": "none"},
        }
    )
    assert restored.simulator.seed == 7
    assert restored.simulator.shift.enabled is False
    assert restored.simulator.shift.latency_mult == 1.30
    assert restored.detector.features == ("latency_ms",)
    assert restored.drift.correction == "none"


@pytest.mark.parametrize("name", ["default.toml", "sensitive.toml"])
def test_committed_configs_load(name: str) -> None:
    config = load_config(CONFIG_DIR / name)
    config.validate()
    assert config.drift.min_window_size <= config.drift.window_size
    assert config.simulator.n_point_anomalies >= 0


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(CONFIG_DIR / "does_not_exist.toml")
