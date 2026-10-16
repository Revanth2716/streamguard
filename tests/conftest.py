"""Shared fixtures.

The session-scoped fixtures are read-only by contract: tests must never mutate a
shared dataset or pipeline result, because they are reused across the suite.
Configurations that need a tweak use :func:`make_config` instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from streamguard.config import AppConfig, default_config
from streamguard.pipeline import PipelineResult, run_pipeline
from streamguard.simulator import Dataset, generate_dataset


def make_config(
    simulator: dict[str, object] | None = None,
    detector: dict[str, object] | None = None,
    drift: dict[str, object] | None = None,
) -> AppConfig:
    """A small, fast, fully validated configuration for tests."""
    config = default_config()
    config.simulator.n_reference = 400
    config.simulator.n_stream = 400
    config.simulator.n_point_anomalies = 10
    config.detector.if_n_estimators = 60
    config.drift.window_size = 100
    config.drift.step = 100
    config.drift.min_window_size = 60
    for section, values in (
        (config.simulator, simulator),
        (config.detector, detector),
        (config.drift, drift),
    ):
        for key, value in (values or {}).items():
            if not hasattr(section, key):
                raise AttributeError(f"unknown configuration field: {key}")
            setattr(section, key, value)
    config.output_dir = "reports"
    config.alert_state_path = "data/processed/alert_state.json"
    config.validate()
    return config


@pytest.fixture(scope="session")
def small_config() -> AppConfig:
    return make_config()


@pytest.fixture
def config_factory():
    return make_config


@pytest.fixture(scope="session")
def small_dataset(small_config: AppConfig) -> Dataset:
    return generate_dataset(small_config.simulator)


@pytest.fixture(scope="session")
def small_result(small_config: AppConfig, small_dataset: Dataset) -> PipelineResult:
    return run_pipeline(small_config, dataset=small_dataset)


@pytest.fixture
def alert_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "alert_state.json"
