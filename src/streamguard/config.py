"""Configuration objects for StreamGuard.

Configuration is intentionally plain (dataclasses + TOML) so that every run can
be reproduced from a single file that is safe to commit: seeds, window sizes,
thresholds and correction methods are all recorded next to the artifacts they
produced.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FEATURE_COLUMNS: tuple[str, ...] = ("latency_ms", "cpu_util", "mem_util", "request_rate")

DEFAULT_BASELINES: dict[str, float] = {
    "latency_ms": 118.0,
    "cpu_util": 0.46,
    "mem_util": 0.58,
    "request_rate": 640.0,
}

#: Relative noise fractions (multiples of each feature's own baseline).
DEFAULT_NOISE: dict[str, float] = {
    "latency_ms": 0.05,
    "cpu_util": 0.028,
    "mem_util": 0.005,
    "request_rate": 0.05,
}

DEFAULT_DIURNAL_AMPLITUDE: dict[str, float] = {
    "latency_ms": 0.06,
    "cpu_util": 0.08,
    "mem_util": 0.015,
    "request_rate": 0.20,
}

DEFAULT_LOAD_SENSITIVITY: dict[str, float] = {
    "latency_ms": 0.12,
    "cpu_util": 0.10,
    "mem_util": 0.03,
    "request_rate": 0.15,
}

#: Features generated with a log-normal (strictly positive) model. Latency and
#: throughput are naturally right-skewed and can never be negative; utilisation
#: is modelled additively and clipped to a physical 0-100% range.
MULTIPLICATIVE_FEATURES: tuple[str, ...] = ("latency_ms", "request_rate")

VALID_CORRECTIONS = ("none", "bonferroni", "holm")
VALID_ANOMALY_KINDS = ("spike", "burst", "dropout", "latency_storm")


class ConfigError(ValueError):
    """Raised when a configuration value is out of range or inconsistent."""


@dataclass
class ShiftConfig:
    """Parameters of the injected distribution shift (drift) block."""

    enabled: bool = True
    #: Fraction of the stream that precedes the injected shift.
    start_fraction: float = 0.60
    latency_mult: float = 1.18
    # NOTE: magnitudes are deliberately moderate. A larger step change makes the
    # fixed-reference detectors flag almost the whole shifted period as point
    # anomalies, which inflates their false positives without testing anything
    # new. Measured effects are reported in reports/metrics/evaluation.json.
    cpu_delta: float = 0.055
    mem_delta: float = 0.018
    request_rate_mult: float = 1.10
    noise_mult: float = 1.15
    component: str = "mean_and_noise_shift"

    def validate(self) -> None:
        if not 0.0 < self.start_fraction < 1.0:
            raise ConfigError("simulator.shift.start_fraction must be within (0, 1)")
        if self.latency_mult <= 0 or self.request_rate_mult <= 0:
            raise ConfigError("simulator.shift multipliers must be positive")
        if self.noise_mult <= 0:
            raise ConfigError("simulator.shift.noise_mult must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulatorConfig:
    """Everything needed to generate a reproducible, labelled event stream."""

    seed: int = 20261016
    n_reference: int = 1200
    n_stream: int = 1400
    interval_seconds: int = 60
    #: Period, in observations, of the recurring usage cycle. It stands in for a
    #: daily seasonality; it is deliberately short relative to the drift window
    #: so that phase differences between the reference block and a window cannot
    #: masquerade as distribution drift (see LIMITATIONS.md).
    cycle_points: int = 48
    start_timestamp: str = "2026-01-01T00:00:00"
    #: AR(1) coefficient of the latent load. Kept moderate (0.35) because the
    #: two-sample KS test assumes independent samples: strong autocorrelation
    #: shrinks the effective sample size and makes the test over-reject. The
    #: effect and its consequence were measured, see LIMITATIONS.md.
    ar1_phi: float = 0.35
    load_sigma: float = 1.0
    multiplicative: tuple[str, ...] = MULTIPLICATIVE_FEATURES
    anomaly_fraction_pre_drift: float = 0.7
    n_point_anomalies: int = 24
    anomaly_min_len: int = 1
    anomaly_max_len: int = 8
    anomaly_sigma_low: float = 2.6
    anomaly_sigma_high: float = 9.0
    anomaly_kinds: tuple[str, ...] = VALID_ANOMALY_KINDS
    baselines: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BASELINES))
    noise: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NOISE))
    diurnal_amplitude: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DIURNAL_AMPLITUDE)
    )
    load_sensitivity: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_LOAD_SENSITIVITY)
    )
    shift: ShiftConfig = field(default_factory=ShiftConfig)

    def validate(self) -> None:
        if self.n_reference < 100:
            raise ConfigError("simulator.n_reference must be at least 100")
        if self.n_stream < 100:
            raise ConfigError("simulator.n_stream must be at least 100")
        if self.interval_seconds < 1:
            raise ConfigError("simulator.interval_seconds must be >= 1")
        if self.cycle_points < 4:
            raise ConfigError("simulator.cycle_points must be >= 4")
        if not 0.0 <= self.ar1_phi < 1.0:
            raise ConfigError("simulator.ar1_phi must be within [0, 1)")
        if not 0.0 <= self.anomaly_fraction_pre_drift <= 1.0:
            raise ConfigError("simulator.anomaly_fraction_pre_drift must be within [0, 1]")
        if self.n_point_anomalies < 0:
            raise ConfigError("simulator.n_point_anomalies must be >= 0")
        if self.anomaly_min_len < 1 or self.anomaly_max_len < self.anomaly_min_len:
            raise ConfigError("simulator anomaly length bounds are inconsistent")
        if self.anomaly_sigma_low <= 0 or self.anomaly_sigma_high < self.anomaly_sigma_low:
            raise ConfigError("simulator anomaly magnitude bounds are inconsistent")
        unknown = set(self.anomaly_kinds) - set(VALID_ANOMALY_KINDS)
        if unknown:
            raise ConfigError(f"simulator.anomaly_kinds contains unknown kinds: {sorted(unknown)}")
        unknown_features = set(self.multiplicative) - set(FEATURE_COLUMNS)
        if unknown_features:
            raise ConfigError(
                f"simulator.multiplicative contains unknown features: {sorted(unknown_features)}"
            )
        for mapping in (self.baselines, self.noise, self.diurnal_amplitude, self.load_sensitivity):
            missing = set(FEATURE_COLUMNS) - set(mapping)
            if missing:
                raise ConfigError(f"simulator config is missing features: {sorted(missing)}")
        self.shift.validate()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["anomaly_kinds"] = list(self.anomaly_kinds)
        data["multiplicative"] = list(self.multiplicative)
        return data


@dataclass
class DetectorConfig:
    """Thresholds and fitting choices for the anomaly detectors."""

    features: tuple[str, ...] = FEATURE_COLUMNS
    zscore_window: int = 120
    zscore_threshold: float = 3.5
    zscore_min_periods: int = 30
    contamination: float = 0.02
    if_n_estimators: int = 200
    if_random_state: int = 20261016
    if_severity_sigma: tuple[float, float, float] = (2.0, 3.0, 4.0)
    static_threshold_quantile: float = 0.995

    def validate(self) -> None:
        if self.zscore_window < 10:
            raise ConfigError("detector.zscore_window must be >= 10")
        if self.zscore_threshold <= 0:
            raise ConfigError("detector.zscore_threshold must be positive")
        if not 0.0 < self.contamination < 0.5:
            raise ConfigError("detector.contamination must be within (0, 0.5)")
        if self.if_n_estimators < 10:
            raise ConfigError("detector.if_n_estimators must be >= 10")
        if not 0.5 < self.static_threshold_quantile < 1.0:
            raise ConfigError("detector.static_threshold_quantile must be within (0.5, 1)")
        if sorted(self.if_severity_sigma) != list(self.if_severity_sigma):
            raise ConfigError("detector.if_severity_sigma must be non-decreasing")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["features"] = list(self.features)
        data["if_severity_sigma"] = list(self.if_severity_sigma)
        return data


@dataclass
class DriftConfig:
    """Reference-versus-window comparison settings for the drift detector."""

    features: tuple[str, ...] = FEATURE_COLUMNS
    window_size: int = 200
    step: int = 200
    alpha: float = 0.01
    correction: str = "holm"
    min_window_size: int = 120
    psi_bins: int = 10
    #: Practical-significance floor on the KS statistic, measured on the default
    #: seed: with 200-point windows the largest clean-window KS statistic is
    #: 0.124, while the injected shift reaches 0.418-0.534. At these settings the
    #: Holm-corrected p-value already suppresses every clean window, so the floor
    #: is a margin rather than the binding guard; it binds when the telemetry is
    #: more autocorrelated (clean-window maximum 0.152 at phi=0.50 and 0.195 at
    #: phi=0.72, where a 0.10 floor would raise false drift alerts).
    #: See reports/metrics/sensitivity.json for the raw measurements.
    effect_size_floor: float = 0.20

    def validate(self) -> None:
        if self.window_size < 10:
            raise ConfigError("drift.window_size must be >= 10")
        if self.step < 1:
            raise ConfigError("drift.step must be >= 1")
        if self.min_window_size < 10:
            raise ConfigError("drift.min_window_size must be >= 10")
        if self.min_window_size > self.window_size:
            raise ConfigError("drift.min_window_size must not exceed drift.window_size")
        if not 0.0 < self.alpha < 0.5:
            raise ConfigError("drift.alpha must be within (0, 0.5)")
        if self.correction not in VALID_CORRECTIONS:
            raise ConfigError(f"drift.correction must be one of {VALID_CORRECTIONS}")
        if self.psi_bins < 2:
            raise ConfigError("drift.psi_bins must be >= 2")
        if self.effect_size_floor < 0:
            raise ConfigError("drift.effect_size_floor must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["features"] = list(self.features)
        return data


@dataclass
class AppConfig:
    """Top-level configuration bundle."""

    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    output_dir: str = "reports"
    alert_state_path: str = "data/processed/alert_state.json"

    def validate(self) -> None:
        self.simulator.validate()
        self.detector.validate()
        self.drift.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulator": self.simulator.to_dict(),
            "detector": self.detector.to_dict(),
            "drift": self.drift.to_dict(),
            "output_dir": self.output_dir,
            "alert_state_path": self.alert_state_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        sim = data.get("simulator", {})
        shift = sim.pop("shift", {}) if isinstance(sim, dict) else {}
        det = dict(data.get("detector", {}))
        drift = dict(data.get("drift", {}))
        for mapping in (det, drift):
            for key in ("features",):
                if key in mapping:
                    mapping[key] = tuple(mapping[key])
        if "if_severity_sigma" in det:
            det["if_severity_sigma"] = tuple(det["if_severity_sigma"])
        for key in ("anomaly_kinds", "multiplicative"):
            if key in sim:
                sim[key] = tuple(sim[key])
        cfg = cls(
            simulator=SimulatorConfig(**sim, shift=ShiftConfig(**shift))
            if shift
            else SimulatorConfig(**sim),
            detector=DetectorConfig(**det),
            drift=DriftConfig(**drift),
            output_dir=data.get("output_dir", "reports"),
            alert_state_path=data.get("alert_state_path", "data/processed/alert_state.json"),
        )
        cfg.validate()
        return cfg


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an :class:`AppConfig` from a TOML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return AppConfig.from_dict(raw)


def default_config() -> AppConfig:
    """Return the validated default configuration."""
    cfg = AppConfig()
    cfg.validate()
    return cfg
