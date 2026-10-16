"""Anomaly detector tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamguard.config import AppConfig, DetectorConfig
from streamguard.detector import (
    OUTPUT_COLUMNS,
    DummyDetector,
    IsolationForestDetector,
    RollingZScoreDetector,
    StaticPercentileDetector,
    build_detectors,
    run_detectors,
    severity_from_sigma,
)
from streamguard.simulator import FEATURE_COLUMNS


def synthetic_reference(rows: int = 600, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "latency_ms": rng.normal(120.0, 8.0, rows),
        "cpu_util": rng.normal(0.45, 0.02, rows),
        "mem_util": rng.normal(0.60, 0.01, rows),
        "request_rate": rng.normal(600.0, 30.0, rows),
    }
    frame = pd.DataFrame(data)
    frame["timestamp"] = pd.date_range("2026-01-01", periods=rows, freq="min")
    frame["event_index"] = np.arange(rows)
    return frame


def test_all_detectors_share_one_output_schema(small_dataset) -> None:
    reference = small_dataset.reference
    stream = small_dataset.stream
    results = run_detectors(reference, stream, DetectorConfig(if_n_estimators=40))
    assert set(results) == {"dummy", "static_percentile", "rolling_zscore", "isolation_forest"}
    for name, frame in results.items():
        assert set(OUTPUT_COLUMNS).issubset(frame.columns), name
        assert len(frame) == len(stream)
        assert frame["detector"].unique().tolist() == [name]
        assert frame["is_anomaly"].dtype == bool
        assert frame["severity"].isin(["low", "medium", "high", "critical"]).all()
        assert frame["event_index"].tolist() == stream["event_index"].tolist()
        flagged = frame.loc[frame["is_anomaly"]]
        assert (flagged["reason"].str.len() > 0).all(), name
        assert (flagged["top_feature"] != "").all(), name


def test_dummy_detector_never_flags_but_still_implements_the_interface(small_dataset) -> None:
    detector = DummyDetector(DetectorConfig())
    with pytest.raises(RuntimeError):
        detector.score(small_dataset.stream)
    output = detector.fit(small_dataset.reference).score(small_dataset.stream)
    assert not output["is_anomaly"].any()
    assert (output["score"] == 0).all()


def test_zscore_flags_a_large_spike_and_names_the_feature() -> None:
    reference = synthetic_reference()
    stream = reference.tail(300).reset_index(drop=True).copy()
    stream.loc[200, "latency_ms"] = 120.0 + 12.0 * 8.0  # ~12 sigma above the baseline
    detector = RollingZScoreDetector(DetectorConfig(zscore_window=100, zscore_threshold=3.5))
    output = detector.fit(reference).score(stream)

    assert bool(output.loc[200, "is_anomaly"])
    assert output.loc[200, "top_feature"] == "latency_ms"
    assert output.loc[200, "score"] >= 3.5
    assert "latency_ms" in output.loc[200, "reason"]
    assert output.loc[200, "severity"] in {"medium", "high", "critical"}
    # Untouched neighbours are not flagged.
    assert not output.loc[190:199, "is_anomaly"].any()
    assert output.loc[200, "z_latency_ms"] > 10


def test_zscore_threshold_is_configurable() -> None:
    reference = synthetic_reference()
    stream = reference.tail(300).reset_index(drop=True).copy()
    stream.loc[200, "cpu_util"] = 0.45 + 0.02 * 4.0  # ~4 sigma
    strict = RollingZScoreDetector(DetectorConfig(zscore_threshold=6.0, zscore_window=100))
    relaxed = RollingZScoreDetector(DetectorConfig(zscore_threshold=3.0, zscore_window=100))
    assert not strict.fit(reference).score(stream).loc[200, "is_anomaly"]
    assert relaxed.fit(reference).score(stream).loc[200, "is_anomaly"]


def test_isolation_forest_flags_an_injected_multivariate_outlier() -> None:
    reference = synthetic_reference()
    stream = reference.tail(300).reset_index(drop=True).copy()
    stream.loc[150, ["latency_ms", "cpu_util", "mem_util"]] = [200.0, 0.9, 0.8]
    detector = IsolationForestDetector(
        DetectorConfig(contamination=0.01, if_n_estimators=60, if_random_state=3)
    )
    output = detector.fit(reference).score(stream)
    assert bool(output.loc[150, "is_anomaly"])
    assert "isolation forest outlier" in output.loc[150, "reason"]
    assert output.loc[150, "score_sigma"] > 2.0
    assert output.loc[150, "top_feature"] in FEATURE_COLUMNS


def test_isolation_forest_respects_contamination_prior() -> None:
    reference = synthetic_reference()
    for contamination in (0.01, 0.05):
        detector = IsolationForestDetector(
            DetectorConfig(contamination=contamination, if_n_estimators=60, if_random_state=1)
        )
        output = detector.fit(reference).score(reference)
        rate = float(output["is_anomaly"].mean())
        # The forest's decision offset is calibrated on the reference, so scoring
        # the reference reproduces roughly the contamination prior.
        assert abs(rate - contamination) < 0.02


def test_isolation_forest_requires_a_reference_block() -> None:
    with pytest.raises(ValueError):
        IsolationForestDetector(DetectorConfig()).fit(synthetic_reference(rows=5))


def test_static_percentile_flags_an_out_of_envelope_value() -> None:
    reference = synthetic_reference()
    stream = reference.tail(300).reset_index(drop=True).copy()
    stream.loc[100, "request_rate"] = 1200.0
    detector = StaticPercentileDetector(DetectorConfig(static_threshold_quantile=0.995))
    output = detector.fit(reference).score(stream)
    assert bool(output.loc[100, "is_anomaly"])
    assert "static percentile" in output.loc[100, "reason"]
    assert float(output["is_anomaly"].mean()) < 0.05


def test_severity_mapping_is_monotone() -> None:
    bounds = (2.0, 3.0, 4.0)
    levels = [severity_from_sigma(value, bounds) for value in (0.5, 2.5, 3.5, 9.0)]
    assert levels == ["low", "medium", "high", "critical"]
    assert severity_from_sigma(2.0, bounds) == "medium"


def test_missing_feature_column_is_rejected(small_dataset) -> None:
    broken = small_dataset.reference.drop(columns=["cpu_util"])
    with pytest.raises(KeyError):
        run_detectors(broken, small_dataset.stream, DetectorConfig(if_n_estimators=20))


def test_severity_agrees_with_score_ordering(small_dataset) -> None:
    output = run_detectors(small_dataset.reference, small_dataset.stream, DetectorConfig())[
        "rolling_zscore"
    ]
    flagged = output.loc[output["is_anomaly"]].copy()
    if len(flagged) < 2:
        pytest.skip("not enough flagged rows in this configuration")
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    ordered = flagged.sort_values("score")
    assert ordered["severity"].map(rank).is_monotonic_increasing


def test_build_detectors_is_stable() -> None:
    config = DetectorConfig()
    assert list(build_detectors(config)) == list(build_detectors(config))
    assert all(hasattr(detector, "fit") for detector in build_detectors(config).values())


def test_detector_config_defaults_are_independent_of_the_app_config() -> None:
    config = AppConfig()
    assert config.detector.zscore_threshold > 0
