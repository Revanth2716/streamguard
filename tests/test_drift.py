"""Drift detector tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamguard.config import DriftConfig
from streamguard.drift import (
    adjust_pvalues,
    compare_window,
    drift_feature_frame,
    drift_summary,
    population_stability_index,
    scan_drift,
)
from streamguard.simulator import FEATURE_COLUMNS


def frame_from(rng: np.random.Generator, rows: int, shift: float = 0.0) -> pd.DataFrame:
    data = {
        "latency_ms": rng.normal(120.0 + shift, 8.0, rows),
        "cpu_util": rng.normal(0.45, 0.02, rows),
        "mem_util": rng.normal(0.60, 0.01, rows),
        "request_rate": rng.normal(600.0, 30.0, rows),
    }
    frame = pd.DataFrame(data)
    frame["timestamp"] = pd.date_range("2026-01-01", periods=rows, freq="min")
    return frame


def test_holm_is_never_less_powerful_than_bonferroni() -> None:
    raw = np.array([0.0001, 0.004, 0.02, 0.2, 0.9])
    bonferroni = adjust_pvalues(raw, "bonferroni")
    holm = adjust_pvalues(raw, "holm")
    assert np.all(holm <= bonferroni + 1e-12)
    assert np.all(holm >= raw - 1e-12)
    assert np.all(holm <= 1.0)
    assert np.allclose(bonferroni, np.minimum(raw * raw.size, 1.0))
    assert np.allclose(adjust_pvalues(raw, "none"), raw)


def test_correction_is_order_independent() -> None:
    raw = np.array([0.02, 0.0004, 0.3, 0.0001])
    adjusted = adjust_pvalues(raw, "holm")
    shuffled = [2, 0, 3, 1]
    reshuffled = adjust_pvalues(raw[shuffled], "holm")
    for new_position, old_position in enumerate(shuffled):
        assert adjusted[old_position] == pytest.approx(reshuffled[new_position])


def test_unknown_correction_is_rejected() -> None:
    with pytest.raises(ValueError):
        adjust_pvalues(np.array([0.1, 0.2]), "fdr")


def test_identical_distributions_do_not_trigger_drift() -> None:
    rng = np.random.default_rng(11)
    reference = frame_from(rng, 1500)
    window = frame_from(rng, 200)
    report = compare_window(reference, window, DriftConfig())
    assert report.evaluated
    assert not report.any_drift
    assert report.max_ks_statistic < 0.20
    assert report.severity == "low"
    assert "no drift" in report.reason


def test_shifted_window_is_detected_with_correct_attribution() -> None:
    rng = np.random.default_rng(12)
    reference = frame_from(rng, 1500)
    window = frame_from(rng, 200, shift=25.0)  # latency only, ~3 sigma
    report = compare_window(reference, window, DriftConfig())
    assert report.evaluated
    assert report.any_drift
    assert report.drifted_features == ["latency_ms"]
    worst = max(report.features, key=lambda item: item.ks_statistic)
    assert worst.feature == "latency_ms"
    assert worst.mean_shift_pct == pytest.approx(20.8, rel=0.2)
    assert worst.p_adjusted < 0.01
    assert report.severity in {"medium", "high", "critical"}
    assert "latency_ms" in report.reason
    # Unchanged features must not be reported as drifted.
    assert not next(item for item in report.features if item.feature == "mem_util").drifted


def test_effect_size_floor_suppresses_trivial_shifts() -> None:
    rng = np.random.default_rng(13)
    reference = frame_from(rng, 4000)
    tiny = frame_from(rng, 1000, shift=3.0)  # statistically detectable, operationally small
    permissive = compare_window(reference, tiny, DriftConfig(effect_size_floor=0.0))
    guarded = compare_window(reference, tiny, DriftConfig(effect_size_floor=0.20))
    assert permissive.any_drift
    assert not guarded.any_drift
    # The shift is highly significant; only the practical-significance floor holds it back.
    smallest = min(item.p_value for item in guarded.features)
    assert smallest < 0.001
    assert guarded.max_ks_statistic < 0.20


def test_tiny_windows_are_skipped_with_a_reason() -> None:
    rng = np.random.default_rng(14)
    reference = frame_from(rng, 500)
    window = frame_from(rng, 30)
    config = DriftConfig(window_size=200, step=200, min_window_size=120)
    report = compare_window(reference, window, config)
    assert not report.evaluated
    assert not report.any_drift
    assert report.features == []
    assert "minimum_window_size is 120" in report.skip_reason
    assert "skipped" in report.reason
    assert drift_summary([report]).iloc[0]["evaluated"] == False  # noqa: E712


def test_missing_feature_is_rejected() -> None:
    rng = np.random.default_rng(15)
    reference = frame_from(rng, 400).drop(columns=["mem_util"])
    with pytest.raises(KeyError):
        compare_window(reference, frame_from(rng, 200), DriftConfig())


def test_population_stability_index_behaves_sensibly() -> None:
    rng = np.random.default_rng(16)
    base = rng.normal(0, 1, 2000)
    same = rng.normal(0, 1, 500)
    shifted = rng.normal(2, 1, 500)
    assert population_stability_index(base, same) < 0.1
    assert population_stability_index(base, shifted) > 0.25
    assert population_stability_index(np.array([]), same) == 0.0


def test_scan_drift_covers_the_stream_in_windows() -> None:
    rng = np.random.default_rng(17)
    reference = frame_from(rng, 800)
    early = frame_from(rng, 400)
    late = frame_from(rng, 400, shift=30.0)
    stream = pd.concat([early, late], ignore_index=True)
    config = DriftConfig(window_size=200, step=200, min_window_size=120)
    reports = scan_drift(reference, stream, config)

    assert [report.window_id for report in reports] == [0, 1, 2, 3]
    assert [report.start_position for report in reports] == [0, 200, 400, 600]
    kinds = [report.any_drift for report in reports]
    assert not kinds[0]
    assert kinds[2] and kinds[3]

    features = drift_feature_frame(reports)
    assert len(features) == len(reports) * 4
    assert set(features.columns) >= {
        "window_id",
        "feature",
        "ks_statistic",
        "p_adjusted",
        "drifted",
    }
    summary = drift_summary(reports)
    assert len(summary) == 4
    assert "reason" in summary.columns


def test_no_windows_when_stream_is_shorter_than_the_minimum() -> None:
    rng = np.random.default_rng(18)
    reference = frame_from(rng, 800)
    short = frame_from(rng, 40)
    reports = scan_drift(reference, short, DriftConfig(min_window_size=120, window_size=200))
    assert len(reports) == 1
    assert not reports[0].evaluated


def test_all_features_are_compared_by_default() -> None:
    rng = np.random.default_rng(19)
    report = compare_window(frame_from(rng, 800), frame_from(rng, 200), DriftConfig())
    assert [item.feature for item in report.features] == list(FEATURE_COLUMNS)
    assert all(item.test == "ks_2samp" for item in report.features)
    assert all(item.unit for item in report.features)
