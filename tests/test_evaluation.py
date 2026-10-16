"""Evaluation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamguard.evaluation import (
    binary_counts,
    confusion_matrix_frame,
    detection_latency,
    evaluate_all,
    evaluate_detector,
    latency_summary,
    metrics_frame,
    oracle_threshold_sweep,
    save_report,
    segment_labels,
    segment_metrics,
)


def test_binary_counts() -> None:
    counts = binary_counts([True, True, False, False], [True, False, True, False])
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1, "n": 4}
    with pytest.raises(ValueError):
        binary_counts([True, False], [True])


def test_perfect_and_imperfect_predictions() -> None:
    truth = [True, True, False, False, True]
    perfect = evaluate_detector("perfect", truth, truth, scores=[0.9, 0.8, 0.1, 0.2, 0.7])
    assert perfect.precision == 1.0
    assert perfect.recall == 1.0
    assert perfect.f1 == 1.0
    assert perfect.roc_auc == 1.0
    assert perfect.average_precision == 1.0

    partial = evaluate_detector(
        "partial", truth, [True, False, True, False, False], scores=[0.9, 0.4, 0.6, 0.1, 0.3]
    )
    assert partial.tp == 1
    assert partial.fp == 1
    assert partial.fn == 2
    assert partial.precision == pytest.approx(0.5)
    assert partial.recall == pytest.approx(1 / 3)
    assert partial.f1 == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))


def test_zero_division_returns_zero_not_nan() -> None:
    metrics = evaluate_detector(
        "never", [True, False, True], [False, False, False], scores=[0.1, 0.2, 0.3]
    )
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.specificity == 1.0
    assert all(
        value is None or not (isinstance(value, float) and np.isnan(value))
        for value in metrics.to_dict().values()
    )


def test_roc_auc_is_none_with_a_single_class() -> None:
    metrics = evaluate_detector("m", [True, True], [True, False], scores=[0.5, 0.2])
    assert metrics.roc_auc is None
    assert metrics.average_precision is None


def test_metric_frames() -> None:
    metrics = evaluate_detector("m", [True, False], [True, False])
    matrix = confusion_matrix_frame(metrics)
    assert matrix.shape == (2, 2)
    assert matrix.loc["actual anomaly", "predicted anomaly"] == 1
    tidy = metrics_frame(metrics)
    assert set(tidy["metric"]) >= {"detector", "precision", "recall", "confusion_tp"}


def test_segment_metrics_split_the_stream(small_dataset, small_result) -> None:
    segments = segment_labels(small_dataset.stream)
    assert set(segments.unique()) == {"pre_drift", "drift"}
    table = segment_metrics(small_result.results, small_dataset.truth(), segments)
    assert set(table["segment"]) == {"pre_drift", "drift"}
    assert set(table["detector"]) == set(small_result.results)
    assert (table["n"] > 0).all()
    with pytest.raises(ValueError):
        segment_metrics(small_result.results, small_dataset.truth().head(10), segments)


def test_detection_latency_matches_event_spans() -> None:
    events = pd.DataFrame(
        [
            {
                "anomaly_id": 1,
                "kind": "spike",
                "features": "latency_ms",
                "duration": 3,
                "start": 10,
                "stop": 12,
            },
            {
                "anomaly_id": 2,
                "kind": "burst",
                "features": "cpu_util",
                "duration": 2,
                "start": 50,
                "stop": 51,
            },
        ]
    )
    flags = np.zeros(100, dtype=bool)
    flags[12] = True  # caught late inside event 1
    results = {"m": pd.DataFrame({"is_anomaly": flags, "event_index": np.arange(100)})}

    latency = detection_latency(events, results)
    assert len(latency) == 2
    first = latency[latency["anomaly_id"] == 1].iloc[0]
    missing = latency[latency["anomaly_id"] == 2].iloc[0]
    assert bool(first["detected"]) and first["latency"] == 2 and first["hits"] == 1
    assert not bool(missing["detected"]) and pd.isna(missing["latency"])

    summary = latency_summary(latency)
    assert summary.iloc[0]["detection_rate"] == pytest.approx(0.5)
    assert "median_latency" in summary.columns


def test_detection_latency_is_empty_safe() -> None:
    empty_events = pd.DataFrame(
        columns=["anomaly_id", "kind", "features", "duration", "start", "stop"]
    )
    result = detection_latency(empty_events, {"m": pd.DataFrame({"is_anomaly": [True]})})
    assert result.empty
    assert "detector" in result.columns


def test_oracle_threshold_sweep_trades_precision_for_recall(small_result, small_dataset) -> None:
    sweep = oracle_threshold_sweep(small_result.results, small_dataset.truth())
    assert not sweep.empty
    assert {"detector", "threshold", "flag_rate", "precision", "recall", "f1"} <= set(sweep.columns)
    assert "dummy" not in set(sweep["detector"])  # a model that never flags has nothing to sweep
    for _, group in sweep.groupby("detector"):
        ordered = group.sort_values("threshold")
        assert ordered["flag_rate"].is_monotonic_decreasing
        assert ordered["recall"].is_monotonic_decreasing


def test_build_evaluation_report_is_self_describing(small_result, small_dataset, tmp_path) -> None:
    report = small_result.report
    assert report["data_origin"] == "simulated"
    assert report["primary_detector"] in small_result.results
    assert report["metadata"]["seed"] == small_dataset.metadata["seed"]
    assert len(report["metrics_by_detector"]) == len(small_result.results)
    assert report["oracle_best_operating_point"]
    assert report["drift"]["windows_evaluated"] >= 1
    assert any("not real-world" in note for note in report["notes"])

    path = save_report(report, tmp_path / "nested" / "evaluation.json")
    assert path.is_file()
    assert "data_origin" in path.read_text(encoding="utf-8")


def test_evaluate_all_sorts_by_f1(small_result, small_dataset) -> None:
    table = evaluate_all(small_result.results, small_dataset.truth())
    assert len(table) == len(small_result.results)
    assert table["f1"].is_monotonic_decreasing
