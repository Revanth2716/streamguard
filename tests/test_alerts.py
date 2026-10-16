"""Alert management tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streamguard.alerts import (
    ALERT_TYPE_ANOMALY,
    ALERT_TYPE_DRIFT,
    Alert,
    AlertStore,
    alert_episodes,
    alerts_from_detector_results,
    alerts_from_drift_reports,
    make_alert_id,
    severity_rank,
)
from streamguard.drift import DriftConfig, compare_window


def make_alert(alert_id: str = "SG-ANOM-abc", **overrides: object) -> Alert:
    payload = {
        "alert_id": alert_id,
        "timestamp": "2026-01-01T00:00:00",
        "alert_type": ALERT_TYPE_ANOMALY,
        "severity": "high",
        "feature": "latency_ms",
        "metric": "latency_ms",
        "detector": "rolling_zscore",
        "score": 5.0,
        "reason": "test alert",
    }
    payload.update(overrides)
    return Alert(**payload)  # type: ignore[arg-type]


def test_alert_ids_are_content_addressed() -> None:
    first = make_alert_id(
        ALERT_TYPE_ANOMALY, "rolling_zscore", "2026-01-01T00:00:00", "latency_ms", "r"
    )
    again = make_alert_id(
        ALERT_TYPE_ANOMALY, "rolling_zscore", "2026-01-01T00:00:00", "latency_ms", "r"
    )
    different = make_alert_id(
        ALERT_TYPE_ANOMALY, "rolling_zscore", "2026-01-01T00:01:00", "latency_ms", "r"
    )
    other_type = make_alert_id(
        ALERT_TYPE_DRIFT, "rolling_zscore", "2026-01-01T00:00:00", "latency_ms", "r"
    )

    assert first == again
    assert first != different
    assert first.startswith("SG-ANOM-")
    assert other_type.startswith("SG-DRIFT-")


def test_alert_round_trip_and_acknowledgement_state() -> None:
    alert = make_alert()
    alert.acknowledge(by="tester", at="2026-01-01T01:00:00")
    restored = Alert.from_dict(alert.to_dict())
    assert restored == alert
    assert restored.acknowledged
    assert restored.acknowledged_by == "tester"
    assert restored.acknowledged_at == "2026-01-01T01:00:00"
    restored.unacknowledge()
    assert restored.acknowledged_at is None
    assert severity_rank("critical") > severity_rank("low")


def test_store_deduplicates_and_counts() -> None:
    store = AlertStore()
    assert store.add(make_alert()) is True
    assert store.add(make_alert()) is False
    store.add(make_alert("SG-ANOM-def", severity="critical"))
    assert len(store) == 2
    assert store.counts()["total"] == 2
    assert store.counts()["unacknowledged_critical"] == 1
    assert store.acknowledge("SG-ANOM-abc") is not None
    assert store.counts()["acknowledged"] == 1
    with pytest.raises(KeyError):
        store.acknowledge("missing")


def test_acknowledgements_persist_across_a_new_store(alert_path: Path) -> None:
    store = AlertStore(alert_path)
    store.add(make_alert())
    store.acknowledge("SG-ANOM-abc", by="operator-1")
    store.save()
    assert alert_path.is_file()

    reloaded = AlertStore(alert_path)
    assert len(reloaded) == 1
    assert reloaded.get("SG-ANOM-abc").acknowledged  # type: ignore[union-attr]
    assert reloaded.get("SG-ANOM-abc").acknowledged_by == "operator-1"  # type: ignore[union-attr]


def test_journal_only_mode_reapplies_acks_to_regenerated_alerts(alert_path: Path) -> None:
    """The dashboard mode: alert records are regenerated, acks come from disk."""
    store = AlertStore(alert_path, persist_alerts=False)
    store.add(make_alert())
    store.acknowledge("SG-ANOM-abc", by="operator-2")
    store.save()

    regenerated = AlertStore(alert_path, persist_alerts=False)
    assert len(regenerated) == 0  # alert records are not restored
    assert regenerated.add(make_alert()) is True  # the same deterministic id
    assert regenerated.get("SG-ANOM-abc").acknowledged  # type: ignore[union-attr]
    assert regenerated.get("SG-ANOM-abc").acknowledged_by == "operator-2"  # type: ignore[union-attr]


def test_reset_acknowledgements_clears_the_journal(alert_path: Path) -> None:
    store = AlertStore(alert_path)
    store.add(make_alert())
    store.acknowledge("SG-ANOM-abc")
    store.save()
    store.clear_journal()
    store.save()

    payload = json.loads(alert_path.read_text(encoding="utf-8"))
    assert payload["acknowledgements"] == {}

    fresh = AlertStore(alert_path)
    assert len(fresh) == 1  # alert records are still persisted in this mode
    assert not fresh.get("SG-ANOM-abc").acknowledged  # type: ignore[union-attr]
    assert fresh.counts()["acknowledged"] == 0


def test_filters_and_dataframe_order_puts_severe_alerts_first() -> None:
    store = AlertStore()
    store.add(make_alert("SG-ANOM-a", severity="critical"))
    store.add(make_alert("SG-ANOM-b", severity="low"))
    store.add(make_alert("SG-DRIFT-c", alert_type=ALERT_TYPE_DRIFT, detector="ks_drift"))
    assert len(store.all(severity="critical")) == 1
    assert len(store.all(alert_type=ALERT_TYPE_DRIFT)) == 1
    assert len(store.all(acknowledged=False)) == 3
    frame = store.to_dataframe()
    # Same timestamp, so ordering is by descending severity: critical, high, low.
    assert list(frame["alert_id"]) == ["SG-ANOM-a", "SG-DRIFT-c", "SG-ANOM-b"]
    assert "reason" in frame.columns


def test_alerts_from_detector_results_match_the_flags(small_dataset, small_result) -> None:
    flagged = {
        name: int(frame["is_anomaly"].sum())
        for name, frame in small_result.results.items()
        if name != "dummy"
    }
    alerts = alerts_from_detector_results(
        small_result.results, small_dataset.stream["timestamp"], detectors=list(flagged)
    )
    assert len(alerts) == sum(flagged.values())
    assert len({alert.alert_id for alert in alerts}) == len(alerts)
    assert {alert.detector for alert in alerts} == set(flagged)
    assert all(alert.alert_type == ALERT_TYPE_ANOMALY for alert in alerts)
    assert all(alert.severity in {"low", "medium", "high", "critical"} for alert in alerts)
    assert all(alert.reason for alert in alerts)
    assert all(alert.data_origin == "simulated" for alert in alerts)


def test_alerts_from_drift_reports_only_for_drifted_features() -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(23)
    reference = pd.DataFrame(
        {"latency_ms": rng.normal(0, 1, 800), "cpu_util": rng.normal(0, 1, 800)}
    )
    window = pd.DataFrame({"latency_ms": rng.normal(3, 1, 200), "cpu_util": rng.normal(0, 1, 200)})
    config = DriftConfig(features=("latency_ms", "cpu_util"), window_size=200, min_window_size=120)
    report = compare_window(reference, window, config)
    alerts = alerts_from_drift_reports([report])
    assert len(alerts) == len(report.drifted_features) == 1
    assert alerts[0].feature == "latency_ms"
    assert alerts[0].alert_type == ALERT_TYPE_DRIFT
    assert "NOT evidence of model degradation" in alerts[0].reason

    clean = compare_window(
        reference,
        pd.DataFrame({"latency_ms": rng.normal(0, 1, 200), "cpu_util": rng.normal(0, 1, 200)}),
        config,
    )
    assert alerts_from_drift_reports([clean]) == []


def test_alert_episodes_collapse_consecutive_observations() -> None:
    alerts = [
        make_alert(f"SG-ANOM-{position}", evidence={"row_position": position})
        for position in (10, 11, 12, 40, 41)
    ]
    episodes = alert_episodes(alerts, gap=3)
    assert len(episodes) == 2
    assert episodes["alerts"].tolist() == [3, 2]
    assert episodes["first_position"].tolist() == [10, 40]
    assert alert_episodes([]).empty


def test_alert_episodes_ignore_drift_alerts() -> None:
    drift = make_alert("SG-DRIFT-x", alert_type=ALERT_TYPE_DRIFT, evidence={"row_position": 10})
    assert alert_episodes([drift]).empty
