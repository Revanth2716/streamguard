"""End-to-end pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from streamguard.alerts import AlertStore
from streamguard.cli import main
from streamguard.pipeline import (
    PRIMARY_METRICS_COLUMNS,
    build_alert_store,
    flagged_frame,
    run_pipeline,
    save_artifacts,
)

MINIMAL_CONFIG = """
[simulator]
n_reference = 250
n_stream = 250
n_point_anomalies = 6

[detector]
if_n_estimators = 40

[drift]
window_size = 100
step = 100
min_window_size = 60
"""


def test_pipeline_is_structurally_consistent(small_result, small_dataset) -> None:
    assert len(small_result.stream) == len(small_dataset.stream)
    for name, frame in small_result.results.items():
        assert len(frame) == len(small_result.stream), name
    assert len(small_result.flagged) == len(small_result.stream)
    assert small_result.flagged["is_anomaly_true"].sum() == int(small_dataset.truth().sum())
    assert set(small_result.metrics["detector"]) == set(small_result.results)
    assert not small_result.events.empty
    assert small_result.drift_reports
    assert small_result.alerts
    assert set(PRIMARY_METRICS_COLUMNS) <= set(small_result.detector_status().columns)


def test_flagged_frame_carries_truth_and_every_detector(small_result) -> None:
    frame = small_result.flagged
    for name in small_result.results:
        assert f"flag_{name}" in frame.columns
        assert f"score_{name}" in frame.columns
        assert f"reason_{name}" in frame.columns
    assert frame["flag_any"].dtype == bool
    assert frame["segment"].isin(["pre_drift", "drift"]).all()


def test_flagged_frame_is_recomputable_from_its_inputs(small_dataset, small_result) -> None:
    rebuilt = flagged_frame(small_dataset, small_result.results)
    pd.testing.assert_frame_equal(rebuilt, small_result.flagged)


def test_two_runs_with_the_same_seed_agree(small_config, small_dataset) -> None:
    first = run_pipeline(small_config, dataset=small_dataset)
    second = run_pipeline(small_config, dataset=small_dataset)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    pd.testing.assert_frame_equal(first.segments, second.segments)
    assert [alert.alert_id for alert in first.alerts] == [alert.alert_id for alert in second.alerts]
    assert json.dumps(first.report, default=str, sort_keys=True) == json.dumps(
        second.report, default=str, sort_keys=True
    )


def test_report_records_seed_and_configuration(small_result, small_config) -> None:
    metadata = small_result.report["metadata"]
    assert metadata["seed"] == small_config.simulator.seed
    assert metadata["config"]["seed"] == small_config.simulator.seed
    assert metadata["config"]["shift"]["latency_mult"] == small_config.simulator.shift.latency_mult
    assert small_result.report["drift"]["correction"] == small_config.drift.correction
    assert small_result.report["latency_summary"]


def test_drift_and_anomaly_alerts_coexist(small_result) -> None:
    types = {alert.alert_type for alert in small_result.alerts}
    assert "anomaly_point" in types
    assert "drift" in types


def test_save_artifacts_writes_every_expected_file(small_result, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    written = save_artifacts(small_result, tmp_path / "reports")
    for key in (
        "evaluation",
        "metrics_by_detector",
        "latency",
        "segments",
        "drift_windows",
        "labelled_stream",
        "run_metadata",
    ):
        assert key in written
        assert Path(written[key]).is_file(), key
    payload = json.loads(Path(written["evaluation"]).read_text(encoding="utf-8"))
    assert payload["data_origin"] == "simulated"
    assert payload["metadata"]["seed"] == small_result.dataset.metadata["seed"]
    stream = pd.read_csv(written["labelled_stream"])
    assert len(stream) == len(small_result.stream)
    assert (stream["data_origin"] == "simulated").all()


def test_build_alert_store_persists_alerts(small_result, alert_path: Path) -> None:
    store = build_alert_store(small_result, alert_path)
    assert alert_path.is_file()
    assert len(store) == len(small_result.alerts)
    reloaded = AlertStore(alert_path)
    assert len(reloaded) == len(small_result.alerts)


def test_cli_runs_headless_and_prints_measured_numbers(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "tiny.toml"
    config_path.write_text(MINIMAL_CONFIG, encoding="utf-8")

    assert main(["--config", str(config_path), "--no-save"]) == 0
    output = capsys.readouterr().out
    assert "simulated telemetry only" in output
    assert "evaluation on the labelled simulated stream" in output
    assert "drift scan" in output


def test_cli_json_mode_emits_a_parseable_report(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "tiny.toml"
    config_path.write_text(MINIMAL_CONFIG, encoding="utf-8")

    assert main(["--config", str(config_path), "--no-save", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data_origin"] == "simulated"
    assert payload["metrics_by_detector"]


def test_cli_rejects_an_absurd_stream_length(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "tiny.toml"
    config_path.write_text(MINIMAL_CONFIG, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--config", str(config_path), "--no-save", "--n-stream", "10"])
