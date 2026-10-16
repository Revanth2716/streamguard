"""End-to-end orchestration.

``run_pipeline`` is the single entry point used by both the CLI and the
dashboard, so the numbers shown in the UI and the numbers written to
``reports/`` always come from the same code path.

Order of operations (and why it matters):

1. Generate the labelled reference/stream pair from a seed.
2. Fit every detector on the **reference block only**. Labels never reach a fit.
3. Score the stream and collect alerts.
4. Scan the stream for distribution drift against the same reference block.
5. Evaluate the flags against the stream's ground-truth labels, overall and per
   segment, and persist the report together with the exact configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from streamguard.alerts import (
    Alert,
    AlertStore,
    alerts_from_detector_results,
    alerts_from_drift_reports,
)
from streamguard.config import AppConfig
from streamguard.detector import run_detectors
from streamguard.drift import DriftReport, drift_feature_frame, drift_summary, scan_drift
from streamguard.evaluation import (
    SEGMENT_COLUMN,
    build_evaluation_report,
    detection_latency,
    evaluate_all,
    latency_summary,
    save_report,
    segment_labels,
    segment_metrics,
)
from streamguard.simulator import (
    FEATURE_COLUMNS,
    SIMULATED_ORIGIN,
    Dataset,
    anomaly_events,
    generate_dataset,
)

__all__ = [
    "PRIMARY_METRICS_COLUMNS",
    "PipelineResult",
    "build_alert_store",
    "flagged_frame",
    "run_pipeline",
    "save_artifacts",
]

#: Columns shown by default in the headless report and the dashboard overview.
PRIMARY_METRICS_COLUMNS = [
    "detector",
    "fit_rows",
    "flags",
    "flag_rate",
    "threshold",
    "window",
    "precision",
    "recall",
    "f1",
]


@dataclass
class PipelineResult:
    """Container for everything one run produced."""

    dataset: Dataset
    results: dict[str, pd.DataFrame]
    drift_reports: list[DriftReport]
    alerts: list[Alert]
    metrics: pd.DataFrame
    latency: pd.DataFrame
    segments: pd.DataFrame
    report: dict[str, Any]
    config: AppConfig
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    flagged: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def reference(self) -> pd.DataFrame:
        return self.dataset.reference

    @property
    def stream(self) -> pd.DataFrame:
        return self.dataset.stream

    @property
    def drift_features(self) -> pd.DataFrame:
        return drift_feature_frame(self.drift_reports)

    @property
    def drift_windows(self) -> pd.DataFrame:
        return drift_summary(self.drift_reports)

    def detector_status(self) -> pd.DataFrame:
        """One row per detector: configuration plus measured stream behaviour."""
        truth = self.dataset.truth()
        rows: list[dict[str, Any]] = []
        for name, frame in self.results.items():
            metrics = self.metrics[self.metrics["detector"] == name]
            record: dict[str, Any] = {
                "detector": name,
                "fit_rows": int(self.reference.shape[0]),
                "flags": int(frame["is_anomaly"].sum()),
                "flag_rate": round(float(frame["is_anomaly"].mean()), 4),
                "last_flag": (
                    str(
                        self.stream.loc[frame["is_anomaly"].to_numpy(dtype=bool), "timestamp"].iloc[
                            -1
                        ]
                    )
                    if bool(frame["is_anomaly"].any())
                    else None
                ),
                "precision": None,
                "recall": None,
                "f1": None,
            }
            if not metrics.empty:
                record.update(
                    {
                        "precision": float(metrics["precision"].iloc[0]),
                        "recall": float(metrics["recall"].iloc[0]),
                        "f1": float(metrics["f1"].iloc[0]),
                    }
                )
            if name == "rolling_zscore":
                record["threshold"] = self.config.detector.zscore_threshold
                record["window"] = self.config.detector.zscore_window
            elif name == "isolation_forest":
                record["threshold"] = self.config.detector.contamination
                record["window"] = None
            elif name == "static_percentile":
                record["threshold"] = self.config.detector.static_threshold_quantile
                record["window"] = None
            else:
                record["threshold"] = None
                record["window"] = None
            record["truth_positives"] = int(truth.sum())
            rows.append(record)
        return pd.DataFrame(rows)


def flagged_frame(dataset: Dataset, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the stream, its ground-truth labels and every detector's output."""
    base_columns = ["timestamp", "event_index", "data_origin", *FEATURE_COLUMNS]
    frame = dataset.stream.loc[:, [c for c in base_columns if c in dataset.stream.columns]].copy()
    frame["is_anomaly_true"] = dataset.stream["is_anomaly"].astype(bool)
    frame["anomaly_id"] = dataset.stream["anomaly_id"]
    frame["anomaly_kind"] = dataset.stream["anomaly_kind"]
    frame["drift_active"] = dataset.stream.get("drift_active", False)
    frame[SEGMENT_COLUMN] = segment_labels(dataset.stream)
    for name, output in results.items():
        frame[f"flag_{name}"] = output["is_anomaly"].to_numpy(dtype=bool)
        frame[f"score_{name}"] = output["score"].to_numpy(dtype=float)
        frame[f"severity_{name}"] = output["severity"].to_numpy()
        frame[f"reason_{name}"] = output["reason"].to_numpy()
    flag_columns = [f"flag_{name}" for name in results]
    frame["flag_any"] = frame[flag_columns].any(axis=1) if flag_columns else False
    # Keep only JSON-serialisable attrs: pandas propagates attributes through
    # copies, and timestamp-bearing attrs make Streamlit's dataframe warning.
    frame.attrs = {"data_origin": SIMULATED_ORIGIN, "simulated": True}
    return frame


def run_pipeline(config: AppConfig, dataset: Dataset | None = None) -> PipelineResult:
    """Run the full simulate -> detect -> drift -> alert -> evaluate workflow."""
    config.validate()
    if dataset is None:
        dataset = generate_dataset(config.simulator)

    results = run_detectors(dataset.reference, dataset.stream, config.detector)
    events = anomaly_events(dataset.stream)
    drift_reports = scan_drift(dataset.reference, dataset.stream, config.drift)
    alerts = alerts_from_detector_results(results, dataset.stream["timestamp"])
    alerts.extend(alerts_from_drift_reports(drift_reports))

    truth = dataset.truth()
    metrics = evaluate_all(results, truth)
    segments = segment_metrics(results, truth, segment_labels(dataset.stream))
    report = build_evaluation_report(
        results,
        truth,
        events,
        metadata=dataset.metadata,
        segments=segment_labels(dataset.stream),
    )
    report["drift"] = {
        "windows_evaluated": int(sum(1 for item in drift_reports if item.evaluated)),
        "windows_with_drift": int(sum(1 for item in drift_reports if item.any_drift)),
        "alpha": config.drift.alpha,
        "correction": config.drift.correction,
        "window_size": config.drift.window_size,
        "min_window_size": config.drift.min_window_size,
        "effect_size_floor": config.drift.effect_size_floor,
        "summary": drift_summary(drift_reports).to_dict(orient="records"),
    }
    report["latency_summary"] = latency_summary(detection_latency(events, results)).to_dict(
        orient="records"
    )

    latency = detection_latency(events, results)
    return PipelineResult(
        dataset=dataset,
        results=results,
        drift_reports=drift_reports,
        alerts=alerts,
        metrics=metrics,
        latency=latency,
        segments=segments,
        report=report,
        config=config,
        events=events,
        flagged=flagged_frame(dataset, results),
    )


def build_alert_store(result: PipelineResult, path: str | Path | None = None) -> AlertStore:
    """Create a store, add this run's alerts and persist the result."""
    store = AlertStore(path if path is not None else result.config.alert_state_path)
    store.add_many(result.alerts)
    store.save()
    return store


def save_artifacts(result: PipelineResult, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Persist metrics, labelled data and metadata for the current run."""
    root = Path(output_dir if output_dir is not None else result.config.output_dir)
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    data_dir = Path("data/processed")
    for directory in (metrics_dir, figures_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    written["evaluation"] = save_report(result.report, metrics_dir / "evaluation.json")
    written["metrics_by_detector"] = _write_csv(
        result.metrics, metrics_dir / "metrics_by_detector.csv"
    )
    written["latency"] = _write_csv(result.latency, metrics_dir / "detection_latency.csv")
    written["segments"] = _write_csv(result.segments, metrics_dir / "metrics_by_segment.csv")
    written["drift_windows"] = _write_csv(result.drift_windows, metrics_dir / "drift_windows.csv")
    written["labelled_stream"] = _write_csv(result.flagged, data_dir / "stream_labelled.csv")
    written["run_metadata"] = _write_json(
        {
            "data_origin": "simulated",
            "config": result.config.to_dict(),
            "dataset_metadata": result.dataset.metadata,
        },
        metrics_dir / "run_metadata.json",
    )
    return written


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
