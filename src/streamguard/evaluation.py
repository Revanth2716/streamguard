"""Detection-quality evaluation on a separate labelled dataset.

Everything measured here is measured on the *simulated* stream block whose labels
were injected by :mod:`streamguard.simulator`. The reference block used to fit
detectors contains no anomalies, so detector fitting never sees a label. The
single-label convention is:

* ``is_anomaly == True`` marks an injected point anomaly;
* a period of injected *distribution shift* is **not** labelled as an anomaly.

Consequences are reported honestly rather than hidden:

* rows inside the shifted period that a fixed-reference detector flags count as
  false positives under this convention (the metrics are therefore pessimistic
  for the detectors that cannot adapt), and
* the segmented table (``segment_metrics``) reports the reference-like segment and
  the shifted segment separately so the effect is visible instead of averaged away.

No metric in this module is hardcoded; every number comes from comparing a
detector's flags with the labels of the run that produced them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "Metrics",
    "binary_counts",
    "build_evaluation_report",
    "confusion_matrix_frame",
    "detection_latency",
    "evaluate_all",
    "evaluate_detector",
    "format_metric_value",
    "latency_summary",
    "metrics_frame",
    "oracle_threshold_sweep",
    "save_report",
    "segment_labels",
    "segment_metrics",
]

PRIMARY_DETECTOR = "isolation_forest"
SEGMENT_COLUMN = "segment"


@dataclass
class Metrics:
    """Binary classification metrics for one detector on one labelled set."""

    detector: str
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    false_positive_rate: float
    balanced_accuracy: float
    average_precision: float | None = None
    roc_auc: float | None = None
    positives: int = 0
    predicted_positives: int = 0
    segment: str = "all"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = None if np.isnan(value) else round(value, 6)
        return data


def binary_counts(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, int]:
    """Confusion-matrix counts; no sklearn label-order surprises."""
    truth = np.asarray(y_true).astype(bool)
    predicted = np.asarray(y_pred).astype(bool)
    if truth.shape != predicted.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape} vs y_pred {predicted.shape}")
    return {
        "tp": int(np.sum(truth & predicted)),
        "fp": int(np.sum(~truth & predicted)),
        "tn": int(np.sum(~truth & ~predicted)),
        "fn": int(np.sum(truth & ~predicted)),
        "n": int(truth.size),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def evaluate_detector(
    detector: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray | None = None,
    segment: str = "all",
) -> Metrics:
    """Compute precision/recall/F1/confusion counts for one detector."""
    counts = binary_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    accuracy = _safe_ratio(tp + tn, counts["n"])
    average_precision = None
    roc_auc = None
    if scores is not None:
        truth = np.asarray(y_true).astype(bool)
        score_values = np.asarray(scores, dtype=float)
        if np.isfinite(score_values).all() and 0 < truth.sum() < truth.size:
            average_precision = float(average_precision_score(truth, score_values))
            roc_auc = float(roc_auc_score(truth, score_values))
    return Metrics(
        detector=detector,
        n=counts["n"],
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        specificity=specificity,
        false_positive_rate=1.0 - specificity,
        balanced_accuracy=(recall + specificity) / 2.0,
        average_precision=average_precision,
        roc_auc=roc_auc,
        positives=int(tp + fn),
        predicted_positives=int(tp + fp),
        segment=segment,
    )


def evaluate_all(
    results: dict[str, pd.DataFrame],
    y_true: pd.Series | np.ndarray,
    segment: str = "all",
) -> pd.DataFrame:
    """Metric table with one row per detector, ordered by F1 (descending)."""
    rows = [
        evaluate_detector(
            name,
            y_true,
            frame["is_anomaly"].to_numpy(dtype=bool),
            frame["score"].to_numpy(dtype=float),
            segment=segment,
        ).to_dict()
        for name, frame in results.items()
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("f1", ascending=False).reset_index(drop=True)
    return frame


def format_metric_value(value: Any) -> str:
    """Render a metric for display.

    Values are stringified on purpose: a column that mixes text, ints and floats
    makes Arrow serialisation fail inside Streamlit, so a tidy table must have a
    single consistent type.
    """
    if value is None:
        return "not computed"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metrics_frame(metrics: Metrics) -> pd.DataFrame:
    """Tidy one-row-per-metric table for display (all values rendered as text)."""
    data = metrics.to_dict()
    confusion = {key: data.pop(key) for key in ("tp", "fp", "tn", "fn")}
    rows = [{"metric": key, "value": format_metric_value(value)} for key, value in data.items()]
    rows.extend(
        {"metric": f"confusion_{key}", "value": format_metric_value(value)}
        for key, value in confusion.items()
    )
    return pd.DataFrame(rows)


def confusion_matrix_frame(metrics: Metrics) -> pd.DataFrame:
    """2x2 confusion matrix as a labelled frame."""
    return pd.DataFrame(
        [[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]],
        index=["actual normal", "actual anomaly"],
        columns=["predicted normal", "predicted anomaly"],
    )


def detection_latency(
    events: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    tolerance: int = 0,
) -> pd.DataFrame:
    """Per-event detection outcome for every detector.

    An event counts as detected when the detector flags at least one row between
    the event's start index and ``start + duration - 1 + tolerance`` (matched on
    ``event_index`` so the definition survives windowing and re-indexing).
    ``latency`` is the number of rows between the first detection and the event
    start, so 0 means "caught at the first anomalous observation"; it is empty
    (``None`` / ``NaN``) when the event was not detected at all.
    """
    columns = [
        "anomaly_id",
        "kind",
        "features",
        "duration",
        "detector",
        "detected",
        "latency",
        "hits",
    ]
    if events.empty or not results:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        start = int(event["start"])
        stop = int(event["stop"]) + int(tolerance)
        for name, frame in results.items():
            if frame.empty or "event_index" not in frame.columns:
                continue
            flagged = frame.loc[frame["is_anomaly"].to_numpy(dtype=bool), "event_index"]
            inside = flagged[(flagged >= start) & (flagged <= stop)]
            rows.append(
                {
                    "anomaly_id": int(event["anomaly_id"]),
                    "kind": str(event["kind"]),
                    "features": str(event["features"]),
                    "duration": int(event["duration"]),
                    "detector": name,
                    "detected": bool(len(inside)),
                    "latency": int(inside.min() - start) if len(inside) else None,
                    "hits": len(inside),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def latency_summary(latency: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-event outcome into a detection-rate table per detector."""
    if latency.empty:
        return pd.DataFrame(
            columns=["detector", "events", "detected", "detection_rate", "median_latency"]
        )
    grouped = latency.groupby("detector", sort=False)
    summary = grouped.agg(
        events=("anomaly_id", "count"),
        detected=("detected", "sum"),
        median_latency=("latency", "median"),
    ).reset_index()
    summary["detection_rate"] = summary["detected"] / summary["events"]
    return summary.sort_values("detection_rate", ascending=False).reset_index(drop=True)


def oracle_threshold_sweep(
    results: dict[str, pd.DataFrame],
    y_true: pd.Series | np.ndarray,
    n_steps: int = 25,
    quantile_low: float = 0.80,
    quantile_high: float = 0.995,
) -> pd.DataFrame:
    """Precision/recall/F1 as each detector's score threshold is raised.

    DIAGNOSTIC ONLY. The thresholds are swept against the labelled stream, so the
    best row in this table is an *upper bound* on what each score function can
    achieve, not a deployable operating point: choosing a threshold with labels
    is label leakage. It is included so that a reader can tell a weak score apart
    from a mis-set threshold -- the isolation forest's default operating point is
    a contamination prior, and this table shows what its ranking could deliver.
    """
    rows: list[dict[str, Any]] = []
    truth = np.asarray(y_true).astype(bool)
    for name, frame in results.items():
        if name == "dummy" or frame.empty:
            continue
        scores = frame["score"].to_numpy(dtype=float)
        candidates = np.unique(
            np.quantile(scores, np.linspace(quantile_low, quantile_high, n_steps))
        )
        for threshold in candidates:
            predicted = scores >= threshold
            metrics = evaluate_detector(name, truth, predicted)
            rows.append(
                {
                    "detector": name,
                    "threshold": float(threshold),
                    "flag_rate": float(predicted.mean()),
                    "tp": metrics.tp,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )
    return pd.DataFrame(rows)


def segment_labels(stream: pd.DataFrame) -> pd.Series:
    """Segment each stream row as ``pre_drift`` or ``drift`` (or ``reference``)."""
    labels = pd.Series("pre_drift", index=stream.index, dtype=object)
    if "drift_active" in stream.columns:
        labels[stream["drift_active"].astype(bool)] = "drift"
    if "is_reference" in stream.columns:
        labels[stream["is_reference"].astype(bool)] = "reference"
    return labels


def segment_metrics(
    results: dict[str, pd.DataFrame],
    truth: pd.Series,
    segments: pd.Series,
) -> pd.DataFrame:
    """Metric table computed per segment (e.g. pre-drift versus shifted period)."""
    if results:
        length = len(next(iter(results.values())))
        if len(truth) != length or len(segments) != length:
            raise ValueError("truth, segments and detector results must be the same length")
    rows: list[dict[str, Any]] = []
    for segment in pd.unique(segments):
        mask = (segments == segment).to_numpy()
        if not mask.any():
            continue
        for name, frame in results.items():
            detected = frame["is_anomaly"].to_numpy(dtype=bool)[mask]
            metrics = evaluate_detector(
                name, truth.to_numpy()[mask], detected, segment=str(segment)
            )
            rows.append(metrics.to_dict())
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["segment", "f1"], ascending=[True, False]).reset_index(drop=True)
    return frame


def build_evaluation_report(
    results: dict[str, pd.DataFrame],
    truth: pd.Series | np.ndarray,
    events: pd.DataFrame,
    metadata: dict[str, Any],
    segments: pd.Series | None = None,
    primary: str = PRIMARY_DETECTOR,
    tolerance: int = 0,
) -> dict[str, Any]:
    """Assemble the full, reproducible evaluation report for one run."""
    if primary not in results:
        primary = next(iter(results))
    metrics = evaluate_all(results, truth)
    latency = detection_latency(events, results, tolerance=tolerance)
    primary_metrics = evaluate_detector(
        primary,
        truth,
        results[primary]["is_anomaly"].to_numpy(dtype=bool),
        results[primary]["score"].to_numpy(dtype=float),
    )
    sweep = oracle_threshold_sweep(results, truth)
    report: dict[str, Any] = {
        "data_origin": "simulated",
        "primary_detector": primary,
        "primary_metrics": primary_metrics.to_dict(),
        "confusion_matrix": confusion_matrix_frame(primary_metrics).to_dict(),
        "metrics_by_detector": metrics.to_dict(orient="records"),
        "detection_latency": latency.to_dict(orient="records"),
        "latency_summary": latency_summary(latency).to_dict(orient="records"),
        "segment_metrics": []
        if segments is None
        else segment_metrics(results, truth, segments).to_dict(orient="records"),
        "oracle_threshold_sweep": sweep.to_dict(orient="records"),
        "oracle_best_operating_point": (
            sweep.sort_values("f1", ascending=False)
            .groupby("detector", as_index=False)
            .first()
            .to_dict(orient="records")
            if not sweep.empty
            else []
        ),
        "metadata": metadata,
        "notes": [
            "All observations are simulated; these numbers are not real-world performance claims.",
            "Detector fitting used the anomaly-free reference block only; stream labels were used for scoring only.",
            "Injected distribution shift is not labelled as a point anomaly, so flags inside the shifted "
            "period count as false positives for detectors that cannot adapt. See segment_metrics.",
            "oracle_threshold_sweep picks each threshold using the labels and is a diagnostic upper bound, "
            "not a procedure that may be deployed.",
        ],
    }
    return report


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the evaluation report to JSON (creating parent directories)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str, sort_keys=False), encoding="utf-8")
    return target


@dataclass
class EvaluationBundle:
    """Convenience container returned by the pipeline."""

    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    latency: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    report: dict[str, Any] = field(default_factory=dict)
