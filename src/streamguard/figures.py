"""Plotly figure builders.

Kept separate from the dashboard so that the same charts can be written to
``reports/figures/`` without importing Streamlit, and so the dashboard file stays
readable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamguard.simulator import FEATURE_LABELS, FEATURE_UNITS

__all__ = [
    "confusion_heatmap",
    "drift_heatmap",
    "drift_ks_lines",
    "feature_timeseries",
    "metrics_bar",
    "score_timeseries",
    "severity_bar",
    "write_figures",
]

TRUTH_COLOR = "#d62728"
DETECTOR_COLOR = "#ff7f0e"
DRIFT_COLOR = "rgba(148, 103, 189, 0.12)"
PALETTE = ("#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf")

_LAYOUT: dict[str, Any] = {
    "template": "plotly_white",
    "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
    "legend": {"orientation": "h", "y": -0.18},
    "hovermode": "x unified",
}


def _drift_spans(frame: pd.DataFrame) -> list[str]:
    """Return contiguous segments where ``drift_active`` is true (as timestamps)."""
    if "drift_active" not in frame.columns or not bool(frame["drift_active"].any()):
        return []
    stamps = frame.loc[frame["drift_active"].astype(bool), "timestamp"]
    return [str(stamps.min()), str(stamps.max())]


def feature_timeseries(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    detector: str,
    show_truth: bool = True,
    max_points: int = 4000,
) -> go.Figure:
    """Small-multiples of the monitored features with anomaly and drift overlays."""
    plot_frame = frame if len(frame) <= max_points else frame.iloc[-max_points:]
    figure = make_subplots(
        rows=len(features),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[FEATURE_LABELS.get(name, name) for name in features],
    )
    flag_column = f"flag_{detector}"
    for row, feature in enumerate(features, start=1):
        figure.add_trace(
            go.Scattergl(
                x=plot_frame["timestamp"],
                y=plot_frame[feature],
                mode="lines",
                name=FEATURE_LABELS.get(feature, feature),
                line={"width": 1, "color": PALETTE[(row - 1) % len(PALETTE)]},
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        if flag_column in plot_frame.columns:
            hits = plot_frame.loc[plot_frame[flag_column].astype(bool)]
            if not hits.empty:
                figure.add_trace(
                    go.Scattergl(
                        x=hits["timestamp"],
                        y=hits[feature],
                        mode="markers",
                        name=f"{detector} alert",
                        marker={"color": DETECTOR_COLOR, "size": 7, "symbol": "x"},
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )
        if show_truth and "is_anomaly_true" in plot_frame.columns:
            truth = plot_frame.loc[plot_frame["is_anomaly_true"].astype(bool)]
            if not truth.empty:
                figure.add_trace(
                    go.Scattergl(
                        x=truth["timestamp"],
                        y=truth[feature],
                        mode="markers",
                        name="injected anomaly (ground truth)",
                        marker={"color": TRUTH_COLOR, "size": 5, "symbol": "circle-open"},
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )
        figure.update_yaxes(title_text=FEATURE_UNITS.get(feature, ""), row=row, col=1)

    spans = _drift_spans(plot_frame)
    if spans:
        figure.add_vrect(
            x0=spans[0],
            x1=spans[1],
            fillcolor=DRIFT_COLOR,
            line_width=0,
            annotation_text="injected distribution shift",
            annotation_position="top left",
            row=1,
            col=1,
        )
    figure.update_layout(
        **_LAYOUT,
        height=200 * len(features) + 120,
        title="Simulated metrics with detector alerts and ground-truth anomalies",
    )
    return figure


def score_timeseries(
    frame: pd.DataFrame, detector: str, threshold: float | None = None
) -> go.Figure:
    """Detector score over time with the alert threshold."""
    flag_column = f"flag_{detector}"
    score_column = f"score_{detector}"
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=frame["timestamp"],
            y=frame[score_column],
            mode="lines",
            name=f"{detector} score",
            line={"width": 1, "color": "#1f77b4"},
        )
    )
    if flag_column in frame.columns:
        hits = frame.loc[frame[flag_column].astype(bool)]
        if not hits.empty:
            figure.add_trace(
                go.Scattergl(
                    x=hits["timestamp"],
                    y=hits[score_column],
                    mode="markers",
                    name="alert",
                    marker={"color": DETECTOR_COLOR, "size": 7, "symbol": "x"},
                )
            )
    if threshold is not None:
        figure.add_hline(
            y=threshold,
            line={"color": DETECTOR_COLOR, "dash": "dash"},
            annotation_text=f"threshold {threshold:g}",
        )
    spans = _drift_spans(frame)
    if spans:
        figure.add_vrect(x0=spans[0], x1=spans[1], fillcolor=DRIFT_COLOR, line_width=0)
    figure.update_layout(**_LAYOUT, height=320, title=f"{detector} anomaly score (simulated data)")
    return figure


def drift_heatmap(drift_features: pd.DataFrame) -> go.Figure:
    """KS statistic per window per feature."""
    if drift_features.empty:
        return go.Figure().update_layout(**_LAYOUT, title="No drift windows evaluated")
    pivot = drift_features.pivot_table(
        index="feature", columns="window_id", values="ks_statistic", aggfunc="max"
    )
    drifted = drift_features.pivot_table(
        index="feature", columns="window_id", values="drifted", aggfunc="max"
    )
    text = [
        [
            f"D={pivot.iloc[i, j]:.2f}{'*' if bool(drifted.iloc[i, j]) else ''}"
            for j in range(pivot.shape[1])
        ]
        for i in range(pivot.shape[0])
    ]
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[f"w{int(column)}" for column in pivot.columns],
            y=list(pivot.index),
            colorscale="YlOrRd",
            colorbar={"title": "KS D"},
            text=text,
            texttemplate="%{text}",
        )
    )
    figure.update_layout(
        **_LAYOUT,
        height=360,
        title="Drift scan: KS statistic per window ( * = corrected p below alpha and effect size floor )",
    )
    return figure


def drift_ks_lines(drift_features: pd.DataFrame, floor: float) -> go.Figure:
    """KS statistic trajectory per feature with the practical-significance floor."""
    figure = go.Figure()
    if drift_features.empty:
        return figure.update_layout(**_LAYOUT, title="No drift windows evaluated")
    for index, (feature, group) in enumerate(drift_features.groupby("feature")):
        group = group.sort_values("window_id")
        figure.add_trace(
            go.Scatter(
                x=group["start_timestamp"],
                y=group["ks_statistic"],
                mode="lines+markers",
                name=feature,
                line={"color": PALETTE[index % len(PALETTE)]},
            )
        )
    figure.add_hline(
        y=floor,
        line={"color": "grey", "dash": "dot"},
        annotation_text=f"effect-size floor {floor:g}",
    )
    figure.update_layout(
        **_LAYOUT,
        height=340,
        title="KS statistic per feature across windows (simulated data)",
        yaxis_title="KS D",
    )
    return figure


def metrics_bar(metrics: pd.DataFrame) -> go.Figure:
    """Precision / recall / F1 per detector."""
    figure = go.Figure()
    if metrics.empty:
        return figure.update_layout(**_LAYOUT, title="No metrics")
    for index, column in enumerate(["precision", "recall", "f1"]):
        figure.add_trace(
            go.Bar(
                x=metrics["detector"],
                y=metrics[column],
                name=column,
                marker={"color": PALETTE[index % len(PALETTE)]},
                text=[f"{value:.2f}" for value in metrics[column]],
                textposition="outside",
            )
        )
    figure.update_layout(
        **_LAYOUT,
        height=360,
        barmode="group",
        yaxis={"range": [0, 1.08], "title": "score"},
        title="Detection quality on labelled simulated stream",
    )
    return figure


def confusion_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """2x2 confusion matrix heatmap."""
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="Blues",
            text=matrix.to_numpy(),
            texttemplate="%{text}",
            showscale=False,
        )
    )
    figure.update_layout(
        **_LAYOUT, height=320, title="Confusion matrix (labelled simulated stream)"
    )
    return figure


def severity_bar(counts: dict[str, int]) -> go.Figure:
    """Alert counts by severity."""
    levels = ["critical", "high", "medium", "low"]
    figure = go.Figure(
        go.Bar(
            x=levels,
            y=[counts.get(level, 0) for level in levels],
            marker={"color": ["#d62728", "#ff7f0e", "#bcbd22", "#7f7f7f"]},
            text=[counts.get(level, 0) for level in levels],
            textposition="outside",
        )
    )
    figure.update_layout(**_LAYOUT, height=300, title="Alerts by severity", yaxis_title="alerts")
    return figure


def write_figures(figures: dict[str, go.Figure], directory: str | Path) -> dict[str, Path]:
    """Write figures as standalone HTML files (no kaleido/Chromium needed)."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, figure in figures.items():
        path = target / f"{name}.html"
        figure.write_html(path, include_plotlyjs="cdn", full_html=True)
        written[name] = path
    return written
