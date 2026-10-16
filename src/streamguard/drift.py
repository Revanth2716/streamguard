"""Distribution drift detection.

Method
------
Each incoming window of ``window_size`` observations is compared with the
reference block using the two-sample Kolmogorov-Smirnov test (``ks_2samp``) on
every monitored feature. KS is distribution-free, sensitive to any difference in
location, spread or shape, and needs no binning choices -- which is why it is the
primary test here.

Two guards keep the detector from crying wolf:

1. **Multiple comparisons.** One test per feature per window means the family of
   comparisons grows quickly. p-values are corrected with Holm's step-down
   method (``bonferroni`` and ``none`` are also available) before the
   ``alpha`` threshold is applied.
2. **Practical significance.** ``effect_size_floor`` requires a minimum KS
   *statistic* in addition to statistical significance (default 0.20). With
   hundreds of observations, KS detects differences far too small to matter
   operationally; the floor deliberately trades sensitivity for usefulness and
   is documented in ``LIMITATIONS.md``. Measured on the default seed, clean
   windows reach a maximum KS statistic of 0.124 and the injected shift reaches
   0.418-0.534, so the floor is a margin here; it becomes the binding guard when
   the telemetry is more strongly autocorrelated (0.152 at phi=0.50, 0.195 at
   phi=0.72). Numbers: ``reports/metrics/sensitivity.json``.

The Population Stability Index (PSI) is reported alongside each test as
supporting evidence (binned, equal-frequency bins from the reference). The drift
*decision* does not use PSI, it is shown so a human can sanity-check the call.

Windows shorter than ``min_window_size`` are never tested; they are returned as
``evaluated=False`` with an explicit reason, because KS p-values on tiny samples
are meaningless.

Known statistical caveat: ``ks_2samp`` assumes independent samples, while
monitored telemetry is autocorrelated. Autocorrelation shrinks the effective
sample size and makes the test over-reject; measured on clean windows with the
effect-size floor disabled, the number of false drift alarms went from 0 of 4 at
AR(1) phi=0.00-0.35 to 1 of 4 at phi=0.50 and phi=0.72. The default simulator
coefficient is a deliberate 0.35 and the mitigation is discussed in
``LIMITATIONS.md``.

Critical framing: a detected shift is a statement about the *input data*. It is
**not** evidence that model accuracy has degraded. Label-free drift detection
cannot measure accuracy; that requires labelled outcomes, which a live stream
does not have. See :mod:`streamguard.evaluation` for the labelled evaluation and
``LIMITATIONS.md`` for the full discussion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from streamguard.config import DriftConfig
from streamguard.simulator import FEATURE_UNITS

__all__ = [
    "DriftReport",
    "FeatureDrift",
    "adjust_pvalues",
    "compare_window",
    "drift_feature_frame",
    "drift_summary",
    "population_stability_index",
    "scan_drift",
]

_PSI_EPSILON = 1e-6


@dataclass
class FeatureDrift:
    """Result of comparing one feature's window against the reference block."""

    feature: str
    n_reference: int
    n_window: int
    ks_statistic: float
    p_value: float
    p_adjusted: float
    drifted: bool
    reference_mean: float
    window_mean: float
    mean_shift_pct: float
    reference_std: float
    window_std: float
    psi: float
    unit: str = ""
    test: str = "ks_2samp"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "n_reference": self.n_reference,
            "n_window": self.n_window,
            "ks_statistic": round(self.ks_statistic, 6),
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "drifted": bool(self.drifted),
            "reference_mean": round(self.reference_mean, 6),
            "window_mean": round(self.window_mean, 6),
            "mean_shift_pct": round(self.mean_shift_pct, 4),
            "reference_std": round(self.reference_std, 6),
            "window_std": round(self.window_std, 6),
            "psi": round(self.psi, 6),
            "unit": self.unit,
            "test": self.test,
        }


@dataclass
class DriftReport:
    """Drift verdict for a single window."""

    window_id: int
    start_position: int
    end_position: int
    n_reference: int
    n_window: int
    alpha: float
    correction: str
    features: list[FeatureDrift] = field(default_factory=list)
    evaluated: bool = True
    skip_reason: str = ""
    start_timestamp: str | None = None
    end_timestamp: str | None = None

    @property
    def drifted_features(self) -> list[str]:
        return [item.feature for item in self.features if item.drifted]

    @property
    def any_drift(self) -> bool:
        return bool(self.drifted_features)

    @property
    def max_ks_statistic(self) -> float:
        return max((item.ks_statistic for item in self.features), default=0.0)

    @property
    def severity(self) -> str:
        if not self.evaluated or not self.any_drift:
            return "low"
        statistic = self.max_ks_statistic
        if statistic >= 0.35:
            base = 3
        elif statistic >= 0.20:
            base = 2
        elif statistic >= 0.10:
            base = 1
        else:
            base = 0
        if len(self.drifted_features) >= 3:
            base = min(base + 1, 3)
        return ("low", "medium", "high", "critical")[base]

    @property
    def reason(self) -> str:
        if not self.evaluated:
            return f"drift test skipped: {self.skip_reason}"
        if not self.any_drift:
            return (
                f"no drift: largest KS statistic {self.max_ks_statistic:.3f} below "
                f"effect-size floor or p_adj >= {self.alpha:g}"
            )
        worst = max(self.features, key=lambda item: item.ks_statistic)
        return (
            f"KS two-sample vs reference (n={self.n_reference}) on {self.n_window}-point window, "
            f"Holm-adjusted alpha={self.alpha:g}: {worst.feature} D={worst.ks_statistic:.3f} "
            f"(p_adj={worst.p_adjusted:.3g}), {len(self.drifted_features)}/{len(self.features)} "
            f"features drifted, largest mean shift {worst.feature} {worst.mean_shift_pct:+.1f}%"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "n_reference": self.n_reference,
            "n_window": self.n_window,
            "alpha": self.alpha,
            "correction": self.correction,
            "evaluated": bool(self.evaluated),
            "skip_reason": self.skip_reason,
            "any_drift": bool(self.any_drift),
            "drifted_features": self.drifted_features,
            "max_ks_statistic": round(self.max_ks_statistic, 6),
            "severity": self.severity,
            "reason": self.reason,
            "features": [item.to_dict() for item in self.features],
        }


def population_stability_index(
    reference: np.ndarray,
    window: np.ndarray,
    bins: int = 10,
) -> float:
    """PSI between ``reference`` and ``window`` using equal-frequency reference bins.

    ``PSI`` below 0.1 is conventionally treated as stable, 0.1-0.25 as a moderate
    shift and above 0.25 as a significant shift. Reported as supporting evidence
    only; the drift decision is made by the corrected KS test.
    """
    reference = np.asarray(reference, dtype=float)
    window = np.asarray(window, dtype=float)
    if reference.size == 0 or window.size == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    window_counts, _ = np.histogram(window, bins=edges)
    reference_share = reference_counts / max(reference_counts.sum(), 1)
    window_share = window_counts / max(window_counts.sum(), 1)
    reference_share = np.clip(reference_share, _PSI_EPSILON, None)
    window_share = np.clip(window_share, _PSI_EPSILON, None)
    return float(np.sum((window_share - reference_share) * np.log(window_share / reference_share)))


def adjust_pvalues(p_values: np.ndarray | list[float], method: str = "holm") -> np.ndarray:
    """Correct a family of p-values for multiple comparisons.

    ``holm`` is the default (uniformly more powerful than Bonferroni while still
    controlling the family-wise error rate). Adjusted values are clipped to 1.0.
    """
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values
    if method == "none":
        return values
    if method == "bonferroni":
        return np.minimum(values * values.size, 1.0)
    if method == "holm":
        # Holm step-down: sort ascending, scale p_(i) by (m - i), then enforce
        # monotonicity with a running maximum before mapping back to the input order.
        order = np.argsort(values, kind="stable")
        ranked = values[order]
        m = values.size
        scaled = np.minimum(ranked * (m - np.arange(m)), 1.0)
        adjusted_ranked = np.maximum.accumulate(scaled)
        out = np.empty_like(adjusted_ranked)
        out[order] = adjusted_ranked
        return np.minimum(out, 1.0)
    raise ValueError(f"unknown p-value correction method: {method!r}")


def compare_window(
    reference: pd.DataFrame,
    window: pd.DataFrame,
    config: DriftConfig,
    window_id: int = 0,
    start_position: int = 0,
) -> DriftReport:
    """Compare one window against the reference block feature by feature."""
    features = tuple(config.features)
    missing = [
        column
        for column in features
        if column not in reference.columns or column not in window.columns
    ]
    if missing:
        raise KeyError(f"drift comparison is missing feature columns {missing}")

    timestamps = _timestamps(window)
    n_window = len(window)
    report = DriftReport(
        window_id=window_id,
        start_position=start_position,
        end_position=start_position + max(n_window - 1, 0),
        n_reference=len(reference),
        n_window=n_window,
        alpha=config.alpha,
        correction=config.correction,
        start_timestamp=timestamps[0],
        end_timestamp=timestamps[1],
    )

    if n_window < config.min_window_size:
        report.evaluated = False
        report.skip_reason = (
            f"window has {n_window} observations, minimum_window_size is {config.min_window_size}"
        )
        return report
    if len(reference) < config.min_window_size:
        report.evaluated = False
        report.skip_reason = f"reference has {len(reference)} observations, minimum_window_size is {config.min_window_size}"
        return report

    raw_p_values: list[float] = []
    drafts: list[FeatureDrift] = []
    for feature in features:
        reference_values = reference[feature].to_numpy(dtype=float)
        window_values = window[feature].to_numpy(dtype=float)
        result = stats.ks_2samp(reference_values, window_values)
        reference_mean = float(np.mean(reference_values))
        window_mean = float(np.mean(window_values))
        shift_pct = (
            (window_mean - reference_mean) / abs(reference_mean) * 100.0
            if reference_mean != 0
            else 0.0
        )
        raw_p_values.append(float(result.pvalue))
        drafts.append(
            FeatureDrift(
                feature=feature,
                n_reference=int(reference_values.size),
                n_window=int(window_values.size),
                ks_statistic=float(result.statistic),
                p_value=float(result.pvalue),
                p_adjusted=float("nan"),
                drifted=False,
                reference_mean=reference_mean,
                window_mean=window_mean,
                mean_shift_pct=shift_pct,
                reference_std=float(np.std(reference_values)),
                window_std=float(np.std(window_values)),
                psi=population_stability_index(reference_values, window_values, config.psi_bins),
                unit=FEATURE_UNITS.get(feature, ""),
            )
        )

    adjusted = adjust_pvalues(raw_p_values, config.correction)
    for draft, adjusted_value in zip(drafts, adjusted, strict=False):
        draft.p_adjusted = float(adjusted_value)
        draft.drifted = bool(
            adjusted_value < config.alpha and draft.ks_statistic >= config.effect_size_floor
        )
    report.features = drafts
    return report


def scan_drift(
    reference: pd.DataFrame,
    stream: pd.DataFrame,
    config: DriftConfig,
    start_position: int = 0,
) -> list[DriftReport]:
    """Slide a window over ``stream`` and compare each window with ``reference``."""
    n_stream = len(stream)
    reports: list[DriftReport] = []
    if n_stream < config.min_window_size:
        reports.append(
            compare_window(reference, stream, config, window_id=0, start_position=start_position)
        )
        return reports
    window_id = 0
    position = 0
    while position + config.min_window_size <= n_stream:
        window = stream.iloc[position : position + config.window_size]
        reports.append(
            compare_window(
                reference,
                window,
                config,
                window_id=window_id,
                start_position=start_position + position,
            )
        )
        window_id += 1
        if position + config.window_size >= n_stream:
            break
        position += config.step
    return reports


def _timestamps(window: pd.DataFrame) -> tuple[str | None, str | None]:
    if "timestamp" not in window.columns or window.empty:
        return None, None
    return str(window["timestamp"].iloc[0]), str(window["timestamp"].iloc[-1])


def drift_feature_frame(reports: list[DriftReport]) -> pd.DataFrame:
    """Long-format frame of per-feature per-window statistics (for plotting)."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        for item in report.features:
            rows.append(
                {
                    "window_id": report.window_id,
                    "start_position": report.start_position,
                    "start_timestamp": report.start_timestamp,
                    "feature": item.feature,
                    "ks_statistic": item.ks_statistic,
                    "p_value": item.p_value,
                    "p_adjusted": item.p_adjusted,
                    "drifted": item.drifted,
                    "psi": item.psi,
                    "mean_shift_pct": item.mean_shift_pct,
                }
            )
    return pd.DataFrame(rows)


def drift_summary(reports: list[DriftReport]) -> pd.DataFrame:
    """One row per window with the headline drift verdict."""
    return pd.DataFrame(
        [
            {
                "window_id": report.window_id,
                "start_position": report.start_position,
                "start_timestamp": report.start_timestamp,
                "end_timestamp": report.end_timestamp,
                "n_window": report.n_window,
                "evaluated": report.evaluated,
                "any_drift": report.any_drift,
                "drifted_features": ",".join(report.drifted_features),
                "n_drifted": len(report.drifted_features),
                "max_ks_statistic": report.max_ks_statistic,
                "severity": report.severity,
                "reason": report.reason,
            }
            for report in reports
        ]
    )
