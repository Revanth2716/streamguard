"""Anomaly detection on simulated telemetry.

Four detectors are provided, all sharing one interface (``fit`` on a reference
block, ``score`` on an incoming block):

``rolling_zscore``
    The statistical baseline. A *causal* rolling z-score: for each observation
    the mean and standard deviation come from the previous ``zscore_window``
    observations only (the current point is excluded), so the score is
    computable online. It adapts to slow level changes -- which is exactly why
    it is a useful counter-example next to a detector that does not adapt.
``isolation_forest``
    The multivariate ML detector. A ``StandardScaler`` and ``IsolationForest``
    are fit on the reference block; the anomaly score is
    ``-score_samples`` (higher = more isolated) and the flag is the forest's own
    ``predict`` output.
``static_percentile``
    A naive fixed-threshold baseline: flag when any feature leaves the
    reference block's two-sided percentile envelope. No adaptivity at all.
``dummy``
    Never alerts. Included so that precision/recall tables have a floor.

Leakage control: ``fit`` receives only feature columns of the reference block.
Ground-truth labels exist in the stream frame but are never read here -- there is
a dedicated test (``tests/test_no_leakage.py``) that scrambles the stream labels
and asserts the scores are byte-for-byte identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from streamguard.config import FEATURE_COLUMNS, DetectorConfig
from streamguard.simulator import robust_scale

__all__ = [
    "SEVERITY_ORDER",
    "AnomalyDetector",
    "DummyDetector",
    "IsolationForestDetector",
    "RollingZScoreDetector",
    "StaticPercentileDetector",
    "build_detectors",
    "run_detectors",
    "severity_from_sigma",
]

SEVERITY_ORDER: tuple[str, ...] = ("low", "medium", "high", "critical")

OUTPUT_COLUMNS = (
    "detector",
    "score",
    "score_sigma",
    "is_anomaly",
    "severity",
    "reason",
    "top_feature",
)


def severity_from_sigma(value: float, bounds: tuple[float, float, float]) -> str:
    """Map a standardised magnitude onto a severity label.

    ``bounds`` holds the low/medium/high cut points in increasing order.
    """
    low, medium, high = bounds
    if value >= high:
        return "critical"
    if value >= medium:
        return "high"
    if value >= low:
        return "medium"
    return "low"


class AnomalyDetector(ABC):
    """Common interface: fit on reference data, score an incoming block."""

    name: str = "detector"

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.features: tuple[str, ...] = tuple(config.features)
        self.fitted = False

    # -- helpers ---------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name}: fit() must be called before score()")

    def _features_of(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.features if column not in frame.columns]
        if missing:
            raise KeyError(f"{self.name}: frame is missing feature columns {missing}")
        return frame.loc[:, list(self.features)]

    def _empty_output(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        result["detector"] = self.name
        result["score"] = 0.0
        result["score_sigma"] = 0.0
        result["is_anomaly"] = False
        result["severity"] = "low"
        result["reason"] = ""
        result["top_feature"] = ""
        return result

    # -- interface -------------------------------------------------------
    @abstractmethod
    def fit(self, reference: pd.DataFrame) -> AnomalyDetector:
        """Fit using reference features only (labels are never accepted here)."""

    @abstractmethod
    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return per-row detector output aligned with ``frame.index``."""

    def fit_score(self, reference: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(reference).score(frame)

    def describe(self) -> dict[str, Any]:
        return {"detector": self.name, "features": list(self.features), "fitted": self.fitted}


class RollingZScoreDetector(AnomalyDetector):
    """Causal rolling z-score baseline with a reference-derived scale floor."""

    name = "rolling_zscore"
    std_floor_ratio = 0.25

    def fit(self, reference: pd.DataFrame) -> RollingZScoreDetector:
        features = self._features_of(reference)
        if features.empty:
            raise ValueError(f"{self.name}: empty reference block")
        self.reference_median_: pd.Series = features.median()
        self.reference_scale_: pd.Series = robust_scale(features, self.features)
        self.reference_tail_ = features.tail(self.config.zscore_window).reset_index(drop=True)
        self.fitted = True
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        features = self._features_of(frame)
        result = self._empty_output(frame)
        if features.empty:
            return result

        history = self.reference_tail_
        combined = pd.concat([history, features], ignore_index=True)
        n_history = len(history)
        window = self.config.zscore_window
        min_periods = min(self.config.zscore_min_periods, max(2, window // 2))

        z_matrix = np.empty((len(features), len(self.features)), dtype=float)
        means = np.empty_like(z_matrix)
        for position, feature in enumerate(self.features):
            series = combined[feature]
            previous = series.shift(1)
            rolling = previous.rolling(window, min_periods=min_periods)
            mean = rolling.mean().fillna(self.reference_median_[feature])
            scale = rolling.std()
            floor = self.reference_scale_[feature] * self.std_floor_ratio
            scale = scale.fillna(self.reference_scale_[feature]).clip(lower=floor)
            z = ((series - mean) / scale).to_numpy()[n_history:]
            z_matrix[:, position] = z
            means[:, position] = mean.to_numpy()[n_history:]

        abs_z = np.abs(z_matrix)
        best = abs_z.argmax(axis=1)
        score = abs_z[np.arange(len(features)), best]
        top_feature = [self.features[i] for i in best]
        threshold = self.config.zscore_threshold

        result["score"] = score
        result["score_sigma"] = score
        result["is_anomaly"] = score >= threshold
        result["top_feature"] = top_feature
        np.arange(len(features))
        flagged_mask = np.asarray(score >= threshold)
        result["reason"] = [
            (
                f"rolling z-score: {top_feature[i]} |z|={score[i]:.2f} >= {threshold:.2f} "
                f"(causal window {window}, baseline mean {means[i, best[i]]:.2f})"
            )
            if flagged_mask[i]
            else ""
            for i in range(len(features))
        ]
        ratio = np.divide(score, threshold, out=np.zeros_like(score), where=threshold > 0)
        result["severity"] = [
            _severity_from_ratio(float(r)) if bool(flag) else "low"
            for flag, r in zip(flagged_mask, ratio, strict=True)
        ]
        for position, feature in enumerate(self.features):
            result[f"z_{feature}"] = z_matrix[:, position]
        return result


def _severity_from_ratio(ratio: float) -> str:
    if ratio >= 2.0:
        return "critical"
    if ratio >= 1.5:
        return "high"
    if ratio >= 1.2:
        return "medium"
    return "low"


class IsolationForestDetector(AnomalyDetector):
    """Multivariate detector: StandardScaler + IsolationForest fit on the reference.

    The forest's ``contamination`` parameter sets the decision offset, so the
    alert rate on the reference block is approximately ``contamination`` by
    construction. Severity uses the reference distribution of anomaly scores:
    a point is ``critical`` at 4 reference-sigma, ``high`` at 3, ``medium`` at 2.

    ``reason`` lists the largest standardised deviations from the reference
    median. This is post-hoc attribution of the flagged point, not a causal
    explanation and not SHAP -- see ``LIMITATIONS.md``.
    """

    name = "isolation_forest"

    def fit(self, reference: pd.DataFrame) -> IsolationForestDetector:
        features = self._features_of(reference)
        if len(features) < 10:
            raise ValueError(
                f"{self.name}: reference block too small to fit ({len(features)} rows)"
            )
        self.scaler_ = StandardScaler().fit(features.to_numpy())
        self.model_ = IsolationForest(
            n_estimators=self.config.if_n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.if_random_state,
            n_jobs=1,
        ).fit(self.scaler_.transform(features.to_numpy()))
        self.reference_median_ = features.median()
        self.reference_scale_ = robust_scale(features, self.features)
        reference_scores = -self.model_.score_samples(self.scaler_.transform(features.to_numpy()))
        self.reference_score_mean_ = float(np.mean(reference_scores))
        self.reference_score_std_ = float(np.std(reference_scores)) or 1.0
        self.reference_quantiles_ = {
            label: float(np.quantile(reference_scores, quantile))
            for label, quantile in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p999", 0.999))
        }
        self.fitted = True
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        features = self._features_of(frame)
        result = self._empty_output(frame)
        if features.empty:
            return result

        scaled = self.scaler_.transform(features.to_numpy())
        raw = -self.model_.score_samples(scaled)
        flagged = self.model_.predict(scaled) == -1
        sigma = (raw - self.reference_score_mean_) / self.reference_score_std_

        deviation = (
            features.to_numpy() - self.reference_median_.to_numpy()
        ) / self.reference_scale_.to_numpy()
        abs_deviation = np.abs(deviation)
        order = np.argsort(-abs_deviation, axis=1)
        top_index = order[:, 0]
        top_feature = [self.features[i] for i in top_index]
        bounds = self.config.if_severity_sigma

        result["score"] = raw
        result["score_sigma"] = sigma
        result["is_anomaly"] = flagged
        result["top_feature"] = top_feature
        reasons = []
        for i in range(len(features)):
            if not flagged[i]:
                reasons.append("")
                continue
            first, second = order[i, 0], order[i, 1]
            reasons.append(
                "isolation forest outlier "
                f"(contamination={self.config.contamination:g}); standardised deviations: "
                f"{self.features[first]} {deviation[i, first]:+.2f}, "
                f"{self.features[second]} {deviation[i, second]:+.2f}"
            )
        result["reason"] = reasons
        result["severity"] = [
            severity_from_sigma(float(s), bounds) if flag else "low"
            for s, flag in zip(sigma, flagged, strict=False)
        ]
        for position, feature in enumerate(self.features):
            result[f"dev_{feature}"] = deviation[:, position]
        return result


class StaticPercentileDetector(AnomalyDetector):
    """Naive baseline: fixed two-sided percentile envelope from the reference."""

    name = "static_percentile"
    #: severity cut points expressed in reference-sigma beyond the envelope
    severity_bounds: tuple[float, float, float] = (1.0, 2.0, 3.0)

    def fit(self, reference: pd.DataFrame) -> StaticPercentileDetector:
        features = self._features_of(reference)
        quantile = self.config.static_threshold_quantile
        self.lower_ = features.quantile(1.0 - quantile)
        self.upper_ = features.quantile(quantile)
        self.reference_median_ = features.median()
        self.reference_scale_ = robust_scale(features, self.features)
        self.fitted = True
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        features = self._features_of(frame)
        result = self._empty_output(frame)
        if features.empty:
            return result

        below = (features < self.lower_).to_numpy()
        above = (features > self.upper_).to_numpy()
        outside = below | above
        values = features.to_numpy()
        # ``to_numpy`` may return a read-only view (pandas 3 / numpy 2), so copy
        # before normalising any feature whose envelope collapsed to a point.
        span = np.array(self.upper_ - self.lower_, dtype=float)
        collapsed = span <= 0
        span[collapsed] = np.maximum(self.reference_scale_.to_numpy(), 1e-12)[collapsed]
        distance = np.where(above, values - self.upper_.to_numpy(), 0.0)
        distance = np.where(below, self.lower_.to_numpy() - values, distance)
        scale = np.maximum(self.reference_scale_.to_numpy(), 1e-12)
        # distance normalised by the reference scale: interpretable as sigma
        normalized = distance / span
        sigma_distance = distance / scale
        score = normalized.max(axis=1)
        best = normalized.argmax(axis=1)
        flagged = outside.any(axis=1)

        result["score"] = score
        result["score_sigma"] = sigma_distance[np.arange(len(features)), best]
        result["is_anomaly"] = flagged
        result["top_feature"] = [self.features[i] for i in best]
        quantile = self.config.static_threshold_quantile
        result["reason"] = [
            (
                f"static percentile: {self.features[best[i]]}="
                f"{features.iloc[i, best[i]]:.4g} outside reference "
                f"[{self.lower_.iloc[best[i]]:.4g}, {self.upper_.iloc[best[i]]:.4g}] "
                f"({quantile:.3%} envelope)"
            )
            if flagged[i]
            else ""
            for i in range(len(features))
        ]
        result["severity"] = [
            severity_from_sigma(float(sigma_distance[i, best[i]]), self.severity_bounds)
            if flagged[i]
            else "low"
            for i in range(len(features))
        ]
        return result


class DummyDetector(AnomalyDetector):
    """Never alerts. A floor for precision/recall tables."""

    name = "dummy"

    def fit(self, reference: pd.DataFrame) -> DummyDetector:
        self._features_of(reference)
        self.fitted = True
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        return self._empty_output(frame)


def build_detectors(config: DetectorConfig) -> dict[str, AnomalyDetector]:
    """Instantiate every detector (unfitted) in a stable order."""
    detectors: list[AnomalyDetector] = [
        DummyDetector(config),
        StaticPercentileDetector(config),
        RollingZScoreDetector(config),
        IsolationForestDetector(config),
    ]
    return {detector.name: detector for detector in detectors}


def run_detectors(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
    config: DetectorConfig,
) -> dict[str, pd.DataFrame]:
    """Fit every detector on ``reference`` and score ``frame``.

    ``reference`` and ``frame`` are positionally aligned only through their
    ``event_index`` values; the returned frames share ``frame``'s index.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in reference.columns]
    if missing:
        raise KeyError(f"reference block is missing feature columns {missing}")
    results: dict[str, pd.DataFrame] = {}
    for name, detector in build_detectors(config).items():
        output = detector.fit(reference).score(frame)
        if "event_index" in frame.columns:
            # Carried through so downstream joins/latency scoring never depend on
            # an index that windowing may have reset.
            output["event_index"] = frame["event_index"].to_numpy()
        results[name] = output
    return results
