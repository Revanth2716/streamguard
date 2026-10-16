"""Simulated event stream generation.

This module fabricates timestamped telemetry for a small set of service metrics.
Every row it produces is synthetic: there is no connection to a real service, no
live ingestion, and no real operational data anywhere in this project. Each
generated row carries a ``data_origin`` column with the value ``"simulated"`` so
that downstream consumers can never confuse it with production data.

The generator is fully deterministic given a seed. It produces:

* a *reference* block of normal behaviour (used to fit detectors), and
* a *stream* block containing normal behaviour, injected point anomalies, and an
  optional distribution shift part-way through.

Ground-truth labels (``is_anomaly``, ``anomaly_id``, ``anomaly_kind``,
``anomaly_features``) are attached only to the stream block and are never used
when fitting a detector.

Reference: the exact simulation config and seed are recorded in
:attr:`Dataset.metadata` and persisted with every evaluation report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from streamguard.config import FEATURE_COLUMNS, ShiftConfig, SimulatorConfig

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_LABELS",
    "FEATURE_UNITS",
    "Dataset",
    "anomaly_events",
    "extend_dataset",
    "generate_dataset",
    "generate_observations",
    "inject_point_anomalies",
    "normal_block",
    "robust_scale",
]

SIMULATED_ORIGIN = "simulated"

FEATURE_LABELS: dict[str, str] = {
    "latency_ms": "Latency (ms)",
    "cpu_util": "CPU utilisation",
    "mem_util": "Memory utilisation",
    "request_rate": "Request rate (req/s)",
}

FEATURE_UNITS: dict[str, str] = {
    "latency_ms": "ms",
    "cpu_util": "fraction of capacity",
    "mem_util": "fraction of capacity",
    "request_rate": "requests/second",
}

# Hard physical/plausible bounds applied after every transform.
_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "latency_ms": (5.0, 5000.0),
    "cpu_util": (0.02, 0.99),
    "mem_util": (0.05, 0.99),
    "request_rate": (1.0, 20000.0),
}


@dataclass
class Dataset:
    """A reproducible reference/stream pair with ground-truth stream labels."""

    reference: pd.DataFrame
    stream: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> tuple[str, ...]:
        return FEATURE_COLUMNS

    def truth(self) -> pd.Series:
        """Boolean ground-truth anomaly labels for the stream block."""
        return self.stream["is_anomaly"].astype(bool)

    def describe(self) -> str:
        labelled = self.stream.loc[self.stream["is_anomaly"].astype(bool), "anomaly_id"]
        n_events = int(labelled.nunique())
        return (
            f"SimulatedDataset(reference={len(self.reference)} rows, "
            f"stream={len(self.stream)} rows, anomaly_events={n_events})"
        )


def robust_scale(frame: pd.DataFrame, features: tuple[str, ...] = FEATURE_COLUMNS) -> pd.Series:
    """Median-absolute-deviation scale per feature (falls back to std then 1.0)."""
    median = frame.loc[:, list(features)].median()
    mad = (frame.loc[:, list(features)] - median).abs().median()
    scale = mad * 1.4826
    std = frame.loc[:, list(features)].std(ddof=0)
    scale = scale.where(scale > 1e-12, std)
    return scale.where(scale > 1e-12, 1.0)


def _window_index(frame: pd.DataFrame, start: int, stop: int) -> pd.Index:
    """Positional index labels for the half-open range ``[start, stop)``."""
    return frame.index.take(np.arange(start, stop))


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) noise with standard deviation ``sigma``."""
    if n <= 0:
        return np.empty(0, dtype=float)
    if phi <= 0:
        return rng.normal(0.0, sigma, size=n)
    innovations = rng.normal(0.0, sigma * np.sqrt(1.0 - phi * phi), size=n)
    return lfilter([1.0], [1.0, -phi], innovations)


def _clip_features(frame: pd.DataFrame) -> pd.DataFrame:
    for column, (low, high) in _CLIP_BOUNDS.items():
        if column in frame.columns:
            frame[column] = frame[column].clip(low, high)
    return frame


def _shift_multiplier(feature: str, shift: ShiftConfig) -> float:
    if feature == "latency_ms":
        return shift.latency_mult
    if feature == "request_rate":
        return shift.request_rate_mult
    return 1.0


def _shift_offset(feature: str, shift: ShiftConfig) -> float:
    if feature == "cpu_util":
        return shift.cpu_delta
    if feature == "mem_util":
        return shift.mem_delta
    return 0.0


def normal_block(
    config: SimulatorConfig,
    rng: np.random.Generator,
    n: int,
    start_index: int,
    shift: ShiftConfig | None = None,
) -> pd.DataFrame:
    """Generate ``n`` normal observations starting at global index ``start_index``.

    The latent "load" process is shared across features so the metrics are
    correlated, which is what makes a multivariate detector (Isolation Forest)
    meaningfully different from a per-feature threshold rule.
    """
    phi = config.ar1_phi
    load = _ar1(rng, n, phi, config.load_sigma)
    indices = np.arange(start_index, start_index + n)
    cycle_points = max(4, int(config.cycle_points))
    diurnal = np.sin(2.0 * np.pi * indices / cycle_points)

    columns: dict[str, np.ndarray] = {}
    for feature in FEATURE_COLUMNS:
        base = config.baselines[feature]
        sensitivity = config.load_sensitivity[feature]
        amplitude = config.diurnal_amplitude[feature]
        # Unit-variance AR(1) shaped noise, scaled by the feature's relative noise.
        noise = _ar1(rng, n, phi * 0.8, 1.0)
        noise_fraction = config.noise[feature]
        if shift is not None:
            base = base * _shift_multiplier(feature, shift)
            noise_fraction = noise_fraction * shift.noise_mult
        if feature in config.multiplicative:
            # Log-normal: strictly positive and right-skewed, like real latency
            # and throughput. The mean shifts multiplicatively with the latent load.
            series = base * np.exp(
                amplitude * diurnal + sensitivity * load + noise_fraction * noise
            )
        else:
            series = (
                base * (1.0 + amplitude * diurnal + sensitivity * load)
                + base * noise_fraction * noise
            )
        if shift is not None:
            series = series + _shift_offset(feature, shift)
        columns[feature] = series

    frame = pd.DataFrame(columns)
    frame["timestamp"] = pd.Timestamp(config.start_timestamp) + pd.to_timedelta(
        indices * config.interval_seconds, unit="s"
    )
    frame["event_index"] = indices
    frame["data_origin"] = SIMULATED_ORIGIN
    return _clip_features(frame)


def inject_point_anomalies(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    config: SimulatorConfig,
    scale: pd.Series,
    n_events: int,
    allowed_ranges: list[tuple[int, int]],
    start_event_id: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Inject labelled point anomalies into ``frame`` in place and return events.

    ``allowed_ranges`` is a list of half-open (start, stop) positional ranges the
    injections must fall inside -- e.g. the pre-drift and post-drift regions.
    """
    frame = frame.copy()
    frame["is_anomaly"] = False
    frame["anomaly_id"] = -1
    frame["anomaly_kind"] = ""
    frame["anomaly_features"] = ""
    n_rows = len(frame)
    if n_events <= 0 or n_rows == 0 or not allowed_ranges:
        return frame, []

    max_len = max(1, config.anomaly_max_len)
    min_len = max(1, config.anomaly_min_len)
    occupied = np.zeros(n_rows, dtype=bool)
    events: list[dict[str, Any]] = []
    attempts = 0
    next_id = start_event_id
    kinds = list(config.anomaly_kinds)

    while len(events) < n_events and attempts < n_events * 40:
        attempts += 1
        lo, hi = allowed_ranges[int(rng.integers(0, len(allowed_ranges)))]
        lo = max(0, int(lo))
        hi = min(n_rows, int(hi))
        if hi - lo <= max_len + 1:
            continue
        kind = str(kinds[int(rng.integers(0, len(kinds)))])
        duration = int(rng.integers(min_len, max_len + 1))
        if kind == "burst":
            duration = max(duration, min(max_len, 4))
        start = int(rng.integers(lo, hi - duration))
        stop = start + duration
        # Keep a two-point buffer so events never touch each other.
        if occupied[max(0, start - 2) : min(n_rows, stop + 2)].any():
            continue
        magnitude = float(rng.uniform(config.anomaly_sigma_low, config.anomaly_sigma_high))
        affected = [
            str(name) for name in _apply_anomaly(frame, rng, kind, start, stop, magnitude, scale)
        ]
        positions = _window_index(frame, start, stop)
        occupied[start:stop] = True
        frame.loc[positions, "is_anomaly"] = True
        frame.loc[positions, "anomaly_id"] = next_id
        frame.loc[positions, "anomaly_kind"] = kind
        frame.loc[positions, "anomaly_features"] = ",".join(affected)
        events.append(
            {
                "anomaly_id": next_id,
                "kind": kind,
                "start_index": int(frame.loc[positions[0], "event_index"]),
                "stop_index": int(frame.loc[positions[-1], "event_index"]),
                "start_position": start,
                "duration": duration,
                "magnitude_sigma": round(magnitude, 3),
                "features": affected,
                "timestamp": frame.loc[positions[0], "timestamp"],
            }
        )
        next_id += 1

    frame[["is_anomaly", "anomaly_id"]] = frame[["is_anomaly", "anomaly_id"]].astype(
        {"is_anomaly": bool, "anomaly_id": int}
    )
    return frame, events


def _apply_anomaly(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    kind: str,
    start: int,
    stop: int,
    magnitude: float,
    scale: pd.Series,
) -> list[str]:
    positions = _window_index(frame, start, stop)
    if kind == "spike":
        features = list(
            rng.choice(FEATURE_COLUMNS, size=1 if rng.random() < 0.6 else 2, replace=False)
        )
        for feature in features:
            frame.loc[positions, feature] = (
                frame.loc[positions, feature] + magnitude * scale[feature]
            )
        return sorted(features)
    if kind == "burst":
        feature = str(rng.choice(FEATURE_COLUMNS))
        frame.loc[positions, feature] = frame.loc[positions, feature] + magnitude * scale[feature]
        return [feature]
    if kind == "dropout":
        frame.loc[positions, "request_rate"] = frame.loc[positions, "request_rate"] * 0.15
        frame.loc[positions, "latency_ms"] = (
            frame.loc[positions, "latency_ms"] + max(1.0, magnitude * 0.6) * scale["latency_ms"]
        )
        return ["request_rate", "latency_ms"]
    # latency_storm
    frame.loc[positions, "latency_ms"] = (
        frame.loc[positions, "latency_ms"] + magnitude * scale["latency_ms"]
    )
    frame.loc[positions, "cpu_util"] = (
        frame.loc[positions, "cpu_util"] + max(1.0, magnitude * 0.5) * scale["cpu_util"]
    )
    return ["latency_ms", "cpu_util"]


def generate_observations(config: SimulatorConfig) -> pd.DataFrame:
    """Generate one labelled observation frame (reference block + stream block).

    Point anomalies are injected only into the stream block. The distribution
    shift, when enabled, begins part-way through the stream block and is marked
    by ``drift_active`` / ``drift_component`` (a shift is *not* labelled as a
    point anomaly: the two are different questions, see ``LIMITATIONS.md``).
    """
    config.validate()
    rng = np.random.default_rng(config.seed)

    reference = normal_block(config, rng, config.n_reference, start_index=0)
    reference["is_anomaly"] = False
    reference["anomaly_id"] = -1
    reference["anomaly_kind"] = ""
    reference["anomaly_features"] = ""
    reference["drift_active"] = False
    reference["drift_component"] = ""
    reference["is_reference"] = True

    shift = config.shift if config.shift.enabled else None
    if shift is not None:
        pre_len = int(config.n_stream * shift.start_fraction)
        post_len = config.n_stream - pre_len
        pre = normal_block(config, rng, pre_len, start_index=config.n_reference)
        post = normal_block(
            config, rng, post_len, start_index=config.n_reference + pre_len, shift=shift
        )
        stream = pd.concat([pre, post], ignore_index=True)
        stream["drift_active"] = [False] * pre_len + [True] * post_len
        stream["drift_component"] = [""] * pre_len + [shift.component] * post_len
        pre_len_rows = pre_len
    else:
        stream = normal_block(config, rng, config.n_stream, start_index=config.n_reference)
        stream["drift_active"] = False
        stream["drift_component"] = ""
        pre_len_rows = config.n_stream
    stream["is_reference"] = False

    scale = robust_scale(reference)
    allowed_ranges: list[tuple[int, int]] = []
    n_pre = round(config.n_point_anomalies * config.anomaly_fraction_pre_drift)
    n_post = config.n_point_anomalies - n_pre
    if shift is not None:
        if n_pre:
            allowed_ranges.append((0, pre_len_rows))
        if n_post:
            allowed_ranges.append((pre_len_rows, len(stream)))
    else:
        n_pre, n_post = config.n_point_anomalies, 0
        allowed_ranges.append((0, len(stream)))

    stream, events = inject_point_anomalies(
        stream,
        rng,
        config,
        scale,
        n_events=n_pre + n_post,
        allowed_ranges=allowed_ranges,
    )

    frame = pd.concat([reference, stream], ignore_index=True)
    frame["data_origin"] = SIMULATED_ORIGIN
    frame.attrs["simulated"] = True
    frame.attrs["anomaly_events"] = events
    frame.attrs["seed"] = config.seed
    return frame


def generate_dataset(config: SimulatorConfig) -> Dataset:
    """Generate a reference/stream :class:`Dataset` with metadata for reports."""
    frame = generate_observations(config)
    events: list[dict[str, Any]] = frame.attrs.get("anomaly_events", [])

    label_columns = ["is_anomaly", "anomaly_id", "anomaly_kind", "anomaly_features"]
    reference = frame[frame["is_reference"]].drop(columns=label_columns)
    stream = frame[~frame["is_reference"]].copy()
    stream = stream.reset_index(drop=True)
    reference = reference.reset_index(drop=True)

    scale = robust_scale(reference)
    drift_rows = int(stream["drift_active"].sum())
    metadata: dict[str, Any] = {
        "data_origin": SIMULATED_ORIGIN,
        "simulated": True,
        "seed": config.seed,
        "n_reference": len(reference),
        "n_stream": len(stream),
        "n_anomaly_events": len(events),
        "n_anomaly_rows": int(stream["is_anomaly"].sum()),
        "anomaly_rate": round(float(stream["is_anomaly"].mean()), 6),
        "drift_enabled": bool(config.shift.enabled),
        "drift_rows": drift_rows,
        "drift_component": config.shift.component if config.shift.enabled else None,
        "drift_start_timestamp": (
            str(stream.loc[stream["drift_active"], "timestamp"].iloc[0]) if drift_rows else None
        ),
        "reference_scale": {k: round(float(v), 6) for k, v in scale.items()},
        "features": list(FEATURE_COLUMNS),
        "feature_units": dict(FEATURE_UNITS),
        "interval_seconds": config.interval_seconds,
        "anomaly_events": [{**event, "timestamp": str(event["timestamp"])} for event in events],
        "config": config.to_dict(),
    }
    return Dataset(reference=reference, stream=stream, metadata=metadata)


def extend_dataset(
    config: SimulatorConfig,
    dataset: Dataset,
    n_new: int,
    *,
    seed: int | None = None,
    anomaly_kinds: Sequence[str] | None = None,
    n_anomaly_events: int = 0,
    shift: bool = False,
) -> Dataset:
    """Append new simulated observations to an existing dataset's stream.

    Used by the dashboard's live controls ("append events", "inject anomaly") and
    by the tests. The reference block is never modified, so a detector that was
    fit once stays valid for every appended row: appending cannot leak labels into
    fitting because fitting has already happened and never sees the stream.

    ``seed=None`` derives a deterministic seed from the stream position. Pass an
    explicit seed to reproduce a specific append.
    """
    if n_new < 1:
        raise ValueError("n_new must be >= 1")
    config.validate()
    stream = dataset.stream
    start_index = int(stream["event_index"].max()) + 1 if len(stream) else config.n_reference
    derived_seed = config.seed + start_index
    rng = np.random.default_rng(derived_seed if seed is None else seed)
    shift_config = config.shift if shift else None
    block = normal_block(config, rng, n_new, start_index=start_index, shift=shift_config)
    block["is_reference"] = False
    block["drift_active"] = bool(shift)
    block["drift_component"] = config.shift.component if shift else ""

    next_event_id = 1
    if len(stream):
        labelled = stream.loc[stream["is_anomaly"].astype(bool), "anomaly_id"]
        next_event_id = int(labelled.max()) + 1 if len(labelled) else 1
    new_events: list[dict[str, Any]] = []
    if n_anomaly_events > 0:
        kinds = tuple(anomaly_kinds) if anomaly_kinds else tuple(config.anomaly_kinds)
        unknown = set(kinds) - set(config.anomaly_kinds)
        if unknown:
            raise ValueError(f"unknown anomaly kinds requested: {sorted(unknown)}")
        injection_config = replace(config, anomaly_kinds=kinds)
        block, new_events = inject_point_anomalies(
            block,
            rng,
            injection_config,
            robust_scale(dataset.reference),
            n_events=n_anomaly_events,
            allowed_ranges=[(0, n_new)],
            start_event_id=next_event_id,
        )
    else:
        block["is_anomaly"] = False
        block["anomaly_id"] = -1
        block["anomaly_kind"] = ""
        block["anomaly_features"] = ""

    combined = pd.concat([stream, block], ignore_index=True)
    combined["data_origin"] = SIMULATED_ORIGIN
    metadata = dict(dataset.metadata)
    metadata["n_stream"] = len(combined)
    metadata["n_anomaly_rows"] = int(combined["is_anomaly"].sum())
    metadata["anomaly_rate"] = round(float(combined["is_anomaly"].mean()), 6)
    metadata["drift_rows"] = int(combined["drift_active"].sum())
    metadata["drift_start_timestamp"] = (
        str(combined.loc[combined["drift_active"].astype(bool), "timestamp"].iloc[0])
        if metadata["drift_rows"]
        else None
    )
    known_events = list(metadata.get("anomaly_events", []))
    known_events.extend({**event, "timestamp": str(event["timestamp"])} for event in new_events)
    metadata["anomaly_events"] = known_events
    metadata["n_anomaly_events"] = len(known_events)
    metadata["appended_rows"] = int(metadata.get("appended_rows", 0)) + n_new
    if shift:
        metadata["runtime_shift_appended"] = True
    return Dataset(reference=dataset.reference, stream=combined, metadata=metadata)


def anomaly_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per injected anomaly event (for detection-latency scoring)."""
    labelled = frame[frame["is_anomaly"].astype(bool)]
    if labelled.empty:
        return pd.DataFrame(
            columns=[
                "anomaly_id",
                "kind",
                "features",
                "start",
                "stop",
                "duration",
                "start_timestamp",
            ]
        )
    grouped = labelled.groupby("anomaly_id", sort=True)
    rows = []
    for anomaly_id, group in grouped:
        rows.append(
            {
                "anomaly_id": int(anomaly_id),
                "kind": str(group["anomaly_kind"].iloc[0]),
                "features": str(group["anomaly_features"].iloc[0]),
                "start": int(group["event_index"].min()),
                "stop": int(group["event_index"].max()),
                "duration": len(group),
                "start_timestamp": group["timestamp"].min(),
            }
        )
    return pd.DataFrame(rows)
