"""Simulator tests: determinism, labelling, plausibility and extension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streamguard.config import VALID_ANOMALY_KINDS, AppConfig
from streamguard.simulator import (
    FEATURE_COLUMNS,
    SIMULATED_ORIGIN,
    anomaly_events,
    extend_dataset,
    generate_dataset,
    generate_observations,
    robust_scale,
)


def test_same_seed_is_bit_for_bit_reproducible(small_config: AppConfig) -> None:
    first = generate_dataset(small_config.simulator)
    second = generate_dataset(small_config.simulator)
    pd.testing.assert_frame_equal(first.reference, second.reference)
    pd.testing.assert_frame_equal(first.stream, second.stream)
    assert first.metadata == second.metadata


def test_different_seed_changes_the_data(small_config: AppConfig) -> None:
    import dataclasses

    other = dataclasses.replace(small_config.simulator, seed=small_config.simulator.seed + 1)
    base = generate_dataset(small_config.simulator)
    changed = generate_dataset(other)
    assert not base.stream["latency_ms"].equals(changed.stream["latency_ms"])
    assert base.metadata["seed"] != changed.metadata["seed"]


def test_reference_block_is_clean_and_labelled_simulated(small_config: AppConfig) -> None:
    dataset = generate_dataset(small_config.simulator)
    assert len(dataset.reference) == small_config.simulator.n_reference
    label_columns = {"is_anomaly", "anomaly_id", "anomaly_kind", "anomaly_features"}
    assert label_columns.isdisjoint(dataset.reference.columns)
    assert len(dataset.stream) == small_config.simulator.n_stream
    assert set(dataset.reference["data_origin"]) == {SIMULATED_ORIGIN}
    assert set(dataset.stream["data_origin"]) == {SIMULATED_ORIGIN}
    assert not dataset.stream["drift_active"].iloc[:100].any()


def test_anomaly_labels_are_consistent(small_config: AppConfig) -> None:
    dataset = generate_dataset(small_config.simulator)
    stream = dataset.stream
    labelled = stream.loc[stream["is_anomaly"].astype(bool)]
    assert len(labelled) > 0
    assert labelled["anomaly_id"].ge(1).all()
    assert set(labelled["anomaly_kind"]) <= set(VALID_ANOMALY_KINDS)
    assert (labelled["anomaly_features"].str.len() > 0).all()

    normal = stream.loc[~stream["is_anomaly"].astype(bool)]
    assert set(normal["anomaly_id"]) == {-1}
    assert set(normal["anomaly_kind"]) == {""}

    events = anomaly_events(stream)
    assert len(events) == dataset.metadata["n_anomaly_events"]
    assert len(events) == small_config.simulator.n_point_anomalies
    assert events["duration"].sum() == int(stream["is_anomaly"].sum())
    assert (events["duration"] >= 1).all()
    assert events["stop"].ge(events["start"]).all()


def test_injected_shift_changes_the_shifted_region(small_config: AppConfig) -> None:
    dataset = generate_dataset(small_config.simulator)
    stream = dataset.stream
    drift = stream.loc[stream["drift_active"].astype(bool)]
    pre = stream.loc[~stream["drift_active"].astype(bool)]
    assert len(drift) > 0
    assert dataset.metadata["drift_start_timestamp"] is not None
    # The injected shift raises latency and CPU; verify it actually did.
    assert drift["latency_ms"].median() > pre["latency_ms"].median() * 1.05
    assert drift["cpu_util"].median() > pre["cpu_util"].median()
    # A shifted period must not be labelled as a point anomaly.
    assert stream.loc[stream["drift_active"].astype(bool), "is_anomaly"].sum() < len(drift)


def test_shift_can_be_disabled(small_config: AppConfig) -> None:
    import dataclasses

    from streamguard.config import ShiftConfig

    config = dataclasses.replace(
        small_config.simulator, shift=dataclasses.replace(ShiftConfig(), enabled=False)
    )
    dataset = generate_dataset(config)
    assert not dataset.stream["drift_active"].any()
    assert dataset.metadata["drift_rows"] == 0
    assert dataset.metadata["drift_start_timestamp"] is None


def test_normal_observations_stay_inside_plausible_bounds(small_config: AppConfig) -> None:
    dataset = generate_dataset(small_config.simulator)
    normal = dataset.stream.loc[~dataset.stream["is_anomaly"].astype(bool)]
    assert (normal["latency_ms"] > 0).all()
    assert (normal["request_rate"] > 0).all()
    assert normal["cpu_util"].between(0.02, 0.99).all()
    assert normal["mem_util"].between(0.05, 0.99).all()
    assert (dataset.reference["latency_ms"] > 0).all()
    assert np.isfinite(dataset.stream[list(FEATURE_COLUMNS)].to_numpy()).all()


def test_timestamps_are_strictly_increasing_and_one_interval_apart(small_config: AppConfig) -> None:
    frame = generate_observations(small_config.simulator)
    deltas = frame["timestamp"].diff().dropna().dt.total_seconds().unique()
    assert list(deltas) == [float(small_config.simulator.interval_seconds)]


def test_robust_scale_is_positive() -> None:
    config = AppConfig()
    frame = generate_observations(config.simulator)
    scale = robust_scale(frame)
    assert (scale > 0).all()
    assert set(scale.index) == set(FEATURE_COLUMNS)


def test_extend_dataset_appends_labelled_rows(small_config: AppConfig) -> None:
    dataset = generate_dataset(small_config.simulator)
    before_reference = dataset.reference.copy()
    extended = extend_dataset(
        small_config.simulator,
        dataset,
        n_new=40,
        seed=1234,
        anomaly_kinds=["spike", "latency_storm"],
        n_anomaly_events=1,
    )

    assert len(extended.stream) == len(dataset.stream) + 40
    pd.testing.assert_frame_equal(extended.reference, before_reference)
    appended = extended.stream.tail(40)
    assert (appended["is_reference"] == False).all()  # noqa: E712 - explicit comparison reads better
    # One injected *event*, which may span several consecutive observations.
    injected = appended.loc[appended["is_anomaly"].astype(bool)]
    assert injected["anomaly_id"].nunique() == 1
    assert 1 <= len(injected) <= small_config.simulator.anomaly_max_len
    assert set(injected["anomaly_kind"]) <= {"spike", "latency_storm"}
    assert extended.metadata["n_anomaly_events"] == dataset.metadata["n_anomaly_events"] + 1
    assert extended.metadata["appended_rows"] == 40
    assert extended.stream["timestamp"].is_monotonic_increasing
    assert extended.stream["event_index"].is_monotonic_increasing

    again = extend_dataset(
        small_config.simulator,
        dataset,
        n_new=40,
        seed=1234,
        anomaly_kinds=["spike", "latency_storm"],
        n_anomaly_events=1,
    )
    pd.testing.assert_frame_equal(extended.stream, again.stream)


def test_extend_dataset_can_append_a_shift(small_config: AppConfig) -> None:
    dataset = generate_dataset(small_config.simulator)
    extended = extend_dataset(small_config.simulator, dataset, n_new=60, seed=99, shift=True)
    appended = extended.stream.tail(60)
    assert appended["drift_active"].all()
    assert extended.metadata["drift_rows"] == dataset.metadata["drift_rows"] + 60
    assert extended.metadata["runtime_shift_appended"] is True


def test_extend_dataset_rejects_bad_input(small_config: AppConfig) -> None:
    dataset = generate_dataset(small_config.simulator)
    with pytest.raises(ValueError):
        extend_dataset(small_config.simulator, dataset, n_new=0)
    with pytest.raises(ValueError):
        extend_dataset(
            small_config.simulator,
            dataset,
            n_new=30,
            anomaly_kinds=["unknown_kind"],
            n_anomaly_events=1,
        )
