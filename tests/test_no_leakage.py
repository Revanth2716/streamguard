"""Leakage tests.

These are the tests that back the claims in the report and the model card:

* detectors are fit on the reference block only, and
* ground-truth stream labels are never read by a detector.

Both are checked by perturbing the labels and asserting the *scores* do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from streamguard.config import DetectorConfig
from streamguard.detector import (
    IsolationForestDetector,
    RollingZScoreDetector,
    run_detectors,
)
from streamguard.pipeline import run_pipeline
from streamguard.simulator import FEATURE_COLUMNS


def test_scrambling_stream_labels_does_not_change_any_score(small_config, small_dataset) -> None:
    reference = small_dataset.reference
    baseline = run_detectors(reference, small_dataset.stream, small_config.detector)

    scrambled = small_dataset.stream.copy()
    scrambled["is_anomaly"] = ~scrambled["is_anomaly"].astype(bool)
    scrambled["anomaly_id"] = np.arange(len(scrambled)) % 5
    scrambled["anomaly_kind"] = "tampered"
    resumed = run_detectors(reference, scrambled, small_config.detector)

    for name, frame in baseline.items():
        assert np.allclose(frame["score"], resumed[name]["score"]), name
        assert frame["is_anomaly"].equals(resumed[name]["is_anomaly"]), name


def test_extra_label_columns_on_the_reference_block_are_ignored(small_dataset) -> None:
    config = DetectorConfig(if_n_estimators=40, if_random_state=7)
    reference = small_dataset.reference
    tagged = reference.copy()
    tagged["is_anomaly"] = True  # a hypothetical leak would change the fit
    tagged["anomaly_kind"] = "not_real"
    tagged["anomaly_id"] = 99

    clean_output = run_detectors(reference, small_dataset.stream, config)["isolation_forest"]
    tagged_output = run_detectors(tagged, small_dataset.stream, config)["isolation_forest"]
    assert np.allclose(clean_output["score"], tagged_output["score"])
    assert clean_output["is_anomaly"].equals(tagged_output["is_anomaly"])


def test_reference_statistics_come_from_the_reference_not_the_stream(small_dataset) -> None:
    detector = RollingZScoreDetector(DetectorConfig())
    detector.fit(small_dataset.reference)
    assert np.allclose(
        detector.reference_median_[list(FEATURE_COLUMNS)].to_numpy(),
        small_dataset.reference[list(FEATURE_COLUMNS)].median().to_numpy(),
    )
    assert not np.allclose(
        detector.reference_median_[list(FEATURE_COLUMNS)].to_numpy(),
        small_dataset.stream[list(FEATURE_COLUMNS)].median().to_numpy(),
    )
    # The tail used for the causal warm-up must also come from the reference.
    assert len(detector.reference_tail_) <= DetectorConfig().zscore_window
    expected_tail = (
        small_dataset.reference[list(FEATURE_COLUMNS)]
        .tail(len(detector.reference_tail_))
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(detector.reference_tail_, expected_tail)


def test_isolation_forest_severity_scale_is_reference_derived(small_dataset) -> None:
    detector = IsolationForestDetector(DetectorConfig(if_n_estimators=40))
    detector.fit(small_dataset.reference)
    assert detector.reference_score_mean_ < detector.reference_quantiles_["p99"]
    assert detector.reference_quantiles_["p999"] > detector.reference_quantiles_["p99"]


def test_labels_change_metrics_but_not_flags(small_config, small_dataset) -> None:
    """Metrics must be label-driven: flipping labels must change the numbers."""
    honest = run_pipeline(small_config, dataset=small_dataset)
    tampered = small_dataset.stream.copy()
    tampered["is_anomaly"] = False
    from streamguard.simulator import Dataset

    tampered_dataset = Dataset(
        reference=small_dataset.reference, stream=tampered, metadata=small_dataset.metadata
    )
    liar = run_pipeline(small_config, dataset=tampered_dataset)

    assert not np.allclose(
        honest.metrics.set_index("detector")["f1"].to_numpy(),
        liar.metrics.set_index("detector")["f1"].to_numpy(),
    )
    for name, frame in honest.results.items():
        assert np.allclose(frame["score"], liar.results[name]["score"]), name
