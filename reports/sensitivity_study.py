"""Measure how the two most important knobs move the metrics.

This is a development/analysis helper (it is not part of the test suite). It runs
the real pipeline several times on the *same* seeded data and records what each
setting does, so the numbers quoted in README.md / MODEL_CARD.md are measured
rather than asserted::

    .venv\\Scripts\\python.exe reports\\sensitivity_study.py

Output: ``reports/metrics/sensitivity.json``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from streamguard.config import AppConfig, default_config
from streamguard.detector import run_detectors
from streamguard.drift import drift_summary, scan_drift
from streamguard.evaluation import evaluate_all
from streamguard.simulator import generate_dataset


def contamination_sweep(config: AppConfig) -> list[dict[str, object]]:
    """Isolation Forest F1 as its contamination prior is varied (same data)."""
    dataset = generate_dataset(config.simulator)
    rows: list[dict[str, object]] = []
    for contamination in (0.02, 0.05, 0.08, 0.12):
        detector_config = replace(config.detector, contamination=contamination)
        results = run_detectors(dataset.reference, dataset.stream, detector_config)
        metrics = evaluate_all(results, dataset.truth())
        row = metrics[metrics["detector"] == "isolation_forest"].iloc[0]
        rows.append(
            {
                "contamination": contamination,
                "flags": int(results["isolation_forest"]["is_anomaly"].sum()),
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
                "f1": round(float(row["f1"]), 4),
                "true_anomaly_rate": round(float(dataset.metadata["anomaly_rate"]), 4),
            }
        )
    return rows


def drift_sweep(config: AppConfig) -> list[dict[str, object]]:
    """False and true drift alerts as the effect-size floor is lowered."""
    dataset = generate_dataset(config.simulator)
    shifted_from = int(np.argmax(dataset.stream["drift_active"].to_numpy()))
    rows: list[dict[str, object]] = []
    for floor in (0.0, 0.10, 0.15, 0.20, 0.30):
        drift_config = replace(config.drift, effect_size_floor=floor)
        reports = scan_drift(dataset.reference, dataset.stream, drift_config)
        summary = drift_summary(reports)
        clean = summary[summary["start_position"] + drift_config.window_size <= shifted_from]
        shifted = summary[summary["start_position"] >= shifted_from]
        rows.append(
            {
                "effect_size_floor": floor,
                "clean_windows": int(len(clean)),
                "clean_windows_flagged": int(clean["any_drift"].sum()),
                "shifted_windows": int(len(shifted)),
                "shifted_windows_flagged": int(shifted["any_drift"].sum()),
                "max_ks_on_clean": round(float(clean["max_ks_statistic"].max()) if len(clean) else 0.0, 4),
            }
        )
    return rows


def autocorrelation_check(config: AppConfig) -> list[dict[str, object]]:
    """How the AR(1) coefficient changes the drift detector's false alerts.

    The two-sample KS test assumes independent samples. Autocorrelated telemetry
    shrinks the effective sample size, so the test over-rejects; this measures the
    size of that effect on clean windows.
    """
    rows: list[dict[str, object]] = []
    for phi in (0.0, 0.20, 0.35, 0.50, 0.72):
        simulator = replace(config.simulator, ar1_phi=phi)
        dataset = generate_dataset(simulator)
        shifted_from = int(np.argmax(dataset.stream["drift_active"].to_numpy()))
        drift_config = replace(config.drift, effect_size_floor=0.0)
        summary = drift_summary(scan_drift(dataset.reference, dataset.stream, drift_config))
        clean = summary[summary["start_position"] + drift_config.window_size <= shifted_from]
        rows.append(
            {
                "ar1_phi": phi,
                "clean_windows": int(len(clean)),
                "clean_windows_flagged_no_floor": int(clean["any_drift"].sum()),
                "max_ks_on_clean": round(float(clean["max_ks_statistic"].max()) if len(clean) else 0.0, 4),
            }
        )
    return rows


def main() -> int:
    config = default_config()
    report = {
        "data_origin": "simulated",
        "config": config.to_dict(),
        "note": (
            "Same seeded dataset (seed %d) for every row of each sweep; only the named "
            "setting changes. Numbers are measured, not assumed." % config.simulator.seed
        ),
        "contamination_sweep": contamination_sweep(config),
        "drift_effect_size_floor_sweep": drift_sweep(config),
        "drift_autocorrelation_sweep": autocorrelation_check(config),
    }
    target = Path(config.output_dir) / "metrics" / "sensitivity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for key in ("contamination_sweep", "drift_effect_size_floor_sweep", "drift_autocorrelation_sweep"):
        print(f"\n{key}")
        for row in report[key]:
            print("  ", row)
    print(f"\nwritten: {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - analysis helper
    raise SystemExit(main())
