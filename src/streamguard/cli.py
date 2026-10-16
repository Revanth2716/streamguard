"""Headless entry point: run the pipeline without Streamlit.

Examples
--------
``python -m streamguard.cli --config configs/default.toml``
``python -m streamguard.cli --seed 7 --n-stream 600 --no-save``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from streamguard.config import AppConfig, default_config, load_config
from streamguard.figures import (
    confusion_heatmap,
    drift_heatmap,
    drift_ks_lines,
    feature_timeseries,
    metrics_bar,
    score_timeseries,
    write_figures,
)
from streamguard.pipeline import (
    PRIMARY_METRICS_COLUMNS,
    PipelineResult,
    run_pipeline,
    save_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streamguard",
        description="Run the StreamGuard anomaly/drift pipeline on simulated telemetry.",
    )
    parser.add_argument("--config", type=Path, default=None, help="path to a TOML config file")
    parser.add_argument("--seed", type=int, default=None, help="override the simulator seed")
    parser.add_argument(
        "--n-stream", type=int, default=None, help="override the number of stream rows"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not write reports/, data/processed/ artifacts",
    )
    parser.add_argument("--json", action="store_true", help="print the evaluation report as JSON")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.seed is not None:
        config.simulator.seed = args.seed
    if args.n_stream is not None:
        if args.n_stream < 100:
            raise SystemExit("--n-stream must be >= 100")
        config.simulator.n_stream = args.n_stream
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else default_config()
    config = _apply_overrides(config, args)

    result = run_pipeline(config)

    if args.json:
        print(json.dumps(result.report, indent=2, default=str))
    else:
        _print_summary(result)
        if not args.no_save:
            written = save_artifacts(result)
            figures = write_figures(
                {
                    "metrics_timeseries": feature_timeseries(
                        result.flagged, result.dataset.features, "isolation_forest"
                    ),
                    "score_timeseries": score_timeseries(
                        result.flagged, "isolation_forest", config.detector.zscore_threshold
                    ),
                    "metrics_bar": metrics_bar(result.metrics),
                    "confusion_matrix": confusion_heatmap(_confusion_frame(result)),
                    "drift_heatmap": drift_heatmap(result.drift_features),
                    "drift_ks": drift_ks_lines(
                        result.drift_features, config.drift.effect_size_floor
                    ),
                },
                Path(config.output_dir) / "figures",
            )
            print("\nartifacts written:")
            for name, path in {**written, **figures}.items():
                print(f"  {name:<20} {path}")
    return 0


def _confusion_frame(result: PipelineResult) -> pd.DataFrame:
    from streamguard.evaluation import confusion_matrix_frame, evaluate_detector

    metrics = evaluate_detector(
        result.report["primary_detector"],
        result.dataset.truth(),
        result.results[result.report["primary_detector"]]["is_anomaly"].to_numpy(dtype=bool),
    )
    return confusion_matrix_frame(metrics)


def _print_summary(result: PipelineResult) -> None:
    metadata = result.dataset.metadata
    print("StreamGuard - simulated telemetry only, no real operational data")
    print("=" * 78)
    print(f"seed .................. {metadata['seed']}")
    print(f"reference rows ........ {metadata['n_reference']} (anomaly-free, used for fitting)")
    print(f"stream rows ........... {metadata['n_stream']}")
    print(
        f"injected events ....... {metadata['n_anomaly_events']} "
        f"({metadata['n_anomaly_rows']} rows, rate {metadata['anomaly_rate']:.3%})"
    )
    print(
        f"distribution shift .... from {metadata['drift_start_timestamp']} "
        f"({metadata['drift_rows']} rows, {metadata['drift_component']})"
    )
    print()
    print("detector status")
    print(result.detector_status()[PRIMARY_METRICS_COLUMNS].to_string(index=False))
    print()
    print("evaluation on the labelled simulated stream (sorted by F1)")
    columns = [
        "detector",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    print(result.metrics[columns].round(3).to_string(index=False))
    print()
    print("per-segment F1 (pre-drift vs injected-shift period)")
    if not result.segments.empty:
        print(
            result.segments.pivot_table(index="detector", columns="segment", values="f1")
            .round(3)
            .to_string()
        )
    print()
    print("drift scan")
    for report in result.drift_reports:
        print(f"  window {report.window_id}: {report.reason}")
    counts = pd.Series([alert.severity for alert in result.alerts]).value_counts().to_dict()
    print()
    print(f"alerts raised ......... {len(result.alerts)} {counts}")


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
