"""StreamGuard Streamlit dashboard.

Run with::

    streamlit run dashboard/app.py

Everything displayed here is produced by the local simulator in
:mod:`streamguard.simulator`. There is no network access, no external service and
no real operational data. The header banner states that explicitly, and every
chart title is labelled "simulated".

The app is a thin layer over :func:`streamguard.pipeline.run_pipeline`, so the
numbers shown here are identical to the ones written to ``reports/`` by the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from streamguard.alerts import (
    ALERT_TYPE_ANOMALY,
    ALERT_TYPE_DRIFT,
    AlertStore,
    alert_episodes,
)
from streamguard.config import AppConfig, default_config, load_config
from streamguard.evaluation import (
    confusion_matrix_frame,
    detection_latency,
    evaluate_detector,
    latency_summary,
    metrics_frame,
    oracle_threshold_sweep,
)
from streamguard.figures import (
    confusion_heatmap,
    drift_heatmap,
    drift_ks_lines,
    feature_timeseries,
    metrics_bar,
    score_timeseries,
    severity_bar,
)
from streamguard.pipeline import PRIMARY_METRICS_COLUMNS, PipelineResult, run_pipeline
from streamguard.simulator import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    FEATURE_UNITS,
    SIMULATED_ORIGIN,
    extend_dataset,
    generate_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
WIDTH = "stretch"
SIMULATED_BANNER = (
    "**All data on this page is SIMULATED.** StreamGuard fabricates telemetry "
    "locally with a seeded generator and injects known anomalies and a known "
    "distribution shift so that detection quality can be measured. No real "
    "service, sensor or production dataset is involved, and none of the numbers "
    "here are real-world performance claims. Every metric is tagged with "
    f"`data_origin = {SIMULATED_ORIGIN}`."
)

st.set_page_config(page_title="StreamGuard", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Cached pipeline
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Simulating telemetry and running detectors ...", max_entries=8)
def build_run(config_payload: str, appends_payload: str) -> PipelineResult:
    """Deterministic, cacheable pipeline run.

    Both payloads are JSON strings so that the cache key is a pure function of the
    configuration and of the list of runtime append operations.
    """
    config = AppConfig.from_dict(json.loads(config_payload))
    dataset = generate_dataset(config.simulator)
    for step in json.loads(appends_payload):
        dataset = extend_dataset(config.simulator, dataset, **step)
    return run_pipeline(config, dataset=dataset)


def build_config() -> AppConfig:
    """Read widget state into a validated :class:`AppConfig`."""
    state = st.session_state
    config = (
        load_config(CONFIG_DIR / state["sg_preset"]) if CONFIG_DIR.is_dir() else default_config()
    )
    simulator = config.simulator
    simulator.seed = int(state["sim_seed"])
    simulator.n_reference = int(state["sim_n_reference"])
    simulator.n_stream = int(state["sim_n_stream"])
    simulator.n_point_anomalies = int(state["sim_n_anomalies"])
    simulator.cycle_points = int(state["sim_cycle"])
    simulator.shift.enabled = bool(state["sim_shift_enabled"])
    simulator.shift.start_fraction = float(state["sim_shift_start"])
    simulator.shift.latency_mult = float(state["sim_shift_latency"])
    simulator.shift.cpu_delta = float(state["sim_shift_cpu"])
    simulator.shift.request_rate_mult = float(state["sim_shift_rate"])
    simulator.shift.noise_mult = float(state["sim_shift_noise"])
    detector = config.detector
    detector.zscore_threshold = float(state["det_z_threshold"])
    detector.zscore_window = int(state["det_z_window"])
    detector.contamination = float(state["det_contamination"])
    drift = config.drift
    drift.alpha = float(state["drift_alpha"])
    drift.correction = str(state["drift_correction"])
    drift.window_size = int(state["drift_window"])
    drift.min_window_size = int(state["drift_min_window"])
    drift.effect_size_floor = float(state["drift_floor"])
    config.validate()
    return config


CONFIG_WIDGET_KEYS = (
    "sim_seed",
    "sim_n_reference",
    "sim_n_stream",
    "sim_n_anomalies",
    "sim_cycle",
    "sim_shift_enabled",
    "sim_shift_start",
    "sim_shift_latency",
    "sim_shift_cpu",
    "sim_shift_rate",
    "sim_shift_noise",
    "det_z_threshold",
    "det_z_window",
    "det_contamination",
    "drift_alpha",
    "drift_correction",
    "drift_window",
    "drift_min_window",
    "drift_floor",
)


def seed_widgets(config: AppConfig) -> None:
    """Seed widget defaults from a preset exactly once per preset selection."""
    state = st.session_state
    simulator, detector, drift = config.simulator, config.detector, config.drift
    state.setdefault("sim_seed", simulator.seed)
    state.setdefault("sim_n_reference", simulator.n_reference)
    state.setdefault("sim_n_stream", simulator.n_stream)
    state.setdefault("sim_n_anomalies", simulator.n_point_anomalies)
    state.setdefault("sim_cycle", simulator.cycle_points)
    state.setdefault("sim_shift_enabled", simulator.shift.enabled)
    state.setdefault("sim_shift_start", simulator.shift.start_fraction)
    state.setdefault("sim_shift_latency", simulator.shift.latency_mult)
    state.setdefault("sim_shift_cpu", simulator.shift.cpu_delta)
    state.setdefault("sim_shift_rate", simulator.shift.request_rate_mult)
    state.setdefault("sim_shift_noise", simulator.shift.noise_mult)
    state.setdefault("det_z_threshold", detector.zscore_threshold)
    state.setdefault("det_z_window", detector.zscore_window)
    state.setdefault("det_contamination", detector.contamination)
    state.setdefault("drift_alpha", drift.alpha)
    state.setdefault("drift_correction", drift.correction)
    state.setdefault("drift_window", drift.window_size)
    state.setdefault("drift_min_window", drift.min_window_size)
    state.setdefault("drift_floor", drift.effect_size_floor)
    state.setdefault("appends", [])
    state.setdefault("append_counter", 0)


def next_append(**step: object) -> None:
    """Queue a runtime append operation and rerun."""
    state = st.session_state
    state["append_counter"] = int(state["append_counter"]) + 1
    step.setdefault("seed", 900_000 + int(state["append_counter"]))
    state["appends"].append(step)
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
presets = sorted(path.name for path in CONFIG_DIR.glob("*.toml")) if CONFIG_DIR.is_dir() else []
with st.sidebar:
    st.title("📈 StreamGuard")
    st.caption("Local-first anomaly & drift monitoring on simulated telemetry")
    preset = st.selectbox("Config preset", presets or ["(built-in defaults)"], key="sg_preset")
    if st.session_state.get("sg_preset_active") != preset:
        for key in CONFIG_WIDGET_KEYS:
            st.session_state.pop(key, None)
        st.session_state["sg_preset_active"] = preset

    seed_widgets(
        load_config(CONFIG_DIR / preset)
        if CONFIG_DIR.is_dir() and preset in presets
        else default_config()
    )

    st.subheader("Simulation")
    st.number_input(
        "Random seed",
        min_value=0,
        # Must comfortably exceed the configured default seeds, otherwise
        # Streamlit clamps the seeded value and silently falls back to the minimum.
        max_value=2_147_483_647,
        step=1,
        key="sim_seed",
        help="Everything is reproducible from this seed.",
    )
    st.slider("Reference rows (fit)", 200, 3000, step=100, key="sim_n_reference")
    st.slider("Stream rows (scored)", 200, 4000, step=100, key="sim_n_stream")
    st.slider("Injected anomaly events", 0, 60, step=1, key="sim_n_anomalies")
    with st.expander("Distribution shift", expanded=False):
        st.checkbox("Enable injected shift", key="sim_shift_enabled")
        st.slider(
            "Shift starts at (fraction of stream)", 0.2, 0.9, step=0.05, key="sim_shift_start"
        )
        st.slider("Latency x", 1.0, 1.6, step=0.02, key="sim_shift_latency")
        st.slider("CPU +", 0.0, 0.2, step=0.005, key="sim_shift_cpu")
        st.slider("Request rate x", 1.0, 1.4, step=0.02, key="sim_shift_rate")
        st.slider("Noise x", 1.0, 1.8, step=0.05, key="sim_shift_noise")
    with st.expander("Cycle / seasonality", expanded=False):
        st.slider("Usage cycle (observations)", 8, 240, step=4, key="sim_cycle")

    st.subheader("Detectors")
    st.slider("Rolling z threshold", 1.5, 6.0, step=0.1, key="det_z_threshold")
    st.slider("Rolling z window", 20, 400, step=10, key="det_z_window")
    st.slider(
        "Isolation Forest contamination",
        0.005,
        0.2,
        step=0.005,
        key="det_contamination",
        help="Prior on the anomaly rate. The simulator injects ~8%, so a small value under-detects.",
    )

    st.subheader("Drift detector")
    st.selectbox("Significance level (alpha)", [0.05, 0.02, 0.01, 0.005], key="drift_alpha")
    st.selectbox(
        "Multiple-comparison correction", ["holm", "bonferroni", "none"], key="drift_correction"
    )
    st.slider("Window size", 40, 600, step=20, key="drift_window")
    st.slider("Minimum window size", 20, 400, step=20, key="drift_min_window")
    st.slider("Effect-size floor (KS D)", 0.0, 0.5, step=0.01, key="drift_floor")

    st.divider()
    st.subheader("Live controls")
    appended = len(st.session_state["appends"])
    st.caption(f"Appended to the stream so far: **{appended}** operation(s)")
    # A drift test needs at least `min_window_size` observations, so the shift
    # control appends a whole window of shifted data instead of a token few rows.
    shift_rows = max(60, int(st.session_state.get("drift_window", 200)))
    if st.button("Append 30 normal events", width=WIDTH):
        next_append(n_new=30)
    if st.button("Inject anomaly burst now", width=WIDTH):
        next_append(n_new=30, anomaly_kinds=["latency_storm", "spike"], n_anomaly_events=1)
    if st.button(
        f"Inject distribution shift now ({shift_rows} rows)",
        width=WIDTH,
        help="Appends one full drift window drawn from the shifted regime, "
        "so the drift detector has enough observations to test.",
    ):
        next_append(n_new=shift_rows, shift=True)
    if st.button("Reset stream to generated data", width=WIDTH):
        st.session_state["appends"] = []
        st.rerun()
    if st.button("Reset acknowledgements", width=WIDTH):
        journal_store = AlertStore(Path("data/processed/alert_state.json"), persist_alerts=False)
        journal_store.clear_journal()
        journal_store.save()
        build_run.clear()
        st.rerun()
    st.caption(
        "Detectors are fit once, on the anomaly-free reference block; appended rows "
        "are only ever scored, never used for fitting."
    )

config = build_config()
signature = json.dumps(config.to_dict(), sort_keys=True)
appends_payload = json.dumps(st.session_state["appends"], sort_keys=True)
result = build_run(signature, appends_payload)

alert_path = Path(config.alert_state_path)
store = AlertStore(alert_path, persist_alerts=False)
store.add_many(result.alerts)
counts = store.counts()


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("StreamGuard — real-time anomaly & drift monitor")
st.warning(SIMULATED_BANNER, icon="⚠️")

metadata = result.dataset.metadata
drift_windows = result.drift_windows
drift_hits = int(drift_windows["any_drift"].sum()) if not drift_windows.empty else 0

columns = st.columns(5)
columns[0].metric(
    "Stream observations", f"{len(result.stream):,}", help="Simulated rows scored by the detectors"
)
columns[1].metric(
    "Labelled anomalies",
    f"{int(result.dataset.truth().sum())}",
    delta=f"{metadata['anomaly_rate']:.1%} of stream",
    delta_color="off",
    help="Injected point anomalies with ground-truth labels (simulated)",
)
columns[2].metric(
    "Alerts",
    f"{counts['total']:,}",
    delta=f"{counts['unacknowledged']} unacknowledged",
    delta_color="off",
)
columns[3].metric(
    "Critical unacknowledged", f"{counts['unacknowledged_critical']}", delta_color="off"
)
columns[4].metric(
    "Drift",
    "detected" if drift_hits else "not detected",
    delta=f"{drift_hits}/{len(drift_windows)} windows" if len(drift_windows) else None,
    delta_color="off",
)

tabs = st.tabs(["Overview", "Metrics & anomalies", "Drift", "Alerts", "Evaluation", "Provenance"])


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Detector status")
        st.caption(
            "Precision/recall/F1 are measured against the simulated ground-truth labels of the "
            "stream shown below them. `fit_rows` is the anomaly-free reference block used for fitting."
        )
        status = result.detector_status()
        st.dataframe(
            status[PRIMARY_METRICS_COLUMNS].round(3),
            width=WIDTH,
            hide_index=True,
            column_config={
                "flag_rate": st.column_config.NumberColumn("flag rate", format="%.3f"),
                "precision": st.column_config.NumberColumn(format="%.3f"),
                "recall": st.column_config.NumberColumn(format="%.3f"),
                "f1": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with right:
        st.subheader("Alerts by severity")
        st.plotly_chart(severity_bar(counts["by_severity"]), width=WIDTH, key="overview_severity")

    st.subheader("Simulation configuration in force")
    sim = config.simulator
    # Every value is rendered as text: an object column mixing str/int/bool makes
    # Streamlit's Arrow serialisation fail.
    st.dataframe(
        pd.DataFrame(
            [
                ("seed", sim.seed),
                ("reference rows", sim.n_reference),
                ("stream rows", len(result.stream)),
                ("injected events", metadata["n_anomaly_events"]),
                ("anomaly rate", f"{metadata['anomaly_rate']:.2%}"),
                ("shift enabled", sim.shift.enabled),
                ("shift starts", metadata["drift_start_timestamp"] or "-"),
                ("shift component", metadata["drift_component"] or "-"),
                ("runtime appends", metadata.get("appended_rows", 0)),
                ("data origin", SIMULATED_ORIGIN),
            ],
            columns=["setting", "value"],
        ).assign(value=lambda frame: frame["value"].astype(str)),
        width=WIDTH,
        hide_index=True,
    )

    st.subheader("Most recent alerts")
    if counts["total"]:
        recent = store.to_dataframe().head(8)
        st.dataframe(recent, width=WIDTH, hide_index=True)
    else:
        st.info("No alerts for this configuration.")


# --------------------------------------------------------------------------- #
# Metrics & anomalies
# --------------------------------------------------------------------------- #
with tabs[1]:
    detector_names = list(result.results)
    controls = st.columns([2, 2, 1])
    detector = controls[0].selectbox(
        "Detector", detector_names, index=detector_names.index("isolation_forest")
    )
    features = controls[1].multiselect(
        "Features",
        list(FEATURE_COLUMNS),
        default=list(FEATURE_COLUMNS),
        format_func=lambda name: FEATURE_LABELS.get(name, name),
    )
    show_truth = controls[2].toggle("Show ground truth", value=True)
    with st.expander("What am I looking at?"):
        st.markdown(
            "- **Lines** are simulated metric values.\n"
            "- **Red circles** mark the injected anomalies (ground truth), which only exist because "
            "this project is a simulation.\n"
            "- **Orange crosses** are the detector's alerts.\n"
            "- The **shaded band** is the injected distribution shift.\n"
            "- Alerts inside the shaded band are usually drift, not point anomalies: a detector fit on "
            "the reference block has never seen the shifted regime. See the *Drift* tab."
        )
    if features:
        st.plotly_chart(
            feature_timeseries(result.flagged, tuple(features), detector, show_truth=show_truth),
            width=WIDTH,
            key="feature_timeseries",
        )
    threshold = config.detector.zscore_threshold if detector == "rolling_zscore" else None
    st.plotly_chart(
        score_timeseries(result.flagged, detector, threshold),
        width=WIDTH,
        key="score_timeseries",
    )

    flagged_rows = result.flagged.loc[result.flagged[f"flag_{detector}"].astype(bool)]
    st.subheader(f"{len(flagged_rows)} alerted observations ({detector})")
    st.caption(
        "Each flagged observation is one alert. The Alerts tab groups them into episodes for triage."
    )
    detail_columns = [
        "timestamp",
        *FEATURE_COLUMNS,
        "is_anomaly_true",
        "anomaly_kind",
        "drift_active",
        f"severity_{detector}",
        f"reason_{detector}",
    ]
    st.dataframe(flagged_rows[detail_columns].head(300), width=WIDTH, hide_index=True)


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Which metrics drifted, and when?")
    st.info(
        "A drift alert means **the input distribution changed**. It is *not* evidence that model "
        "accuracy degraded: detecting degradation needs labelled outcomes, which a live stream does "
        "not provide. StreamGuard reports drift and detection quality as two separate signals.",
        icon="ℹ️",
    )
    drift_features = result.drift_features
    window_size = config.drift.window_size
    reference_size = len(result.reference)
    st.caption(
        f"Two-sample Kolmogorov-Smirnov per feature: each {window_size}-point window against the "
        f"{reference_size}-row reference block; alpha={config.drift.alpha:g}, "
        f"{config.drift.correction} correction, minimum window {config.drift.min_window_size} rows, "
        f"practical-significance floor KS D >= {config.drift.effect_size_floor:g} "
        f"(one test per feature per window)."
    )
    st.plotly_chart(drift_heatmap(drift_features), width=WIDTH, key="drift_heatmap")
    st.plotly_chart(
        drift_ks_lines(drift_features, config.drift.effect_size_floor),
        width=WIDTH,
        key="drift_ks_lines",
    )

    if not drift_windows.empty:
        st.subheader("Window-by-window verdict")
        st.dataframe(drift_windows, width=WIDTH, hide_index=True)
        evaluated = drift_windows.loc[drift_windows["evaluated"] & drift_windows["any_drift"]]
        if not evaluated.empty:
            chosen = st.selectbox("Inspect window", list(evaluated["window_id"]))
            report = result.drift_reports[int(chosen)]
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "feature": item.feature,
                            "KS D": round(item.ks_statistic, 4),
                            "p value": f"{item.p_value:.3g}",
                            "p adjusted": f"{item.p_adjusted:.3g}",
                            "PSI": round(item.psi, 4),
                            "reference mean": round(item.reference_mean, 3),
                            "window mean": round(item.window_mean, 3),
                            "mean shift %": round(item.mean_shift_pct, 2),
                            "drifted": item.drifted,
                        }
                        for item in report.features
                    ]
                ),
                width=WIDTH,
                hide_index=True,
            )
            st.caption(
                f"Window {int(chosen)} spans {report.start_timestamp} to {report.end_timestamp}."
            )
    else:
        st.warning(
            "No drift windows were evaluated: the stream is shorter than the minimum window size. "
            "Increase the stream length or lower the minimum window size."
        )


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Alert table")
    st.caption(
        "Alert ids are content-addressed: regenerating the same simulation reproduces the same ids, "
        f"so acknowledgements survive a restart. Journal file: `{alert_path}`."
    )
    filters = st.columns(4)
    severities = filters[0].multiselect(
        "Severity",
        ["critical", "high", "medium", "low"],
        default=["critical", "high", "medium", "low"],
    )
    types = filters[1].multiselect(
        "Type",
        [ALERT_TYPE_ANOMALY, ALERT_TYPE_DRIFT],
        default=[ALERT_TYPE_ANOMALY, ALERT_TYPE_DRIFT],
    )
    detectors_filter = filters[2].multiselect(
        "Detector", list(result.results), default=list(result.results)
    )
    only_unacked = filters[3].toggle("Only unacknowledged", value=False)

    visible = [
        alert
        for alert in store.all()
        if alert.severity in severities
        and alert.alert_type in types
        and (alert.detector in detectors_filter or alert.alert_type == ALERT_TYPE_DRIFT)
        and (not only_unacked or not alert.acknowledged)
    ]
    table = (
        pd.DataFrame([alert.to_dict() for alert in visible])
        if visible
        else pd.DataFrame(
            columns=[
                "alert_id",
                "timestamp",
                "alert_type",
                "severity",
                "feature",
                "detector",
                "score",
                "reason",
            ]
        )
    )
    shown_columns = [
        "alert_id",
        "timestamp",
        "alert_type",
        "severity",
        "feature",
        "detector",
        "score",
        "acknowledged",
        "reason",
    ]
    # Row selection on the table itself: explicit, and impossible to trigger
    # "select everything" by accident the way a many-option multiselect can.
    event = st.dataframe(
        table.loc[:, [c for c in shown_columns if c in table.columns]],
        width=WIDTH,
        hide_index=True,
        key="alert_table",
        on_select="rerun",
        selection_mode="multi-row",
    )
    selected_ids = [str(table.iloc[position]["alert_id"]) for position in event.selection.rows]

    if visible:
        shown_ids = [str(alert.alert_id) for alert in visible]
        unacknowledged_shown = [str(alert.alert_id) for alert in visible if not alert.acknowledged]
        st.caption(
            f"{len(selected_ids)} row(s) selected · {len(shown_ids)} alerts shown · "
            f"{len(unacknowledged_shown)} of them unacknowledged."
        )
        controls = st.columns([1, 1, 1, 1])
        batch = controls[0].number_input(
            "Batch size",
            min_value=1,
            max_value=200,
            value=1,
            step=1,
            key="ack_batch",
            help="Used by 'Acknowledge newest N', newest first.",
        )
        if controls[1].button("Acknowledge newest N", width=WIDTH):
            for alert_id in unacknowledged_shown[: int(batch)]:
                store.acknowledge(alert_id, by="dashboard-operator")
            store.save()
            st.rerun()
        if controls[2].button(
            "Acknowledge all shown", width=WIDTH, disabled=not unacknowledged_shown
        ):
            for alert_id in unacknowledged_shown:
                store.acknowledge(alert_id, by="dashboard-operator")
            store.save()
            st.rerun()
        if controls[3].button("Unacknowledge selected", width=WIDTH, disabled=not selected_ids):
            for alert_id in selected_ids:
                store.unacknowledge(alert_id)
            store.save()
            st.rerun()
        if selected_ids and st.button("Acknowledge selected rows", width=WIDTH):
            for alert_id in selected_ids:
                store.acknowledge(alert_id, by="dashboard-operator")
            store.save()
            st.rerun()
        st.download_button(
            "Download alert table (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="streamguard_alerts.csv",
            mime="text/csv",
            width=WIDTH,
        )
        st.caption(
            "Acknowledgement state is written to the journal file and re-applied to the same "
            "content-addressed alert ids on the next run, so regenerating a simulation does not "
            "un-acknowledge alerts an operator has already triaged."
        )

    st.subheader("Episodes (consecutive alerts collapsed for triage)")
    episodes = alert_episodes(visible)
    if episodes.empty:
        st.info("No anomaly episodes in the current filter.")
    else:
        st.dataframe(
            episodes.drop(columns=["alert_ids"]).head(200),
            width=WIDTH,
            hide_index=True,
        )
        st.caption(f"{len(visible)} alerts collapsed into {len(episodes)} episodes.")

    st.subheader("Severity split")
    st.plotly_chart(severity_bar(counts["by_severity"]), width=WIDTH, key="alerts_severity")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.info(
        "Evaluation uses a **separate labelled simulated dataset**: the detector is fit on the "
        "anomaly-free reference block and scored on the stream block whose injected anomalies are "
        "known. Injected distribution shift is not labelled as a point anomaly, which is why the "
        "per-segment table below matters.",
        icon="🎯",
    )
    st.subheader("Detection quality by detector")
    st.dataframe(
        result.metrics[
            [
                "detector",
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "roc_auc",
                "average_precision",
            ]
        ].round(3),
        width=WIDTH,
        hide_index=True,
    )
    st.plotly_chart(metrics_bar(result.metrics), width=WIDTH, key="metrics_bar")

    primary = st.selectbox(
        "Confusion matrix for",
        list(result.results),
        index=list(result.results).index(result.report["primary_detector"]),
    )
    left, right = st.columns([2, 3])
    primary_metrics = evaluate_detector(
        primary,
        result.dataset.truth(),
        result.results[primary]["is_anomaly"].to_numpy(dtype=bool),
        result.results[primary]["score"].to_numpy(dtype=float),
    )
    with left:
        st.plotly_chart(
            confusion_heatmap(confusion_matrix_frame(primary_metrics)),
            width=WIDTH,
            key="confusion_heatmap",
        )
    with right:
        st.dataframe(metrics_frame(primary_metrics), width=WIDTH, hide_index=True)

    st.subheader("Metrics per segment")
    st.caption(
        "The same detectors measured on the pre-shift part of the stream and on the injected-shift "
        "period. Detectors fit on the reference block lose precision after the shift because the "
        "shifted regime looks anomalous to them; the rolling z-score adapts and keeps its precision. "
        "This is a property of the detectors, not of the labels."
    )
    if not result.segments.empty:
        st.dataframe(
            result.segments[
                ["segment", "detector", "tp", "fp", "fn", "precision", "recall", "f1"]
            ].round(3),
            width=WIDTH,
            hide_index=True,
        )

    st.subheader("Event-level detection latency")
    st.caption(
        "An injected event counts as detected if the detector flags at least one observation inside "
        "it. `latency` is the number of observations between the event start and the first hit."
    )
    latency_summary_frame = latency_summary(detection_latency(result.events, result.results))
    st.dataframe(latency_summary_frame.round(3), width=WIDTH, hide_index=True)

    st.subheader("Achievable operating points (diagnostic)")
    st.warning(
        "These thresholds are chosen **using the labels**, so the best row per detector is an upper "
        "bound for each score function, not a deployable procedure. It is shown to separate a weak "
        "score from a mis-set threshold: the isolation forest's default operating point is a "
        "contamination prior, and this table shows what its ranking can deliver.",
        icon="🧪",
    )
    sweep = oracle_threshold_sweep(result.results, result.dataset.truth())
    if not sweep.empty:
        best = sweep.sort_values("f1", ascending=False).groupby("detector", as_index=False).first()
        st.dataframe(
            best[
                [
                    "detector",
                    "threshold",
                    "flag_rate",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                ]
            ].round(3),
            width=WIDTH,
            hide_index=True,
        )
        with st.expander("Full sweep (all thresholds)"):
            st.dataframe(sweep.round(4), width=WIDTH, hide_index=True)

    downloads = st.columns(3)
    downloads[0].download_button(
        "Download evaluation report (JSON)",
        data=json.dumps(result.report, indent=2, default=str).encode("utf-8"),
        file_name="streamguard_evaluation.json",
        mime="application/json",
        width=WIDTH,
    )
    downloads[1].download_button(
        "Download metrics by detector (CSV)",
        data=result.metrics.to_csv(index=False).encode("utf-8"),
        file_name="metrics_by_detector.csv",
        mime="text/csv",
        width=WIDTH,
    )
    downloads[2].download_button(
        "Download labelled stream (CSV)",
        data=result.flagged.to_csv(index=False).encode("utf-8"),
        file_name="stream_labelled.csv",
        mime="text/csv",
        width=WIDTH,
    )


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Data provenance")
    st.markdown(
        f"""
**Origin: simulated.** Every observation is generated by `streamguard.simulator` from the seed
`{metadata["seed"]}` and the configuration below. There is no ingestion path, no database, no
network call and no real telemetry anywhere in StreamGuard.

How to tell simulated rows apart:

| Marker | Meaning |
| --- | --- |
| `data_origin = "simulated"` | column on every generated row |
| `is_reference` | `True` for the anomaly-free reference block used to fit detectors |
| `is_anomaly` / `anomaly_id` / `anomaly_kind` | injected point anomalies (simulation ground truth) |
| `drift_active` / `drift_component` | rows drawn from the injected shifted regime |

**What the numbers mean.** Ground-truth labels exist only because the anomalies were injected.
Precision, recall and F1 therefore measure *this* generator, not real detection performance.
Detectors were fit on the reference block only; stream labels are never used for fitting.
"""
    )
    st.subheader("Configuration in force")
    st.json(config.to_dict())
    st.subheader("Run metadata")
    st.json({key: value for key, value in metadata.items() if key != "anomaly_events"})
    st.caption(
        "Feature units: "
        + ", ".join(f"{FEATURE_LABELS[name]} [{FEATURE_UNITS[name]}]" for name in FEATURE_COLUMNS)
    )
    st.caption("See MODEL_CARD.md and LIMITATIONS.md at the repository root for scope and caveats.")
