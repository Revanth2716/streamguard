# StreamGuard — Real-Time Anomaly & Drift Monitor

A local-first monitoring application that **simulates** streaming service metrics,
detects point anomalies, detects distribution drift, evaluates detection quality
against injected ground truth, and presents everything in a Streamlit dashboard.

> **All data in this project is simulated.** A seeded generator fabricates every
> observation, injects known anomalies and a known distribution shift, and labels
> them. There is no ingestion path, no database, no network call and no real
> operational data anywhere in StreamGuard. Every metric reported below measures
> *this generator*, not real-world detection performance. See
> [MODEL_CARD.md](MODEL_CARD.md) and [LIMITATIONS.md](LIMITATIONS.md).

---

## Quick start

Python 3.12. The virtual environment, package cache and generated files live on
`D:` (this checkout is on `D:`); nothing is written outside the project.

```bash
# 1. Environment (already present in this checkout as .venv)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"     # Windows
# source .venv/bin/activate && pip install -e ".[dev]"    # macOS / Linux

# 2. Headless run: simulate -> detect -> drift -> alert -> evaluate, write artifacts
./.venv/Scripts/python.exe -m streamguard.cli --config configs/default.toml

# 3. Dashboard
./.venv/Scripts/python.exe -m streamlit run dashboard/app.py

# 4. Verification
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m ruff format --check .
```

Optional analysis helpers (not part of the package or the test suite):

```bash
./.venv/Scripts/python.exe reports/sensitivity_study.py        # knob sensitivity, writes JSON
./.venv/Scripts/python.exe reports/capture_screenshots.py      # dashboard screenshots (needs Chrome)
```

## Measured results (seed 20261016, `configs/default.toml`)

Reference block: 1200 anomaly-free observations (used for fitting).
Stream block: 1400 observations, 24 injected anomaly events (114 rows, 8.14%),
injected distribution shift starting at 2026-01-02 10:00:00 for the last 560 rows.
Reproduce with `python -m streamguard.cli --config configs/default.toml`;
full detail in [reports/metrics/evaluation.json](reports/metrics/evaluation.json).

### Detection quality on the labelled stream

| detector | flags | flag rate | TP | FP | FN | precision | recall | F1 | ROC-AUC | avg precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `static_percentile` (naive baseline) | 178 | 0.127 | 112 | 66 | 2 | 0.629 | 0.982 | **0.767** | 0.990 | 0.973 |
| `rolling_zscore` (statistical baseline) | 62 | 0.044 | 59 | 3 | 55 | 0.952 | 0.518 | **0.670** | 0.972 | 0.866 |
| `isolation_forest` (ML detector) | 118 | 0.084 | 53 | 65 | 61 | 0.449 | 0.465 | **0.457** | 0.940 | 0.409 |
| `dummy` (never alerts) | 0 | 0.000 | 0 | 0 | 114 | 0.000 | 0.000 | 0.000 | 0.500 | 0.081 |

Confusion matrix for `isolation_forest`: TN 1221, FP 65, FN 61, TP 53.

### Per-segment F1 (the drift penalty, measured rather than averaged away)

| detector | pre-drift segment | injected-shift segment |
| --- | --- | --- |
| `static_percentile` | 0.860 | 0.702 |
| `rolling_zscore` | 0.813 | 0.518 |
| `isolation_forest` | 0.395 | 0.493 |

### Event-level detection

| detector | events | detected | detection rate | median latency (observations) |
| --- | --- | --- | --- | --- |
| `static_percentile` | 24 | 24 | 1.000 | 0 |
| `rolling_zscore` | 24 | 17 | 0.708 | 0 |
| `isolation_forest` | 24 | 16 | 0.667 | 0 |
| `dummy` | 24 | 0 | 0.000 | — |

### Drift scan (KS, Holm-corrected, 200-observation windows)

| window | position | verdict | largest KS D | latency mean shift |
| --- | --- | --- | --- | --- |
| 0–3 | 0–800 | no drift | 0.061 / 0.100 / 0.124 / 0.104 | — |
| 4 | 800 | drift | 0.418 (p_adj 8.6e-27) | +18.5% |
| 5 | 1000 | drift | 0.534 (p_adj 4.2e-45) | +30.0% |
| 6 | 1200 | drift | 0.532 (p_adj 1.2e-44) | +21.6% |

Three of seven windows alert, exactly the windows drawn from the shifted regime.
Alerts: 367 total (105 critical, 124 high, 53 medium, 85 low; 9 of them drift,
the rest per-observation anomaly alerts).

### Threshold sensitivity (same data, only the knob changed)

`contamination` for the isolation forest, against a true anomaly rate of 8.14%:

| contamination | flags | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| 0.02 (default) | 118 | 0.449 | 0.465 | 0.457 |
| 0.05 | 255 | 0.412 | 0.921 | **0.569** |
| 0.08 | 321 | 0.355 | 1.000 | 0.524 |
| 0.12 | 403 | 0.283 | 1.000 | 0.441 |

The prior matters more than the model: at the default 2% the forest under-detects
an 8% anomaly rate. The default is left at a conservative 2% and the measurement
is reported instead of silently tuning the prior on the test data. An
"oracle" sweep (thresholds chosen *with* labels, a diagnostic upper bound only,
**not** a deployable procedure) shows the best achievable F1 per score function:
`static_percentile` 0.938, `rolling_zscore` 0.800, `isolation_forest` 0.571.

Autocorrelation and the drift detector, on clean windows with the effect-size
floor disabled: false drift alerts 0 of 4 at AR(1) φ = 0.00–0.35, 1 of 4 at
φ = 0.50 and φ = 0.72 (largest clean-window KS statistic 0.093 → 0.195). Raw
numbers: [reports/metrics/sensitivity.json](reports/metrics/sensitivity.json).

## How it works

```
configs/*.toml
      │
      ▼
simulator.py ──► reference block (anomaly-free)  ──► detectors fit HERE ONLY
      │        └─► stream block (anomalies + shift, labelled)
      │                     │
      │                     ├─► detector.py  (rolling_zscore, isolation_forest,
      │                     │                 static_percentile, dummy)
      │                     ├─► drift.py     (KS per feature per window,
      │                     │                 Holm/Bonferroni correction,
      │                     │                 effect-size floor, PSI as evidence)
      │                     ▼
      │              evaluation.py ──► precision/recall/F1, confusion matrix,
      │                                per-segment table, event latency,
      │                                oracle threshold sweep
      ▼
alerts.py ──► structured alerts with deterministic ids + acknowledgement journal
      ▼
dashboard/app.py (Streamlit + Plotly)  +  reports/ artifacts
```

| Path | Purpose |
| --- | --- |
| [src/streamguard/simulator.py](src/streamguard/simulator.py) | Seeded, labelled event stream (log-normal latency/throughput, correlated latent load, injected spikes/bursts/dropouts/latency storms, injected distribution shift) |
| [src/streamguard/detector.py](src/streamguard/detector.py) | Causal rolling z-score, Isolation Forest, static-percentile and dummy baselines |
| [src/streamguard/drift.py](src/streamguard/drift.py) | Two-sample KS per feature with multiple-comparison correction and practical-significance floor |
| [src/streamguard/evaluation.py](src/streamguard/evaluation.py) | Metrics, confusion matrix, segments, detection latency, threshold sweep, JSON report |
| [src/streamguard/alerts.py](src/streamguard/alerts.py) | Alert records, deterministic ids, dedup, episode grouping, persisted acknowledgements |
| [src/streamguard/pipeline.py](src/streamguard/pipeline.py) | One entry point used by both the CLI and the dashboard |
| [src/streamguard/config.py](src/streamguard/config.py) | Validated configuration dataclasses (TOML) |
| [src/streamguard/figures.py](src/streamguard/figures.py) | Plotly figures shared by the dashboard and the offline report |
| [dashboard/app.py](dashboard/app.py) | Streamlit dashboard (overview, metrics, drift, alerts, evaluation, provenance) |
| [tests/](tests) | 86 tests, including explicit leakage and dashboard-contract tests |

### Leakage control

* Detectors are fit on the **reference block only**, which contains no anomalies.
* `AnomalyDetector.fit()` accepts a feature frame; labels are never passed in.
* `tests/test_no_leakage.py` flips every stream label and asserts the scores and
  flags are unchanged, adds label columns to the reference block and asserts the
  fit is unchanged, and checks the stored reference statistics come from the
  reference block rather than the stream.
* The threshold sweep is named `oracle_threshold_sweep` and documented as a
  diagnostic upper bound because it selects thresholds with labels.

## Configuration

`configs/default.toml` (calibrated defaults) and `configs/sensitive.toml` (a
deliberately twitchier operating point: lower z threshold, higher contamination,
smaller drift window, lower effect-size floor). Every run records the effective
configuration and seed in `reports/metrics/run_metadata.json` and inside
`reports/metrics/evaluation.json`.

## Dashboard

Six tabs: **Overview**, **Metrics & anomalies**, **Drift**, **Alerts**,
**Evaluation**, **Provenance**. Sidebar controls regenerate the simulation, tune
both detectors and the drift test, and append live events at runtime
("Append 30 normal events", "Inject anomaly burst now", "Inject distribution shift
now", "Reset stream to generated data", "Reset acknowledgements"). Detectors are
fit once on the reference block; appended rows are only ever scored.

Screenshots of the running dashboard: [reports/figures/](reports/figures)
(`dashboard_overview.png`, `dashboard_metrics_anomalies.png`, `dashboard_drift.png`,
`dashboard_alerts.png`, `dashboard_evaluation.png`, `dashboard_provenance.png`).

## Generated artifacts

| Path | Contents |
| --- | --- |
| `reports/metrics/evaluation.json` | Full evaluation report: metrics, confusion matrix, segments, latency, sweep, drift summary, seed and config |
| `reports/metrics/metrics_by_detector.csv`, `latency`, `segments`, `drift_windows` | The same tables as CSVs |
| `reports/metrics/run_metadata.json` | Effective configuration + dataset metadata for the run |
| `reports/metrics/sensitivity.json` | Contamination, effect-size-floor and autocorrelation sweeps |
| `reports/figures/*.html` | Plotly figures (time series with alerts, score, drift heatmap, KS trajectories, metrics bar, confusion matrix) |
| `data/processed/stream_labelled.csv` | The labelled simulated stream (`data_origin = simulated`) |
| `data/processed/alert_state.json` | Acknowledgement journal (no alert rows are persisted by the dashboard) |

## Scope and non-goals

No Kafka, no message broker, no database server, no Kubernetes, no
authentication, no Docker, no cloud services, no paid APIs and no external AI
service. Single process, local files, standard library plus NumPy/pandas/SciPy/
scikit-learn/Streamlit/Plotly. The known weaknesses are listed in
[LIMITATIONS.md](LIMITATIONS.md) rather than hidden.
