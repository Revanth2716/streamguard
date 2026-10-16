# Model Card — StreamGuard detectors

This card covers the anomaly detectors shipped in `src/streamguard/detector.py`
and the drift test in `src/streamguard/drift.py`. It follows the spirit of a
standard model card: what the system is, what it was evaluated on, how it
behaves, and what it must not be used for.

**Read this first:** every number here was produced on synthetic data generated
by this repository. None of it is evidence about how these detectors behave on a
real service.

---

## 1. Overview

| | |
| --- | --- |
| Project | StreamGuard — real-time anomaly & drift monitor |
| Version | 0.1.0 |
| Task | (a) flag anomalous observations in a metric stream; (b) flag changes in the input distribution of that stream |
| Detectors | `rolling_zscore` (statistical baseline), `isolation_forest` (ML), `static_percentile` (naive baseline), `dummy` (floor) |
| Drift test | two-sample Kolmogorov–Smirnov per feature, Holm-corrected, plus a practical-significance floor; Population Stability Index reported as supporting evidence |
| Outputs | anomaly score, alert flag, severity, human-readable reason, affected feature; drift statistic, corrected p-value, PSI, mean shift |
| Data | simulated only (seeded generator, `data_origin = "simulated"`) |
| Runtime | Python 3.12, NumPy / pandas / SciPy / scikit-learn / Streamlit / Plotly, local files only |
| Owner | single local operator (this project has no multi-tenant story) |

## 2. Intended use

**Intended.** Demonstration, teaching and local experimentation with streaming
anomaly detection and drift monitoring: how a simple statistical baseline
compares with an Isolation Forest, how a contamination prior changes the
operating point, how distribution shift degrades a fixed-reference detector, and
how to evaluate all of it honestly on labelled synthetic data.

**Out of scope.** Anything requiring real-world guarantees: production alerting
on live infrastructure, safety-critical decisions, capacity or incident
automation, per-customer or regulated data processing, and any claim of accuracy
on a system this project has never seen. Do not ingest real telemetry and treat
the resulting alerts as validated.

## 3. Data and training

| | |
| --- | --- |
| Generator | `streamguard.simulator`, fully deterministic from `seed` |
| Features | `latency_ms` (ms), `cpu_util` (fraction), `mem_util` (fraction), `request_rate` (req/s) |
| Reference block | 1200 anomaly-free observations — the only data used for fitting |
| Stream block | 1400 observations scored: 24 injected anomaly events (114 rows, 8.14%) plus an injected distribution shift for the last 560 rows |
| Anomaly kinds | `spike`, `burst`, `dropout` (throughput collapse + latency spike), `latency_storm` |
| Anomaly magnitude | 2.6–9.0 reference-sigma, sampled per event; multi-row events last 1–8 observations |
| Latent structure | shared AR(1) load (φ = 0.35) drives all four features, so the metrics are correlated; latency and throughput are log-normal, utilisation is additive and clipped to 0–100% |
| Seasonality | a 48-observation recurring cycle with small amplitude, deliberately shorter than the drift window |
| Shift component | latency ×1.18, CPU +0.055, memory +0.018, request rate ×1.10, noise ×1.15 |

The shift is a *step* change of moderate size. A larger step makes the
fixed-reference detectors flag almost the whole shifted period, inflating their
false positives without testing anything new; the measured trade-off is in
`reports/metrics/sensitivity.json`.

### What the model is fit on

`fit(reference)` receives the feature columns of the reference block. Ground
truth labels exist only on the stream block and are never passed to a fit.
`tests/test_no_leakage.py` enforces this by tampering with labels and asserting
the scores are unchanged.

## 4. Detectors

| Detector | Fitting | Score | Flag rule | Severity |
| --- | --- | --- | --- | --- |
| `rolling_zscore` | reference median + robust (MAD) scale, used as fallback and scale floor | causal rolling z from the previous `zscore_window` points, current point excluded, `max` over features | `|z| ≥ zscore_threshold` (default 3.5) | ratio of score to threshold (≥1.2 medium, ≥1.5 high, ≥2.0 critical) |
| `isolation_forest` | `StandardScaler` + `IsolationForest` on the reference features | `-score_samples`, standardised against the reference score distribution | the forest's own `predict` (contamination prior, default 0.02) | reference-sigma: ≥2 medium, ≥3 high, ≥4 critical |
| `static_percentile` | per-feature 99.5% two-sided envelope from the reference | normalised distance beyond the envelope | outside the envelope | reference-sigma beyond the envelope: ≥1 medium, ≥2 high, ≥3 critical |
| `dummy` | nothing | 0 | never | — |

`reason` strings carry the numbers that drove the decision. For the isolation
forest the reason lists the largest standardised deviations from the reference
median: this is post-hoc attribution of a flagged point, **not** a causal
explanation and not SHAP.

## 5. Evaluation

Protocol: fit on the reference block, score the stream block, compare flags
against the injected labels. Point anomalies are labelled; the injected shift is
**not**, which is why the per-segment table exists — flags inside the shifted
period count as false positives for detectors that cannot adapt.

Seed 20261016, `configs/default.toml` (`reports/metrics/evaluation.json`):

| detector | precision | recall | F1 | ROC-AUC | average precision | F1 pre-drift | F1 shifted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `static_percentile` | 0.629 | 0.982 | 0.767 | 0.990 | 0.973 | 0.860 | 0.702 |
| `rolling_zscore` | 0.952 | 0.518 | 0.670 | 0.972 | 0.866 | 0.813 | 0.518 |
| `isolation_forest` | 0.449 | 0.465 | 0.457 | 0.940 | 0.409 | 0.395 | 0.493 |
| `dummy` | 0.000 | 0.000 | 0.000 | 0.500 | 0.081 | 0.000 | 0.000 |

Event-level detection (an event counts as detected if any of its observations is
flagged): `static_percentile` 24/24, `rolling_zscore` 17/24, `isolation_forest`
16/24, all with median latency 0 observations.

### What the numbers say

1. **A naive baseline is hard to beat on this data.** The static percentile rule
   wins on F1 (0.767) because injected anomalies are few, large and the
   percentile envelope rarely fires in the pre-drift segment. A weak result for
   the ML detector, reported as measured.
2. **The Isolation Forest's operating point dominates its score quality.** Its
   ROC-AUC is 0.940 — the ranking is informative — but at `contamination = 0.02`
   against an 8.14% anomaly rate it flags 118 rows and misses 61. Raising the
   prior to 0.05 lifts F1 to 0.569 (recall 0.921) at lower precision 0.412.
   The oracle ceiling on this data is F1 0.571, so the loss is not just the
   threshold: this score function is genuinely weaker here than the simple rule.
3. **Fixed-reference detectors pay for drift.** 49 of the static-percentile's 66
   false positives fall in the shifted period; the rolling z-score, which adapts,
   has 1 there (out of 3 overall). That is the intended contrast, and it is why precision and recall must
   be read per segment.
4. **Drift is detected unambiguously when it is real.** Three of seven windows
   alert (KS D 0.418–0.534, Holm-adjusted p from 8.6e-27 to 1.2e-44, latency mean
   shift +18.5% to +30.0%); the four clean windows stay silent (largest KS
   statistic 0.124).
5. **No free lunch on thresholds.** All figures move materially with the
   contamination prior and with the drift effect-size floor; the sweeps in
   `reports/metrics/sensitivity.json` show exactly how.

### Confidence and reliability

* Deterministic: the same seed and config reproduce the same frames, scores,
  alert ids and metrics (asserted in `tests/test_pipeline.py`).
* Four anomaly kinds and one shift component are exercised; other failure modes
  (network partitions, clock skew, metric restatements, missing data, correlated
  outages) are **not** simulated at all.
* No missing values, no schema changes, no late-arriving data, no duplicate
  timestamps: the stream is well-formed by construction. Real streams are not.

## 6. Drift test details

* One two-sample KS test per feature per window (4 features × windows).
* p-values corrected for the family of comparisons (Holm by default;
  Bonferroni and none available).
* `min_window_size` (default 120) prevents testing windows too small for KS to
  mean anything; such windows return `evaluated=False` with a reason.
* `effect_size_floor` (default 0.20) requires a minimum KS statistic on top of
  significance. Measured margins: clean-window maximum 0.124, injected shift
  0.418–0.534.
* PSI is computed and displayed but does **not** influence the decision.
* KS assumes independent samples; autocorrelated telemetry breaks that
  assumption and the test over-rejects (measured: false drift alerts on 1 of 4
  clean windows at AR(1) φ = 0.50 and 0.72 with the floor disabled, versus 0 of 4
  at φ ≤ 0.35). See [LIMITATIONS.md](LIMITATIONS.md).

## 7. Ethical, safety and reporting notes

* No personal data, no credentials, no network egress; the simulator fabricates
  numbers and the acknowledgement journal stores only alert ids and timestamps.
* Alerts are labelled `data_origin = simulated`, and the dashboard repeats the
  simulation warning in the page header and in the provenance tab.
* Reported metrics are measured by the code in this repository and are stored
  with the seed and configuration that produced them. Nothing is hardcoded.
* The oracle threshold sweep is explicitly named and labelled as a diagnostic
  upper bound because it uses labels; presenting it as a deployable result would
  be leakage.
* If you reuse this project's structure for real monitoring, re-derive every
  threshold on your own data and validate against labelled incidents before
  trusting an alert.
