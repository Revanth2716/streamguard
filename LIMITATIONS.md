# Limitations

Everything below is a known, measured or structural weakness of StreamGuard. It is
deliberately explicit: the honest output of a monitoring project is a list of
things it cannot do, next to the things it can.

## 0. The one that matters most

**All data is simulated, so all metrics are statements about the simulator.**
StreamGuard fabricates telemetry with a seeded generator, injects anomalies, and
then measures how well the detectors find the anomalies it just placed. A
detector "achieving F1 0.767" means it found 76%-weighted agreement with *this
generator's* idea of an anomaly. It says nothing about a real service. The
simulated data does not contain real failure modes: cascading dependencies,
deploy-induced regressions, clock skew, metric restatements, missing or delayed
data, schema drift, or correlated outages across services.

## 1. Labelling conventions

* Only *injected point anomalies* set `is_anomaly = True`. A period of injected
  distribution shift is **not** labelled as an anomaly.
* Consequence: any observation inside the shifted period that a
  fixed-reference detector flags counts as a false positive. This is why
  `isolation_forest` and `static_percentile` show heavy false positives overall
  (65 and 66) while the adapting `rolling_zscore` shows 3, and why the
  per-segment table (`reports/metrics/metrics_by_segment.csv`) is the fairer
  comparison.
* The convention is a choice, not a truth. A deployment might prefer to treat
  drift-period flags as "expected given drift" and exclude them, which would
  raise every fixed-reference detector's precision substantially.

## 2. Detector limitations

* **Rolling z-score.** Causal and cheap, but it adapts to whatever happened in the
  last `zscore_window` points. A slow ramp will be absorbed and never flagged; a
  single extreme point inflates the window's standard deviation and masks the
  next anomaly. It is univariate in practice (the flag is the maximum over
  features), so it cannot see an anomaly that is only multivariate.
* **Isolation Forest.** `contamination` sets the decision offset, so the number
  of flags on reference-like data is that prior *by construction* — the model
  does not estimate the anomaly rate. Measured against this stream's true rate of
  8.14%: 0.02 → F1 0.457, 0.05 → 0.569, 0.08 → 0.524, 0.12 → 0.441. The forest is
  also fit once and never updated, so its behaviour changes with the regime: on
  this run its F1 is 0.395 on the pre-drift segment and 0.493 on the shifted
  segment, because points from a dense shifted regime are *easier* to isolate.
  A detector that treats a whole shifted period as anomalous can therefore score
  better on exactly the segment where it is least trustworthy — the segment table
  must always be read together with the drift results.
* **Static percentile.** No adaptivity at all, and with four features at a 99.5%
  envelope it has an expected in-distribution false positive rate of roughly 2%
  per observation, which is already visible in the numbers.
* **Attribution.** `reason` strings list the largest standardised deviations of
  the flagged point. This is descriptive, post-hoc and not causal; it is not
  SHAP and it does not tell you which feature *caused* the alert.
* **No seasonality handling.** The simulator's recurring cycle is deliberately
  short (48 observations) relative to the detectors' windows. On real data with
  daily or weekly cycles, a 120-point rolling window may be shorter than the
  cycle, and the z-score baseline will chase the seasonality instead of the
  anomaly.

## 3. Drift test limitations

* **Statistically significant is not operationally important.** With thousands of
  reference observations KS detects differences far too small to care about.
  The `effect_size_floor` is a blunt mitigation, not a solution.
* **Autocorrelation invalidates the KS null distribution.** KS assumes
  independent samples; telemetry is usually autocorrelated, which shrinks the
  effective sample size and makes the test over-reject. Measured on clean windows
  with the effect-size floor disabled: 0 of 4 windows falsely alerted at AR(1)
  φ ≤ 0.35, 1 of 4 at φ = 0.50 and φ = 0.72 (largest clean-window KS statistic
  0.093 at φ = 0 → 0.195 at φ = 0.72). The default simulator coefficient is a
  deliberate 0.35 and the default floor adds margin, but a real stream with
  stronger persistence needs a corrected test (block bootstrap, or a
  variance-corrected statistic) rather than a lower alpha.
* **Phase and seasonality confound window comparisons.** A short window samples a
  different seasonal phase than the reference block. The simulator keeps that
  effect small on purpose; a real deployment must detrend or compare windows with
  matched time-of-day.
* **Multiple comparisons are corrected, but windows are not independent.**
  Overlapping or adjacent windows share data, so the family-wise error rate is not
  exactly what Holm promises. `step = window_size` limits overlap but does not
  remove dependence.
* **Minimum window size is a blunt guard.** 120 observations is a reasonable
  default for a 60-second interval, not a universal constant.
* **Drift ≠ degradation.** A detected shift is a statement about the input
  distribution. It is not evidence that model accuracy has fallen, and this
  project cannot make that claim: measuring accuracy requires labelled outcomes,
  which a live stream does not have. StreamGuard reports drift and detection
  quality separately and never merges them into one score.
* **Single-feature tests.** Multivariate drift (a change in *correlation*
  structure with unchanged marginals) is invisible to per-feature KS. PSI is shown
  for context only and does not fix this.

## 4. Evaluation limitations

* **One seed, one configuration.** The reported metrics come from seed 20261016
  and `configs/default.toml`. There is no repeated-seed confidence interval, so
  small differences between detectors should not be over-interpreted.
* **Anomaly rate and event lengths are fixed by the generator.** Detection
  quality depends strongly on how large and how long the injected anomalies are;
  the sweep of magnitudes (2.6–9.0 sigma) is uniform, not tuned to be
  representative.
* **The oracle threshold sweep uses labels.** It is a diagnostic upper bound and
  is labelled as such everywhere it appears. Choosing an operating point that way
  in production is label leakage.
* **Detection latency is measured only inside events.** An event detected one
  observation late counts as missed. No tolerance window is applied by default.
* **Alert counts are per observation.** A single 8-observation anomaly produces
  up to 8 alerts per detector. Episode grouping exists for triage, but the raw
  counts (367 alerts = 358 per-observation detector flags plus 9 drift-window
  alerts) are not comparable to an
  event-level alerting system that suppresses duplicates.

## 5. Engineering and operational gaps

* No persistence beyond JSON files: no database, no retention policy, no
  concurrent-writer protection beyond an atomic file replace and an in-process
  lock. Two operators writing the acknowledgement journal at once will race.
* No authentication, no authorisation, no audit trail of *who* acknowledged what
  beyond the free-text `acknowledged_by` field.
* No backfill, no late data, no out-of-order handling, and no handling of missing
  observations — the simulator always emits a complete, evenly spaced series.
* The dashboard reruns the entire pipeline when settings change; fine for 1400
  observations, not a streaming architecture. The "live" controls append in
  batches on demand rather than consuming a real feed.
* Ingestion is simulated in-process, so throughput, back-pressure, and
  exactly-once/late-arrival semantics are entirely untested.
* Figures are written as Plotly HTML (browser-rendered) because PNG export would
  add a Chromium/kaleido dependency. Dashboard screenshots are captured
  separately by an optional Playwright helper.

## 6. Scope boundaries (deliberate)

No Kafka or broker, no database server, no Kubernetes, no authentication, no
Docker, no cloud services, no paid APIs, no external AI services, no
multi-tenant isolation, no GPU work. These were excluded by the project brief and
are not silently "missing features"; they are the boundary of what this
implementation claims to be.
