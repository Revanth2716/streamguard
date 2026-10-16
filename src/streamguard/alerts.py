"""Structured alert management.

An alert is a plain record with a stable, reproducible identifier derived from
its content (`alert_type`, detector, observation timestamp, feature, reason).
That makes re-running the same simulation idempotent: regenerating the stream
does not duplicate alerts, and acknowledgement state survives a restart because
it is stored next to the alert.

Acknowledgement state is persisted to a JSON file (default
``data/processed/alert_state.json``). The store is deliberately simple: a dict
plus a lock, atomic file replacement on save, no database.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from streamguard.simulator import SIMULATED_ORIGIN

__all__ = [
    "ALERT_TYPE_ANOMALY",
    "ALERT_TYPE_DRIFT",
    "SEVERITY_RANK",
    "Alert",
    "AlertStore",
    "alert_episodes",
    "alerts_from_detector_results",
    "alerts_from_drift_reports",
    "make_alert_id",
    "severity_rank",
    "summary_frame",
]

ALERT_TYPE_ANOMALY = "anomaly_point"
ALERT_TYPE_DRIFT = "drift"
SEVERITY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_TYPE_CODE = {ALERT_TYPE_ANOMALY: "ANOM", ALERT_TYPE_DRIFT: "DRIFT"}


def severity_rank(severity: str) -> int:
    return SEVERITY_RANK.get(severity, 0)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def make_alert_id(
    alert_type: str,
    detector: str,
    timestamp: Any,
    feature: str,
    reason: str = "",
) -> str:
    """Deterministic alert identifier (content-addressed, 12 hex chars)."""
    payload = "|".join([alert_type, detector, str(timestamp), str(feature), reason.strip()])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"SG-{_TYPE_CODE.get(alert_type, 'GEN')}-{digest}"


@dataclass
class Alert:
    """A single structured alert."""

    alert_id: str
    timestamp: str
    alert_type: str
    severity: str
    feature: str
    metric: str
    detector: str
    score: float
    reason: str
    detected_at: str = field(default_factory=_now_iso)
    acknowledged: bool = False
    acknowledged_at: str | None = None
    acknowledged_by: str | None = None
    data_origin: str = SIMULATED_ORIGIN
    evidence: dict[str, Any] = field(default_factory=dict)

    def acknowledge(self, by: str = "operator", at: str | None = None) -> Alert:
        if not self.acknowledged:
            self.acknowledged = True
            self.acknowledged_at = at or _now_iso()
            self.acknowledged_by = by
        return self

    def unacknowledge(self) -> Alert:
        self.acknowledged = False
        self.acknowledged_at = None
        self.acknowledged_by = None
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Alert:
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def severity_rank(self) -> int:
        return severity_rank(self.severity)


def alerts_from_detector_results(
    results: dict[str, pd.DataFrame],
    timestamps: pd.Series,
    detectors: Iterable[str] | None = None,
) -> list[Alert]:
    """Convert detector output frames into alerts.

    ``timestamps`` must align with the row order of every frame in ``results``.
    """
    alerts: list[Alert] = []
    selected = set(detectors) if detectors is not None else None
    for name, frame in results.items():
        if selected is not None and name not in selected:
            continue
        if frame.empty or "is_anomaly" not in frame.columns:
            continue
        flagged = frame["is_anomaly"].to_numpy(dtype=bool)
        stamp_values = timestamps.to_numpy()
        for position in range(len(frame)):
            if not flagged[position]:
                continue
            row = frame.iloc[position]
            feature = str(row.get("top_feature") or "")
            metric = feature
            timestamp = str(stamp_values[position])
            reason = str(row.get("reason") or f"{name} flagged the observation")
            alert_type = ALERT_TYPE_ANOMALY
            alerts.append(
                Alert(
                    alert_id=make_alert_id(alert_type, name, timestamp, feature, reason),
                    timestamp=timestamp,
                    alert_type=alert_type,
                    severity=str(row.get("severity") or "low"),
                    feature=feature,
                    metric=metric,
                    detector=name,
                    score=float(row.get("score") or 0.0),
                    reason=reason,
                    evidence={
                        "score": round(float(row.get("score") or 0.0), 6),
                        "score_sigma": round(float(row.get("score_sigma") or 0.0), 6),
                        "row_position": int(position),
                        "row_event_index": int(row["event_index"])
                        if "event_index" in row
                        else None,
                    },
                )
            )
    return alerts


def alerts_from_drift_reports(
    reports: Iterable[Any],
    detector: str = "ks_drift",
) -> list[Alert]:
    """Create one alert per drifted feature per window."""
    alerts: list[Alert] = []
    for report in reports:
        if not getattr(report, "evaluated", False) or not report.any_drift:
            continue
        timestamp = report.start_timestamp or str(report.start_position)
        severity = report.severity
        for item in report.features:
            if not item.drifted:
                continue
            reason = (
                f"KS drift detected: {item.feature} D={item.ks_statistic:.3f}, "
                f"p_adj={item.p_adjusted:.3g} < alpha={report.alpha:g} "
                f"({report.correction} corrected), mean shift {item.mean_shift_pct:+.1f}%, "
                f"PSI={item.psi:.3f}, window n={item.n_window} vs reference n={item.n_reference}. "
                "Input distribution differs; this is NOT evidence of model degradation."
            )
            alerts.append(
                Alert(
                    alert_id=make_alert_id(
                        ALERT_TYPE_DRIFT, detector, timestamp, item.feature, reason
                    ),
                    timestamp=timestamp,
                    alert_type=ALERT_TYPE_DRIFT,
                    severity=severity,
                    feature=item.feature,
                    metric=item.feature,
                    detector=detector,
                    score=float(item.ks_statistic),
                    reason=reason,
                    evidence={
                        "window_id": report.window_id,
                        "n_window": item.n_window,
                        "ks_statistic": round(item.ks_statistic, 6),
                        "p_value": item.p_value,
                        "p_adjusted": item.p_adjusted,
                        "psi": round(item.psi, 6),
                        "mean_shift_pct": round(item.mean_shift_pct, 4),
                        "reference_mean": round(item.reference_mean, 6),
                        "window_mean": round(item.window_mean, 6),
                    },
                )
            )
    return alerts


class AlertStore:
    """In-memory alert store with optional JSON persistence and dedup by id."""

    schema_version = 1

    def __init__(
        self,
        path: str | Path | None = None,
        autoload: bool = True,
        persist_alerts: bool = True,
    ) -> None:
        """Create a store.

        ``persist_alerts=False`` keeps only an acknowledgement *journal* on disk.
        That is the right mode for a dashboard: alert records are regenerated
        deterministically for the current run, so persisting them would only
        accumulate rows from superseded configurations, while the operator's
        acknowledgements are genuine user state that must survive a restart.
        """
        self.path = Path(path) if path is not None else None
        self.persist_alerts = persist_alerts
        self.ack_journal: dict[str, dict[str, str]] = {}
        self._alerts: dict[str, Alert] = {}
        self._lock = threading.RLock()
        if autoload and self.path is not None and self.path.is_file():
            self.load()

    # -- mutation --------------------------------------------------------
    def add(self, alert: Alert) -> bool:
        """Add an alert. Returns False when the id already exists (no duplicates).

        A previously recorded acknowledgement for the same content-addressed id
        is re-applied, so regenerating the same run does not silently un-ack an
        alert an operator already triaged.
        """
        with self._lock:
            if alert.alert_id in self._alerts:
                return False
            entry = self.ack_journal.get(alert.alert_id)
            if entry:
                alert.acknowledged = True
                alert.acknowledged_at = entry.get("acknowledged_at")
                alert.acknowledged_by = entry.get("acknowledged_by")
            self._alerts[alert.alert_id] = alert
            return True

    def add_many(self, alerts: Iterable[Alert]) -> int:
        return sum(1 for alert in alerts if self.add(alert))

    def acknowledge(self, alert_id: str, by: str = "operator") -> Alert:
        with self._lock:
            if alert_id not in self._alerts:
                raise KeyError(f"unknown alert id: {alert_id}")
            alert = self._alerts[alert_id].acknowledge(by=by)
            self.ack_journal[alert_id] = {
                "acknowledged_at": alert.acknowledged_at or _now_iso(),
                "acknowledged_by": alert.acknowledged_by or by,
            }
            return alert

    def acknowledge_many(self, alert_ids: Iterable[str], by: str = "operator") -> int:
        return sum(1 for alert_id in alert_ids if self.acknowledge(alert_id, by=by))

    def unacknowledge(self, alert_id: str) -> Alert:
        with self._lock:
            if alert_id not in self._alerts:
                raise KeyError(f"unknown alert id: {alert_id}")
            self.ack_journal.pop(alert_id, None)
            return self._alerts[alert_id].unacknowledge()

    def clear(self) -> None:
        """Drop every alert from memory, keeping the acknowledgement journal."""
        with self._lock:
            self._alerts.clear()

    def clear_journal(self) -> None:
        """Forget every acknowledgement (used by the dashboard reset control)."""
        with self._lock:
            self.ack_journal.clear()
            for alert in self._alerts.values():
                alert.unacknowledge()

    # -- reads -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._alerts)

    def get(self, alert_id: str) -> Alert | None:
        return self._alerts.get(alert_id)

    def all(
        self,
        severity: str | None = None,
        acknowledged: bool | None = None,
        alert_type: str | None = None,
    ) -> list[Alert]:
        with self._lock:
            alerts = list(self._alerts.values())
        if severity is not None:
            alerts = [alert for alert in alerts if alert.severity == severity]
        if acknowledged is not None:
            alerts = [alert for alert in alerts if alert.acknowledged is acknowledged]
        if alert_type is not None:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]
        return sorted(alerts, key=lambda alert: (alert.timestamp, -alert.severity_rank))

    def counts(self) -> dict[str, Any]:
        alerts = self.all()
        by_severity = dict.fromkeys(SEVERITY_RANK, 0)
        by_type: dict[str, int] = {}
        for alert in alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1
        unacknowledged = [alert for alert in alerts if not alert.acknowledged]
        return {
            "total": len(alerts),
            "unacknowledged": len(unacknowledged),
            "acknowledged": len(alerts) - len(unacknowledged),
            "by_severity": by_severity,
            "by_type": by_type,
            "unacknowledged_critical": sum(
                1 for alert in unacknowledged if alert.severity == "critical"
            ),
        }

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            "alert_id",
            "timestamp",
            "alert_type",
            "severity",
            "feature",
            "detector",
            "score",
            "acknowledged",
            "acknowledged_at",
            "reason",
        ]
        alerts = self.all()
        if not alerts:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame([alert.to_dict() for alert in alerts])
        return frame.loc[:, columns]

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "data_origin": SIMULATED_ORIGIN,
            "acknowledgements": self.ack_journal,
            "alerts": [alert.to_dict() for alert in self.all()] if self.persist_alerts else [],
        }

    def save(self) -> Path | None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        # Write to a sibling temp file and replace atomically, so a crash cannot
        # leave a half-written acknowledgement journal behind.
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)
        return self.path

    def load(self) -> int:
        if self.path is None or not self.path.is_file():
            return 0
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, list):  # schema-less list of alerts
            raw = {"alerts": raw, "acknowledgements": {}}
        journal = raw.get("acknowledgements", {}) or {}
        entries = raw.get("alerts", []) or []
        loaded = 0
        with self._lock:
            self.ack_journal.update({str(key): value for key, value in journal.items()})
            for entry in entries:
                alert = Alert.from_dict(entry)
                if alert.alert_id in self.ack_journal and not alert.acknowledged:
                    recorded = self.ack_journal[alert.alert_id]
                    alert.acknowledged = True
                    alert.acknowledged_at = recorded.get("acknowledged_at")
                    alert.acknowledged_by = recorded.get("acknowledged_by")
                self._alerts[alert.alert_id] = alert
                loaded += 1
        return loaded


def alert_episodes(
    alerts: Iterable[Alert], gap: int = 3, detector: str | None = None
) -> pd.DataFrame:
    """Collapse consecutive per-observation alerts into episodes.

    A single anomaly that spans several observations raises one alert per row (so
    nobody can quietly drop a spike). For human triage that is noisy, so the same
    alerts are also summarised as episodes: alerts from one detector affected by
    the same feature whose observation positions are at most ``gap`` rows apart.
    """
    selected = [
        alert
        for alert in alerts
        if alert.alert_type == ALERT_TYPE_ANOMALY
        and (detector is None or alert.detector == detector)
    ]
    selected.sort(key=lambda alert: (alert.detector, alert.feature, alert.timestamp))
    episodes: list[dict[str, Any]] = []
    for alert in selected:
        position = alert.evidence.get("row_position")
        if episodes:
            last = episodes[-1]
            same_group = (
                last["detector"] == alert.detector
                and last["feature"] == alert.feature
                and position is not None
                and last["last_position"] is not None
                and position - last["last_position"] <= gap
            )
            if same_group:
                last["alerts"] += 1
                last["last_position"] = position
                last["end_timestamp"] = alert.timestamp
                last["severity"] = max((last["severity"], alert.severity), key=severity_rank)
                last["alert_ids"].append(alert.alert_id)
                continue
        episodes.append(
            {
                "detector": alert.detector,
                "feature": alert.feature,
                "severity": alert.severity,
                "alerts": 1,
                "start_timestamp": alert.timestamp,
                "end_timestamp": alert.timestamp,
                "first_position": position,
                "last_position": position,
                "acknowledged": alert.acknowledged,
                "reason": alert.reason,
                "alert_ids": [alert.alert_id],
            }
        )
    if not episodes:
        return pd.DataFrame(
            columns=[
                "detector",
                "feature",
                "severity",
                "alerts",
                "start_timestamp",
                "end_timestamp",
                "first_position",
                "last_position",
                "acknowledged",
                "reason",
                "alert_ids",
            ]
        )
    return pd.DataFrame(episodes).sort_values(
        "severity", key=lambda s: s.map(severity_rank), ascending=False
    )


def summary_frame(store: AlertStore) -> pd.DataFrame:
    """Severity-count table for dashboards and reports."""
    counts = store.counts()
    rows = [
        {"severity": level, "count": counts["by_severity"].get(level, 0)}
        for level in ("critical", "high", "medium", "low")
    ]
    return pd.DataFrame(rows)
