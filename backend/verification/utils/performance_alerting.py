"""
Performance Alerting Thresholds with Early Termination Benefits - Task 5.3.5

Provides a unified PerformanceAlertingSystem that:
- Monitors latency, throughput, resource usage, and token costs
- Fires structured PerformanceAlert objects when thresholds are exceeded
- Integrates early termination benefits: slow/expensive workflows trigger
  early termination recommendations to avoid wasting further resources
- Supports configurable thresholds via PerformanceAlertThresholds
- Provides async-compatible alert checking

Integrates with:
- AsyncStepLatencyTracker (performance.py)
- ThroughputMonitor (performance.py)
- ResourceUsageMonitor (resource_monitor.py)
- LLMTokenTracker (token_tracker.py)
- EarlyTerminationManager (early_termination.py)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from backend.verification.utils.performance import AsyncStepLatencyTracker, ThroughputMonitor
from backend.verification.utils.resource_monitor import ResourceUsageMonitor, ResourceUsageReport
from backend.verification.utils.token_tracker import LLMTokenTracker, TokenUsageSummary


# ---------------------------------------------------------------------------
# Alert severity and types
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE = "resource"
    TOKEN_COST = "token_cost"
    EARLY_TERMINATION_RECOMMENDED = "early_termination_recommended"


# ---------------------------------------------------------------------------
# Alert data class
# ---------------------------------------------------------------------------

@dataclass
class PerformanceAlert:
    """A single performance alert fired when a threshold is exceeded."""

    alert_type: AlertType
    severity: AlertSeverity
    threshold: float
    actual_value: float
    message: str
    recommend_early_termination: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "threshold": self.threshold,
            "actual_value": round(self.actual_value, 4),
            "message": self.message,
            "recommend_early_termination": self.recommend_early_termination,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

@dataclass
class PerformanceAlertThresholds:
    """
    Configurable thresholds for performance alerting.

    Latency thresholds (milliseconds):
        warning_latency_ms  - single step latency warning
        critical_latency_ms - single step latency critical (triggers early termination recommendation)
        total_latency_warning_ms  - aggregate workflow latency warning
        total_latency_critical_ms - aggregate workflow latency critical

    Throughput thresholds:
        min_wps_warning  - minimum workflows/sec before warning
        max_fail_rate_warning  - maximum FAIL ratio (0-1) before warning
        max_fail_rate_critical - maximum FAIL ratio before critical alert

    Resource thresholds:
        max_memory_mb_warning  - peak memory warning (MB)
        max_memory_mb_critical - peak memory critical (MB)
        max_cpu_percent_warning  - peak CPU % warning
        max_cpu_percent_critical - peak CPU % critical

    Token / cost thresholds:
        max_tokens_per_workflow_warning  - total tokens warning
        max_tokens_per_workflow_critical - total tokens critical
        max_cost_per_workflow_warning    - total cost warning (USD)
        max_cost_per_workflow_critical   - total cost critical (USD)

    Early termination:
        enable_early_termination_recommendations - whether to set
            recommend_early_termination=True on critical alerts
    """

    # Latency (ms)
    warning_latency_ms: float = 3_000.0
    critical_latency_ms: float = 8_000.0
    total_latency_warning_ms: float = 8_000.0
    total_latency_critical_ms: float = 15_000.0

    # Throughput
    min_wps_warning: float = 0.05
    max_fail_rate_warning: float = 0.5
    max_fail_rate_critical: float = 0.8

    # Resource
    max_memory_mb_warning: float = 512.0
    max_memory_mb_critical: float = 1_024.0
    max_cpu_percent_warning: float = 70.0
    max_cpu_percent_critical: float = 90.0

    # Token / cost
    max_tokens_per_workflow_warning: int = 20_000
    max_tokens_per_workflow_critical: int = 50_000
    max_cost_per_workflow_warning: float = 0.05
    max_cost_per_workflow_critical: float = 0.20

    # Early termination integration
    enable_early_termination_recommendations: bool = True

    # Optional callback invoked for every alert
    alert_callback: Optional[Callable[[PerformanceAlert], None]] = None


# ---------------------------------------------------------------------------
# Main alerting system
# ---------------------------------------------------------------------------

class PerformanceAlertingSystem:
    """
    Unified performance alerting system that monitors all verification metrics
    and fires PerformanceAlert objects when thresholds are exceeded.

    Integrates early termination benefits: when a workflow is detected as
    critically slow or expensive, the system recommends early termination
    to avoid wasting further LLM calls and resources.

    Usage::

        thresholds = PerformanceAlertThresholds(critical_latency_ms=5000)
        alerting = PerformanceAlertingSystem(thresholds=thresholds)

        # After verification steps complete:
        alerting.check_latency(latency_tracker)
        alerting.check_resources(resource_report)
        alerting.check_tokens(token_tracker)
        alerting.check_throughput(throughput_monitor)

        if alerting.should_recommend_early_termination():
            # Skip remaining checks — workflow is already too slow/expensive
            return early_termination_result()

        print(alerting.get_alert_summary())
    """

    def __init__(
        self,
        thresholds: Optional[PerformanceAlertThresholds] = None,
    ) -> None:
        self._thresholds = thresholds or PerformanceAlertThresholds()
        self._alerts: List[PerformanceAlert] = []

    # ------------------------------------------------------------------
    # Latency checks
    # ------------------------------------------------------------------

    def check_latency(self, tracker: AsyncStepLatencyTracker) -> List[PerformanceAlert]:
        """
        Check per-step and aggregate latency against thresholds.

        Returns newly fired alerts (also stored internally).
        """
        new_alerts: List[PerformanceAlert] = []
        t = self._thresholds

        for record in tracker.get_all_records():
            ms = record.duration_ms
            if ms >= t.critical_latency_ms:
                alert = self._make_alert(
                    alert_type=AlertType.LATENCY,
                    severity=AlertSeverity.CRITICAL,
                    threshold=t.critical_latency_ms,
                    actual_value=ms,
                    message=(
                        f"Step '{record.step_name}' latency {ms:.1f}ms exceeds critical "
                        f"threshold {t.critical_latency_ms:.0f}ms"
                    ),
                    recommend_early_termination=t.enable_early_termination_recommendations,
                    metadata={"step_name": record.step_name},
                )
                new_alerts.append(alert)
            elif ms >= t.warning_latency_ms:
                alert = self._make_alert(
                    alert_type=AlertType.LATENCY,
                    severity=AlertSeverity.WARNING,
                    threshold=t.warning_latency_ms,
                    actual_value=ms,
                    message=(
                        f"Step '{record.step_name}' latency {ms:.1f}ms exceeds warning "
                        f"threshold {t.warning_latency_ms:.0f}ms"
                    ),
                    metadata={"step_name": record.step_name},
                )
                new_alerts.append(alert)

        # Aggregate total latency
        metrics = tracker.get_all_metrics()
        agg = metrics.get("aggregate_latency", {})
        total_ms = agg.get("total_ms", 0.0)

        if total_ms >= t.total_latency_critical_ms:
            alert = self._make_alert(
                alert_type=AlertType.LATENCY,
                severity=AlertSeverity.CRITICAL,
                threshold=t.total_latency_critical_ms,
                actual_value=total_ms,
                message=(
                    f"Total workflow latency {total_ms:.1f}ms exceeds critical "
                    f"threshold {t.total_latency_critical_ms:.0f}ms"
                ),
                recommend_early_termination=t.enable_early_termination_recommendations,
                metadata={"aggregate": True},
            )
            new_alerts.append(alert)
        elif total_ms >= t.total_latency_warning_ms:
            alert = self._make_alert(
                alert_type=AlertType.LATENCY,
                severity=AlertSeverity.WARNING,
                threshold=t.total_latency_warning_ms,
                actual_value=total_ms,
                message=(
                    f"Total workflow latency {total_ms:.1f}ms exceeds warning "
                    f"threshold {t.total_latency_warning_ms:.0f}ms"
                ),
                metadata={"aggregate": True},
            )
            new_alerts.append(alert)

        return new_alerts

    # ------------------------------------------------------------------
    # Throughput checks
    # ------------------------------------------------------------------

    def check_throughput(
        self,
        monitor: ThroughputMonitor,
        window_seconds: float = 60.0,
    ) -> List[PerformanceAlert]:
        """
        Check throughput (WPS and fail rate) against thresholds.

        Returns newly fired alerts.
        """
        new_alerts: List[PerformanceAlert] = []
        t = self._thresholds
        snap = monitor.snapshot(window_seconds=window_seconds)

        # Low throughput warning
        if snap.total_completed > 0 and snap.wps < t.min_wps_warning:
            alert = self._make_alert(
                alert_type=AlertType.THROUGHPUT,
                severity=AlertSeverity.WARNING,
                threshold=t.min_wps_warning,
                actual_value=snap.wps,
                message=(
                    f"Throughput {snap.wps:.4f} WPS is below minimum warning "
                    f"threshold {t.min_wps_warning:.4f} WPS "
                    f"(window={window_seconds:.0f}s)"
                ),
                metadata={"window_seconds": window_seconds},
            )
            new_alerts.append(alert)

        # High fail rate
        if snap.total_completed > 0:
            fail_rate = snap.fail_count / snap.total_completed
            if fail_rate >= t.max_fail_rate_critical:
                alert = self._make_alert(
                    alert_type=AlertType.THROUGHPUT,
                    severity=AlertSeverity.CRITICAL,
                    threshold=t.max_fail_rate_critical,
                    actual_value=fail_rate,
                    message=(
                        f"Verification fail rate {fail_rate:.1%} exceeds critical "
                        f"threshold {t.max_fail_rate_critical:.1%} "
                        f"({snap.fail_count}/{snap.total_completed} workflows failed)"
                    ),
                    recommend_early_termination=t.enable_early_termination_recommendations,
                    metadata={"fail_count": snap.fail_count, "total": snap.total_completed},
                )
                new_alerts.append(alert)
            elif fail_rate >= t.max_fail_rate_warning:
                alert = self._make_alert(
                    alert_type=AlertType.THROUGHPUT,
                    severity=AlertSeverity.WARNING,
                    threshold=t.max_fail_rate_warning,
                    actual_value=fail_rate,
                    message=(
                        f"Verification fail rate {fail_rate:.1%} exceeds warning "
                        f"threshold {t.max_fail_rate_warning:.1%} "
                        f"({snap.fail_count}/{snap.total_completed} workflows failed)"
                    ),
                    metadata={"fail_count": snap.fail_count, "total": snap.total_completed},
                )
                new_alerts.append(alert)

        return new_alerts

    # ------------------------------------------------------------------
    # Resource checks
    # ------------------------------------------------------------------

    def check_resources(self, report: ResourceUsageReport) -> List[PerformanceAlert]:
        """
        Check memory and CPU usage against thresholds.

        Returns newly fired alerts.
        """
        new_alerts: List[PerformanceAlert] = []
        t = self._thresholds

        # Memory
        mem = report.peak_memory_rss_mb
        if mem >= t.max_memory_mb_critical:
            alert = self._make_alert(
                alert_type=AlertType.RESOURCE,
                severity=AlertSeverity.CRITICAL,
                threshold=t.max_memory_mb_critical,
                actual_value=mem,
                message=(
                    f"Peak memory {mem:.1f}MB exceeds critical threshold "
                    f"{t.max_memory_mb_critical:.0f}MB"
                ),
                recommend_early_termination=t.enable_early_termination_recommendations,
                metadata={"resource": "memory_rss_mb"},
            )
            new_alerts.append(alert)
        elif mem >= t.max_memory_mb_warning:
            alert = self._make_alert(
                alert_type=AlertType.RESOURCE,
                severity=AlertSeverity.WARNING,
                threshold=t.max_memory_mb_warning,
                actual_value=mem,
                message=(
                    f"Peak memory {mem:.1f}MB exceeds warning threshold "
                    f"{t.max_memory_mb_warning:.0f}MB"
                ),
                metadata={"resource": "memory_rss_mb"},
            )
            new_alerts.append(alert)

        # CPU
        cpu = report.peak_cpu_percent
        if cpu >= t.max_cpu_percent_critical:
            alert = self._make_alert(
                alert_type=AlertType.RESOURCE,
                severity=AlertSeverity.CRITICAL,
                threshold=t.max_cpu_percent_critical,
                actual_value=cpu,
                message=(
                    f"Peak CPU {cpu:.1f}% exceeds critical threshold "
                    f"{t.max_cpu_percent_critical:.0f}%"
                ),
                recommend_early_termination=t.enable_early_termination_recommendations,
                metadata={"resource": "cpu_percent"},
            )
            new_alerts.append(alert)
        elif cpu >= t.max_cpu_percent_warning:
            alert = self._make_alert(
                alert_type=AlertType.RESOURCE,
                severity=AlertSeverity.WARNING,
                threshold=t.max_cpu_percent_warning,
                actual_value=cpu,
                message=(
                    f"Peak CPU {cpu:.1f}% exceeds warning threshold "
                    f"{t.max_cpu_percent_warning:.0f}%"
                ),
                metadata={"resource": "cpu_percent"},
            )
            new_alerts.append(alert)

        return new_alerts

    # ------------------------------------------------------------------
    # Token / cost checks
    # ------------------------------------------------------------------

    def check_tokens(self, tracker: LLMTokenTracker) -> List[PerformanceAlert]:
        """
        Check token usage and estimated cost against thresholds.

        Returns newly fired alerts.
        """
        new_alerts: List[PerformanceAlert] = []
        t = self._thresholds
        summary = tracker.summary()

        # Token count
        tokens = summary.total_tokens
        if tokens >= t.max_tokens_per_workflow_critical:
            alert = self._make_alert(
                alert_type=AlertType.TOKEN_COST,
                severity=AlertSeverity.CRITICAL,
                threshold=float(t.max_tokens_per_workflow_critical),
                actual_value=float(tokens),
                message=(
                    f"Workflow token usage {tokens:,} exceeds critical threshold "
                    f"{t.max_tokens_per_workflow_critical:,} tokens"
                ),
                recommend_early_termination=t.enable_early_termination_recommendations,
                metadata={"metric": "total_tokens"},
            )
            new_alerts.append(alert)
        elif tokens >= t.max_tokens_per_workflow_warning:
            alert = self._make_alert(
                alert_type=AlertType.TOKEN_COST,
                severity=AlertSeverity.WARNING,
                threshold=float(t.max_tokens_per_workflow_warning),
                actual_value=float(tokens),
                message=(
                    f"Workflow token usage {tokens:,} exceeds warning threshold "
                    f"{t.max_tokens_per_workflow_warning:,} tokens"
                ),
                metadata={"metric": "total_tokens"},
            )
            new_alerts.append(alert)

        # Cost
        cost = summary.total_cost_usd
        if cost >= t.max_cost_per_workflow_critical:
            alert = self._make_alert(
                alert_type=AlertType.TOKEN_COST,
                severity=AlertSeverity.CRITICAL,
                threshold=t.max_cost_per_workflow_critical,
                actual_value=cost,
                message=(
                    f"Workflow cost ${cost:.4f} exceeds critical threshold "
                    f"${t.max_cost_per_workflow_critical:.4f}"
                ),
                recommend_early_termination=t.enable_early_termination_recommendations,
                metadata={"metric": "cost_usd"},
            )
            new_alerts.append(alert)
        elif cost >= t.max_cost_per_workflow_warning:
            alert = self._make_alert(
                alert_type=AlertType.TOKEN_COST,
                severity=AlertSeverity.WARNING,
                threshold=t.max_cost_per_workflow_warning,
                actual_value=cost,
                message=(
                    f"Workflow cost ${cost:.4f} exceeds warning threshold "
                    f"${t.max_cost_per_workflow_warning:.4f}"
                ),
                metadata={"metric": "cost_usd"},
            )
            new_alerts.append(alert)

        return new_alerts

    # ------------------------------------------------------------------
    # Early termination recommendation
    # ------------------------------------------------------------------

    def should_recommend_early_termination(self) -> bool:
        """
        Return True if any fired alert recommends early termination.

        This integrates the early termination benefit: when performance is
        critically degraded, the system recommends stopping the current
        workflow to avoid wasting further resources on a slow/expensive path.
        """
        return any(a.recommend_early_termination for a in self._alerts)

    def get_early_termination_reason(self) -> Optional[str]:
        """
        Return a human-readable reason for early termination, or None.

        Aggregates messages from all alerts that recommend early termination.
        """
        reasons = [
            a.message for a in self._alerts if a.recommend_early_termination
        ]
        if not reasons:
            return None
        return "; ".join(reasons)

    # ------------------------------------------------------------------
    # Summary and reporting
    # ------------------------------------------------------------------

    @property
    def alerts(self) -> List[PerformanceAlert]:
        """All alerts fired since last reset."""
        return list(self._alerts)

    def has_alerts(self) -> bool:
        return bool(self._alerts)

    def has_critical_alerts(self) -> bool:
        return any(a.severity == AlertSeverity.CRITICAL for a in self._alerts)

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Return a summary dict suitable for ExecutionStep.metrics or logging.
        """
        critical = [a for a in self._alerts if a.severity == AlertSeverity.CRITICAL]
        warnings = [a for a in self._alerts if a.severity == AlertSeverity.WARNING]
        return {
            "total_alerts": len(self._alerts),
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "recommend_early_termination": self.should_recommend_early_termination(),
            "early_termination_reason": self.get_early_termination_reason(),
            "alerts": [a.to_dict() for a in self._alerts],
        }

    def reset(self) -> None:
        """Clear all fired alerts."""
        self._alerts.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        threshold: float,
        actual_value: float,
        message: str,
        recommend_early_termination: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceAlert:
        alert = PerformanceAlert(
            alert_type=alert_type,
            severity=severity,
            threshold=threshold,
            actual_value=actual_value,
            message=message,
            recommend_early_termination=recommend_early_termination,
            metadata=metadata or {},
        )
        self._alerts.append(alert)
        if self._thresholds.alert_callback is not None:
            try:
                self._thresholds.alert_callback(alert)
            except Exception:
                pass
        return alert


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_performance_alerting_system(
    environment: Optional[str] = None,
    alert_callback: Optional[Callable[[PerformanceAlert], None]] = None,
) -> PerformanceAlertingSystem:
    """
    Create a PerformanceAlertingSystem with environment-appropriate thresholds.

    Args:
        environment: "development", "production", or "testing".
            - development: relaxed thresholds, early termination disabled
            - production:  strict thresholds, early termination enabled
            - testing:     very relaxed thresholds, early termination disabled
        alert_callback: Optional callable invoked for every alert.
    """
    env_thresholds: Dict[str, PerformanceAlertThresholds] = {
        "development": PerformanceAlertThresholds(
            warning_latency_ms=10_000.0,
            critical_latency_ms=30_000.0,
            total_latency_warning_ms=30_000.0,
            total_latency_critical_ms=60_000.0,
            max_memory_mb_warning=1_024.0,
            max_memory_mb_critical=2_048.0,
            max_tokens_per_workflow_warning=100_000,
            max_tokens_per_workflow_critical=200_000,
            max_cost_per_workflow_warning=0.50,
            max_cost_per_workflow_critical=1.00,
            enable_early_termination_recommendations=False,
            alert_callback=alert_callback,
        ),
        "production": PerformanceAlertThresholds(
            warning_latency_ms=2_000.0,
            critical_latency_ms=5_000.0,
            total_latency_warning_ms=5_000.0,
            total_latency_critical_ms=10_000.0,
            max_memory_mb_warning=256.0,
            max_memory_mb_critical=512.0,
            max_tokens_per_workflow_warning=10_000,
            max_tokens_per_workflow_critical=25_000,
            max_cost_per_workflow_warning=0.02,
            max_cost_per_workflow_critical=0.10,
            enable_early_termination_recommendations=True,
            alert_callback=alert_callback,
        ),
        "testing": PerformanceAlertThresholds(
            warning_latency_ms=60_000.0,
            critical_latency_ms=120_000.0,
            total_latency_warning_ms=120_000.0,
            total_latency_critical_ms=300_000.0,
            max_memory_mb_warning=4_096.0,
            max_memory_mb_critical=8_192.0,
            max_tokens_per_workflow_warning=1_000_000,
            max_tokens_per_workflow_critical=2_000_000,
            max_cost_per_workflow_warning=10.0,
            max_cost_per_workflow_critical=50.0,
            enable_early_termination_recommendations=False,
            alert_callback=alert_callback,
        ),
    }

    thresholds = env_thresholds.get(environment or "")
    if thresholds is None:
        thresholds = PerformanceAlertThresholds(alert_callback=alert_callback)

    return PerformanceAlertingSystem(thresholds=thresholds)


__all__ = [
    "AlertSeverity",
    "AlertType",
    "PerformanceAlert",
    "PerformanceAlertThresholds",
    "PerformanceAlertingSystem",
    "create_performance_alerting_system",
]
