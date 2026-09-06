"""
Critical Alert Manager for Automated Alerting on Critical Failures - Task 6.2.4

Detects critical failures from VerificationResult and fires structured alerts
with full issue context (PriceIssue, PolicyIssue, RelevanceIssue details).

Features:
- Alert detection from VerificationResult critical issues
- Multiple alert channels: logging (always), callback handlers, file-based alert log
- Alert deduplication within a configurable time window (avoid alert storms)
- Alert history tracking with timestamps and correlation IDs
- Integration with ErrorRateTracker to trigger alerts on threshold breaches
- Alert severity levels: CRITICAL (immediate), HIGH (urgent), MEDIUM (warning)
- Fire-and-forget design — never blocks workflow execution

Requirements:
- 8.1: Error handling with logging and correlation IDs
- 8.4: Circuit breaker / error monitoring integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..models.verification import (
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    VerificationResult,
)
from .error_rate_tracker import ErrorRateTracker, get_error_rate_tracker

logger = logging.getLogger("backend.verification.critical_alert_manager")


# ---------------------------------------------------------------------------
# Alert severity levels
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    """Alert severity levels for the alerting system."""
    CRITICAL = "CRITICAL"   # Immediate action required (fabricated policy, >30% price deviation)
    HIGH = "HIGH"           # Urgent attention needed (multiple major issues, high error rate)
    MEDIUM = "MEDIUM"       # Warning — monitor closely (threshold approaching, minor critical)


# ---------------------------------------------------------------------------
# Alert data model
# ---------------------------------------------------------------------------

@dataclass
class CriticalAlert:
    """
    A structured critical alert with full issue context.

    **Validates: Requirements 8.1** - structured error handling with context
    """
    alert_id: str
    alert_severity: AlertSeverity
    timestamp: float                        # Unix epoch seconds
    correlation_id: str
    workflow_id: str
    message: str
    issue_context: Dict[str, Any]           # Full structured issue details
    source: str                             # "verification_result" | "error_rate_threshold"
    dedup_key: str                          # Key used for deduplication

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to a JSON-serializable dict."""
        return {
            "alert_id": self.alert_id,
            "alert_severity": self.alert_severity.value,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "correlation_id": self.correlation_id,
            "workflow_id": self.workflow_id,
            "message": self.message,
            "issue_context": self.issue_context,
            "source": self.source,
            "dedup_key": self.dedup_key,
        }


# ---------------------------------------------------------------------------
# Alert callback type
# ---------------------------------------------------------------------------

AlertCallback = Callable[[CriticalAlert], None]


# ---------------------------------------------------------------------------
# CriticalAlertManager
# ---------------------------------------------------------------------------

class CriticalAlertManager:
    """
    Automated alerting system for critical verification failures.

    Detects critical issues from VerificationResult, deduplicates alerts
    within a configurable time window, and dispatches to multiple channels:
    - Structured JSON logging (always active)
    - Registered callback handlers (optional)
    - File-based alert log (optional)

    Also integrates with ErrorRateTracker to fire HIGH-severity alerts when
    error rates exceed configured thresholds.

    Design: fire-and-forget — alert dispatch never raises exceptions that
    would block the calling workflow node.

    Usage::

        manager = CriticalAlertManager(alert_log_path="logs/alerts.jsonl")

        # Register optional callback
        manager.register_callback(lambda alert: send_to_slack(alert))

        # After verification node
        manager.check_and_alert(verification_result, correlation_id, workflow_id)

        # Check error rate thresholds
        manager.check_error_rate_threshold("verification", error_rate=0.45)

    **Validates: Requirements 8.1** - error handling with logging and correlation IDs
    **Validates: Requirements 8.4** - error monitoring integration
    """

    def __init__(
        self,
        dedup_window_seconds: float = 300.0,
        error_rate_threshold: float = 0.5,
        alert_log_path: Optional[str] = None,
        error_rate_tracker: Optional[ErrorRateTracker] = None,
    ) -> None:
        """
        Initialize the CriticalAlertManager.

        Args:
            dedup_window_seconds: Time window (seconds) within which duplicate
                                  alerts for the same issue are suppressed.
                                  Default: 300 (5 minutes).
            error_rate_threshold: Error rate (0.0–1.0) above which a HIGH alert
                                  is fired for a workflow component. Default: 0.5.
            alert_log_path: Optional path to a JSONL file where all alerts are
                            appended. Directory is created if it doesn't exist.
            error_rate_tracker: Optional ErrorRateTracker instance. Uses the
                                module-level singleton if None.
        """
        self.dedup_window_seconds = dedup_window_seconds
        self.error_rate_threshold = error_rate_threshold
        self.alert_log_path = Path(alert_log_path) if alert_log_path else None

        self._error_tracker: ErrorRateTracker = (
            error_rate_tracker if error_rate_tracker is not None
            else get_error_rate_tracker()
        )

        # Registered callback handlers
        self._callbacks: List[AlertCallback] = []

        # Alert history: list of CriticalAlert (most recent last)
        self._alert_history: List[CriticalAlert] = []

        # Deduplication: dedup_key → last_fired_timestamp
        self._dedup_cache: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Ensure alert log directory exists
        if self.alert_log_path:
            self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, callback: AlertCallback) -> None:
        """
        Register an alert callback handler.

        The callback receives a CriticalAlert and is called synchronously
        during alert dispatch. Exceptions in callbacks are caught and logged
        so they never block the workflow.

        Args:
            callback: Callable that accepts a CriticalAlert.
        """
        with self._lock:
            self._callbacks.append(callback)
        logger.debug("Registered alert callback: %s", getattr(callback, "__name__", repr(callback)))

    def check_and_alert(
        self,
        verification_result: VerificationResult,
        correlation_id: str = "",
        workflow_id: str = "",
    ) -> List[CriticalAlert]:
        """
        Inspect a VerificationResult and fire alerts for any critical issues.

        Fires a CRITICAL alert if any issue has severity="critical".
        Fires a HIGH alert if escalation_priority is "immediate" or "high"
        but no individual critical-severity issues were found.

        Args:
            verification_result: The VerificationResult to inspect.
            correlation_id: Correlation ID for distributed tracing.
            workflow_id: Workflow identifier for context.

        Returns:
            List of CriticalAlert objects that were actually fired (after
            deduplication). May be empty if all alerts were deduplicated.
        """
        fired: List[CriticalAlert] = []

        criteria = verification_result.criteria

        # Collect all critical-severity issues
        critical_price = [i for i in criteria.price_issues if i.severity.value == "critical"]
        critical_policy = [i for i in criteria.policy_issues if i.severity.value == "critical"]
        critical_relevance = [i for i in criteria.relevance_issues if i.severity.value == "critical"]

        # Fire per-issue CRITICAL alerts
        for issue in critical_price:
            alert = self._build_price_alert(
                issue, AlertSeverity.CRITICAL, correlation_id, workflow_id
            )
            if self._fire_alert(alert):
                fired.append(alert)

        for issue in critical_policy:
            alert = self._build_policy_alert(
                issue, AlertSeverity.CRITICAL, correlation_id, workflow_id
            )
            if self._fire_alert(alert):
                fired.append(alert)

        for issue in critical_relevance:
            alert = self._build_relevance_alert(
                issue, AlertSeverity.CRITICAL, correlation_id, workflow_id
            )
            if self._fire_alert(alert):
                fired.append(alert)

        # If no individual critical issues but escalation priority is high/immediate,
        # fire a HIGH aggregate alert
        if not (critical_price or critical_policy or critical_relevance):
            priority = criteria.get_escalation_priority()
            if priority in ("immediate", "high"):
                alert = self._build_aggregate_alert(
                    verification_result, AlertSeverity.HIGH, correlation_id, workflow_id
                )
                if self._fire_alert(alert):
                    fired.append(alert)

        return fired

    def check_error_rate_threshold(
        self,
        component: str,
        error_rate: Optional[float] = None,
        correlation_id: str = "",
        workflow_id: str = "",
    ) -> Optional[CriticalAlert]:
        """
        Fire a HIGH alert if the component's error rate exceeds the threshold.

        Args:
            component: Workflow component name (research, verification, etc.).
            error_rate: Pre-computed error rate (0.0–1.0). If None, fetches
                        from the ErrorRateTracker using a 60-second window.
            correlation_id: Correlation ID for distributed tracing.
            workflow_id: Workflow identifier for context.

        Returns:
            The fired CriticalAlert, or None if rate is below threshold or
            the alert was deduplicated.
        """
        if error_rate is None:
            error_rate = self._error_tracker.get_error_rate(component, window_seconds=60)

        if error_rate < self.error_rate_threshold:
            return None

        severity = (
            AlertSeverity.CRITICAL if error_rate >= 0.8 else AlertSeverity.HIGH
        )

        dedup_key = f"error_rate:{component}"
        message = (
            f"Error rate threshold exceeded for component '{component}': "
            f"{error_rate:.1%} >= {self.error_rate_threshold:.1%}"
        )

        issue_context = {
            "component": component,
            "error_rate": error_rate,
            "threshold": self.error_rate_threshold,
            "component_stats": self._error_tracker.get_component_stats(component),
        }

        alert = self._create_alert(
            alert_severity=severity,
            message=message,
            issue_context=issue_context,
            source="error_rate_threshold",
            dedup_key=dedup_key,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
        )

        if self._fire_alert(alert):
            return alert
        return None

    def get_alert_history(
        self,
        limit: Optional[int] = None,
        severity_filter: Optional[AlertSeverity] = None,
    ) -> List[CriticalAlert]:
        """
        Return alert history, optionally filtered and limited.

        Args:
            limit: Maximum number of alerts to return (most recent first).
            severity_filter: If provided, only return alerts of this severity.

        Returns:
            List of CriticalAlert objects (most recent first).
        """
        with self._lock:
            history = list(reversed(self._alert_history))

        if severity_filter is not None:
            history = [a for a in history if a.alert_severity == severity_filter]

        if limit is not None:
            history = history[:limit]

        return history

    def get_alert_count(self, severity_filter: Optional[AlertSeverity] = None) -> int:
        """Return total number of alerts fired (optionally filtered by severity)."""
        with self._lock:
            if severity_filter is None:
                return len(self._alert_history)
            return sum(1 for a in self._alert_history if a.alert_severity == severity_filter)

    def clear_history(self) -> None:
        """Clear alert history and deduplication cache (useful for testing)."""
        with self._lock:
            self._alert_history.clear()
            self._dedup_cache.clear()
        logger.debug("Alert history and dedup cache cleared")

    # ------------------------------------------------------------------
    # Internal: alert building helpers
    # ------------------------------------------------------------------

    def _build_price_alert(
        self,
        issue: PriceIssue,
        severity: AlertSeverity,
        correlation_id: str,
        workflow_id: str,
    ) -> CriticalAlert:
        """Build a CriticalAlert from a PriceIssue."""
        dedup_key = f"price:{issue.product_name}:{issue.deviation_percent}"
        message = (
            f"Critical price deviation detected for '{issue.product_name}': "
            f"{issue.deviation_percent:.1f}% deviation "
            f"(mentioned={issue.mentioned_price}, actual={issue.actual_price})"
        )
        issue_context: Dict[str, Any] = {
            "issue_type": "price",
            "product_name": issue.product_name,
            "product_sku": issue.product_sku,
            "mentioned_price": issue.mentioned_price,
            "actual_price": issue.actual_price,
            "deviation_percent": issue.deviation_percent,
            "currency": issue.currency,
            "severity": issue.severity.value,
            "explanation": issue.explanation,
            "correction_suggestion": issue.correction_suggestion,
        }
        return self._create_alert(
            alert_severity=severity,
            message=message,
            issue_context=issue_context,
            source="verification_result",
            dedup_key=dedup_key,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
        )

    def _build_policy_alert(
        self,
        issue: PolicyIssue,
        severity: AlertSeverity,
        correlation_id: str,
        workflow_id: str,
    ) -> CriticalAlert:
        """Build a CriticalAlert from a PolicyIssue."""
        fabricated_tag = "FABRICATED" if issue.is_fabricated else "INACCURATE"
        dedup_key = f"policy:{issue.policy_type}:{fabricated_tag}"
        message = (
            f"Critical policy issue detected [{fabricated_tag}]: "
            f"policy_type='{issue.policy_type}' — {issue.explanation}"
        )
        issue_context: Dict[str, Any] = {
            "issue_type": "policy",
            "mentioned_policy": issue.mentioned_policy,
            "policy_type": issue.policy_type,
            "is_fabricated": issue.is_fabricated,
            "is_inaccurate": issue.is_inaccurate,
            "is_incomplete": issue.is_incomplete,
            "correct_policy": issue.correct_policy,
            "severity": issue.severity.value,
            "explanation": issue.explanation,
            "source_document": issue.source_document,
            "policy_section": issue.policy_section,
            "correction_suggestion": issue.correction_suggestion,
        }
        return self._create_alert(
            alert_severity=severity,
            message=message,
            issue_context=issue_context,
            source="verification_result",
            dedup_key=dedup_key,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
        )

    def _build_relevance_alert(
        self,
        issue: RelevanceIssue,
        severity: AlertSeverity,
        correlation_id: str,
        workflow_id: str,
    ) -> CriticalAlert:
        """Build a CriticalAlert from a RelevanceIssue."""
        dedup_key = f"relevance:{issue.objection_intent}:{issue.response_coverage:.2f}"
        message = (
            f"Critical relevance issue detected: "
            f"coverage={issue.response_coverage:.0%} for intent='{issue.objection_intent}'"
        )
        issue_context: Dict[str, Any] = {
            "issue_type": "relevance",
            "objection_intent": issue.objection_intent,
            "detected_intents": issue.detected_intents,
            "response_coverage": issue.response_coverage,
            "missing_aspects": issue.missing_aspects,
            "off_topic_content": issue.off_topic_content,
            "empathy_score": issue.empathy_score,
            "severity": issue.severity.value,
            "explanation": issue.explanation,
            "correction_suggestion": issue.correction_suggestion,
        }
        return self._create_alert(
            alert_severity=severity,
            message=message,
            issue_context=issue_context,
            source="verification_result",
            dedup_key=dedup_key,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
        )

    def _build_aggregate_alert(
        self,
        verification_result: VerificationResult,
        severity: AlertSeverity,
        correlation_id: str,
        workflow_id: str,
    ) -> CriticalAlert:
        """Build an aggregate alert from a VerificationResult with high escalation priority."""
        criteria = verification_result.criteria
        summary = criteria.get_detailed_issue_summary()
        dedup_key = (
            f"aggregate:{criteria.critical_issues_count}:"
            f"{criteria.get_major_issues_count()}:{workflow_id}"
        )
        message = (
            f"Verification failed with high escalation priority "
            f"({criteria.get_escalation_priority().upper()}): "
            f"{summary['total_issues']} total issues "
            f"({summary['critical_count']} critical, {summary['major_count']} major)"
        )
        issue_context: Dict[str, Any] = {
            "issue_type": "aggregate",
            "escalation_priority": criteria.get_escalation_priority(),
            "issue_summary": summary,
            "price_accuracy_pass": criteria.price_accuracy_pass,
            "policy_authenticity_pass": criteria.policy_authenticity_pass,
            "topic_relevance_pass": criteria.topic_relevance_pass,
            "verification_reasoning": verification_result.verification_reasoning,
            "execution_time_seconds": verification_result.execution_time_seconds,
        }
        return self._create_alert(
            alert_severity=severity,
            message=message,
            issue_context=issue_context,
            source="verification_result",
            dedup_key=dedup_key,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
        )

    def _create_alert(
        self,
        alert_severity: AlertSeverity,
        message: str,
        issue_context: Dict[str, Any],
        source: str,
        dedup_key: str,
        correlation_id: str,
        workflow_id: str,
    ) -> CriticalAlert:
        """Create a CriticalAlert with a deterministic alert_id."""
        now = time.time()
        # Stable alert_id based on dedup_key + minute bucket (so same issue in same minute = same id)
        minute_bucket = int(now // 60)
        raw = f"{dedup_key}:{minute_bucket}"
        alert_id = f"alert_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

        return CriticalAlert(
            alert_id=alert_id,
            alert_severity=alert_severity,
            timestamp=now,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            message=message,
            issue_context=issue_context,
            source=source,
            dedup_key=dedup_key,
        )

    # ------------------------------------------------------------------
    # Internal: deduplication and dispatch
    # ------------------------------------------------------------------

    def _is_duplicate(self, alert: CriticalAlert) -> bool:
        """Return True if this alert was already fired within the dedup window."""
        last_fired = self._dedup_cache.get(alert.dedup_key)
        if last_fired is None:
            return False
        return (alert.timestamp - last_fired) < self.dedup_window_seconds

    def _fire_alert(self, alert: CriticalAlert) -> bool:
        """
        Dispatch an alert to all channels if it passes deduplication.

        Returns True if the alert was actually fired, False if deduplicated.
        """
        with self._lock:
            if self._is_duplicate(alert):
                logger.debug(
                    "Alert deduplicated (dedup_key=%s, window=%.0fs)",
                    alert.dedup_key,
                    self.dedup_window_seconds,
                )
                return False

            # Mark as fired
            self._dedup_cache[alert.dedup_key] = alert.timestamp
            self._alert_history.append(alert)
            callbacks = list(self._callbacks)

        # Dispatch outside the lock to avoid deadlocks in callbacks
        self._dispatch_to_logger(alert)
        self._dispatch_to_file(alert)
        for cb in callbacks:
            self._dispatch_to_callback(cb, alert)

        return True

    def _dispatch_to_logger(self, alert: CriticalAlert) -> None:
        """Log the alert as structured JSON at the appropriate level."""
        log_data = alert.to_dict()
        log_message = (
            f"[ALERT:{alert.alert_severity.value}] {alert.message} "
            f"(correlation_id={alert.correlation_id}, workflow_id={alert.workflow_id})"
        )
        if alert.alert_severity == AlertSeverity.CRITICAL:
            logger.critical(log_message, extra={"alert": log_data})
        elif alert.alert_severity == AlertSeverity.HIGH:
            logger.error(log_message, extra={"alert": log_data})
        else:
            logger.warning(log_message, extra={"alert": log_data})

    def _dispatch_to_file(self, alert: CriticalAlert) -> None:
        """Append the alert as a JSON line to the alert log file."""
        if self.alert_log_path is None:
            return
        try:
            with open(self.alert_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict(), ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write alert to file %s: %s", self.alert_log_path, exc)

    def _dispatch_to_callback(self, callback: AlertCallback, alert: CriticalAlert) -> None:
        """Invoke a callback, catching and logging any exceptions."""
        try:
            callback(alert)
        except Exception as exc:
            logger.warning(
                "Alert callback %s raised an exception: %s",
                getattr(callback, "__name__", repr(callback)),
                exc,
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_manager: Optional[CriticalAlertManager] = None
_manager_lock = threading.Lock()


def get_critical_alert_manager(
    dedup_window_seconds: float = 300.0,
    error_rate_threshold: float = 0.5,
    alert_log_path: Optional[str] = None,
) -> CriticalAlertManager:
    """
    Return the module-level singleton CriticalAlertManager.

    Args:
        dedup_window_seconds: Only used on first call to create the singleton.
        error_rate_threshold: Only used on first call to create the singleton.
        alert_log_path: Only used on first call to create the singleton.

    Returns:
        Shared CriticalAlertManager instance.
    """
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = CriticalAlertManager(
                dedup_window_seconds=dedup_window_seconds,
                error_rate_threshold=error_rate_threshold,
                alert_log_path=alert_log_path,
            )
    return _default_manager


def reset_critical_alert_manager() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_manager
    with _manager_lock:
        _default_manager = None


__all__ = [
    "AlertSeverity",
    "CriticalAlert",
    "AlertCallback",
    "CriticalAlertManager",
    "get_critical_alert_manager",
    "reset_critical_alert_manager",
]
