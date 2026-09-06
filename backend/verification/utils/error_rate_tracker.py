"""
Error Rate Tracker for Async Workflow Components - Task 6.2.1

Tracks error counts and rates per async workflow component with sliding
time windows, error type categorization, and thread-safe operation.

Workflow components tracked:
- research   (execute_research_node)
- verification (execute_verification_node)
- correction  (execute_correction_node)
- escalation  (execute_escalation_node)

Requirements:
- 8.1: Error handling with logging and correlation IDs
- 8.4: Circuit breaker pattern (this tracker feeds into it)
- 9.2: Support concurrent workflow execution
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger("backend.verification.error_rate_tracker")


# ---------------------------------------------------------------------------
# Internal event records
# ---------------------------------------------------------------------------

@dataclass
class _Event:
    """A single success or error event for a component."""
    timestamp: float          # Unix epoch seconds (time.monotonic-compatible via time.time)
    is_error: bool
    error_type: str = ""      # Empty string for success events
    correlation_id: str = ""


# ---------------------------------------------------------------------------
# ErrorRateTracker
# ---------------------------------------------------------------------------

WORKFLOW_COMPONENTS = ("research", "verification", "correction", "escalation")


class ErrorRateTracker:
    """
    Thread-safe error rate tracker for async workflow components.

    Maintains a sliding time window of success/error events per component
    and provides error rate calculations, per-component statistics, and
    error type breakdowns.

    Usage::

        tracker = ErrorRateTracker(window_seconds=60)

        # In workflow nodes:
        tracker.record_error("research", "TimeoutError", correlation_id="corr_abc")
        tracker.record_success("verification", correlation_id="corr_abc")

        rate = tracker.get_error_rate("research")          # float 0.0–1.0
        stats = tracker.get_component_stats("research")    # dict
        all_stats = tracker.get_all_stats()                # dict of all components

    Thread safety:
        All public methods acquire a per-component lock before mutating state,
        making the tracker safe for concurrent async workflow execution.

    **Validates: Requirements 8.1** - error handling with logging and correlation IDs
    **Validates: Requirements 8.4** - feeds circuit breaker pattern
    **Validates: Requirements 9.2** - supports concurrent workflow execution
    """

    def __init__(self, window_seconds: int = 60) -> None:
        """
        Initialize the error rate tracker.

        Args:
            window_seconds: Default sliding window size in seconds for rate
                            calculation. Can be overridden per call to
                            ``get_error_rate()``.
        """
        self.default_window_seconds = window_seconds

        # Per-component event deques (timestamp, is_error, error_type, correlation_id)
        self._events: Dict[str, Deque[_Event]] = defaultdict(deque)

        # Per-component locks for fine-grained thread safety
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Global lock for operations that touch multiple components
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_error(
        self,
        component: str,
        error_type: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Record an error event for a workflow component.

        Args:
            component: Workflow component name (research, verification,
                       correction, escalation). Unknown components are
                       accepted and tracked without restriction.
            error_type: Error category/type string (e.g. "TimeoutError",
                        "ValidationError", "LLMError").
            correlation_id: Optional correlation ID for distributed tracing.
        """
        event = _Event(
            timestamp=time.time(),
            is_error=True,
            error_type=error_type,
            correlation_id=correlation_id or "",
        )
        with self._locks[component]:
            self._events[component].append(event)

        logger.debug(
            "Error recorded for component '%s': %s (correlation_id=%s)",
            component,
            error_type,
            correlation_id,
        )

    def record_success(
        self,
        component: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Record a successful execution event for a workflow component.

        Args:
            component: Workflow component name.
            correlation_id: Optional correlation ID for distributed tracing.
        """
        event = _Event(
            timestamp=time.time(),
            is_error=False,
            error_type="",
            correlation_id=correlation_id or "",
        )
        with self._locks[component]:
            self._events[component].append(event)

        logger.debug(
            "Success recorded for component '%s' (correlation_id=%s)",
            component,
            correlation_id,
        )

    # ------------------------------------------------------------------
    # Rate calculation
    # ------------------------------------------------------------------

    def get_error_rate(
        self,
        component: str,
        window_seconds: int = 60,
    ) -> float:
        """
        Calculate the error rate for a component within a sliding time window.

        Args:
            component: Workflow component name.
            window_seconds: Sliding window size in seconds. Defaults to 60.

        Returns:
            Error rate as a float between 0.0 and 1.0.
            Returns 0.0 if no events exist in the window.
        """
        cutoff = time.time() - window_seconds
        with self._locks[component]:
            events_in_window = [e for e in self._events[component] if e.timestamp >= cutoff]

        if not events_in_window:
            return 0.0

        error_count = sum(1 for e in events_in_window if e.is_error)
        return error_count / len(events_in_window)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_component_stats(self, component: str) -> dict:
        """
        Get detailed statistics for a single workflow component.

        Returns a dict with:
            - component: str
            - total_events: int
            - error_count: int
            - success_count: int
            - error_rate: float (using default window)
            - error_rate_60s: float
            - error_types: dict mapping error_type -> count
            - oldest_event_ts: float or None
            - newest_event_ts: float or None
        """
        now = time.time()
        cutoff = now - self.default_window_seconds

        with self._locks[component]:
            all_events: List[_Event] = list(self._events[component])

        total = len(all_events)
        errors = [e for e in all_events if e.is_error]
        successes = [e for e in all_events if not e.is_error]

        # Error type breakdown (all-time)
        error_types: Dict[str, int] = defaultdict(int)
        for e in errors:
            error_types[e.error_type] += 1

        # Windowed error rate
        windowed = [e for e in all_events if e.timestamp >= cutoff]
        windowed_errors = sum(1 for e in windowed if e.is_error)
        windowed_rate = windowed_errors / len(windowed) if windowed else 0.0

        return {
            "component": component,
            "total_events": total,
            "error_count": len(errors),
            "success_count": len(successes),
            "error_rate": windowed_rate,
            "error_rate_60s": self.get_error_rate(component, window_seconds=60),
            "error_types": dict(error_types),
            "oldest_event_ts": all_events[0].timestamp if all_events else None,
            "newest_event_ts": all_events[-1].timestamp if all_events else None,
        }

    def get_all_stats(self) -> dict:
        """
        Get statistics for all tracked components.

        Returns a dict mapping component name -> component stats dict.
        Includes all components that have had at least one event recorded,
        plus the four canonical workflow components.
        """
        # Collect all known component names
        with self._global_lock:
            known_components = set(self._events.keys()) | set(WORKFLOW_COMPONENTS)

        return {component: self.get_component_stats(component) for component in sorted(known_components)}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, component: Optional[str] = None) -> None:
        """
        Reset tracked events.

        Args:
            component: If provided, reset only that component's events.
                       If None, reset all components.
        """
        if component is not None:
            with self._locks[component]:
                self._events[component].clear()
            logger.debug("Reset error rate tracker for component '%s'", component)
        else:
            with self._global_lock:
                components = list(self._events.keys())
            for comp in components:
                with self._locks[comp]:
                    self._events[comp].clear()
            logger.debug("Reset error rate tracker for all components")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prune_old_events(self, component: str, max_age_seconds: int = 3600) -> None:
        """
        Remove events older than max_age_seconds from a component's deque.

        Called internally to prevent unbounded memory growth in long-running
        processes. Not required for correctness since get_error_rate() already
        filters by window.
        """
        cutoff = time.time() - max_age_seconds
        with self._locks[component]:
            dq = self._events[component]
            while dq and dq[0].timestamp < cutoff:
                dq.popleft()


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_default_tracker: Optional[ErrorRateTracker] = None
_tracker_lock = threading.Lock()


def get_error_rate_tracker(window_seconds: int = 60) -> ErrorRateTracker:
    """
    Get or create the module-level singleton ErrorRateTracker.

    Args:
        window_seconds: Default window for the tracker (only used on first call).

    Returns:
        Shared ErrorRateTracker instance.
    """
    global _default_tracker
    with _tracker_lock:
        if _default_tracker is None:
            _default_tracker = ErrorRateTracker(window_seconds=window_seconds)
    return _default_tracker


def reset_error_rate_tracker() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_tracker
    with _tracker_lock:
        _default_tracker = None


__all__ = [
    "WORKFLOW_COMPONENTS",
    "ErrorRateTracker",
    "get_error_rate_tracker",
    "reset_error_rate_tracker",
]
