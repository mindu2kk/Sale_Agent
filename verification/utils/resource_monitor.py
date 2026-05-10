"""
Resource Usage Metrics - Task 5.3.3 / 5.2.5

Tracks memory, CPU, and async task resource usage for verification workflows.
Integrates with ExecutionStep.metrics and WorkflowMetrics for observability.

Task 5.2.5 additions:
- ResourceThresholds: configurable limits for memory, CPU, and async tasks
- Automatic cleanup callbacks fired when thresholds are exceeded
- Cleanup on monitoring session end (context manager __exit__)
- CleanupAction enum for structured cleanup responses
"""

import asyncio
import gc
import logging
import time
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cleanup support (Task 5.2.5)
# ---------------------------------------------------------------------------

class CleanupAction(str, Enum):
    """Actions taken during automatic resource cleanup."""
    GC_COLLECT = "gc_collect"           # Force Python garbage collection
    CLEAR_SNAPSHOTS = "clear_snapshots" # Drop in-memory snapshot history
    CUSTOM = "custom"                   # User-supplied callback


@dataclass
class ResourceThresholds:
    """
    Configurable resource limits.  When a snapshot exceeds any limit the
    monitor fires registered cleanup callbacks.

    Set a field to ``None`` to disable that particular check.
    """
    max_memory_rss_mb: Optional[float] = None   # e.g. 512.0
    max_cpu_percent: Optional[float] = None     # e.g. 90.0
    max_async_tasks: Optional[int] = None       # e.g. 100

    def is_exceeded_by(self, snap: "ResourceSnapshot") -> bool:
        """Return True if *any* threshold is exceeded by the snapshot."""
        if self.max_memory_rss_mb is not None and snap.memory_rss_mb > self.max_memory_rss_mb:
            return True
        if self.max_cpu_percent is not None and snap.cpu_percent > self.max_cpu_percent:
            return True
        if self.max_async_tasks is not None and snap.active_async_tasks > self.max_async_tasks:
            return True
        return False

    def exceeded_fields(self, snap: "ResourceSnapshot") -> List[str]:
        """Return names of fields that are exceeded."""
        exceeded = []
        if self.max_memory_rss_mb is not None and snap.memory_rss_mb > self.max_memory_rss_mb:
            exceeded.append("memory_rss_mb")
        if self.max_cpu_percent is not None and snap.cpu_percent > self.max_cpu_percent:
            exceeded.append("cpu_percent")
        if self.max_async_tasks is not None and snap.active_async_tasks > self.max_async_tasks:
            exceeded.append("async_tasks")
        return exceeded


@dataclass
class ResourceSnapshot:
    """Point-in-time resource usage snapshot."""
    timestamp: float
    memory_rss_mb: float        # Resident Set Size
    memory_vms_mb: float        # Virtual Memory Size
    cpu_percent: float          # CPU usage %
    active_async_tasks: int     # asyncio tasks currently running
    thread_count: int           # OS threads

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "memory_rss_mb": round(self.memory_rss_mb, 2),
            "memory_vms_mb": round(self.memory_vms_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "active_async_tasks": self.active_async_tasks,
            "thread_count": self.thread_count,
        }


@dataclass
class ResourceUsageReport:
    """Aggregated resource usage over a measured interval."""
    duration_seconds: float
    peak_memory_rss_mb: float
    avg_memory_rss_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    peak_async_tasks: int
    avg_async_tasks: float
    peak_thread_count: int
    sample_count: int
    snapshots: List[ResourceSnapshot] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 3),
            "peak_memory_rss_mb": round(self.peak_memory_rss_mb, 2),
            "avg_memory_rss_mb": round(self.avg_memory_rss_mb, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "peak_async_tasks": self.peak_async_tasks,
            "avg_async_tasks": round(self.avg_async_tasks, 1),
            "peak_thread_count": self.peak_thread_count,
            "sample_count": self.sample_count,
        }


def _take_snapshot() -> ResourceSnapshot:
    """Capture current resource usage. Falls back gracefully if psutil unavailable."""
    now = time.time()

    # Count asyncio tasks (works even without psutil)
    try:
        loop = asyncio.get_running_loop()
        async_tasks = len([t for t in asyncio.all_tasks(loop) if not t.done()])
    except RuntimeError:
        async_tasks = 0

    thread_count = threading.active_count()

    if _PSUTIL_AVAILABLE:
        proc = psutil.Process()
        mem = proc.memory_info()
        # interval=None returns cached value (non-blocking)
        cpu = proc.cpu_percent(interval=None)
        return ResourceSnapshot(
            timestamp=now,
            memory_rss_mb=mem.rss / 1024 / 1024,
            memory_vms_mb=mem.vms / 1024 / 1024,
            cpu_percent=cpu,
            active_async_tasks=async_tasks,
            thread_count=thread_count,
        )

    # Fallback: zeros for memory/cpu when psutil not installed
    return ResourceSnapshot(
        timestamp=now,
        memory_rss_mb=0.0,
        memory_vms_mb=0.0,
        cpu_percent=0.0,
        active_async_tasks=async_tasks,
        thread_count=thread_count,
    )


def _build_report(snapshots: List[ResourceSnapshot], duration: float) -> ResourceUsageReport:
    """Aggregate a list of snapshots into a ResourceUsageReport."""
    if not snapshots:
        return ResourceUsageReport(
            duration_seconds=duration,
            peak_memory_rss_mb=0.0,
            avg_memory_rss_mb=0.0,
            peak_cpu_percent=0.0,
            avg_cpu_percent=0.0,
            peak_async_tasks=0,
            avg_async_tasks=0.0,
            peak_thread_count=0,
            sample_count=0,
        )

    n = len(snapshots)
    return ResourceUsageReport(
        duration_seconds=duration,
        peak_memory_rss_mb=max(s.memory_rss_mb for s in snapshots),
        avg_memory_rss_mb=sum(s.memory_rss_mb for s in snapshots) / n,
        peak_cpu_percent=max(s.cpu_percent for s in snapshots),
        avg_cpu_percent=sum(s.cpu_percent for s in snapshots) / n,
        peak_async_tasks=max(s.active_async_tasks for s in snapshots),
        avg_async_tasks=sum(s.active_async_tasks for s in snapshots) / n,
        peak_thread_count=max(s.thread_count for s in snapshots),
        sample_count=n,
        snapshots=snapshots,
    )


class ResourceUsageMonitor:
    """
    Monitors memory, CPU, and async task usage during verification operations.

    Supports both sync and async context managers for measuring a code block,
    plus a background polling mode for long-running operations.

    Task 5.2.5 – automatic cleanup:
    - Pass ``thresholds`` to enable threshold monitoring.
    - Register callbacks via ``add_cleanup_callback()``.  Built-in actions
      (GC_COLLECT, CLEAR_SNAPSHOTS) are always available; CUSTOM callbacks
      receive the offending ``ResourceSnapshot`` as their sole argument.
    - On context-manager exit (both sync and async) a final cleanup pass runs
      automatically via ``_run_cleanup()``.

    Usage (async context manager)::

        monitor = ResourceUsageMonitor(
            poll_interval=0.1,
            thresholds=ResourceThresholds(max_memory_rss_mb=512),
        )
        async with monitor.measure("verification_node") as report_ref:
            await run_verification()
        report = report_ref.report
        print(report.to_dict())

    Usage (sync context manager)::

        monitor = ResourceUsageMonitor()
        with monitor.measure_sync("price_check") as report_ref:
            run_price_check()
        report = report_ref.report

    Usage (manual)::

        monitor = ResourceUsageMonitor()
        monitor.start()
        # ... do work ...
        report = monitor.stop()
    """

    def __init__(
        self,
        poll_interval: float = 0.05,
        thresholds: Optional[ResourceThresholds] = None,
    ) -> None:
        """
        Args:
            poll_interval: Seconds between background samples (default 50ms).
            thresholds: Optional resource limits; when exceeded, cleanup
                        callbacks are fired automatically.
        """
        self.poll_interval = poll_interval
        self.thresholds = thresholds
        self._snapshots: List[ResourceSnapshot] = []
        self._start_time: Optional[float] = None
        self._polling_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._label: str = "unnamed"
        # Cleanup callbacks: list of (CleanupAction, optional callable)
        self._cleanup_callbacks: List[tuple] = []
        self._cleanup_count: int = 0  # number of times cleanup was triggered

    # ------------------------------------------------------------------
    # Cleanup API (Task 5.2.5)
    # ------------------------------------------------------------------

    def add_cleanup_callback(
        self,
        action: CleanupAction,
        callback: Optional[Callable[["ResourceSnapshot"], None]] = None,
    ) -> None:
        """
        Register a cleanup action.

        Args:
            action: One of the ``CleanupAction`` enum values.
            callback: Required when ``action == CleanupAction.CUSTOM``.
                      Receives the snapshot that triggered the cleanup.
        """
        if action == CleanupAction.CUSTOM and callback is None:
            raise ValueError("callback is required for CleanupAction.CUSTOM")
        self._cleanup_callbacks.append((action, callback))

    def _run_cleanup(self, snap: Optional[ResourceSnapshot] = None) -> None:
        """
        Execute all registered cleanup actions (threshold-triggered).

        Called automatically when a polled snapshot exceeds configured thresholds.
        Fires all registered callbacks including CUSTOM ones.
        """
        if not self._cleanup_callbacks:
            gc.collect()
            return

        for action, callback in self._cleanup_callbacks:
            try:
                if action == CleanupAction.GC_COLLECT:
                    gc.collect()
                elif action == CleanupAction.CLEAR_SNAPSHOTS:
                    self._snapshots.clear()
                elif action == CleanupAction.CUSTOM and callback is not None:
                    callback(snap)
            except Exception as exc:  # pragma: no cover
                logger.warning("Cleanup callback %s raised: %s", action, exc)

        self._cleanup_count += 1

    def _run_session_end_cleanup(self, snap: Optional[ResourceSnapshot] = None) -> None:
        """
        Execute session-end cleanup (called on context manager exit / stop).

        Only runs GC_COLLECT and CLEAR_SNAPSHOTS actions — CUSTOM callbacks are
        NOT fired here, as they are reserved for threshold-triggered events.
        If no callbacks are registered, a default gc.collect() is performed.
        """
        if not self._cleanup_callbacks:
            gc.collect()
            return

        for action, callback in self._cleanup_callbacks:
            try:
                if action == CleanupAction.GC_COLLECT:
                    gc.collect()
                elif action == CleanupAction.CLEAR_SNAPSHOTS:
                    self._snapshots.clear()
                # CUSTOM callbacks are intentionally skipped on session end
            except Exception as exc:  # pragma: no cover
                logger.warning("Session-end cleanup %s raised: %s", action, exc)

    def _check_thresholds(self, snap: ResourceSnapshot) -> None:
        """Fire cleanup if thresholds are configured and exceeded."""
        if self.thresholds is None:
            return
        if self.thresholds.is_exceeded_by(snap):
            exceeded = self.thresholds.exceeded_fields(snap)
            logger.warning(
                "Resource thresholds exceeded [%s=%s]: %s",
                self._label,
                exceeded,
                snap.to_dict(),
            )
            self._run_cleanup(snap)

    # ------------------------------------------------------------------
    # Manual start / stop API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Take an initial snapshot and record start time (sync, no polling)."""
        self._snapshots = []
        self._start_time = time.perf_counter()
        snap = _take_snapshot()
        self._snapshots.append(snap)
        self._check_thresholds(snap)

    def stop(self) -> ResourceUsageReport:
        """Take a final snapshot, run session-end cleanup, and return the aggregated report."""
        snap = _take_snapshot()
        self._snapshots.append(snap)
        self._check_thresholds(snap)
        duration = time.perf_counter() - (self._start_time or 0.0)
        report = _build_report(self._snapshots, duration)
        # Session-end cleanup: GC/clear only, no CUSTOM callbacks
        self._run_session_end_cleanup(snap)
        return report

    # ------------------------------------------------------------------
    # Async background polling
    # ------------------------------------------------------------------

    async def start_async(self) -> None:
        """Start async background polling."""
        self._snapshots = []
        self._start_time = time.perf_counter()
        self._stop_event = asyncio.Event()
        self._polling_task = asyncio.create_task(self._poll_loop())

    async def stop_async(self) -> ResourceUsageReport:
        """Stop async polling, run session-end cleanup, and return the aggregated report."""
        if self._stop_event:
            self._stop_event.set()
        if self._polling_task:
            await self._polling_task
        duration = time.perf_counter() - (self._start_time or 0.0)
        report = _build_report(self._snapshots, duration)
        # Session-end cleanup: GC/clear only, no CUSTOM callbacks
        snap = _take_snapshot()
        self._run_session_end_cleanup(snap)
        return report

    async def _poll_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            snap = _take_snapshot()
            self._snapshots.append(snap)
            self._check_thresholds(snap)
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self.poll_interval,
                )
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def measure(self, label: str = "operation"):
        """Async context manager with background polling and automatic cleanup."""
        self._label = label

        class _Ref:
            report: Optional[ResourceUsageReport] = None

        ref = _Ref()
        await self.start_async()
        try:
            yield ref
        finally:
            ref.report = await self.stop_async()
            # Fire full cleanup (including CUSTOM callbacks) on context manager exit
            snap = _take_snapshot()
            self._run_cleanup(snap)

    @contextmanager
    def measure_sync(self, label: str = "operation"):
        """Sync context manager (start/stop snapshots only, no polling) with automatic cleanup."""
        self._label = label

        class _Ref:
            report: Optional[ResourceUsageReport] = None

        ref = _Ref()
        self.start()
        try:
            yield ref
        finally:
            ref.report = self.stop()
            # Fire full cleanup (including CUSTOM callbacks) on context manager exit
            snap = _take_snapshot()
            self._run_cleanup(snap)

    # ------------------------------------------------------------------
    # Convenience: single snapshot
    # ------------------------------------------------------------------

    @staticmethod
    def snapshot() -> ResourceSnapshot:
        """Take a single point-in-time resource snapshot."""
        return _take_snapshot()


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_global_monitor: Optional[ResourceUsageMonitor] = None


def get_resource_monitor(poll_interval: float = 0.05) -> ResourceUsageMonitor:
    """Return (or create) the module-level ResourceUsageMonitor singleton."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceUsageMonitor(poll_interval=poll_interval)
    return _global_monitor


def reset_resource_monitor() -> None:
    """Reset the module-level singleton (useful in tests)."""
    global _global_monitor
    _global_monitor = None


__all__ = [
    "CleanupAction",
    "ResourceThresholds",
    "ResourceSnapshot",
    "ResourceUsageReport",
    "ResourceUsageMonitor",
    "get_resource_monitor",
    "reset_resource_monitor",
]
