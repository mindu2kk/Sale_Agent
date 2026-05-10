"""
Performance Monitoring Utilities

Performance tracking và optimization helpers:
- Execution time measurement
- Memory usage monitoring  
- Token usage tracking
- Cache performance metrics
- Async step latency tracking (Task 5.3.1)
"""

import time
import asyncio
import functools
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "tokens_used": self.tokens_used,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls": self.api_calls,
            "cache_hit_rate": self.cache_hit_rate,
            "efficiency_score": self.efficiency_score
        }
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (lower is better)"""
        if self.tokens_used == 0:
            return 0.0
        return self.execution_time / self.tokens_used * 1000  # ms per 1k tokens


class PerformanceMonitor:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._start_time: Optional[float] = None
    
    def start_timing(self):
        """Start timing measurement"""
        self._start_time = time.time()
    
    def stop_timing(self) -> float:
        """Stop timing and return duration"""
        if self._start_time is None:
            return 0.0
        
        duration = time.time() - self._start_time
        self.metrics.execution_time += duration
        self._start_time = None
        return duration
    
    def record_tokens(self, tokens: int):
        """Record token usage"""
        self.metrics.tokens_used += tokens
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics.cache_misses += 1
    
    def record_api_call(self):
        """Record API call"""
        self.metrics.api_calls += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics"""
        return self.metrics
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = PerformanceMetrics()
        self._start_time = None


@contextmanager
def measure_execution_time():
    """Context manager to measure execution time"""
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        return execution_time


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        monitor.start_timing()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            monitor.stop_timing()
            
            # Add metrics to result if it's a dict
            if isinstance(result, dict):
                result["_performance_metrics"] = monitor.get_metrics().to_dict()
    
    return wrapper


@dataclass
class StepLatencyRecord:
    """Latency record for a single async verification step"""
    step_name: str
    start_time: float          # Unix timestamp (seconds)
    end_time: float            # Unix timestamp (seconds)
    duration_ms: float         # Duration in milliseconds
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage in ExecutionStep.metrics"""
        return {
            "step_name": self.step_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 3),
            "success": self.success,
            "error": self.error,
        }


class AsyncStepLatencyTracker:
    """
    Tracks per-step latency for async verification operations.

    Usage as async context manager:
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            result = await run_price_check()
        record = tracker.get_record("price_check")

    Usage as decorator:
        tracker = AsyncStepLatencyTracker()

        @tracker.track_async("price_check")
        async def run_price_check():
            ...

    After all steps complete, call ``get_all_metrics()`` to retrieve a dict
    suitable for storing in ``ExecutionStep.metrics`` or ``WorkflowMetrics``.
    """

    def __init__(self) -> None:
        self._records: Dict[str, StepLatencyRecord] = {}

    @asynccontextmanager
    async def track(self, step_name: str):
        """
        Async context manager that measures latency for a named step.

        Records start_time, end_time, and duration_ms.  On exception the
        record is still saved with success=False and the error message.
        """
        start = time.perf_counter()
        start_wall = time.time()
        error_msg: Optional[str] = None
        success = True
        try:
            yield
        except Exception as exc:
            success = False
            error_msg = str(exc)
            raise
        finally:
            end = time.perf_counter()
            end_wall = time.time()
            duration_ms = (end - start) * 1000.0
            self._records[step_name] = StepLatencyRecord(
                step_name=step_name,
                start_time=start_wall,
                end_time=end_wall,
                duration_ms=duration_ms,
                success=success,
                error=error_msg,
            )

    def track_async(self, step_name: str) -> Callable:
        """
        Decorator that wraps an async function with latency tracking.

        Example::

            tracker = AsyncStepLatencyTracker()

            @tracker.track_async("price_check")
            async def check_price(state):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                async with self.track(step_name):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    def get_record(self, step_name: str) -> Optional[StepLatencyRecord]:
        """Return the latency record for a specific step, or None if not tracked."""
        return self._records.get(step_name)

    def get_all_records(self) -> List[StepLatencyRecord]:
        """Return all recorded latency records in insertion order."""
        return list(self._records.values())

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Return a flat metrics dict suitable for ``ExecutionStep.metrics``.

        Keys follow the pattern ``step_latency_{step_name}_ms`` for easy
        querying, plus an ``aggregate_latency`` sub-dict with summary stats.

        Example output::

            {
                "step_latency_price_check_ms": 123.4,
                "step_latency_policy_check_ms": 98.7,
                "step_latency_relevance_check_ms": 210.1,
                "step_latencies": {
                    "price_check": {"step_name": ..., "duration_ms": 123.4, ...},
                    ...
                },
                "aggregate_latency": {
                    "total_ms": 432.2,
                    "max_ms": 210.1,
                    "min_ms": 98.7,
                    "avg_ms": 144.1,
                    "step_count": 3,
                },
            }
        """
        metrics: Dict[str, Any] = {}
        step_details: Dict[str, Any] = {}

        durations: List[float] = []
        for record in self._records.values():
            key = f"step_latency_{record.step_name}_ms"
            metrics[key] = round(record.duration_ms, 3)
            step_details[record.step_name] = record.to_dict()
            durations.append(record.duration_ms)

        metrics["step_latencies"] = step_details

        if durations:
            metrics["aggregate_latency"] = {
                "total_ms": round(sum(durations), 3),
                "max_ms": round(max(durations), 3),
                "min_ms": round(min(durations), 3),
                "avg_ms": round(sum(durations) / len(durations), 3),
                "step_count": len(durations),
            }
        else:
            metrics["aggregate_latency"] = {
                "total_ms": 0.0,
                "max_ms": 0.0,
                "min_ms": 0.0,
                "avg_ms": 0.0,
                "step_count": 0,
            }

        return metrics

    def reset(self) -> None:
        """Clear all recorded latency data."""
        self._records.clear()


# ---------------------------------------------------------------------------
# Task 5.3.2: Throughput monitoring for concurrent binary workflows
# ---------------------------------------------------------------------------

import asyncio
import collections
from typing import Deque, Tuple


@dataclass
class ThroughputSnapshot:
    """Throughput metrics snapshot for a given time window."""
    window_seconds: float
    total_completed: int
    pass_count: int
    fail_count: int
    wps: float                  # workflows per second
    pass_wps: float             # PASS workflows per second
    fail_wps: float             # FAIL workflows per second
    concurrent_workflows: int   # active at snapshot time
    queue_depth: int            # pending / queued workflows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "total_completed": self.total_completed,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "wps": round(self.wps, 4),
            "pass_wps": round(self.pass_wps, 4),
            "fail_wps": round(self.fail_wps, 4),
            "concurrent_workflows": self.concurrent_workflows,
            "queue_depth": self.queue_depth,
        }


class ThroughputMonitor:
    """
    Async-compatible throughput monitor for concurrent binary verification workflows.

    Tracks:
    - Workflows per second (WPS) over sliding time windows (1s, 10s, 60s)
    - Concurrent (active) workflow count
    - Queue depth / pending workflows
    - Binary PASS vs FAIL throughput rates per time window

    Usage::

        monitor = ThroughputMonitor()

        # When a workflow starts:
        monitor.workflow_started()

        # When a workflow finishes (binary result):
        monitor.workflow_completed(passed=True)

        # Increment queue depth when a workflow is enqueued:
        monitor.workflow_enqueued()
        monitor.workflow_dequeued()

        # Get snapshot for a specific window:
        snap = monitor.snapshot(window_seconds=10)
    """

    # Default sliding windows in seconds
    DEFAULT_WINDOWS: Tuple[float, ...] = (1.0, 10.0, 60.0)

    def __init__(self, windows: Tuple[float, ...] = DEFAULT_WINDOWS) -> None:
        self.windows = windows
        # Each completion event stored as (timestamp, passed: bool)
        self._completions: Deque[Tuple[float, bool]] = collections.deque()
        self._concurrent: int = 0   # active workflows right now
        self._queue_depth: int = 0  # pending / queued workflows
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Sync-safe helpers (no await needed for simple counters)
    # ------------------------------------------------------------------

    def workflow_started(self) -> None:
        """Call when a workflow begins execution (leaves queue, enters active)."""
        self._concurrent += 1

    def workflow_completed(self, passed: bool) -> None:
        """
        Call when a workflow finishes.

        Args:
            passed: True if the binary verification result was PASS, False for FAIL.
        """
        ts = time.time()
        self._completions.append((ts, passed))
        if self._concurrent > 0:
            self._concurrent -= 1
        # Prune old events beyond the largest window
        self._prune(ts)

    def workflow_enqueued(self) -> None:
        """Call when a workflow is added to the pending queue."""
        self._queue_depth += 1

    def workflow_dequeued(self) -> None:
        """Call when a workflow leaves the pending queue (starts or is cancelled)."""
        if self._queue_depth > 0:
            self._queue_depth -= 1

    # ------------------------------------------------------------------
    # Async variants (acquire lock for thread-safe async usage)
    # ------------------------------------------------------------------

    async def async_workflow_started(self) -> None:
        """Async-safe version of workflow_started."""
        async with self._lock:
            self.workflow_started()

    async def async_workflow_completed(self, passed: bool) -> None:
        """Async-safe version of workflow_completed."""
        async with self._lock:
            self.workflow_completed(passed)

    async def async_workflow_enqueued(self) -> None:
        async with self._lock:
            self.workflow_enqueued()

    async def async_workflow_dequeued(self) -> None:
        async with self._lock:
            self.workflow_dequeued()

    # ------------------------------------------------------------------
    # Snapshot / metrics
    # ------------------------------------------------------------------

    def snapshot(self, window_seconds: float = 60.0) -> ThroughputSnapshot:
        """
        Return a throughput snapshot for the given sliding window.

        Args:
            window_seconds: Length of the sliding window in seconds.

        Returns:
            ThroughputSnapshot with WPS, pass/fail rates, concurrency, queue depth.
        """
        now = time.time()
        cutoff = now - window_seconds

        total = 0
        passes = 0
        fails = 0
        for ts, passed in self._completions:
            if ts >= cutoff:
                total += 1
                if passed:
                    passes += 1
                else:
                    fails += 1

        wps = total / window_seconds if window_seconds > 0 else 0.0
        pass_wps = passes / window_seconds if window_seconds > 0 else 0.0
        fail_wps = fails / window_seconds if window_seconds > 0 else 0.0

        return ThroughputSnapshot(
            window_seconds=window_seconds,
            total_completed=total,
            pass_count=passes,
            fail_count=fails,
            wps=wps,
            pass_wps=pass_wps,
            fail_wps=fail_wps,
            concurrent_workflows=self._concurrent,
            queue_depth=self._queue_depth,
        )

    def all_snapshots(self) -> Dict[str, ThroughputSnapshot]:
        """
        Return snapshots for all configured windows.

        Returns:
            Dict keyed by ``"window_{n}s"`` (e.g. ``"window_1s"``, ``"window_10s"``).
        """
        return {
            f"window_{int(w)}s": self.snapshot(w)
            for w in self.windows
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return a flat metrics dict suitable for ``ExecutionStep.metrics``.

        Includes per-window WPS, pass/fail rates, concurrency, and queue depth.
        """
        metrics: Dict[str, Any] = {
            "concurrent_workflows": self._concurrent,
            "queue_depth": self._queue_depth,
        }
        for w in self.windows:
            snap = self.snapshot(w)
            key = f"throughput_{int(w)}s"
            metrics[key] = snap.to_dict()
        return metrics

    @property
    def concurrent_count(self) -> int:
        """Current number of active (concurrent) workflows."""
        return self._concurrent

    @property
    def queue_depth(self) -> int:
        """Current number of pending / queued workflows."""
        return self._queue_depth

    def reset(self) -> None:
        """Clear all recorded data."""
        self._completions.clear()
        self._concurrent = 0
        self._queue_depth = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        """Remove completion events older than the largest configured window."""
        if not self.windows:
            return
        max_window = max(self.windows)
        cutoff = now - max_window
        while self._completions and self._completions[0][0] < cutoff:
            self._completions.popleft()
