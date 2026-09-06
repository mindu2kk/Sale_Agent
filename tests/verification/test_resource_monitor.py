"""
Tests for ResourceUsageMonitor - Task 5.3.3 / 5.2.5

Covers:
- ResourceSnapshot structure and serialization
- ResourceUsageReport aggregation
- Sync context manager (measure_sync)
- Async context manager with background polling (measure)
- Manual start/stop API
- Module-level singleton helpers
- Graceful fallback when psutil is unavailable
- Automatic cleanup on threshold exceeded (Task 5.2.5)
- Automatic cleanup on session end (Task 5.2.5)
"""

import asyncio
import gc
import time
import pytest
from unittest.mock import patch, MagicMock, call

from backend.verification.utils.resource_monitor import (
    CleanupAction,
    ResourceThresholds,
    ResourceSnapshot,
    ResourceUsageReport,
    ResourceUsageMonitor,
    get_resource_monitor,
    reset_resource_monitor,
    _take_snapshot,
    _build_report,
)


# ---------------------------------------------------------------------------
# ResourceSnapshot
# ---------------------------------------------------------------------------

class TestResourceSnapshot:
    def test_to_dict_keys(self):
        snap = ResourceSnapshot(
            timestamp=1000.0,
            memory_rss_mb=128.5,
            memory_vms_mb=256.0,
            cpu_percent=12.3,
            active_async_tasks=3,
            thread_count=5,
        )
        d = snap.to_dict()
        assert set(d.keys()) == {
            "timestamp", "memory_rss_mb", "memory_vms_mb",
            "cpu_percent", "active_async_tasks", "thread_count",
        }

    def test_to_dict_values_rounded(self):
        snap = ResourceSnapshot(
            timestamp=1000.0,
            memory_rss_mb=128.123456,
            memory_vms_mb=256.789,
            cpu_percent=12.3456,
            active_async_tasks=2,
            thread_count=4,
        )
        d = snap.to_dict()
        assert d["memory_rss_mb"] == round(128.123456, 2)
        assert d["cpu_percent"] == round(12.3456, 2)


# ---------------------------------------------------------------------------
# _build_report
# ---------------------------------------------------------------------------

class TestBuildReport:
    def _make_snap(self, mem=100.0, cpu=10.0, tasks=2, threads=4):
        return ResourceSnapshot(
            timestamp=time.time(),
            memory_rss_mb=mem,
            memory_vms_mb=mem * 2,
            cpu_percent=cpu,
            active_async_tasks=tasks,
            thread_count=threads,
        )

    def test_empty_snapshots(self):
        report = _build_report([], 1.0)
        assert report.sample_count == 0
        assert report.peak_memory_rss_mb == 0.0
        assert report.peak_cpu_percent == 0.0

    def test_aggregation(self):
        snaps = [
            self._make_snap(mem=100.0, cpu=10.0, tasks=1),
            self._make_snap(mem=200.0, cpu=50.0, tasks=5),
            self._make_snap(mem=150.0, cpu=30.0, tasks=3),
        ]
        report = _build_report(snaps, 3.0)
        assert report.peak_memory_rss_mb == 200.0
        assert report.avg_memory_rss_mb == pytest.approx(150.0)
        assert report.peak_cpu_percent == 50.0
        assert report.avg_cpu_percent == pytest.approx(30.0)
        assert report.peak_async_tasks == 5
        assert report.sample_count == 3

    def test_to_dict_keys(self):
        snaps = [self._make_snap()]
        report = _build_report(snaps, 1.0)
        d = report.to_dict()
        expected_keys = {
            "duration_seconds", "peak_memory_rss_mb", "avg_memory_rss_mb",
            "peak_cpu_percent", "avg_cpu_percent", "peak_async_tasks",
            "avg_async_tasks", "peak_thread_count", "sample_count",
        }
        assert expected_keys.issubset(d.keys())


# ---------------------------------------------------------------------------
# ResourceUsageMonitor - sync
# ---------------------------------------------------------------------------

class TestResourceUsageMonitorSync:
    def test_manual_start_stop(self):
        monitor = ResourceUsageMonitor()
        monitor.start()
        time.sleep(0.01)
        report = monitor.stop()
        assert isinstance(report, ResourceUsageReport)
        assert report.sample_count >= 2  # start + stop snapshots
        assert report.duration_seconds >= 0.0

    def test_measure_sync_context_manager(self):
        monitor = ResourceUsageMonitor()
        with monitor.measure_sync("test_op") as ref:
            time.sleep(0.01)
        assert ref.report is not None
        assert isinstance(ref.report, ResourceUsageReport)
        assert ref.report.sample_count >= 2

    def test_measure_sync_report_to_dict(self):
        monitor = ResourceUsageMonitor()
        with monitor.measure_sync("price_check") as ref:
            pass
        d = ref.report.to_dict()
        assert "peak_memory_rss_mb" in d
        assert "peak_cpu_percent" in d
        assert "peak_async_tasks" in d


# ---------------------------------------------------------------------------
# ResourceUsageMonitor - async
# ---------------------------------------------------------------------------

class TestResourceUsageMonitorAsync:
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        monitor = ResourceUsageMonitor(poll_interval=0.02)
        async with monitor.measure("async_op") as ref:
            await asyncio.sleep(0.05)
        assert ref.report is not None
        assert ref.report.sample_count >= 2
        assert ref.report.duration_seconds >= 0.04

    @pytest.mark.asyncio
    async def test_async_tracks_tasks(self):
        """active_async_tasks should be > 0 while tasks are running."""
        monitor = ResourceUsageMonitor(poll_interval=0.01)

        async def dummy():
            await asyncio.sleep(0.1)

        task = asyncio.create_task(dummy())
        async with monitor.measure("task_tracking") as ref:
            await asyncio.sleep(0.05)
        await task

        # At least one snapshot should have seen the running task
        assert ref.report is not None
        assert ref.report.peak_async_tasks >= 1

    @pytest.mark.asyncio
    async def test_async_start_stop_manual(self):
        monitor = ResourceUsageMonitor(poll_interval=0.02)
        await monitor.start_async()
        await asyncio.sleep(0.06)
        report = await monitor.stop_async()
        assert report.sample_count >= 2
        assert report.duration_seconds >= 0.05

    @pytest.mark.asyncio
    async def test_report_to_dict_integration(self):
        monitor = ResourceUsageMonitor(poll_interval=0.02)
        async with monitor.measure("verification_node") as ref:
            await asyncio.sleep(0.04)
        d = ref.report.to_dict()
        assert isinstance(d["peak_memory_rss_mb"], float)
        assert isinstance(d["peak_async_tasks"], int)
        assert d["sample_count"] >= 2


# ---------------------------------------------------------------------------
# Static snapshot helper
# ---------------------------------------------------------------------------

class TestTakeSnapshot:
    def test_snapshot_fields(self):
        snap = _take_snapshot()
        assert snap.timestamp > 0
        assert snap.memory_rss_mb >= 0.0
        assert snap.cpu_percent >= 0.0
        assert snap.active_async_tasks >= 0
        assert snap.thread_count >= 1

    def test_snapshot_without_psutil(self):
        """Should return zeros for memory/cpu but still track tasks/threads."""
        with patch("backend.verification.utils.resource_monitor._PSUTIL_AVAILABLE", False):
            snap = _take_snapshot()
        assert snap.memory_rss_mb == 0.0
        assert snap.cpu_percent == 0.0
        assert snap.thread_count >= 1

    @pytest.mark.asyncio
    async def test_snapshot_counts_async_tasks(self):
        async def dummy():
            await asyncio.sleep(0.5)

        task = asyncio.create_task(dummy())
        snap = _take_snapshot()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The running task should have been counted
        assert snap.active_async_tasks >= 1


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    def setup_method(self):
        reset_resource_monitor()

    def test_get_resource_monitor_returns_instance(self):
        monitor = get_resource_monitor()
        assert isinstance(monitor, ResourceUsageMonitor)

    def test_get_resource_monitor_singleton(self):
        m1 = get_resource_monitor()
        m2 = get_resource_monitor()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_resource_monitor()
        reset_resource_monitor()
        m2 = get_resource_monitor()
        assert m1 is not m2


# ---------------------------------------------------------------------------
# ResourceThresholds (Task 5.2.5)
# ---------------------------------------------------------------------------

class TestResourceThresholds:
    def _snap(self, mem=100.0, cpu=10.0, tasks=2):
        return ResourceSnapshot(
            timestamp=time.time(),
            memory_rss_mb=mem,
            memory_vms_mb=mem * 2,
            cpu_percent=cpu,
            active_async_tasks=tasks,
            thread_count=4,
        )

    def test_no_threshold_never_exceeded(self):
        thresholds = ResourceThresholds()
        snap = self._snap(mem=9999.0, cpu=100.0, tasks=1000)
        assert not thresholds.is_exceeded_by(snap)

    def test_memory_threshold_exceeded(self):
        thresholds = ResourceThresholds(max_memory_rss_mb=200.0)
        assert thresholds.is_exceeded_by(self._snap(mem=201.0))
        assert not thresholds.is_exceeded_by(self._snap(mem=199.0))

    def test_cpu_threshold_exceeded(self):
        thresholds = ResourceThresholds(max_cpu_percent=80.0)
        assert thresholds.is_exceeded_by(self._snap(cpu=81.0))
        assert not thresholds.is_exceeded_by(self._snap(cpu=79.0))

    def test_async_tasks_threshold_exceeded(self):
        thresholds = ResourceThresholds(max_async_tasks=5)
        assert thresholds.is_exceeded_by(self._snap(tasks=6))
        assert not thresholds.is_exceeded_by(self._snap(tasks=5))

    def test_exceeded_fields_returns_correct_names(self):
        thresholds = ResourceThresholds(max_memory_rss_mb=50.0, max_cpu_percent=5.0)
        snap = self._snap(mem=100.0, cpu=10.0)
        fields = thresholds.exceeded_fields(snap)
        assert "memory_rss_mb" in fields
        assert "cpu_percent" in fields

    def test_exceeded_fields_empty_when_within_limits(self):
        thresholds = ResourceThresholds(max_memory_rss_mb=500.0, max_cpu_percent=90.0)
        snap = self._snap(mem=100.0, cpu=10.0)
        assert thresholds.exceeded_fields(snap) == []


# ---------------------------------------------------------------------------
# Automatic cleanup – callbacks (Task 5.2.5)
# ---------------------------------------------------------------------------

class TestAutomaticCleanupCallbacks:
    def test_add_gc_collect_callback(self):
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.GC_COLLECT)
        assert len(monitor._cleanup_callbacks) == 1

    def test_add_clear_snapshots_callback(self):
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.CLEAR_SNAPSHOTS)
        assert len(monitor._cleanup_callbacks) == 1

    def test_add_custom_callback_requires_callable(self):
        monitor = ResourceUsageMonitor()
        with pytest.raises(ValueError):
            monitor.add_cleanup_callback(CleanupAction.CUSTOM)

    def test_add_custom_callback_with_callable(self):
        monitor = ResourceUsageMonitor()
        called_with = []
        monitor.add_cleanup_callback(CleanupAction.CUSTOM, lambda snap: called_with.append(snap))
        assert len(monitor._cleanup_callbacks) == 1

    def test_custom_callback_receives_snapshot(self):
        received = []
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.CUSTOM, lambda snap: received.append(snap))
        snap = ResourceSnapshot(
            timestamp=time.time(), memory_rss_mb=10.0, memory_vms_mb=20.0,
            cpu_percent=5.0, active_async_tasks=0, thread_count=1,
        )
        monitor._run_cleanup(snap)
        assert len(received) == 1
        assert received[0] is snap

    def test_run_cleanup_increments_count(self):
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.GC_COLLECT)
        assert monitor._cleanup_count == 0
        monitor._run_cleanup()
        assert monitor._cleanup_count == 1
        monitor._run_cleanup()
        assert monitor._cleanup_count == 2

    def test_clear_snapshots_action_empties_list(self):
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.CLEAR_SNAPSHOTS)
        monitor._snapshots = [
            ResourceSnapshot(time.time(), 10.0, 20.0, 5.0, 0, 1)
            for _ in range(5)
        ]
        monitor._run_cleanup()
        assert monitor._snapshots == []

    def test_gc_collect_action_runs(self):
        """GC_COLLECT should not raise and should call gc.collect."""
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.GC_COLLECT)
        with patch("gc.collect") as mock_gc:
            monitor._run_cleanup()
        mock_gc.assert_called_once()


# ---------------------------------------------------------------------------
# Automatic cleanup – threshold-triggered (Task 5.2.5)
# ---------------------------------------------------------------------------

class TestThresholdTriggeredCleanup:
    def test_cleanup_fires_when_memory_threshold_exceeded(self):
        fired = []
        thresholds = ResourceThresholds(max_memory_rss_mb=1.0)  # very low → always exceeded
        monitor = ResourceUsageMonitor(thresholds=thresholds)
        monitor.add_cleanup_callback(CleanupAction.CUSTOM, lambda snap: fired.append(snap))

        monitor.start()
        monitor.stop()

        # At least one threshold-triggered cleanup should have fired
        assert len(fired) >= 1

    def test_cleanup_not_fired_when_within_threshold(self):
        fired = []
        thresholds = ResourceThresholds(max_memory_rss_mb=999999.0)  # very high → never exceeded
        monitor = ResourceUsageMonitor(thresholds=thresholds)
        monitor.add_cleanup_callback(CleanupAction.CUSTOM, lambda snap: fired.append(snap))

        monitor.start()
        # _cleanup_count starts at 0; threshold not exceeded so no threshold-triggered cleanup
        # (session-end cleanup still runs via stop())
        monitor.stop()

        # No threshold-triggered cleanups (only session-end cleanup)
        assert len(fired) == 0

    def test_check_thresholds_calls_run_cleanup(self):
        thresholds = ResourceThresholds(max_memory_rss_mb=0.001)  # always exceeded
        monitor = ResourceUsageMonitor(thresholds=thresholds)
        monitor.add_cleanup_callback(CleanupAction.GC_COLLECT)

        snap = ResourceSnapshot(
            timestamp=time.time(), memory_rss_mb=100.0, memory_vms_mb=200.0,
            cpu_percent=5.0, active_async_tasks=0, thread_count=1,
        )
        initial_count = monitor._cleanup_count
        monitor._check_thresholds(snap)
        assert monitor._cleanup_count == initial_count + 1

    def test_check_thresholds_no_op_when_no_thresholds(self):
        monitor = ResourceUsageMonitor(thresholds=None)
        monitor.add_cleanup_callback(CleanupAction.GC_COLLECT)
        snap = ResourceSnapshot(
            timestamp=time.time(), memory_rss_mb=9999.0, memory_vms_mb=9999.0,
            cpu_percent=100.0, active_async_tasks=100, thread_count=1,
        )
        monitor._check_thresholds(snap)
        # No threshold configured → cleanup_count stays 0
        assert monitor._cleanup_count == 0


# ---------------------------------------------------------------------------
# Automatic cleanup – session end (Task 5.2.5)
# ---------------------------------------------------------------------------

class TestSessionEndCleanup:
    def test_sync_stop_triggers_cleanup(self):
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.GC_COLLECT)
        monitor.start()
        with patch("gc.collect") as mock_gc:
            monitor.stop()
        mock_gc.assert_called()

    def test_sync_context_manager_triggers_cleanup_on_exit(self):
        cleanup_called = []
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.CUSTOM, lambda snap: cleanup_called.append(True))

        with monitor.measure_sync("test"):
            pass

        assert len(cleanup_called) >= 1

    def test_sync_context_manager_cleanup_on_exception(self):
        """Cleanup must run even when the body raises."""
        cleanup_called = []
        monitor = ResourceUsageMonitor()
        monitor.add_cleanup_callback(CleanupAction.CUSTOM, lambda snap: cleanup_called.append(True))

        with pytest.raises(ValueError):
            with monitor.measure_sync("test"):
                raise ValueError("intentional")

        assert len(cleanup_called) >= 1

    @pytest.mark.asyncio
    async def test_async_stop_triggers_cleanup(self):
        monitor = ResourceUsageMonitor(poll_interval=0.02)
        monitor.add_cleanup_callback(CleanupAction.GC_COLLECT)
        await monitor.start_async()
        await asyncio.sleep(0.03)
        with patch("gc.collect") as mock_gc:
            await monitor.stop_async()
        mock_gc.assert_called()

    @pytest.mark.asyncio
    async def test_async_context_manager_triggers_cleanup_on_exit(self):
        cleanup_called = []
        monitor = ResourceUsageMonitor(poll_interval=0.02)
        monitor.add_cleanup_callback(CleanupAction.CUSTOM, lambda snap: cleanup_called.append(True))

        async with monitor.measure("async_test"):
            await asyncio.sleep(0.03)

        assert len(cleanup_called) >= 1

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup_on_exception(self):
        """Cleanup must run even when the async body raises."""
        cleanup_called = []
        monitor = ResourceUsageMonitor(poll_interval=0.02)
        monitor.add_cleanup_callback(CleanupAction.CUSTOM, lambda snap: cleanup_called.append(True))

        with pytest.raises(RuntimeError):
            async with monitor.measure("async_test"):
                raise RuntimeError("intentional")

        assert len(cleanup_called) >= 1

    def test_default_cleanup_runs_gc_even_without_callbacks(self):
        """Without explicit callbacks, stop() still runs gc.collect() by default."""
        monitor = ResourceUsageMonitor()
        monitor.start()
        with patch("gc.collect") as mock_gc:
            monitor.stop()
        mock_gc.assert_called()
