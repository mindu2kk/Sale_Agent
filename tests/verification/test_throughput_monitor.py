"""
Tests for Task 5.3.2: Throughput monitoring for concurrent binary workflows.

Covers:
- ThroughputMonitor tracks workflows per second (WPS) over sliding windows.
- Concurrent workflow count increments on start, decrements on completion.
- Queue depth increments on enqueue, decrements on dequeue.
- Binary PASS vs FAIL throughput rates are tracked separately.
- Sliding time windows (1s, 10s, 60s) return correct counts.
- Async-compatible methods work correctly with asyncio.
- ThroughputSnapshot.to_dict() returns all required fields.
- get_metrics() returns flat dict with per-window data.
- reset() clears all state.
"""

import asyncio
import time
import pytest

from verification.utils.performance import ThroughputMonitor, ThroughputSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs) -> ThroughputMonitor:
    return ThroughputMonitor(**kwargs)


# ---------------------------------------------------------------------------
# ThroughputSnapshot.to_dict()
# ---------------------------------------------------------------------------

class TestThroughputSnapshotToDict:
    def test_to_dict_has_all_required_fields(self):
        snap = ThroughputSnapshot(
            window_seconds=10.0,
            total_completed=5,
            pass_count=3,
            fail_count=2,
            wps=0.5,
            pass_wps=0.3,
            fail_wps=0.2,
            concurrent_workflows=2,
            queue_depth=1,
        )
        d = snap.to_dict()
        for key in ("window_seconds", "total_completed", "pass_count", "fail_count",
                    "wps", "pass_wps", "fail_wps", "concurrent_workflows", "queue_depth"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_rounds_wps(self):
        snap = ThroughputSnapshot(
            window_seconds=10.0,
            total_completed=1,
            pass_count=1,
            fail_count=0,
            wps=0.123456789,
            pass_wps=0.123456789,
            fail_wps=0.0,
            concurrent_workflows=0,
            queue_depth=0,
        )
        d = snap.to_dict()
        assert d["wps"] == round(0.123456789, 4)
        assert d["pass_wps"] == round(0.123456789, 4)


# ---------------------------------------------------------------------------
# Concurrent workflow count
# ---------------------------------------------------------------------------

class TestConcurrentWorkflowCount:
    def test_starts_at_zero(self):
        m = _make_monitor()
        assert m.concurrent_count == 0

    def test_increments_on_start(self):
        m = _make_monitor()
        m.workflow_started()
        assert m.concurrent_count == 1

    def test_decrements_on_completion(self):
        m = _make_monitor()
        m.workflow_started()
        m.workflow_completed(passed=True)
        assert m.concurrent_count == 0

    def test_multiple_concurrent_workflows(self):
        m = _make_monitor()
        m.workflow_started()
        m.workflow_started()
        m.workflow_started()
        assert m.concurrent_count == 3

    def test_concurrent_count_never_goes_below_zero(self):
        m = _make_monitor()
        # Complete without starting — should not go negative
        m.workflow_completed(passed=True)
        assert m.concurrent_count == 0

    def test_concurrent_count_after_mixed_operations(self):
        m = _make_monitor()
        m.workflow_started()
        m.workflow_started()
        m.workflow_completed(passed=True)
        assert m.concurrent_count == 1


# ---------------------------------------------------------------------------
# Queue depth
# ---------------------------------------------------------------------------

class TestQueueDepth:
    def test_starts_at_zero(self):
        m = _make_monitor()
        assert m.queue_depth == 0

    def test_increments_on_enqueue(self):
        m = _make_monitor()
        m.workflow_enqueued()
        assert m.queue_depth == 1

    def test_decrements_on_dequeue(self):
        m = _make_monitor()
        m.workflow_enqueued()
        m.workflow_dequeued()
        assert m.queue_depth == 0

    def test_queue_depth_never_goes_below_zero(self):
        m = _make_monitor()
        m.workflow_dequeued()
        assert m.queue_depth == 0

    def test_queue_depth_multiple_enqueues(self):
        m = _make_monitor()
        for _ in range(5):
            m.workflow_enqueued()
        assert m.queue_depth == 5

    def test_queue_depth_partial_dequeue(self):
        m = _make_monitor()
        for _ in range(4):
            m.workflow_enqueued()
        m.workflow_dequeued()
        m.workflow_dequeued()
        assert m.queue_depth == 2


# ---------------------------------------------------------------------------
# PASS / FAIL throughput tracking
# ---------------------------------------------------------------------------

class TestPassFailThroughput:
    def test_pass_count_increments(self):
        m = _make_monitor()
        m.workflow_started()
        m.workflow_completed(passed=True)
        snap = m.snapshot(window_seconds=60.0)
        assert snap.pass_count == 1
        assert snap.fail_count == 0

    def test_fail_count_increments(self):
        m = _make_monitor()
        m.workflow_started()
        m.workflow_completed(passed=False)
        snap = m.snapshot(window_seconds=60.0)
        assert snap.fail_count == 1
        assert snap.pass_count == 0

    def test_mixed_pass_fail_counts(self):
        m = _make_monitor()
        for _ in range(3):
            m.workflow_started()
            m.workflow_completed(passed=True)
        for _ in range(2):
            m.workflow_started()
            m.workflow_completed(passed=False)
        snap = m.snapshot(window_seconds=60.0)
        assert snap.pass_count == 3
        assert snap.fail_count == 2
        assert snap.total_completed == 5

    def test_pass_wps_positive_when_passes_exist(self):
        m = _make_monitor()
        m.workflow_started()
        m.workflow_completed(passed=True)
        snap = m.snapshot(window_seconds=10.0)
        assert snap.pass_wps > 0.0

    def test_fail_wps_positive_when_fails_exist(self):
        m = _make_monitor()
        m.workflow_started()
        m.workflow_completed(passed=False)
        snap = m.snapshot(window_seconds=10.0)
        assert snap.fail_wps > 0.0

    def test_pass_wps_plus_fail_wps_equals_total_wps(self):
        m = _make_monitor()
        for _ in range(2):
            m.workflow_completed(passed=True)
        for _ in range(3):
            m.workflow_completed(passed=False)
        snap = m.snapshot(window_seconds=10.0)
        assert abs(snap.pass_wps + snap.fail_wps - snap.wps) < 1e-9


# ---------------------------------------------------------------------------
# Sliding window behaviour
# ---------------------------------------------------------------------------

class TestSlidingWindows:
    def test_snapshot_returns_correct_window_seconds(self):
        m = _make_monitor()
        snap = m.snapshot(window_seconds=10.0)
        assert snap.window_seconds == 10.0

    def test_empty_monitor_returns_zero_wps(self):
        m = _make_monitor()
        snap = m.snapshot(window_seconds=60.0)
        assert snap.wps == 0.0
        assert snap.total_completed == 0

    def test_wps_calculated_over_window(self):
        m = _make_monitor(windows=(60.0,))
        # Complete 6 workflows; WPS over 60s window = 6/60 = 0.1
        for _ in range(6):
            m.workflow_completed(passed=True)
        snap = m.snapshot(window_seconds=60.0)
        assert snap.total_completed == 6
        assert abs(snap.wps - 6 / 60.0) < 1e-9

    def test_all_snapshots_returns_all_windows(self):
        m = _make_monitor(windows=(1.0, 10.0, 60.0))
        snaps = m.all_snapshots()
        assert "window_1s" in snaps
        assert "window_10s" in snaps
        assert "window_60s" in snaps

    def test_all_snapshots_window_seconds_match(self):
        m = _make_monitor(windows=(1.0, 10.0, 60.0))
        snaps = m.all_snapshots()
        assert snaps["window_1s"].window_seconds == 1.0
        assert snaps["window_10s"].window_seconds == 10.0
        assert snaps["window_60s"].window_seconds == 60.0

    def test_old_events_excluded_from_small_window(self):
        """Events older than the window should not be counted."""
        m = _make_monitor(windows=(1.0, 60.0))
        # Manually inject an old completion event (61 seconds ago)
        old_ts = time.time() - 61.0
        m._completions.append((old_ts, True))
        # Add a recent one
        m.workflow_completed(passed=True)

        snap_1s = m.snapshot(window_seconds=1.0)
        snap_60s = m.snapshot(window_seconds=60.0)

        # The old event should NOT appear in the 1s window
        assert snap_1s.total_completed == 1
        # The old event should NOT appear in the 60s window either (61s ago)
        assert snap_60s.total_completed == 1

    def test_recent_events_included_in_window(self):
        m = _make_monitor(windows=(10.0,))
        m.workflow_completed(passed=True)
        m.workflow_completed(passed=False)
        snap = m.snapshot(window_seconds=10.0)
        assert snap.total_completed == 2


# ---------------------------------------------------------------------------
# get_metrics()
# ---------------------------------------------------------------------------

class TestGetMetrics:
    def test_get_metrics_has_concurrent_workflows(self):
        m = _make_monitor()
        m.workflow_started()
        metrics = m.get_metrics()
        assert "concurrent_workflows" in metrics
        assert metrics["concurrent_workflows"] == 1

    def test_get_metrics_has_queue_depth(self):
        m = _make_monitor()
        m.workflow_enqueued()
        metrics = m.get_metrics()
        assert "queue_depth" in metrics
        assert metrics["queue_depth"] == 1

    def test_get_metrics_has_per_window_keys(self):
        m = _make_monitor(windows=(1.0, 10.0, 60.0))
        metrics = m.get_metrics()
        assert "throughput_1s" in metrics
        assert "throughput_10s" in metrics
        assert "throughput_60s" in metrics

    def test_get_metrics_per_window_is_dict(self):
        m = _make_monitor(windows=(10.0,))
        metrics = m.get_metrics()
        assert isinstance(metrics["throughput_10s"], dict)

    def test_get_metrics_per_window_has_wps(self):
        m = _make_monitor(windows=(10.0,))
        metrics = m.get_metrics()
        assert "wps" in metrics["throughput_10s"]


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_completions(self):
        m = _make_monitor()
        m.workflow_completed(passed=True)
        m.reset()
        snap = m.snapshot(window_seconds=60.0)
        assert snap.total_completed == 0

    def test_reset_clears_concurrent_count(self):
        m = _make_monitor()
        m.workflow_started()
        m.reset()
        assert m.concurrent_count == 0

    def test_reset_clears_queue_depth(self):
        m = _make_monitor()
        m.workflow_enqueued()
        m.reset()
        assert m.queue_depth == 0


# ---------------------------------------------------------------------------
# Async interface
# ---------------------------------------------------------------------------

class TestAsyncInterface:
    @pytest.mark.asyncio
    async def test_async_workflow_started_increments_concurrent(self):
        m = _make_monitor()
        await m.async_workflow_started()
        assert m.concurrent_count == 1

    @pytest.mark.asyncio
    async def test_async_workflow_completed_decrements_concurrent(self):
        m = _make_monitor()
        await m.async_workflow_started()
        await m.async_workflow_completed(passed=True)
        assert m.concurrent_count == 0

    @pytest.mark.asyncio
    async def test_async_workflow_completed_records_pass(self):
        m = _make_monitor()
        await m.async_workflow_completed(passed=True)
        snap = m.snapshot(window_seconds=60.0)
        assert snap.pass_count == 1

    @pytest.mark.asyncio
    async def test_async_workflow_completed_records_fail(self):
        m = _make_monitor()
        await m.async_workflow_completed(passed=False)
        snap = m.snapshot(window_seconds=60.0)
        assert snap.fail_count == 1

    @pytest.mark.asyncio
    async def test_async_enqueue_dequeue(self):
        m = _make_monitor()
        await m.async_workflow_enqueued()
        assert m.queue_depth == 1
        await m.async_workflow_dequeued()
        assert m.queue_depth == 0

    @pytest.mark.asyncio
    async def test_concurrent_async_completions(self):
        """Multiple concurrent async completions are all recorded."""
        m = _make_monitor()

        async def run_workflow(passed: bool):
            await m.async_workflow_started()
            await asyncio.sleep(0)  # yield
            await m.async_workflow_completed(passed=passed)

        await asyncio.gather(
            run_workflow(True),
            run_workflow(True),
            run_workflow(False),
        )

        snap = m.snapshot(window_seconds=60.0)
        assert snap.total_completed == 3
        assert snap.pass_count == 2
        assert snap.fail_count == 1
        assert m.concurrent_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_count_peaks_during_execution(self):
        """Concurrent count should peak > 1 when multiple workflows run simultaneously."""
        m = _make_monitor()
        peak = 0

        async def run_workflow():
            nonlocal peak
            await m.async_workflow_started()
            peak = max(peak, m.concurrent_count)
            await asyncio.sleep(0.01)
            await m.async_workflow_completed(passed=True)

        await asyncio.gather(run_workflow(), run_workflow(), run_workflow())
        assert peak >= 2  # at least 2 concurrent at some point
