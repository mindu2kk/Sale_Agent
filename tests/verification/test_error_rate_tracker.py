"""
Tests for ErrorRateTracker - Task 6.2.1

Covers:
- Recording errors and successes
- Error rate calculation over time windows
- Per-component isolation
- Thread safety basics
- Stats retrieval
"""

import threading
import time
from unittest.mock import patch

import pytest

from verification.utils.error_rate_tracker import (
    ErrorRateTracker,
    WORKFLOW_COMPONENTS,
    get_error_rate_tracker,
    reset_error_rate_tracker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure the module-level singleton is reset between tests."""
    reset_error_rate_tracker()
    yield
    reset_error_rate_tracker()


@pytest.fixture
def tracker():
    return ErrorRateTracker(window_seconds=60)


# ---------------------------------------------------------------------------
# Recording errors and successes
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_error_increments_error_count(self, tracker):
        tracker.record_error("research", "TimeoutError")
        stats = tracker.get_component_stats("research")
        assert stats["error_count"] == 1
        assert stats["success_count"] == 0
        assert stats["total_events"] == 1

    def test_record_success_increments_success_count(self, tracker):
        tracker.record_success("verification")
        stats = tracker.get_component_stats("verification")
        assert stats["success_count"] == 1
        assert stats["error_count"] == 0
        assert stats["total_events"] == 1

    def test_record_multiple_events(self, tracker):
        tracker.record_error("correction", "LLMError")
        tracker.record_error("correction", "ValidationError")
        tracker.record_success("correction")
        stats = tracker.get_component_stats("correction")
        assert stats["total_events"] == 3
        assert stats["error_count"] == 2
        assert stats["success_count"] == 1

    def test_record_error_with_correlation_id(self, tracker):
        tracker.record_error("escalation", "NetworkError", correlation_id="corr_123")
        stats = tracker.get_component_stats("escalation")
        assert stats["error_count"] == 1

    def test_record_success_with_correlation_id(self, tracker):
        tracker.record_success("research", correlation_id="corr_456")
        stats = tracker.get_component_stats("research")
        assert stats["success_count"] == 1

    def test_error_types_tracked_per_component(self, tracker):
        tracker.record_error("research", "TimeoutError")
        tracker.record_error("research", "TimeoutError")
        tracker.record_error("research", "LLMError")
        stats = tracker.get_component_stats("research")
        assert stats["error_types"]["TimeoutError"] == 2
        assert stats["error_types"]["LLMError"] == 1

    def test_unknown_component_accepted(self, tracker):
        """Non-canonical components should be tracked without error."""
        tracker.record_error("custom_node", "SomeError")
        stats = tracker.get_component_stats("custom_node")
        assert stats["error_count"] == 1


# ---------------------------------------------------------------------------
# Error rate calculation
# ---------------------------------------------------------------------------

class TestErrorRateCalculation:
    def test_error_rate_all_errors(self, tracker):
        for _ in range(5):
            tracker.record_error("research", "Error")
        rate = tracker.get_error_rate("research")
        assert rate == 1.0

    def test_error_rate_no_errors(self, tracker):
        for _ in range(5):
            tracker.record_success("research")
        rate = tracker.get_error_rate("research")
        assert rate == 0.0

    def test_error_rate_mixed(self, tracker):
        tracker.record_error("verification", "Error")
        tracker.record_success("verification")
        tracker.record_success("verification")
        tracker.record_success("verification")
        rate = tracker.get_error_rate("verification")
        assert abs(rate - 0.25) < 1e-9

    def test_error_rate_empty_component(self, tracker):
        rate = tracker.get_error_rate("correction")
        assert rate == 0.0

    def test_error_rate_respects_window(self, tracker):
        """Events outside the window should not count."""
        now = time.time()
        # Inject old events directly by patching time.time
        with patch("time.time", return_value=now - 120):
            tracker.record_error("research", "OldError")
            tracker.record_error("research", "OldError")

        # Recent events
        tracker.record_success("research")
        tracker.record_success("research")

        # With 60s window, old events are excluded → rate = 0.0
        rate = tracker.get_error_rate("research", window_seconds=60)
        assert rate == 0.0

    def test_error_rate_custom_window(self, tracker):
        """Custom window_seconds parameter is respected."""
        now = time.time()
        with patch("time.time", return_value=now - 30):
            tracker.record_error("research", "Error")

        # 60s window includes the 30s-old event
        rate_60 = tracker.get_error_rate("research", window_seconds=60)
        assert rate_60 == 1.0

        # 10s window excludes the 30s-old event
        rate_10 = tracker.get_error_rate("research", window_seconds=10)
        assert rate_10 == 0.0

    def test_error_rate_window_boundary(self, tracker):
        """Events exactly at the boundary should be included."""
        now = time.time()
        with patch("time.time", return_value=now - 59):
            tracker.record_error("verification", "BoundaryError")

        rate = tracker.get_error_rate("verification", window_seconds=60)
        assert rate == 1.0


# ---------------------------------------------------------------------------
# Per-component isolation
# ---------------------------------------------------------------------------

class TestComponentIsolation:
    def test_errors_isolated_per_component(self, tracker):
        tracker.record_error("research", "Error")
        tracker.record_error("research", "Error")
        tracker.record_success("verification")

        research_rate = tracker.get_error_rate("research")
        verification_rate = tracker.get_error_rate("verification")

        assert research_rate == 1.0
        assert verification_rate == 0.0

    def test_reset_single_component_does_not_affect_others(self, tracker):
        tracker.record_error("research", "Error")
        tracker.record_error("verification", "Error")

        tracker.reset("research")

        assert tracker.get_error_rate("research") == 0.0
        assert tracker.get_error_rate("verification") == 1.0

    def test_all_canonical_components_independent(self, tracker):
        for i, comp in enumerate(WORKFLOW_COMPONENTS):
            for _ in range(i + 1):
                tracker.record_error(comp, "Error")

        for i, comp in enumerate(WORKFLOW_COMPONENTS):
            stats = tracker.get_component_stats(comp)
            assert stats["error_count"] == i + 1


# ---------------------------------------------------------------------------
# Stats retrieval
# ---------------------------------------------------------------------------

class TestStatsRetrieval:
    def test_get_component_stats_structure(self, tracker):
        tracker.record_error("research", "TimeoutError", correlation_id="c1")
        tracker.record_success("research", correlation_id="c2")
        stats = tracker.get_component_stats("research")

        assert "component" in stats
        assert "total_events" in stats
        assert "error_count" in stats
        assert "success_count" in stats
        assert "error_rate" in stats
        assert "error_rate_60s" in stats
        assert "error_types" in stats
        assert "oldest_event_ts" in stats
        assert "newest_event_ts" in stats

    def test_get_component_stats_empty(self, tracker):
        stats = tracker.get_component_stats("correction")
        assert stats["total_events"] == 0
        assert stats["error_count"] == 0
        assert stats["success_count"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["oldest_event_ts"] is None
        assert stats["newest_event_ts"] is None

    def test_get_all_stats_includes_canonical_components(self, tracker):
        all_stats = tracker.get_all_stats()
        for comp in WORKFLOW_COMPONENTS:
            assert comp in all_stats

    def test_get_all_stats_includes_custom_components(self, tracker):
        tracker.record_error("custom_node", "Error")
        all_stats = tracker.get_all_stats()
        assert "custom_node" in all_stats

    def test_get_all_stats_values_match_component_stats(self, tracker):
        tracker.record_error("research", "Error")
        tracker.record_success("verification")

        all_stats = tracker.get_all_stats()
        assert all_stats["research"]["error_count"] == 1
        assert all_stats["verification"]["success_count"] == 1

    def test_timestamps_ordered(self, tracker):
        tracker.record_success("research")
        time.sleep(0.01)
        tracker.record_error("research", "Error")
        stats = tracker.get_component_stats("research")
        assert stats["oldest_event_ts"] <= stats["newest_event_ts"]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_all_clears_all_components(self, tracker):
        for comp in WORKFLOW_COMPONENTS:
            tracker.record_error(comp, "Error")

        tracker.reset()

        for comp in WORKFLOW_COMPONENTS:
            assert tracker.get_error_rate(comp) == 0.0
            assert tracker.get_component_stats(comp)["total_events"] == 0

    def test_reset_single_component(self, tracker):
        tracker.record_error("research", "Error")
        tracker.reset("research")
        assert tracker.get_component_stats("research")["total_events"] == 0

    def test_reset_none_resets_all(self, tracker):
        tracker.record_error("research", "Error")
        tracker.record_error("verification", "Error")
        tracker.reset(None)
        assert tracker.get_component_stats("research")["total_events"] == 0
        assert tracker.get_component_stats("verification")["total_events"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_errors(self, tracker):
        """Multiple threads recording errors should not corrupt state."""
        errors_per_thread = 50
        num_threads = 10

        def record_errors():
            for _ in range(errors_per_thread):
                tracker.record_error("research", "ConcurrentError")

        threads = [threading.Thread(target=record_errors) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = tracker.get_component_stats("research")
        assert stats["error_count"] == errors_per_thread * num_threads

    def test_concurrent_mixed_operations(self, tracker):
        """Concurrent errors and successes should not cause data races."""
        results = []

        def worker(component, is_error):
            for _ in range(20):
                if is_error:
                    tracker.record_error(component, "Error")
                else:
                    tracker.record_success(component)
            results.append(tracker.get_error_rate(component))

        threads = [
            threading.Thread(target=worker, args=("verification", i % 2 == 0))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No assertion on exact rate — just verify no exception and rate is valid
        rate = tracker.get_error_rate("verification")
        assert 0.0 <= rate <= 1.0

    def test_concurrent_reset_and_record(self, tracker):
        """Reset and record operations should not deadlock or corrupt state."""
        stop_event = threading.Event()

        def recorder():
            while not stop_event.is_set():
                tracker.record_error("correction", "Error")
                tracker.record_success("correction")

        def resetter():
            for _ in range(10):
                tracker.reset("correction")
                time.sleep(0.001)

        t1 = threading.Thread(target=recorder)
        t2 = threading.Thread(target=resetter)
        t1.start()
        t2.start()
        time.sleep(0.05)
        stop_event.set()
        t1.join()
        t2.join()

        # Should complete without deadlock or exception
        rate = tracker.get_error_rate("correction")
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    def test_get_error_rate_tracker_returns_same_instance(self):
        t1 = get_error_rate_tracker()
        t2 = get_error_rate_tracker()
        assert t1 is t2

    def test_reset_error_rate_tracker_creates_new_instance(self):
        t1 = get_error_rate_tracker()
        reset_error_rate_tracker()
        t2 = get_error_rate_tracker()
        assert t1 is not t2

    def test_singleton_state_persists_across_calls(self):
        t = get_error_rate_tracker()
        t.record_error("research", "Error")
        t2 = get_error_rate_tracker()
        assert t2.get_component_stats("research")["error_count"] == 1
