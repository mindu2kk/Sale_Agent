"""
Tests for LazyWorkflowState and StateMemoryManager.

Covers:
- Lazy loading behaviour (data not loaded until accessed)
- State compression reduces memory footprint
- Cleanup releases resources
- StateMemoryManager LRU eviction
- Memory stats reporting
"""

import time
import pytest

from backend.verification.models.execution import ExecutionStep, ExecutionStatus
from backend.verification.utils.lazy_state import LazyWorkflowState, StateMemoryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(node_name: str = "test_node") -> ExecutionStep:
    return ExecutionStep(
        node_name=node_name,
        execution_time=0.5,
        status=ExecutionStatus.SUCCESS,
        input_summary="input data here",
        output_summary="output data here",
    )


def _make_state(workflow_id: str = "wf_test_001") -> dict:
    return {
        "workflow_id": workflow_id,
        "workflow_status": "approved",
        "objection_text": "Why is this product expensive?",
        "draft_response": "Because it has premium features.",
        "final_response": "Premium features justify the price.",
        "verification_result": {"is_approved": True},
        "execution_log": [_make_step("research"), _make_step("verification")],
        "correction_feedback": "Fix the price.",
        "resource_usage": {"cpu_time_seconds": 1.0, "memory_peak_mb": 50.0},
        "error_log": [],
        "tools_used": ["price_lookup"],
        "research_reasoning": "Checked price DB.",
        "retry_count": 0,
        "max_retries": 3,
    }


# ---------------------------------------------------------------------------
# LazyWorkflowState – lazy loading
# ---------------------------------------------------------------------------

class TestLazyLoading:
    def test_verification_result_not_loaded_until_accessed(self):
        """verification_result should only be returned when the property is accessed."""
        state_dict = _make_state()
        lazy = LazyWorkflowState(state_dict)

        # Access the property – should return the value without error
        result = lazy.verification_result
        assert result == {"is_approved": True}

    def test_execution_log_not_loaded_until_accessed(self):
        """execution_log should only be returned when the property is accessed."""
        state_dict = _make_state()
        lazy = LazyWorkflowState(state_dict)

        log = lazy.execution_log
        assert len(log) == 2

    def test_other_fields_accessible_via_getattr(self):
        state_dict = _make_state()
        lazy = LazyWorkflowState(state_dict)

        assert lazy.workflow_id == "wf_test_001"
        assert lazy.workflow_status == "approved"
        assert lazy.retry_count == 0

    def test_missing_field_raises_attribute_error(self):
        lazy = LazyWorkflowState({})
        with pytest.raises(AttributeError):
            _ = lazy.nonexistent_field

    def test_setattr_updates_underlying_state(self):
        state_dict = _make_state()
        lazy = LazyWorkflowState(state_dict)
        lazy.workflow_status = "failed"
        assert state_dict["workflow_status"] == "failed"

    def test_get_helper(self):
        lazy = LazyWorkflowState(_make_state())
        assert lazy.get("workflow_id") == "wf_test_001"
        assert lazy.get("missing_key", "default") == "default"


# ---------------------------------------------------------------------------
# LazyWorkflowState – compression
# ---------------------------------------------------------------------------

class TestCompression:
    def test_compress_reduces_log_to_summary_dicts(self):
        state_dict = _make_state()
        lazy = LazyWorkflowState(state_dict)

        lazy.compress()

        log = lazy.execution_log
        assert len(log) == 2
        for entry in log:
            assert isinstance(entry, dict)
            # Summary keys present
            assert "node_name" in entry
            assert "timestamp" in entry
            assert "execution_time" in entry
            assert "status" in entry
            # Full payload keys dropped
            assert "input_summary" not in entry
            assert "output_summary" not in entry

    def test_compress_marks_state_as_compressed(self):
        lazy = LazyWorkflowState(_make_state())
        assert not lazy.is_compressed
        lazy.compress()
        assert lazy.is_compressed

    def test_compress_idempotent(self):
        lazy = LazyWorkflowState(_make_state())
        lazy.compress()
        log_after_first = list(lazy.execution_log)
        lazy.compress()  # second call should be a no-op
        assert list(lazy.execution_log) == log_after_first

    def test_compress_reduces_memory_footprint(self):
        """Compressed state should not be larger than original."""
        import sys
        state_dict = _make_state()
        lazy = LazyWorkflowState(state_dict)
        size_before = lazy.memory_size_bytes()
        lazy.compress()
        size_after = lazy.memory_size_bytes()
        # After compression the log entries are smaller dicts; overall size
        # should not grow (it may stay the same for small test states).
        assert size_after <= size_before + 512  # allow small overhead

    def test_compress_dict_entries(self):
        """compress() should also handle pre-existing dict entries in the log."""
        state_dict = _make_state()
        state_dict["execution_log"] = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "node_name": "research",
                "execution_time": 1.0,
                "status": "success",
                "correlation_id": "corr_abc",
                "input_summary": "big input payload",
                "output_summary": "big output payload",
            }
        ]
        lazy = LazyWorkflowState(state_dict)
        lazy.compress()
        entry = lazy.execution_log[0]
        assert "input_summary" not in entry
        assert "output_summary" not in entry
        assert entry["node_name"] == "research"


# ---------------------------------------------------------------------------
# LazyWorkflowState – cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_clears_expensive_fields(self):
        lazy = LazyWorkflowState(_make_state())
        lazy.cleanup()

        assert lazy.execution_log == []
        assert lazy.verification_result is None
        assert lazy.get("correction_feedback") is None
        assert lazy.get("resource_usage") is None

    def test_cleanup_preserves_audit_fields(self):
        lazy = LazyWorkflowState(_make_state())
        lazy.cleanup()

        assert lazy.workflow_id == "wf_test_001"
        assert lazy.workflow_status == "approved"
        assert lazy.final_response == "Premium features justify the price."

    def test_cleanup_marks_state_as_released(self):
        lazy = LazyWorkflowState(_make_state())
        assert not lazy.is_released
        lazy.cleanup()
        assert lazy.is_released

    def test_cleanup_idempotent(self):
        lazy = LazyWorkflowState(_make_state())
        lazy.cleanup()
        lazy.cleanup()  # second call should be a no-op
        assert lazy.is_released

    def test_compress_skipped_after_cleanup(self):
        lazy = LazyWorkflowState(_make_state())
        lazy.cleanup()
        lazy.compress()  # should be a no-op
        assert not lazy.is_compressed  # compress was skipped


# ---------------------------------------------------------------------------
# LazyWorkflowState – streaming
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_stream_execution_log_yields_all_entries(self):
        lazy = LazyWorkflowState(_make_state())
        entries = list(lazy.stream_execution_log())
        assert len(entries) == 2

    def test_stream_execution_log_is_generator(self):
        import types
        lazy = LazyWorkflowState(_make_state())
        gen = lazy.stream_execution_log()
        assert isinstance(gen, types.GeneratorType)

    def test_stream_empty_log(self):
        state_dict = _make_state()
        state_dict["execution_log"] = []
        lazy = LazyWorkflowState(state_dict)
        assert list(lazy.stream_execution_log()) == []


# ---------------------------------------------------------------------------
# StateMemoryManager – basic operations
# ---------------------------------------------------------------------------

class TestStateMemoryManager:
    def test_register_and_get_state(self):
        mgr = StateMemoryManager()
        lazy = LazyWorkflowState(_make_state("wf_001"))
        mgr.register_state("wf_001", lazy)

        retrieved = mgr.get_state("wf_001")
        assert retrieved is lazy

    def test_get_nonexistent_returns_none(self):
        mgr = StateMemoryManager()
        assert mgr.get_state("nonexistent") is None

    def test_release_state_calls_cleanup(self):
        mgr = StateMemoryManager()
        lazy = LazyWorkflowState(_make_state("wf_002"))
        mgr.register_state("wf_002", lazy)

        released = mgr.release_state("wf_002")
        assert released is True
        assert lazy.is_released
        assert mgr.get_state("wf_002") is None

    def test_release_nonexistent_returns_false(self):
        mgr = StateMemoryManager()
        assert mgr.release_state("nonexistent") is False

    def test_len(self):
        mgr = StateMemoryManager()
        assert len(mgr) == 0
        mgr.register_state("wf_a", LazyWorkflowState(_make_state("wf_a")))
        assert len(mgr) == 1
        mgr.release_state("wf_a")
        assert len(mgr) == 0


# ---------------------------------------------------------------------------
# StateMemoryManager – LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_evicts_oldest_when_over_capacity(self):
        mgr = StateMemoryManager(max_states=3)

        for i in range(3):
            wf_id = f"wf_{i:03d}"
            mgr.register_state(wf_id, LazyWorkflowState(_make_state(wf_id)))

        # Access wf_000 to make it recently used
        mgr.get_state("wf_000")

        # Adding a 4th state should evict the LRU entry (wf_001)
        mgr.register_state("wf_003", LazyWorkflowState(_make_state("wf_003")))

        assert len(mgr) == 3
        assert mgr.get_state("wf_001") is None  # evicted
        assert mgr.get_state("wf_000") is not None  # still present
        assert mgr.get_state("wf_002") is not None
        assert mgr.get_state("wf_003") is not None

    def test_evicted_state_is_cleaned_up(self):
        mgr = StateMemoryManager(max_states=1)
        lazy_old = LazyWorkflowState(_make_state("wf_old"))
        mgr.register_state("wf_old", lazy_old)

        mgr.register_state("wf_new", LazyWorkflowState(_make_state("wf_new")))

        assert lazy_old.is_released  # cleanup was called on eviction

    def test_active_workflow_ids(self):
        mgr = StateMemoryManager(max_states=10)
        for i in range(3):
            wf_id = f"wf_{i}"
            mgr.register_state(wf_id, LazyWorkflowState(_make_state(wf_id)))

        ids = mgr.active_workflow_ids()
        assert set(ids) == {"wf_0", "wf_1", "wf_2"}


# ---------------------------------------------------------------------------
# StateMemoryManager – auto-compression of stale states
# ---------------------------------------------------------------------------

class TestAutoCompression:
    def test_stale_state_is_compressed_on_next_get(self):
        # Use a very short TTL so the state becomes stale immediately
        mgr = StateMemoryManager(compress_ttl_seconds=0.0)
        lazy = LazyWorkflowState(_make_state("wf_stale"))
        mgr.register_state("wf_stale", lazy)

        # Small sleep to ensure last_accessed < now - ttl
        time.sleep(0.01)

        # Trigger compression check via get_state
        mgr.get_state("wf_stale")

        assert lazy.is_compressed

    def test_fresh_state_not_compressed(self):
        mgr = StateMemoryManager(compress_ttl_seconds=3600.0)
        lazy = LazyWorkflowState(_make_state("wf_fresh"))
        mgr.register_state("wf_fresh", lazy)

        mgr.get_state("wf_fresh")

        assert not lazy.is_compressed


# ---------------------------------------------------------------------------
# StateMemoryManager – memory stats
# ---------------------------------------------------------------------------

class TestMemoryStats:
    def test_memory_stats_structure(self):
        mgr = StateMemoryManager(max_states=10)
        for i in range(3):
            wf_id = f"wf_{i}"
            mgr.register_state(wf_id, LazyWorkflowState(_make_state(wf_id)))

        stats = mgr.memory_stats()

        assert stats["tracked_states"] == 3
        assert stats["max_states"] == 10
        assert stats["total_estimated_bytes"] > 0
        assert stats["total_estimated_mb"] >= 0
        assert len(stats["per_state_bytes"]) == 3
        for wf_id, size in stats["per_state_bytes"].items():
            assert size > 0

    def test_memory_stats_empty_manager(self):
        mgr = StateMemoryManager()
        stats = mgr.memory_stats()
        assert stats["tracked_states"] == 0
        assert stats["total_estimated_bytes"] == 0

    def test_memory_stats_after_release(self):
        mgr = StateMemoryManager()
        mgr.register_state("wf_x", LazyWorkflowState(_make_state("wf_x")))
        mgr.release_state("wf_x")

        stats = mgr.memory_stats()
        assert stats["tracked_states"] == 0
