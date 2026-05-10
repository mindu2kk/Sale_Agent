"""
Tests for ErrorRecoveryManager - Task 6.2.5

Tests:
- LLM timeout recovery flow
- DB connection loss recovery
- State rollback mechanism
- Fallback to cached data
- Correlation ID propagation

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verification.utils.error_recovery import (
    ErrorRecoveryManager,
    RecoveryErrorType,
    RecoveryResult,
    get_error_recovery_manager,
    reset_error_recovery_manager,
)
from verification.utils.circuit_breaker import CircuitBreaker, reset_circuit_breaker_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons before each test."""
    reset_error_recovery_manager()
    reset_circuit_breaker_registry()
    yield
    reset_error_recovery_manager()
    reset_circuit_breaker_registry()


def _make_state(workflow_id: str = "wf_test_001", status: str = "verifying") -> Dict[str, Any]:
    """Create a minimal workflow state dict for testing."""
    return {
        "workflow_id": workflow_id,
        "workflow_status": status,
        "retry_count": 0,
        "max_retries": 3,
        "correlation_id": "corr_test_abc123",
        "error_log": [],
        "objection_text": "Test objection",
        "draft_response": "Test draft",
    }


# ---------------------------------------------------------------------------
# LLM Timeout Recovery
# ---------------------------------------------------------------------------

class TestLLMTimeoutRecovery:
    @pytest.mark.asyncio
    async def test_recovery_succeeds_after_retries(self):
        """LLM timeout recovery should succeed if retry_fn eventually succeeds."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        call_count = []

        async def retry_fn():
            call_count.append(1)
            if len(call_count) < 3:
                raise TimeoutError("LLM timeout")
            return "llm_response"

        result = await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=TimeoutError("initial timeout"),
            state=state,
            retry_fn=retry_fn,
            correlation_id="corr_llm_test",
        )

        assert result.success is True
        assert result.error_type == RecoveryErrorType.LLM_TIMEOUT
        assert result.strategy_used == "exponential_backoff_retry"
        assert result.correlation_id == "corr_llm_test"
        assert len(call_count) == 3

    @pytest.mark.asyncio
    async def test_recovery_escalates_when_all_retries_fail(self):
        """LLM timeout recovery should escalate when all retries fail."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        async def always_fail():
            raise TimeoutError("always timeout")

        result = await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=TimeoutError("initial timeout"),
            state=state,
            retry_fn=always_fail,
            correlation_id="corr_escalate_test",
        )

        assert result.success is False
        assert result.escalation_required is True
        assert result.correlation_id == "corr_escalate_test"
        assert state["workflow_status"] == "escalated"

    @pytest.mark.asyncio
    async def test_recovery_escalates_without_retry_fn(self):
        """LLM timeout recovery without retry_fn should escalate immediately."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        result = await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=TimeoutError("timeout"),
            state=state,
            retry_fn=None,
            correlation_id="corr_no_fn",
        )

        assert result.success is False
        assert result.escalation_required is True

    @pytest.mark.asyncio
    async def test_recovery_logs_error_to_state(self):
        """LLM timeout recovery should log errors to workflow state."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        async def always_fail():
            raise TimeoutError("timeout")

        await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=TimeoutError("initial"),
            state=state,
            retry_fn=always_fail,
            correlation_id="corr_log_test",
        )

        assert len(state["error_log"]) > 0
        assert state["error_log"][-1]["error_type"] == "LLMTimeoutEscalation"

    @pytest.mark.asyncio
    async def test_correlation_id_propagated(self):
        """Correlation ID should appear in recovery result."""
        manager = ErrorRecoveryManager()

        result = await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=TimeoutError("timeout"),
            retry_fn=None,
            correlation_id="corr_propagation_test",
        )

        assert result.correlation_id == "corr_propagation_test"


# ---------------------------------------------------------------------------
# DB Connection Loss Recovery
# ---------------------------------------------------------------------------

class TestDBConnectionLossRecovery:
    @pytest.mark.asyncio
    async def test_recovery_uses_cached_data(self):
        """DB connection loss recovery should use cached data when available."""
        manager = ErrorRecoveryManager()
        manager.cache_data("product_prices", {"iPhone15": 29990000})
        state = _make_state()

        result = await manager.recover(
            error_type=RecoveryErrorType.DB_CONNECTION_LOST,
            exception=ConnectionError("DB connection lost"),
            state=state,
            correlation_id="corr_db_test",
        )

        assert result.success is True
        assert result.fallback_used is True
        assert result.details["cached_data_available"] is True
        assert result.details["cache_entries"] == 1

    @pytest.mark.asyncio
    async def test_recovery_lowers_thresholds(self):
        """DB connection loss recovery should lower verification thresholds."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        await manager.recover(
            error_type=RecoveryErrorType.DB_CONNECTION_LOST,
            exception=ConnectionError("DB down"),
            state=state,
            correlation_id="corr_threshold_test",
        )

        thresholds = manager.get_lowered_thresholds()
        assert "price_tolerance_percent" in thresholds
        assert thresholds["price_tolerance_percent"] > 1.0  # Relaxed from default 1%
        assert "relevance_min_coverage" in thresholds
        assert thresholds["relevance_min_coverage"] < 0.7  # Relaxed from default 0.7

    @pytest.mark.asyncio
    async def test_recovery_queues_for_manual_review(self):
        """DB connection loss recovery should queue workflow for manual review."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        await manager.recover(
            error_type=RecoveryErrorType.DB_CONNECTION_LOST,
            exception=ConnectionError("DB down"),
            state=state,
            correlation_id="corr_queue_test",
        )

        queue = manager.get_manual_review_queue()
        assert len(queue) == 1
        assert queue[0]["correlation_id"] == "corr_queue_test"
        assert "DB connection lost" in queue[0]["reason"]

    @pytest.mark.asyncio
    async def test_recovery_fails_without_cache(self):
        """DB connection loss recovery should fail (escalate) without cached data."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        result = await manager.recover(
            error_type=RecoveryErrorType.DB_CONNECTION_LOST,
            exception=ConnectionError("DB down"),
            state=state,
            correlation_id="corr_no_cache",
        )

        assert result.success is False
        assert result.escalation_required is True
        assert result.details["cached_data_available"] is False

    @pytest.mark.asyncio
    async def test_recovery_logs_error_to_state(self):
        """DB connection loss recovery should log to workflow state."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        await manager.recover(
            error_type=RecoveryErrorType.DB_CONNECTION_LOST,
            exception=ConnectionError("DB down"),
            state=state,
            correlation_id="corr_db_log",
        )

        assert len(state["error_log"]) > 0
        assert state["error_log"][-1]["error_type"] == "DBConnectionLost"

    def test_restore_thresholds(self):
        """restore_thresholds should clear all lowered thresholds."""
        manager = ErrorRecoveryManager()
        manager._lowered_thresholds["price_tolerance_percent"] = 5.0

        manager.restore_thresholds()

        assert manager.get_lowered_thresholds() == {}


# ---------------------------------------------------------------------------
# State Rollback Mechanism
# ---------------------------------------------------------------------------

class TestStateRollback:
    @pytest.mark.asyncio
    async def test_rollback_to_last_valid_state(self):
        """State corruption recovery should rollback to last valid snapshot."""
        manager = ErrorRecoveryManager()
        state = _make_state(status="verifying")

        # Save a valid snapshot
        valid_state = _make_state(status="researching")
        manager.save_state_snapshot("wf_test_001", valid_state)

        result = await manager.recover(
            error_type=RecoveryErrorType.STATE_CORRUPTION,
            exception=KeyError("missing key"),
            state=state,
            correlation_id="corr_rollback_test",
        )

        assert result.success is True
        assert result.strategy_used == "rollback_to_last_valid_state"
        assert result.recovered_state is not None
        assert result.recovered_state["workflow_status"] == "researching"

    @pytest.mark.asyncio
    async def test_rollback_escalates_without_snapshot(self):
        """State corruption recovery should escalate when no snapshot exists."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        result = await manager.recover(
            error_type=RecoveryErrorType.STATE_CORRUPTION,
            exception=KeyError("missing key"),
            state=state,
            correlation_id="corr_no_snapshot",
        )

        assert result.success is False
        assert result.escalation_required is True
        assert state["workflow_status"] == "escalated"

    def test_save_and_retrieve_snapshot(self):
        """save_state_snapshot and get_last_valid_state should work correctly."""
        manager = ErrorRecoveryManager()
        state = _make_state(workflow_id="wf_001", status="verifying")

        manager.save_state_snapshot("wf_001", state)
        retrieved = manager.get_last_valid_state("wf_001")

        assert retrieved is not None
        assert retrieved["workflow_status"] == "verifying"
        assert retrieved["workflow_id"] == "wf_001"

    def test_snapshot_is_deep_copy(self):
        """Snapshots should be independent copies of the state."""
        manager = ErrorRecoveryManager()
        state = _make_state()

        manager.save_state_snapshot("wf_001", state)
        state["workflow_status"] = "escalated"  # Mutate original

        retrieved = manager.get_last_valid_state("wf_001")
        assert retrieved["workflow_status"] == "verifying"  # Snapshot unchanged

    def test_keeps_only_last_5_snapshots(self):
        """Only the last 5 snapshots should be retained per workflow."""
        manager = ErrorRecoveryManager()

        for i in range(7):
            state = _make_state(status=f"status_{i}")
            manager.save_state_snapshot("wf_001", state)

        assert len(manager._state_snapshots["wf_001"]) == 5

    def test_get_last_valid_state_returns_none_for_unknown_workflow(self):
        """get_last_valid_state should return None for unknown workflow IDs."""
        manager = ErrorRecoveryManager()
        assert manager.get_last_valid_state("unknown_wf") is None

    @pytest.mark.asyncio
    async def test_rollback_logs_error_in_recovered_state(self):
        """Rollback should log the corruption error in the recovered state."""
        manager = ErrorRecoveryManager()
        valid_state = _make_state(status="researching")
        manager.save_state_snapshot("wf_test_001", valid_state)

        state = _make_state(status="verifying")

        result = await manager.recover(
            error_type=RecoveryErrorType.STATE_CORRUPTION,
            exception=KeyError("corrupted"),
            state=state,
            correlation_id="corr_log_rollback",
        )

        assert result.recovered_state is not None
        assert len(result.recovered_state.get("error_log", [])) > 0


# ---------------------------------------------------------------------------
# Fallback to Cached Data
# ---------------------------------------------------------------------------

class TestCachedDataFallback:
    def test_cache_and_retrieve_data(self):
        """cache_data and get_cached_data should store and retrieve values."""
        manager = ErrorRecoveryManager()

        manager.cache_data("product_prices", {"iPhone15": 29990000})
        retrieved = manager.get_cached_data("product_prices")

        assert retrieved == {"iPhone15": 29990000}

    def test_get_cached_data_returns_none_for_missing_key(self):
        """get_cached_data should return None for missing keys."""
        manager = ErrorRecoveryManager()
        assert manager.get_cached_data("nonexistent_key") is None

    def test_cache_overwrites_existing_entry(self):
        """Caching the same key twice should overwrite the previous value."""
        manager = ErrorRecoveryManager()

        manager.cache_data("key", "value1")
        manager.cache_data("key", "value2")

        assert manager.get_cached_data("key") == "value2"

    @pytest.mark.asyncio
    async def test_db_recovery_reports_cache_entries_count(self):
        """DB recovery result should include the number of cache entries."""
        manager = ErrorRecoveryManager()
        manager.cache_data("prices", {"A": 1})
        manager.cache_data("policies", {"B": 2})

        result = await manager.recover(
            error_type=RecoveryErrorType.DB_CONNECTION_LOST,
            exception=ConnectionError("DB down"),
            correlation_id="corr_cache_count",
        )

        assert result.details["cache_entries"] == 2


# ---------------------------------------------------------------------------
# Correlation ID Propagation
# ---------------------------------------------------------------------------

class TestCorrelationIDPropagation:
    @pytest.mark.asyncio
    async def test_correlation_id_in_all_recovery_types(self):
        """Correlation ID should be present in all recovery result types."""
        manager = ErrorRecoveryManager()
        cid = "corr_universal_test_xyz"

        # LLM timeout (no retry_fn → immediate escalation)
        r1 = await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=TimeoutError("t"),
            correlation_id=cid,
        )
        assert r1.correlation_id == cid

        # DB connection lost
        r2 = await manager.recover(
            error_type=RecoveryErrorType.DB_CONNECTION_LOST,
            exception=ConnectionError("c"),
            correlation_id=cid,
        )
        assert r2.correlation_id == cid

        # State corruption (no snapshot)
        r3 = await manager.recover(
            error_type=RecoveryErrorType.STATE_CORRUPTION,
            exception=KeyError("k"),
            correlation_id=cid,
        )
        assert r3.correlation_id == cid

        # Verification failure
        r4 = await manager.recover(
            error_type=RecoveryErrorType.VERIFICATION_FAILURE,
            exception=RuntimeError("v"),
            correlation_id=cid,
        )
        assert r4.correlation_id == cid

    @pytest.mark.asyncio
    async def test_auto_generated_correlation_id_when_not_provided(self):
        """A correlation ID should be auto-generated when not provided."""
        manager = ErrorRecoveryManager()

        result = await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=TimeoutError("t"),
        )

        assert result.correlation_id is not None
        assert len(result.correlation_id) > 0

    @pytest.mark.asyncio
    async def test_correlation_id_in_state_error_log(self):
        """Correlation ID should appear in the state error log entries."""
        manager = ErrorRecoveryManager()
        state = _make_state()
        cid = "corr_state_log_test"

        async def always_fail():
            raise TimeoutError("timeout")

        await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=TimeoutError("initial"),
            state=state,
            retry_fn=always_fail,
            correlation_id=cid,
        )

        assert len(state["error_log"]) > 0
        error_entry = state["error_log"][-1]
        assert cid in str(error_entry)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_error_recovery_manager_returns_same_instance(self):
        manager1 = get_error_recovery_manager()
        manager2 = get_error_recovery_manager()
        assert manager1 is manager2

    def test_reset_creates_new_instance(self):
        manager1 = get_error_recovery_manager()
        reset_error_recovery_manager()
        manager2 = get_error_recovery_manager()
        assert manager1 is not manager2
