"""
Error Recovery Manager - Task 6.2.5

Provides async recovery strategies for different error types in the
verification workflow, with fallback mechanisms and correlation ID tracking.

Recovery strategies:
- LLM_TIMEOUT: retry with exponential backoff, escalate if all fail
- DB_CONNECTION_LOST: use cached data, lower threshold temporarily, queue for review
- STATE_CORRUPTION: rollback to last valid state, log error, attempt recovery
- VERIFICATION_FAILURE: fallback verification mode, lower thresholds

Requirements:
- 8.1: Error handling with logging and correlation IDs
- 8.2: LLM API timeout/error → retry with exponential backoff, escalate if all fail
- 8.3: DB connection lost → cached data, lower threshold, queue for manual review
- 8.4: StateGraph execution error → rollback to last valid state, log, recover or escalate
- 8.5: All errors logged with correlation IDs
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import ConfigDict, BaseModel, Field

from .async_retry import AsyncRetryStrategy, RetryConfig, RetryResult
from .circuit_breaker import CircuitBreaker, get_circuit_breaker_registry
from .error_classifier import ErrorCategory, ErrorClassifier, get_error_classifier

logger = logging.getLogger("backend.verification.error_recovery")


# ---------------------------------------------------------------------------
# Recovery Error Types
# ---------------------------------------------------------------------------

class RecoveryErrorType(str, Enum):
    """Error types that have dedicated recovery strategies."""
    LLM_TIMEOUT = "llm_timeout"
    DB_CONNECTION_LOST = "db_connection_lost"
    STATE_CORRUPTION = "state_corruption"
    VERIFICATION_FAILURE = "verification_failure"


# ---------------------------------------------------------------------------
# Recovery Result
# ---------------------------------------------------------------------------

class RecoveryResult(BaseModel):
    """
    Result of an error recovery attempt.

    **Validates: Requirements 8.1** - structured error tracking with correlation IDs
    """

    success: bool = Field(description="Whether recovery succeeded")
    error_type: RecoveryErrorType = Field(description="Type of error that was recovered")
    strategy_used: str = Field(description="Name of the recovery strategy applied")
    correlation_id: str = Field(description="Correlation ID for distributed tracing")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When recovery was attempted",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the recovery attempt",
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether a fallback mechanism was used",
    )
    escalation_required: bool = Field(
        default=False,
        description="Whether human escalation is required after recovery",
    )
    recovered_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The recovered workflow state (if applicable)",
    )
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "error_type": "llm_timeout",
            "strategy_used": "exponential_backoff_retry",
            "correlation_id": "corr_abc123",
            "fallback_used": False,
            "escalation_required": False,
        }
    })


# ---------------------------------------------------------------------------
# ErrorRecoveryManager
# ---------------------------------------------------------------------------

class ErrorRecoveryManager:
    """
    Manages async error recovery strategies for the verification workflow.

    Provides dedicated recovery handlers for each error type:
    - LLM_TIMEOUT: exponential backoff retry (max 3 attempts), then escalate
    - DB_CONNECTION_LOST: use cached data, lower threshold, queue for review
    - STATE_CORRUPTION: rollback to last valid state, log, attempt recovery
    - VERIFICATION_FAILURE: fallback verification with lowered thresholds

    Usage::

        manager = ErrorRecoveryManager()

        # Recover from LLM timeout
        result = await manager.recover(
            error_type=RecoveryErrorType.LLM_TIMEOUT,
            exception=exc,
            state=workflow_state,
            retry_fn=call_llm,
            correlation_id="corr_abc",
        )

    **Validates: Requirements 8.1** - error handling with correlation IDs
    **Validates: Requirements 8.2** - LLM timeout retry with exponential backoff
    **Validates: Requirements 8.3** - DB connection loss recovery with cached data
    **Validates: Requirements 8.4** - StateGraph error rollback to last valid state
    """

    def __init__(
        self,
        error_classifier: Optional[ErrorClassifier] = None,
        llm_circuit_breaker: Optional[CircuitBreaker] = None,
        db_circuit_breaker: Optional[CircuitBreaker] = None,
        cache: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the error recovery manager.

        Args:
            error_classifier: Optional ErrorClassifier. Uses singleton if None.
            llm_circuit_breaker: Optional CircuitBreaker for LLM API calls.
            db_circuit_breaker: Optional CircuitBreaker for DB connections.
            cache: Optional shared cache dict for DB fallback data.
        """
        self.error_classifier = error_classifier or get_error_classifier()

        # Circuit breakers — use registry if not provided
        registry = get_circuit_breaker_registry()
        self.llm_circuit_breaker = llm_circuit_breaker or registry.get_or_create(
            "llm_api", failure_threshold=3, cooldown_seconds=60.0
        )
        self.db_circuit_breaker = db_circuit_breaker or registry.get_or_create(
            "internal_db", failure_threshold=3, cooldown_seconds=30.0
        )

        # Shared cache for DB fallback
        self._cache: Dict[str, Any] = cache if cache is not None else {}

        # State snapshots for rollback (keyed by workflow_id)
        self._state_snapshots: Dict[str, List[Dict[str, Any]]] = {}

        # Queue for manual review (DB connection loss scenario)
        self._manual_review_queue: List[Dict[str, Any]] = []

        # Temporarily lowered thresholds (DB connection loss scenario)
        self._lowered_thresholds: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def recover(
        self,
        error_type: RecoveryErrorType,
        exception: Exception,
        state: Optional[Dict[str, Any]] = None,
        retry_fn: Optional[Callable] = None,
        retry_args: tuple = (),
        retry_kwargs: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> RecoveryResult:
        """
        Dispatch to the appropriate recovery strategy.

        Args:
            error_type: The type of error to recover from.
            exception: The original exception that triggered recovery.
            state: Current workflow state (used for rollback/fallback).
            retry_fn: Callable to retry (for LLM_TIMEOUT recovery).
            retry_args: Positional args for retry_fn.
            retry_kwargs: Keyword args for retry_fn.
            correlation_id: Correlation ID for logging.

        Returns:
            RecoveryResult describing the outcome.
        """
        cid = correlation_id or f"corr_{uuid.uuid4().hex[:12]}"
        retry_kwargs = retry_kwargs or {}

        logger.info(
            "Starting error recovery: error_type=%s, correlation_id=%s, exception=%s",
            error_type.value,
            cid,
            exception,
        )

        if error_type == RecoveryErrorType.LLM_TIMEOUT:
            return await self._recover_llm_timeout(
                exception, retry_fn, retry_args, retry_kwargs, state, cid
            )
        elif error_type == RecoveryErrorType.DB_CONNECTION_LOST:
            return await self._recover_db_connection_lost(exception, state, cid)
        elif error_type == RecoveryErrorType.STATE_CORRUPTION:
            return await self._recover_state_corruption(exception, state, cid)
        elif error_type == RecoveryErrorType.VERIFICATION_FAILURE:
            return await self._recover_verification_failure(exception, state, cid)
        else:
            return RecoveryResult(
                success=False,
                error_type=error_type,
                strategy_used="unknown",
                correlation_id=cid,
                details={"error": str(exception)},
                escalation_required=True,
            )

    def save_state_snapshot(self, workflow_id: str, state: Dict[str, Any]) -> None:
        """
        Save a snapshot of the workflow state for potential rollback.

        **Validates: Requirements 8.4** - rollback to last valid state
        """
        if workflow_id not in self._state_snapshots:
            self._state_snapshots[workflow_id] = []

        # Deep copy to prevent mutation
        snapshot = copy.deepcopy(state)
        snapshot["_snapshot_timestamp"] = datetime.now().isoformat()
        self._state_snapshots[workflow_id].append(snapshot)

        # Keep only last 5 snapshots per workflow
        if len(self._state_snapshots[workflow_id]) > 5:
            self._state_snapshots[workflow_id].pop(0)

        logger.debug(
            "State snapshot saved for workflow_id=%s (total_snapshots=%d)",
            workflow_id,
            len(self._state_snapshots[workflow_id]),
        )

    def get_last_valid_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent valid state snapshot for rollback.

        **Validates: Requirements 8.4** - rollback to last valid state
        """
        snapshots = self._state_snapshots.get(workflow_id, [])
        if not snapshots:
            return None
        return copy.deepcopy(snapshots[-1])

    def cache_data(self, key: str, value: Any) -> None:
        """Store data in the fallback cache for DB connection loss scenarios."""
        self._cache[key] = {
            "value": value,
            "cached_at": datetime.now().isoformat(),
        }

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data for DB fallback."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        return entry.get("value")

    def get_manual_review_queue(self) -> List[Dict[str, Any]]:
        """Return items queued for manual review (DB connection loss scenario)."""
        return list(self._manual_review_queue)

    def get_lowered_thresholds(self) -> Dict[str, float]:
        """Return currently active lowered thresholds."""
        return dict(self._lowered_thresholds)

    def restore_thresholds(self) -> None:
        """Clear all temporarily lowered thresholds."""
        self._lowered_thresholds.clear()
        logger.info("Verification thresholds restored to normal")

    # ------------------------------------------------------------------
    # Recovery strategies
    # ------------------------------------------------------------------

    async def _recover_llm_timeout(
        self,
        exception: Exception,
        retry_fn: Optional[Callable],
        retry_args: tuple,
        retry_kwargs: Dict[str, Any],
        state: Optional[Dict[str, Any]],
        correlation_id: str,
    ) -> RecoveryResult:
        """
        Recovery strategy for LLM API timeout/error.

        Strategy:
        1. Retry with exponential backoff (max 3 attempts)
        2. If all retries fail → escalate to human

        **Validates: Requirements 8.2**
        """
        logger.warning(
            "LLM timeout recovery started (correlation_id=%s): %s",
            correlation_id,
            exception,
        )

        if retry_fn is None:
            logger.error(
                "LLM timeout recovery: no retry_fn provided, escalating (correlation_id=%s)",
                correlation_id,
            )
            return RecoveryResult(
                success=False,
                error_type=RecoveryErrorType.LLM_TIMEOUT,
                strategy_used="escalation_no_retry_fn",
                correlation_id=correlation_id,
                details={"original_error": str(exception)},
                escalation_required=True,
            )

        config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True,
        )
        strategy = AsyncRetryStrategy(
            config=config,
            circuit_breaker=self.llm_circuit_breaker,
        )

        retry_result: RetryResult = await strategy.execute(
            retry_fn,
            *retry_args,
            correlation_id=correlation_id,
            **retry_kwargs,
        )

        if retry_result.success:
            logger.info(
                "LLM timeout recovery succeeded after %d attempts (correlation_id=%s)",
                retry_result.attempts,
                correlation_id,
            )
            return RecoveryResult(
                success=True,
                error_type=RecoveryErrorType.LLM_TIMEOUT,
                strategy_used="exponential_backoff_retry",
                correlation_id=correlation_id,
                details={
                    "attempts": retry_result.attempts,
                    "total_time": retry_result.total_time,
                    "original_error": str(exception),
                },
                recovered_state=state,
            )
        else:
            logger.error(
                "LLM timeout recovery failed after %d attempts — escalating (correlation_id=%s)",
                retry_result.attempts,
                correlation_id,
            )
            # Update state to escalated if provided
            if state is not None:
                state["workflow_status"] = "escalated"
                _append_error_log(
                    state,
                    error_type="LLMTimeoutEscalation",
                    message=f"All {retry_result.attempts} LLM retry attempts failed",
                    details={
                        "correlation_id": correlation_id,
                        "final_exception": retry_result.final_exception,
                    },
                )

            return RecoveryResult(
                success=False,
                error_type=RecoveryErrorType.LLM_TIMEOUT,
                strategy_used="exponential_backoff_retry",
                correlation_id=correlation_id,
                details={
                    "attempts": retry_result.attempts,
                    "total_time": retry_result.total_time,
                    "final_exception": retry_result.final_exception,
                    "original_error": str(exception),
                },
                escalation_required=True,
                recovered_state=state,
            )

    async def _recover_db_connection_lost(
        self,
        exception: Exception,
        state: Optional[Dict[str, Any]],
        correlation_id: str,
    ) -> RecoveryResult:
        """
        Recovery strategy for internal DB connection loss.

        Strategy:
        1. Use cached data if available
        2. Lower verification threshold temporarily
        3. Queue workflow for manual review when DB is restored

        **Validates: Requirements 8.3**
        """
        logger.warning(
            "DB connection loss recovery started (correlation_id=%s): %s",
            correlation_id,
            exception,
        )

        # Record failure in DB circuit breaker
        self.db_circuit_breaker.record_failure(
            error_type="ConnectionError",
            correlation_id=correlation_id,
        )

        # 1. Check for cached data
        cached_available = len(self._cache) > 0
        if cached_available:
            logger.info(
                "DB connection loss: using cached data (%d entries, correlation_id=%s)",
                len(self._cache),
                correlation_id,
            )

        # 2. Lower verification thresholds temporarily
        self._lowered_thresholds["price_tolerance_percent"] = 5.0  # Relax from 1% to 5%
        self._lowered_thresholds["relevance_min_coverage"] = 0.5   # Relax from 0.7 to 0.5
        logger.info(
            "DB connection loss: verification thresholds temporarily lowered (correlation_id=%s)",
            correlation_id,
        )

        # 3. Queue for manual review
        queue_entry = {
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(exception),
            "workflow_state_summary": _summarize_state(state),
            "reason": "DB connection lost — requires manual review when DB is restored",
        }
        self._manual_review_queue.append(queue_entry)
        logger.info(
            "DB connection loss: workflow queued for manual review (queue_size=%d, correlation_id=%s)",
            len(self._manual_review_queue),
            correlation_id,
        )

        # Update state if provided
        if state is not None:
            _append_error_log(
                state,
                error_type="DBConnectionLost",
                message="Internal DB connection lost — using cached data with lowered thresholds",
                details={
                    "correlation_id": correlation_id,
                    "cached_data_available": cached_available,
                    "lowered_thresholds": dict(self._lowered_thresholds),
                    "queued_for_review": True,
                },
            )

        return RecoveryResult(
            success=cached_available,
            error_type=RecoveryErrorType.DB_CONNECTION_LOST,
            strategy_used="cached_data_fallback_with_lowered_thresholds",
            correlation_id=correlation_id,
            details={
                "cached_data_available": cached_available,
                "cache_entries": len(self._cache),
                "lowered_thresholds": dict(self._lowered_thresholds),
                "queued_for_manual_review": True,
                "original_error": str(exception),
            },
            fallback_used=cached_available,
            escalation_required=not cached_available,
            recovered_state=state,
        )

    async def _recover_state_corruption(
        self,
        exception: Exception,
        state: Optional[Dict[str, Any]],
        correlation_id: str,
    ) -> RecoveryResult:
        """
        Recovery strategy for StateGraph execution errors / state corruption.

        Strategy:
        1. Rollback to last valid state snapshot
        2. Log error with full stack trace
        3. Attempt recovery or escalate

        **Validates: Requirements 8.4**
        """
        logger.error(
            "State corruption recovery started (correlation_id=%s): %s",
            correlation_id,
            exception,
        )

        workflow_id = state.get("workflow_id", "unknown") if state else "unknown"
        last_valid = self.get_last_valid_state(workflow_id)

        if last_valid is not None:
            logger.info(
                "State corruption: rolling back to last valid state "
                "(workflow_id=%s, snapshot_ts=%s, correlation_id=%s)",
                workflow_id,
                last_valid.get("_snapshot_timestamp", "unknown"),
                correlation_id,
            )

            # Log the error in the recovered state
            _append_error_log(
                last_valid,
                error_type="StateCorruption",
                message=f"State corruption detected — rolled back to snapshot: {exception}",
                details={
                    "correlation_id": correlation_id,
                    "original_error": str(exception),
                    "rollback_from_status": state.get("workflow_status") if state else None,
                    "rollback_to_snapshot": last_valid.get("_snapshot_timestamp"),
                },
            )

            return RecoveryResult(
                success=True,
                error_type=RecoveryErrorType.STATE_CORRUPTION,
                strategy_used="rollback_to_last_valid_state",
                correlation_id=correlation_id,
                details={
                    "workflow_id": workflow_id,
                    "snapshot_timestamp": last_valid.get("_snapshot_timestamp"),
                    "original_error": str(exception),
                },
                recovered_state=last_valid,
            )
        else:
            logger.error(
                "State corruption: no valid snapshot available — escalating "
                "(workflow_id=%s, correlation_id=%s)",
                workflow_id,
                correlation_id,
            )

            # Update current state to failed/escalated if possible
            if state is not None:
                state["workflow_status"] = "escalated"
                _append_error_log(
                    state,
                    error_type="StateCorruptionEscalation",
                    message="State corruption with no valid snapshot — escalating to human",
                    details={
                        "correlation_id": correlation_id,
                        "original_error": str(exception),
                    },
                )

            return RecoveryResult(
                success=False,
                error_type=RecoveryErrorType.STATE_CORRUPTION,
                strategy_used="escalation_no_snapshot",
                correlation_id=correlation_id,
                details={
                    "workflow_id": workflow_id,
                    "original_error": str(exception),
                    "reason": "No valid state snapshot available for rollback",
                },
                escalation_required=True,
                recovered_state=state,
            )

    async def _recover_verification_failure(
        self,
        exception: Exception,
        state: Optional[Dict[str, Any]],
        correlation_id: str,
    ) -> RecoveryResult:
        """
        Recovery strategy for verification failures.

        Strategy:
        1. Log the failure with full context
        2. Apply fallback verification mode with lowered thresholds
        3. Mark for manual review if thresholds already lowered

        **Validates: Requirements 8.1, 8.3**
        """
        logger.warning(
            "Verification failure recovery started (correlation_id=%s): %s",
            correlation_id,
            exception,
        )

        # Check if thresholds are already lowered (DB connection loss scenario)
        already_lowered = bool(self._lowered_thresholds)

        if not already_lowered:
            # Apply fallback thresholds
            self._lowered_thresholds["price_tolerance_percent"] = 3.0
            self._lowered_thresholds["relevance_min_coverage"] = 0.6
            logger.info(
                "Verification failure: applying fallback thresholds (correlation_id=%s)",
                correlation_id,
            )

        if state is not None:
            _append_error_log(
                state,
                error_type="VerificationFailure",
                message=f"Verification failure — applying fallback mode: {exception}",
                details={
                    "correlation_id": correlation_id,
                    "fallback_thresholds": dict(self._lowered_thresholds),
                    "already_lowered": already_lowered,
                },
            )

        return RecoveryResult(
            success=not already_lowered,
            error_type=RecoveryErrorType.VERIFICATION_FAILURE,
            strategy_used="fallback_verification_mode",
            correlation_id=correlation_id,
            details={
                "fallback_thresholds": dict(self._lowered_thresholds),
                "already_lowered": already_lowered,
                "original_error": str(exception),
            },
            fallback_used=True,
            escalation_required=already_lowered,
            recovered_state=state,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_error_log(
    state: Dict[str, Any],
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a structured error entry to the workflow state's error_log."""
    if "error_log" not in state:
        state["error_log"] = []

    state["error_log"].append({
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "message": message,
        "details": details or {},
    })


def _summarize_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a lightweight summary of workflow state for logging."""
    if state is None:
        return {}
    return {
        "workflow_id": state.get("workflow_id", "unknown"),
        "workflow_status": state.get("workflow_status", "unknown"),
        "retry_count": state.get("retry_count", 0),
        "correlation_id": state.get("correlation_id", "unknown"),
    }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_manager: Optional[ErrorRecoveryManager] = None


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Return the module-level singleton ErrorRecoveryManager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ErrorRecoveryManager()
    return _default_manager


def reset_error_recovery_manager() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_manager
    _default_manager = None


__all__ = [
    "RecoveryErrorType",
    "RecoveryResult",
    "ErrorRecoveryManager",
    "get_error_recovery_manager",
    "reset_error_recovery_manager",
]
