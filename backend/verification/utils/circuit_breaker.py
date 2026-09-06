"""
Circuit Breaker Pattern for External Services - Task 6.2.2

Implements the standard circuit breaker pattern for external service calls
with three states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery).

Supports:
- Configurable failure threshold and cooldown period
- Async operation support (async context manager + async call wrapper)
- Fallback logic when circuit is OPEN (cached data, fallback function, or exception)
- Integration with ErrorRateTracker for observability
- State transition logging with correlation IDs

External services covered:
- LLM API (OpenAI/Anthropic)
- RAG Pipeline / Internal DB
- ChromaDB

Requirements:
- 8.4: Circuit breaker pattern for external service calls
- 8.1: Error handling with logging and correlation IDs
- 9.2: Support concurrent workflow execution
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .error_rate_tracker import ErrorRateTracker, get_error_rate_tracker

logger = logging.getLogger("backend.verification.circuit_breaker")


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    """Standard circuit breaker states."""
    CLOSED = "closed"       # Normal — requests pass through
    OPEN = "open"           # Failing — requests blocked, fallback used
    HALF_OPEN = "half_open" # Testing recovery — one probe request allowed


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""

    def __init__(self, service_name: str, failure_count: int) -> None:
        self.service_name = service_name
        self.failure_count = failure_count
        super().__init__(
            f"Circuit breaker OPEN for service '{service_name}' "
            f"(failure_count={failure_count})"
        )


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    State machine:
        CLOSED  ──(failures >= threshold)──► OPEN
        OPEN    ──(cooldown elapsed)───────► HALF_OPEN
        HALF_OPEN ──(success)──────────────► CLOSED
        HALF_OPEN ──(failure)──────────────► OPEN

    Usage — async context manager::

        cb = CircuitBreaker("llm_api")
        async with cb.protect():
            result = await call_llm(prompt)

    Usage — async call wrapper::

        result = await cb.call_async(call_llm, prompt)

    Usage — with fallback::

        result = await cb.call_async(
            call_llm, prompt,
            fallback=lambda: cached_response
        )

    **Validates: Requirements 8.4** - circuit breaker for external service calls
    **Validates: Requirements 8.1** - error handling with correlation IDs
    **Validates: Requirements 9.2** - concurrent workflow execution support
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        error_rate_tracker: Optional[ErrorRateTracker] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the circuit breaker.

        Args:
            service_name: Identifier for the external service being protected.
            failure_threshold: Number of consecutive failures before opening
                               the circuit. Default: 5.
            cooldown_seconds: Seconds to wait in OPEN state before transitioning
                              to HALF_OPEN. Default: 60.
            error_rate_tracker: Optional ErrorRateTracker for observability
                                integration. Uses module singleton if None.
            correlation_id: Optional correlation ID for log messages.
        """
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.correlation_id = correlation_id

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

        # ErrorRateTracker integration
        self._error_tracker: ErrorRateTracker = (
            error_rate_tracker if error_rate_tracker is not None
            else get_error_rate_tracker()
        )

    # ------------------------------------------------------------------
    # State properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may auto-transition OPEN → HALF_OPEN)."""
        with self._lock:
            return self._compute_state()

    def _compute_state(self) -> CircuitState:
        """Compute current state, auto-transitioning OPEN → HALF_OPEN if cooldown elapsed.
        Must be called with self._lock held."""
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self.cooldown_seconds
        ):
            self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def is_open(self) -> bool:
        """Return True if circuit is OPEN (requests should be blocked)."""
        return self.state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """Return True if a request should be allowed through."""
        with self._lock:
            state = self._compute_state()
            return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    # ------------------------------------------------------------------
    # Recording outcomes
    # ------------------------------------------------------------------

    def record_success(self, correlation_id: Optional[str] = None) -> None:
        """
        Record a successful call.

        - CLOSED: reset failure count
        - HALF_OPEN: close the circuit
        """
        cid = correlation_id or self.correlation_id
        with self._lock:
            self._success_count += 1
            prev_state = self._state
            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                self._failure_count = 0
                if self._state == CircuitState.HALF_OPEN:
                    self._transition_to(CircuitState.CLOSED)

        self._error_tracker.record_success(self.service_name, correlation_id=cid)
        logger.debug(
            "Circuit breaker '%s': success recorded (prev_state=%s, failure_count=%d)",
            self.service_name,
            prev_state.value,
            self._failure_count,
        )

    def record_failure(
        self,
        error_type: str = "UnknownError",
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Record a failed call.

        - Increments failure count
        - Opens circuit if failure_threshold reached
        - HALF_OPEN failure immediately reopens circuit
        """
        cid = correlation_id or self.correlation_id
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            prev_state = self._state

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — reopen immediately
                self._transition_to(CircuitState.OPEN)
            elif self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

        self._error_tracker.record_error(
            self.service_name, error_type, correlation_id=cid
        )
        logger.debug(
            "Circuit breaker '%s': failure recorded (error_type=%s, failure_count=%d, state=%s)",
            self.service_name,
            error_type,
            self._failure_count,
            self._state.value,
        )

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def protect(
        self,
        fallback: Optional[Callable[[], Any]] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Async context manager that protects a block of code with the circuit breaker.

        If the circuit is OPEN:
        - Calls fallback() and yields its result if fallback is provided
        - Raises CircuitOpenError otherwise

        Records success/failure automatically.

        Usage::

            async with cb.protect(fallback=lambda: cached_data) as result:
                # result is None when circuit is CLOSED/HALF_OPEN (call proceeds)
                # result is fallback() value when circuit is OPEN
                data = await fetch_data()

        Note: When circuit is OPEN and fallback is provided, the body is NOT
        executed. Use call_async() for a simpler API.
        """
        cid = correlation_id or self.correlation_id
        if not self.allow_request():
            if fallback is not None:
                logger.warning(
                    "Circuit breaker '%s' OPEN — using fallback (correlation_id=%s)",
                    self.service_name,
                    cid,
                )
                yield fallback()
                return
            raise CircuitOpenError(self.service_name, self._failure_count)

        try:
            yield None
            self.record_success(correlation_id=cid)
        except Exception as exc:
            self.record_failure(
                error_type=type(exc).__name__,
                correlation_id=cid,
            )
            raise

    async def call_async(
        self,
        fn: Callable[..., Any],
        *args: Any,
        fallback: Optional[Callable[[], Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call an async or sync callable protected by the circuit breaker.

        If the circuit is OPEN:
        - Returns fallback() result if fallback is provided
        - Raises CircuitOpenError otherwise

        Args:
            fn: Callable to invoke (async or sync).
            *args: Positional arguments for fn.
            fallback: Optional zero-argument callable returning a default value
                      when the circuit is OPEN.
            correlation_id: Optional correlation ID for logging.
            **kwargs: Keyword arguments for fn.

        Returns:
            Result of fn(*args, **kwargs) or fallback() if circuit is OPEN.

        Raises:
            CircuitOpenError: If circuit is OPEN and no fallback is provided.
        """
        cid = correlation_id or self.correlation_id
        if not self.allow_request():
            if fallback is not None:
                logger.warning(
                    "Circuit breaker '%s' OPEN — using fallback (correlation_id=%s)",
                    self.service_name,
                    cid,
                )
                return fallback()
            raise CircuitOpenError(self.service_name, self._failure_count)

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
            self.record_success(correlation_id=cid)
            return result
        except Exception as exc:
            self.record_failure(
                error_type=type(exc).__name__,
                correlation_id=cid,
            )
            raise

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of circuit breaker state for observability."""
        with self._lock:
            state = self._compute_state()
            return {
                "service": self.service_name,
                "state": state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state (useful for testing)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
        logger.debug("Circuit breaker '%s': manually reset to CLOSED", self.service_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state and log the change. Lock must be held."""
        if self._state == new_state:
            return
        old_state = self._state
        self._state = new_state
        logger.info(
            "Circuit breaker '%s': %s → %s (failure_count=%d, correlation_id=%s)",
            self.service_name,
            old_state.value,
            new_state.value,
            self._failure_count,
            self.correlation_id,
        )


# ---------------------------------------------------------------------------
# Registry — manage multiple circuit breakers by service name
# ---------------------------------------------------------------------------

class CircuitBreakerRegistry:
    """
    Registry for managing multiple CircuitBreaker instances by service name.

    Usage::

        registry = CircuitBreakerRegistry()
        cb = registry.get_or_create("llm_api", failure_threshold=3)
        result = await cb.call_async(call_llm, prompt)
    """

    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        service_name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        error_rate_tracker: Optional[ErrorRateTracker] = None,
    ) -> CircuitBreaker:
        """Get existing or create new CircuitBreaker for a service."""
        with self._lock:
            if service_name not in self._breakers:
                self._breakers[service_name] = CircuitBreaker(
                    service_name=service_name,
                    failure_threshold=failure_threshold,
                    cooldown_seconds=cooldown_seconds,
                    error_rate_tracker=error_rate_tracker,
                )
            return self._breakers[service_name]

    def get(self, service_name: str) -> Optional[CircuitBreaker]:
        """Return existing CircuitBreaker or None."""
        with self._lock:
            return self._breakers.get(service_name)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Return status snapshots for all registered circuit breakers."""
        with self._lock:
            names = list(self._breakers.keys())
        return {name: self._breakers[name].get_status() for name in names}

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        with self._lock:
            names = list(self._breakers.keys())
        for name in names:
            self._breakers[name].reset()


# ---------------------------------------------------------------------------
# Module-level singleton registry
# ---------------------------------------------------------------------------

_default_registry: Optional[CircuitBreakerRegistry] = None
_registry_lock = threading.Lock()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Return the module-level singleton CircuitBreakerRegistry."""
    global _default_registry
    with _registry_lock:
        if _default_registry is None:
            _default_registry = CircuitBreakerRegistry()
    return _default_registry


def reset_circuit_breaker_registry() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_registry
    with _registry_lock:
        _default_registry = None


__all__ = [
    "CircuitState",
    "CircuitOpenError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "reset_circuit_breaker_registry",
]
