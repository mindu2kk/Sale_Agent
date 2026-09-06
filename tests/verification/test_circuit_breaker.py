"""
Tests for CircuitBreaker - Task 6.2.2

Covers:
- Circuit starts CLOSED
- Circuit opens after threshold failures
- Circuit transitions to HALF_OPEN after cooldown
- Successful call in HALF_OPEN closes circuit
- Failed call in HALF_OPEN reopens circuit
- Fallback is called when circuit is OPEN
- Async operation support
- Registry management
- ErrorRateTracker integration
"""

import asyncio
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from backend.verification.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    get_circuit_breaker_registry,
    reset_circuit_breaker_registry,
)
from backend.verification.utils.error_rate_tracker import ErrorRateTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registry():
    reset_circuit_breaker_registry()
    yield
    reset_circuit_breaker_registry()


@pytest.fixture
def tracker():
    return ErrorRateTracker(window_seconds=60)


@pytest.fixture
def cb(tracker):
    """Circuit breaker with low threshold for easy testing."""
    return CircuitBreaker(
        service_name="test_service",
        failure_threshold=3,
        cooldown_seconds=60.0,
        error_rate_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_circuit_starts_closed(self, cb):
        assert cb.state == CircuitState.CLOSED

    def test_allow_request_when_closed(self, cb):
        assert cb.allow_request() is True

    def test_is_open_false_when_closed(self, cb):
        assert cb.is_open() is False

    def test_initial_failure_count_zero(self, cb):
        status = cb.get_status()
        assert status["failure_count"] == 0
        assert status["success_count"] == 0


# ---------------------------------------------------------------------------
# Opening the circuit
# ---------------------------------------------------------------------------

class TestCircuitOpening:
    def test_circuit_opens_after_threshold_failures(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")
        assert cb.state == CircuitState.OPEN

    def test_circuit_stays_closed_below_threshold(self, cb):
        for _ in range(2):
            cb.record_failure("TimeoutError")
        assert cb.state == CircuitState.CLOSED

    def test_allow_request_false_when_open(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")
        assert cb.allow_request() is False

    def test_is_open_true_when_open(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")
        assert cb.is_open() is True

    def test_success_resets_failure_count(self, cb):
        cb.record_failure("Error")
        cb.record_failure("Error")
        cb.record_success()
        # Failure count reset — need 3 more failures to open
        cb.record_failure("Error")
        cb.record_failure("Error")
        assert cb.state == CircuitState.CLOSED

    def test_failure_count_in_status(self, cb):
        cb.record_failure("Error")
        cb.record_failure("Error")
        status = cb.get_status()
        assert status["failure_count"] == 2


# ---------------------------------------------------------------------------
# HALF_OPEN transition
# ---------------------------------------------------------------------------

class TestHalfOpenTransition:
    def test_circuit_transitions_to_half_open_after_cooldown(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")
        assert cb.state == CircuitState.OPEN

        # Simulate cooldown elapsed
        with patch("time.time", return_value=time.time() + 61):
            assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_stays_open_before_cooldown(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")
        # No time advance — still OPEN
        assert cb.state == CircuitState.OPEN

    def test_allow_request_true_in_half_open(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")
        with patch("time.time", return_value=time.time() + 61):
            assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# HALF_OPEN → CLOSED (success)
# ---------------------------------------------------------------------------

class TestHalfOpenSuccess:
    def test_success_in_half_open_closes_circuit(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")

        # Force HALF_OPEN
        with patch("time.time", return_value=time.time() + 61):
            _ = cb.state  # trigger transition

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_count_reset_after_half_open_success(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")
        with patch("time.time", return_value=time.time() + 61):
            _ = cb.state
        cb.record_success()
        assert cb.get_status()["failure_count"] == 0


# ---------------------------------------------------------------------------
# HALF_OPEN → OPEN (failure)
# ---------------------------------------------------------------------------

class TestHalfOpenFailure:
    def test_failure_in_half_open_reopens_circuit(self, cb):
        for _ in range(3):
            cb.record_failure("TimeoutError")

        # Force HALF_OPEN
        with patch("time.time", return_value=time.time() + 61):
            _ = cb.state

        cb.record_failure("TimeoutError")
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Fallback logic
# ---------------------------------------------------------------------------

class TestFallbackLogic:
    def test_call_async_uses_fallback_when_open(self, cb):
        for _ in range(3):
            cb.record_failure("Error")

        async def run():
            return await cb.call_async(
                lambda: "real_result",
                fallback=lambda: "fallback_result",
            )

        result = asyncio.run(run())
        assert result == "fallback_result"

    def test_call_async_raises_circuit_open_error_without_fallback(self, cb):
        for _ in range(3):
            cb.record_failure("Error")

        async def run():
            return await cb.call_async(lambda: "real_result")

        with pytest.raises(CircuitOpenError) as exc_info:
            asyncio.run(run())
        assert exc_info.value.service_name == "test_service"

    def test_protect_context_manager_uses_fallback_when_open(self, cb):
        for _ in range(3):
            cb.record_failure("Error")

        async def run():
            async with cb.protect(fallback=lambda: "cached") as result:
                return result

        result = asyncio.run(run())
        assert result == "cached"

    def test_protect_raises_circuit_open_error_without_fallback(self, cb):
        for _ in range(3):
            cb.record_failure("Error")

        async def run():
            async with cb.protect():
                pass

        with pytest.raises(CircuitOpenError):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# Async operation support
# ---------------------------------------------------------------------------

class TestAsyncSupport:
    def test_call_async_with_async_function(self, cb):
        async def async_fn():
            return "async_result"

        async def run():
            return await cb.call_async(async_fn)

        result = asyncio.run(run())
        assert result == "async_result"

    def test_call_async_with_sync_function(self, cb):
        def sync_fn():
            return "sync_result"

        async def run():
            return await cb.call_async(sync_fn)

        result = asyncio.run(run())
        assert result == "sync_result"

    def test_call_async_records_success_on_success(self, cb):
        async def run():
            await cb.call_async(lambda: "ok")

        asyncio.run(run())
        assert cb.get_status()["success_count"] == 1
        assert cb.get_status()["failure_count"] == 0

    def test_call_async_records_failure_on_exception(self, cb):
        async def failing_fn():
            raise ValueError("boom")

        async def run():
            try:
                await cb.call_async(failing_fn)
            except ValueError:
                pass

        asyncio.run(run())
        assert cb.get_status()["failure_count"] == 1

    def test_call_async_opens_circuit_after_threshold_async_failures(self, cb):
        async def failing_fn():
            raise ConnectionError("timeout")

        async def run():
            for _ in range(3):
                try:
                    await cb.call_async(failing_fn)
                except ConnectionError:
                    pass

        asyncio.run(run())
        assert cb.state == CircuitState.OPEN

    def test_protect_context_manager_records_success(self, cb):
        async def run():
            async with cb.protect():
                pass  # no exception

        asyncio.run(run())
        assert cb.get_status()["success_count"] == 1

    def test_protect_context_manager_records_failure(self, cb):
        async def run():
            try:
                async with cb.protect():
                    raise RuntimeError("fail")
            except RuntimeError:
                pass

        asyncio.run(run())
        assert cb.get_status()["failure_count"] == 1

    def test_call_async_passes_args_and_kwargs(self, cb):
        async def fn(a, b, c=0):
            return a + b + c

        async def run():
            return await cb.call_async(fn, 1, 2, c=3)

        result = asyncio.run(run())
        assert result == 6


# ---------------------------------------------------------------------------
# ErrorRateTracker integration
# ---------------------------------------------------------------------------

class TestErrorRateTrackerIntegration:
    def test_failure_recorded_in_tracker(self, cb, tracker):
        cb.record_failure("TimeoutError")
        stats = tracker.get_component_stats("test_service")
        assert stats["error_count"] == 1

    def test_success_recorded_in_tracker(self, cb, tracker):
        cb.record_success()
        stats = tracker.get_component_stats("test_service")
        assert stats["success_count"] == 1

    def test_error_type_tracked(self, cb, tracker):
        cb.record_failure("TimeoutError")
        cb.record_failure("ConnectionError")
        stats = tracker.get_component_stats("test_service")
        assert stats["error_types"]["TimeoutError"] == 1
        assert stats["error_types"]["ConnectionError"] == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_closes_open_circuit(self, cb):
        for _ in range(3):
            cb.record_failure("Error")
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_reset_clears_failure_count(self, cb):
        cb.record_failure("Error")
        cb.record_failure("Error")
        cb.reset()
        assert cb.get_status()["failure_count"] == 0

    def test_reset_clears_success_count(self, cb):
        cb.record_success()
        cb.reset()
        assert cb.get_status()["success_count"] == 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_or_create_returns_same_instance(self):
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create("svc_a")
        cb2 = registry.get_or_create("svc_a")
        assert cb1 is cb2

    def test_get_or_create_different_services(self):
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create("svc_a")
        cb2 = registry.get_or_create("svc_b")
        assert cb1 is not cb2

    def test_get_returns_none_for_unknown_service(self):
        registry = CircuitBreakerRegistry()
        assert registry.get("unknown") is None

    def test_get_all_statuses(self):
        registry = CircuitBreakerRegistry()
        registry.get_or_create("svc_a")
        registry.get_or_create("svc_b")
        statuses = registry.get_all_statuses()
        assert "svc_a" in statuses
        assert "svc_b" in statuses

    def test_reset_all(self):
        registry = CircuitBreakerRegistry()
        cb = registry.get_or_create("svc_a", failure_threshold=2)
        cb.record_failure("Error")
        cb.record_failure("Error")
        assert cb.state == CircuitState.OPEN
        registry.reset_all()
        assert cb.state == CircuitState.CLOSED

    def test_singleton_registry(self):
        r1 = get_circuit_breaker_registry()
        r2 = get_circuit_breaker_registry()
        assert r1 is r2

    def test_reset_singleton_creates_new_instance(self):
        r1 = get_circuit_breaker_registry()
        reset_circuit_breaker_registry()
        r2 = get_circuit_breaker_registry()
        assert r1 is not r2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_failures_open_circuit_exactly_once(self, cb):
        """Multiple threads recording failures should not corrupt state."""
        errors = []

        def fail():
            for _ in range(5):
                cb.record_failure("ConcurrentError")

        threads = [threading.Thread(target=fail) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Circuit must be OPEN and failure_count must be consistent
        assert cb.state == CircuitState.OPEN
        assert cb.get_status()["failure_count"] >= cb.failure_threshold

    def test_concurrent_success_and_failure(self, cb):
        """Mixed concurrent operations should not deadlock."""
        stop = threading.Event()

        def recorder(is_error):
            for _ in range(20):
                if is_error:
                    cb.record_failure("Error")
                else:
                    cb.record_success()

        threads = [
            threading.Thread(target=recorder, args=(i % 2 == 0,))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No assertion on exact state — just verify no exception
        state = cb.state
        assert state in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN)
