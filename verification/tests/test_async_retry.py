"""
Tests for AsyncRetryStrategy and async_retry decorator - Task 6.2.5

Tests:
- Exponential backoff timing
- Max attempts enforcement
- Retryable vs non-retryable errors
- Jitter behavior
- Integration with circuit breaker

Requirements: 8.1, 8.2, 8.4, 8.5
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verification.utils.async_retry import (
    AsyncRetryStrategy,
    RetryConfig,
    RetryResult,
    async_retry,
    create_llm_retry_strategy,
)
from verification.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from verification.utils.error_classifier import ErrorCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RetryableError(ConnectionError):
    """Simulates a retryable connection error."""


class _NonRetryableError(ValueError):
    """Simulates a non-retryable validation error."""


async def _always_fail(*args, **kwargs):
    raise _RetryableError("always fails")


async def _fail_then_succeed(call_count: list):
    """Fails on first two calls, succeeds on third."""
    call_count.append(1)
    if len(call_count) < 3:
        raise _RetryableError(f"fail #{len(call_count)}")
    return "success"


# ---------------------------------------------------------------------------
# RetryConfig tests
# ---------------------------------------------------------------------------

class TestRetryConfig:
    def test_defaults(self):
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        config = RetryConfig(max_attempts=5, base_delay=0.5, max_delay=10.0, backoff_multiplier=3.0, jitter=False)
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.backoff_multiplier == 3.0
        assert config.jitter is False

    def test_max_attempts_bounds(self):
        with pytest.raises(Exception):
            RetryConfig(max_attempts=0)
        with pytest.raises(Exception):
            RetryConfig(max_attempts=11)


# ---------------------------------------------------------------------------
# Exponential backoff timing
# ---------------------------------------------------------------------------

class TestExponentialBackoff:
    def test_delay_increases_exponentially(self):
        config = RetryConfig(base_delay=1.0, backoff_multiplier=2.0, max_delay=100.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        delays = [strategy.compute_delay(i) for i in range(4)]
        # 1.0, 2.0, 4.0, 8.0
        assert delays[0] == pytest.approx(1.0)
        assert delays[1] == pytest.approx(2.0)
        assert delays[2] == pytest.approx(4.0)
        assert delays[3] == pytest.approx(8.0)

    def test_delay_capped_at_max(self):
        config = RetryConfig(base_delay=1.0, backoff_multiplier=10.0, max_delay=5.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        for i in range(5):
            assert strategy.compute_delay(i) <= 5.0

    def test_delay_non_negative(self):
        config = RetryConfig(base_delay=0.1, backoff_multiplier=2.0, jitter=True, jitter_max_fraction=0.5)
        strategy = AsyncRetryStrategy(config=config)

        for i in range(10):
            assert strategy.compute_delay(i) >= 0.0

    def test_jitter_adds_randomness(self):
        config = RetryConfig(base_delay=1.0, backoff_multiplier=2.0, jitter=True, jitter_max_fraction=0.25)
        strategy = AsyncRetryStrategy(config=config)

        delays = {strategy.compute_delay(0) for _ in range(20)}
        # With jitter, delays should vary
        assert len(delays) > 1

    def test_no_jitter_deterministic(self):
        config = RetryConfig(base_delay=1.0, backoff_multiplier=2.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        delays = {strategy.compute_delay(0) for _ in range(10)}
        assert len(delays) == 1  # All the same


# ---------------------------------------------------------------------------
# Max attempts enforcement
# ---------------------------------------------------------------------------

class TestMaxAttempts:
    @pytest.mark.asyncio
    async def test_stops_at_max_attempts(self):
        config = RetryConfig(max_attempts=3, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        call_count = []

        async def failing_fn():
            call_count.append(1)
            raise _RetryableError("fail")

        result = await strategy.execute(failing_fn)

        assert result.success is False
        assert result.attempts == 3
        assert len(call_count) == 3

    @pytest.mark.asyncio
    async def test_single_attempt_config(self):
        config = RetryConfig(max_attempts=1, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        call_count = []

        async def failing_fn():
            call_count.append(1)
            raise _RetryableError("fail")

        result = await strategy.execute(failing_fn)

        assert result.success is False
        assert len(call_count) == 1

    @pytest.mark.asyncio
    async def test_succeeds_before_max_attempts(self):
        config = RetryConfig(max_attempts=5, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        call_count = []
        result = await strategy.execute(_fail_then_succeed, call_count)

        assert result.success is True
        assert result.attempts == 3
        assert result.result == "success"

    @pytest.mark.asyncio
    async def test_result_contains_total_time(self):
        config = RetryConfig(max_attempts=2, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        result = await strategy.execute(_always_fail)

        assert result.total_time >= 0.0

    @pytest.mark.asyncio
    async def test_result_contains_final_exception(self):
        config = RetryConfig(max_attempts=2, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        result = await strategy.execute(_always_fail)

        assert result.success is False
        assert result.final_exception is not None
        assert "always fails" in result.final_exception


# ---------------------------------------------------------------------------
# Retryable vs non-retryable errors
# ---------------------------------------------------------------------------

class TestRetryableErrors:
    @pytest.mark.asyncio
    async def test_non_retryable_stops_immediately(self):
        config = RetryConfig(max_attempts=5, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        call_count = []

        async def non_retryable_fn():
            call_count.append(1)
            raise _NonRetryableError("validation error")

        result = await strategy.execute(non_retryable_fn)

        assert result.success is False
        assert len(call_count) == 1  # Only one attempt

    @pytest.mark.asyncio
    async def test_explicit_retryable_exceptions(self):
        config = RetryConfig(max_attempts=3, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(
            config=config,
            retryable_exceptions=(_RetryableError,),
        )

        call_count = []

        async def fn():
            call_count.append(1)
            raise _RetryableError("retryable")

        result = await strategy.execute(fn)

        assert len(call_count) == 3  # All 3 attempts made

    @pytest.mark.asyncio
    async def test_explicit_non_retryable_exceptions(self):
        config = RetryConfig(max_attempts=5, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(
            config=config,
            non_retryable_exceptions=(_RetryableError,),
        )

        call_count = []

        async def fn():
            call_count.append(1)
            raise _RetryableError("should not retry")

        result = await strategy.execute(fn)

        assert len(call_count) == 1  # Stopped immediately

    @pytest.mark.asyncio
    async def test_timeout_error_is_retryable(self):
        """TimeoutError should be classified as retryable (LLM_TIMEOUT)."""
        config = RetryConfig(max_attempts=3, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        call_count = []

        async def fn():
            call_count.append(1)
            raise TimeoutError("LLM timeout")

        result = await strategy.execute(fn)

        assert len(call_count) == 3  # All retries attempted

    @pytest.mark.asyncio
    async def test_value_error_is_not_retryable(self):
        """ValueError should be classified as non-retryable (VALIDATION_ERROR)."""
        config = RetryConfig(max_attempts=5, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        call_count = []

        async def fn():
            call_count.append(1)
            raise ValueError("invalid input")

        result = await strategy.execute(fn)

        assert len(call_count) == 1  # Stopped immediately

    @pytest.mark.asyncio
    async def test_sync_callable_supported(self):
        """AsyncRetryStrategy should also work with sync callables."""
        config = RetryConfig(max_attempts=3, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config)

        call_count = []

        def sync_fn():
            call_count.append(1)
            if len(call_count) < 2:
                raise _RetryableError("fail")
            return "sync_result"

        result = await strategy.execute(sync_fn)

        assert result.success is True
        assert result.result == "sync_result"
        assert len(call_count) == 2


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------

class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_aborts_retry(self):
        """When circuit is OPEN, retry should abort immediately."""
        cb = CircuitBreaker("test_service", failure_threshold=1, cooldown_seconds=9999)
        # Force circuit open
        cb.record_failure("TestError")

        config = RetryConfig(max_attempts=5, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config, circuit_breaker=cb)

        call_count = []

        async def fn():
            call_count.append(1)
            raise _RetryableError("fail")

        result = await strategy.execute(fn)

        # Circuit is open from the start, so no calls should be made
        assert len(call_count) == 0
        assert result.success is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self):
        """Successful call should record success in circuit breaker."""
        cb = CircuitBreaker("test_service")

        config = RetryConfig(max_attempts=3, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config, circuit_breaker=cb)

        async def fn():
            return "ok"

        result = await strategy.execute(fn)

        assert result.success is True
        assert cb._success_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure(self):
        """Failed calls should record failures in circuit breaker."""
        cb = CircuitBreaker("test_service", failure_threshold=10)

        config = RetryConfig(max_attempts=2, base_delay=0.0, jitter=False)
        strategy = AsyncRetryStrategy(config=config, circuit_breaker=cb)

        async def fn():
            raise _RetryableError("fail")

        result = await strategy.execute(fn)

        assert result.success is False
        assert cb._failure_count == 2  # One per attempt


# ---------------------------------------------------------------------------
# async_retry decorator
# ---------------------------------------------------------------------------

class TestAsyncRetryDecorator:
    @pytest.mark.asyncio
    async def test_decorator_retries_on_failure(self):
        call_count = []

        @async_retry(config=RetryConfig(max_attempts=3, base_delay=0.0, jitter=False))
        async def fn():
            call_count.append(1)
            if len(call_count) < 3:
                raise _RetryableError("fail")
            return "done"

        result = await fn()
        assert result == "done"
        assert len(call_count) == 3

    @pytest.mark.asyncio
    async def test_decorator_raises_on_all_failures(self):
        @async_retry(config=RetryConfig(max_attempts=2, base_delay=0.0, jitter=False))
        async def fn():
            raise _RetryableError("always fails")

        with pytest.raises(RuntimeError, match="retry attempts failed"):
            await fn()

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        @async_retry(config=RetryConfig(max_attempts=1))
        async def my_function():
            return "ok"

        assert my_function.__name__ == "my_function"


# ---------------------------------------------------------------------------
# create_llm_retry_strategy factory
# ---------------------------------------------------------------------------

class TestCreateLlmRetryStrategy:
    def test_creates_strategy_with_3_attempts(self):
        strategy = create_llm_retry_strategy()
        assert strategy.config.max_attempts == 3

    def test_creates_strategy_with_exponential_backoff(self):
        strategy = create_llm_retry_strategy()
        assert strategy.config.backoff_multiplier == 2.0
        assert strategy.config.base_delay == 1.0

    def test_creates_strategy_with_circuit_breaker(self):
        cb = CircuitBreaker("llm_api")
        strategy = create_llm_retry_strategy(circuit_breaker=cb)
        assert strategy.circuit_breaker is cb

    @pytest.mark.asyncio
    async def test_llm_strategy_retries_timeout(self):
        strategy = create_llm_retry_strategy()
        config = RetryConfig(max_attempts=3, base_delay=0.0, jitter=False)
        strategy.config = config  # Override delay for fast test

        call_count = []

        async def fn():
            call_count.append(1)
            if len(call_count) < 3:
                raise TimeoutError("LLM timeout")
            return "llm_response"

        result = await strategy.execute(fn)

        assert result.success is True
        assert result.result == "llm_response"
        assert len(call_count) == 3
