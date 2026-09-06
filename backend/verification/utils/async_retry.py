"""
Async Retry Utility with Exponential Backoff - Task 6.2.5

Provides async retry strategies with exponential backoff, jitter, and
integration with the existing circuit breaker and error classifier.

Supports:
- RetryConfig Pydantic model (max_attempts, base_delay, max_delay, backoff_multiplier, jitter)
- RetryResult Pydantic model (success, attempts, final_exception, total_time)
- AsyncRetryStrategy class with configurable backoff
- async_retry decorator for wrapping async callables
- Integration with CircuitBreaker and ErrorClassifier for smart retry decisions

Requirements:
- 8.1: Error handling with logging and correlation IDs
- 8.2: LLM API timeout/error → retry with exponential backoff (max 3 attempts)
- 8.4: Circuit breaker pattern for external service calls
- 8.5: All errors logged with correlation IDs
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Type, Tuple

from pydantic import ConfigDict, BaseModel, Field

from .circuit_breaker import CircuitBreaker
from .error_classifier import ErrorClassifier, get_error_classifier

logger = logging.getLogger("backend.verification.async_retry")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class RetryConfig(BaseModel):
    """
    Configuration for async retry strategy.

    **Validates: Requirements 8.2** - exponential backoff with configurable max attempts
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts (including the first call)",
    )
    base_delay: float = Field(
        default=1.0,
        ge=0.0,
        le=60.0,
        description="Base delay in seconds before the first retry",
    )
    max_delay: float = Field(
        default=30.0,
        ge=0.0,
        le=300.0,
        description="Maximum delay cap in seconds between retries",
    )
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Multiplier applied to delay on each retry (exponential backoff)",
    )
    jitter: bool = Field(
        default=True,
        description="Add random jitter to delay to avoid thundering herd",
    )
    jitter_max_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Maximum jitter as a fraction of the computed delay",
    )
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "backoff_multiplier": 2.0,
            "jitter": True,
        }
    })


class RetryResult(BaseModel):
    """
    Result of an async retry operation.

    **Validates: Requirements 8.1** - structured error tracking with context
    """

    success: bool = Field(description="Whether the operation eventually succeeded")
    attempts: int = Field(ge=1, description="Total number of attempts made")
    final_exception: Optional[str] = Field(
        default=None,
        description="String representation of the final exception if all attempts failed",
    )
    total_time: float = Field(
        ge=0.0,
        description="Total elapsed time in seconds across all attempts and delays",
    )
    result: Optional[Any] = Field(
        default=None,
        description="Return value of the callable on success",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# AsyncRetryStrategy
# ---------------------------------------------------------------------------

class AsyncRetryStrategy:
    """
    Async retry strategy with exponential backoff and optional circuit breaker integration.

    Usage — standalone::

        strategy = AsyncRetryStrategy(RetryConfig(max_attempts=3, base_delay=1.0))
        result = await strategy.execute(call_llm, prompt, correlation_id="corr_abc")

    Usage — with circuit breaker::

        cb = CircuitBreaker("llm_api")
        strategy = AsyncRetryStrategy(config, circuit_breaker=cb)
        result = await strategy.execute(call_llm, prompt)

    **Validates: Requirements 8.2** - retry with exponential backoff, max 3 attempts
    **Validates: Requirements 8.4** - circuit breaker integration
    **Validates: Requirements 8.5** - all errors logged with correlation IDs
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        error_classifier: Optional[ErrorClassifier] = None,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> None:
        """
        Initialize the retry strategy.

        Args:
            config: RetryConfig with backoff parameters. Defaults to 3 attempts, 1s base delay.
            circuit_breaker: Optional CircuitBreaker to check before each attempt.
            error_classifier: Optional ErrorClassifier for smart retry decisions.
                              Uses module singleton if None.
            retryable_exceptions: Explicit tuple of exception types to retry.
                                  If None, uses error_classifier to decide.
            non_retryable_exceptions: Explicit tuple of exception types to never retry.
        """
        self.config = config or RetryConfig()
        self.circuit_breaker = circuit_breaker
        self.error_classifier = error_classifier or get_error_classifier()
        self.retryable_exceptions = retryable_exceptions
        self.non_retryable_exceptions = non_retryable_exceptions or ()

    def compute_delay(self, attempt: int) -> float:
        """
        Compute the delay before the next retry attempt.

        Args:
            attempt: Zero-based attempt index (0 = before first retry).

        Returns:
            Delay in seconds, capped at max_delay, with optional jitter.
        """
        delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_max_fraction
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay)

        return delay

    def _is_retryable(self, exc: Exception) -> bool:
        """
        Determine whether an exception should trigger a retry.

        Priority:
        1. If exception is in non_retryable_exceptions → False
        2. If retryable_exceptions is explicitly set → check membership
        3. Otherwise use ErrorClassifier.classify().is_retriable
        """
        if isinstance(exc, self.non_retryable_exceptions):
            return False

        if self.retryable_exceptions is not None:
            return isinstance(exc, self.retryable_exceptions)

        classified = self.error_classifier.classify(exc)
        return classified.is_retriable

    async def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RetryResult:
        """
        Execute an async callable with retry logic.

        Args:
            fn: Async or sync callable to invoke.
            *args: Positional arguments for fn.
            correlation_id: Optional correlation ID for logging.
            **kwargs: Keyword arguments for fn.

        Returns:
            RetryResult with success status, attempt count, and result or exception.
        """
        start_time = time.monotonic()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_attempts + 1):
            # Check circuit breaker before each attempt
            if self.circuit_breaker is not None and not self.circuit_breaker.allow_request():
                logger.warning(
                    "Circuit breaker OPEN — aborting retry (attempt=%d, correlation_id=%s)",
                    attempt,
                    correlation_id,
                )
                break

            try:
                logger.debug(
                    "Retry attempt %d/%d (correlation_id=%s)",
                    attempt,
                    self.config.max_attempts,
                    correlation_id,
                )

                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)

                # Record success in circuit breaker
                if self.circuit_breaker is not None:
                    self.circuit_breaker.record_success(correlation_id=correlation_id)

                total_time = time.monotonic() - start_time
                logger.info(
                    "Operation succeeded on attempt %d/%d (total_time=%.3fs, correlation_id=%s)",
                    attempt,
                    self.config.max_attempts,
                    total_time,
                    correlation_id,
                )
                return RetryResult(
                    success=True,
                    attempts=attempt,
                    total_time=total_time,
                    result=result,
                )

            except Exception as exc:
                last_exc = exc

                # Record failure in circuit breaker
                if self.circuit_breaker is not None:
                    self.circuit_breaker.record_failure(
                        error_type=type(exc).__name__,
                        correlation_id=correlation_id,
                    )

                is_retryable = self._is_retryable(exc)
                logger.warning(
                    "Attempt %d/%d failed: %s (retryable=%s, correlation_id=%s)",
                    attempt,
                    self.config.max_attempts,
                    exc,
                    is_retryable,
                    correlation_id,
                )

                if not is_retryable:
                    logger.error(
                        "Non-retryable error — aborting retry (correlation_id=%s): %s",
                        correlation_id,
                        exc,
                    )
                    break

                if attempt < self.config.max_attempts:
                    delay = self.compute_delay(attempt - 1)
                    logger.debug(
                        "Waiting %.3fs before retry %d (correlation_id=%s)",
                        delay,
                        attempt + 1,
                        correlation_id,
                    )
                    await asyncio.sleep(delay)

        total_time = time.monotonic() - start_time
        logger.error(
            "All %d attempts failed (total_time=%.3fs, correlation_id=%s): %s",
            self.config.max_attempts,
            total_time,
            correlation_id,
            last_exc,
        )
        return RetryResult(
            success=False,
            attempts=min(self.config.max_attempts, self.config.max_attempts),
            final_exception=str(last_exc) if last_exc else "Unknown error",
            total_time=total_time,
        )


# ---------------------------------------------------------------------------
# async_retry decorator
# ---------------------------------------------------------------------------

def async_retry(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    correlation_id: Optional[str] = None,
) -> Callable:
    """
    Decorator that wraps an async function with retry logic.

    Usage::

        @async_retry(config=RetryConfig(max_attempts=3, base_delay=1.0))
        async def call_llm(prompt: str) -> str:
            ...

        # Or with circuit breaker:
        cb = CircuitBreaker("llm_api")

        @async_retry(config=RetryConfig(max_attempts=3), circuit_breaker=cb)
        async def call_llm(prompt: str) -> str:
            ...

    The decorated function raises the last exception if all attempts fail.

    **Validates: Requirements 8.2** - retry with exponential backoff
    """
    strategy = AsyncRetryStrategy(
        config=config,
        circuit_breaker=circuit_breaker,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
    )

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cid = correlation_id
            result = await strategy.execute(fn, *args, correlation_id=cid, **kwargs)
            if result.success:
                return result.result
            raise RuntimeError(
                f"All {strategy.config.max_attempts} retry attempts failed: {result.final_exception}"
            )

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Convenience factory for LLM API retries (Requirement 8.2)
# ---------------------------------------------------------------------------

def create_llm_retry_strategy(
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> AsyncRetryStrategy:
    """
    Create a pre-configured retry strategy for LLM API calls.

    Uses 3 attempts with 1s base delay and exponential backoff as specified
    in Requirement 8.2.

    **Validates: Requirements 8.2** - LLM API timeout/error retry with exponential backoff
    """
    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
        jitter=True,
    )
    return AsyncRetryStrategy(config=config, circuit_breaker=circuit_breaker)


__all__ = [
    "RetryConfig",
    "RetryResult",
    "AsyncRetryStrategy",
    "async_retry",
    "create_llm_retry_strategy",
]
