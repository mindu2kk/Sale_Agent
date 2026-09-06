"""
Rate Limiting and DDoS Protection with Async Request Handling - Task 7.2.5

Provides token-bucket-based rate limiting to prevent abuse and DDoS attacks:
- RateLimitExceededError — exception with retry_after_seconds attribute
- RateLimitConfig — Pydantic model (requests_per_second, requests_per_minute, burst_size)
- RateLimiter — per-client token bucket with sync and async check methods
- RateLimitMiddleware — decorator/context manager for workflow functions
- get_rate_limiter() — module-level singleton

Token Bucket Algorithm:
  Each client has an independent bucket that fills at `requests_per_second` tokens/sec
  up to `burst_size`. A request consumes one token; if the bucket is empty the request
  is rejected with a RateLimitExceededError that tells the caller how long to wait.

Requirements:
- 9.2: Support concurrent workflow execution (≥10 parallel workflows)
- 9.5: Monitor and alert on performance degradation
- Security: Rate limiting and DDoS protection
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("backend.verification.rate_limiter")


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RateLimitExceededError(Exception):
    """
    Raised when a client exceeds the configured rate limit.

    Attributes:
        client_id: The client whose limit was exceeded.
        retry_after_seconds: How many seconds to wait before retrying.
    """

    def __init__(self, client_id: str, retry_after_seconds: float) -> None:
        self.client_id = client_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Rate limit exceeded for client '{client_id}'. "
            f"Retry after {retry_after_seconds:.2f} seconds."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class RateLimitConfig(BaseModel):
    """
    Configuration for the token bucket rate limiter.

    Attributes:
        requests_per_second: Sustained token refill rate (tokens/sec).
        requests_per_minute: Optional per-minute cap (enforced as a secondary
            bucket). When None, only the per-second bucket is used.
        burst_size: Maximum bucket capacity (allows short bursts above the
            sustained rate). Defaults to requests_per_second * 2.
        stale_client_ttl_seconds: Seconds of inactivity before a client bucket
            is eligible for cleanup. Default 300 (5 minutes).
    """

    requests_per_second: float = Field(
        default=10.0,
        gt=0,
        description="Sustained token refill rate in tokens per second",
    )
    requests_per_minute: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional per-minute cap. None means no per-minute limit.",
    )
    burst_size: Optional[float] = Field(
        default=None,
        gt=0,
        description="Max bucket capacity. Defaults to requests_per_second * 2.",
    )
    stale_client_ttl_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Seconds of inactivity before a client bucket is cleaned up",
    )

    def effective_burst_size(self) -> float:
        """Return the effective burst size (explicit or derived)."""
        return self.burst_size if self.burst_size is not None else self.requests_per_second * 2


# ---------------------------------------------------------------------------
# Internal per-client bucket state
# ---------------------------------------------------------------------------


class _ClientBucket:
    """Thread-safe token bucket for a single client."""

    __slots__ = ("tokens", "last_refill", "minute_tokens", "minute_last_refill", "last_seen")

    def __init__(self, burst_size: float, minute_cap: Optional[float]) -> None:
        self.tokens: float = burst_size
        self.last_refill: float = time.monotonic()
        self.last_seen: float = time.monotonic()

        # Per-minute secondary bucket
        self.minute_tokens: Optional[float] = minute_cap
        self.minute_last_refill: float = time.monotonic()


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Per-client token bucket rate limiter with sync and async interfaces.

    Each client identified by a string ``client_id`` gets an independent
    token bucket.  Buckets are lazily created on first use and automatically
    cleaned up after ``config.stale_client_ttl_seconds`` of inactivity.

    Usage — sync::

        limiter = RateLimiter(RateLimitConfig(requests_per_second=5))
        limiter.check_rate_limit("user_123")   # raises RateLimitExceededError if exceeded

    Usage — async::

        await limiter.async_check_rate_limit("user_123")

    Usage — decorator::

        middleware = RateLimitMiddleware(limiter, client_id="api_client")

        @middleware.limit
        def handle_request(data):
            ...
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, _ClientBucket] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_bucket(self, client_id: str) -> _ClientBucket:
        """Return the bucket for *client_id*, creating it if necessary."""
        if client_id not in self._buckets:
            self._buckets[client_id] = _ClientBucket(
                burst_size=self.config.effective_burst_size(),
                minute_cap=self.config.requests_per_minute,
            )
        return self._buckets[client_id]

    def _refill_bucket(self, bucket: _ClientBucket) -> None:
        """Refill *bucket* based on elapsed time since last refill."""
        now = time.monotonic()

        # Per-second bucket refill
        elapsed = now - bucket.last_refill
        new_tokens = elapsed * self.config.requests_per_second
        bucket.tokens = min(
            self.config.effective_burst_size(),
            bucket.tokens + new_tokens,
        )
        bucket.last_refill = now

        # Per-minute secondary bucket refill
        if self.config.requests_per_minute is not None and bucket.minute_tokens is not None:
            minute_elapsed = now - bucket.minute_last_refill
            minute_new = minute_elapsed * (self.config.requests_per_minute / 60.0)
            bucket.minute_tokens = min(
                self.config.requests_per_minute,
                bucket.minute_tokens + minute_new,
            )
            bucket.minute_last_refill = now

    def _consume_token(self, client_id: str) -> float:
        """
        Attempt to consume one token for *client_id*.

        Returns:
            0.0 if the token was consumed successfully.
            Positive float (seconds to wait) if the bucket is empty.
        """
        bucket = self._get_or_create_bucket(client_id)
        self._refill_bucket(bucket)
        bucket.last_seen = time.monotonic()

        # Check per-second bucket
        if bucket.tokens < 1.0:
            wait = (1.0 - bucket.tokens) / self.config.requests_per_second
            logger.debug(
                "Rate limit exceeded for client '%s' (per-second). Wait %.3fs",
                client_id,
                wait,
            )
            return wait

        # Check per-minute bucket
        if self.config.requests_per_minute is not None and bucket.minute_tokens is not None:
            if bucket.minute_tokens < 1.0:
                wait = (1.0 - bucket.minute_tokens) / (self.config.requests_per_minute / 60.0)
                logger.debug(
                    "Rate limit exceeded for client '%s' (per-minute). Wait %.3fs",
                    client_id,
                    wait,
                )
                return wait

        # Consume token
        bucket.tokens -= 1.0
        if self.config.requests_per_minute is not None and bucket.minute_tokens is not None:
            bucket.minute_tokens -= 1.0

        return 0.0

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def check_rate_limit(self, client_id: str) -> None:
        """
        Synchronous rate limit check for *client_id*.

        Args:
            client_id: Unique identifier for the client/caller.

        Raises:
            RateLimitExceededError: If the client has exceeded the rate limit,
                with ``retry_after_seconds`` set to the wait time.
        """
        with self._lock:
            wait = self._consume_token(client_id)

        if wait > 0.0:
            raise RateLimitExceededError(client_id=client_id, retry_after_seconds=wait)

        logger.debug("Rate limit check passed for client '%s'", client_id)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def async_check_rate_limit(self, client_id: str) -> None:
        """
        Asynchronous rate limit check for *client_id*.

        Runs the token-bucket check in a thread-safe manner and raises
        :class:`RateLimitExceededError` if the limit is exceeded.

        Args:
            client_id: Unique identifier for the client/caller.

        Raises:
            RateLimitExceededError: If the client has exceeded the rate limit.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.check_rate_limit, client_id)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_stale_clients(self) -> int:
        """
        Remove buckets for clients that have been inactive longer than
        ``config.stale_client_ttl_seconds``.

        Returns:
            Number of client buckets removed.
        """
        now = time.monotonic()
        ttl = self.config.stale_client_ttl_seconds
        removed = 0

        with self._lock:
            stale = [
                cid
                for cid, bucket in self._buckets.items()
                if (now - bucket.last_seen) > ttl
            ]
            for cid in stale:
                del self._buckets[cid]
                removed += 1

        if removed:
            logger.info("Cleaned up %d stale client bucket(s)", removed)

        return removed

    def client_count(self) -> int:
        """Return the number of tracked client buckets."""
        with self._lock:
            return len(self._buckets)

    def reset_client(self, client_id: str) -> None:
        """Remove the bucket for *client_id* (useful for testing)."""
        with self._lock:
            self._buckets.pop(client_id, None)


# ---------------------------------------------------------------------------
# RateLimitMiddleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware:
    """
    Decorator and context manager that applies rate limiting to workflow functions.

    Usage — function decorator::

        limiter = RateLimiter(RateLimitConfig(requests_per_second=5))
        middleware = RateLimitMiddleware(limiter, client_id="default")

        @middleware.limit
        def handle_request(data):
            ...

        @middleware.async_limit
        async def handle_async_request(data):
            ...

    Usage — sync context manager::

        with middleware.context("user_123"):
            process_request()

    Usage — async context manager::

        async with middleware.async_context("user_123"):
            await process_request()
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        client_id: str = "default",
    ) -> None:
        self.rate_limiter = rate_limiter
        self.client_id = client_id

    # ------------------------------------------------------------------
    # Sync decorator
    # ------------------------------------------------------------------

    def limit(self, func: Callable) -> Callable:
        """
        Decorator that applies sync rate limiting before calling *func*.

        Raises:
            RateLimitExceededError: If the rate limit is exceeded.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.rate_limiter.check_rate_limit(self.client_id)
            return func(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Async decorator
    # ------------------------------------------------------------------

    def async_limit(self, func: Callable) -> Callable:
        """
        Decorator that applies async rate limiting before calling *func*.

        Raises:
            RateLimitExceededError: If the rate limit is exceeded.
        """

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await self.rate_limiter.async_check_rate_limit(self.client_id)
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Sync context manager
    # ------------------------------------------------------------------

    @contextmanager
    def context(self, client_id: Optional[str] = None):
        """
        Sync context manager that checks the rate limit on entry.

        Args:
            client_id: Override the middleware's default client_id.

        Raises:
            RateLimitExceededError: If the rate limit is exceeded.
        """
        cid = client_id or self.client_id
        self.rate_limiter.check_rate_limit(cid)
        yield

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def async_context(self, client_id: Optional[str] = None):
        """
        Async context manager that checks the rate limit on entry.

        Args:
            client_id: Override the middleware's default client_id.

        Raises:
            RateLimitExceededError: If the rate limit is exceeded.
        """
        cid = client_id or self.client_id
        await self.rate_limiter.async_check_rate_limit(cid)
        yield


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_rate_limiter: Optional[RateLimiter] = None
_singleton_lock = threading.Lock()


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """
    Return the module-level singleton :class:`RateLimiter`.

    Creates the instance on first call.  If *config* is provided on the first
    call it is used to initialise the singleton; subsequent calls ignore it.

    Args:
        config: Optional :class:`RateLimitConfig` for first-time initialisation.

    Returns:
        The singleton :class:`RateLimiter` instance.
    """
    global _default_rate_limiter
    with _singleton_lock:
        if _default_rate_limiter is None:
            _default_rate_limiter = RateLimiter(config)
    return _default_rate_limiter


def reset_rate_limiter() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_rate_limiter
    with _singleton_lock:
        _default_rate_limiter = None


__all__ = [
    "RateLimitExceededError",
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitMiddleware",
    "get_rate_limiter",
    "reset_rate_limiter",
]
