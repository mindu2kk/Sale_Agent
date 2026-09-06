"""
Tests for RateLimiter - Task 7.2.5

Covers:
- RateLimitExceededError exception structure
- RateLimitConfig Pydantic model validation
- RateLimiter token bucket algorithm (sync and async)
- Per-client independent buckets
- Burst size behaviour
- Per-minute secondary bucket
- Stale client cleanup
- RateLimitMiddleware decorator (sync and async)
- RateLimitMiddleware context managers (sync and async)
- Singleton factory get_rate_limiter()
"""

import asyncio
import time

import pytest

from backend.verification.utils.rate_limiter import (
    RateLimitConfig,
    RateLimitExceededError,
    RateLimitMiddleware,
    RateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_rate_limiter()
    yield
    reset_rate_limiter()


@pytest.fixture
def config():
    """Config with burst_size=3 and 10 req/s for fast tests."""
    return RateLimitConfig(requests_per_second=10.0, burst_size=3.0)


@pytest.fixture
def limiter(config):
    return RateLimiter(config)


@pytest.fixture
def middleware(limiter):
    return RateLimitMiddleware(limiter, client_id="test_client")


# ---------------------------------------------------------------------------
# RateLimitExceededError
# ---------------------------------------------------------------------------


class TestRateLimitExceededError:
    def test_is_exception(self):
        err = RateLimitExceededError("client_1", 1.5)
        assert isinstance(err, Exception)

    def test_client_id_attribute(self):
        err = RateLimitExceededError("client_1", 1.5)
        assert err.client_id == "client_1"

    def test_retry_after_seconds_attribute(self):
        err = RateLimitExceededError("client_1", 2.0)
        assert err.retry_after_seconds == 2.0

    def test_str_contains_client_id(self):
        err = RateLimitExceededError("client_abc", 1.0)
        assert "client_abc" in str(err)

    def test_str_contains_retry_after(self):
        err = RateLimitExceededError("client_abc", 3.5)
        assert "3.50" in str(err)

    def test_retry_after_is_float(self):
        err = RateLimitExceededError("c", 0.5)
        assert isinstance(err.retry_after_seconds, float)


# ---------------------------------------------------------------------------
# RateLimitConfig
# ---------------------------------------------------------------------------


class TestRateLimitConfig:
    def test_default_requests_per_second(self):
        cfg = RateLimitConfig()
        assert cfg.requests_per_second == 10.0

    def test_default_requests_per_minute_is_none(self):
        cfg = RateLimitConfig()
        assert cfg.requests_per_minute is None

    def test_default_burst_size_is_none(self):
        cfg = RateLimitConfig()
        assert cfg.burst_size is None

    def test_effective_burst_size_default(self):
        cfg = RateLimitConfig(requests_per_second=5.0)
        assert cfg.effective_burst_size() == 10.0  # 5 * 2

    def test_effective_burst_size_explicit(self):
        cfg = RateLimitConfig(requests_per_second=5.0, burst_size=20.0)
        assert cfg.effective_burst_size() == 20.0

    def test_custom_requests_per_second(self):
        cfg = RateLimitConfig(requests_per_second=100.0)
        assert cfg.requests_per_second == 100.0

    def test_custom_requests_per_minute(self):
        cfg = RateLimitConfig(requests_per_minute=60.0)
        assert cfg.requests_per_minute == 60.0

    def test_invalid_requests_per_second_zero(self):
        with pytest.raises(Exception):
            RateLimitConfig(requests_per_second=0)

    def test_invalid_requests_per_second_negative(self):
        with pytest.raises(Exception):
            RateLimitConfig(requests_per_second=-1.0)

    def test_stale_client_ttl_default(self):
        cfg = RateLimitConfig()
        assert cfg.stale_client_ttl_seconds == 300.0


# ---------------------------------------------------------------------------
# RateLimiter — basic token bucket
# ---------------------------------------------------------------------------


class TestRateLimiterBasic:
    def test_first_request_passes(self, limiter):
        # Should not raise
        limiter.check_rate_limit("client_1")

    def test_burst_requests_pass(self, limiter):
        # burst_size=3, so 3 consecutive requests should pass
        for _ in range(3):
            limiter.check_rate_limit("client_1")

    def test_exceeding_burst_raises(self, limiter):
        # Exhaust the burst bucket (3 tokens)
        for _ in range(3):
            limiter.check_rate_limit("client_1")
        # 4th request should fail
        with pytest.raises(RateLimitExceededError):
            limiter.check_rate_limit("client_1")

    def test_exceeded_error_has_client_id(self, limiter):
        for _ in range(3):
            limiter.check_rate_limit("client_x")
        with pytest.raises(RateLimitExceededError) as exc_info:
            limiter.check_rate_limit("client_x")
        assert exc_info.value.client_id == "client_x"

    def test_exceeded_error_has_positive_retry_after(self, limiter):
        for _ in range(3):
            limiter.check_rate_limit("client_x")
        with pytest.raises(RateLimitExceededError) as exc_info:
            limiter.check_rate_limit("client_x")
        assert exc_info.value.retry_after_seconds > 0

    def test_tokens_refill_over_time(self):
        """After waiting, tokens should refill and allow new requests."""
        cfg = RateLimitConfig(requests_per_second=100.0, burst_size=1.0)
        lim = RateLimiter(cfg)
        lim.check_rate_limit("client_1")  # consume the 1 token
        with pytest.raises(RateLimitExceededError):
            lim.check_rate_limit("client_1")
        # Wait for refill (1 token at 100 req/s = 0.01s)
        time.sleep(0.02)
        lim.check_rate_limit("client_1")  # should pass now

    def test_default_config_used_when_none(self):
        lim = RateLimiter()
        # Default burst = 10*2 = 20, so 20 requests should pass
        for _ in range(20):
            lim.check_rate_limit("c")


# ---------------------------------------------------------------------------
# RateLimiter — per-client independence
# ---------------------------------------------------------------------------


class TestRateLimiterPerClient:
    def test_different_clients_independent(self, limiter):
        # Exhaust client_a
        for _ in range(3):
            limiter.check_rate_limit("client_a")
        with pytest.raises(RateLimitExceededError):
            limiter.check_rate_limit("client_a")
        # client_b should still have full bucket
        limiter.check_rate_limit("client_b")

    def test_multiple_clients_tracked(self, limiter):
        limiter.check_rate_limit("a")
        limiter.check_rate_limit("b")
        limiter.check_rate_limit("c")
        assert limiter.client_count() == 3

    def test_reset_client_removes_bucket(self, limiter):
        limiter.check_rate_limit("client_1")
        limiter.reset_client("client_1")
        assert limiter.client_count() == 0

    def test_reset_nonexistent_client_is_safe(self, limiter):
        limiter.reset_client("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# RateLimiter — per-minute secondary bucket
# ---------------------------------------------------------------------------


class TestRateLimiterPerMinute:
    def test_per_minute_limit_enforced(self):
        # 3 req/min = 1 token per 20s; burst_size=3 so we can burst 3 then minute bucket runs out
        cfg = RateLimitConfig(
            requests_per_second=100.0,
            requests_per_minute=3.0,
            burst_size=3.0,
        )
        lim = RateLimiter(cfg)
        for _ in range(3):
            lim.check_rate_limit("c")
        with pytest.raises(RateLimitExceededError):
            lim.check_rate_limit("c")

    def test_per_minute_error_has_positive_retry_after(self):
        cfg = RateLimitConfig(
            requests_per_second=100.0,
            requests_per_minute=2.0,
            burst_size=2.0,
        )
        lim = RateLimiter(cfg)
        for _ in range(2):
            lim.check_rate_limit("c")
        with pytest.raises(RateLimitExceededError) as exc_info:
            lim.check_rate_limit("c")
        assert exc_info.value.retry_after_seconds > 0


# ---------------------------------------------------------------------------
# RateLimiter — stale client cleanup
# ---------------------------------------------------------------------------


class TestRateLimiterCleanup:
    def test_cleanup_removes_stale_clients(self):
        cfg = RateLimitConfig(requests_per_second=10.0, burst_size=5.0, stale_client_ttl_seconds=0.01)
        lim = RateLimiter(cfg)
        lim.check_rate_limit("stale_client")
        assert lim.client_count() == 1
        time.sleep(0.05)
        removed = lim.cleanup_stale_clients()
        assert removed == 1
        assert lim.client_count() == 0

    def test_cleanup_keeps_active_clients(self):
        cfg = RateLimitConfig(requests_per_second=10.0, burst_size=5.0, stale_client_ttl_seconds=60.0)
        lim = RateLimiter(cfg)
        lim.check_rate_limit("active_client")
        removed = lim.cleanup_stale_clients()
        assert removed == 0
        assert lim.client_count() == 1

    def test_cleanup_returns_count(self):
        cfg = RateLimitConfig(requests_per_second=10.0, burst_size=5.0, stale_client_ttl_seconds=0.01)
        lim = RateLimiter(cfg)
        for i in range(3):
            lim.check_rate_limit(f"client_{i}")
        time.sleep(0.05)
        removed = lim.cleanup_stale_clients()
        assert removed == 3

    def test_cleanup_no_stale_returns_zero(self, limiter):
        removed = limiter.cleanup_stale_clients()
        assert removed == 0


# ---------------------------------------------------------------------------
# RateLimiter — async API
# ---------------------------------------------------------------------------


class TestRateLimiterAsync:
    def test_async_check_passes(self, limiter):
        asyncio.get_event_loop().run_until_complete(
            limiter.async_check_rate_limit("async_client")
        )

    def test_async_check_raises_when_exceeded(self, limiter):
        async def run():
            for _ in range(3):
                await limiter.async_check_rate_limit("async_client")
            with pytest.raises(RateLimitExceededError):
                await limiter.async_check_rate_limit("async_client")

        asyncio.get_event_loop().run_until_complete(run())

    def test_async_check_error_has_client_id(self, limiter):
        async def run():
            for _ in range(3):
                await limiter.async_check_rate_limit("async_x")
            with pytest.raises(RateLimitExceededError) as exc_info:
                await limiter.async_check_rate_limit("async_x")
            assert exc_info.value.client_id == "async_x"

        asyncio.get_event_loop().run_until_complete(run())

    def test_async_concurrent_requests(self):
        """Multiple concurrent async requests should be handled safely."""
        cfg = RateLimitConfig(requests_per_second=100.0, burst_size=50.0)
        lim = RateLimiter(cfg)

        async def run():
            tasks = [lim.async_check_rate_limit("concurrent") for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # All 10 should pass since burst_size=50
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0

        asyncio.get_event_loop().run_until_complete(run())


# ---------------------------------------------------------------------------
# RateLimitMiddleware — sync decorator
# ---------------------------------------------------------------------------


class TestRateLimitMiddlewareDecorator:
    def test_limit_decorator_allows_call(self, middleware):
        results = []

        @middleware.limit
        def fn():
            results.append("called")

        fn()
        assert results == ["called"]

    def test_limit_decorator_raises_when_exceeded(self, middleware):
        # Exhaust the bucket (burst_size=3)
        for _ in range(3):
            middleware.rate_limiter.check_rate_limit("test_client")

        @middleware.limit
        def fn():
            pass

        with pytest.raises(RateLimitExceededError):
            fn()

    def test_limit_decorator_preserves_function_name(self, middleware):
        @middleware.limit
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_limit_decorator_passes_args(self, middleware):
        @middleware.limit
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_limit_decorator_passes_kwargs(self, middleware):
        @middleware.limit
        def greet(name="world"):
            return f"hello {name}"

        assert greet(name="test") == "hello test"


# ---------------------------------------------------------------------------
# RateLimitMiddleware — async decorator
# ---------------------------------------------------------------------------


class TestRateLimitMiddlewareAsyncDecorator:
    def test_async_limit_decorator_allows_call(self, middleware):
        async def run():
            results = []

            @middleware.async_limit
            async def fn():
                results.append("called")

            await fn()
            assert results == ["called"]

        asyncio.get_event_loop().run_until_complete(run())

    def test_async_limit_decorator_raises_when_exceeded(self, middleware):
        async def run():
            for _ in range(3):
                await middleware.rate_limiter.async_check_rate_limit("test_client")

            @middleware.async_limit
            async def fn():
                pass

            with pytest.raises(RateLimitExceededError):
                await fn()

        asyncio.get_event_loop().run_until_complete(run())

    def test_async_limit_preserves_function_name(self, middleware):
        @middleware.async_limit
        async def my_async_fn():
            pass

        assert my_async_fn.__name__ == "my_async_fn"

    def test_async_limit_wraps_sync_function(self, middleware):
        """async_limit should also work on sync functions."""
        async def run():
            results = []

            @middleware.async_limit
            def sync_fn():
                results.append("sync_called")

            await sync_fn()
            assert results == ["sync_called"]

        asyncio.get_event_loop().run_until_complete(run())


# ---------------------------------------------------------------------------
# RateL

# ---------------------------------------------------------------------------
# RateLimitMiddleware — sync context manager
# ---------------------------------------------------------------------------


class TestRateLimitMiddlewareContext:
    def test_context_allows_entry(self, middleware):
        with middleware.context():
            pass  # should not raise

    def test_context_raises_when_exceeded(self, middleware):
        for _ in range(3):
            middleware.rate_limiter.check_rate_limit("test_client")
        with pytest.raises(RateLimitExceededError):
            with middleware.context():
                pass

    def test_context_with_override_client_id(self, limiter):
        mw = RateLimitMiddleware(limiter, client_id="default_client")
        # Exhaust "override_client" bucket
        for _ in range(3):
            limiter.check_rate_limit("override_client")
        with pytest.raises(RateLimitExceededError):
            with mw.context("override_client"):
                pass
        # default_client should still work
        with mw.context("default_client"):
            pass

    def test_context_body_executes(self, middleware):
        executed = []
        with middleware.context():
            executed.append(True)
        assert executed == [True]


# ---------------------------------------------------------------------------
# RateLimitMiddleware — async context manager
# ---------------------------------------------------------------------------


class TestRateLimitMiddlewareAsyncContext:
    def test_async_context_allows_entry(self, middleware):
        async def run():
            async with middleware.async_context():
                pass  # should not raise

        asyncio.get_event_loop().run_until_complete(run())

    def test_async_context_raises_when_exceeded(self, middleware):
        async def run():
            for _ in range(3):
                await middleware.rate_limiter.async_check_rate_limit("test_client")
            with pytest.raises(RateLimitExceededError):
                async with middleware.async_context():
                    pass

        asyncio.get_event_loop().run_until_complete(run())

    def test_async_context_with_override_client_id(self, limiter):
        async def run():
            mw = RateLimitMiddleware(limiter, client_id="default_client")
            for _ in range(3):
                await limiter.async_check_rate_limit("override_client")
            with pytest.raises(RateLimitExceededError):
                async with mw.async_context("override_client"):
                    pass
            # default_client should still work
            async with mw.async_context("default_client"):
                pass

        asyncio.get_event_loop().run_until_complete(run())

    def test_async_context_body_executes(self, middleware):
        async def run():
            executed = []
            async with middleware.async_context():
                executed.append(True)
            assert executed == [True]

        asyncio.get_event_loop().run_until_complete(run())


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_rate_limiter_returns_same_instance(self):
        l1 = get_rate_limiter()
        l2 = get_rate_limiter()
        assert l1 is l2

    def test_reset_creates_new_instance(self):
        l1 = get_rate_limiter()
        reset_rate_limiter()
        l2 = get_rate_limiter()
        assert l1 is not l2

    def test_singleton_is_rate_limiter(self):
        assert isinstance(get_rate_limiter(), RateLimiter)

    def test_singleton_with_custom_config(self):
        cfg = RateLimitConfig(requests_per_second=50.0)
        lim = get_rate_limiter(cfg)
        assert lim.config.requests_per_second == 50.0

    def test_singleton_config_not_overridden_on_second_call(self):
        cfg1 = RateLimitConfig(requests_per_second=50.0)
        get_rate_limiter(cfg1)
        cfg2 = RateLimitConfig(requests_per_second=99.0)
        lim = get_rate_limiter(cfg2)
        # First config wins
        assert lim.config.requests_per_second == 50.0

    def test_singleton_functional(self):
        lim = get_rate_limiter()
        lim.check_rate_limit("singleton_client")  # should not raise
