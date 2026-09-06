"""
Tests for ShutdownManager - Task 7.3.3

Covers:
- register_task / register_cleanup_handler
- shutdown() cancels tasks and runs cleanup handlers
- Timeout handling: tasks that don't cancel are force-killed
- is_shutting_down property
- Duplicate shutdown() calls are ignored
- lifespan_with_shutdown context manager
- Module-level singleton helpers

Requirements: 8.1, 8.3
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.verification.utils.graceful_shutdown import (
    ShutdownManager,
    get_shutdown_manager,
    lifespan_with_shutdown,
    reset_shutdown_manager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _noop() -> None:
    await asyncio.sleep(0)


async def _long_running(duration: float = 60.0) -> None:
    """Simulates a long-running task that respects cancellation."""
    await asyncio.sleep(duration)


async def _stubborn_task() -> None:
    """Task that catches CancelledError and keeps running briefly."""
    try:
        await asyncio.sleep(60.0)
    except asyncio.CancelledError:
        # Simulate slow cleanup — but eventually finish
        await asyncio.sleep(0)
        raise


# ---------------------------------------------------------------------------
# ShutdownManager — basic state
# ---------------------------------------------------------------------------

class TestShutdownManagerState:
    def test_not_shutting_down_initially(self):
        manager = ShutdownManager()
        assert manager.is_shutting_down is False

    @pytest.mark.asyncio
    async def test_is_shutting_down_after_shutdown(self):
        manager = ShutdownManager()
        await manager.shutdown(timeout_seconds=1.0)
        assert manager.is_shutting_down is True


# ---------------------------------------------------------------------------
# register_task
# ---------------------------------------------------------------------------

class TestRegisterTask:
    @pytest.mark.asyncio
    async def test_register_task_adds_to_internal_set(self):
        manager = ShutdownManager()
        task = asyncio.create_task(_noop())
        manager.register_task(task)
        assert task in manager._tasks
        await task  # let it finish

    @pytest.mark.asyncio
    async def test_completed_task_removed_automatically(self):
        manager = ShutdownManager()
        task = asyncio.create_task(_noop())
        manager.register_task(task)
        await task  # task completes → done callback fires
        assert task not in manager._tasks

    @pytest.mark.asyncio
    async def test_multiple_tasks_registered(self):
        manager = ShutdownManager()
        tasks = [asyncio.create_task(_noop()) for _ in range(3)]
        for t in tasks:
            manager.register_task(t)
        assert len(manager._tasks) == 3
        await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# register_cleanup_handler
# ---------------------------------------------------------------------------

class TestRegisterCleanupHandler:
    def test_register_sync_handler(self):
        manager = ShutdownManager()
        handler = MagicMock()
        manager.register_cleanup_handler(handler)
        assert handler in manager._cleanup_handlers

    def test_register_multiple_handlers(self):
        manager = ShutdownManager()
        h1, h2 = MagicMock(), MagicMock()
        manager.register_cleanup_handler(h1)
        manager.register_cleanup_handler(h2)
        assert manager._cleanup_handlers == [h1, h2]


# ---------------------------------------------------------------------------
# shutdown() — task cancellation
# ---------------------------------------------------------------------------

class TestShutdownTaskCancellation:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_running_task(self):
        manager = ShutdownManager()
        task = asyncio.create_task(_long_running(60.0))
        manager.register_task(task)

        await manager.shutdown(timeout_seconds=2.0)

        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_shutdown_with_no_tasks(self):
        """Shutdown with no tasks should complete without error."""
        manager = ShutdownManager()
        await manager.shutdown(timeout_seconds=1.0)
        assert manager.is_shutting_down is True

    @pytest.mark.asyncio
    async def test_shutdown_cancels_multiple_tasks(self):
        manager = ShutdownManager()
        tasks = [asyncio.create_task(_long_running(60.0)) for _ in range(3)]
        for t in tasks:
            manager.register_task(t)

        await manager.shutdown(timeout_seconds=2.0)

        for t in tasks:
            assert t.cancelled() or t.done()

    @pytest.mark.asyncio
    async def test_already_done_tasks_not_cancelled(self):
        """Tasks that finish before shutdown are not double-cancelled."""
        manager = ShutdownManager()
        task = asyncio.create_task(_noop())
        manager.register_task(task)
        await task  # let it finish naturally

        # Should not raise
        await manager.shutdown(timeout_seconds=1.0)


# ---------------------------------------------------------------------------
# shutdown() — cleanup handlers
# ---------------------------------------------------------------------------

class TestShutdownCleanupHandlers:
    @pytest.mark.asyncio
    async def test_sync_cleanup_handler_called(self):
        manager = ShutdownManager()
        handler = MagicMock()
        manager.register_cleanup_handler(handler)

        await manager.shutdown(timeout_seconds=1.0)

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_cleanup_handler_called(self):
        manager = ShutdownManager()
        handler = AsyncMock()
        manager.register_cleanup_handler(handler)

        await manager.shutdown(timeout_seconds=1.0)

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handlers_called_in_order(self):
        manager = ShutdownManager()
        call_order = []

        def h1():
            call_order.append("h1")

        def h2():
            call_order.append("h2")

        manager.register_cleanup_handler(h1)
        manager.register_cleanup_handler(h2)

        await manager.shutdown(timeout_seconds=1.0)

        assert call_order == ["h1", "h2"]

    @pytest.mark.asyncio
    async def test_failing_handler_does_not_abort_shutdown(self):
        """A handler that raises should not prevent other handlers from running."""
        manager = ShutdownManager()
        bad_handler = MagicMock(side_effect=RuntimeError("boom"))
        good_handler = MagicMock()

        manager.register_cleanup_handler(bad_handler)
        manager.register_cleanup_handler(good_handler)

        # Should not raise
        await manager.shutdown(timeout_seconds=1.0)

        good_handler.assert_called_once()


# ---------------------------------------------------------------------------
# shutdown() — duplicate call guard
# ---------------------------------------------------------------------------

class TestShutdownDuplicateCall:
    @pytest.mark.asyncio
    async def test_duplicate_shutdown_ignored(self):
        manager = ShutdownManager()
        handler = MagicMock()
        manager.register_cleanup_handler(handler)

        await manager.shutdown(timeout_seconds=1.0)
        await manager.shutdown(timeout_seconds=1.0)  # second call — should be ignored

        # Handler should only be called once
        handler.assert_called_once()


# ---------------------------------------------------------------------------
# shutdown() — timeout / force-kill
# ---------------------------------------------------------------------------

class TestShutdownTimeout:
    @pytest.mark.asyncio
    async def test_force_kills_task_after_timeout(self):
        """
        A task that ignores cancellation should be force-killed after timeout.
        We use a very short timeout to trigger the force-kill path.
        """
        manager = ShutdownManager()

        async def unkillable():
            while True:
                try:
                    await asyncio.sleep(60.0)
                except asyncio.CancelledError:
                    # Swallow cancellation — keep running
                    continue

        task = asyncio.create_task(unkillable())
        manager.register_task(task)

        # Very short timeout to trigger force-kill path
        await manager.shutdown(timeout_seconds=0.1)

        # After shutdown, the task should be done (cancelled or finished)
        assert task.done()


# ---------------------------------------------------------------------------
# lifespan_with_shutdown
# ---------------------------------------------------------------------------

class TestLifespanWithShutdown:
    @pytest.mark.asyncio
    async def test_lifespan_yields_and_shuts_down(self):
        """lifespan_with_shutdown should yield control and then shut down."""
        manager = ShutdownManager()
        cleanup = MagicMock()
        manager.register_cleanup_handler(cleanup)

        app_mock = MagicMock()

        async with lifespan_with_shutdown(app_mock, shutdown_manager=manager, shutdown_timeout=1.0):
            pass  # simulate app running

        assert manager.is_shutting_down is True
        cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_uses_default_manager_when_none_given(self):
        """When no manager is passed, the module singleton is used."""
        reset_shutdown_manager()
        app_mock = MagicMock()

        async with lifespan_with_shutdown(app_mock, shutdown_timeout=1.0):
            pass

        singleton = get_shutdown_manager()
        assert singleton.is_shutting_down is True
        reset_shutdown_manager()

    @pytest.mark.asyncio
    async def test_lifespan_shuts_down_even_on_exception(self):
        """Shutdown should run even if the app body raises."""
        manager = ShutdownManager()
        cleanup = MagicMock()
        manager.register_cleanup_handler(cleanup)

        app_mock = MagicMock()

        with pytest.raises(ValueError):
            async with lifespan_with_shutdown(app_mock, shutdown_manager=manager, shutdown_timeout=1.0):
                raise ValueError("app error")

        assert manager.is_shutting_down is True
        cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    def setup_method(self):
        reset_shutdown_manager()

    def teardown_method(self):
        reset_shutdown_manager()

    def test_get_returns_shutdown_manager_instance(self):
        manager = get_shutdown_manager()
        assert isinstance(manager, ShutdownManager)

    def test_get_returns_same_instance(self):
        m1 = get_shutdown_manager()
        m2 = get_shutdown_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_shutdown_manager()
        reset_shutdown_manager()
        m2 = get_shutdown_manager()
        assert m1 is not m2


# ---------------------------------------------------------------------------
# API returns 503 during shutdown
# ---------------------------------------------------------------------------

class TestApiShutdownBehavior:
    """Verify that the API returns 503 when the shutdown manager is shutting down."""

    @pytest.mark.asyncio
    async def test_api_returns_503_during_shutdown(self):
        """
        When ShutdownManager.is_shutting_down is True, the middleware should
        return 503 for regular endpoints (not /health/live).
        """
        from httpx import AsyncClient, ASGITransport
        from backend.verification.api import app
        from backend.verification.utils.graceful_shutdown import reset_shutdown_manager, get_shutdown_manager

        reset_shutdown_manager()
        manager = get_shutdown_manager()

        # Trigger shutdown so is_shutting_down becomes True
        await manager.shutdown(timeout_seconds=0.1)
        assert manager.is_shutting_down is True

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 503
            assert "shutting down" in response.json().get("detail", "").lower()

        reset_shutdown_manager()

    @pytest.mark.asyncio
    async def test_liveness_probe_still_works_during_shutdown(self):
        """
        /health/live should remain accessible even during shutdown so that
        the process manager knows the process is still alive.
        """
        from httpx import AsyncClient, ASGITransport
        from backend.verification.api import app
        from backend.verification.utils.graceful_shutdown import reset_shutdown_manager, get_shutdown_manager

        reset_shutdown_manager()
        manager = get_shutdown_manager()

        await manager.shutdown(timeout_seconds=0.1)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/live")
            assert response.status_code == 200

        reset_shutdown_manager()
