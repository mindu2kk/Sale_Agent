"""
Graceful Shutdown with Async Task Cleanup - Task 7.3.3

Manages clean shutdown of the Verification Agent on SIGTERM/SIGINT:
- Registers asyncio tasks for cleanup
- Registers async cleanup callbacks
- Cancels all tasks within a configurable timeout
- Force-kills tasks that don't cancel in time
- Integrates with FastAPI lifespan context manager

Requirements:
- 8.1: Error handling with logging and correlation IDs
- 8.3: StateGraph execution error recovery
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable, List, Optional, Set

logger = logging.getLogger("backend.verification.graceful_shutdown")


class ShutdownManager:
    """
    Manages graceful shutdown of async tasks and cleanup handlers.

    Usage::

        manager = ShutdownManager()
        manager.setup_signal_handlers()

        task = asyncio.create_task(some_coroutine())
        manager.register_task(task)

        async def cleanup():
            await db.close()

        manager.register_cleanup_handler(cleanup)

        # On SIGTERM/SIGINT, call:
        await manager.shutdown(timeout_seconds=30)
    """

    def __init__(self) -> None:
        self._tasks: Set[asyncio.Task] = set()
        self._cleanup_handlers: List[Callable[[], object]] = []
        self._shutting_down: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_shutting_down(self) -> bool:
        """True once shutdown() has been called."""
        return self._shutting_down

    def register_task(self, task: asyncio.Task) -> None:
        """
        Register an asyncio task for cleanup on shutdown.

        Automatically removes the task from the registry when it completes.
        """
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.debug("Registered task for shutdown cleanup: %s", task.get_name())

    def register_cleanup_handler(self, handler: Callable[[], object]) -> None:
        """
        Register an async (or sync) cleanup callback to run during shutdown.

        Handlers are called in registration order after all tasks are cancelled.
        """
        self._cleanup_handlers.append(handler)
        logger.debug("Registered cleanup handler: %s", getattr(handler, "__name__", repr(handler)))

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """
        Gracefully cancel all registered tasks and run cleanup handlers.

        Steps:
        1. Mark shutdown in progress.
        2. Cancel all registered tasks.
        3. Wait up to *timeout_seconds* for tasks to finish.
        4. Force-cancel any tasks still running after the timeout.
        5. Run all registered cleanup handlers in order.

        Args:
            timeout_seconds: Maximum seconds to wait for tasks to cancel
                             before force-killing them.
        """
        if self._shutting_down:
            logger.warning("shutdown() called more than once — ignoring duplicate call")
            return

        self._shutting_down = True
        logger.info(
            "Graceful shutdown initiated — cancelling %d task(s), timeout=%.1fs",
            len(self._tasks),
            timeout_seconds,
        )

        # Step 1: Cancel all registered tasks
        pending = list(self._tasks)
        for task in pending:
            if not task.done():
                task.cancel()
                logger.debug("Cancelled task: %s", task.get_name())

        # Step 2: Wait for tasks to finish within timeout
        if pending:
            done, still_running = await asyncio.wait(
                pending,
                timeout=timeout_seconds,
                return_when=asyncio.ALL_COMPLETED,
            )
            logger.info(
                "Tasks finished: %d done, %d still running after timeout",
                len(done),
                len(still_running),
            )

            # Step 3: Force-kill tasks that didn't cancel in time
            for task in still_running:
                logger.warning(
                    "Force-killing task that did not cancel within %.1fs: %s",
                    timeout_seconds,
                    task.get_name(),
                )
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass  # Task is gone — that's the goal

        # Step 4: Run cleanup handlers
        for handler in self._cleanup_handlers:
            handler_name = getattr(handler, "__name__", repr(handler))
            try:
                result = handler()
                if asyncio.iscoroutine(result):
                    await result
                logger.debug("Cleanup handler completed: %s", handler_name)
            except Exception as exc:
                logger.error(
                    "Cleanup handler '%s' raised an exception: %s: %s",
                    handler_name,
                    type(exc).__name__,
                    exc,
                )

        logger.info("Graceful shutdown complete")

    def setup_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """
        Register SIGTERM and SIGINT handlers that trigger shutdown.

        On Windows, only SIGINT is supported via signal.signal().
        On Unix, both SIGTERM and SIGINT are registered on the event loop.

        Args:
            loop: Event loop to register handlers on.  Defaults to the
                  running loop (must be called from within an async context
                  or after loop creation).
        """
        if sys.platform == "win32":
            # Windows: use signal.signal for SIGINT only
            def _win_handler(signum, frame):  # type: ignore[no-untyped-def]
                logger.info("Received signal %d — scheduling shutdown", signum)
                if loop is not None:
                    loop.call_soon_threadsafe(
                        lambda: loop.create_task(self.shutdown())
                    )
                else:
                    # Best-effort: schedule on the running loop
                    try:
                        running = asyncio.get_event_loop()
                        running.call_soon_threadsafe(
                            lambda: running.create_task(self.shutdown())
                        )
                    except RuntimeError:
                        pass

            signal.signal(signal.SIGINT, _win_handler)
            logger.debug("Registered SIGINT handler (Windows)")
        else:
            # Unix: register on the event loop for thread-safe async scheduling
            _loop = loop
            if _loop is None:
                try:
                    _loop = asyncio.get_running_loop()
                except RuntimeError:
                    _loop = asyncio.get_event_loop()

            def _unix_handler(signum: int) -> None:
                logger.info("Received signal %d — scheduling shutdown", signum)
                _loop.create_task(self.shutdown())  # type: ignore[union-attr]

            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    _loop.add_signal_handler(sig, _unix_handler, sig)
                    logger.debug("Registered signal handler for %s", signal.Signals(sig).name)
                except (NotImplementedError, OSError) as exc:
                    # Some environments (e.g. threads) don't support add_signal_handler
                    logger.warning(
                        "Could not register signal handler for %s: %s",
                        signal.Signals(sig).name,
                        exc,
                    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_manager: Optional[ShutdownManager] = None


def get_shutdown_manager() -> ShutdownManager:
    """Return the module-level singleton ShutdownManager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ShutdownManager()
    return _default_manager


def reset_shutdown_manager() -> None:
    """Reset the module-level singleton (useful in tests)."""
    global _default_manager
    _default_manager = None


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan_with_shutdown(
    app: object,
    shutdown_manager: Optional[ShutdownManager] = None,
    shutdown_timeout: float = 30.0,
) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager that integrates ShutdownManager.

    Usage in api.py::

        from backend.verification.utils.graceful_shutdown import lifespan_with_shutdown

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with lifespan_with_shutdown(app):
                yield

        app = FastAPI(lifespan=lifespan)

    On startup: registers signal handlers.
    On shutdown: calls ShutdownManager.shutdown() to clean up tasks.
    """
    manager = shutdown_manager or get_shutdown_manager()

    # Startup
    logger.info("Application startup — registering signal handlers")
    try:
        manager.setup_signal_handlers()
    except Exception as exc:
        logger.warning("Could not register signal handlers: %s", exc)

    try:
        yield
    finally:
        # Shutdown
        logger.info("Application shutdown — running graceful cleanup")
        await manager.shutdown(timeout_seconds=shutdown_timeout)


__all__ = [
    "ShutdownManager",
    "get_shutdown_manager",
    "reset_shutdown_manager",
    "lifespan_with_shutdown",
]
