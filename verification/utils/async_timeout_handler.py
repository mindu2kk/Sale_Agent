"""
Async Timeout Handler with Configurable Thresholds

Wraps async coroutines with per-operation timeout values loaded from the
thresholds configuration system (thresholds.yaml / TimeoutConfig).

Raises a structured OperationTimeoutError with full context (operation name,
elapsed time, configured threshold) and optionally triggers escalation for
critical-path operations.

Supports Task 6.3.2: Async timeout handling with configurable thresholds.

Requirements:
- 8.1: Error handling with full context logging
- 9.1: Verification process SHALL complete in ≤ 10 seconds for typical objections
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar

from ..config.thresholds_config import TimeoutConfig, enhanced_load_thresholds_config

logger = logging.getLogger("verification.async_timeout_handler")

T = TypeVar("T")

# Default path to the shared thresholds YAML file
_DEFAULT_THRESHOLDS_PATH = str(
    Path(__file__).parent.parent / "config" / "thresholds.yaml"
)


# ---------------------------------------------------------------------------
# Structured timeout error
# ---------------------------------------------------------------------------

class OperationTimeoutError(TimeoutError):
    """
    Raised when an async operation exceeds its configured timeout.

    Carries structured context for logging and escalation decisions.

    Attributes:
        operation: Name of the timed-out operation.
        elapsed: Actual elapsed time in seconds before the timeout fired.
        threshold: Configured timeout threshold in seconds.
        escalation_required: Whether the caller should escalate to a human.
    """

    def __init__(
        self,
        operation: str,
        elapsed: float,
        threshold: float,
        escalation_required: bool = False,
    ) -> None:
        self.operation = operation
        self.elapsed = elapsed
        self.threshold = threshold
        self.escalation_required = escalation_required
        super().__init__(
            f"Operation '{operation}' timed out after {elapsed:.3f}s "
            f"(threshold={threshold:.3f}s, escalate={escalation_required})"
        )

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "elapsed_seconds": round(self.elapsed, 3),
            "threshold_seconds": round(self.threshold, 3),
            "escalation_required": self.escalation_required,
        }


# ---------------------------------------------------------------------------
# AsyncTimeoutHandler
# ---------------------------------------------------------------------------

class AsyncTimeoutHandler:
    """
    Wraps async coroutines with configurable per-operation timeouts.

    Timeout values are loaded from the shared ``TimeoutConfig`` (which in turn
    is populated from ``thresholds.yaml``).  Each call to :meth:`run` or
    :meth:`wrap` enforces the timeout for the named operation and raises
    :class:`OperationTimeoutError` with full context on expiry.

    Usage — run a coroutine directly::

        handler = AsyncTimeoutHandler()
        result = await handler.run(call_llm(prompt), "llm_call")

    Usage — wrap a callable::

        handler = AsyncTimeoutHandler()
        result = await handler.wrap(call_llm, "llm_call", prompt)

    Usage — custom config::

        cfg = TimeoutConfig(llm_call=15.0, price_check=3.0)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        result = await handler.run(call_llm(prompt), "llm_call")
    """

    def __init__(
        self,
        timeout_config: Optional[TimeoutConfig] = None,
        thresholds_path: Optional[str] = None,
    ) -> None:
        """
        Initialise the handler.

        Args:
            timeout_config: Pre-built :class:`TimeoutConfig`.  When *None* the
                config is loaded from ``thresholds_path`` (or the default YAML).
            thresholds_path: Path to ``thresholds.yaml``.  Ignored when
                ``timeout_config`` is provided explicitly.
        """
        if timeout_config is not None:
            self._config = timeout_config
        else:
            path = thresholds_path or _DEFAULT_THRESHOLDS_PATH
            try:
                full_config = enhanced_load_thresholds_config(path)
                self._config = full_config.timeouts
            except Exception as exc:
                logger.warning(
                    "AsyncTimeoutHandler: failed to load thresholds from '%s' (%s). "
                    "Using default TimeoutConfig.",
                    path,
                    exc,
                )
                self._config = TimeoutConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> TimeoutConfig:
        """Return the active :class:`TimeoutConfig`."""
        return self._config

    def get_threshold(self, operation: str) -> float:
        """
        Return the configured timeout (seconds) for *operation*.

        Falls back to ``llm_call`` timeout for unknown operation names.
        """
        return self._config.get_timeout(operation)

    async def run(
        self,
        coro: Awaitable[T],
        operation: str,
        *,
        timeout_override: Optional[float] = None,
    ) -> T:
        """
        Await *coro* with the timeout configured for *operation*.

        Args:
            coro: Awaitable to execute.
            operation: Operation name used to look up the timeout threshold.
            timeout_override: If provided, use this value instead of the
                configured threshold (useful for testing).

        Returns:
            The result of awaiting *coro*.

        Raises:
            OperationTimeoutError: If *coro* does not complete within the
                configured timeout.
        """
        threshold = timeout_override if timeout_override is not None else self.get_threshold(operation)
        start = time.perf_counter()
        try:
            return await asyncio.wait_for(coro, timeout=threshold)
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start
            escalate = self._config.escalate_on_critical_timeout
            logger.error(
                "Timeout: operation='%s' elapsed=%.3fs threshold=%.3fs escalate=%s",
                operation,
                elapsed,
                threshold,
                escalate,
            )
            raise OperationTimeoutError(
                operation=operation,
                elapsed=elapsed,
                threshold=threshold,
                escalation_required=escalate,
            )

    async def wrap(
        self,
        fn: Callable[..., Awaitable[T]],
        operation: str,
        *args: Any,
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """
        Call *fn* with *args*/*kwargs* and enforce the timeout for *operation*.

        Convenience wrapper around :meth:`run` for callables.

        Args:
            fn: Async callable to invoke.
            operation: Operation name used to look up the timeout threshold.
            *args: Positional arguments forwarded to *fn*.
            timeout_override: Optional explicit timeout in seconds.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            The result of ``fn(*args, **kwargs)``.

        Raises:
            OperationTimeoutError: If the call exceeds the configured timeout.
        """
        return await self.run(fn(*args, **kwargs), operation, timeout_override=timeout_override)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_handler: Optional[AsyncTimeoutHandler] = None


def get_async_timeout_handler(
    timeout_config: Optional[TimeoutConfig] = None,
    thresholds_path: Optional[str] = None,
) -> AsyncTimeoutHandler:
    """
    Return the module-level singleton :class:`AsyncTimeoutHandler`.

    Creates a new instance on first call; subsequent calls return the same
    instance (arguments are ignored after initialisation).
    """
    global _default_handler
    if _default_handler is None:
        _default_handler = AsyncTimeoutHandler(
            timeout_config=timeout_config,
            thresholds_path=thresholds_path,
        )
    return _default_handler


def reset_async_timeout_handler() -> None:
    """Reset the module-level singleton (useful in tests)."""
    global _default_handler
    _default_handler = None


__all__ = [
    "OperationTimeoutError",
    "AsyncTimeoutHandler",
    "get_async_timeout_handler",
    "reset_async_timeout_handler",
]
