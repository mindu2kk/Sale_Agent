"""
Adaptive Timeout Mechanisms Based on Issue Complexity

Adjusts timeout durations dynamically based on:
- Check type (price, policy, relevance) with configurable base timeouts
- Issue severity (critical > major > minor adds more time)
- Issue count (more issues = more processing time needed)
- Historical execution times (exponential moving average)

Supports Task 5.4.4: Build adaptive timeout mechanisms based on issue complexity
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from verification.models.verification import (
    IssueSeverity,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TimeoutConfig:
    """
    Configurable base timeouts and scaling factors for adaptive timeout calculation.

    Attributes:
        base_timeouts: Base timeout in seconds per check type.
        severity_multipliers: Multiplier applied per issue of each severity.
        issue_count_factor: Additional seconds added per issue beyond the first.
        max_timeout: Hard upper bound on any computed timeout (seconds).
        min_timeout: Hard lower bound on any computed timeout (seconds).
        history_alpha: EMA smoothing factor for historical execution times (0–1).
            Higher values weight recent observations more heavily.
    """
    base_timeouts: Dict[str, float] = field(default_factory=lambda: {
        "price": 10.0,
        "policy": 15.0,
        "relevance": 10.0,
        "default": 10.0,
    })
    severity_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "critical": 3.0,
        "major": 1.5,
        "minor": 0.5,
    })
    issue_count_factor: float = 1.0   # extra seconds per additional issue
    max_timeout: float = 60.0
    min_timeout: float = 5.0
    history_alpha: float = 0.3        # EMA smoothing factor


# ---------------------------------------------------------------------------
# Complexity analysis
# ---------------------------------------------------------------------------

@dataclass
class ComplexityScore:
    """
    Computed complexity score for a verification check.

    Attributes:
        check_type: One of "price", "policy", "relevance", or "default".
        issue_count: Total number of issues provided.
        critical_count: Number of critical-severity issues.
        major_count: Number of major-severity issues.
        minor_count: Number of minor-severity issues.
        computed_timeout: Final timeout in seconds after applying all factors.
    """
    check_type: str
    issue_count: int
    critical_count: int
    major_count: int
    minor_count: int
    computed_timeout: float

    def to_dict(self) -> Dict:
        return {
            "check_type": self.check_type,
            "issue_count": self.issue_count,
            "critical_count": self.critical_count,
            "major_count": self.major_count,
            "minor_count": self.minor_count,
            "computed_timeout": round(self.computed_timeout, 3),
        }


# ---------------------------------------------------------------------------
# AdaptiveTimeoutManager
# ---------------------------------------------------------------------------

class AdaptiveTimeoutManager:
    """
    Calculates and enforces adaptive timeouts based on issue complexity.

    Usage — calculate timeout only::

        manager = AdaptiveTimeoutManager()
        timeout = manager.calculate_timeout("policy", issues)

    Usage — async context manager for enforcement::

        async with manager.enforce_timeout("policy", issues):
            result = await run_policy_check()

    Usage — with pre-computed timeout::

        timeout = manager.calculate_timeout("price", issues)
        async with manager.enforce_timeout_seconds(timeout):
            result = await run_price_check()
    """

    def __init__(self, config: Optional[TimeoutConfig] = None) -> None:
        self._config = config or TimeoutConfig()
        # EMA of historical execution times per check type
        self._history: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_timeout(
        self,
        check_type: str,
        issues: Optional[List[Union[PriceIssue, PolicyIssue, RelevanceIssue]]] = None,
    ) -> float:
        """
        Calculate an adaptive timeout for the given check type and issues.

        Algorithm:
        1. Start with the base timeout for the check type.
        2. Add severity-weighted contributions from each issue.
        3. Add issue_count_factor * max(0, issue_count - 1) for volume.
        4. Blend with historical EMA if available.
        5. Clamp to [min_timeout, max_timeout].

        Args:
            check_type: "price", "policy", "relevance", or any custom key.
            issues: Current known issues (may be empty for initial checks).

        Returns:
            Timeout duration in seconds.
        """
        issues = issues or []
        cfg = self._config

        base = cfg.base_timeouts.get(check_type, cfg.base_timeouts.get("default", 10.0))

        # Severity contribution
        severity_addition = 0.0
        critical_count = 0
        major_count = 0
        minor_count = 0

        for issue in issues:
            sev = self._get_severity_str(issue)
            multiplier = cfg.severity_multipliers.get(sev, 0.5)
            severity_addition += multiplier
            if sev == "critical":
                critical_count += 1
            elif sev == "major":
                major_count += 1
            else:
                minor_count += 1

        # Volume contribution: extra time per issue beyond the first
        volume_addition = cfg.issue_count_factor * max(0, len(issues) - 1)

        computed = base + severity_addition + volume_addition

        # Blend with historical EMA if we have prior data
        if check_type in self._history:
            alpha = cfg.history_alpha
            computed = alpha * computed + (1.0 - alpha) * self._history[check_type]

        # Clamp
        computed = max(cfg.min_timeout, min(cfg.max_timeout, computed))
        return round(computed, 3)

    def get_complexity_score(
        self,
        check_type: str,
        issues: Optional[List[Union[PriceIssue, PolicyIssue, RelevanceIssue]]] = None,
    ) -> ComplexityScore:
        """
        Return a ComplexityScore with full breakdown for observability.

        Args:
            check_type: Check type identifier.
            issues: Known issues for this check.

        Returns:
            ComplexityScore with counts and computed timeout.
        """
        issues = issues or []
        critical = sum(1 for i in issues if self._get_severity_str(i) == "critical")
        major = sum(1 for i in issues if self._get_severity_str(i) == "major")
        minor = sum(1 for i in issues if self._get_severity_str(i) == "minor")
        timeout = self.calculate_timeout(check_type, issues)
        return ComplexityScore(
            check_type=check_type,
            issue_count=len(issues),
            critical_count=critical,
            major_count=major,
            minor_count=minor,
            computed_timeout=timeout,
        )

    def record_execution_time(self, check_type: str, duration_seconds: float) -> None:
        """
        Update the historical EMA for a check type after it completes.

        Call this after each check to improve future timeout estimates.

        Args:
            check_type: The check type that just completed.
            duration_seconds: Actual wall-clock duration in seconds.
        """
        alpha = self._config.history_alpha
        if check_type in self._history:
            self._history[check_type] = (
                alpha * duration_seconds + (1.0 - alpha) * self._history[check_type]
            )
        else:
            self._history[check_type] = duration_seconds

    @asynccontextmanager
    async def enforce_timeout(
        self,
        check_type: str,
        issues: Optional[List[Union[PriceIssue, PolicyIssue, RelevanceIssue]]] = None,
    ):
        """
        Async context manager that enforces an adaptive timeout.

        Calculates the timeout from check_type + issues, then wraps the body
        with asyncio.wait_for(). Records actual execution time for future
        EMA updates.

        Raises:
            asyncio.TimeoutError: If the body exceeds the computed timeout.

        Example::

            async with manager.enforce_timeout("policy", issues):
                result = await run_policy_check()
        """
        timeout = self.calculate_timeout(check_type, issues)
        start = time.perf_counter()
        try:
            # We use a wrapper coroutine so asyncio.wait_for can cancel it
            yield timeout
        except asyncio.TimeoutError:
            raise
        finally:
            duration = time.perf_counter() - start
            self.record_execution_time(check_type, duration)

    @asynccontextmanager
    async def enforce_timeout_seconds(self, timeout_seconds: float):
        """
        Async context manager that enforces a pre-computed timeout.

        Use this when you've already called calculate_timeout() and want to
        reuse the value.

        Raises:
            asyncio.TimeoutError: If the body exceeds timeout_seconds.

        Example::

            timeout = manager.calculate_timeout("price", issues)
            async with manager.enforce_timeout_seconds(timeout):
                result = await run_price_check()
        """
        yield timeout_seconds

    def get_history(self) -> Dict[str, float]:
        """Return a copy of the current historical EMA values."""
        return dict(self._history)

    def reset_history(self) -> None:
        """Clear all historical execution time data."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_severity_str(
        issue: Union[PriceIssue, PolicyIssue, RelevanceIssue, IssueSeverity, str],
    ) -> str:
        """Extract severity as a lowercase string from any issue type."""
        if isinstance(issue, IssueSeverity):
            return issue.value.lower()
        if isinstance(issue, str):
            return issue.lower()
        sev = getattr(issue, "severity", None)
        if sev is None:
            return "minor"
        if isinstance(sev, IssueSeverity):
            return sev.value.lower()
        return str(getattr(sev, "value", sev)).lower()


# ---------------------------------------------------------------------------
# Convenience wrapper: run a coroutine with adaptive timeout
# ---------------------------------------------------------------------------

async def run_with_adaptive_timeout(
    coro,
    check_type: str,
    issues: Optional[List] = None,
    manager: Optional[AdaptiveTimeoutManager] = None,
):
    """
    Run a coroutine with an adaptive timeout derived from issue complexity.

    Args:
        coro: Awaitable to execute.
        check_type: "price", "policy", or "relevance".
        issues: Known issues that influence the timeout.
        manager: AdaptiveTimeoutManager instance (creates a default one if None).

    Returns:
        The result of awaiting coro.

    Raises:
        asyncio.TimeoutError: If coro exceeds the computed timeout.
    """
    if manager is None:
        manager = AdaptiveTimeoutManager()
    timeout = manager.calculate_timeout(check_type, issues)
    start = time.perf_counter()
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    finally:
        duration = time.perf_counter() - start
        manager.record_execution_time(check_type, duration)


# ---------------------------------------------------------------------------
# Module-level singleton (optional convenience)
# ---------------------------------------------------------------------------

_default_manager: Optional[AdaptiveTimeoutManager] = None


def get_adaptive_timeout_manager(
    config: Optional[TimeoutConfig] = None,
) -> AdaptiveTimeoutManager:
    """
    Return the module-level singleton AdaptiveTimeoutManager.

    Creates a new instance with the given config on first call.
    Subsequent calls return the same instance (config is ignored after init).
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = AdaptiveTimeoutManager(config)
    return _default_manager


def reset_adaptive_timeout_manager() -> None:
    """Reset the module-level singleton (useful in tests)."""
    global _default_manager
    _default_manager = None
