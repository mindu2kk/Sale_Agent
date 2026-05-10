"""
Severity-Based Processing Priorities

Provides priority ordering and processing for verification issues based on
severity levels (critical > major > minor). Integrates with early termination
to short-circuit processing as soon as a critical issue is found.

Supports Task 5.4.3: Create issue severity-based processing priorities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Generator, List, Optional, Set, Union

from verification.models.verification import (
    IssueSeverity,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
)
from verification.utils.early_termination import EarlyTerminationManager


# ---------------------------------------------------------------------------
# Priority constants
# ---------------------------------------------------------------------------

class SeverityLevel(IntEnum):
    """
    Numeric priority for issue severity — lower value = higher priority.

    CRITICAL=0 is processed first, enabling faster early termination.
    """
    CRITICAL = 0
    MAJOR = 1
    MINOR = 2

    @classmethod
    def from_issue_severity(cls, severity: IssueSeverity) -> "SeverityLevel":
        """Convert IssueSeverity enum to SeverityLevel."""
        mapping = {
            IssueSeverity.CRITICAL: cls.CRITICAL,
            IssueSeverity.MAJOR: cls.MAJOR,
            IssueSeverity.MINOR: cls.MINOR,
        }
        return mapping[severity]


# Checker name ordering: run the checker most likely to surface critical issues first.
# This default order can be overridden by historical severity patterns.
DEFAULT_CHECKER_ORDER = ["price", "policy", "relevance"]


# ---------------------------------------------------------------------------
# IssuePriority wrapper
# ---------------------------------------------------------------------------

@dataclass(order=True)
class IssuePriority:
    """
    Wraps a verification issue with its computed priority.

    Ordering is based on (priority, issue_type) so that issues with the same
    severity are grouped by type for consistent processing.
    """
    priority: SeverityLevel = field(compare=True)
    issue_type: str = field(compare=True)
    issue: object = field(compare=False)

    @classmethod
    def from_issue(
        cls,
        issue: Union[PriceIssue, PolicyIssue, RelevanceIssue],
    ) -> "IssuePriority":
        """Create an IssuePriority from any issue model."""
        severity: IssueSeverity = issue.severity  # type: ignore[union-attr]
        priority = SeverityLevel.from_issue_severity(severity)
        issue_type = type(issue).__name__
        return cls(priority=priority, issue_type=issue_type, issue=issue)


# ---------------------------------------------------------------------------
# SeverityBasedProcessor
# ---------------------------------------------------------------------------

class SeverityBasedProcessor:
    """
    Processes verification issues in severity-priority order (critical first).

    Features:
    - Accepts mixed issue types (PriceIssue, PolicyIssue, RelevanceIssue)
    - Sorts by severity: CRITICAL → MAJOR → MINOR
    - Integrates with EarlyTerminationManager to stop on critical issues
    - Tracks which severity levels have been encountered
    - Recommends checker execution order based on historical severity patterns
    """

    def __init__(
        self,
        early_termination_manager: Optional[EarlyTerminationManager] = None,
    ) -> None:
        self._early_termination = early_termination_manager or EarlyTerminationManager()
        # Historical severity counts per checker type for ordering recommendations
        self._checker_severity_counts: dict[str, dict[str, int]] = {
            "price": {"critical": 0, "major": 0, "minor": 0},
            "policy": {"critical": 0, "major": 0, "minor": 0},
            "relevance": {"critical": 0, "major": 0, "minor": 0},
        }
        self._encountered_severities: Set[SeverityLevel] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_by_priority(
        self,
        issues: List[Union[PriceIssue, PolicyIssue, RelevanceIssue]],
    ) -> Generator[IssuePriority, None, None]:
        """
        Yield issues in priority order (critical first).

        Stops yielding as soon as a critical issue is found and early
        termination is triggered by the EarlyTerminationManager.

        Args:
            issues: Mixed list of PriceIssue, PolicyIssue, RelevanceIssue.

        Yields:
            IssuePriority wrappers in severity order.
        """
        if not issues:
            return

        prioritized = sorted(
            [IssuePriority.from_issue(i) for i in issues],
        )

        processed: List[object] = []
        for item in prioritized:
            self._encountered_severities.add(item.priority)
            processed.append(item.issue)
            yield item

            # Check early termination after each issue
            termination = self._early_termination.should_terminate(processed)
            if termination.should_terminate:
                break

    def get_sorted_issues(
        self,
        issues: List[Union[PriceIssue, PolicyIssue, RelevanceIssue]],
    ) -> List[IssuePriority]:
        """
        Return all issues sorted by priority without early termination.

        Useful when you need the full sorted list without side effects.
        """
        return sorted([IssuePriority.from_issue(i) for i in issues])

    def get_processing_order(self) -> List[str]:
        """
        Recommend checker execution order based on historical severity patterns.

        Checkers that have historically produced more critical issues are
        recommended to run first, enabling faster early termination.

        Returns:
            List of checker names in recommended execution order.
            Falls back to DEFAULT_CHECKER_ORDER when no history is available.
        """
        def _critical_score(checker: str) -> int:
            counts = self._checker_severity_counts.get(checker, {})
            # Weight: critical=100, major=10, minor=1
            return (
                counts.get("critical", 0) * 100
                + counts.get("major", 0) * 10
                + counts.get("minor", 0)
            )

        # Only reorder if we have any historical data
        has_history = any(
            sum(counts.values()) > 0
            for counts in self._checker_severity_counts.values()
        )
        if not has_history:
            return list(DEFAULT_CHECKER_ORDER)

        return sorted(
            DEFAULT_CHECKER_ORDER,
            key=_critical_score,
            reverse=True,  # highest score first
        )

    def record_checker_results(
        self,
        checker_name: str,
        issues: List[Union[PriceIssue, PolicyIssue, RelevanceIssue]],
    ) -> None:
        """
        Record issue severity counts for a checker to inform future ordering.

        Args:
            checker_name: One of "price", "policy", "relevance".
            issues: Issues produced by that checker.
        """
        if checker_name not in self._checker_severity_counts:
            self._checker_severity_counts[checker_name] = {
                "critical": 0, "major": 0, "minor": 0,
            }
        for issue in issues:
            sev = getattr(issue, "severity", None)
            if sev is not None:
                sev_str = sev.value if isinstance(sev, IssueSeverity) else str(sev).lower()
                if sev_str in self._checker_severity_counts[checker_name]:
                    self._checker_severity_counts[checker_name][sev_str] += 1

    @property
    def encountered_severities(self) -> Set[SeverityLevel]:
        """Return the set of severity levels seen during process_by_priority calls."""
        return frozenset(self._encountered_severities)  # type: ignore[return-value]

    def has_encountered_critical(self) -> bool:
        """Return True if any critical issue has been processed."""
        return SeverityLevel.CRITICAL in self._encountered_severities

    def reset_history(self) -> None:
        """Reset historical severity counts and encountered severities."""
        for checker in self._checker_severity_counts:
            self._checker_severity_counts[checker] = {
                "critical": 0, "major": 0, "minor": 0,
            }
        self._encountered_severities.clear()
