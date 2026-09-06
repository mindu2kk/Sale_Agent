"""
Early Termination Manager

Wires up EscalationThresholds.should_terminate_early() into a usable manager
that the verification agent can call during parallel checks.

Also provides CriticalIssueDetector for immediate workflow termination when
any single critical issue is found across all issue types.

Supports Task 1.3.2: Configurable early termination rules cho critical issues
Supports Task 5.4.1: Critical issue detection với immediate workflow termination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union, TYPE_CHECKING

from backend.verification.config.thresholds_config import (
    EscalationThresholds,
    IssueSeverity,
    VerificationThresholdsConfig,
    get_default_thresholds_config,
)

if TYPE_CHECKING:
    from backend.verification.models.verification import VerificationResult


@dataclass
class TerminationResult:
    """Result of an early termination check."""

    should_terminate: bool
    reason: str
    critical_count: int
    total_count: int

    def __bool__(self) -> bool:
        return self.should_terminate


@dataclass
class TerminationDecision:
    """
    Structured decision for immediate workflow termination.

    Produced by CriticalIssueDetector.should_terminate_immediately() and
    consumed by routing logic to bypass the correction loop entirely.
    """

    should_terminate: bool
    reason: str
    # The specific critical issues that triggered termination
    critical_issues: List[object] = field(default_factory=list)
    # Counts per issue type
    critical_price_count: int = 0
    critical_policy_count: int = 0
    critical_relevance_count: int = 0

    @property
    def total_critical_count(self) -> int:
        return self.critical_price_count + self.critical_policy_count + self.critical_relevance_count

    def __bool__(self) -> bool:
        return self.should_terminate


class EarlyTerminationManager:
    """
    Manages early termination logic during parallel verification checks.

    Loads rules from VerificationThresholdsConfig and exposes a simple
    ``should_terminate(issues)`` interface that the verification agent can
    call after each checker completes.

    Two termination modes (controlled by EscalationThresholds):
    - stop_on_first_critical=True  → terminate as soon as 1 critical issue found
    - stop_on_first_critical=False → terminate when critical_count >= multiple_critical_threshold
    - early_termination_enabled=False → never terminate early (useful in dev/test)
    """

    def __init__(self, config: Optional[VerificationThresholdsConfig] = None) -> None:
        if config is None:
            config = get_default_thresholds_config()
        self._thresholds: EscalationThresholds = config.escalation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether early termination is currently enabled."""
        return self._thresholds.early_termination_enabled

    @property
    def stop_on_first_critical(self) -> bool:
        return self._thresholds.stop_on_first_critical

    @property
    def multiple_critical_threshold(self) -> int:
        return self._thresholds.multiple_critical_threshold

    def should_terminate(
        self,
        issues: List[Union[object, IssueSeverity]],
    ) -> TerminationResult:
        """
        Decide whether to terminate early given the current list of issues.

        Parameters
        ----------
        issues:
            A list of objects that either *are* ``IssueSeverity`` values or
            have a ``.severity`` attribute of type ``IssueSeverity``.

        Returns
        -------
        TerminationResult
            ``.should_terminate`` is True when early termination is triggered.
        """
        critical_count = self._count_critical(issues)
        total_count = len(issues)

        terminate = self._thresholds.should_terminate_early(critical_count)

        if not terminate:
            reason = (
                "No early termination: "
                f"{critical_count} critical issue(s) found "
                f"(threshold={self.multiple_critical_threshold}, "
                f"stop_on_first={self.stop_on_first_critical}, "
                f"enabled={self.enabled})"
            )
        elif not self.enabled:
            reason = "Early termination disabled for this environment"
        elif self.stop_on_first_critical:
            reason = f"Early termination: first critical issue detected ({critical_count} total)"
        else:
            reason = (
                f"Early termination: {critical_count} critical issue(s) "
                f">= threshold {self.multiple_critical_threshold}"
            )

        return TerminationResult(
            should_terminate=terminate,
            reason=reason,
            critical_count=critical_count,
            total_count=total_count,
        )

    def should_terminate_for_count(self, critical_count: int) -> TerminationResult:
        """
        Convenience method when you already know the critical issue count.
        """
        terminate = self._thresholds.should_terminate_early(critical_count)
        reason = (
            f"Early termination triggered: {critical_count} critical issue(s)"
            if terminate
            else f"No early termination: {critical_count} critical issue(s)"
        )
        return TerminationResult(
            should_terminate=terminate,
            reason=reason,
            critical_count=critical_count,
            total_count=critical_count,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_critical(issues: List) -> int:
        count = 0
        for issue in issues:
            if isinstance(issue, IssueSeverity):
                if issue == IssueSeverity.CRITICAL:
                    count += 1
            elif hasattr(issue, "severity"):
                if issue.severity == IssueSeverity.CRITICAL:
                    count += 1
        return count


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_early_termination_manager(
    environment: Optional[str] = None,
    config: Optional[VerificationThresholdsConfig] = None,
) -> EarlyTerminationManager:
    """
    Create an EarlyTerminationManager, optionally applying environment overrides.

    Parameters
    ----------
    environment:
        One of "development", "production", "testing" (or None for defaults).
        - development: early termination disabled
        - production:  strict (max_critical_before_escalation=1)
        - testing:     early termination disabled
    config:
        Pre-built config to use instead of loading defaults.
    """
    if config is None:
        config = get_default_thresholds_config()
    if environment is not None:
        config = config.get_environment_config(environment)
    return EarlyTerminationManager(config)


# ---------------------------------------------------------------------------
# CriticalIssueDetector — Task 5.4.1
# ---------------------------------------------------------------------------

class CriticalIssueDetector:
    """
    Detects critical issues across all issue types and decides whether to
    terminate the workflow immediately (bypassing the correction loop).

    Design rule (Task 5.4.1):
        ANY single issue with severity="critical" triggers immediate termination.
        This is stricter than EarlyTerminationManager which uses a count threshold.

    Supports:
    - PriceIssue, PolicyIssue, RelevanceIssue (objects with .severity attribute)
    - Direct IssueSeverity enum values
    - VerificationResult objects (inspects all three issue lists)
    """

    def check_issues(
        self,
        price_issues: Optional[List] = None,
        policy_issues: Optional[List] = None,
        relevance_issues: Optional[List] = None,
    ) -> TerminationDecision:
        """
        Check lists of issues for any critical severity item.

        Parameters
        ----------
        price_issues:    List of PriceIssue objects (or objects with .severity)
        policy_issues:   List of PolicyIssue objects (or objects with .severity)
        relevance_issues: List of RelevanceIssue objects (or objects with .severity)

        Returns
        -------
        TerminationDecision
            .should_terminate=True if any critical issue found.
        """
        price_issues = price_issues or []
        policy_issues = policy_issues or []
        relevance_issues = relevance_issues or []

        critical_price = [i for i in price_issues if self._is_critical(i)]
        critical_policy = [i for i in policy_issues if self._is_critical(i)]
        critical_relevance = [i for i in relevance_issues if self._is_critical(i)]

        all_critical = critical_price + critical_policy + critical_relevance

        if not all_critical:
            return TerminationDecision(
                should_terminate=False,
                reason="No critical issues detected — workflow continues normally",
                critical_issues=[],
                critical_price_count=0,
                critical_policy_count=0,
                critical_relevance_count=0,
            )

        # Build a human-readable reason
        parts = []
        if critical_price:
            parts.append(f"{len(critical_price)} critical price issue(s)")
        if critical_policy:
            parts.append(f"{len(critical_policy)} critical policy issue(s)")
        if critical_relevance:
            parts.append(f"{len(critical_relevance)} critical relevance issue(s)")

        reason = (
            f"Immediate termination: {', '.join(parts)} detected — "
            "bypassing correction loop and escalating to human review"
        )

        return TerminationDecision(
            should_terminate=True,
            reason=reason,
            critical_issues=all_critical,
            critical_price_count=len(critical_price),
            critical_policy_count=len(critical_policy),
            critical_relevance_count=len(critical_relevance),
        )

    def check_verification_result(self, verification_result: "VerificationResult") -> TerminationDecision:
        """
        Inspect a VerificationResult for any critical issues.

        Parameters
        ----------
        verification_result:
            A VerificationResult whose .criteria contains price_issues,
            policy_issues, and relevance_issues lists.

        Returns
        -------
        TerminationDecision
        """
        criteria = verification_result.criteria
        return self.check_issues(
            price_issues=criteria.price_issues,
            policy_issues=criteria.policy_issues,
            relevance_issues=criteria.relevance_issues,
        )

    @staticmethod
    def _is_critical(issue: object) -> bool:
        """Return True if the issue has severity=CRITICAL.

        Handles both IssueSeverity enums (thresholds_config and models.verification)
        as well as plain string values.
        """
        if isinstance(issue, IssueSeverity):
            return issue == IssueSeverity.CRITICAL
        if hasattr(issue, "severity"):
            sev = issue.severity
            if isinstance(sev, IssueSeverity):
                return sev == IssueSeverity.CRITICAL
            # Handle other enum types or strings — check .value if available
            sev_str = getattr(sev, "value", None) or str(sev)
            return str(sev_str).lower() == "critical"
        return False


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def should_terminate_immediately(verification_result: "VerificationResult") -> TerminationDecision:
    """
    Convenience function: check a VerificationResult for critical issues.

    Any single critical issue (price, policy, or relevance) triggers
    immediate workflow termination.

    Parameters
    ----------
    verification_result:
        The VerificationResult to inspect.

    Returns
    -------
    TerminationDecision
        .should_terminate=True if any critical issue found.
    """
    detector = CriticalIssueDetector()
    return detector.check_verification_result(verification_result)
