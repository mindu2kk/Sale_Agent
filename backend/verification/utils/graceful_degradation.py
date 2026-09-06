"""
Graceful Degradation for Partial Verification Failures

When one or two checkers fail (due to exceptions, timeouts, or service errors),
the workflow continues with the remaining successful checker results rather than
failing the entire verification.

Design principles:
- At most 1 checker failure is acceptable for degradation
- If 2+ checkers fail, escalate (too unreliable to continue)
- Failed checkers default to PASS with a warning issue (not a hard failure)
- All degradation events are logged with correlation IDs

Requirements: 8.1, 8.2, 8.3
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models.verification import (
    IssueSeverity,
    PolicyIssue,
    PriceIssue,
    RelevanceIssue,
    RubricCriteria,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# Maximum number of checker failures that still allow degraded operation
MAX_DEGRADABLE_FAILURES = 1


@dataclass
class PartialVerificationResult:
    """
    Holds the result of a single checker invocation, including whether it
    succeeded or failed with an exception.
    """

    checker_name: str
    success: bool
    result: Optional[Tuple[bool, List]] = None  # (pass_flag, issues_list)
    error: Optional[Exception] = None

    @property
    def pass_flag(self) -> bool:
        """Return the checker's pass flag, or True (default) if checker failed."""
        if self.success and self.result is not None:
            return self.result[0]
        return True  # default to pass when checker itself failed

    @property
    def issues(self) -> List:
        """Return the checker's issues list, or empty list if checker failed."""
        if self.success and self.result is not None:
            return self.result[1]
        return []


class GracefulDegradationHandler:
    """
    Wraps checker calls to catch exceptions and aggregate partial results
    into a VerificationResult with sensible defaults for failed checkers.
    """

    def __init__(self, correlation_id: Optional[str] = None):
        self._correlation_id = correlation_id or "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_checker_safely(
        self,
        checker_name: str,
        checker_fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> PartialVerificationResult:
        """
        Wrap a checker call, catching any exception.

        Returns a PartialVerificationResult indicating success/failure.
        On exception, logs the event with checker name, error type, and
        correlation ID.
        """
        try:
            result = await asyncio.to_thread(checker_fn, *args, **kwargs)
            logger.debug(
                "Checker completed successfully",
                extra={
                    "checker_name": checker_name,
                    "correlation_id": self._correlation_id,
                    "pass_flag": result[0] if result else None,
                },
            )
            return PartialVerificationResult(
                checker_name=checker_name,
                success=True,
                result=result,
            )
        except Exception as exc:
            logger.warning(
                "Checker failed — applying graceful degradation",
                extra={
                    "checker_name": checker_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "correlation_id": self._correlation_id,
                    "fallback_decision": "default_pass_with_warning",
                },
            )
            return PartialVerificationResult(
                checker_name=checker_name,
                success=False,
                error=exc,
            )

    def should_degrade(self, failed_checkers: List[str]) -> bool:
        """
        Determine whether graceful degradation is acceptable.

        Degradation is acceptable when at most MAX_DEGRADABLE_FAILURES
        checkers have failed. If more fail, the result is too unreliable
        and the caller should escalate.

        Args:
            failed_checkers: Names of checkers that raised exceptions.

        Returns:
            True if degradation is acceptable, False if escalation is needed.
        """
        acceptable = len(failed_checkers) <= MAX_DEGRADABLE_FAILURES
        if not acceptable:
            logger.error(
                "Too many checker failures — degradation not acceptable, escalating",
                extra={
                    "failed_checkers": failed_checkers,
                    "failed_count": len(failed_checkers),
                    "max_degradable": MAX_DEGRADABLE_FAILURES,
                    "correlation_id": self._correlation_id,
                },
            )
        else:
            logger.info(
                "Graceful degradation applied",
                extra={
                    "failed_checkers": failed_checkers,
                    "failed_count": len(failed_checkers),
                    "correlation_id": self._correlation_id,
                },
            )
        return acceptable

    def aggregate_partial_results(
        self,
        partial_results: Dict[str, PartialVerificationResult],
        verification_reasoning: str = "Partial verification with graceful degradation applied",
    ) -> VerificationResult:
        """
        Build a VerificationResult from partial checker results.

        For failed checkers, the corresponding pass flag defaults to True
        (we don't penalise the draft for a checker service error) but a
        warning-severity issue is injected so the degradation is visible
        in the result.

        Args:
            partial_results: Mapping of checker_name → PartialVerificationResult.
                Expected keys: "price", "policy", "relevance".
            verification_reasoning: Human-readable reasoning string.

        Returns:
            VerificationResult built from available data.
        """
        price_pr = partial_results.get("price")
        policy_pr = partial_results.get("policy")
        relevance_pr = partial_results.get("relevance")

        price_pass, price_issues = self._resolve_checker(price_pr, "price")
        policy_pass, policy_issues = self._resolve_checker(policy_pr, "policy")
        relevance_pass, relevance_issues = self._resolve_checker(relevance_pr, "relevance")

        criteria = RubricCriteria(
            price_accuracy_pass=price_pass,
            price_issues=price_issues,
            policy_authenticity_pass=policy_pass,
            policy_issues=policy_issues,
            topic_relevance_pass=relevance_pass,
            relevance_issues=relevance_issues,
        )

        return VerificationResult(
            criteria=criteria,
            verification_reasoning=verification_reasoning,
            execution_time_seconds=0.0,
            llm_tokens_used=0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_checker(
        self,
        partial: Optional[PartialVerificationResult],
        checker_type: str,
    ) -> Tuple[bool, List]:
        """
        Resolve a PartialVerificationResult into (pass_flag, issues_list).

        If the checker failed, returns (True, [warning_issue]) so the draft
        is not penalised for a service error.
        """
        if partial is None:
            # Checker was never run — treat as degraded
            warning = self._build_warning_issue(checker_type, "Checker was not executed")
            return True, [warning]

        if partial.success:
            return partial.result[0], partial.result[1]

        # Checker raised an exception — inject a warning issue
        error_msg = str(partial.error) if partial.error else "Unknown error"
        warning = self._build_warning_issue(checker_type, error_msg)
        return True, [warning]

    @staticmethod
    def _build_warning_issue(checker_type: str, error_message: str) -> Any:
        """Build a MINOR warning issue for a failed checker."""
        msg = f"Checker failed — result unavailable. Error: {error_message}"
        if checker_type == "price":
            return PriceIssue(
                product_name="[Checker Unavailable]",
                severity=IssueSeverity.MINOR,
                explanation=msg,
            )
        elif checker_type == "policy":
            return PolicyIssue(
                mentioned_policy="[Checker Unavailable]",
                policy_type="warranty",
                is_fabricated=False,
                is_inaccurate=False,
                severity=IssueSeverity.MINOR,
                explanation=msg,
            )
        else:  # relevance
            return RelevanceIssue(
                objection_intent="[Checker Unavailable]",
                response_coverage=1.0,
                missing_aspects=[],
                severity=IssueSeverity.MINOR,
                explanation=msg,
            )
