"""
Error Classification with Structured Issue Severity Mapping - Task 6.2.3

Maps exceptions and structured verification issues to classified errors with
severity levels, retriability flags, and suggested remediation actions.

Supports:
- ErrorCategory enum covering all system error types
- ClassifiedError Pydantic model with full context
- ErrorClassifier.classify() for exception-based classification
- ErrorClassifier.classify_from_issue() for PriceIssue/PolicyIssue/RelevanceIssue mapping
- CATEGORY_SEVERITY_MAP and CATEGORY_ACTION_MAP for deterministic mapping

Severity mapping guidelines (from design doc):
- critical: fabricated policies, price deviation >30%, state corruption, DB connection loss
- major: price deviation 15-30%, inaccurate policies, LLM timeout/API errors, rate limits
- minor: price deviation <15%, relevance issues, validation errors

Retriable errors: LLM_TIMEOUT, API_RATE_LIMIT, DB_CONNECTION
Non-retriable: VALIDATION_ERROR, STATE_CORRUPTION, UNKNOWN

Requirements:
- 8.1: Error handling with logging and correlation IDs
- 8.4: Circuit breaker pattern integration
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union

from pydantic import ConfigDict, BaseModel, Field

from ..models.verification import PriceIssue, PolicyIssue, RelevanceIssue, IssueSeverity

logger = logging.getLogger("backend.verification.error_classifier")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    """Categories of errors that can occur in the verification workflow."""
    LLM_TIMEOUT = "llm_timeout"
    DB_CONNECTION = "db_connection"
    VALIDATION_ERROR = "validation_error"
    PRICE_MISMATCH = "price_mismatch"
    POLICY_FABRICATION = "policy_fabrication"
    RELEVANCE_FAILURE = "relevance_failure"
    STATE_CORRUPTION = "state_corruption"
    API_RATE_LIMIT = "api_rate_limit"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Severity and action mappings
# ---------------------------------------------------------------------------

CATEGORY_SEVERITY_MAP: Dict[ErrorCategory, str] = {
    ErrorCategory.LLM_TIMEOUT: "major",
    ErrorCategory.DB_CONNECTION: "critical",
    ErrorCategory.VALIDATION_ERROR: "minor",
    ErrorCategory.PRICE_MISMATCH: "major",       # default; overridden by deviation %
    ErrorCategory.POLICY_FABRICATION: "critical",
    ErrorCategory.RELEVANCE_FAILURE: "minor",
    ErrorCategory.STATE_CORRUPTION: "critical",
    ErrorCategory.API_RATE_LIMIT: "major",
    ErrorCategory.UNKNOWN: "minor",
}

CATEGORY_ACTION_MAP: Dict[ErrorCategory, str] = {
    ErrorCategory.LLM_TIMEOUT: "Retry the LLM call with exponential backoff",
    ErrorCategory.DB_CONNECTION: "Check database connectivity and retry after reconnection",
    ErrorCategory.VALIDATION_ERROR: "Inspect input data and fix validation issues before retrying",
    ErrorCategory.PRICE_MISMATCH: "Cross-check price against the product catalog and correct the draft",
    ErrorCategory.POLICY_FABRICATION: "Remove fabricated policy claims and replace with verified official policies",
    ErrorCategory.RELEVANCE_FAILURE: "Revise the draft to better address the customer objection",
    ErrorCategory.STATE_CORRUPTION: "Reset workflow state and restart from a clean checkpoint",
    ErrorCategory.API_RATE_LIMIT: "Wait for rate limit window to reset, then retry",
    ErrorCategory.UNKNOWN: "Investigate the error manually and determine the appropriate action",
}

# Categories that are safe to retry automatically
_RETRIABLE_CATEGORIES = frozenset({
    ErrorCategory.LLM_TIMEOUT,
    ErrorCategory.API_RATE_LIMIT,
    ErrorCategory.DB_CONNECTION,
})


# ---------------------------------------------------------------------------
# ClassifiedError model
# ---------------------------------------------------------------------------

class ClassifiedError(BaseModel):
    """
    A fully classified error with severity, retriability, and remediation context.

    **Validates: Requirements 8.1** - structured error handling with context
    """

    error_category: ErrorCategory = Field(description="Classified error category")
    severity: str = Field(description="Severity level: critical, major, or minor")
    original_error: str = Field(description="String representation of the original error")
    context: Dict = Field(default_factory=dict, description="Additional context for debugging")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error was classified")
    is_retriable: bool = Field(description="Whether the error can be retried automatically")
    suggested_action: str = Field(description="Recommended remediation action")
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error_category": "llm_timeout",
            "severity": "major",
            "original_error": "TimeoutError: LLM call exceeded 30s",
            "context": {"component": "verification", "correlation_id": "corr_abc123"},
            "is_retriable": True,
            "suggested_action": "Retry the LLM call with exponential backoff",
        }
    })


# ---------------------------------------------------------------------------
# ErrorClassifier
# ---------------------------------------------------------------------------

class ErrorClassifier:
    """
    Classifies exceptions and structured verification issues into ClassifiedError.

    Usage::

        classifier = ErrorClassifier()

        # From an exception
        try:
            await call_llm(prompt)
        except Exception as exc:
            classified = classifier.classify(exc, context={"component": "verification"})

        # From a structured issue
        issue = PriceIssue(product_name="iPhone 15", deviation_percent=35.0, ...)
        classified = classifier.classify_from_issue(issue)
    """

    # Exception type name → ErrorCategory mapping (checked via isinstance / type name)
    _EXCEPTION_CATEGORY_MAP: Dict[str, ErrorCategory] = {
        "TimeoutError": ErrorCategory.LLM_TIMEOUT,
        "asyncio.TimeoutError": ErrorCategory.LLM_TIMEOUT,
        "concurrent.futures.TimeoutError": ErrorCategory.LLM_TIMEOUT,
        "ConnectionError": ErrorCategory.DB_CONNECTION,
        "ConnectionRefusedError": ErrorCategory.DB_CONNECTION,
        "ConnectionResetError": ErrorCategory.DB_CONNECTION,
        "ConnectionAbortedError": ErrorCategory.DB_CONNECTION,
        "OSError": ErrorCategory.DB_CONNECTION,
        "ValueError": ErrorCategory.VALIDATION_ERROR,
        "TypeError": ErrorCategory.VALIDATION_ERROR,
        "ValidationError": ErrorCategory.VALIDATION_ERROR,
        "pydantic.ValidationError": ErrorCategory.VALIDATION_ERROR,
        "RateLimitError": ErrorCategory.API_RATE_LIMIT,
        "openai.RateLimitError": ErrorCategory.API_RATE_LIMIT,
        "anthropic.RateLimitError": ErrorCategory.API_RATE_LIMIT,
        "KeyError": ErrorCategory.STATE_CORRUPTION,
        "AttributeError": ErrorCategory.STATE_CORRUPTION,
    }

    def classify(self, exception: Exception, context: Optional[Dict] = None) -> ClassifiedError:
        """
        Classify an exception into a ClassifiedError.

        The classification uses the exception's type name (and MRO) to look up
        the most specific ErrorCategory. Falls back to UNKNOWN for unrecognised
        exception types.

        Args:
            exception: The exception to classify.
            context: Optional dict with additional debugging context
                     (e.g. component name, correlation_id).

        Returns:
            ClassifiedError with populated severity, retriability, and action.
        """
        category = self._categorize_exception(exception)
        severity = self.get_severity_for_category(category)
        is_retriable = category in _RETRIABLE_CATEGORIES

        logger.debug(
            "Classified exception %s → category=%s severity=%s retriable=%s",
            type(exception).__name__,
            category.value,
            severity,
            is_retriable,
        )

        return ClassifiedError(
            error_category=category,
            severity=severity,
            original_error=str(exception),
            context=context or {},
            is_retriable=is_retriable,
            suggested_action=CATEGORY_ACTION_MAP[category],
        )

    def classify_from_issue(
        self,
        issue: Union[PriceIssue, PolicyIssue, RelevanceIssue],
    ) -> ClassifiedError:
        """
        Map a structured verification issue to a ClassifiedError.

        Severity is derived from the issue's own severity field, with special
        handling for PriceIssue deviation thresholds and PolicyIssue fabrication.

        Args:
            issue: A PriceIssue, PolicyIssue, or RelevanceIssue instance.

        Returns:
            ClassifiedError reflecting the issue's category and severity.
        """
        if isinstance(issue, PriceIssue):
            return self._classify_price_issue(issue)
        elif isinstance(issue, PolicyIssue):
            return self._classify_policy_issue(issue)
        elif isinstance(issue, RelevanceIssue):
            return self._classify_relevance_issue(issue)
        else:
            raise TypeError(f"Unsupported issue type: {type(issue)}")

    def get_severity_for_category(self, category: ErrorCategory) -> str:
        """
        Return the default severity string for an ErrorCategory.

        Args:
            category: An ErrorCategory enum value.

        Returns:
            One of "critical", "major", or "minor".
        """
        return CATEGORY_SEVERITY_MAP[category]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _categorize_exception(self, exception: Exception) -> ErrorCategory:
        """
        Determine the ErrorCategory for an exception by inspecting its MRO.

        Checks the full qualified name first, then the simple class name,
        then walks the MRO for base classes.
        """
        # Check fully-qualified name first (e.g. "openai.RateLimitError")
        fqn = f"{type(exception).__module__}.{type(exception).__name__}"
        if fqn in self._EXCEPTION_CATEGORY_MAP:
            return self._EXCEPTION_CATEGORY_MAP[fqn]

        # Check simple class name
        simple_name = type(exception).__name__
        if simple_name in self._EXCEPTION_CATEGORY_MAP:
            return self._EXCEPTION_CATEGORY_MAP[simple_name]

        # Walk MRO for base class matches (skip object)
        for base in type(exception).__mro__[1:]:
            if base is object:
                continue
            base_name = base.__name__
            if base_name in self._EXCEPTION_CATEGORY_MAP:
                return self._EXCEPTION_CATEGORY_MAP[base_name]

        return ErrorCategory.UNKNOWN

    def _classify_price_issue(self, issue: PriceIssue) -> ClassifiedError:
        """Map a PriceIssue to a ClassifiedError using deviation thresholds."""
        deviation = issue.deviation_percent

        # Override severity based on deviation thresholds from design doc
        if deviation is not None:
            if deviation > 30.0:
                severity = "critical"
            elif deviation >= 15.0:
                severity = "major"
            else:
                severity = "minor"
        else:
            # Fall back to the issue's own severity field
            severity = issue.severity.value

        context: Dict = {
            "product_name": issue.product_name,
            "deviation_percent": deviation,
            "mentioned_price": issue.mentioned_price,
            "actual_price": issue.actual_price,
        }
        if issue.product_sku:
            context["product_sku"] = issue.product_sku

        return ClassifiedError(
            error_category=ErrorCategory.PRICE_MISMATCH,
            severity=severity,
            original_error=issue.explanation,
            context=context,
            is_retriable=False,
            suggested_action=CATEGORY_ACTION_MAP[ErrorCategory.PRICE_MISMATCH],
        )

    def _classify_policy_issue(self, issue: PolicyIssue) -> ClassifiedError:
        """Map a PolicyIssue to a ClassifiedError."""
        if issue.is_fabricated:
            category = ErrorCategory.POLICY_FABRICATION
            severity = "critical"
        else:
            # Inaccurate or incomplete policy — use the issue's severity
            category = ErrorCategory.POLICY_FABRICATION
            severity = issue.severity.value

        context: Dict = {
            "policy_type": issue.policy_type,
            "is_fabricated": issue.is_fabricated,
            "is_inaccurate": issue.is_inaccurate,
            "is_incomplete": issue.is_incomplete,
        }
        if issue.source_document:
            context["source_document"] = issue.source_document

        return ClassifiedError(
            error_category=category,
            severity=severity,
            original_error=issue.explanation,
            context=context,
            is_retriable=False,
            suggested_action=CATEGORY_ACTION_MAP[category],
        )

    def _classify_relevance_issue(self, issue: RelevanceIssue) -> ClassifiedError:
        """Map a RelevanceIssue to a ClassifiedError."""
        context: Dict = {
            "objection_intent": issue.objection_intent,
            "response_coverage": issue.response_coverage,
            "missing_aspects": issue.missing_aspects,
        }
        if issue.empathy_score is not None:
            context["empathy_score"] = issue.empathy_score

        return ClassifiedError(
            error_category=ErrorCategory.RELEVANCE_FAILURE,
            severity=issue.severity.value,
            original_error=issue.explanation,
            context=context,
            is_retriable=False,
            suggested_action=CATEGORY_ACTION_MAP[ErrorCategory.RELEVANCE_FAILURE],
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_classifier: Optional[ErrorClassifier] = None


def get_error_classifier() -> ErrorClassifier:
    """Return the module-level singleton ErrorClassifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = ErrorClassifier()
    return _default_classifier


__all__ = [
    "ErrorCategory",
    "CATEGORY_SEVERITY_MAP",
    "CATEGORY_ACTION_MAP",
    "ClassifiedError",
    "ErrorClassifier",
    "get_error_classifier",
]
