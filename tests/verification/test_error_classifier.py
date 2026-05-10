"""
Tests for ErrorClassifier - Task 6.2.3

Covers:
- Classification of different exception types
- Mapping from PriceIssue/PolicyIssue/RelevanceIssue to ClassifiedError
- Severity mapping correctness
- is_retriable logic
- CATEGORY_SEVERITY_MAP and CATEGORY_ACTION_MAP completeness
"""

import pytest
from datetime import datetime

from verification.utils.error_classifier import (
    ErrorCategory,
    ClassifiedError,
    ErrorClassifier,
    CATEGORY_SEVERITY_MAP,
    CATEGORY_ACTION_MAP,
    get_error_classifier,
)
from verification.models.verification import (
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier():
    return ErrorClassifier()


# ---------------------------------------------------------------------------
# CATEGORY_SEVERITY_MAP and CATEGORY_ACTION_MAP completeness
# ---------------------------------------------------------------------------

def test_severity_map_covers_all_categories():
    """Every ErrorCategory must have a severity entry."""
    for category in ErrorCategory:
        assert category in CATEGORY_SEVERITY_MAP, f"Missing severity for {category}"


def test_action_map_covers_all_categories():
    """Every ErrorCategory must have an action entry."""
    for category in ErrorCategory:
        assert category in CATEGORY_ACTION_MAP, f"Missing action for {category}"


def test_severity_values_are_valid():
    """All severity values must be one of the three allowed levels."""
    valid = {"critical", "major", "minor"}
    for category, severity in CATEGORY_SEVERITY_MAP.items():
        assert severity in valid, f"{category} has invalid severity '{severity}'"


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------

class TestClassifyExceptions:

    def test_timeout_error_is_llm_timeout(self, classifier):
        exc = TimeoutError("LLM call timed out")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.LLM_TIMEOUT

    def test_connection_error_is_db_connection(self, classifier):
        exc = ConnectionError("Cannot connect to database")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.DB_CONNECTION

    def test_connection_refused_is_db_connection(self, classifier):
        exc = ConnectionRefusedError("Connection refused")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.DB_CONNECTION

    def test_value_error_is_validation_error(self, classifier):
        exc = ValueError("Invalid input")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.VALIDATION_ERROR

    def test_type_error_is_validation_error(self, classifier):
        exc = TypeError("Wrong type")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.VALIDATION_ERROR

    def test_key_error_is_state_corruption(self, classifier):
        exc = KeyError("missing_key")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.STATE_CORRUPTION

    def test_attribute_error_is_state_corruption(self, classifier):
        exc = AttributeError("object has no attribute 'x'")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.STATE_CORRUPTION

    def test_unknown_exception_is_unknown(self, classifier):
        class WeirdError(Exception):
            pass
        exc = WeirdError("something strange")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.UNKNOWN

    def test_context_is_stored(self, classifier):
        exc = TimeoutError("timeout")
        ctx = {"component": "verification", "correlation_id": "corr_abc"}
        result = classifier.classify(exc, context=ctx)
        assert result.context == ctx

    def test_original_error_is_str(self, classifier):
        exc = ValueError("bad value")
        result = classifier.classify(exc)
        assert isinstance(result.original_error, str)
        assert "bad value" in result.original_error

    def test_timestamp_is_set(self, classifier):
        exc = TimeoutError("t")
        result = classifier.classify(exc)
        assert isinstance(result.timestamp, datetime)

    def test_subclass_inherits_base_category(self, classifier):
        """A subclass of ConnectionError should map to DB_CONNECTION."""
        class CustomDBError(ConnectionError):
            pass
        exc = CustomDBError("custom db error")
        result = classifier.classify(exc)
        assert result.error_category == ErrorCategory.DB_CONNECTION


# ---------------------------------------------------------------------------
# is_retriable logic
# ---------------------------------------------------------------------------

class TestRetriable:

    def test_timeout_is_retriable(self, classifier):
        result = classifier.classify(TimeoutError("t"))
        assert result.is_retriable is True

    def test_connection_error_is_retriable(self, classifier):
        result = classifier.classify(ConnectionError("c"))
        assert result.is_retriable is True

    def test_validation_error_is_not_retriable(self, classifier):
        result = classifier.classify(ValueError("v"))
        assert result.is_retriable is False

    def test_state_corruption_is_not_retriable(self, classifier):
        result = classifier.classify(KeyError("k"))
        assert result.is_retriable is False

    def test_unknown_is_not_retriable(self, classifier):
        class Weird(Exception):
            pass
        result = classifier.classify(Weird("w"))
        assert result.is_retriable is False

    def test_api_rate_limit_category_is_retriable(self, classifier):
        """API_RATE_LIMIT should be retriable per design doc."""
        severity = classifier.get_severity_for_category(ErrorCategory.API_RATE_LIMIT)
        assert severity == "major"
        # Verify retriability via direct category check
        from verification.utils.error_classifier import _RETRIABLE_CATEGORIES
        assert ErrorCategory.API_RATE_LIMIT in _RETRIABLE_CATEGORIES


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

class TestSeverityMapping:

    def test_db_connection_is_critical(self, classifier):
        assert classifier.get_severity_for_category(ErrorCategory.DB_CONNECTION) == "critical"

    def test_policy_fabrication_is_critical(self, classifier):
        assert classifier.get_severity_for_category(ErrorCategory.POLICY_FABRICATION) == "critical"

    def test_state_corruption_is_critical(self, classifier):
        assert classifier.get_severity_for_category(ErrorCategory.STATE_CORRUPTION) == "critical"

    def test_llm_timeout_is_major(self, classifier):
        assert classifier.get_severity_for_category(ErrorCategory.LLM_TIMEOUT) == "major"

    def test_api_rate_limit_is_major(self, classifier):
        assert classifier.get_severity_for_category(ErrorCategory.API_RATE_LIMIT) == "major"

    def test_validation_error_is_minor(self, classifier):
        assert classifier.get_severity_for_category(ErrorCategory.VALIDATION_ERROR) == "minor"

    def test_relevance_failure_is_minor(self, classifier):
        assert classifier.get_severity_for_category(ErrorCategory.RELEVANCE_FAILURE) == "minor"


# ---------------------------------------------------------------------------
# PriceIssue → ClassifiedError
# ---------------------------------------------------------------------------

class TestClassifyPriceIssue:

    def _make_price_issue(self, deviation: float, severity: IssueSeverity) -> PriceIssue:
        return PriceIssue(
            product_name="iPhone 15",
            mentioned_price="35,000,000 VND",
            actual_price="29,990,000 VND",
            deviation_percent=deviation,
            severity=severity,
            explanation=f"Price deviation {deviation}%",
        )

    def test_deviation_over_30_is_critical(self, classifier):
        issue = self._make_price_issue(35.0, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.error_category == ErrorCategory.PRICE_MISMATCH
        assert result.severity == "critical"

    def test_deviation_between_15_and_30_is_major(self, classifier):
        issue = self._make_price_issue(20.0, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.severity == "major"

    def test_deviation_under_15_is_minor(self, classifier):
        issue = self._make_price_issue(5.0, IssueSeverity.MINOR)
        result = classifier.classify_from_issue(issue)
        assert result.severity == "minor"

    def test_deviation_exactly_15_is_major(self, classifier):
        issue = self._make_price_issue(15.0, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.severity == "major"

    def test_deviation_exactly_30_is_major(self, classifier):
        """30% is not strictly > 30, so it should be major."""
        issue = self._make_price_issue(30.0, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.severity == "major"

    def test_no_deviation_falls_back_to_issue_severity(self, classifier):
        issue = PriceIssue(
            product_name="Unknown",
            severity=IssueSeverity.MAJOR,
            explanation="Cannot verify price",
        )
        result = classifier.classify_from_issue(issue)
        assert result.severity == "major"

    def test_price_issue_is_not_retriable(self, classifier):
        issue = self._make_price_issue(10.0, IssueSeverity.MINOR)
        result = classifier.classify_from_issue(issue)
        assert result.is_retriable is False

    def test_context_contains_product_info(self, classifier):
        issue = self._make_price_issue(20.0, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.context["product_name"] == "iPhone 15"
        assert result.context["deviation_percent"] == 20.0


# ---------------------------------------------------------------------------
# PolicyIssue → ClassifiedError
# ---------------------------------------------------------------------------

class TestClassifyPolicyIssue:

    def _make_policy_issue(
        self,
        is_fabricated: bool,
        is_inaccurate: bool,
        severity: IssueSeverity,
    ) -> PolicyIssue:
        return PolicyIssue(
            mentioned_policy="Bảo hành 5 năm miễn phí",
            policy_type="warranty",
            is_fabricated=is_fabricated,
            is_inaccurate=is_inaccurate,
            severity=severity,
            explanation="Policy issue detected",
        )

    def test_fabricated_policy_is_critical(self, classifier):
        issue = self._make_policy_issue(True, False, IssueSeverity.CRITICAL)
        result = classifier.classify_from_issue(issue)
        assert result.error_category == ErrorCategory.POLICY_FABRICATION
        assert result.severity == "critical"

    def test_inaccurate_policy_uses_issue_severity(self, classifier):
        issue = self._make_policy_issue(False, True, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.severity == "major"

    def test_incomplete_policy_uses_issue_severity(self, classifier):
        issue = PolicyIssue(
            mentioned_policy="Bảo hành 1 năm",
            policy_type="warranty",
            is_fabricated=False,
            is_inaccurate=False,
            is_incomplete=True,
            severity=IssueSeverity.MINOR,
            explanation="Policy is incomplete",
        )
        result = classifier.classify_from_issue(issue)
        assert result.severity == "minor"

    def test_policy_issue_is_not_retriable(self, classifier):
        issue = self._make_policy_issue(True, False, IssueSeverity.CRITICAL)
        result = classifier.classify_from_issue(issue)
        assert result.is_retriable is False

    def test_context_contains_policy_type(self, classifier):
        issue = self._make_policy_issue(True, False, IssueSeverity.CRITICAL)
        result = classifier.classify_from_issue(issue)
        assert result.context["policy_type"] == "warranty"
        assert result.context["is_fabricated"] is True


# ---------------------------------------------------------------------------
# RelevanceIssue → ClassifiedError
# ---------------------------------------------------------------------------

class TestClassifyRelevanceIssue:

    def _make_relevance_issue(self, coverage: float, severity: IssueSeverity) -> RelevanceIssue:
        return RelevanceIssue(
            objection_intent="So sánh giá iPhone vs Samsung",
            response_coverage=coverage,
            severity=severity,
            explanation=f"Coverage only {coverage:.0%}",
        )

    def test_relevance_issue_maps_to_relevance_failure(self, classifier):
        issue = self._make_relevance_issue(0.5, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.error_category == ErrorCategory.RELEVANCE_FAILURE

    def test_relevance_issue_uses_issue_severity(self, classifier):
        issue = self._make_relevance_issue(0.5, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.severity == "major"

    def test_minor_relevance_issue(self, classifier):
        issue = self._make_relevance_issue(0.8, IssueSeverity.MINOR)
        result = classifier.classify_from_issue(issue)
        assert result.severity == "minor"

    def test_relevance_issue_is_not_retriable(self, classifier):
        issue = self._make_relevance_issue(0.3, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.is_retriable is False

    def test_context_contains_coverage(self, classifier):
        issue = self._make_relevance_issue(0.6, IssueSeverity.MAJOR)
        result = classifier.classify_from_issue(issue)
        assert result.context["response_coverage"] == 0.6
        assert result.context["objection_intent"] == "So sánh giá iPhone vs Samsung"


# ---------------------------------------------------------------------------
# classify_from_issue type guard
# ---------------------------------------------------------------------------

def test_classify_from_issue_raises_for_unknown_type(classifier):
    with pytest.raises(TypeError):
        classifier.classify_from_issue("not an issue")  # type: ignore


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_get_error_classifier_returns_same_instance():
    c1 = get_error_classifier()
    c2 = get_error_classifier()
    assert c1 is c2


def test_get_error_classifier_is_error_classifier():
    assert isinstance(get_error_classifier(), ErrorClassifier)
