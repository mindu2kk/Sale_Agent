"""
Unit tests for binary decision logic in VerificationAgent (Task 2.4.3)

Tests cover:
- PASS decision when all 3 checks pass (no critical issues)
- FAIL decision when any check fails, with issue aggregation
- Critical issues count computed correctly across all issue lists
- overall_pass = price_accuracy_pass AND policy_authenticity_pass AND topic_relevance_pass
- Immediate escalation flag when critical_issues_count >= 3
- is_approved reflects overall_pass correctly

Requirements validated: 1.1, 1.2, 1.3, 4.3, 5.3, 6.3
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from backend.verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
)
from backend.verification.agent.verification_agent import VerificationAgent
from backend.verification.config import VerificationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**kwargs) -> VerificationConfig:
    """Create a minimal VerificationConfig for testing."""
    defaults = dict(
        price_tolerance_percent=1.0,
        price_critical_threshold=30.0,
        parallel_verification=False,
        early_termination=False,
        enable_caching=False,
        async_timeout_seconds=30,
        retry_backoff_seconds=0.1,
        max_draft_length=10000,
        max_objection_length=5000,
    )
    defaults.update(kwargs)
    return VerificationConfig(**defaults)


def make_agent() -> VerificationAgent:
    """Create a VerificationAgent with mocked dependencies."""
    llm = MagicMock()
    rag = MagicMock()
    config = make_config()
    return VerificationAgent(llm, rag, config)


def make_state(draft: str = "Test draft response", objection: str = "Test objection") -> dict:
    return {"draft_response": draft, "objection_text": objection}


# ---------------------------------------------------------------------------
# Tests for _build_verification_result (binary decision core)
# ---------------------------------------------------------------------------

class TestBuildVerificationResult:
    """Tests for the _build_verification_result helper method."""

    def setup_method(self):
        self.agent = make_agent()
        self.state = make_state()

    def test_pass_when_all_checks_pass_no_issues(self):
        """PASS: all 3 checks pass, no issues → overall_pass=True, is_approved=True."""
        result = self.agent._build_verification_result(
            self.state,
            price_pass=True, price_issues=[],
            policy_pass=True, policy_issues=[],
            relevance_pass=True, relevance_issues=[],
        )

        assert result.criteria.overall_pass is True
        assert result.is_approved is True
        assert result.criteria.critical_issues_count == 0
        assert result.requires_correction is False

    def test_fail_when_price_check_fails(self):
        """FAIL: price check fails → overall_pass=False, is_approved=False."""
        price_issue = PriceIssue(
            product_name="iPhone 15",
            mentioned_price="50,000,000 VND",
            actual_price="30,000,000 VND",
            deviation_percent=66.7,
            severity=IssueSeverity.CRITICAL,
            explanation="Price deviation 66.7% exceeds tolerance",
        )

        result = self.agent._build_verification_result(
            self.state,
            price_pass=False, price_issues=[price_issue],
            policy_pass=True, policy_issues=[],
            relevance_pass=True, relevance_issues=[],
        )

        assert result.criteria.overall_pass is False
        assert result.is_approved is False
        assert result.criteria.price_accuracy_pass is False
        assert result.criteria.policy_authenticity_pass is True
        assert result.criteria.topic_relevance_pass is True
        assert result.criteria.critical_issues_count == 1

    def test_fail_when_policy_check_fails(self):
        """FAIL: policy check fails → overall_pass=False."""
        policy_issue = PolicyIssue(
            mentioned_policy="Bảo hành 5 năm miễn phí",
            policy_type="warranty",
            is_fabricated=True,
            is_inaccurate=False,
            severity=IssueSeverity.CRITICAL,
            explanation="Fabricated policy",
        )

        result = self.agent._build_verification_result(
            self.state,
            price_pass=True, price_issues=[],
            policy_pass=False, policy_issues=[policy_issue],
            relevance_pass=True, relevance_issues=[],
        )

        assert result.criteria.overall_pass is False
        assert result.is_approved is False
        assert result.criteria.policy_authenticity_pass is False
        assert result.criteria.critical_issues_count == 1

    def test_fail_when_relevance_check_fails(self):
        """FAIL: relevance check fails → overall_pass=False."""
        relevance_issue = RelevanceIssue(
            objection_intent="Price comparison",
            response_coverage=0.1,
            severity=IssueSeverity.CRITICAL,
            explanation="Response severely off-topic",
        )

        result = self.agent._build_verification_result(
            self.state,
            price_pass=True, price_issues=[],
            policy_pass=True, policy_issues=[],
            relevance_pass=False, relevance_issues=[relevance_issue],
        )

        assert result.criteria.overall_pass is False
        assert result.is_approved is False
        assert result.criteria.topic_relevance_pass is False
        assert result.criteria.critical_issues_count == 1

    def test_fail_when_all_checks_fail(self):
        """FAIL: all 3 checks fail → overall_pass=False."""
        result = self.agent._build_verification_result(
            self.state,
            price_pass=False, price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.MAJOR, explanation="x")
            ],
            policy_pass=False, policy_issues=[
                PolicyIssue(
                    mentioned_policy="p", policy_type="warranty",
                    is_fabricated=False, is_inaccurate=True,
                    severity=IssueSeverity.MAJOR, explanation="y",
                )
            ],
            relevance_pass=False, relevance_issues=[
                RelevanceIssue(
                    objection_intent="z", response_coverage=0.3,
                    severity=IssueSeverity.MAJOR, explanation="w",
                )
            ],
        )

        assert result.criteria.overall_pass is False
        assert result.is_approved is False
        assert result.criteria.price_accuracy_pass is False
        assert result.criteria.policy_authenticity_pass is False
        assert result.criteria.topic_relevance_pass is False

    def test_critical_issues_aggregated_across_all_lists(self):
        """Critical issues are counted across price + policy + relevance lists."""
        result = self.agent._build_verification_result(
            self.state,
            price_pass=False, price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.CRITICAL, explanation="c1"),
                PriceIssue(product_name="B", severity=IssueSeverity.MINOR, explanation="m1"),
            ],
            policy_pass=False, policy_issues=[
                PolicyIssue(
                    mentioned_policy="p", policy_type="warranty",
                    is_fabricated=True, is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL, explanation="c2",
                )
            ],
            relevance_pass=False, relevance_issues=[
                RelevanceIssue(
                    objection_intent="z", response_coverage=0.1,
                    severity=IssueSeverity.CRITICAL, explanation="c3",
                )
            ],
        )

        # 3 critical issues: 1 price + 1 policy + 1 relevance
        assert result.criteria.critical_issues_count == 3
        assert result.is_approved is False

    def test_non_critical_issues_not_counted_as_critical(self):
        """MAJOR and MINOR issues do not increment critical_issues_count."""
        result = self.agent._build_verification_result(
            self.state,
            price_pass=False, price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.MAJOR, explanation="major"),
                PriceIssue(product_name="B", severity=IssueSeverity.MINOR, explanation="minor"),
            ],
            policy_pass=True, policy_issues=[],
            relevance_pass=True, relevance_issues=[],
        )

        assert result.criteria.critical_issues_count == 0
        assert result.is_approved is False  # price failed

    def test_overall_pass_is_and_of_all_three(self):
        """overall_pass = price AND policy AND relevance (all must be True)."""
        # Only price fails
        r1 = self.agent._build_verification_result(
            self.state,
            price_pass=False, price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="x")
            ],
            policy_pass=True, policy_issues=[],
            relevance_pass=True, relevance_issues=[],
        )
        assert r1.criteria.overall_pass is False

        # Only policy fails
        r2 = self.agent._build_verification_result(
            self.state,
            price_pass=True, price_issues=[],
            policy_pass=False, policy_issues=[
                PolicyIssue(
                    mentioned_policy="p", policy_type="return",
                    is_fabricated=False, is_inaccurate=True,
                    severity=IssueSeverity.MINOR, explanation="y",
                )
            ],
            relevance_pass=True, relevance_issues=[],
        )
        assert r2.criteria.overall_pass is False

        # Only relevance fails
        r3 = self.agent._build_verification_result(
            self.state,
            price_pass=True, price_issues=[],
            policy_pass=True, policy_issues=[],
            relevance_pass=False, relevance_issues=[
                RelevanceIssue(
                    objection_intent="z", response_coverage=0.5,
                    severity=IssueSeverity.MINOR, explanation="w",
                )
            ],
        )
        assert r3.criteria.overall_pass is False

        # All pass
        r4 = self.agent._build_verification_result(
            self.state,
            price_pass=True, price_issues=[],
            policy_pass=True, policy_issues=[],
            relevance_pass=True, relevance_issues=[],
        )
        assert r4.criteria.overall_pass is True


# ---------------------------------------------------------------------------
# Tests for reasoning generation with escalation flag
# ---------------------------------------------------------------------------

class TestVerificationReasoning:
    """Tests for _generate_verification_reasoning with escalation logic."""

    def setup_method(self):
        self.agent = make_agent()

    def _make_criteria(self, price_pass, policy_pass, relevance_pass,
                       price_issues=None, policy_issues=None, relevance_issues=None):
        return RubricCriteria(
            price_accuracy_pass=price_pass,
            policy_authenticity_pass=policy_pass,
            topic_relevance_pass=relevance_pass,
            price_issues=price_issues or [],
            policy_issues=policy_issues or [],
            relevance_issues=relevance_issues or [],
        )

    def test_reasoning_approved_when_all_pass(self):
        criteria = self._make_criteria(True, True, True)
        reasoning = self.agent._generate_verification_reasoning(criteria)
        assert "APPROVED" in reasoning
        assert "✅ Price Accuracy: PASS" in reasoning
        assert "✅ Policy Authenticity: PASS" in reasoning
        assert "✅ Topic Relevance: PASS" in reasoning

    def test_reasoning_requires_correction_when_any_fails(self):
        criteria = self._make_criteria(
            False, True, True,
            price_issues=[PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="err")]
        )
        reasoning = self.agent._generate_verification_reasoning(criteria)
        assert "REQUIRES CORRECTION" in reasoning
        assert "❌ Price Accuracy: FAIL" in reasoning

    def test_reasoning_flags_critical_issues(self):
        criteria = self._make_criteria(
            False, True, True,
            price_issues=[PriceIssue(product_name="A", severity=IssueSeverity.CRITICAL, explanation="crit")]
        )
        reasoning = self.agent._generate_verification_reasoning(criteria)
        assert "1 critical issue" in reasoning

    def test_reasoning_flags_immediate_escalation_at_3_critical(self):
        """When critical_issues_count >= 3, reasoning must flag immediate escalation."""
        criteria = self._make_criteria(
            False, False, False,
            price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.CRITICAL, explanation="c1")
            ],
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="p", policy_type="warranty",
                    is_fabricated=True, is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL, explanation="c2",
                )
            ],
            relevance_issues=[
                RelevanceIssue(
                    objection_intent="z", response_coverage=0.1,
                    severity=IssueSeverity.CRITICAL, explanation="c3",
                )
            ],
        )
        assert criteria.critical_issues_count == 3
        reasoning = self.agent._generate_verification_reasoning(criteria)
        assert "IMMEDIATE ESCALATION REQUIRED" in reasoning

    def test_reasoning_no_escalation_flag_below_3_critical(self):
        """When critical_issues_count < 3, no immediate escalation flag."""
        criteria = self._make_criteria(
            False, True, True,
            price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.CRITICAL, explanation="c1"),
                PriceIssue(product_name="B", severity=IssueSeverity.CRITICAL, explanation="c2"),
            ],
        )
        assert criteria.critical_issues_count == 2
        reasoning = self.agent._generate_verification_reasoning(criteria)
        assert "IMMEDIATE ESCALATION REQUIRED" not in reasoning
        assert "2 critical issue" in reasoning


# ---------------------------------------------------------------------------
# Tests for issue aggregation in VerificationResult
# ---------------------------------------------------------------------------

class TestIssueAggregation:
    """Tests that all issue types are properly aggregated into RubricCriteria."""

    def setup_method(self):
        self.agent = make_agent()
        self.state = make_state()

    def test_all_issue_types_stored_in_criteria(self):
        """All PriceIssue, PolicyIssue, RelevanceIssue objects are in criteria."""
        price_issue = PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="p")
        policy_issue = PolicyIssue(
            mentioned_policy="q", policy_type="return",
            is_fabricated=False, is_inaccurate=True,
            severity=IssueSeverity.MAJOR, explanation="r",
        )
        relevance_issue = RelevanceIssue(
            objection_intent="s", response_coverage=0.5,
            severity=IssueSeverity.MINOR, explanation="t",
        )

        result = self.agent._build_verification_result(
            self.state,
            price_pass=False, price_issues=[price_issue],
            policy_pass=False, policy_issues=[policy_issue],
            relevance_pass=False, relevance_issues=[relevance_issue],
        )

        assert len(result.criteria.price_issues) == 1
        assert len(result.criteria.policy_issues) == 1
        assert len(result.criteria.relevance_issues) == 1
        assert result.criteria.price_issues[0].product_name == "A"
        assert result.criteria.policy_issues[0].mentioned_policy == "q"
        assert result.criteria.relevance_issues[0].objection_intent == "s"

    def test_empty_issue_lists_when_all_pass(self):
        """When all checks pass, all issue lists are empty."""
        result = self.agent._build_verification_result(
            self.state,
            price_pass=True, price_issues=[],
            policy_pass=True, policy_issues=[],
            relevance_pass=True, relevance_issues=[],
        )

        assert result.criteria.price_issues == []
        assert result.criteria.policy_issues == []
        assert result.criteria.relevance_issues == []

    def test_multiple_issues_per_category(self):
        """Multiple issues per category are all preserved."""
        result = self.agent._build_verification_result(
            self.state,
            price_pass=False, price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.CRITICAL, explanation="c1"),
                PriceIssue(product_name="B", severity=IssueSeverity.MAJOR, explanation="m1"),
                PriceIssue(product_name="C", severity=IssueSeverity.MINOR, explanation="n1"),
            ],
            policy_pass=True, policy_issues=[],
            relevance_pass=True, relevance_issues=[],
        )

        assert len(result.criteria.price_issues) == 3
        assert result.criteria.critical_issues_count == 1  # only 1 critical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
