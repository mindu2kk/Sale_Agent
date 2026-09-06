"""
Tests for structured feedback generation (Task 2.4.4)

Covers:
- FeedbackReport and FailedCriterion Pydantic models
- VerificationAgent.generate_structured_feedback()
- SelfCorrectionNode.generate_correction_feedback() with FeedbackReport
- Severity-aware prioritization (critical first)
- Deterministic / template-based output (no LLM calls)
"""

import pytest
from unittest.mock import MagicMock
from backend.verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
    FeedbackReport,
    FailedCriterion,
)
from backend.verification.agent.verification_agent import VerificationAgent
from backend.verification.workflow.correction import SelfCorrectionNode
from backend.verification.config import VerificationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> VerificationConfig:
    return VerificationConfig()


def _make_agent() -> VerificationAgent:
    config = _make_config()
    return VerificationAgent(
        llm=MagicMock(),
        rag_pipeline=MagicMock(),
        config=config,
    )


def _make_result(
    price_pass=True,
    policy_pass=True,
    relevance_pass=True,
    price_issues=None,
    policy_issues=None,
    relevance_issues=None,
) -> VerificationResult:
    criteria = RubricCriteria(
        price_accuracy_pass=price_pass,
        policy_authenticity_pass=policy_pass,
        topic_relevance_pass=relevance_pass,
        price_issues=price_issues or [],
        policy_issues=policy_issues or [],
        relevance_issues=relevance_issues or [],
    )
    return VerificationResult(
        criteria=criteria,
        verification_reasoning="Test reasoning for structured feedback",
        execution_time_seconds=1.0,
        llm_tokens_used=500,
    )


# ---------------------------------------------------------------------------
# FeedbackReport model tests
# ---------------------------------------------------------------------------

class TestFeedbackReportModel:
    def test_approved_report_has_no_failed_criteria(self):
        report = FeedbackReport(
            is_approved=True,
            total_issues=0,
            critical_issues_count=0,
            escalation_priority="low",
            failed_criteria=[],
            correction_prompt="✅ No corrections needed.",
        )
        assert report.is_approved is True
        assert report.failed_criteria == []
        assert report.total_issues == 0

    def test_failed_report_contains_criteria(self):
        fc = FailedCriterion(
            criterion_name="Price Accuracy",
            criterion_key="price_accuracy",
            explanation="1 price issue detected",
            correction_suggestions=["Update price to 29,990,000 VND"],
            severity=IssueSeverity.MAJOR,
            issue_count=1,
        )
        report = FeedbackReport(
            is_approved=False,
            total_issues=1,
            critical_issues_count=0,
            escalation_priority="medium",
            failed_criteria=[fc],
            correction_prompt="🔄 CORRECTION REQUIRED",
        )
        assert report.is_approved is False
        assert len(report.failed_criteria) == 1
        assert report.failed_criteria[0].criterion_key == "price_accuracy"

    def test_failed_criterion_severity_values(self):
        for severity in IssueSeverity:
            fc = FailedCriterion(
                criterion_name="Test",
                criterion_key="price_accuracy",
                explanation="test",
                severity=severity,
                issue_count=1,
            )
            assert fc.severity == severity


# ---------------------------------------------------------------------------
# VerificationAgent.generate_structured_feedback() tests
# ---------------------------------------------------------------------------

class TestGenerateStructuredFeedback:
    def setup_method(self):
        self.agent = _make_agent()

    def test_approved_result_returns_approved_report(self):
        result = _make_result()  # all pass
        report = self.agent.generate_structured_feedback(result)
        assert report.is_approved is True
        assert report.total_issues == 0
        assert report.failed_criteria == []
        assert "No corrections needed" in report.correction_prompt

    def test_price_failure_produces_price_criterion(self):
        result = _make_result(
            price_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="iPhone 15",
                    mentioned_price="35,000,000 VND",
                    actual_price="29,990,000 VND",
                    deviation_percent=16.7,
                    severity=IssueSeverity.MAJOR,
                    explanation="Price deviation 16.7%",
                    correction_suggestion="Update iPhone 15 price to 29,990,000 VND",
                )
            ],
        )
        report = self.agent.generate_structured_feedback(result)
        assert report.is_approved is False
        assert report.total_issues == 1
        assert len(report.failed_criteria) == 1
        fc = report.failed_criteria[0]
        assert fc.criterion_key == "price_accuracy"
        assert fc.severity == IssueSeverity.MAJOR
        assert "iPhone 15" in fc.explanation
        assert any("29,990,000" in s for s in fc.correction_suggestions)

    def test_policy_failure_produces_policy_criterion(self):
        result = _make_result(
            policy_pass=False,
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="Bảo hành 5 năm miễn phí",
                    policy_type="warranty",
                    is_fabricated=True,
                    is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL,
                    explanation="Fabricated warranty policy",
                )
            ],
        )
        report = self.agent.generate_structured_feedback(result)
        assert report.is_approved is False
        assert report.critical_issues_count == 1
        fc = report.failed_criteria[0]
        assert fc.criterion_key == "policy_authenticity"
        assert fc.severity == IssueSeverity.CRITICAL
        assert any("fabricated" in s.lower() or "remove" in s.lower() for s in fc.correction_suggestions)

    def test_relevance_failure_produces_relevance_criterion(self):
        result = _make_result(
            relevance_pass=False,
            relevance_issues=[
                RelevanceIssue(
                    objection_intent="Camera comparison",
                    response_coverage=0.3,
                    missing_aspects=["Camera specs", "AI features"],
                    severity=IssueSeverity.MAJOR,
                    explanation="Coverage only 30%",
                )
            ],
        )
        report = self.agent.generate_structured_feedback(result)
        fc = report.failed_criteria[0]
        assert fc.criterion_key == "topic_relevance"
        assert "Camera comparison" in fc.explanation
        assert any("Camera specs" in s or "missing" in s.lower() for s in fc.correction_suggestions)

    def test_severity_prioritization_critical_first(self):
        """Critical issues must appear before major/minor in failed_criteria."""
        result = _make_result(
            price_pass=False,
            policy_pass=False,
            relevance_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="Product A",
                    severity=IssueSeverity.MINOR,
                    explanation="Minor price issue",
                )
            ],
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="Fake policy",
                    policy_type="warranty",
                    is_fabricated=True,
                    is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL,
                    explanation="Critical fabricated policy",
                )
            ],
            relevance_issues=[
                RelevanceIssue(
                    objection_intent="Test",
                    response_coverage=0.5,
                    severity=IssueSeverity.MAJOR,
                    explanation="Major relevance issue",
                )
            ],
        )
        report = self.agent.generate_structured_feedback(result)
        severities = [fc.severity for fc in report.failed_criteria]
        # Critical must come before major, major before minor
        order = {IssueSeverity.CRITICAL: 0, IssueSeverity.MAJOR: 1, IssueSeverity.MINOR: 2}
        assert severities == sorted(severities, key=lambda s: order[s])

    def test_correction_prompt_contains_key_sections(self):
        result = _make_result(
            price_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="Samsung S24",
                    severity=IssueSeverity.MAJOR,
                    explanation="Price mismatch",
                    correction_suggestion="Update price to 24,990,000 VND",
                )
            ],
        )
        report = self.agent.generate_structured_feedback(result)
        prompt = report.correction_prompt
        assert "VERIFICATION FAILED" in prompt
        assert "Price Accuracy" in prompt
        assert "Samsung S24" in prompt or "24,990,000" in prompt
        assert "Instructions for retry" in prompt

    def test_deterministic_output(self):
        """Same input must produce identical FeedbackReport (no randomness)."""
        result = _make_result(
            price_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="iPhone 15",
                    deviation_percent=5.0,
                    severity=IssueSeverity.MAJOR,
                    explanation="Price deviation 5%",
                )
            ],
        )
        report1 = self.agent.generate_structured_feedback(result)
        report2 = self.agent.generate_structured_feedback(result)
        assert report1.correction_prompt == report2.correction_prompt
        assert report1.total_issues == report2.total_issues
        assert report1.critical_issues_count == report2.critical_issues_count

    def test_escalation_priority_reflected_in_prompt(self):
        """Immediate/high escalation priority should appear in correction_prompt."""
        result = _make_result(
            policy_pass=False,
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="Fabricated policy",
                    policy_type="warranty",
                    is_fabricated=True,
                    is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL,
                    explanation="Fabricated",
                )
            ],
        )
        report = self.agent.generate_structured_feedback(result)
        assert report.escalation_priority in ("immediate", "high")
        assert "ESCALATION" in report.correction_prompt.upper()

    def test_multiple_issues_counted_correctly(self):
        result = _make_result(
            price_pass=False,
            policy_pass=False,
            price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="e1"),
                PriceIssue(product_name="B", severity=IssueSeverity.MAJOR, explanation="e2"),
            ],
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="p",
                    policy_type="return",
                    is_fabricated=False,
                    is_inaccurate=True,
                    severity=IssueSeverity.MAJOR,
                    explanation="e3",
                )
            ],
        )
        report = self.agent.generate_structured_feedback(result)
        assert report.total_issues == 3
        assert report.critical_issues_count == 0


# ---------------------------------------------------------------------------
# SelfCorrectionNode.generate_correction_feedback() with FeedbackReport
# ---------------------------------------------------------------------------

class TestSelfCorrectionNodeWithFeedbackReport:
    def setup_method(self):
        self.node = SelfCorrectionNode(config=_make_config())
        self.agent = _make_agent()

    def test_uses_feedback_report_when_provided(self):
        result = _make_result(
            price_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="iPhone 15",
                    severity=IssueSeverity.MAJOR,
                    explanation="Price mismatch",
                    correction_suggestion="Update to 29,990,000 VND",
                )
            ],
        )
        feedback_report = self.agent.generate_structured_feedback(result)
        correction = self.node.generate_correction_feedback(
            original_objection="Why is iPhone so expensive?",
            failed_draft="iPhone costs 35 million VND...",
            verification_result=result,
            feedback_report=feedback_report,
        )
        # Should contain the structured correction prompt content
        assert "VERIFICATION FAILED" in correction
        assert "ORIGINAL OBJECTION" in correction or "Why is iPhone" in correction

    def test_fallback_without_feedback_report(self):
        result = _make_result(
            price_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="Samsung S24",
                    severity=IssueSeverity.MAJOR,
                    explanation="Price deviation",
                )
            ],
        )
        correction = self.node.generate_correction_feedback(
            original_objection="Is Samsung cheaper?",
            failed_draft="Samsung costs 30 million...",
            verification_result=result,
        )
        assert "VERIFICATION FAILED" in correction or "CORRECTION" in correction

    def test_approved_result_returns_no_correction_needed(self):
        result = _make_result()  # all pass
        correction = self.node.generate_correction_feedback(
            original_objection="Test objection",
            failed_draft="Test draft",
            verification_result=result,
        )
        assert "No corrections needed" in correction

    def test_correction_includes_original_objection(self):
        result = _make_result(
            relevance_pass=False,
            relevance_issues=[
                RelevanceIssue(
                    objection_intent="Price comparison",
                    response_coverage=0.2,
                    severity=IssueSeverity.MAJOR,
                    explanation="Low coverage",
                )
            ],
        )
        feedback_report = self.agent.generate_structured_feedback(result)
        objection = "Compare iPhone vs Samsung price"
        correction = self.node.generate_correction_feedback(
            original_objection=objection,
            failed_draft="Some draft...",
            verification_result=result,
            feedback_report=feedback_report,
        )
        assert objection in correction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
