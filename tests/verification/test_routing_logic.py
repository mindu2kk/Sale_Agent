"""
Unit tests for WorkflowRouter routing logic (Task 3.2.5)

Tests cover:
- route_after_verification: approved / correction / escalation paths
- route_after_correction: retry / escalation paths
- Immediate escalation on critical issues, fabricated policies, too many issues
- Retry-limit escalation
- Complex issue pattern detection (_is_complex_issue_pattern)
- Routing decision summary and metrics helpers

Requirements validated: 2.3, 2.4, 3.3, 3.4
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from backend.verification.workflow.routing import WorkflowRouter
from backend.verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
)
from backend.verification.config import VerificationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**kwargs) -> VerificationConfig:
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
        critical_issue_escalation=True,
    )
    defaults.update(kwargs)
    return VerificationConfig(**defaults)


def make_router(**config_kwargs) -> WorkflowRouter:
    return WorkflowRouter(config=make_config(**config_kwargs))


def make_verification_result(
    price_pass: bool = True,
    policy_pass: bool = True,
    relevance_pass: bool = True,
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
        verification_reasoning="Test reasoning for routing tests",
        execution_time_seconds=0.5,
        llm_tokens_used=100,
    )


def make_state(
    retry_count: int = 0,
    max_retries: int = 3,
    verification_result=None,
    workflow_status: str = "verifying",
) -> dict:
    return {
        "objection_text": "Test objection",
        "draft_response": "Test draft",
        "tools_used": [],
        "research_reasoning": "",
        "research_sources": [],
        "verification_result": verification_result,
        "correction_feedback": None,
        "retry_count": retry_count,
        "max_retries": max_retries,
        "final_response": "",
        "workflow_status": workflow_status,
        "execution_log": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "resource_usage": {},
        "error_log": [],
        "config": {},
        "workflow_id": "wf_test_001",
        "correlation_id": "corr_test_001",
    }


# ---------------------------------------------------------------------------
# route_after_verification — APPROVED path
# ---------------------------------------------------------------------------

class TestRouteAfterVerificationApproved:

    def test_approved_when_all_criteria_pass(self):
        router = make_router()
        result = make_verification_result(price_pass=True, policy_pass=True, relevance_pass=True)
        state = make_state(verification_result=result)

        assert router.route_after_verification(state) == "approved"

    def test_approved_regardless_of_retry_count(self):
        """Even if retries were used, approval is returned when result passes."""
        router = make_router()
        result = make_verification_result()
        state = make_state(retry_count=2, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "approved"


# ---------------------------------------------------------------------------
# route_after_verification — CORRECTION path
# ---------------------------------------------------------------------------

class TestRouteAfterVerificationCorrection:

    def test_correction_when_price_fails_no_critical_issues(self):
        router = make_router()
        result = make_verification_result(
            price_pass=False,
            price_issues=[PriceIssue(
                product_name="iPhone 15",
                mentioned_price="30,000,000 VND",
                actual_price="29,990,000 VND",
                deviation_percent=0.03,
                severity=IssueSeverity.MINOR,
                explanation="Minor price deviation",
            )],
        )
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "correction"

    def test_correction_when_policy_fails_non_critical(self):
        router = make_router()
        result = make_verification_result(
            policy_pass=False,
            policy_issues=[PolicyIssue(
                mentioned_policy="Bảo hành 1 năm",
                policy_type="warranty",
                is_fabricated=False,
                is_inaccurate=True,
                severity=IssueSeverity.MAJOR,
                explanation="Inaccurate warranty term",
            )],
        )
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "correction"

    def test_correction_when_relevance_fails_non_critical(self):
        router = make_router()
        result = make_verification_result(
            relevance_pass=False,
            relevance_issues=[RelevanceIssue(
                objection_intent="Price comparison",
                response_coverage=0.5,
                severity=IssueSeverity.MAJOR,
                explanation="Partial coverage",
            )],
        )
        state = make_state(retry_count=1, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "correction"

    def test_correction_when_multiple_minor_issues_below_threshold(self):
        """4 minor issues (< 5 total threshold) → correction, not escalation."""
        router = make_router()
        result = make_verification_result(
            price_pass=False,
            policy_pass=False,
            price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="m1"),
                PriceIssue(product_name="B", severity=IssueSeverity.MINOR, explanation="m2"),
            ],
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="p1", policy_type="return",
                    is_fabricated=False, is_inaccurate=True,
                    severity=IssueSeverity.MINOR, explanation="m3",
                ),
                PolicyIssue(
                    mentioned_policy="p2", policy_type="warranty",
                    is_fabricated=False, is_inaccurate=True,
                    severity=IssueSeverity.MINOR, explanation="m4",
                ),
            ],
        )
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "correction"


# ---------------------------------------------------------------------------
# route_after_verification — ESCALATION path
# ---------------------------------------------------------------------------

class TestRouteAfterVerificationEscalation:

    def test_escalation_when_no_verification_result(self):
        router = make_router()
        state = make_state(verification_result=None)

        assert router.route_after_verification(state) == "escalation"

    def test_escalation_when_retry_count_equals_max_retries(self):
        router = make_router()
        result = make_verification_result(
            price_pass=False,
            price_issues=[PriceIssue(
                product_name="A", severity=IssueSeverity.MINOR, explanation="x"
            )],
        )
        state = make_state(retry_count=3, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "escalation"

    def test_escalation_when_retry_count_exceeds_max_retries(self):
        router = make_router()
        result = make_verification_result(price_pass=False, price_issues=[
            PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="x")
        ])
        state = make_state(retry_count=5, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "escalation"

    def test_escalation_on_critical_issue_when_escalation_enabled(self):
        router = make_router(critical_issue_escalation=True)
        result = make_verification_result(
            price_pass=False,
            price_issues=[PriceIssue(
                product_name="iPhone 15",
                mentioned_price="50,000,000 VND",
                actual_price="30,000,000 VND",
                deviation_percent=66.7,
                severity=IssueSeverity.CRITICAL,
                explanation="Critical price deviation",
            )],
        )
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "escalation"

    def test_no_immediate_escalation_on_critical_when_disabled(self):
        """When critical_issue_escalation=False, critical issues go to correction."""
        router = make_router(critical_issue_escalation=False)
        result = make_verification_result(
            price_pass=False,
            price_issues=[PriceIssue(
                product_name="A",
                deviation_percent=66.7,
                severity=IssueSeverity.CRITICAL,
                explanation="Critical",
            )],
        )
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "correction"

    def test_escalation_on_fabricated_policy_critical(self):
        """Fabricated critical policy → immediate escalation (compliance risk)."""
        router = make_router()
        result = make_verification_result(
            policy_pass=False,
            policy_issues=[PolicyIssue(
                mentioned_policy="Bảo hành 5 năm miễn phí",
                policy_type="warranty",
                is_fabricated=True,
                is_inaccurate=False,
                severity=IssueSeverity.CRITICAL,
                explanation="Fabricated warranty policy",
            )],
        )
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "escalation"

    def test_escalation_when_total_issues_reach_five(self):
        """5 or more total issues → systemic problem → escalation."""
        router = make_router(critical_issue_escalation=False)  # disable critical escalation
        result = make_verification_result(
            price_pass=False,
            policy_pass=False,
            relevance_pass=False,
            price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="1"),
                PriceIssue(product_name="B", severity=IssueSeverity.MINOR, explanation="2"),
            ],
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="p1", policy_type="return",
                    is_fabricated=False, is_inaccurate=True,
                    severity=IssueSeverity.MINOR, explanation="3",
                ),
                PolicyIssue(
                    mentioned_policy="p2", policy_type="warranty",
                    is_fabricated=False, is_inaccurate=True,
                    severity=IssueSeverity.MINOR, explanation="4",
                ),
            ],
            relevance_issues=[
                RelevanceIssue(
                    objection_intent="z", response_coverage=0.5,
                    severity=IssueSeverity.MINOR, explanation="5",
                ),
            ],
        )
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "escalation"

    def test_escalation_on_two_critical_price_issues(self):
        """≥2 critical price issues → high business risk → escalation."""
        router = make_router()
        result = make_verification_result(
            price_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="A", deviation_percent=66.7,
                    severity=IssueSeverity.CRITICAL, explanation="c1",
                ),
                PriceIssue(
                    product_name="B", deviation_percent=80.0,
                    severity=IssueSeverity.CRITICAL, explanation="c2",
                ),
            ],
        )
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_verification(state) == "escalation"


# ---------------------------------------------------------------------------
# route_after_correction — RETRY path
# ---------------------------------------------------------------------------

class TestRouteAfterCorrectionRetry:

    def test_retry_when_below_max_retries_no_critical_issues(self):
        router = make_router()
        result = make_verification_result(
            price_pass=False,
            price_issues=[PriceIssue(
                product_name="A", severity=IssueSeverity.MINOR, explanation="x"
            )],
        )
        state = make_state(retry_count=1, max_retries=3, verification_result=result)

        assert router.route_after_correction(state) == "retry"

    def test_retry_when_no_verification_result_and_below_limit(self):
        router = make_router()
        state = make_state(retry_count=0, max_retries=3, verification_result=None)

        assert router.route_after_correction(state) == "retry"

    def test_retry_on_first_attempt(self):
        router = make_router()
        result = make_verification_result(policy_pass=False, policy_issues=[
            PolicyIssue(
                mentioned_policy="p", policy_type="return",
                is_fabricated=False, is_inaccurate=True,
                severity=IssueSeverity.MAJOR, explanation="y",
            )
        ])
        state = make_state(retry_count=0, max_retries=3, verification_result=result)

        assert router.route_after_correction(state) == "retry"


# ---------------------------------------------------------------------------
# route_after_correction — ESCALATION path
# ---------------------------------------------------------------------------

class TestRouteAfterCorrectionEscalation:

    def test_escalation_when_retry_count_equals_max(self):
        router = make_router()
        result = make_verification_result(price_pass=False, price_issues=[
            PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="x")
        ])
        state = make_state(retry_count=3, max_retries=3, verification_result=result)

        assert router.route_after_correction(state) == "escalation"

    def test_escalation_when_retry_count_exceeds_max(self):
        router = make_router()
        state = make_state(retry_count=5, max_retries=3, verification_result=None)

        assert router.route_after_correction(state) == "escalation"

    def test_escalation_after_multiple_retries_with_critical_issues(self):
        """retry_count ≥ 2 AND critical issues → escalate after correction."""
        router = make_router()
        result = make_verification_result(
            price_pass=False,
            price_issues=[PriceIssue(
                product_name="A", deviation_percent=66.7,
                severity=IssueSeverity.CRITICAL, explanation="critical",
            )],
        )
        state = make_state(retry_count=2, max_retries=3, verification_result=result)

        assert router.route_after_correction(state) == "escalation"

    def test_escalation_on_complex_issue_pattern_all_criteria_failed(self):
        """All 3 criteria failed → complex pattern → escalate after correction."""
        router = make_router()
        result = make_verification_result(
            price_pass=False,
            policy_pass=False,
            relevance_pass=False,
            price_issues=[PriceIssue(product_name="A", severity=IssueSeverity.MAJOR, explanation="p")],
            policy_issues=[PolicyIssue(
                mentioned_policy="q", policy_type="warranty",
                is_fabricated=False, is_inaccurate=True,
                severity=IssueSeverity.MAJOR, explanation="r",
            )],
            relevance_issues=[RelevanceIssue(
                objection_intent="s", response_coverage=0.5,
                severity=IssueSeverity.MAJOR, explanation="t",
            )],
        )
        state = make_state(retry_count=1, max_retries=3, verification_result=result)

        assert router.route_after_correction(state) == "escalation"


# ---------------------------------------------------------------------------
# Complex issue pattern detection
# ---------------------------------------------------------------------------

class TestComplexIssuePatternDetection:

    def setup_method(self):
        self.router = make_router()

    def _make_result(self, **kwargs) -> VerificationResult:
        return make_verification_result(**kwargs)

    def test_complex_pattern_when_all_three_criteria_fail(self):
        result = self._make_result(
            price_pass=False,
            policy_pass=False,
            relevance_pass=False,
            price_issues=[PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="x")],
            policy_issues=[PolicyIssue(
                mentioned_policy="p", policy_type="return",
                is_fabricated=False, is_inaccurate=True,
                severity=IssueSeverity.MINOR, explanation="y",
            )],
            relevance_issues=[RelevanceIssue(
                objection_intent="z", response_coverage=0.5,
                severity=IssueSeverity.MINOR, explanation="w",
            )],
        )
        assert self.router._is_complex_issue_pattern(result) is True

    def test_not_complex_when_only_one_criterion_fails(self):
        result = self._make_result(
            price_pass=False,
            price_issues=[PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="x")],
        )
        assert self.router._is_complex_issue_pattern(result) is False

    def test_complex_pattern_on_high_price_deviation(self):
        """≥2 price issues with >50% deviation → complex pattern."""
        result = self._make_result(
            price_pass=False,
            price_issues=[
                PriceIssue(product_name="A", deviation_percent=60.0, severity=IssueSeverity.CRITICAL, explanation="c1"),
                PriceIssue(product_name="B", deviation_percent=55.0, severity=IssueSeverity.CRITICAL, explanation="c2"),
            ],
        )
        assert self.router._is_complex_issue_pattern(result) is True

    def test_complex_pattern_on_multiple_fabricated_policies(self):
        """≥2 fabricated policies → complex pattern."""
        result = self._make_result(
            policy_pass=False,
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="p1", policy_type="warranty",
                    is_fabricated=True, is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL, explanation="fab1",
                ),
                PolicyIssue(
                    mentioned_policy="p2", policy_type="return",
                    is_fabricated=True, is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL, explanation="fab2",
                ),
            ],
        )
        assert self.router._is_complex_issue_pattern(result) is True

    def test_complex_pattern_on_very_low_relevance_coverage(self):
        """Relevance coverage < 30% → complex pattern."""
        result = self._make_result(
            relevance_pass=False,
            relevance_issues=[RelevanceIssue(
                objection_intent="price comparison",
                response_coverage=0.2,
                severity=IssueSeverity.CRITICAL,
                explanation="Severely off-topic",
            )],
        )
        assert self.router._is_complex_issue_pattern(result) is True

    def test_complex_pattern_on_mixed_critical_across_criteria(self):
        """Critical price + critical policy + major relevance → complex pattern."""
        result = self._make_result(
            price_pass=False,
            policy_pass=False,
            relevance_pass=False,
            price_issues=[PriceIssue(
                product_name="A", deviation_percent=66.7,
                severity=IssueSeverity.CRITICAL, explanation="c1",
            )],
            policy_issues=[PolicyIssue(
                mentioned_policy="p", policy_type="warranty",
                is_fabricated=True, is_inaccurate=False,
                severity=IssueSeverity.CRITICAL, explanation="c2",
            )],
            relevance_issues=[RelevanceIssue(
                objection_intent="z", response_coverage=0.4,
                severity=IssueSeverity.MAJOR, explanation="m1",
            )],
        )
        assert self.router._is_complex_issue_pattern(result) is True

    def test_not_complex_when_only_minor_issues_single_criterion(self):
        result = self._make_result(
            price_pass=False,
            price_issues=[
                PriceIssue(product_name="A", deviation_percent=2.0, severity=IssueSeverity.MINOR, explanation="m1"),
            ],
        )
        assert self.router._is_complex_issue_pattern(result) is False


# ---------------------------------------------------------------------------
# Total issue counting
# ---------------------------------------------------------------------------

class TestCountTotalIssues:

    def setup_method(self):
        self.router = make_router()

    def test_zero_issues_when_all_pass(self):
        result = make_verification_result()
        assert self.router._count_total_issues(result) == 0

    def test_counts_across_all_three_categories(self):
        result = make_verification_result(
            price_pass=False,
            policy_pass=False,
            relevance_pass=False,
            price_issues=[
                PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="1"),
                PriceIssue(product_name="B", severity=IssueSeverity.MINOR, explanation="2"),
            ],
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="p", policy_type="return",
                    is_fabricated=False, is_inaccurate=True,
                    severity=IssueSeverity.MINOR, explanation="3",
                ),
            ],
            relevance_issues=[
                RelevanceIssue(objection_intent="z", response_coverage=0.5, severity=IssueSeverity.MINOR, explanation="4"),
                RelevanceIssue(objection_intent="z", response_coverage=0.6, severity=IssueSeverity.MINOR, explanation="5"),
            ],
        )
        assert self.router._count_total_issues(result) == 5


# ---------------------------------------------------------------------------
# Routing decision summary
# ---------------------------------------------------------------------------

class TestRoutingDecisionSummary:

    def setup_method(self):
        self.router = make_router()

    def test_summary_approved(self):
        state = make_state(verification_result=make_verification_result())
        summary = self.router.get_routing_decision_summary(state, "approved")
        assert "approved" in summary.lower() or "✅" in summary

    def test_summary_escalation_includes_reason(self):
        result = make_verification_result(
            price_pass=False,
            price_issues=[PriceIssue(
                product_name="A", deviation_percent=66.7,
                severity=IssueSeverity.CRITICAL, explanation="c1",
            )],
        )
        state = make_state(retry_count=3, max_retries=3, verification_result=result)
        summary = self.router.get_routing_decision_summary(state, "escalation")
        assert "escalat" in summary.lower() or "🚨" in summary

    def test_summary_correction_includes_retry_info(self):
        result = make_verification_result(price_pass=False, price_issues=[
            PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="x")
        ])
        state = make_state(retry_count=0, max_retries=3, verification_result=result)
        summary = self.router.get_routing_decision_summary(state, "correction")
        assert "correction" in summary.lower() or "🔄" in summary

    def test_summary_retry(self):
        state = make_state(retry_count=1, max_retries=3)
        summary = self.router.get_routing_decision_summary(state, "retry")
        assert "retry" in summary.lower() or "🔄" in summary


# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------

class TestRoutingMetrics:

    def setup_method(self):
        self.router = make_router()

    def test_metrics_include_retry_info(self):
        result = make_verification_result()
        state = make_state(retry_count=1, max_retries=3, verification_result=result)
        metrics = self.router.get_routing_metrics(state)

        assert metrics["retry_count"] == 1
        assert metrics["max_retries"] == 3

    def test_metrics_include_verification_flags_when_result_present(self):
        result = make_verification_result(
            price_pass=False,
            price_issues=[PriceIssue(
                product_name="A", severity=IssueSeverity.CRITICAL, explanation="c"
            )],
        )
        state = make_state(verification_result=result)
        metrics = self.router.get_routing_metrics(state)

        assert metrics["verification_passed"] is False
        assert metrics["price_accuracy_pass"] is False
        assert metrics["policy_authenticity_pass"] is True
        assert metrics["topic_relevance_pass"] is True
        assert metrics["critical_issues_count"] == 1

    def test_metrics_without_verification_result(self):
        state = make_state(verification_result=None)
        metrics = self.router.get_routing_metrics(state)

        assert "retry_count" in metrics
        assert "verification_passed" not in metrics

    def test_metrics_complex_pattern_flag(self):
        result = make_verification_result(
            price_pass=False,
            policy_pass=False,
            relevance_pass=False,
            price_issues=[PriceIssue(product_name="A", severity=IssueSeverity.MINOR, explanation="x")],
            policy_issues=[PolicyIssue(
                mentioned_policy="p", policy_type="return",
                is_fabricated=False, is_inaccurate=True,
                severity=IssueSeverity.MINOR, explanation="y",
            )],
            relevance_issues=[RelevanceIssue(
                objection_intent="z", response_coverage=0.5,
                severity=IssueSeverity.MINOR, explanation="w",
            )],
        )
        state = make_state(verification_result=result)
        metrics = self.router.get_routing_metrics(state)

        assert metrics["complex_pattern_detected"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
