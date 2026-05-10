"""
Tests for Self-Correction Loop Mechanism (Task 3.4.5)

Covers:
- Price / Policy / Relevance issue correction scenarios
- Multiple issue types combined correction
- Retry count tracking and state preservation
- Max retries escalation
- Critical issue immediate escalation
- Correction prompt quality

Requirements validated: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from verification.workflow.correction import SelfCorrectionNode
from verification.workflow.routing import WorkflowRouter
from verification.workflow.workflow import VerificationWorkflow
from verification.config.config import VerificationConfig
from verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
)
from verification.models.execution import ExecutionStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> VerificationConfig:
    defaults = dict(
        price_tolerance_percent=1.0,
        price_critical_threshold=30.0,
        max_retries=3,
        parallel_verification=False,
        early_termination=False,
        enable_caching=False,
        async_timeout_seconds=10,
        critical_issue_escalation=True,
        llm_model_name="gpt-4",
    )
    defaults.update(overrides)
    return VerificationConfig(**defaults)


def _make_criteria(
    price_pass=True, policy_pass=True, relevance_pass=True,
    price_issues=None, policy_issues=None, relevance_issues=None,
) -> RubricCriteria:
    return RubricCriteria(
        price_accuracy_pass=price_pass,
        policy_authenticity_pass=policy_pass,
        topic_relevance_pass=relevance_pass,
        price_issues=price_issues or [],
        policy_issues=policy_issues or [],
        relevance_issues=relevance_issues or [],
    )


def _make_vr(criteria: RubricCriteria, reasoning: str = "Verification completed") -> VerificationResult:
    if len(reasoning) < 10:
        reasoning = reasoning.ljust(10)
    return VerificationResult(
        criteria=criteria,
        verification_reasoning=reasoning,
        execution_time_seconds=0.5,
        llm_tokens_used=100,
    )


def _make_state(**overrides) -> dict:
    base = {
        "objection_text": "iPhone quá đắt so với Samsung",
        "draft_response": "iPhone 15 Pro Max giá 35,000,000 VND với bảo hành 2 năm",
        "tools_used": [],
        "research_reasoning": "",
        "research_sources": [],
        "verification_result": None,
        "correction_feedback": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_response": "",
        "workflow_status": "verifying",
        "execution_log": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "config": {},
        "customer_context": {},
        "resource_usage": {},
        "error_log": [],
        "workflow_id": "wf_test",
        "correlation_id": "corr_test",
    }
    base.update(overrides)
    return base


def _make_price_issue(severity=IssueSeverity.MAJOR, deviation=15.0) -> PriceIssue:
    return PriceIssue(
        product_name="iPhone 15 Pro Max",
        product_sku="IP15PM-256",
        mentioned_price="35,000,000 VND",
        actual_price="29,990,000 VND",
        deviation_percent=deviation,
        severity=severity,
        explanation=f"Price deviation {deviation:.1f}% exceeds tolerance",
        correction_suggestion="Update price to 29,990,000 VND (SKU: IP15PM-256)",
    )


def _make_policy_issue(is_fabricated=False, severity=IssueSeverity.MAJOR) -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Bảo hành 2 năm toàn diện miễn phí",
        policy_type="warranty",
        is_fabricated=is_fabricated,
        is_inaccurate=not is_fabricated,
        severity=severity,
        explanation="Fabricated warranty policy" if is_fabricated else "Inaccurate warranty duration",
        correct_policy=None if is_fabricated else "Bảo hành 1 năm theo chính sách Apple",
        correction_suggestion="Remove fabricated policy" if is_fabricated else "Use official warranty policy",
    )


def _make_relevance_issue(coverage=0.4, severity=IssueSeverity.MAJOR) -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="So sánh giá iPhone vs Samsung",
        response_coverage=coverage,
        missing_aspects=["Camera comparison", "Performance benchmarks"],
        off_topic_content=["Apple history"],
        severity=severity,
        explanation=f"Response only covers {coverage:.0%} of objection",
        correction_suggestion="Address price comparison and feature comparison directly",
    )


def _make_correction_node(config=None) -> SelfCorrectionNode:
    return SelfCorrectionNode(config or _make_config())


def _make_router(config=None) -> WorkflowRouter:
    return WorkflowRouter(config or _make_config())


def _make_workflow(config=None) -> VerificationWorkflow:
    config = config or _make_config()
    ra = MagicMock()
    va = MagicMock()
    va.verify_draft = MagicMock(return_value=_make_vr(_make_criteria()))
    return VerificationWorkflow(research_agent=ra, verification_agent=va, config=config)


# ---------------------------------------------------------------------------
# 1. Price Issue Correction Scenarios
# ---------------------------------------------------------------------------

class TestPriceIssueCorrection:
    """Draft with wrong price triggers correction with PriceIssue feedback."""

    def test_price_issue_generates_correction_feedback(self):
        """Price failure produces non-empty correction feedback."""
        node = _make_correction_node()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "iPhone 15 giá 35 triệu", vr
        )

        assert feedback
        assert len(feedback) > 0

    def test_price_correction_prompt_contains_product_name(self):
        """Correction prompt references the product with the price issue."""
        node = _make_correction_node()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "iPhone 15 giá 35 triệu", vr
        )

        assert "iPhone 15 Pro Max" in feedback

    def test_price_correction_prompt_contains_price_section(self):
        """Correction prompt includes a price accuracy section."""
        node = _make_correction_node()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "iPhone 15 giá 35 triệu", vr
        )

        assert "PRICE" in feedback.upper() or "GIÁ" in feedback.upper()

    def test_price_correction_includes_original_objection(self):
        """Correction prompt preserves original objection context."""
        node = _make_correction_node()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        objection = "iPhone quá đắt so với Samsung"

        feedback = node.generate_correction_feedback(
            objection, "iPhone 15 giá 35 triệu", vr
        )

        assert objection in feedback

    def test_critical_price_issue_flagged_in_correction(self):
        """Critical price deviation is prominently flagged."""
        node = _make_correction_node()
        issue = _make_price_issue(severity=IssueSeverity.CRITICAL, deviation=66.7)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "iPhone 15 giá 35 triệu", vr
        )

        assert "CRITICAL" in feedback.upper() or "🚨" in feedback

    def test_price_correction_suggestion_included(self):
        """Correction prompt includes the correction suggestion from the issue."""
        node = _make_correction_node()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "iPhone 15 giá 35 triệu", vr
        )

        # Either the actual price or the SKU should appear
        assert "29,990,000" in feedback or "IP15PM-256" in feedback


# ---------------------------------------------------------------------------
# 2. Policy Issue Correction Scenarios
# ---------------------------------------------------------------------------

class TestPolicyIssueCorrection:
    """Draft with fabricated/inaccurate policy triggers correction with PolicyIssue feedback."""

    def test_policy_issue_generates_correction_feedback(self):
        """Policy failure produces non-empty correction feedback."""
        node = _make_correction_node()
        issue = _make_policy_issue()
        vr = _make_vr(_make_criteria(policy_pass=False, policy_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "Chính sách bảo hành thế nào?", "Bảo hành 2 năm miễn phí", vr
        )

        assert feedback
        assert len(feedback) > 0

    def test_fabricated_policy_flagged_in_correction(self):
        """Fabricated policy is explicitly flagged in correction prompt."""
        node = _make_correction_node()
        issue = _make_policy_issue(is_fabricated=True, severity=IssueSeverity.CRITICAL)
        vr = _make_vr(_make_criteria(policy_pass=False, policy_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "Chính sách bảo hành?", "Bảo hành 2 năm miễn phí", vr
        )

        assert "fabricat" in feedback.lower() or "FABRICATED" in feedback or "bịa" in feedback.lower()

    def test_policy_correction_contains_policy_section(self):
        """Correction prompt includes a policy authenticity section."""
        node = _make_correction_node()
        issue = _make_policy_issue()
        vr = _make_vr(_make_criteria(policy_pass=False, policy_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "Chính sách bảo hành?", "Bảo hành 2 năm miễn phí", vr
        )

        assert "POLICY" in feedback.upper() or "CHÍNH SÁCH" in feedback.upper()

    def test_policy_correction_references_correct_policy(self):
        """When correct_policy is available, it appears in the correction."""
        node = _make_correction_node()
        issue = _make_policy_issue(is_fabricated=False)  # inaccurate, has correct_policy
        vr = _make_vr(_make_criteria(policy_pass=False, policy_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "Chính sách bảo hành?", "Bảo hành 2 năm miễn phí", vr
        )

        assert "Bảo hành 1 năm" in feedback or "official" in feedback.lower()

    def test_critical_fabricated_policy_triggers_escalation_flag(self):
        """Critical fabricated policy should be flagged for escalation."""
        node = _make_correction_node()
        issue = _make_policy_issue(is_fabricated=True, severity=IssueSeverity.CRITICAL)
        vr = _make_vr(_make_criteria(policy_pass=False, policy_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "Chính sách bảo hành?", "Bảo hành 2 năm miễn phí", vr
        )

        # Should mention escalation or critical severity
        assert "ESCALAT" in feedback.upper() or "CRITICAL" in feedback.upper() or "🚨" in feedback


# ---------------------------------------------------------------------------
# 3. Relevance Issue Correction Scenarios
# ---------------------------------------------------------------------------

class TestRelevanceIssueCorrection:
    """Off-topic draft triggers correction with RelevanceIssue feedback."""

    def test_relevance_issue_generates_correction_feedback(self):
        """Relevance failure produces non-empty correction feedback."""
        node = _make_correction_node()
        issue = _make_relevance_issue()
        vr = _make_vr(_make_criteria(relevance_pass=False, relevance_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "So sánh giá iPhone vs Samsung", "Apple được thành lập năm 1976", vr
        )

        assert feedback
        assert len(feedback) > 0

    def test_relevance_correction_contains_missing_aspects(self):
        """Missing aspects from the issue appear in correction prompt."""
        node = _make_correction_node()
        issue = _make_relevance_issue()
        vr = _make_vr(_make_criteria(relevance_pass=False, relevance_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "So sánh giá iPhone vs Samsung", "Apple được thành lập năm 1976", vr
        )

        assert "Camera comparison" in feedback or "Performance benchmarks" in feedback

    def test_relevance_correction_contains_relevance_section(self):
        """Correction prompt includes a topic relevance section."""
        node = _make_correction_node()
        issue = _make_relevance_issue()
        vr = _make_vr(_make_criteria(relevance_pass=False, relevance_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "So sánh giá iPhone vs Samsung", "Apple được thành lập năm 1976", vr
        )

        assert "RELEVANCE" in feedback.upper() or "TOPIC" in feedback.upper()

    def test_relevance_correction_guides_toward_objection_intent(self):
        """Correction prompt references the objection intent."""
        node = _make_correction_node()
        issue = _make_relevance_issue()
        vr = _make_vr(_make_criteria(relevance_pass=False, relevance_issues=[issue]))
        objection = "So sánh giá iPhone vs Samsung"

        feedback = node.generate_correction_feedback(
            objection, "Apple được thành lập năm 1976", vr
        )

        assert objection in feedback

    def test_low_coverage_relevance_issue_prompts_expansion(self):
        """Very low coverage (<50%) triggers expansion instruction."""
        node = _make_correction_node()
        issue = _make_relevance_issue(coverage=0.2, severity=IssueSeverity.CRITICAL)
        vr = _make_vr(_make_criteria(relevance_pass=False, relevance_issues=[issue]))

        feedback = node.generate_correction_feedback(
            "So sánh giá iPhone vs Samsung", "Apple được thành lập năm 1976", vr
        )

        # Should mention expanding or addressing the objection
        assert "expand" in feedback.lower() or "address" in feedback.lower() or "objection" in feedback.lower()


# ---------------------------------------------------------------------------
# 4. Multiple Issue Types Correction
# ---------------------------------------------------------------------------

class TestMultipleIssueCorrection:
    """Draft failing multiple criteria generates combined correction feedback."""

    def test_all_three_issue_types_in_single_correction(self):
        """Combined correction addresses price, policy, and relevance issues."""
        node = _make_correction_node()
        criteria = _make_criteria(
            price_pass=False, policy_pass=False, relevance_pass=False,
            price_issues=[_make_price_issue()],
            policy_issues=[_make_policy_issue()],
            relevance_issues=[_make_relevance_issue()],
        )
        vr = _make_vr(criteria)

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "Draft with multiple issues", vr
        )

        # All three issue types should be mentioned
        assert "PRICE" in feedback.upper() or "GIÁ" in feedback.upper()
        assert "POLICY" in feedback.upper() or "CHÍNH SÁCH" in feedback.upper()
        assert "RELEVANCE" in feedback.upper() or "TOPIC" in feedback.upper()

    def test_combined_correction_is_non_empty(self):
        """Combined correction feedback is non-empty."""
        node = _make_correction_node()
        criteria = _make_criteria(
            price_pass=False, policy_pass=False, relevance_pass=False,
            price_issues=[_make_price_issue()],
            policy_issues=[_make_policy_issue()],
            relevance_issues=[_make_relevance_issue()],
        )
        vr = _make_vr(criteria)

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "Draft with multiple issues", vr
        )

        assert len(feedback) > 100  # Should be substantial

    def test_price_and_policy_issues_combined(self):
        """Price + policy issues both appear in correction."""
        node = _make_correction_node()
        criteria = _make_criteria(
            price_pass=False, policy_pass=False,
            price_issues=[_make_price_issue()],
            policy_issues=[_make_policy_issue()],
        )
        vr = _make_vr(criteria)

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "Draft with price and policy issues", vr
        )

        assert "iPhone 15 Pro Max" in feedback  # price issue product
        assert "warranty" in feedback.lower() or "bảo hành" in feedback.lower()  # policy issue

    def test_no_correction_when_approved(self):
        """Approved verification returns no-correction message."""
        node = _make_correction_node()
        vr = _make_vr(_make_criteria())  # all pass

        feedback = node.generate_correction_feedback(
            "iPhone quá đắt", "Good draft", vr
        )

        assert "No corrections needed" in feedback or "passed" in feedback.lower()


# ---------------------------------------------------------------------------
# 5. Retry Count Tracking
# ---------------------------------------------------------------------------

class TestRetryCountTracking:
    """retry_count increments correctly and state is preserved across retries."""

    def test_correction_node_increments_retry_count(self):
        """_execute_correction_node increments retry_count by 1."""
        wf = _make_workflow()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, verification_result=vr)

        result = wf._execute_correction_node(state)

        assert result["retry_count"] == 1

    def test_retry_count_increments_on_each_correction(self):
        """retry_count increments from 1 to 2 on second correction."""
        wf = _make_workflow()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=1, verification_result=vr)

        result = wf._execute_correction_node(state)

        assert result["retry_count"] == 2

    def test_original_objection_preserved_across_retries(self):
        """objection_text is unchanged after correction node execution."""
        wf = _make_workflow()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        original_objection = "iPhone quá đắt so với Samsung"
        state = _make_state(
            objection_text=original_objection,
            retry_count=1,
            verification_result=vr,
        )

        result = wf._execute_correction_node(state)

        assert result["objection_text"] == original_objection

    def test_correction_feedback_set_in_state(self):
        """correction_feedback is populated after correction node runs."""
        wf = _make_workflow()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, verification_result=vr)

        result = wf._execute_correction_node(state)

        assert result["correction_feedback"] is not None
        assert len(result["correction_feedback"]) > 0

    def test_execution_log_records_correction_step(self):
        """Correction node appends an ExecutionStep to execution_log."""
        wf = _make_workflow()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, verification_result=vr)

        result = wf._execute_correction_node(state)

        assert len(result["execution_log"]) == 1
        step = result["execution_log"][0]
        assert step.node_name == "correction"
        assert step.status == ExecutionStatus.SUCCESS

    def test_execution_log_records_retry_count_in_metrics(self):
        """Correction step metrics include retry_count."""
        wf = _make_workflow()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, verification_result=vr)

        result = wf._execute_correction_node(state)

        metrics = result["execution_log"][0].metrics
        assert "retry_count" in metrics
        assert metrics["retry_count"] == 1  # incremented

    def test_execution_log_records_max_retries_in_metrics(self):
        """Correction step metrics include max_retries."""
        wf = _make_workflow()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, max_retries=3, verification_result=vr)

        result = wf._execute_correction_node(state)

        metrics = result["execution_log"][0].metrics
        assert metrics["max_retries"] == 3


# ---------------------------------------------------------------------------
# 6. Max Retries Escalation
# ---------------------------------------------------------------------------

class TestMaxRetriesEscalation:
    """When retry_count >= max_retries, routing goes to escalation."""

    def test_route_to_escalation_when_retries_exhausted(self):
        """route_after_verification returns 'escalation' when retry_count >= max_retries."""
        router = _make_router()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(
            retry_count=3,
            max_retries=3,
            verification_result=vr,
        )

        decision = router.route_after_verification(state)

        assert decision == "escalation"

    def test_route_to_correction_when_retries_remaining(self):
        """route_after_verification returns 'correction' when retries remain (no critical issues)."""
        router = _make_router(_make_config(critical_issue_escalation=False))
        issue = _make_price_issue(severity=IssueSeverity.MAJOR)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(
            retry_count=1,
            max_retries=3,
            verification_result=vr,
        )

        decision = router.route_after_verification(state)

        assert decision == "correction"

    def test_route_after_correction_escalates_when_retries_exhausted(self):
        """route_after_correction returns 'escalation' when retry_count >= max_retries."""
        router = _make_router()
        state = _make_state(retry_count=3, max_retries=3)

        decision = router.route_after_correction(state)

        assert decision == "escalation"

    def test_route_after_correction_retries_when_retries_remaining(self):
        """route_after_correction returns 'retry' when retries remain."""
        router = _make_router()
        issue = _make_price_issue(severity=IssueSeverity.MINOR)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=1, max_retries=3, verification_result=vr)

        decision = router.route_after_correction(state)

        assert decision == "retry"

    def test_escalation_node_sets_escalated_status(self):
        """_execute_escalation_node sets workflow_status to 'escalated'."""
        wf = _make_workflow()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=3, max_retries=3, verification_result=vr)

        result = wf._execute_escalation_node(state)

        assert result["workflow_status"] == "escalated"

    def test_escalation_node_sets_final_response(self):
        """_execute_escalation_node sets final_response with escalation message."""
        wf = _make_workflow()
        state = _make_state(retry_count=3, max_retries=3)

        result = wf._execute_escalation_node(state)

        assert "ESCALATED" in result["final_response"].upper() or "HUMAN" in result["final_response"].upper()

    def test_escalation_node_appends_execution_step(self):
        """_execute_escalation_node appends an ExecutionStep to execution_log."""
        wf = _make_workflow()
        state = _make_state(retry_count=3, max_retries=3)

        result = wf._execute_escalation_node(state)

        assert len(result["execution_log"]) == 1
        assert result["execution_log"][0].node_name == "escalation"
        assert result["execution_log"][0].status == ExecutionStatus.SUCCESS

    def test_escalation_reason_includes_max_retries_exceeded(self):
        """Escalation reason mentions max retries exceeded."""
        wf = _make_workflow()
        state = _make_state(retry_count=3, max_retries=3)

        reason = wf._generate_escalation_reason(state)

        assert "retries" in reason.lower() or "retry" in reason.lower() or "3" in reason


# ---------------------------------------------------------------------------
# 7. Critical Issue Immediate Escalation
# ---------------------------------------------------------------------------

class TestCriticalIssueEscalation:
    """Critical severity issues trigger immediate escalation."""

    def test_critical_price_issue_routes_to_escalation(self):
        """Single critical price issue triggers immediate escalation."""
        router = _make_router(_make_config(critical_issue_escalation=True))
        issue = _make_price_issue(severity=IssueSeverity.CRITICAL, deviation=66.7)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, max_retries=3, verification_result=vr)

        decision = router.route_after_verification(state)

        assert decision == "escalation"

    def test_critical_fabricated_policy_routes_to_escalation(self):
        """Critical fabricated policy triggers immediate escalation."""
        router = _make_router(_make_config(critical_issue_escalation=True))
        issue = _make_policy_issue(is_fabricated=True, severity=IssueSeverity.CRITICAL)
        vr = _make_vr(_make_criteria(policy_pass=False, policy_issues=[issue]))
        state = _make_state(retry_count=0, max_retries=3, verification_result=vr)

        decision = router.route_after_verification(state)

        assert decision == "escalation"

    def test_critical_relevance_issue_routes_to_escalation(self):
        """Critical relevance issue triggers immediate escalation."""
        router = _make_router(_make_config(critical_issue_escalation=True))
        issue = _make_relevance_issue(coverage=0.1, severity=IssueSeverity.CRITICAL)
        vr = _make_vr(_make_criteria(relevance_pass=False, relevance_issues=[issue]))
        state = _make_state(retry_count=0, max_retries=3, verification_result=vr)

        decision = router.route_after_verification(state)

        assert decision == "escalation"

    def test_major_issue_does_not_trigger_immediate_escalation(self):
        """MAJOR issue (non-critical) routes to correction, not escalation."""
        router = _make_router(_make_config(critical_issue_escalation=True))
        issue = _make_price_issue(severity=IssueSeverity.MAJOR, deviation=15.0)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, max_retries=3, verification_result=vr)

        decision = router.route_after_verification(state)

        assert decision == "correction"

    def test_critical_escalation_disabled_routes_to_correction(self):
        """When critical_issue_escalation=False, critical issues route to correction."""
        router = _make_router(_make_config(critical_issue_escalation=False))
        issue = _make_price_issue(severity=IssueSeverity.CRITICAL, deviation=66.7)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, max_retries=3, verification_result=vr)

        decision = router.route_after_verification(state)

        # Without critical escalation, should go to correction (retries remain)
        assert decision == "correction"

    def test_should_escalate_immediately_with_critical_issue(self):
        """SelfCorrectionNode.should_escalate_immediately returns True for critical issues."""
        node = _make_correction_node(_make_config(critical_issue_escalation=True))
        issue = _make_price_issue(severity=IssueSeverity.CRITICAL, deviation=66.7)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))

        assert node.should_escalate_immediately(vr) is True

    def test_should_not_escalate_immediately_for_major_issues(self):
        """SelfCorrectionNode.should_escalate_immediately returns False for major issues."""
        node = _make_correction_node(_make_config(critical_issue_escalation=True))
        issue = _make_price_issue(severity=IssueSeverity.MAJOR, deviation=15.0)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))

        assert node.should_escalate_immediately(vr) is False

    def test_escalation_includes_critical_issues_count_in_reason(self):
        """Escalation reason mentions critical issues count."""
        wf = _make_workflow()
        issue = _make_price_issue(severity=IssueSeverity.CRITICAL, deviation=66.7)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=0, max_retries=3, verification_result=vr)

        reason = wf._generate_escalation_reason(state)

        assert "critical" in reason.lower() or "1" in reason


# ---------------------------------------------------------------------------
# 8. Correction Prompt Quality
# ---------------------------------------------------------------------------

class TestCorrectionPromptQuality:
    """Correction prompts are non-empty and contain relevant issue details."""

    def test_correction_prompt_is_non_empty_for_price_failure(self):
        """Price failure produces non-empty correction prompt."""
        node = _make_correction_node()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[_make_price_issue()]))

        feedback = node.generate_correction_feedback("objection", "draft", vr)

        assert feedback.strip() != ""

    def test_correction_prompt_is_non_empty_for_policy_failure(self):
        """Policy failure produces non-empty correction prompt."""
        node = _make_correction_node()
        vr = _make_vr(_make_criteria(policy_pass=False, policy_issues=[_make_policy_issue()]))

        feedback = node.generate_correction_feedback("objection", "draft", vr)

        assert feedback.strip() != ""

    def test_correction_prompt_is_non_empty_for_relevance_failure(self):
        """Relevance failure produces non-empty correction prompt."""
        node = _make_correction_node()
        vr = _make_vr(_make_criteria(relevance_pass=False, relevance_issues=[_make_relevance_issue()]))

        feedback = node.generate_correction_feedback("objection", "draft", vr)

        assert feedback.strip() != ""

    def test_price_and_policy_produce_different_corrections(self):
        """Price issue and policy issue produce different correction prompts."""
        node = _make_correction_node()

        price_vr = _make_vr(_make_criteria(price_pass=False, price_issues=[_make_price_issue()]))
        policy_vr = _make_vr(_make_criteria(policy_pass=False, policy_issues=[_make_policy_issue()]))

        price_feedback = node.generate_correction_feedback("objection", "draft", price_vr)
        policy_feedback = node.generate_correction_feedback("objection", "draft", policy_vr)

        assert price_feedback != policy_feedback

    def test_correction_contains_retry_instructions(self):
        """Correction prompt includes retry instructions for the Research Agent."""
        node = _make_correction_node()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[_make_price_issue()]))

        feedback = node.generate_correction_feedback("objection", "draft", vr)

        assert "RETRY" in feedback.upper() or "INSTRUCTION" in feedback.upper() or "CORRECT" in feedback.upper()

    def test_correction_contains_quality_checklist(self):
        """Correction prompt includes a quality checklist."""
        node = _make_correction_node()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[_make_price_issue()]))

        feedback = node.generate_correction_feedback("objection", "draft", vr)

        assert "CHECKLIST" in feedback.upper() or "□" in feedback or "verify" in feedback.lower()

    def test_correction_preserves_objection_context(self):
        """Correction prompt always includes the original objection."""
        node = _make_correction_node()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[_make_price_issue()]))
        objection = "Tại sao iPhone đắt hơn Samsung Galaxy?"

        feedback = node.generate_correction_feedback(objection, "draft", vr)

        assert objection in feedback

    def test_correction_header_indicates_failure(self):
        """Correction prompt header clearly indicates verification failed."""
        node = _make_correction_node()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[_make_price_issue()]))

        feedback = node.generate_correction_feedback("objection", "draft", vr)

        assert "FAILED" in feedback.upper() or "CORRECTION" in feedback.upper() or "REQUIRED" in feedback.upper()

    def test_issue_explanation_appears_in_correction(self):
        """The issue explanation text appears in the correction prompt."""
        node = _make_correction_node()
        issue = _make_price_issue()
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))

        feedback = node.generate_correction_feedback("objection", "draft", vr)

        # The explanation or product name should appear
        assert issue.product_name in feedback or "deviation" in feedback.lower()


# ---------------------------------------------------------------------------
# 9. Routing Decision Summary
# ---------------------------------------------------------------------------

class TestRoutingDecisionSummary:
    """WorkflowRouter.get_routing_decision_summary returns human-readable summaries."""

    def test_approved_summary(self):
        """Approved routing produces positive summary."""
        router = _make_router()
        state = _make_state(verification_result=_make_vr(_make_criteria()))

        summary = router.get_routing_decision_summary(state, "approved")

        assert "approved" in summary.lower() or "passed" in summary.lower()

    def test_escalation_summary_mentions_reason(self):
        """Escalation summary mentions the reason."""
        router = _make_router()
        issue = _make_price_issue(severity=IssueSeverity.CRITICAL, deviation=66.7)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=3, max_retries=3, verification_result=vr)

        summary = router.get_routing_decision_summary(state, "escalation")

        assert "escalat" in summary.lower() or "human" in summary.lower()

    def test_correction_summary_mentions_retry_number(self):
        """Correction summary mentions the retry attempt number."""
        router = _make_router()
        issue = _make_price_issue(severity=IssueSeverity.MAJOR)
        vr = _make_vr(_make_criteria(price_pass=False, price_issues=[issue]))
        state = _make_state(retry_count=1, max_retries=3, verification_result=vr)

        summary = router.get_routing_decision_summary(state, "correction")

        assert "correction" in summary.lower() or "retry" in summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
