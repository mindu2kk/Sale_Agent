"""
Tests for execute_verification_node() - Task 3.1.2

Covers:
- Parallel verification via VerificationAgent.verify_draft()
- Early termination on critical issues
- State updates (verification_result, workflow_status, final_response)
- ExecutionStep logging with correct metrics
- Error handling via _handle_node_error()
- Async event loop handling (running loop vs new loop)
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from backend.verification.workflow.workflow import VerificationWorkflow
from backend.verification.config.config import VerificationConfig
from backend.verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
)
from backend.verification.models.execution import ExecutionStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> VerificationConfig:
    defaults = dict(
        price_tolerance_percent=1.0,
        price_critical_threshold=30.0,
        max_retries=3,
        parallel_verification=True,
        early_termination=True,
        async_timeout_seconds=10,
        enable_caching=False,
        llm_model_name="gpt-4",
    )
    defaults.update(overrides)
    return VerificationConfig(**defaults)


def _make_criteria(
    price_pass=True,
    policy_pass=True,
    relevance_pass=True,
    price_issues=None,
    policy_issues=None,
    relevance_issues=None,
) -> RubricCriteria:
    return RubricCriteria(
        price_accuracy_pass=price_pass,
        policy_authenticity_pass=policy_pass,
        topic_relevance_pass=relevance_pass,
        price_issues=price_issues or [],
        policy_issues=policy_issues or [],
        relevance_issues=relevance_issues or [],
    )


def _make_verification_result(criteria: RubricCriteria, reasoning="Verification passed successfully.") -> VerificationResult:
    return VerificationResult(
        criteria=criteria,
        verification_reasoning=reasoning,
        execution_time_seconds=0.5,
        llm_tokens_used=100,
    )


def _make_state(**overrides) -> dict:
    base = {
        "objection_text": "San pham co bao hanh khong?",
        "draft_response": "San pham duoc bao hanh 12 thang theo chinh sach chinh hang.",
        "tools_used": [],
        "research_reasoning": "",
        "research_sources": [],
        "verification_result": None,
        "correction_feedback": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_response": "",
        "workflow_status": "researching",
        "execution_log": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "config": {},
        "customer_context": {},
        "resource_usage": {},
        "error_log": [],
        "workflow_id": "wf_test_001",
        "correlation_id": "corr_test_001",
    }
    base.update(overrides)
    return base


def _make_workflow(config=None, verification_result=None):
    config = config or _make_config()
    research_agent = MagicMock()
    verification_agent = MagicMock()

    if verification_result is None:
        verification_result = _make_verification_result(_make_criteria())

    verification_agent.verify_draft = AsyncMock(return_value=verification_result)

    workflow = VerificationWorkflow(
        research_agent=research_agent,
        verification_agent=verification_agent,
        config=config,
    )
    return workflow, verification_agent



# ---------------------------------------------------------------------------
# 1. Happy path — verification passes
# ---------------------------------------------------------------------------

class TestVerificationNodeHappyPath:

    def test_approved_sets_workflow_status(self):
        """When verification passes, workflow_status becomes 'approved'."""
        workflow, _ = _make_workflow()
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["workflow_status"] == "approved"

    def test_approved_sets_final_response(self):
        """When approved, final_response is set to draft_response."""
        workflow, _ = _make_workflow()
        draft = "San pham duoc bao hanh 12 thang."
        state = _make_state(draft_response=draft)
        result = workflow._execute_verification_node(state)
        assert result["final_response"] == draft

    def test_verification_result_stored_in_state(self):
        """VerificationResult is stored in state['verification_result']."""
        vr = _make_verification_result(_make_criteria())
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["verification_result"] is vr

    def test_execution_step_appended(self):
        """An ExecutionStep is appended to execution_log."""
        workflow, _ = _make_workflow()
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert len(result["execution_log"]) == 1
        step = result["execution_log"][0]
        assert step.node_name == "verification"
        assert step.status == ExecutionStatus.SUCCESS

    def test_execution_step_metrics_overall_pass(self):
        """ExecutionStep metrics reflect overall_pass=True."""
        workflow, _ = _make_workflow()
        state = _make_state()
        result = workflow._execute_verification_node(state)
        metrics = result["execution_log"][0].metrics
        assert metrics["overall_pass"] is True
        assert metrics["price_accuracy_pass"] is True
        assert metrics["policy_authenticity_pass"] is True
        assert metrics["topic_relevance_pass"] is True

    def test_execution_step_has_non_negative_execution_time(self):
        """ExecutionStep records a non-negative execution_time."""
        workflow, _ = _make_workflow()
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].execution_time >= 0.0


# ---------------------------------------------------------------------------
# 2. Verification fails — correction needed
# ---------------------------------------------------------------------------

class TestVerificationNodeFailure:

    def test_failed_sets_correction_needed_status(self):
        """When verification fails, workflow_status becomes 'correction_needed'."""
        criteria = _make_criteria(price_pass=False)
        vr = _make_verification_result(criteria, reasoning="Price mismatch detected")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["workflow_status"] == "correction_needed"

    def test_failed_does_not_set_final_response(self):
        """When verification fails, final_response is NOT set to draft."""
        criteria = _make_criteria(policy_pass=False)
        vr = _make_verification_result(criteria, reasoning="Policy fabricated")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state(final_response="")
        result = workflow._execute_verification_node(state)
        assert result["final_response"] == ""

    def test_failed_execution_step_metrics_overall_pass_false(self):
        """ExecutionStep metrics reflect overall_pass=False when failed."""
        criteria = _make_criteria(price_pass=False, policy_pass=False)
        vr = _make_verification_result(criteria, reasoning="Multiple failures")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        metrics = result["execution_log"][0].metrics
        assert metrics["overall_pass"] is False
        assert metrics["price_accuracy_pass"] is False
        assert metrics["policy_authenticity_pass"] is False

    def test_failed_execution_step_status_is_success(self):
        """Even when verification fails, the node itself succeeded (no exception)."""
        criteria = _make_criteria(relevance_pass=False)
        vr = _make_verification_result(criteria, reasoning="Off-topic response")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].status == ExecutionStatus.SUCCESS



# ---------------------------------------------------------------------------
# 3. Early termination — critical issues
# ---------------------------------------------------------------------------

class TestVerificationNodeEarlyTermination:

    def test_critical_price_issue_tracked_in_metrics(self):
        """Critical price issue is reflected in execution_log metrics."""
        critical_price = PriceIssue(
            product_name="iPhone 15",
            mentioned_price="10,000,000 VND",
            actual_price="29,990,000 VND",
            deviation_percent=66.7,
            severity=IssueSeverity.CRITICAL,
            explanation="Price deviation 66.7% exceeds critical threshold",
        )
        criteria = _make_criteria(price_pass=False, price_issues=[critical_price])
        vr = _make_verification_result(criteria, reasoning="Critical price deviation")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].metrics["critical_issues"] == 1

    def test_early_termination_flag_in_metrics_when_critical(self):
        """early_termination_triggered is True when critical issues exist and config enables it."""
        critical_policy = PolicyIssue(
            mentioned_policy="Bao hanh vinh vien",
            policy_type="warranty",
            is_fabricated=True,
            is_inaccurate=False,
            severity=IssueSeverity.CRITICAL,
            explanation="Fabricated warranty policy",
        )
        criteria = _make_criteria(policy_pass=False, policy_issues=[critical_policy])
        vr = _make_verification_result(criteria, reasoning="Fabricated policy detected")
        config = _make_config(early_termination=True)
        workflow, _ = _make_workflow(config=config, verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].metrics["early_termination_triggered"] is True

    def test_no_early_termination_flag_when_disabled(self):
        """early_termination_triggered is False when config disables early termination."""
        critical_price = PriceIssue(
            product_name="Samsung S24",
            severity=IssueSeverity.CRITICAL,
            explanation="Critical price issue",
        )
        criteria = _make_criteria(price_pass=False, price_issues=[critical_price])
        vr = _make_verification_result(criteria, reasoning="Critical price")
        config = _make_config(early_termination=False)
        workflow, _ = _make_workflow(config=config, verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].metrics["early_termination_triggered"] is False

    def test_no_early_termination_flag_when_no_critical_issues(self):
        """early_termination_triggered is False when there are no critical issues."""
        minor_issue = PriceIssue(
            product_name="iPhone 15",
            deviation_percent=0.5,
            severity=IssueSeverity.MINOR,
            explanation="Minor price deviation",
        )
        criteria = _make_criteria(price_pass=False, price_issues=[minor_issue])
        vr = _make_verification_result(criteria, reasoning="Minor price issue")
        config = _make_config(early_termination=True)
        workflow, _ = _make_workflow(config=config, verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].metrics["early_termination_triggered"] is False


# ---------------------------------------------------------------------------
# 4. Parallel verification — verify_draft() is called once
# ---------------------------------------------------------------------------

class TestVerificationNodeParallelExecution:

    def test_verify_draft_called_exactly_once(self):
        """verify_draft() is called exactly once per node execution."""
        workflow, verification_agent = _make_workflow()
        state = _make_state()
        workflow._execute_verification_node(state)
        verification_agent.verify_draft.assert_called_once()

    def test_verify_draft_receives_full_state(self):
        """verify_draft() receives the full workflow state dict."""
        workflow, verification_agent = _make_workflow()
        state = _make_state(
            draft_response="Test draft",
            objection_text="Test objection",
        )
        workflow._execute_verification_node(state)
        call_args = verification_agent.verify_draft.call_args
        passed_state = call_args[0][0]
        assert passed_state["draft_response"] == "Test draft"
        assert passed_state["objection_text"] == "Test objection"

    def test_workflow_status_set_to_verifying_during_execution(self):
        """workflow_status is set to 'verifying' before verify_draft() is called."""
        captured_status = []

        async def capture_status(state):
            captured_status.append(state["workflow_status"])
            return _make_verification_result(_make_criteria())

        workflow, verification_agent = _make_workflow()
        verification_agent.verify_draft = capture_status
        state = _make_state()
        workflow._execute_verification_node(state)
        assert "verifying" in captured_status



# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------

class TestVerificationNodeErrorHandling:

    def test_exception_sets_failed_status(self):
        """When verify_draft() raises, workflow_status becomes 'failed'."""
        workflow, verification_agent = _make_workflow()
        verification_agent.verify_draft = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["workflow_status"] == "failed"

    def test_exception_appends_failed_execution_step(self):
        """When verify_draft() raises, a FAILED ExecutionStep is logged."""
        workflow, verification_agent = _make_workflow()
        verification_agent.verify_draft = AsyncMock(side_effect=ConnectionError("DB down"))
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert len(result["execution_log"]) == 1
        step = result["execution_log"][0]
        assert step.status == ExecutionStatus.FAILED
        assert step.node_name == "verification"

    def test_exception_sets_final_response_with_error_context(self):
        """When verify_draft() raises, final_response contains error context."""
        workflow, verification_agent = _make_workflow()
        verification_agent.verify_draft = AsyncMock(side_effect=ValueError("Invalid state"))
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert len(result["final_response"]) > 0

    def test_exception_does_not_propagate(self):
        """Exceptions inside verify_draft() are caught and do not propagate."""
        workflow, verification_agent = _make_workflow()
        verification_agent.verify_draft = AsyncMock(side_effect=Exception("Unexpected error"))
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result is not None


# ---------------------------------------------------------------------------
# 6. Execution log output summary format
# ---------------------------------------------------------------------------

class TestVerificationNodeOutputSummary:

    def test_output_summary_contains_approved_when_passed(self):
        """output_summary contains 'APPROVED' when verification passes."""
        workflow, _ = _make_workflow()
        state = _make_state()
        result = workflow._execute_verification_node(state)
        summary = result["execution_log"][0].output_summary
        assert "APPROVED" in summary

    def test_output_summary_contains_failed_when_not_passed(self):
        """output_summary contains 'FAILED' when verification fails."""
        criteria = _make_criteria(price_pass=False)
        vr = _make_verification_result(criteria, reasoning="Price mismatch")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        summary = result["execution_log"][0].output_summary
        assert "FAILED" in summary

    def test_output_summary_within_200_chars(self):
        """output_summary is truncated to at most 200 characters."""
        workflow, _ = _make_workflow()
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert len(result["execution_log"][0].output_summary) <= 200

    def test_output_summary_contains_criterion_results(self):
        """output_summary includes pass/fail info for each criterion."""
        criteria = _make_criteria(price_pass=True, policy_pass=False, relevance_pass=True)
        vr = _make_verification_result(criteria, reasoning="Policy issue")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        summary = result["execution_log"][0].output_summary
        assert "price" in summary.lower()
        assert "policy" in summary.lower()

    def test_input_summary_truncated_for_long_draft(self):
        """input_summary is truncated when draft_response exceeds 100 chars."""
        long_draft = "A" * 200
        workflow, _ = _make_workflow()
        state = _make_state(draft_response=long_draft)
        result = workflow._execute_verification_node(state)
        input_summary = result["execution_log"][0].input_summary
        assert len(input_summary) <= 120


# ---------------------------------------------------------------------------
# 7. Metrics tracking
# ---------------------------------------------------------------------------

class TestVerificationNodeMetrics:

    def test_tokens_used_tracked_in_metrics(self):
        """llm tokens_used from VerificationResult is tracked in execution metrics."""
        vr = _make_verification_result(_make_criteria())
        vr.llm_tokens_used = 512
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].metrics["tokens_used"] == 512

    def test_retry_count_tracked_in_metrics(self):
        """retry_count from state is tracked in execution metrics."""
        workflow, _ = _make_workflow()
        state = _make_state(retry_count=2)
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].metrics["retry_count"] == 2

    def test_critical_issues_count_zero_when_only_minor(self):
        """critical_issues metric is 0 when only minor issues exist."""
        minor_issue = PriceIssue(
            product_name="iPhone 15",
            deviation_percent=0.5,
            severity=IssueSeverity.MINOR,
            explanation="Minor deviation",
        )
        criteria = _make_criteria(price_pass=False, price_issues=[minor_issue])
        vr = _make_verification_result(criteria, reasoning="Minor price issue")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].metrics["critical_issues"] == 0

    def test_multiple_critical_issues_counted_correctly(self):
        """Multiple critical issues across criteria are counted correctly."""
        critical_price = PriceIssue(
            product_name="iPhone 15",
            severity=IssueSeverity.CRITICAL,
            explanation="Critical price",
        )
        critical_policy = PolicyIssue(
            mentioned_policy="Bao hanh vinh vien",
            policy_type="warranty",
            is_fabricated=True,
            is_inaccurate=False,
            severity=IssueSeverity.CRITICAL,
            explanation="Fabricated policy",
        )
        criteria = _make_criteria(
            price_pass=False,
            policy_pass=False,
            price_issues=[critical_price],
            policy_issues=[critical_policy],
        )
        vr = _make_verification_result(criteria, reasoning="Multiple critical issues")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        assert result["execution_log"][0].metrics["critical_issues"] == 2

    def test_all_criteria_fail_tracked_in_metrics(self):
        """When all 3 criteria fail, all metric flags are False."""
        criteria = _make_criteria(price_pass=False, policy_pass=False, relevance_pass=False)
        vr = _make_verification_result(criteria, reasoning="All criteria failed")
        workflow, _ = _make_workflow(verification_result=vr)
        state = _make_state()
        result = workflow._execute_verification_node(state)
        metrics = result["execution_log"][0].metrics
        assert metrics["price_accuracy_pass"] is False
        assert metrics["policy_authenticity_pass"] is False
        assert metrics["topic_relevance_pass"] is False
