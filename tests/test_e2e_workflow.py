"""
End-to-end integration tests for task 4.1.3.

Tests the complete verification workflow using realistic mocked Sales Research
Agent output and binary PASS/FAIL decisions.

Scenarios covered:
1. Happy path  – good draft → Verification PASS → workflow ends approved
2. Correction path – price issue → FAIL → self-correction → retry → PASS
3. Escalation path – repeated failures → max retries exceeded → escalation
4. Binary decision validation – decisions are strictly PASS/FAIL (not numeric)
5. State integrity – WorkflowState is properly maintained throughout

Requirements validated:
  Req 1  – Binary PASS/FAIL verification
  Req 2  – LangGraph StateGraph orchestration
  Req 3  – Self-correction loop
  Req 4  – Price accuracy verification
  Req 5  – Policy authenticity verification
  Req 6  – Topic relevance assessment
  Req 7  – Workflow state management & observability
  Req 8  – Error handling & resilience
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, patch, call

import pytest

from agent.sales_research_agent import AgentResult, SalesResearchAgent
from verification.agent.verification_agent import VerificationAgent
from verification.config.config import VerificationConfig
from verification.models.state import WorkflowState
from verification.models.verification import (
    IssueSeverity,
    PolicyIssue,
    PriceIssue,
    RubricCriteria,
    RelevanceIssue,
    VerificationResult,
)
from verification.workflow.correction import SelfCorrectionNode
from verification.workflow.routing import WorkflowRouter
from verification.workflow.workflow import VerificationWorkflow


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_config(max_retries: int = 3, critical_issue_escalation: bool = True) -> VerificationConfig:
    """Return a VerificationConfig suitable for tests (no real LLM/DB needed)."""
    return VerificationConfig(
        max_retries=max_retries,
        critical_issue_escalation=critical_issue_escalation,
        parallel_verification=False,   # sequential is simpler to mock
        early_termination=False,
        enable_caching=False,
        async_timeout_seconds=30,
        price_tolerance_percent=1.0,
        price_critical_threshold=30.0,
    )


def _pass_criteria() -> RubricCriteria:
    """All-pass RubricCriteria with no issues."""
    return RubricCriteria(
        price_accuracy_pass=True,
        policy_authenticity_pass=True,
        topic_relevance_pass=True,
        price_issues=[],
        policy_issues=[],
        relevance_issues=[],
    )


def _fail_price_criteria(severity: IssueSeverity = IssueSeverity.MAJOR) -> RubricCriteria:
    """Fail criteria with one price issue."""
    return RubricCriteria(
        price_accuracy_pass=False,
        policy_authenticity_pass=True,
        topic_relevance_pass=True,
        price_issues=[
            PriceIssue(
                product_name="iPhone 15 Pro Max",
                mentioned_price="35,000,000 VND",
                actual_price="29,990,000 VND",
                deviation_percent=16.7,
                severity=severity,
                explanation="Price deviation 16.7% exceeds tolerance",
                correction_suggestion="Update price to 29,990,000 VND",
            )
        ],
        policy_issues=[],
        relevance_issues=[],
    )


def _fail_policy_criteria(fabricated: bool = True) -> RubricCriteria:
    """Fail criteria with one fabricated policy issue (CRITICAL)."""
    return RubricCriteria(
        price_accuracy_pass=True,
        policy_authenticity_pass=False,
        topic_relevance_pass=True,
        price_issues=[],
        policy_issues=[
            PolicyIssue(
                mentioned_policy="Bảo hành 5 năm toàn bộ linh kiện",
                policy_type="warranty",
                is_fabricated=fabricated,
                is_inaccurate=not fabricated,
                severity=IssueSeverity.CRITICAL if fabricated else IssueSeverity.MAJOR,
                explanation="Fabricated warranty policy not found in official documents",
                correction_suggestion="Remove fabricated policy claim",
            )
        ],
        relevance_issues=[],
    )


def _fail_relevance_criteria() -> RubricCriteria:
    """Fail criteria with one relevance issue."""
    return RubricCriteria(
        price_accuracy_pass=True,
        policy_authenticity_pass=True,
        topic_relevance_pass=False,
        price_issues=[],
        policy_issues=[],
        relevance_issues=[
            RelevanceIssue(
                objection_intent="So sánh giá iPhone vs Samsung",
                response_coverage=0.4,
                missing_aspects=["camera comparison", "battery life"],
                severity=IssueSeverity.MAJOR,
                explanation="Response only covers 40% of objection",
            )
        ],
    )


def _make_verification_result(criteria: RubricCriteria, reasoning: str = "Test verification") -> VerificationResult:
    return VerificationResult(
        criteria=criteria,
        timestamp=datetime.now(),
        verification_reasoning=reasoning,
        execution_time_seconds=0.1,
        llm_tokens_used=100,
    )


def _make_agent_result(draft: str, tools: Optional[List[str]] = None) -> AgentResult:
    return AgentResult(
        objection_text="iPhone quá đắt so với Samsung",
        draft_response=draft,
        tools_used=tools or ["internal_db_search"],
    )


def _make_mock_research_agent(drafts: List[str]) -> MagicMock:
    """
    Return a mock research agent whose run() returns successive AgentResults.
    Each call pops the next draft from the list.
    """
    agent = MagicMock()
    results = [_make_agent_result(d) for d in drafts]
    agent.run.side_effect = results
    return agent


def _make_mock_verification_agent(results: List[VerificationResult]) -> MagicMock:
    """Return a mock VerificationAgent whose verify_draft() returns successive results.

    The workflow's _execute_verification_node calls verify_draft() as an async coroutine.
    We use AsyncMock so that each call returns the next result from the list.
    """
    from unittest.mock import AsyncMock

    agent = MagicMock(spec=VerificationAgent)
    # verify_draft_sync is the sync wrapper used in some paths
    agent.verify_draft_sync.side_effect = list(results)
    # verify_draft is the async method called by the workflow node
    agent.verify_draft = AsyncMock(side_effect=list(results))
    return agent


# ---------------------------------------------------------------------------
# Scenario 1: Happy path – PASS on first attempt
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Research Agent produces a good draft → Verification PASS → approved."""

    GOOD_DRAFT = (
        "Dạ, em hiểu anh/chị đang so sánh về giá. iPhone 15 Pro Max hiện có giá "
        "29,990,000 VND với chip A17 Pro, camera 48MP và bảo hành chính hãng 12 tháng. "
        "Đây là mức giá cạnh tranh cho cấu hình cao cấp này."
    )

    def test_workflow_ends_with_approved_status(self):
        """Happy path: workflow_status == 'approved' after PASS verification."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        pass_result = _make_verification_result(_pass_criteria(), "All checks passed")
        verification_agent = _make_mock_verification_agent([pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["workflow_status"] == "approved"

    def test_final_response_equals_draft_on_pass(self):
        """Happy path: final_response is set to the draft when verification passes."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        pass_result = _make_verification_result(_pass_criteria())
        verification_agent = _make_mock_verification_agent([pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["final_response"] == self.GOOD_DRAFT

    def test_verification_result_is_approved_on_pass(self):
        """Happy path: verification_result.is_approved is True."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        pass_result = _make_verification_result(_pass_criteria())
        verification_agent = _make_mock_verification_agent([pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["verification_result"] is not None
        assert final_state["verification_result"].is_approved is True

    def test_retry_count_is_zero_on_first_pass(self):
        """Happy path: no retries needed."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        pass_result = _make_verification_result(_pass_criteria())
        verification_agent = _make_mock_verification_agent([pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["retry_count"] == 0

    def test_execution_log_contains_research_and_verification_nodes(self):
        """Happy path: execution log records both research and verification steps."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        pass_result = _make_verification_result(_pass_criteria())
        verification_agent = _make_mock_verification_agent([pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        node_names = [step.node_name for step in final_state["execution_log"]]
        assert "research" in node_names
        assert "verification" in node_names


# ---------------------------------------------------------------------------
# Scenario 2: Correction path – FAIL → correction → retry → PASS
# ---------------------------------------------------------------------------

class TestCorrectionPath:
    """Research Agent produces a draft with price issues → FAIL → correction → PASS."""

    BAD_DRAFT = (
        "iPhone 15 Pro Max có giá 35,000,000 VND với chip A17 Pro. "
        "Bảo hành chính hãng 12 tháng."
    )
    CORRECTED_DRAFT = (
        "iPhone 15 Pro Max có giá 29,990,000 VND với chip A17 Pro. "
        "Bảo hành chính hãng 12 tháng theo chính sách Apple."
    )

    def test_workflow_ends_approved_after_correction(self):
        """Correction path: workflow_status == 'approved' after retry succeeds."""
        config = _make_config(max_retries=3, critical_issue_escalation=False)
        research_agent = _make_mock_research_agent([self.BAD_DRAFT, self.CORRECTED_DRAFT])
        fail_result = _make_verification_result(_fail_price_criteria(), "Price deviation detected")
        pass_result = _make_verification_result(_pass_criteria(), "All checks passed after correction")
        verification_agent = _make_mock_verification_agent([fail_result, pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["workflow_status"] == "approved"

    def test_retry_count_incremented_after_correction(self):
        """Correction path: retry_count is 1 after one correction cycle."""
        config = _make_config(max_retries=3, critical_issue_escalation=False)
        research_agent = _make_mock_research_agent([self.BAD_DRAFT, self.CORRECTED_DRAFT])
        fail_result = _make_verification_result(_fail_price_criteria())
        pass_result = _make_verification_result(_pass_criteria())
        verification_agent = _make_mock_verification_agent([fail_result, pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["retry_count"] == 1

    def test_correction_feedback_is_set_after_fail(self):
        """Correction path: correction_feedback is populated after verification FAIL."""
        config = _make_config(max_retries=3, critical_issue_escalation=False)
        research_agent = _make_mock_research_agent([self.BAD_DRAFT, self.CORRECTED_DRAFT])
        fail_result = _make_verification_result(_fail_price_criteria())
        pass_result = _make_verification_result(_pass_criteria())
        verification_agent = _make_mock_verification_agent([fail_result, pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        # correction_feedback must be a non-empty string after correction
        assert final_state["correction_feedback"] is not None
        assert len(final_state["correction_feedback"]) > 0

    def test_execution_log_contains_correction_node(self):
        """Correction path: execution log records the correction node."""
        config = _make_config(max_retries=3, critical_issue_escalation=False)
        research_agent = _make_mock_research_agent([self.BAD_DRAFT, self.CORRECTED_DRAFT])
        fail_result = _make_verification_result(_fail_price_criteria())
        pass_result = _make_verification_result(_pass_criteria())
        verification_agent = _make_mock_verification_agent([fail_result, pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        node_names = [step.node_name for step in final_state["execution_log"]]
        assert "correction" in node_names

    def test_research_agent_called_twice_in_correction_path(self):
        """Correction path: research agent is called once for initial draft and once for retry."""
        config = _make_config(max_retries=3, critical_issue_escalation=False)
        research_agent = _make_mock_research_agent([self.BAD_DRAFT, self.CORRECTED_DRAFT])
        fail_result = _make_verification_result(_fail_price_criteria())
        pass_result = _make_verification_result(_pass_criteria())
        verification_agent = _make_mock_verification_agent([fail_result, pass_result])

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        asyncio.run(workflow.execute_workflow("iPhone quá đắt so với Samsung"))

        assert research_agent.run.call_count == 2


# ---------------------------------------------------------------------------
# Scenario 3: Escalation path – max retries exceeded
# ---------------------------------------------------------------------------

class TestEscalationPath:
    """Research Agent repeatedly produces bad drafts → escalation after max retries."""

    BAD_DRAFT = (
        "iPhone 15 Pro Max có giá 35,000,000 VND. "
        "Bảo hành 5 năm toàn bộ linh kiện theo chính sách tự bịa."
    )

    def test_workflow_escalated_after_max_retries(self):
        """Escalation path: workflow_status == 'escalated' when max retries exceeded."""
        config = _make_config(max_retries=2, critical_issue_escalation=False)
        # Provide enough bad drafts for all retries + initial attempt
        research_agent = _make_mock_research_agent([self.BAD_DRAFT] * 10)
        fail_results = [
            _make_verification_result(_fail_price_criteria(), f"Fail attempt {i}")
            for i in range(10)
        ]
        verification_agent = _make_mock_verification_agent(fail_results)

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["workflow_status"] == "escalated"

    def test_final_response_contains_escalation_marker(self):
        """Escalation path: final_response contains escalation indicator."""
        config = _make_config(max_retries=2, critical_issue_escalation=False)
        research_agent = _make_mock_research_agent([self.BAD_DRAFT] * 10)
        fail_results = [
            _make_verification_result(_fail_price_criteria()) for _ in range(10)
        ]
        verification_agent = _make_mock_verification_agent(fail_results)

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert "ESCALATED" in final_state["final_response"].upper()

    def test_immediate_escalation_on_fabricated_policy(self):
        """Escalation path: fabricated policy (CRITICAL) triggers immediate escalation."""
        config = _make_config(max_retries=3, critical_issue_escalation=True)
        research_agent = _make_mock_research_agent([self.BAD_DRAFT] * 10)
        # Fabricated policy → CRITICAL → immediate escalation
        critical_fail = _make_verification_result(
            _fail_policy_criteria(fabricated=True),
            "Fabricated policy detected",
        )
        verification_agent = _make_mock_verification_agent([critical_fail] * 10)

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["workflow_status"] == "escalated"
        # Should escalate immediately without exhausting retries
        assert final_state["retry_count"] == 0

    def test_retry_count_does_not_exceed_max_retries(self):
        """Escalation path: retry_count never exceeds max_retries."""
        max_retries = 2
        config = _make_config(max_retries=max_retries, critical_issue_escalation=False)
        research_agent = _make_mock_research_agent([self.BAD_DRAFT] * 10)
        fail_results = [
            _make_verification_result(_fail_price_criteria()) for _ in range(10)
        ]
        verification_agent = _make_mock_verification_agent(fail_results)

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["retry_count"] <= max_retries


# ---------------------------------------------------------------------------
# Scenario 4: Binary decision validation
# ---------------------------------------------------------------------------

class TestBinaryDecisionValidation:
    """Verify that decisions are strictly binary PASS/FAIL, not numeric scores."""

    def test_is_approved_is_boolean(self):
        """VerificationResult.is_approved must be a bool, not a numeric score."""
        result = _make_verification_result(_pass_criteria())
        assert isinstance(result.is_approved, bool)
        assert result.is_approved is True

    def test_is_approved_false_when_any_criterion_fails(self):
        """is_approved is False when any single criterion fails."""
        for criteria in [
            _fail_price_criteria(),
            _fail_policy_criteria(fabricated=False),
            _fail_relevance_criteria(),
        ]:
            result = _make_verification_result(criteria)
            assert result.is_approved is False, (
                f"Expected is_approved=False for criteria: {criteria}"
            )

    def test_is_approved_true_only_when_all_criteria_pass(self):
        """is_approved is True only when all three criteria pass."""
        result = _make_verification_result(_pass_criteria())
        assert result.is_approved is True
        assert result.criteria.price_accuracy_pass is True
        assert result.criteria.policy_authenticity_pass is True
        assert result.criteria.topic_relevance_pass is True

    def test_overall_pass_matches_is_approved(self):
        """criteria.overall_pass must equal is_approved."""
        for criteria in [_pass_criteria(), _fail_price_criteria(), _fail_policy_criteria()]:
            result = _make_verification_result(criteria)
            assert result.criteria.overall_pass == result.is_approved

    def test_routing_approved_on_pass(self):
        """WorkflowRouter routes to 'approved' when verification passes."""
        config = _make_config()
        router = WorkflowRouter(config)

        state: WorkflowState = {
            "objection_text": "test",
            "customer_context": {},
            "draft_response": "test draft",
            "tools_used": [],
            "research_reasoning": "",
            "research_sources": [],
            "verification_result": _make_verification_result(_pass_criteria()),
            "correction_feedback": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_response": "",
            "workflow_status": "verifying",
            "execution_log": [],
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": {},
        }

        decision = router.route_after_verification(state)
        assert decision == "approved"

    def test_routing_correction_on_major_fail(self):
        """WorkflowRouter routes to 'correction' on MAJOR fail with retries remaining."""
        config = _make_config(max_retries=3, critical_issue_escalation=False)
        router = WorkflowRouter(config)

        state: WorkflowState = {
            "objection_text": "test",
            "customer_context": {},
            "draft_response": "test draft",
            "tools_used": [],
            "research_reasoning": "",
            "research_sources": [],
            "verification_result": _make_verification_result(
                _fail_price_criteria(IssueSeverity.MAJOR)
            ),
            "correction_feedback": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_response": "",
            "workflow_status": "verifying",
            "execution_log": [],
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": {},
        }

        decision = router.route_after_verification(state)
        assert decision == "correction"

    def test_routing_escalation_on_critical_fail(self):
        """WorkflowRouter routes to 'escalation' on CRITICAL fabricated policy."""
        config = _make_config(max_retries=3, critical_issue_escalation=True)
        router = WorkflowRouter(config)

        state: WorkflowState = {
            "objection_text": "test",
            "customer_context": {},
            "draft_response": "test draft",
            "tools_used": [],
            "research_reasoning": "",
            "research_sources": [],
            "verification_result": _make_verification_result(
                _fail_policy_criteria(fabricated=True)
            ),
            "correction_feedback": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_response": "",
            "workflow_status": "verifying",
            "execution_log": [],
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": {},
        }

        decision = router.route_after_verification(state)
        assert decision == "escalation"

    def test_routing_escalation_when_max_retries_reached(self):
        """WorkflowRouter routes to 'escalation' when retry_count >= max_retries."""
        config = _make_config(max_retries=2, critical_issue_escalation=False)
        router = WorkflowRouter(config)

        state: WorkflowState = {
            "objection_text": "test",
            "customer_context": {},
            "draft_response": "test draft",
            "tools_used": [],
            "research_reasoning": "",
            "research_sources": [],
            "verification_result": _make_verification_result(
                _fail_price_criteria(IssueSeverity.MAJOR)
            ),
            "correction_feedback": None,
            "retry_count": 2,   # equals max_retries
            "max_retries": 2,
            "final_response": "",
            "workflow_status": "verifying",
            "execution_log": [],
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": {},
        }

        decision = router.route_after_verification(state)
        assert decision == "escalation"


# ---------------------------------------------------------------------------
# Scenario 5: State integrity
# ---------------------------------------------------------------------------

class TestStateIntegrity:
    """WorkflowState is properly maintained throughout the workflow."""

    GOOD_DRAFT = (
        "iPhone 15 Pro Max có giá 29,990,000 VND. Bảo hành 12 tháng chính hãng."
    )

    def test_objection_text_preserved_in_final_state(self):
        """State integrity: objection_text is unchanged from input to final state."""
        objection = "iPhone quá đắt so với Samsung"
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(workflow.execute_workflow(objection))

        assert final_state["objection_text"] == objection

    def test_draft_response_populated_after_research(self):
        """State integrity: draft_response is set after research node executes."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["draft_response"] == self.GOOD_DRAFT

    def test_start_time_is_set(self):
        """State integrity: start_time is a valid ISO timestamp."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["start_time"] is not None
        # Should be parseable as ISO datetime
        datetime.fromisoformat(final_state["start_time"])

    def test_end_time_is_set_after_completion(self):
        """State integrity: end_time is set when workflow completes."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["end_time"] is not None

    def test_execution_log_steps_have_required_fields(self):
        """State integrity: each ExecutionStep has node_name, status, execution_time."""
        config = _make_config()
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        for step in final_state["execution_log"]:
            assert step.node_name, "ExecutionStep must have node_name"
            assert step.status is not None, "ExecutionStep must have status"
            assert step.execution_time >= 0.0, "ExecutionStep execution_time must be non-negative"

    def test_max_retries_preserved_from_config(self):
        """State integrity: max_retries in state matches config."""
        max_retries = 5
        config = _make_config(max_retries=max_retries)
        research_agent = _make_mock_research_agent([self.GOOD_DRAFT])
        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert final_state["max_retries"] == max_retries

    def test_tools_used_populated_from_research_agent(self):
        """State integrity: tools_used reflects what the research agent reported."""
        config = _make_config()
        agent_result = AgentResult(
            objection_text="iPhone quá đắt so với Samsung",
            draft_response=self.GOOD_DRAFT,
            tools_used=["internal_db_search", "price_lookup"],
        )
        research_agent = MagicMock()
        research_agent.run.return_value = agent_result
        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("iPhone quá đắt so với Samsung")
        )

        assert "internal_db_search" in final_state["tools_used"]
        assert "price_lookup" in final_state["tools_used"]


# ---------------------------------------------------------------------------
# Scenario 6: Real Research Agent output format compatibility
# ---------------------------------------------------------------------------

class TestResearchAgentOutputCompatibility:
    """Verify that the workflow correctly handles the real AgentResult format."""

    def test_agent_result_fields_map_to_workflow_state(self):
        """AgentResult fields are correctly mapped into WorkflowState."""
        config = _make_config()

        agent_result = AgentResult(
            objection_text="Samsung đắt hơn iPhone",
            draft_response="Samsung Galaxy S24 Ultra có giá 31,990,000 VND.",
            tools_used=["internal_db_search"],
        )
        research_agent = MagicMock()
        research_agent.run.return_value = agent_result

        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(
            workflow.execute_workflow("Samsung đắt hơn iPhone")
        )

        assert final_state["draft_response"] == agent_result.draft_response
        assert final_state["tools_used"] == agent_result.tools_used

    def test_workflow_handles_empty_tools_used(self):
        """Workflow handles AgentResult with empty tools_used list."""
        config = _make_config()

        agent_result = AgentResult(
            objection_text="test",
            draft_response="Dạ, đây là phản hồi.",
            tools_used=[],
        )
        research_agent = MagicMock()
        research_agent.run.return_value = agent_result

        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(workflow.execute_workflow("test"))

        assert final_state["tools_used"] == []
        assert final_state["workflow_status"] == "approved"

    def test_workflow_handles_long_draft_response(self):
        """Workflow handles a long draft response without truncation errors."""
        config = _make_config()
        long_draft = "Dạ, " + ("iPhone 15 Pro Max có giá 29,990,000 VND. " * 50)

        agent_result = AgentResult(
            objection_text="iPhone quá đắt",
            draft_response=long_draft,
            tools_used=["internal_db_search"],
        )
        research_agent = MagicMock()
        research_agent.run.return_value = agent_result

        verification_agent = _make_mock_verification_agent(
            [_make_verification_result(_pass_criteria())]
        )

        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

        import asyncio
        final_state = asyncio.run(workflow.execute_workflow("iPhone quá đắt"))

        assert final_state["workflow_status"] == "approved"
        assert final_state["final_response"] == long_draft


# ---------------------------------------------------------------------------
# Scenario 7: SelfCorrectionNode generates structured feedback
# ---------------------------------------------------------------------------

class TestSelfCorrectionFeedback:
    """SelfCorrectionNode generates actionable feedback for each issue type."""

    def test_correction_feedback_mentions_price_issue(self):
        """Correction feedback includes price issue details."""
        config = _make_config()
        node = SelfCorrectionNode(config)
        result = _make_verification_result(_fail_price_criteria())

        feedback = node.generate_correction_feedback(
            original_objection="iPhone quá đắt",
            failed_draft="iPhone 15 Pro Max có giá 35,000,000 VND.",
            verification_result=result,
        )

        assert "iPhone 15 Pro Max" in feedback
        assert "CORRECTION" in feedback.upper() or "FAILED" in feedback.upper()

    def test_correction_feedback_mentions_policy_issue(self):
        """Correction feedback includes policy issue details."""
        config = _make_config()
        node = SelfCorrectionNode(config)
        result = _make_verification_result(_fail_policy_criteria(fabricated=True))

        feedback = node.generate_correction_feedback(
            original_objection="Bảo hành thế nào?",
            failed_draft="Bảo hành 5 năm toàn bộ linh kiện.",
            verification_result=result,
        )

        assert "warranty" in feedback.lower() or "policy" in feedback.lower() or "bảo hành" in feedback.lower()

    def test_correction_feedback_mentions_relevance_issue(self):
        """Correction feedback includes relevance issue details."""
        config = _make_config()
        node = SelfCorrectionNode(config)
        result = _make_verification_result(_fail_relevance_criteria())

        feedback = node.generate_correction_feedback(
            original_objection="So sánh giá iPhone vs Samsung",
            failed_draft="iPhone rất tốt.",
            verification_result=result,
        )

        assert len(feedback) > 50  # non-trivial feedback

    def test_no_correction_needed_when_approved(self):
        """SelfCorrectionNode returns no-correction message when result is approved."""
        config = _make_config()
        node = SelfCorrectionNode(config)
        result = _make_verification_result(_pass_criteria())

        feedback = node.generate_correction_feedback(
            original_objection="test",
            failed_draft="test draft",
            verification_result=result,
        )

        assert "no corrections needed" in feedback.lower() or "passed" in feedback.lower()
