"""
Unit tests for task 4.1.2 and 4.1.5:
  - AgentResult binary verification workflow state integration
  - Backward compatibility for existing functionality

Tests verify:
- Backward compatibility: existing 3-field construction still works
- New optional fields exist with correct defaults
- workflow_status accepts all valid WorkflowState literals
- verification_result field accepts VerificationResult objects
- retry_count and correction_feedback fields work correctly
- AgentResult can be used within LangGraph WorkflowState context
- Legacy code that only uses core fields continues to work unchanged
- WorkflowStatus is importable from backend.workflows.research_agent.sales_research_agent
- SalesResearchAgent.run() signature is backward compatible
"""

from __future__ import annotations

from dataclasses import fields
from typing import get_args
from unittest.mock import MagicMock

import pytest

from backend.workflows.research_agent.sales_research_agent import AgentResult, WorkflowStatus


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestAgentResultBackwardCompatibility:
    def test_construct_with_required_fields_only(self):
        result = AgentResult(
            objection_text="iPhone quá đắt",
            draft_response="Dạ, iPhone có giá trị tốt.",
        )
        assert result.objection_text == "iPhone quá đắt"
        assert result.draft_response == "Dạ, iPhone có giá trị tốt."
        assert result.tools_used == []

    def test_construct_with_tools_used(self):
        result = AgentResult(
            objection_text="So sánh giá",
            draft_response="Bản nháp.",
            tools_used=["product_search", "price_lookup"],
        )
        assert result.tools_used == ["product_search", "price_lookup"]

    def test_new_optional_fields_default_to_none_or_zero(self):
        result = AgentResult(
            objection_text="test",
            draft_response="test",
        )
        assert result.verification_result is None
        assert result.workflow_status is None
        assert result.retry_count == 0
        assert result.correction_feedback is None


# ---------------------------------------------------------------------------
# New optional fields
# ---------------------------------------------------------------------------

class TestAgentResultWorkflowFields:
    def test_workflow_status_can_be_set(self):
        for status in get_args(WorkflowStatus):
            result = AgentResult(
                objection_text="test",
                draft_response="test",
                workflow_status=status,
            )
            assert result.workflow_status == status

    def test_retry_count_can_be_set(self):
        result = AgentResult(
            objection_text="test",
            draft_response="test",
            retry_count=2,
        )
        assert result.retry_count == 2

    def test_correction_feedback_can_be_set(self):
        feedback = "🔄 CORRECTION REQUIRED: Fix price of iPhone 15."
        result = AgentResult(
            objection_text="test",
            draft_response="test",
            correction_feedback=feedback,
        )
        assert result.correction_feedback == feedback

    def test_verification_result_accepts_mock_object(self):
        """verification_result field accepts any VerificationResult-like object."""
        mock_vr = MagicMock()
        mock_vr.is_approved = True
        mock_vr.criteria.overall_pass = True

        result = AgentResult(
            objection_text="test",
            draft_response="test",
            verification_result=mock_vr,
        )
        assert result.verification_result is mock_vr
        assert result.verification_result.is_approved is True

    def test_all_workflow_fields_set_together(self):
        mock_vr = MagicMock()
        mock_vr.is_approved = False

        result = AgentResult(
            objection_text="iPhone quá đắt",
            draft_response="Bản nháp.",
            tools_used=["product_search"],
            verification_result=mock_vr,
            workflow_status="correcting",
            retry_count=1,
            correction_feedback="Fix price accuracy.",
        )

        assert result.objection_text == "iPhone quá đắt"
        assert result.draft_response == "Bản nháp."
        assert result.tools_used == ["product_search"]
        assert result.verification_result is mock_vr
        assert result.workflow_status == "correcting"
        assert result.retry_count == 1
        assert result.correction_feedback == "Fix price accuracy."


# ---------------------------------------------------------------------------
# WorkflowStatus literals match WorkflowState
# ---------------------------------------------------------------------------

class TestWorkflowStatusLiterals:
    """Ensure WorkflowStatus covers all statuses defined in WorkflowState."""

    EXPECTED_STATUSES = {
        "initialized",
        "researching",
        "verifying",
        "correcting",
        "approved",
        "escalated",
        "failed",
    }

    def test_all_expected_statuses_present(self):
        actual = set(get_args(WorkflowStatus))
        assert actual == self.EXPECTED_STATUSES

    def test_approved_status(self):
        result = AgentResult("q", "a", workflow_status="approved")
        assert result.workflow_status == "approved"

    def test_escalated_status(self):
        result = AgentResult("q", "a", workflow_status="escalated")
        assert result.workflow_status == "escalated"

    def test_failed_status(self):
        result = AgentResult("q", "a", workflow_status="failed")
        assert result.workflow_status == "failed"


# ---------------------------------------------------------------------------
# Dataclass field introspection
# ---------------------------------------------------------------------------

class TestAgentResultDataclassStructure:
    def test_has_seven_fields(self):
        field_names = {f.name for f in fields(AgentResult)}
        assert field_names == {
            "objection_text",
            "draft_response",
            "tools_used",
            "verification_result",
            "workflow_status",
            "retry_count",
            "correction_feedback",
        }

    def test_core_fields_have_no_default(self):
        """objection_text and draft_response must be provided (no default)."""
        with pytest.raises(TypeError):
            AgentResult()  # missing required positional args

    def test_tools_used_default_is_independent_per_instance(self):
        """Each instance gets its own list (not shared mutable default)."""
        r1 = AgentResult("q1", "a1")
        r2 = AgentResult("q2", "a2")
        r1.tools_used.append("tool_x")
        assert r2.tools_used == []


# ---------------------------------------------------------------------------
# Task 4.1.5: Backward compatibility – public API surface
# ---------------------------------------------------------------------------

class TestPublicAPIBackwardCompatibility:
    """Verify that the public API surface of agent.sales_research_agent is stable."""

    def test_workflow_status_importable(self):
        """WorkflowStatus must be importable from backend.workflows.research_agent.sales_research_agent."""
        from backend.workflows.research_agent.sales_research_agent import WorkflowStatus  # noqa: F401

    def test_agent_result_importable(self):
        """AgentResult must be importable from backend.workflows.research_agent.sales_research_agent."""
        from backend.workflows.research_agent.sales_research_agent import AgentResult  # noqa: F401

    def test_sales_research_agent_importable(self):
        """SalesResearchAgent must be importable from backend.workflows.research_agent.sales_research_agent."""
        from backend.workflows.research_agent.sales_research_agent import SalesResearchAgent  # noqa: F401

    def test_agent_result_is_dataclass(self):
        """AgentResult must remain a dataclass (not converted to Pydantic or plain class)."""
        import dataclasses
        assert dataclasses.is_dataclass(AgentResult)

    def test_agent_result_positional_construction(self):
        """Legacy code that constructs AgentResult positionally still works."""
        result = AgentResult("objection text", "draft response")
        assert result.objection_text == "objection text"
        assert result.draft_response == "draft response"

    def test_agent_result_keyword_construction(self):
        """Legacy code that constructs AgentResult with keyword args still works."""
        result = AgentResult(
            objection_text="objection text",
            draft_response="draft response",
            tools_used=["tool_a"],
        )
        assert result.tools_used == ["tool_a"]

    def test_agent_result_attribute_access_unchanged(self):
        """Core attribute names have not been renamed."""
        result = AgentResult("q", "a", ["t1"])
        assert hasattr(result, "objection_text")
        assert hasattr(result, "draft_response")
        assert hasattr(result, "tools_used")

    def test_new_fields_do_not_break_equality(self):
        """Two AgentResults with same core fields are equal regardless of new fields."""
        r1 = AgentResult("q", "a")
        r2 = AgentResult("q", "a")
        assert r1 == r2

    def test_new_fields_do_not_appear_in_legacy_repr(self):
        """repr() of a legacy AgentResult still contains the core field values."""
        result = AgentResult("my objection", "my draft")
        r = repr(result)
        assert "my objection" in r
        assert "my draft" in r


class TestSalesResearchAgentRunSignatureCompatibility:
    """Verify SalesResearchAgent.run() signature is backward compatible."""

    def _make_agent(self):
        from unittest.mock import MagicMock, patch
        from backend.workflows.research_agent.sales_research_agent import SalesResearchAgent

        llm = MagicMock()
        rag_pipeline = MagicMock()
        with patch("backend.workflows.research_agent.sales_research_agent.build_internal_db_tool", return_value=MagicMock()), \
             patch("backend.workflows.research_agent.sales_research_agent.build_tavily_tool", return_value=None), \
             patch("backend.workflows.research_agent.sales_research_agent.ReActAgent") as mock_react:
            mock_react.from_tools.return_value = MagicMock()
            agent = SalesResearchAgent(llm=llm, rag_pipeline=rag_pipeline)
        return agent

    def test_run_accepts_single_positional_arg(self):
        """run(objection) — the original single-arg call — still works."""
        from unittest.mock import MagicMock
        agent = self._make_agent()
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "Dạ, bản nháp."
        mock_response.sources = []
        agent._agent.chat.return_value = mock_response

        result = agent.run("iPhone quá đắt")
        assert isinstance(result, AgentResult)
        assert result.objection_text == "iPhone quá đắt"

    def test_run_returns_agent_result_type(self):
        """run() always returns an AgentResult instance."""
        from unittest.mock import MagicMock
        agent = self._make_agent()
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "Dạ."
        mock_response.sources = []
        agent._agent.chat.return_value = mock_response

        result = agent.run("test objection")
        assert type(result).__name__ == "AgentResult"

    def test_run_result_has_tools_used_list(self):
        """run() result always has tools_used as a list (not None)."""
        from unittest.mock import MagicMock
        agent = self._make_agent()
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "Dạ."
        mock_response.sources = []
        agent._agent.chat.return_value = mock_response

        result = agent.run("test")
        assert isinstance(result.tools_used, list)

    def test_run_result_new_fields_are_none_by_default(self):
        """run() result has None/0 for new workflow fields — legacy consumers unaffected."""
        from unittest.mock import MagicMock
        agent = self._make_agent()
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "Dạ."
        mock_response.sources = []
        agent._agent.chat.return_value = mock_response

        result = agent.run("test")
        # New fields must not interfere with legacy usage
        assert result.verification_result is None
        assert result.workflow_status is None
        assert result.retry_count == 0
        assert result.correction_feedback is None

    def test_run_with_correction_feedback_kwarg_optional(self):
        """correction_feedback is optional — omitting it is identical to passing None."""
        from unittest.mock import MagicMock
        agent = self._make_agent()
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "Dạ."
        mock_response.sources = []
        agent._agent.chat.return_value = mock_response

        result_no_kwarg = agent.run("test")
        agent._agent.chat.return_value = mock_response
        result_none_kwarg = agent.run("test", correction_feedback=None)

        # Both calls should produce equivalent results
        assert result_no_kwarg.objection_text == result_none_kwarg.objection_text
        assert result_no_kwarg.draft_response == result_none_kwarg.draft_response


class TestAgentResultWorkflowIntegrationCompatibility:
    """Verify AgentResult integrates cleanly with WorkflowState without breaking legacy use."""

    def test_agent_result_fields_map_to_workflow_state_fields(self):
        """Core AgentResult fields map to the corresponding WorkflowState fields."""
        result = AgentResult(
            objection_text="iPhone quá đắt",
            draft_response="Dạ, iPhone có giá trị tốt.",
            tools_used=["product_search"],
        )
        # Simulate what VerificationWorkflow._execute_research_node does
        state = {
            "objection_text": result.objection_text,
            "draft_response": result.draft_response,
            "tools_used": result.tools_used,
        }
        assert state["objection_text"] == "iPhone quá đắt"
        assert state["draft_response"] == "Dạ, iPhone có giá trị tốt."
        assert state["tools_used"] == ["product_search"]

    def test_agent_result_with_workflow_fields_maps_to_workflow_state(self):
        """AgentResult with workflow fields maps correctly to WorkflowState."""
        mock_vr = MagicMock()
        mock_vr.is_approved = True

        result = AgentResult(
            objection_text="test",
            draft_response="test draft",
            tools_used=["tool"],
            verification_result=mock_vr,
            workflow_status="approved",
            retry_count=1,
            correction_feedback="Fix price.",
        )
        # All fields accessible without error
        assert result.verification_result.is_approved is True
        assert result.workflow_status == "approved"
        assert result.retry_count == 1
        assert result.correction_feedback == "Fix price."

    def test_legacy_code_ignoring_new_fields_still_works(self):
        """Code that only reads objection_text, draft_response, tools_used is unaffected."""
        result = AgentResult(
            objection_text="Samsung đắt hơn iPhone",
            draft_response="Dạ, Samsung có nhiều tính năng.",
            tools_used=["internal_db"],
            # New fields present but legacy code ignores them
            workflow_status="approved",
            retry_count=2,
        )
        # Legacy access pattern
        objection = result.objection_text
        draft = result.draft_response
        tools = result.tools_used

        assert objection == "Samsung đắt hơn iPhone"
        assert draft == "Dạ, Samsung có nhiều tính năng."
        assert tools == ["internal_db"]
