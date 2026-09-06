"""
Unit tests for task 4.1.1: SalesResearchAgent structured issue feedback support.

Tests verify:
- Backward compatibility: run(objection) still works unchanged
- Correction feedback is incorporated into the query when provided
- Structured verification_issues are formatted into the query
- build_correction_context helper formats issues correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.workflows.research_agent.prompts import build_correction_context
from backend.workflows.research_agent.sales_research_agent import AgentResult, SalesResearchAgent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_agent() -> SalesResearchAgent:
    """Return a SalesResearchAgent with mocked LLM and RAG pipeline."""
    llm = MagicMock()
    rag_pipeline = MagicMock()

    with patch("backend.workflows.research_agent.sales_research_agent.build_internal_db_tool", return_value=MagicMock()), \
         patch("backend.workflows.research_agent.sales_research_agent.build_tavily_tool", return_value=None), \
         patch("backend.workflows.research_agent.sales_research_agent.ReActAgent") as mock_react:
        mock_react.from_tools.return_value = MagicMock()
        agent = SalesResearchAgent(llm=llm, rag_pipeline=rag_pipeline)

    return agent


def _mock_chat_response(text: str = "Dạ, đây là bản nháp.") -> MagicMock:
    response = MagicMock()
    response.__str__ = lambda self: text
    response.sources = []
    return response


# ---------------------------------------------------------------------------
# build_correction_context tests
# ---------------------------------------------------------------------------

class TestBuildCorrectionContext:
    def test_returns_string_with_feedback(self):
        ctx = build_correction_context("Fix the price of iPhone 15.")
        assert "Fix the price of iPhone 15." in ctx

    def test_no_issues_no_detail_section(self):
        ctx = build_correction_context("Some feedback", verification_issues=None)
        assert "CHI TIẾT" not in ctx

    def test_price_issue_formatted(self):
        price_issue = MagicMock()
        price_issue.__class__.__name__ = "PriceIssue"
        price_issue.product_name = "iPhone 15"
        price_issue.mentioned_price = "35,000,000 VND"
        price_issue.actual_price = "29,990,000 VND"
        price_issue.deviation_percent = 16.7
        price_issue.explanation = "Price deviation"

        ctx = build_correction_context("Fix price", verification_issues=[price_issue])
        assert "iPhone 15" in ctx
        assert "35,000,000 VND" in ctx
        assert "29,990,000 VND" in ctx
        assert "16.7%" in ctx

    def test_policy_issue_fabricated_formatted(self):
        policy_issue = MagicMock()
        policy_issue.__class__.__name__ = "PolicyIssue"
        policy_issue.policy_type = "warranty"
        policy_issue.mentioned_policy = "Bảo hành 3 năm"
        policy_issue.is_fabricated = True
        policy_issue.correct_policy = "Bảo hành 1 năm"

        ctx = build_correction_context("Fix policy", verification_issues=[policy_issue])
        assert "BỊA ĐẶT" in ctx
        assert "warranty" in ctx
        assert "Bảo hành 3 năm" in ctx
        assert "Bảo hành 1 năm" in ctx

    def test_relevance_issue_formatted(self):
        rel_issue = MagicMock()
        rel_issue.__class__.__name__ = "RelevanceIssue"
        rel_issue.response_coverage = 0.45
        rel_issue.explanation = "Response misses key aspects"
        rel_issue.missing_aspects = ["camera", "battery"]

        ctx = build_correction_context("Fix relevance", verification_issues=[rel_issue])
        assert "45%" in ctx
        assert "camera" in ctx
        assert "battery" in ctx

    def test_multiple_issues_all_present(self):
        price_issue = MagicMock()
        price_issue.__class__.__name__ = "PriceIssue"
        price_issue.product_name = "Samsung S24"
        price_issue.mentioned_price = "20M"
        price_issue.actual_price = "22M"
        price_issue.deviation_percent = 9.1
        price_issue.explanation = "Price off"

        policy_issue = MagicMock()
        policy_issue.__class__.__name__ = "PolicyIssue"
        policy_issue.policy_type = "return"
        policy_issue.mentioned_policy = "30-day return"
        policy_issue.is_fabricated = False
        policy_issue.correct_policy = "14-day return"

        ctx = build_correction_context("Fix both", verification_issues=[price_issue, policy_issue])
        assert "Samsung S24" in ctx
        assert "return" in ctx


# ---------------------------------------------------------------------------
# SalesResearchAgent.run() backward compatibility
# ---------------------------------------------------------------------------

class TestRunBackwardCompatibility:
    def test_run_without_correction_feedback(self):
        agent = _make_agent()
        agent._agent.chat.return_value = _mock_chat_response("Dạ, bản nháp.")

        result = agent.run("iPhone quá đắt")

        assert isinstance(result, AgentResult)
        assert result.objection_text == "iPhone quá đắt"
        assert result.draft_response == "Dạ, bản nháp."
        # The grounded prompt must preserve the original objection.
        call_args = agent._agent.chat.call_args[0][0]
        assert "iPhone quá đắt" in call_args
        assert "DỮ LIỆU NỘI BỘ" in call_args

    def test_run_returns_agent_result_on_exception(self):
        agent = _make_agent()
        agent._agent.chat.side_effect = RuntimeError("LLM timeout")

        result = agent.run("Some objection")

        assert isinstance(result, AgentResult)
        assert "chưa tìm thấy dữ liệu nội bộ" in result.draft_response
        assert result.tools_used == ["internal_db_search"]


# ---------------------------------------------------------------------------
# SalesResearchAgent.run() with correction feedback
# ---------------------------------------------------------------------------

class TestRunWithCorrectionFeedback:
    def test_correction_feedback_prepended_to_query(self):
        agent = _make_agent()
        agent._agent.chat.return_value = _mock_chat_response("Dạ, bản nháp sửa.")

        result = agent.run(
            "iPhone quá đắt",
            correction_feedback="Fix the price of iPhone 15.",
        )

        assert isinstance(result, AgentResult)
        assert result.objection_text == "iPhone quá đắt"
        call_args = agent._agent.chat.call_args[0][0]
        # Correction context must be in the query
        assert "Fix the price of iPhone 15." in call_args
        # Original objection must still be present
        assert "iPhone quá đắt" in call_args

    def test_correction_feedback_with_verification_issues(self):
        agent = _make_agent()
        agent._agent.chat.return_value = _mock_chat_response("Dạ, đã sửa.")

        price_issue = MagicMock()
        price_issue.__class__.__name__ = "PriceIssue"
        price_issue.product_name = "iPhone 15"
        price_issue.mentioned_price = "35M"
        price_issue.actual_price = "30M"
        price_issue.deviation_percent = 16.7
        price_issue.explanation = "Price deviation"

        result = agent.run(
            "iPhone quá đắt",
            correction_feedback="Correct the price.",
            verification_issues=[price_issue],
        )

        call_args = agent._agent.chat.call_args[0][0]
        assert "iPhone 15" in call_args
        assert "35M" in call_args
        assert "30M" in call_args

    def test_none_correction_feedback_behaves_like_no_feedback(self):
        agent = _make_agent()
        agent._agent.chat.return_value = _mock_chat_response("Dạ.")

        result = agent.run("objection", correction_feedback=None)

        call_args = agent._agent.chat.call_args[0][0]
        assert "objection" in call_args
        assert "DỮ LIỆU NỘI BỘ" in call_args

    def test_objection_text_preserved_in_result(self):
        agent = _make_agent()
        agent._agent.chat.return_value = _mock_chat_response("Dạ.")

        result = agent.run(
            "Samsung đắt hơn iPhone",
            correction_feedback="Fix relevance.",
        )

        assert result.objection_text == "Samsung đắt hơn iPhone"
