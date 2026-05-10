from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from llama_index.core.agent import ReActAgent

from agent.tools import build_internal_db_tool, build_tavily_tool
from agent.prompts import AGENT_SYSTEM_PROMPT, build_correction_context

if TYPE_CHECKING:
    from verification.models.verification import VerificationResult

logger = logging.getLogger(__name__)

# Workflow status literals matching WorkflowState in verification/models/state.py
WorkflowStatus = Literal[
    "initialized",
    "researching",
    "verifying",
    "correcting",
    "approved",
    "escalated",
    "failed",
]


@dataclass
class AgentResult:
    """
    Result returned by SalesResearchAgent.run().

    Core fields (always populated):
        objection_text  – the original customer objection
        draft_response  – the generated draft answer
        tools_used      – list of tool names called during research

    Optional fields for binary verification workflow integration:
        verification_result – VerificationResult from the Verification Agent
                              (None until verification has been executed)
        workflow_status     – current status in the LangGraph WorkflowState
                              (None when used outside the workflow)
        retry_count         – how many correction retries have been attempted
        correction_feedback – structured correction instructions injected on retry
    """

    # --- core fields (required) ---
    objection_text: str
    draft_response: str
    tools_used: list[str] = field(default_factory=list)

    # --- binary verification workflow integration (optional) ---
    verification_result: Optional["VerificationResult"] = field(default=None)
    workflow_status: Optional[WorkflowStatus] = field(default=None)
    retry_count: int = field(default=0)
    correction_feedback: Optional[str] = field(default=None)


class SalesResearchAgent:
    def __init__(self, llm, rag_pipeline, tavily_api_key: str | None = None) -> None:
        if llm is None:
            raise ValueError("llm cannot be None")
        if rag_pipeline is None:
            raise ValueError("rag_pipeline cannot be None")

        tools = [build_internal_db_tool(rag_pipeline)]

        tavily_tool = build_tavily_tool(tavily_api_key)
        if tavily_tool is not None:
            tools.append(tavily_tool)

        if not tools:
            raise ValueError("tools list cannot be empty")

        self._agent = ReActAgent.from_tools(
            tools,
            llm=llm,
            verbose=True,
            max_iterations=2,
            context=AGENT_SYSTEM_PROMPT,
        )

    def run(
        self,
        objection: str,
        correction_feedback: Optional[str] = None,
        verification_issues: Optional[list] = None,
    ) -> AgentResult:
        """
        Run the Sales Research Agent for a given customer objection.

        Args:
            objection: The customer objection text to respond to.
            correction_feedback: Optional structured correction instructions from the
                Self-Correction Node when retrying after a failed verification.
                When provided, the feedback is prepended to the query so the agent
                knows exactly what to fix.
            verification_issues: Optional list of PriceIssue / PolicyIssue /
                RelevanceIssue objects with granular issue details.  Used together
                with ``correction_feedback`` to give the agent precise instructions.

        Returns:
            AgentResult with the draft response and tool usage metadata.
        """
        try:
            # Build the query: prepend correction context when retrying
            if correction_feedback:
                correction_ctx = build_correction_context(
                    correction_feedback=correction_feedback,
                    verification_issues=verification_issues,
                )
                query = f"{correction_ctx}\n\n---\n\nCÂU HỎI GỐC CỦA KHÁCH HÀNG:\n{objection}"
                logger.info(
                    "Running agent with correction feedback for objection: %s", objection
                )
            else:
                query = objection
                logger.info("Running agent for objection: %s", objection)

            response = self._agent.chat(query)

            tools_used = [s.tool_name for s in response.sources]

            for source in response.sources:
                logger.info(
                    "Tool called: %s | Input: %s",
                    source.tool_name,
                    source.raw_input,
                )
                logger.info("Observation: %s", source.raw_output)

            return AgentResult(
                objection_text=objection,
                draft_response=str(response),
                tools_used=tools_used,
            )
        except Exception as exc:
            logger.error("Agent error: %s", exc, exc_info=True)
            return AgentResult(
                objection_text=objection,
                draft_response="Dạ, hiện tại hệ thống đang gặp sự cố kỹ thuật. Xin phép ghi nhận để báo cáo quản lý ạ.",
                tools_used=[],
            )
