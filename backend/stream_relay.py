"""
SSE Stream Relay

Wraps VerificationWorkflow.execute_workflow() into an async generator
that yields properly formatted Server-Sent Events.

Since execute_workflow() returns a complete WorkflowState (not a token
stream), we simulate streaming by emitting one word at a time with a small
delay. This gives the UI a natural typewriter effect while preserving the
full AI reasoning pipeline.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_workflow(
    message: str,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """
    Run the AI workflow and stream the response word-by-word as SSE events.

    Yields:
        SSE-formatted strings. Event types:
        - {"type": "token", "data": "<word>"}
        - {"type": "stream_end", "thread_id": "...", "workflow_status": "..."}
        - {"type": "error", "message": "..."}
    """
    from backend.workflow_factory import get_workflow

    try:
        workflow = get_workflow()
    except Exception as exc:
        logger.error("Workflow not available: %s", exc)
        yield _sse({"type": "error", "message": "Hệ thống AI chưa sẵn sàng. Vui lòng thử lại sau."})
        return

    try:
        # execute_workflow is async — call directly without run_in_executor
        final_state = await workflow.execute_workflow(
            objection_text=message,
            customer_context={"thread_id": thread_id},
        )
    except Exception as exc:
        logger.error("Workflow execution error (thread=%s): %s", thread_id, exc, exc_info=True)
        yield _sse({"type": "error", "message": "Đã xảy ra lỗi khi xử lý yêu cầu. Vui lòng thử lại."})
        return

    # Extract the final response text
    response_text: str = final_state.get("final_response") or final_state.get("draft_response", "")
    workflow_status: str = final_state.get("workflow_status", "unknown")

    if not response_text:
        response_text = "Xin lỗi, tôi không thể xử lý câu hỏi này lúc này."

    # Handle failed workflow (e.g., API quota exhausted)
    if workflow_status == "failed":
        # Check if this is an API quota error
        error_msg = str(final_state.get("final_response", ""))
        if "quota" in error_msg.lower() or "api" in error_msg.lower():
            response_text = (
                "Xin lỗi, hệ thống tạm thời không thể xử lý yêu cầu do giới hạn API. "
                "Vui lòng thử lại sau hoặc liên hệ quản trị viên."
            )
        else:
            response_text = (
                "Xin lỗi, đã xảy ra lỗi kỹ thuật khi xử lý yêu cầu. "
                "Vui lòng thử lại sau."
            )
    # Handle escalation
    elif workflow_status == "escalated":
        response_text = (
            "Câu hỏi của bạn cần được xử lý bởi chuyên viên tư vấn. "
            "Vui lòng để lại thông tin liên hệ để được hỗ trợ trực tiếp.\n\n"
            + response_text
        )

    # Stream word by word for typewriter effect
    words = response_text.split()
    for i, word in enumerate(words):
        # Add space before words (except first)
        token = (" " + word) if i > 0 else word
        yield _sse({"type": "token", "data": token})
        # Small delay between tokens — gives ~80 words/sec typewriter speed
        await asyncio.sleep(0.015)

    # Signal stream completion
    yield _sse({
        "type": "stream_end",
        "thread_id": thread_id,
        "workflow_status": workflow_status,
    })
