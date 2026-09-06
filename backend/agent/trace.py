"""Structured trace payloads for deterministic advisor turns."""

from __future__ import annotations

from backend.agent.evidence_confidence import EvidenceConfidence, confidence_summary


def build_agent_trace(
    *,
    user_text: str,
    intent: str,
    query_frame: dict,
    product_resolution: dict | None,
    tool_call: dict,
    tool_result_summary: dict,
    display_specs: dict,
    evidence_confidence: tuple[EvidenceConfidence, ...],
    verifier_result: dict,
    response_mode: str,
    related_product_codes: list[str],
    state_after: dict,
    latency_ms: float,
) -> dict:
    return {
        "user_text": user_text,
        "intent": intent,
        "query_frame": query_frame,
        "product_resolution": product_resolution or {},
        "tool_call": tool_call,
        "tool_result_summary": tool_result_summary,
        "display_specs": display_specs,
        "evidence_confidence": confidence_summary(evidence_confidence),
        "verifier_result": verifier_result,
        "response_mode": response_mode,
        "related_product_codes": related_product_codes,
        "state_after": state_after,
        "latency_ms": latency_ms,
        "trace_collector": {
            "terminalEvent": {
                "status": "succeeded",
                "message": "domain_contract_response_completed",
            },
            "phases": [
                "perceive",
                "query_frame",
                "tool",
                "evidence",
                "compose",
                "verify",
                "commit",
            ]
        },
    }
