"""Clarification policy for vague commerce requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.agent.intent_router import IntentRoute


ClarificationLevel = Literal["clear", "assume_and_search", "ask_clarifying_question"]


@dataclass(frozen=True)
class ClarificationDecision:
    level: ClarificationLevel
    question: str | None = None
    assumption: str | None = None


def decide_clarification(route: IntentRoute, user_text: str) -> ClarificationDecision:
    if route.has_new_constraints:
        return ClarificationDecision("clear")
    normalized = user_text.casefold()
    if any(term in normalized for term in ("hiệu năng", "hieu nang", "giá thành", "gia thanh", "đáng tiền", "dang tien")):
        return ClarificationDecision("clear")
    if "văn phòng" in normalized or "van phong" in normalized or "tầm" in normalized or "tam" in normalized:
        return ClarificationDecision("assume_and_search", assumption="Mình sẽ giả định bạn đang tìm laptop theo nhu cầu phổ thông.")
    if route.intent in {"unknown", "broad_consulting"}:
        return ClarificationDecision(
            "ask_clarifying_question",
            question="Bạn đang tìm laptop hay điện thoại, ưu tiên học tập/văn phòng, gaming hay đồ họa, và ngân sách khoảng bao nhiêu?",
        )
    return ClarificationDecision("clear")
