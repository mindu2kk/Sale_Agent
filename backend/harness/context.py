"""Context lifecycle management for the advisor harness.

The harness keeps durable decisions and catalog references, while aggressively
discarding reconstructible assistant prose. This prevents long chat history
from becoming an accidental source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from backend.services.catalog import CatalogService
from backend.services.conversation import DecisionContext, normalize_text


DECISION_TERMS = (
    "uu tien",
    "ngan sach",
    "tam gia",
    "khong chon",
    "loai",
    "can",
    "muon",
    "choi game",
    "do ben",
    "hieu nang",
    "pin",
    "mong nhe",
)


@dataclass(frozen=True)
class ContextSlice:
    history: tuple[Any, ...]
    product_codes: tuple[str, ...]
    user_decisions: tuple[str, ...]
    input_turns: int
    retained_turns: int
    dropped_turns: int
    compacted: bool

    def public_summary(self) -> dict[str, Any]:
        return {
            "input_turns": self.input_turns,
            "retained_turns": self.retained_turns,
            "dropped_turns": self.dropped_turns,
            "compacted": self.compacted,
            "product_codes": list(self.product_codes),
            "user_decisions": list(self.user_decisions),
        }


class ContextLifecycleManager:
    """Build a bounded working context from client state and recent turns."""

    def __init__(self, *, max_turns: int = 8, max_text_chars: int = 3_200) -> None:
        self.max_turns = max_turns
        self.max_text_chars = max_text_chars

    def prepare(
        self,
        history: Iterable[Any],
        state: DecisionContext,
        catalog: CatalogService,
    ) -> ContextSlice:
        turns = list(history)
        retained: list[Any] = []
        total_chars = 0

        # Newest turns have the highest operational value. Assistant prose is
        # retained only while bounded; product IDs remain separately durable.
        for turn in reversed(turns):
            text = str(getattr(turn, "text", "") or "")
            if len(retained) >= self.max_turns:
                break
            if total_chars + len(text) > self.max_text_chars and retained:
                continue
            retained.append(turn)
            total_chars += len(text)
        retained.reverse()

        codes: list[str] = []
        for code in (
            [state.active_product_code] if state.active_product_code else []
        ) + state.compared_codes + state.candidate_codes:
            if code and catalog.get(code) is not None and code not in codes:
                codes.append(code)
        for turn in retained:
            for code in getattr(turn, "product_codes", []) or []:
                if catalog.get(code) is not None and code not in codes:
                    codes.append(code)
            text = str(getattr(turn, "text", "") or "").upper()
            for code in re.findall(r"\b[A-Z0-9]{8}\b", text):
                if catalog.get(code) is not None and code not in codes:
                    codes.append(code)

        decisions: list[str] = []
        for turn in retained:
            if getattr(turn, "role", "") != "user":
                continue
            text = str(getattr(turn, "text", "") or "").strip()
            normalized = normalize_text(text)
            if any(term in normalized for term in DECISION_TERMS):
                decisions.append(text[:280])

        return ContextSlice(
            history=tuple(retained),
            product_codes=tuple(codes[:20]),
            user_decisions=tuple(decisions[-4:]),
            input_turns=len(turns),
            retained_turns=len(retained),
            dropped_turns=max(0, len(turns) - len(retained)),
            compacted=len(retained) < len(turns),
        )
