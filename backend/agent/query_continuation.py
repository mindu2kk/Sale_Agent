"""Query-continuation helpers for inheriting filters safely."""

from __future__ import annotations

from backend.agent.intent_router import IntentRoute
from backend.agent.query_frame import build_query_frame, continuation_exclude_codes, is_query_continuation
from backend.agent.state import AgentState, QueryFrame


def build_continuation_frame(route: IntentRoute, state: AgentState) -> QueryFrame:
    """Build a QueryFrame that carries the previous filter and excludes shown products."""

    return build_query_frame(route, state)


def continuation_excluded_codes(route: IntentRoute, state: AgentState) -> tuple[str, ...]:
    """Return product codes that a continuation must not show again."""

    return tuple(sorted(continuation_exclude_codes(route, state)))


__all__ = [
    "build_continuation_frame",
    "continuation_excluded_codes",
    "is_query_continuation",
]
