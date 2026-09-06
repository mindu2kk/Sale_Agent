"""Schema planner that selects deterministic agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.agent.intent_router import IntentRoute
from backend.agent.query_frame import constraints_from_route, continuation_exclude_codes
from backend.agent.state import AgentState, ProductConstraints


PlanType = Literal[
    "field_lookup",
    "filtered_search",
    "product_detail",
    "comparison",
    "strong_claim_ranking",
    "hardware_explanation",
    "clarify",
]


@dataclass(frozen=True)
class PlanStep:
    tool: str
    args: dict[str, object]


@dataclass(frozen=True)
class AgentPlan:
    plan_type: PlanType
    steps: tuple[PlanStep, ...]
    reason: str
    requires_focused_product: bool = False


def build_plan(
    user_query: str,
    route: IntentRoute,
    state: AgentState,
) -> AgentPlan:
    if route.intent == "focused_product_field_question":
        if not state.focused_product_code:
            return AgentPlan(
                plan_type="clarify",
                steps=(),
                reason="Focused field question needs a focused product.",
                requires_focused_product=True,
            )
        return AgentPlan(
            plan_type="field_lookup",
            steps=(
                PlanStep("get_product_by_code", {"code": state.focused_product_code}),
                PlanStep(
                    "get_product_field",
                    {
                        "code": state.focused_product_code,
                        "field": route.field_requested,
                    },
                ),
            ),
            reason="Focused product field lookup must not broad search.",
            requires_focused_product=True,
        )

    if route.intent in {"new_filtered_search", "query_continuation", "correction"} and route.has_new_constraints:
        args: dict[str, object] = {"filters": constraints_from_route(route)}
        if route.constraints.get("exclude_previous") is True:
            args["exclude_codes"] = continuation_exclude_codes(route, state)
        return AgentPlan(
            plan_type="filtered_search",
            steps=(
                PlanStep(
                    "search_products",
                    args,
                ),
            ),
            reason="New constraints require schema-filtered search.",
        )

    if route.intent == "comparison":
        return AgentPlan(
            plan_type="comparison",
            steps=(
                PlanStep(
                    "search_products",
                    {"filters": constraints_from_route(route)},
                ),
                PlanStep("compare_products", {"objective": route.field_requested}),
            ),
            reason="Comparison needs candidate resolution before compare.",
        )

    if route.intent == "strong_claim_question":
        return AgentPlan(
            plan_type="strong_claim_ranking",
            steps=(
                PlanStep(
                    "search_products",
                    {"filters": constraints_from_route(route)},
                ),
                PlanStep(
                    "rank_products",
                    {"objective": route.field_requested or _ranking_objective(user_query)},
                ),
            ),
            reason="Strong claim requires evidence-aware ranking.",
        )

    if route.intent in {"product_selection", "product_detail"}:
        return AgentPlan(
            plan_type="product_detail",
            steps=(),
            reason="Product detail depends on resolver-selected product.",
        )

    if route.intent == "hardware_explanation":
        return AgentPlan(
            plan_type="hardware_explanation",
            steps=(),
            reason="Hardware explanation does not need product retrieval.",
        )

    return AgentPlan(
        plan_type="clarify",
        steps=(),
        reason="No executable deterministic plan matched.",
    )


def _constraints_from_route(route: IntentRoute) -> ProductConstraints:
    return constraints_from_route(route)


def _ranking_objective(user_query: str) -> str:
    normalized = user_query.casefold()
    if "bền" in normalized or "ben" in normalized:
        return "durability"
    if "pin" in normalized:
        return "battery"
    if "mạnh" in normalized or "khỏe" in normalized or "manh" in normalized or "khoe" in normalized:
        return "performance"
    if "rẻ" in normalized or "re" in normalized:
        return "price"
    return "overall"
