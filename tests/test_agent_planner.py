from __future__ import annotations

from backend.agent.intent_router import route_intent
from backend.agent.planner import build_plan
from backend.agent.state import AgentState


def _focused_state() -> AgentState:
    return AgentState(
        active_category="Laptop",
        focused_product_code="00921548",
        focused_product_name="HP 14-ep1179TU Core 5 120U (C89ZSPA)",
    )


def test_focused_field_question_plans_get_product_field_not_search() -> None:
    state = _focused_state()
    query = "máy đó nặng bao kg"
    route = route_intent(query, state)
    plan = build_plan(query, route, state)

    assert plan.plan_type == "field_lookup"
    assert [step.tool for step in plan.steps] == [
        "get_product_by_code",
        "get_product_field",
    ]
    assert plan.steps[1].args == {
        "code": "00921548",
        "field": "weight_kg",
    }


def test_dell_under_30m_dedicated_gpu_plans_filtered_search() -> None:
    state = _focused_state()
    query = "Dell dưới 30 triệu có card rời không"
    route = route_intent(query, state)
    plan = build_plan(query, route, state)

    assert plan.plan_type == "filtered_search"
    assert [step.tool for step in plan.steps] == ["search_products"]
    filters = plan.steps[0].args["filters"]
    assert filters.brand == "Dell"
    assert filters.max_price == 30_000_000
    assert filters.gpu_type == "dedicated"


def test_correction_with_new_constraints_plans_filtered_search() -> None:
    state = _focused_state()
    query = "không phải con đó, Dell i7 cơ mà"
    route = route_intent(query, state)
    plan = build_plan(query, route, state)

    assert route.intent == "correction"
    assert plan.plan_type == "filtered_search"
    filters = plan.steps[0].args["filters"]
    assert filters.brand == "Dell"
    assert filters.cpu_tier == "i7"


def test_comparison_plans_search_then_compare() -> None:
    state = _focused_state()
    query = "So sánh Dell với Acer cùng tầm giá"
    route = route_intent(query, state)
    plan = build_plan(query, route, state)

    assert plan.plan_type == "comparison"
    assert [step.tool for step in plan.steps] == ["search_products", "compare_products"]


def test_strong_claim_question_plans_evidence_aware_ranking() -> None:
    state = _focused_state()
    query = "Con nào bền nhất?"
    route = route_intent(query, state)
    plan = build_plan(query, route, state)

    assert plan.plan_type == "strong_claim_ranking"
    assert [step.tool for step in plan.steps] == ["search_products", "rank_products"]
    assert plan.steps[1].args["objective"] == "durability"


def test_focused_field_question_without_focus_clarifies() -> None:
    query = "máy đó nặng bao kg"
    state = AgentState()
    route = route_intent(query, state)
    plan = build_plan(query, route, state)

    assert plan.plan_type == "clarify"
    assert plan.requires_focused_product is True
