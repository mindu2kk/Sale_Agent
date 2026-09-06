from __future__ import annotations

from backend.agent.intent_router import route_intent
from backend.agent.state import AgentState, CandidateRef


def _focused_state() -> AgentState:
    return AgentState(
        active_category="Laptop",
        focused_product_code="00921548",
        focused_product_name="HP 14-ep1179TU Core 5 120U (C89ZSPA)",
    )


def test_dell_i7_under_30m_routes_to_new_filtered_search() -> None:
    route = route_intent("Dell i7 dưới 30 triệu", _focused_state())

    assert route.intent == "new_filtered_search"
    assert route.has_new_constraints is True
    assert route.constraints["brand"] == "Dell"
    assert route.constraints["cpu_tier"] == "i7"
    assert route.constraints["max_price"] == 30_000_000


def test_focused_weight_question_routes_to_focused_product_field_question() -> None:
    route = route_intent("máy đó nặng bao kg", _focused_state())

    assert route.intent == "focused_product_field_question"
    assert route.field_requested == "weight_kg"
    assert route.has_new_constraints is False


def test_strong_claim_question_routes_to_evidence_aware_intent() -> None:
    route = route_intent("con nào bền nhất", _focused_state())

    assert route.intent == "strong_claim_question"
    assert route.field_requested is None


def test_two_brand_question_routes_to_comparison() -> None:
    route = route_intent("so sánh Dell với Acer cùng tầm giá", _focused_state())

    assert route.intent == "comparison"
    assert route.constraints["brands"] == ("Dell", "Acer")


def test_correction_beats_new_constraints_but_preserves_constraints() -> None:
    route = route_intent("không phải con đó, Dell i7 cơ mà", _focused_state())

    assert route.intent == "correction"
    assert route.is_correction is True
    assert route.has_new_constraints is True
    assert route.constraints["brand"] == "Dell"
    assert route.constraints["cpu_tier"] == "i7"


def test_product_selection_routes_to_selection() -> None:
    route = route_intent("con thứ 2 đi", _focused_state())

    assert route.intent == "product_selection"


def test_hardware_explanation_without_product_context_routes_to_explanation() -> None:
    route = route_intent("card rời là gì", AgentState())

    assert route.intent == "hardware_explanation"


def test_compare_previous_two_models_beats_selection_suffix() -> None:
    state = AgentState(
        active_category="Laptop",
        last_shown_candidates=[
            CandidateRef(
                code="00927778",
                name="Dell Pro 15 Essential PV15250 Core 3-100U (VKVKD)",
                brand="Dell",
                category="Laptop",
            ),
            CandidateRef(
                code="00927423",
                name="Dell 15 DC15255 R7-7730U_884116430117",
                brand="Dell",
                category="Laptop",
            ),
        ],
    )

    route = route_intent("so sanh 2 mau nay di", state)

    assert route.intent == "comparison"


def test_top_two_notable_followup_routes_to_comparison() -> None:
    state = AgentState(
        active_category="Laptop",
        last_shown_candidates=[
            CandidateRef(code="00927778", name="Dell A", brand="Dell", category="Laptop"),
            CandidateRef(code="00927423", name="Dell B", brand="Dell", category="Laptop"),
        ],
    )

    assert route_intent("2 mau dang chu y nay di", state).intent == "comparison"
    assert route_intent("2 mau dell vua hoi co ma", state).intent == "comparison"
