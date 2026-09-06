from __future__ import annotations

from backend.agent.intent_router import route_intent
from backend.agent.query_frame import build_query_frame
from backend.agent.product_resolver import resolve_product_reference
from backend.agent.state import AgentState, CandidateRef, ProductConstraints, QueryFrame
from backend.agent.verifier import AdvisorResponseContract, verify_response
from backend.agent.evidence import build_evidence_ledger
from backend.agent.product_facts import normalize_product
from backend.services.catalog import CatalogProduct


def _product(code: str, name: str, brand: str, specs: tuple[str, ...]) -> CatalogProduct:
    return CatalogProduct(
        code=code,
        category="Laptop",
        brand=brand,
        price="19.990.000 VND",
        context="",
        specs=specs,
        title=name,
    )


HP = _product(
    "00921548",
    "HP 14-ep1179TU Core 5 120U (C89ZSPA)",
    "HP",
    ("CPU Core 5 120U", "Card do hoa Intel Graphics", "RAM 16GB"),
)

DELL_I5 = _product(
    "00927402",
    "Dell 15 DC15250 i5-1334U (71092479)",
    "Dell",
    ("CPU Core i5", "Card do hoa Intel UHD Graphics", "RAM 16GB"),
)

DELL_I7 = _product(
    "00927992",
    "Laptop Dell 15 DC15250 i7-1355U (DC5I7952W1)",
    "Dell",
    ("CPU Core i7", "Card do hoa Intel UHD Graphics", "RAM 16GB"),
)


def test_new_filtered_search_does_not_resolve_unique_previous_brand_candidate() -> None:
    state = AgentState(
        focused_product_code=HP.code,
        focused_product_name=HP.name,
        last_shown_candidates=[CandidateRef.from_product(HP), CandidateRef.from_product(DELL_I5)],
    )

    route = route_intent("co Dell i7 duoi 30 trieu khong?", state)
    resolution = resolve_product_reference("co Dell i7 duoi 30 trieu khong?", state)

    assert route.intent == "new_filtered_search"
    assert resolution.resolved is False
    assert resolution.source == "unresolved"


def test_explicit_comparison_with_constraints_is_not_downgraded_to_search() -> None:
    state = AgentState(last_shown_candidates=[CandidateRef.from_product(DELL_I5)])

    route = route_intent("So sanh Dell i7 voi Acer cung tam gia", state)

    assert route.intent == "comparison"


def test_verifier_blocks_answer_text_and_card_drift_regression() -> None:
    hp_facts = normalize_product(HP)
    dell_facts = normalize_product(DELL_I7)
    ledger = build_evidence_ledger([hp_facts, dell_facts])

    result = verify_response(
        AdvisorResponseContract(
            answer_text="Minh dang noi ve HP 14-ep1179TU Core 5 120U (C89ZSPA).",
            related_product_codes=(DELL_I7.code,),
            answer_mode="filtered_search_result",
        ),
        [hp_facts, dell_facts],
        ledger,
    )

    assert result.passed is False
    assert any(failure.code == "answer_cards_mismatch" for failure in result.failures)


def test_brand_refinement_inherits_previous_dedicated_gpu_constraint() -> None:
    state = AgentState(
        active_category="Laptop",
        query_frame=QueryFrame(
            intent="new_filtered_search",
            constraints=ProductConstraints(
                category="Laptop",
                max_price=30_000_000,
                gpu_type="dedicated",
            ),
            requested_attributes=("gpu_type", "price_value"),
        ),
    )

    route = route_intent("Co laptop Lenovo nao duoi 30 trieu khong?", state)
    frame = build_query_frame(route, state)

    assert route.intent == "new_filtered_search"
    assert frame.inherit_from_last_query_frame is True
    assert frame.constraints.category == "Laptop"
    assert frame.constraints.brand == "Lenovo"
    assert frame.constraints.max_price == 30_000_000
    assert frame.constraints.gpu_type == "dedicated"
    assert "gpu_type" in frame.requested_attributes


def test_use_case_refinement_inherits_previous_gpu_and_category() -> None:
    state = AgentState(
        active_category="Laptop",
        query_frame=QueryFrame(
            intent="new_filtered_search",
            constraints=ProductConstraints(
                category="Laptop",
                max_price=30_000_000,
                gpu_type="dedicated",
            ),
        ),
    )

    route = route_intent("Phan tich xem may nao phu hop cho tac vu choi game", state)
    frame = build_query_frame(route, state)

    assert route.intent == "new_filtered_search"
    assert frame.inherit_from_last_query_frame is True
    assert frame.constraints.category == "Laptop"
    assert frame.constraints.max_price == 30_000_000
    assert frame.constraints.gpu_type == "dedicated"
    assert frame.constraints.use_case == "gaming"
