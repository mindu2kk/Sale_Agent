from __future__ import annotations

from dataclasses import dataclass

from backend.agent.evidence import build_evidence_ledger
from backend.agent.intent_router import IntentRoute, route_intent
from backend.agent.planner import AgentPlan, build_plan
from backend.agent.product_facts import NormalizedProductFacts, normalize_product
from backend.agent.product_resolver import ProductResolution, resolve_product_reference
from backend.agent.response_composer import ResponseDraftInput, AdvisorResponse, compose_response
from backend.agent.state import AgentState, CandidateRef, ProductConstraints
from backend.agent.tools import get_product_field, search_products
from backend.agent.verifier import AdvisorResponseContract, VerifierResult, verify_response
from backend.services.catalog import CatalogProduct


def _product(
    code: str,
    name: str,
    brand: str,
    specs: tuple[str, ...],
    *,
    price: str,
) -> CatalogProduct:
    return CatalogProduct(
        code=code,
        category="Laptop",
        brand=brand,
        price=price,
        context="",
        specs=specs,
        title=name,
    )


HP = _product(
    "00921548",
    "HP 14-ep1179TU Core 5 120U (C89ZSPA)",
    "HP",
    (
        "CPU Core 5 120U",
        "Card do hoa Intel Graphics",
        "RAM 16GB",
        "Dung luong Pin 41Wh",
    ),
    price="19.490.000 VND",
)

DELL_I7 = _product(
    "00927992",
    "Laptop Dell 15 DC15250 i7-1355U (DC5I7952W1)",
    "Dell",
    (
        "CPU Core i7",
        "Card do hoa Intel UHD Graphics",
        "RAM 16GB",
        "O cung SSD 512GB",
    ),
    price="23.790.000 VND",
)

DELL_I5 = _product(
    "00927402",
    "Dell 15 DC15250 i5-1334U (71092479)",
    "Dell",
    (
        "CPU Core i5",
        "Card do hoa Intel UHD Graphics",
        "RAM 16GB",
    ),
    price="19.990.000 VND",
)

DELL_RTX_OVER_BUDGET = _product(
    "00929999",
    "Dell G15 i7 RTX 4050",
    "Dell",
    (
        "CPU Core i7",
        "Card do hoa NVIDIA RTX 4050",
        "RAM 16GB",
        "O cung SSD 512GB",
    ),
    price="32.990.000 VND",
)

ACER_RTX = _product(
    "00930001",
    "Acer Aspire Gaming 15 i5 RTX 2050",
    "Acer",
    (
        "CPU Core i5",
        "Card do hoa NVIDIA RTX 2050",
        "RAM 16GB",
        "O cung SSD 512GB",
    ),
    price="18.990.000 VND",
)


class MemoryCatalog:
    def __init__(self, products: tuple[CatalogProduct, ...]) -> None:
        self.products = list(products)
        self.by_code = {product.code.casefold(): product for product in products}

    def get(self, code: str) -> CatalogProduct | None:
        return self.by_code.get(code.casefold())


CATALOG = MemoryCatalog((HP, DELL_I7, DELL_I5, DELL_RTX_OVER_BUDGET, ACER_RTX))


@dataclass(frozen=True)
class TurnResult:
    route: IntentRoute
    resolution: ProductResolution
    plan: AgentPlan
    response: AdvisorResponse
    verifier: VerifierResult
    facts: tuple[NormalizedProductFacts, ...]


def _run_turn(query: str, state: AgentState) -> TurnResult:
    route = route_intent(query, state)
    resolution = resolve_product_reference(query, state)
    plan = build_plan(query, route, state)
    facts: tuple[NormalizedProductFacts, ...] = ()
    constraints: ProductConstraints | None = None
    asked_field = route.field_requested

    if plan.plan_type == "filtered_search":
        constraints = plan.steps[0].args["filters"]
        assert isinstance(constraints, ProductConstraints)
        search_result = search_products(CATALOG, constraints)
        facts = search_result.facts
        response_mode = "filtered_search_result" if facts else "no_result"
        ledger = build_evidence_ledger(list(facts), constraints_checked=route.constraints)
        response = compose_response(
            ResponseDraftInput(
                response_mode=response_mode,
                products=facts,
                evidence_ledger=ledger,
                constraints=constraints,
                alternative_brands=("MSI", "Asus", "Acer"),
            )
        )
    elif plan.plan_type == "field_lookup":
        code = state.focused_product_code
        assert code is not None
        product = CATALOG.get(code)
        assert product is not None
        field = route.field_requested
        assert field is not None
        field_result = get_product_field(CATALOG, code, field)
        facts = (normalize_product(product),)
        ledger = build_evidence_ledger(list(facts), requested_fields=(field,))
        response = compose_response(
            ResponseDraftInput(
                response_mode="missing_field" if field_result.missing else "focused_product_detail",
                products=facts,
                evidence_ledger=ledger,
                missing_fields=(field,) if field_result.missing else (),
                focused_product_code=code,
            )
        )
    elif plan.plan_type == "product_detail" and resolution.resolved:
        product = CATALOG.get(resolution.code or "")
        assert product is not None
        facts = (normalize_product(product),)
        ledger = build_evidence_ledger(list(facts))
        response = compose_response(
            ResponseDraftInput(
                response_mode="focused_product_detail",
                products=facts,
                evidence_ledger=ledger,
                focused_product_code=product.code,
            )
        )
    elif plan.plan_type == "comparison":
        products = (DELL_I7, ACER_RTX)
        facts = tuple(normalize_product(product) for product in products)
        ledger = build_evidence_ledger(list(facts))
        response = compose_response(
            ResponseDraftInput(
                response_mode="comparison",
                products=facts,
                evidence_ledger=ledger,
            )
        )
    elif plan.plan_type == "strong_claim_ranking":
        candidates = state.last_shown_candidates or [CandidateRef.from_product(HP), CandidateRef.from_product(DELL_I7)]
        facts = tuple(
            normalize_product(product)
            for candidate in candidates
            if (product := CATALOG.get(candidate.code)) is not None
        )
        ledger = build_evidence_ledger(list(facts))
        response = compose_response(
            ResponseDraftInput(
                response_mode="tradeoff",
                products=facts,
                evidence_ledger=ledger,
            )
        )
    else:
        ledger = build_evidence_ledger([])
        response = compose_response(ResponseDraftInput(response_mode="no_result"))

    verifier = verify_response(
        AdvisorResponseContract(
            answer_text=response.answer_text,
            related_product_codes=response.related_product_codes,
            answer_mode=response.answer_mode,
            missing_fields=response.missing_fields,
        ),
        list(facts),
        ledger,
        constraints=constraints,
        asked_field=asked_field,
        focused_product_code=state.focused_product_code if plan.plan_type == "field_lookup" else None,
    )
    _commit_state(state, route, response, facts, resolution)
    return TurnResult(route, resolution, plan, response, verifier, facts)


def _commit_state(
    state: AgentState,
    route: IntentRoute,
    response: AdvisorResponse,
    facts: tuple[NormalizedProductFacts, ...],
    resolution: ProductResolution,
) -> None:
    state.last_intent = route.intent
    if response.related_product_codes:
        candidates = [
            CandidateRef(
                code=fact.code,
                name=fact.name,
                brand=fact.brand,
                category=fact.category,
                price_value=fact.price_value,
            )
            for fact in facts
        ]
        state.remember_candidates(candidates)
    focus_code = resolution.code or (response.related_product_codes[0] if len(response.related_product_codes) == 1 else None)
    if focus_code:
        focused = next((candidate for candidate in state.last_shown_candidates if candidate.code == focus_code), None)
        if focused:
            state.set_focus(focused)


def test_realistic_multiturn_sales_flow_keeps_focus_and_filter_boundaries() -> None:
    state = AgentState(
        active_category="Laptop",
        last_shown_candidates=[
            CandidateRef.from_product(HP),
            CandidateRef.from_product(DELL_I5),
            CandidateRef.from_product(ACER_RTX),
        ],
    )

    no_result = _run_turn("Dell duoi 30 trieu co card roi khong?", state)
    assert no_result.route.intent == "new_filtered_search"
    assert no_result.plan.plan_type == "filtered_search"
    assert no_result.response.answer_mode == "no_result"
    assert no_result.response.related_product_codes == ()
    assert "HP 14-ep1179TU" not in no_result.response.answer_text

    selected_hp = _run_turn("HP 14-ep1179TU di", state)
    assert selected_hp.resolution.resolved is True
    assert selected_hp.resolution.code == HP.code
    assert state.focused_product_code == HP.code
    assert selected_hp.response.related_product_codes == (HP.code,)
    assert "Dell 15" not in selected_hp.response.answer_text

    missing_weight = _run_turn("may do nang bao kg?", state)
    assert missing_weight.route.intent == "focused_product_field_question"
    assert missing_weight.plan.plan_type == "field_lookup"
    assert missing_weight.response.answer_mode == "missing_field"
    assert missing_weight.response.related_product_codes == (HP.code,)
    assert "chưa có dữ liệu trọng lượng" in missing_weight.response.answer_text
    assert missing_weight.verifier.passed is True

    dell_i7 = _run_turn("co Dell i7 duoi 30 trieu khong?", state)
    assert dell_i7.route.intent == "new_filtered_search"
    assert dell_i7.response.answer_mode == "filtered_search_result"
    assert dell_i7.response.related_product_codes == (DELL_I7.code,)
    assert DELL_I5.code not in dell_i7.response.related_product_codes
    assert dell_i7.verifier.passed is True

    comparison = _run_turn("So sanh Dell i7 voi Acer cung tam gia", state)
    assert comparison.route.intent == "comparison"
    assert comparison.plan.plan_type == "comparison"
    assert comparison.response.answer_mode == "comparison"
    assert comparison.response.related_product_codes == (DELL_I7.code, ACER_RTX.code)

    durable = _run_turn("Con nao ben nhat?", state)
    assert durable.route.intent == "strong_claim_question"
    assert "ben nhat" not in durable.response.answer_text.lower()
    assert durable.verifier.passed is True
