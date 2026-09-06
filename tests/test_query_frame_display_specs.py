from __future__ import annotations

from backend.agent.evidence import build_evidence_ledger
from backend.agent.intent_router import route_intent
from backend.agent.planner import build_plan
from backend.agent.product_facts import normalize_product
from backend.agent.product_resolver import resolve_product_reference
from backend.agent.response_composer import ResponseDraftInput, compose_response
from backend.agent.state import AgentState, CandidateRef, ProductConstraints, QueryFrame
from backend.agent.tools import search_products
from backend.agent.verifier import AdvisorResponseContract, verify_response
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


DELL_RTX_1 = _product(
    "00930011",
    "Dell Gaming G15 i7 RTX 4050",
    "Dell",
    ("CPU Core i7", "Card do hoa NVIDIA RTX 4050", "RAM 16GB", "O cung SSD 512GB", "Man hinh 15.6 inch"),
    price="28.990.000 VND",
)

DELL_RTX_2 = _product(
    "00930012",
    "Dell Gaming G15 i5 RTX 3050",
    "Dell",
    ("CPU Core i5", "Card do hoa NVIDIA RTX 3050", "RAM 16GB", "O cung SSD 512GB", "Man hinh 15.6 inch"),
    price="26.990.000 VND",
)

HP_INTEGRATED = _product(
    "00921548",
    "HP 14-ep1179TU Core 5 120U (C89ZSPA)",
    "HP",
    ("CPU Core 5 120U", "Card do hoa Intel Graphics", "RAM 16GB", "O cung SSD 512GB", "Man hinh 14 inch"),
    price="19.490.000 VND",
)

DELL_RYZEN_FOCUSED = _product(
    "00927403",
    "Dell 14 DC14255 Ryzen AI 5",
    "Dell",
    ("CPU Ryzen AI 5", "Card do hoa AMD Radeon Graphics", "RAM 16GB", "O cung SSD 512GB", "Man hinh 14 inch"),
    price="24.990.000 VND",
)


class MemoryCatalog:
    def __init__(self, products: tuple[CatalogProduct, ...]) -> None:
        self.products = list(products)
        self.by_code = {product.code.casefold(): product for product in products}

    def get(self, code: str) -> CatalogProduct | None:
        return self.by_code.get(code.casefold())


CATALOG = MemoryCatalog((DELL_RTX_1, DELL_RTX_2, HP_INTEGRATED, DELL_RYZEN_FOCUSED))


def test_filtered_ram_ssd_query_displays_requested_specs_first_and_verifies() -> None:
    query = "Laptop Dell RAM 16GB SSD 512GB duoi 30 trieu"
    state = AgentState()
    route = route_intent(query, state)
    plan = build_plan(query, route, state)
    filters = plan.steps[0].args["filters"]
    assert isinstance(filters, ProductConstraints)

    result = search_products(CATALOG, filters)
    requested = route.constraints["requested_attributes"]
    facts = result.facts
    ledger = build_evidence_ledger(list(facts), constraints_checked=route.constraints)
    response = compose_response(
        ResponseDraftInput(
            response_mode="filtered_search_result",
            products=facts,
            evidence_ledger=ledger,
            constraints=filters,
            requested_attributes=requested,
        )
    )

    assert route.intent == "new_filtered_search"
    assert filters.ram_gb == 16
    assert filters.storage_gb == 512
    assert "RAM 16GB" in response.answer_text
    assert "SSD 512GB" in response.answer_text
    assert response.displayed_attributes == ("ram_gb", "storage_gb", "price_value")
    assert any(
        action.type == "SHOW_RELATED_PRODUCTS"
        and action.payload == {"display_specs": requested}
        for action in response.ui_actions
    )

    verifier = verify_response(
        AdvisorResponseContract(
            answer_text=response.answer_text,
            related_product_codes=response.related_product_codes,
            answer_mode=response.answer_mode,
            displayed_attributes=response.displayed_attributes,
        ),
        list(facts),
        ledger,
        constraints=filters,
        requested_attributes=requested,
    )
    assert verifier.passed is True


def test_continuation_inherits_query_frame_and_excludes_previous_candidates() -> None:
    frame = QueryFrame(
        constraints=ProductConstraints(
            category="Laptop",
            brand="Dell",
            max_price=30_000_000,
            gpu_type="dedicated",
        ),
        requested_attributes=("gpu_type", "price_value"),
    )
    state = AgentState(
        query_frame=frame,
        last_constraints=frame.constraints,
        last_shown_candidates=[CandidateRef.from_product(DELL_RTX_1)],
    )

    route = route_intent("con may nao nua khong?", state)
    plan = build_plan("con may nao nua khong?", route, state)
    filters = plan.steps[0].args["filters"]
    exclude_codes = plan.steps[0].args["exclude_codes"]
    result = search_products(CATALOG, filters, exclude_codes=exclude_codes)

    assert route.intent == "query_continuation"
    assert filters.brand == "Dell"
    assert filters.max_price == 30_000_000
    assert filters.gpu_type == "dedicated"
    assert exclude_codes == {DELL_RTX_1.code}
    assert tuple(product.code for product in result.products) == (DELL_RTX_2.code,)
    assert HP_INTEGRATED.code not in tuple(product.code for product in result.products)


def test_brand_scoped_followup_resolves_unique_previous_dell_candidate() -> None:
    state = AgentState(
        last_shown_candidates=[CandidateRef.from_product(DELL_RTX_1), CandidateRef.from_product(HP_INTEGRATED)],
    )
    query = "mau Dell do bao nhieu inch?"

    route = route_intent(query, state)
    resolution = resolve_product_reference(query, state)

    assert route.intent == "focused_product_field_question"
    assert route.field_requested == "screen_inches"
    assert resolution.resolved is True
    assert resolution.code == DELL_RTX_1.code


def test_brand_scoped_field_answer_is_short_not_full_product_dump() -> None:
    state = AgentState(
        focused_product_code=DELL_RTX_1.code,
        focused_product_name=DELL_RTX_1.name,
        last_shown_candidates=[CandidateRef.from_product(DELL_RTX_1)],
    )
    query = "mau Dell do bao nhieu inch?"
    route = route_intent(query, state)
    resolution = resolve_product_reference(query, state)
    product = CATALOG.get(resolution.code or "")
    assert product is not None
    facts = (normalize_product(product),)

    response = compose_response(
        ResponseDraftInput(
            response_mode="focused_product_field_answer",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts), requested_fields=("screen_inches",)),
            focused_product_code=product.code,
            requested_attributes=(route.field_requested or "",),
        )
    )

    assert route.intent == "focused_product_field_question"
    assert response.answer_mode == "focused_product_field_answer"
    assert "15.6 inch" in response.answer_text
    assert "Danh gia nhanh" not in response.answer_text
    assert "Thong tin catalog hien co" not in response.answer_text


def test_new_brand_filter_with_focused_dell_does_not_resolve_to_old_focused_product() -> None:
    state = AgentState(
        focused_product_code=DELL_RYZEN_FOCUSED.code,
        focused_product_name=DELL_RYZEN_FOCUSED.name,
        last_shown_candidates=[CandidateRef.from_product(DELL_RYZEN_FOCUSED)],
    )

    route = route_intent("Dell co card roi duoi 30 trieu khong?", state)
    resolution = resolve_product_reference("Dell co card roi duoi 30 trieu khong?", state)

    assert route.intent == "new_filtered_search"
    assert resolution.resolved is False
    assert resolution.source == "unresolved"


def test_verifier_blocks_dell_i7_answer_if_product_is_ryzen_ai_5() -> None:
    facts = (normalize_product(DELL_RYZEN_FOCUSED),)
    ledger = build_evidence_ledger(list(facts))
    constraints = ProductConstraints(brand="Dell", cpu_tier="i7", max_price=30_000_000)

    verifier = verify_response(
        AdvisorResponseContract(
            answer_text="Dell 14 DC14255 Ryzen AI 5 khop yeu cau Dell i7 duoi 30 trieu.",
            related_product_codes=(DELL_RYZEN_FOCUSED.code,),
            answer_mode="filtered_search_result",
        ),
        list(facts),
        ledger,
        constraints=constraints,
    )

    assert verifier.passed is False
    assert any(failure.code == "constraint_mismatch_cpu_tier" for failure in verifier.failures)
