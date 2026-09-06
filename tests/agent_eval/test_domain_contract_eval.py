from __future__ import annotations

from backend.agent.domain_contract import DOMAIN_RULES, check_domain_contract
from backend.agent.evidence import build_evidence_ledger
from backend.agent.intent_router import route_intent
from backend.agent.planner import build_plan
from backend.agent.product_facts import normalize_product
from backend.agent.query_frame import build_query_frame
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


DELL_RTX = _product(
    "00930011",
    "Dell Gaming G15 i7 RTX 4050",
    "Dell",
    ("CPU Core i7", "Card do hoa NVIDIA RTX 4050", "RAM 16GB", "O cung SSD 512GB", "Man hinh 15.6 inch"),
    price="28.990.000 VND",
)

DELL_RTX_OTHER = _product(
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


class MemoryCatalog:
    def __init__(self, products: tuple[CatalogProduct, ...]) -> None:
        self.products = list(products)
        self.by_code = {product.code.casefold(): product for product in products}

    def get(self, code: str) -> CatalogProduct | None:
        return self.by_code.get(code.casefold())


CATALOG = MemoryCatalog((DELL_RTX, DELL_RTX_OTHER, HP_INTEGRATED))


def test_domain_contract_declares_the_core_aura_rules() -> None:
    assert tuple(rule.rule_id for rule in DOMAIN_RULES) == tuple(f"R{index}" for index in range(1, 11))
    assert any("Requested attributes" in rule.summary for rule in DOMAIN_RULES)
    assert any("Fresh brand/category/CPU/GPU/RAM/SSD/budget constraints" in rule.summary for rule in DOMAIN_RULES)


def test_eval_ram_ssd_transcript_exposes_matching_facts_in_answer_and_cards() -> None:
    state = AgentState()
    route = route_intent("Co laptop nao RAM 16GB SSD 512GB khong?", state)
    frame = build_query_frame(route, state)
    plan = build_plan("Co laptop nao RAM 16GB SSD 512GB khong?", route, state)
    search_result = search_products(CATALOG, frame.constraints)
    facts = search_result.facts
    response = compose_response(
        ResponseDraftInput(
            response_mode="filtered_search_result",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts)),
            constraints=frame.constraints,
            requested_attributes=frame.requested_attributes,
        )
    )

    assert route.intent == "new_filtered_search"
    assert plan.plan_type == "filtered_search"
    assert frame.constraints.ram_gb == 16
    assert frame.constraints.storage_gb == 512
    assert frame.requested_attributes[:2] == ("ram_gb", "storage_gb")
    assert "RAM 16GB" in response.answer_text
    assert "SSD 512GB" in response.answer_text
    assert "Mình tìm thấy" in response.answer_text
    assert "Nhận định nhanh" in response.answer_text or "Nếu ưu tiên giá thấp" in response.answer_text
    assert response.related_products[0].matching_facts[:2] == ("RAM 16GB", "SSD 512GB")

    contract = check_domain_contract(
        route=route,
        query_frame=frame,
        response=response,
        products=facts,
    )
    assert contract.passed is True


def test_eval_continuation_inherits_query_frame_and_excludes_previous_products() -> None:
    previous_frame = QueryFrame(
        constraints=ProductConstraints(
            category="Laptop",
            brand="Dell",
            gpu_type="dedicated",
            max_price=30_000_000,
        ),
        requested_attributes=("gpu_type", "price_value"),
    )
    state = AgentState(
        query_frame=previous_frame,
        last_constraints=previous_frame.constraints,
        last_shown_candidates=[CandidateRef.from_product(DELL_RTX)],
    )

    route = route_intent("Con may Dell nao khac khong?", state)
    frame = build_query_frame(route, state)
    plan = build_plan("Con may Dell nao khac khong?", route, state)
    exclude_codes = plan.steps[0].args["exclude_codes"]
    search_result = search_products(CATALOG, frame.constraints, exclude_codes=exclude_codes)
    facts = search_result.facts
    response = compose_response(
        ResponseDraftInput(
            response_mode="filtered_search_result",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts)),
            constraints=frame.constraints,
            requested_attributes=frame.requested_attributes,
        )
    )

    assert route.intent == "query_continuation"
    assert route.constraints["exclude_previous"] is True
    assert frame.intent == "query_continuation"
    assert frame.inherit_from_last_query_frame is True
    assert frame.exclude_product_codes == (DELL_RTX.code,)
    assert frame.constraints.brand == "Dell"
    assert frame.constraints.gpu_type == "dedicated"
    assert frame.constraints.max_price == 30_000_000
    assert exclude_codes == {DELL_RTX.code}
    assert response.related_product_codes == (DELL_RTX_OTHER.code,)

    contract = check_domain_contract(
        route=route,
        query_frame=frame,
        response=response,
        products=facts,
        exclude_codes=exclude_codes,
    )
    assert contract.passed is True


def test_eval_new_constraint_overrides_focused_product() -> None:
    state = AgentState(
        focused_product_code=HP_INTEGRATED.code,
        focused_product_name=HP_INTEGRATED.name,
        last_shown_candidates=[CandidateRef.from_product(HP_INTEGRATED)],
    )

    route = route_intent("Co laptop Dell nao co card roi duoi 30 trieu khong?", state)
    frame = build_query_frame(route, state)
    result = search_products(CATALOG, frame.constraints)

    assert route.intent == "new_filtered_search"
    assert frame.constraints.brand == "Dell"
    assert frame.constraints.category == "Laptop"
    assert frame.constraints.gpu_type == "dedicated"
    assert frame.constraints.max_price == 30_000_000
    assert HP_INTEGRATED.code not in tuple(product.code for product in result.products)
    assert all(fact.brand == "Dell" and fact.gpu_type == "dedicated" for fact in result.facts)


def test_eval_field_question_is_concise() -> None:
    state = AgentState(
        focused_product_code=DELL_RTX.code,
        focused_product_name=DELL_RTX.name,
        last_shown_candidates=[CandidateRef.from_product(DELL_RTX)],
    )

    route = route_intent("Mau Dell do bao nhieu inch?", state)
    facts = (normalize_product(DELL_RTX),)
    response = compose_response(
        ResponseDraftInput(
            response_mode="focused_product_field_answer",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts), requested_fields=("screen_inches",)),
            focused_product_code=DELL_RTX.code,
            requested_attributes=(route.field_requested or "",),
        )
    )

    assert route.intent == "focused_product_field_question"
    assert route.field_requested == "screen_inches"
    assert response.answer_mode == "focused_product_field_answer"
    assert "15.6 inch" in response.answer_text
    assert "Đánh giá nhanh" not in response.answer_text
    assert "Thông tin catalog hiện có" not in response.answer_text


def test_eval_strict_i7_filter_rejects_non_i7_products() -> None:
    state = AgentState()
    route = route_intent("Co laptop Dell i7 duoi 30 trieu khong?", state)
    frame = build_query_frame(route, state)
    result = search_products(CATALOG, frame.constraints)

    assert route.intent == "new_filtered_search"
    assert frame.constraints.brand == "Dell"
    assert frame.constraints.cpu_tier == "i7"
    assert frame.constraints.max_price == 30_000_000
    assert result.products == (DELL_RTX,)
    assert all(fact.cpu_tier == "i7" for fact in result.facts)


def test_eval_missing_weight_field_does_not_broad_search_or_hallucinate() -> None:
    state = AgentState(
        focused_product_code=DELL_RTX.code,
        focused_product_name=DELL_RTX.name,
        last_shown_candidates=[CandidateRef.from_product(DELL_RTX)],
    )

    route = route_intent("May do nang bao kg?", state)
    facts = (normalize_product(DELL_RTX),)
    response = compose_response(
        ResponseDraftInput(
            response_mode="missing_field",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts), requested_fields=("weight_kg",)),
            missing_fields=("weight_kg",),
            focused_product_code=DELL_RTX.code,
            requested_attributes=(route.field_requested or "",),
        )
    )

    assert route.intent == "focused_product_field_question"
    assert response.answer_mode == "missing_field"
    assert response.related_product_codes == (DELL_RTX.code,)
    assert "Catalog hiện chưa có dữ liệu trọng lượng" in response.answer_text


def test_eval_vietnamese_output_has_diacritics_and_verifies() -> None:
    route = route_intent("Có laptop nào RAM 16GB SSD 512GB không?", AgentState())
    frame = build_query_frame(route, AgentState())
    result = search_products(CATALOG, frame.constraints)
    facts = result.facts
    response = compose_response(
        ResponseDraftInput(
            response_mode="filtered_search_result",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts)),
            constraints=frame.constraints,
            requested_attributes=frame.requested_attributes,
        )
    )

    assert "Mình tìm thấy" in response.answer_text
    assert "Giá" in response.answer_text
    assert "VNĐ" in response.answer_text
    forbidden = ("Minh tim thay", "Gia ", "Neu muon", "man hinh", "bo loc", " VND")
    assert not any(item in response.answer_text for item in forbidden)

    verifier = verify_response(
        AdvisorResponseContract(
            answer_text=response.answer_text,
            related_product_codes=response.related_product_codes,
            answer_mode=response.answer_mode,
            displayed_attributes=response.displayed_attributes,
        ),
        list(facts),
        build_evidence_ledger(list(facts)),
        constraints=frame.constraints,
        requested_attributes=frame.requested_attributes,
    )
    assert verifier.passed is True


def test_eval_dell_dedicated_gpu_under_30m_display_specs_include_gpu_and_price() -> None:
    route = route_intent("Có laptop Dell nào có card rời dưới 30 triệu không?", AgentState())
    frame = build_query_frame(route, AgentState())
    result = search_products(CATALOG, frame.constraints)
    response = compose_response(
        ResponseDraftInput(
            response_mode="filtered_search_result",
            products=result.facts,
            evidence_ledger=build_evidence_ledger(list(result.facts)),
            constraints=frame.constraints,
            requested_attributes=frame.requested_attributes,
        )
    )

    assert frame.constraints.brand == "Dell"
    assert frame.constraints.category == "Laptop"
    assert frame.constraints.gpu_type == "dedicated"
    assert frame.constraints.max_price == 30_000_000
    assert all(fact.brand == "Dell" and fact.gpu_type == "dedicated" for fact in result.facts)
    assert response.related_products
    assert "GPU" in " ".join(response.related_products[0].display_specs)
    assert "Giá" in " ".join(response.related_products[0].display_specs)


def test_eval_advisory_tradeoff_does_not_claim_unsupported_pin_or_durability() -> None:
    route = route_intent("Có laptop nào RAM 16GB SSD 512GB không?", AgentState())
    frame = build_query_frame(route, AgentState())
    result = search_products(CATALOG, frame.constraints)
    response = compose_response(
        ResponseDraftInput(
            response_mode="filtered_search_result",
            products=result.facts,
            evidence_ledger=build_evidence_ledger(list(result.facts)),
            constraints=frame.constraints,
            requested_attributes=frame.requested_attributes,
        )
    )

    assert "Nếu ưu tiên giá thấp" in response.answer_text or "Nhận định nhanh" in response.answer_text
    assert "pin tốt" not in response.answer_text.casefold()
    assert "bền hơn" not in response.answer_text.casefold()
    assert "chạy mát" not in response.answer_text.casefold()


def test_eval_next_best_question_is_contextual_for_dell_gpu() -> None:
    route = route_intent("Có laptop Dell nào có card rời dưới 30 triệu không?", AgentState())
    frame = build_query_frame(route, AgentState())
    result = search_products(CATALOG, frame.constraints)
    response = compose_response(
        ResponseDraftInput(
            response_mode="filtered_search_result",
            products=result.facts,
            evidence_ledger=build_evidence_ledger(list(result.facts)),
            constraints=frame.constraints,
            requested_attributes=frame.requested_attributes,
        )
    )

    assert "Asus/MSI/Acer" in response.answer_text
    assert "GPU rời" in response.answer_text
    assert "SKU" not in response.answer_text


def test_eval_comparison_mode_outputs_markdown_table() -> None:
    facts = (normalize_product(DELL_RTX), normalize_product(HP_INTEGRATED))
    response = compose_response(
        ResponseDraftInput(
            response_mode="comparison",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts)),
        )
    )

    assert response.answer_mode == "comparison"
    assert "| Tiêu chí |" in response.answer_text
    assert "| Giá |" in response.answer_text
    assert "Mình không kết luận độ bền/pin" in response.answer_text
