from __future__ import annotations

from fastapi.testclient import TestClient

from backend.services.ai_service import AIService
from backend.services.catalog import get_catalog
from backend.services.conversation import CandidateRef, DecisionContext
from backend.harness.product_resolver import resolve_product_reference
from backend.api.main import app, _catalog_revision


client = TestClient(app)


def _state_with_candidates(candidate_codes: list[str]) -> DecisionContext:
    catalog = get_catalog()
    products = [catalog.get(code) for code in candidate_codes]
    refs = [
        CandidateRef(
            code=product.code,
            name=product.name,
            brand=product.brand,
            category=product.category,
            price=product.price,
            specs_summary=", ".join(product.specs[:4]) if product and product.specs else None,
        )
        for product in products
        if product is not None
    ]
    return DecisionContext(
        category=products[0].category if products and products[0] else "Laptop",
        candidate_codes=[product.code for product in products if product is not None],
        last_shown_candidates=refs,
        last_category=products[0].category if products and products[0] else "Laptop",
        catalog_revision=_catalog_revision(catalog),
    )


def _ask(message: str, state: DecisionContext | dict | None = None) -> dict:
    payload = {
        "message": message,
        "history": [],
        "conversation_state": (
            state.model_dump() if isinstance(state, DecisionContext) else state
        ),
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    return response.json()


def test_resolve_exact_previous_candidate_by_name() -> None:
    state = _state_with_candidates(["00921510", "00912993", "00921548"])
    resolution = resolve_product_reference(
        "HP 14-ep1179TU Core 5 120U đi",
        state,
    )

    assert resolution.resolved is True
    assert resolution.code == "00921548"
    assert resolution.source in {"previous_candidate_name", "correction"}

    result = _ask("HP 14-ep1179TU Core 5 120U đi", state)
    assert result["answer_type"] in {
        "select_previous_candidate",
        "exact_product_detail",
        "product_detail",
        "product_correction",
    }
    assert "HP 14-ep1179TU Core 5 120U" in result["text"]
    assert "MSI Gaming Thin 15" not in result["text"]
    assert "linh kiện này" not in result["text"]


def test_user_correction_resolves_to_previous_hp_only() -> None:
    state = _state_with_candidates(["00921510", "00912993", "00921548"])
    state.focused_product_code = "00921510"
    state.focused_product_name = "MSI Gaming Thin 15 B13UC-3247VN i5 13420H"

    resolution = resolve_product_reference("máy HP cơ mà", state)
    assert resolution.resolved is True
    assert resolution.code == "00921548"

    result = _ask("máy HP cơ mà", state)
    assert "HP 14-ep1179TU Core 5 120U" in result["text"]
    assert "HP 14-em0024AU" not in result["text"]
    assert "MSI Gaming Thin 15" not in result["text"]


def test_exact_model_after_correction_stays_on_same_hp() -> None:
    state = _state_with_candidates(["00921510", "00912993", "00921548"])
    state.focused_product_code = "00921510"
    state.focused_product_name = "MSI Gaming Thin 15 B13UC-3247VN i5 13420H"

    result = _ask("HP 14-ep1179TU Core 5 120U cơ mà, phân tích kĩ đi", state)
    assert "HP 14-ep1179TU Core 5 120U" in result["text"]
    assert "HP 14-em0024AU" not in result["text"]
    assert "MSI Gaming Thin 15" not in result["text"]
    assert "linh kiện này" not in result["text"]


def test_focused_follow_up_uses_selected_product() -> None:
    state = _state_with_candidates(["00921510", "00912993", "00921548"])
    state.focused_product_code = "00921548"
    state.focused_product_name = "HP 14-ep1179TU Core 5 120U (C89ZSPA)"

    result = _ask("phân tích kĩ đi", state)
    assert result["answer_type"] in {"focused_product_analysis", "product_detail_followup", "product_detail"}
    assert "HP 14-ep1179TU Core 5 120U" in result["text"]
    assert "MSI Gaming Thin 15" not in result["text"]


def test_ordinal_selection_resolves_third_candidate() -> None:
    state = _state_with_candidates(["00921510", "00912993", "00921548"])
    resolution = resolve_product_reference("con thứ 3 đi", state)
    assert resolution.resolved is True
    assert resolution.code == "00921548"

    result = _ask("con thứ 3 đi", state)
    assert "HP 14-ep1179TU Core 5 120U" in result["text"]
    assert result["conversation_state"]["focused_product_code"] == "00921548"


def test_ambiguous_brand_correction_lists_only_previous_hp_candidates() -> None:
    state = _state_with_candidates(["00921548", "00926893", "00921510"])

    result = _ask("máy HP cơ mà", state)
    assert result["answer_type"] == "clarify"
    assert "HP 14-ep1179TU" in result["text"]
    assert "HP 14-em0024AU" in result["text"]
    assert "MSI Gaming Thin 15" not in result["text"]


def test_ai_unavailable_fallback_for_selected_product_is_detail_not_generic() -> None:
    catalog = get_catalog()
    product = catalog.get("00921548")
    assert product is not None

    answer = AIService()._fallback(catalog, [product])
    assert "HP 14-ep1179TU Core 5 120U" in answer.text
    assert "Bạn muốn mình phân tích sâu hơn mẫu nào" not in answer.text
    assert "các mẫu sau đáng cân nhắc" not in answer.text


def test_no_previous_candidates_does_not_pretend_it_knows_hp() -> None:
    result = _ask("máy HP cơ mà")
    assert result["answer_type"] in {"clarify", "catalog_search", "catalog_ranking", "product_detail", "comparison"} or isinstance(result["text"], str)
    assert "mẫu HP vừa xem" not in result["text"]


def test_no_hardware_explanation_misroute_for_selected_product() -> None:
    state = _state_with_candidates(["00921510", "00912993", "00921548"])
    result = _ask("HP 14-ep1179TU Core 5 120U đi", state)
    assert "linh kiện này" not in result["text"]
    assert "áp dụng tiêu chí này" not in result["text"]
