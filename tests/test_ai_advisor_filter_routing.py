from __future__ import annotations

from fastapi.testclient import TestClient

from backend.services.catalog import get_catalog
from backend.services.conversation import CandidateRef, DecisionContext
from backend.api.main import app, _catalog_revision


client = TestClient(app)


def _candidate_ref(code: str) -> CandidateRef:
    catalog = get_catalog()
    product = catalog.get(code)
    assert product is not None
    return CandidateRef(
        code=product.code,
        name=product.name,
        brand=product.brand,
        category=product.category,
        price=product.price,
        specs_summary=", ".join(product.specs[:4]) if product.specs else None,
    )


def _state_for_focus(code: str) -> DecisionContext:
    catalog = get_catalog()
    product = catalog.get(code)
    assert product is not None
    return DecisionContext(
        category=product.category,
        active_product_code=product.code,
        focused_product_code=product.code,
        focused_product_name=product.name,
        candidate_codes=[product.code],
        last_shown_candidates=[_candidate_ref(product.code)],
        last_category=product.category,
        catalog_revision=_catalog_revision(catalog),
    )


def _ask(message: str, state: DecisionContext | None = None) -> dict:
    response = client.post(
        "/api/chat",
        json={
            "message": message,
            "history": [],
            "conversation_state": state.model_dump() if state else None,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_new_brand_cpu_filter_does_not_stick_to_old_focused_product() -> None:
    state = _state_for_focus("00927402")

    payload = _ask("Co may Dell nao i7 khong", state)
    assert payload["products"]
    assert "00927402" not in {product["code"] for product in payload["products"]}
    assert {product["brand"] for product in payload["products"]} == {"Dell"}
    assert all("i7" in product["name"].lower() for product in payload["products"])


def test_budget_cpu_follow_up_runs_fresh_search() -> None:
    state = _state_for_focus("00927402")

    payload = _ask("duoi 30 trieu thi co may nao i7 khong", state)
    assert payload["products"]
    assert all(product["price_value"] <= 30_000_000 for product in payload["products"])
    assert all(
        "i7" in " ".join([product["name"], *product["specs"]]).lower()
        for product in payload["products"]
    )
    assert payload["answer_type"] in {"catalog_search", "catalog_ranking", "product_detail"}


def test_dell_under_30m_with_dedicated_gpu_filters_exactly() -> None:
    payload = _ask("co may Dell nao duoi 30 trieu co card roi khong")

    assert payload["products"]
    assert {product["brand"] for product in payload["products"]} == {"Dell"}
    assert all(product["price_value"] <= 30_000_000 for product in payload["products"])
    assert any("mx570a" in " ".join(product["specs"]).lower() for product in payload["products"])


def test_focused_weight_question_reports_missing_catalog_field_without_searching_random_products() -> None:
    state = _state_for_focus("00927992")

    payload = _ask("may Dell ban dang noi nang bao kg", state)
    assert payload["products"]
    assert [product["code"] for product in payload["products"]] == ["00927992"]
    assert "chưa có dữ liệu trọng lượng" in payload["text"].lower()


def test_focused_battery_question_uses_catalog_pin_fact_when_available() -> None:
    state = _state_for_focus("00927778")

    payload = _ask("may do pin co trau khong", state)
    assert [product["code"] for product in payload["products"]] == ["00927778"]
    assert "41" in payload["text"] and "Wh" in payload["text"]
    assert "chưa có dữ liệu pin" not in payload["text"].lower()


def test_answer_text_and_cards_stay_on_same_products() -> None:
    payload = _ask("Co may Dell nao i7 khong")

    assert payload["products"]
    for product in payload["products"]:
        assert product["name"] in payload["text"]
