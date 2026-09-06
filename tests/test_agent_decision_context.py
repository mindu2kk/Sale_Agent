from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from backend.services.conversation import ConversationPlanner, DecisionContext
from backend.services.decision_engine import (
    DecisionPacket,
    DecisionPacketVerifier,
)
from backend.services.advisor import AdvisoryResult
from backend.api.main import app
from backend.services.catalog import get_catalog


def _ask(client: TestClient, message: str, state: dict | None = None) -> dict:
    response = client.post(
        "/api/chat",
        json={
            "message": message,
            "history": [],
            "conversation_state": state,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_five_turn_comparison_preserves_budget_brands_and_candidates() -> None:
    client = TestClient(app)
    first = _ask(client, "Nên chọn Dell hay Asus tầm 20 triệu?")
    expected_codes = first["active_context"]["candidate_codes"]

    assert first["answer_type"] == "comparison"
    assert first["active_context"]["budget_target"] == 20_000_000
    assert first["active_context"]["compared_brands"] == ["Dell", "Asus"]
    assert len(expected_codes) == 2
    assert all(product["category"] == "Laptop" for product in first["products"])

    second = _ask(
        client,
        "Ưu tiên độ bền bỉ và hiệu năng.",
        first["conversation_state"],
    )
    assert second["answer_type"] == "comparison"
    assert second["active_context"]["candidate_codes"] == expected_codes
    assert second["active_context"]["preferences"]["durability"] == 1.0
    assert second["active_context"]["preferences"]["performance"] == 1.0

    third = _ask(
        client,
        "Máy nào đáng tiền hơn?",
        second["conversation_state"],
    )
    assert third["active_context"]["candidate_codes"] == expected_codes
    assert third["active_context"]["preferences"]["value"] == 1.0

    fourth = _ask(
        client,
        "Tại sao Dell lại rẻ hơn?",
        third["conversation_state"],
    )
    assert fourth["answer_type"] == "price_causality"
    assert fourth["active_context"]["candidate_codes"] == expected_codes
    assert {product["brand"] for product in fourth["products"]} == {"Dell", "Asus"}
    assert "tìm được các mẫu" not in fourth["text"].lower()

    fifth = _ask(
        client,
        "Nếu chơi game thì sao?",
        fourth["conversation_state"],
    )
    assert fifth["answer_type"] == "comparison"
    assert fifth["active_context"]["candidate_codes"] == expected_codes
    assert fifth["active_context"]["preferences"]["gaming"] == 1.0
    assert {product["brand"] for product in fifth["products"]} == {"Dell", "Asus"}


def test_price_causality_is_not_classified_as_alternative_search() -> None:
    catalog = get_catalog()
    planner = ConversationPlanner(catalog)
    state = DecisionContext(
        category="Laptop",
        compared_brands=["Dell", "Apple"],
        compared_codes=["00927423", "00920365"],
        candidate_codes=["00927423", "00920365"],
    )

    plan = planner.plan(
        "Tại sao laptop Dell lại rẻ hơn MacBook với cùng hiệu năng?",
        state,
    )

    assert plan.dialogue_act == "price_causality"
    assert plan.product_codes == ("00927423", "00920365")


def test_expired_state_is_not_reused() -> None:
    catalog = get_catalog()
    planner = ConversationPlanner(catalog)
    expired = DecisionContext(
        category="Laptop",
        budget_target=20_000_000,
        compared_brands=["Dell", "Asus"],
        compared_codes=["00927423", "00917721"],
        updated_at=(datetime.now(UTC) - timedelta(minutes=6)).isoformat(),
    )

    plan = planner.plan("Ưu tiên độ bền", expired)

    assert plan.dialogue_act == "catalog_search"
    assert plan.brands == ()
    assert plan.price_intent is None


def test_detail_follow_up_stays_on_active_sku() -> None:
    client = TestClient(app)
    first = _ask(client, "Hãy tư vấn chi tiết sản phẩm mã 00928595.")
    second = _ask(
        client,
        "Nói kỹ hơn cấu hình máy cho tôi.",
        first["conversation_state"],
    )

    assert second["answer_type"] == "product_detail"
    assert [product["code"] for product in second["products"]] == ["00928595"]
    assert "00928595" in second["text"]


def test_vague_request_asks_before_searching_random_products() -> None:
    client = TestClient(app)
    payload = _ask(client, "Tư vấn máy cho tôi")

    assert payload["answer_type"] == "clarify"
    assert payload["products"] == []
    assert payload["follow_up_question"]
    assert "laptop hay điện thoại" in payload["text"].lower()


def test_new_explicit_sku_starts_a_clean_topic() -> None:
    client = TestClient(app)
    comparison = _ask(client, "Nên chọn Dell hay Asus tầm 20 triệu?")
    detail = _ask(
        client,
        "Hãy tư vấn chi tiết sản phẩm mã 00928595.",
        comparison["conversation_state"],
    )

    assert detail["conversation_state"]["topic_id"] != comparison["conversation_state"]["topic_id"]
    assert detail["active_context"]["compared_brands"] == []
    assert detail["active_context"]["candidate_codes"] == ["00928595"]
    assert detail["active_context"]["preferences"] == {}


def test_packet_verifier_ignores_eight_letter_product_words_but_rejects_unknown_sku() -> None:
    catalog = get_catalog()
    product = catalog.get("00927423")
    assert product is not None
    packet = DecisionPacket(
        answer_type="comparison",
        products=(product,),
        scores=(),
        recommendation_code=product.code,
        facts=(),
        warnings=(),
    )
    verifier = DecisionPacketVerifier()

    safe = verifier.verify(
        packet,
        AdvisoryResult(
            text=f"Asus Vivobook; SKU {product.code}.",
            product_codes=(product.code,),
        ),
    )
    unsafe = verifier.verify(
        packet,
        AdvisoryResult(
            text=f"SKU {product.code}; ngoài ra SKU 00928595.",
            product_codes=(product.code,),
        ),
    )

    assert safe.approved is True
    assert unsafe.approved is False
    assert "unknown SKU" in unsafe.issues[0]


def test_shadow_mode_reports_router_mismatch_without_changing_answer(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENT_SHADOW_MODE", "true")
    monkeypatch.setenv("EXPOSE_DECISION_TRACE", "true")
    client = TestClient(app)
    comparison = _ask(client, "Nên chọn Dell hay Asus tầm 20 triệu?")
    payload = _ask(
        client,
        "Tại sao Dell lại rẻ hơn Asus?",
        comparison["conversation_state"],
    )

    assert payload["answer_type"] == "price_causality"
    assert payload["decision_trace"]["shadow"]["enabled"] is True
    assert payload["decision_trace"]["shadow"]["legacy_route"] == "cheaper_alternatives"
    assert payload["decision_trace"]["shadow"]["planner_route"] == "price_causality"
    assert payload["decision_trace"]["shadow"]["mismatch"] is True


def test_metrics_endpoint_exposes_agent_quality_counters() -> None:
    client = TestClient(app)
    _ask(client, "Tư vấn máy cho tôi")
    payload = client.get("/metrics").json()

    assert payload["agent"]["requests"] >= 1
    assert payload["agent"]["clarifications"] >= 1
    assert payload["agent"]["latency_p95_ms"] >= 0


def test_rejected_recommendation_is_recorded_and_not_returned_again() -> None:
    client = TestClient(app)
    first = _ask(client, "Tư vấn laptop tầm 20 triệu")
    rejected_code = first["products"][0]["code"]
    second = _ask(
        client,
        "Không chọn mẫu này, đổi mẫu khác đi",
        first["conversation_state"],
    )

    assert second["answer_type"] == "replacement_search"
    assert rejected_code in second["conversation_state"]["rejected_codes"]
    assert rejected_code not in {
        product["code"] for product in second["products"]
    }
