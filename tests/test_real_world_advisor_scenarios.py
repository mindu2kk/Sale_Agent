from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from backend.services.catalog import CatalogService, get_catalog
from backend.services.conversation import ConversationPlanner, DecisionContext
from backend.api.main import app


client = TestClient(app)


def _named_phone_catalog(tmp_path: Path) -> CatalogService:
    """Provide stable exact-model names for product-resolution contracts."""
    path = tmp_path / "phones.csv"
    fieldnames = [
        "Product Code",
        "Product",
        "Brand",
        "Name",
        "Price",
        "Price Valid Until",
        "Fetched At",
        "Source URL",
        "LLM_Context",
    ]
    rows = [
        {
            "Product Code": "00928862",
            "Product": "Mobile Phone",
            "Brand": "Oppo",
            "Name": "Oppo A6C 4GB",
            "Price": "4.690.000 VNĐ",
            "Price Valid Until": "2099-01-01",
            "Fetched At": datetime.now(UTC).isoformat(),
            "Source URL": "https://example.com/oppo-a6c",
            "LLM_Context": "RAM 4GB, bộ nhớ trong 64GB, pin 7000mAh.",
        },
        {
            "Product Code": "00928700",
            "Product": "Mobile Phone",
            "Brand": "Tecno",
            "Name": "Tecno Spark 50 4GB",
            "Price": "4.790.000 VNĐ",
            "Price Valid Until": "2099-01-01",
            "Fetched At": datetime.now(UTC).isoformat(),
            "Source URL": "https://example.com/tecno-spark-50",
            "LLM_Context": "RAM 4GB, bộ nhớ trong 128GB, pin 6700mAh.",
        },
        {
            "Product Code": "00922779",
            "Product": "Mobile Phone",
            "Brand": "Xiaomi",
            "Name": "Xiaomi Redmi 15C 4GB",
            "Price": "4.490.000 VNĐ",
            "Price Valid Until": "2099-01-01",
            "Fetched At": datetime.now(UTC).isoformat(),
            "Source URL": "https://example.com/redmi-15c",
            "LLM_Context": "RAM 4GB, bộ nhớ trong 128GB, pin 6000mAh.",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return CatalogService(catalog_path=path)


def ask(message: str, state: dict | None = None) -> dict:
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


def test_brand_follow_up_keeps_budget_and_never_substitutes_other_brands() -> None:
    first = ask("Tư vấn laptop tầm 20 triệu")
    result = ask(
        "Có MacBook trong tầm giá không?",
        first["conversation_state"],
    )

    assert result["active_context"]["budget_target"] == 20_000_000
    assert result["products"]
    assert {product["brand"] for product in result["products"]} == {"Apple"}
    assert {product["category"] for product in result["products"]} == {"Laptop"}
    assert any("MacBook" in product["name"] for product in result["products"])


def test_three_long_product_names_resolve_to_exact_models_and_compare() -> None:
    result = ask(
        "So sánh giữa MSI Gaming Thin 15 B13UC-3247VN, "
        "Dell 15 DC15255 R7-7730U và HP 14-ep1179TU Core 5"
    )

    assert result["answer_type"] == "comparison"
    assert result["verification"]["approved"] is True
    assert {product["code"] for product in result["products"]} == {
        "00921510",
        "00927423",
        "00921548",
    }
    assert result["active_context"]["preferences"].get("gaming") is None
    assert all(
        product["name"] in result["text"]
        for product in result["products"]
    )


def test_model_resolver_does_not_confuse_shared_cpu_with_named_product() -> None:
    catalog = get_catalog()
    products = catalog.resolve_products(
        "So sánh MSI Gaming Thin 15 B13UC-3247VN, "
        "Dell 15 DC15255 R7-7730U và HP 14-ep1179TU Core 5"
    )
    assert [product.code for product in products] == [
        "00921510",
        "00927423",
        "00921548",
    ]


def test_verifier_does_not_treat_eight_character_model_as_unknown_sku() -> None:
    result = ask("So sánh HP 14-ep1179TU Core 5 với Dell 15 DC15255 R7-7730U")
    assert result["verification"]["approved"] is True
    assert result["answer_type"] == "comparison"


def test_brand_search_respects_category_and_does_not_mix_iphone_with_macbook() -> None:
    result = ask("Có MacBook nào giá khoảng 20 triệu không?")
    assert result["products"]
    assert {product["brand"] for product in result["products"]} == {"Apple"}
    assert {product["category"] for product in result["products"]} == {"Laptop"}


def test_phone_brand_constraints_are_hard_filters() -> None:
    for message, brand, maximum in (
        ("Điện thoại Samsung tầm 10 triệu chụp ảnh tốt", "Samsung", 10_000_000),
        ("Có điện thoại Oppo dưới 5 triệu không?", "Oppo", 5_000_000),
    ):
        result = ask(message)

        assert result["products"]
        assert {product["brand"] for product in result["products"]} == {brand}
        assert {product["category"] for product in result["products"]} == {"Mobile Phone"}
        assert all(product["price_value"] <= maximum for product in result["products"])


def test_preference_follow_up_preserves_three_product_candidate_set() -> None:
    first = ask(
        "So sánh MSI Gaming Thin 15 B13UC-3247VN, "
        "Dell 15 DC15255 R7-7730U và HP 14-ep1179TU Core 5"
    )
    expected = set(first["active_context"]["candidate_codes"])
    assert set(first["conversation_state"]["compared_codes"]) == expected
    second = ask(
        "Tôi ưu tiên hiệu năng và dùng để chơi game",
        first["conversation_state"],
    )
    assert second["answer_type"] == "comparison"
    assert set(second["active_context"]["candidate_codes"]) == expected
    assert set(product["code"] for product in second["products"]) == expected


def test_explicit_phone_switch_does_not_keep_laptop_candidates() -> None:
    first = ask("Nên chọn Dell hay Asus tầm 20 triệu?")
    second = ask("Chuyển sang tư vấn điện thoại tầm 8 triệu", first["conversation_state"])
    assert second["active_context"]["category"] == "Mobile Phone"
    assert all(product["category"] == "Mobile Phone" for product in second["products"])
    assert set(second["active_context"]["candidate_codes"]).isdisjoint(
        first["active_context"]["candidate_codes"]
    )


def test_planner_keeps_single_named_brand_as_a_hard_filter() -> None:
    planner = ConversationPlanner(get_catalog())
    plan = planner.plan(
        "Có MacBook trong tầm giá không?",
        DecisionContext(category="Laptop", budget_target=20_000_000),
    )
    assert plan.brands == ("Apple",)
    assert plan.category == "Laptop"
    assert plan.price_intent is not None
    assert plan.price_intent.target == 20_000_000


def test_compare_two_short_phone_names_resolves_exact_products(
    monkeypatch,
    tmp_path: Path,
) -> None:
    catalog = _named_phone_catalog(tmp_path)
    monkeypatch.setattr("backend.api.main.get_catalog", lambda: catalog)
    result = ask("So sánh Oppo A6C 4GB, Tecno Spark 50 4GB")

    assert result["answer_type"] == "comparison"
    assert result["verification"]["approved"] is True
    assert result["active_context"]["category"] == "Mobile Phone"
    assert {product["code"] for product in result["products"]} == {
        "00928862",
        "00928700",
    }


def test_three_phone_comparison_abstains_when_chip_evidence_is_incomplete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    catalog = _named_phone_catalog(tmp_path)
    monkeypatch.setattr("backend.api.main.get_catalog", lambda: catalog)
    result = ask(
        "So sánh Oppo A6C 4GB, Tecno Spark 50 4GB, Xiaomi Redmi 15C 4GB"
    )

    assert result["answer_type"] == "comparison"
    assert result["verification"]["approved"] is True
    assert {product["code"] for product in result["products"]} == {
        "00928862",
        "00928700",
        "00922779",
    }
    # Missing CPU evidence must be represented as missing; do not infer a
    # performance winner from unrelated fields. A price-only observation is
    # still allowed when every displayed price is sourced.
    assert "| CPU | Chưa có dữ liệu |" in result["text"]
    assert "mạnh nhất" not in result["text"]
    assert "GPU | Chưa có dữ liệu" in result["text"]
    assert "SSD | Chưa có dữ liệu" in result["text"]
    assert "chịu lực/độ bền" not in result["text"]
    assert result["conversation_state"]["last_recommendation_code"] == result["products"][0]["code"]
    assert result["conversation_state"]["active_product_code"] == result["products"][0]["code"]


def test_catalog_superlative_ranks_laptops_instead_of_keyword_search() -> None:
    result = ask("Tôi cần laptop xịn nhất shop của bạn")

    assert result["answer_type"] == "catalog_ranking"
    assert result["tools_used"] == ["catalog_capability_ranking"]
    assert result["verification"]["approved"] is True
    assert result["products"]
    assert {product["category"] for product in result["products"]} == {"Laptop"}
    assert result["conversation_state"]["goal"] == "best_overall"
    assert result["conversation_state"]["last_recommendation_code"] == result["products"][0]["code"]
    assert "Vì sao mẫu này đứng đầu" in result["text"]
    assert "Mình chưa thể khóa đúng sản phẩm" not in result["text"]


def test_strongest_follow_up_inherits_category_and_reranks_full_catalog() -> None:
    first = ask("Tôi cần laptop xịn nhất shop của bạn")
    second = ask("máy nào khỏe nhất", first["conversation_state"])

    assert second["answer_type"] == "catalog_ranking"
    assert second["active_context"]["category"] == "Laptop"
    assert second["conversation_state"]["goal"] == "max_performance"
    assert second["products"]
    assert {product["category"] for product in second["products"]} == {"Laptop"}
    assert second["conversation_state"]["last_recommendation_code"] == second["products"][0]["code"]
    assert "khỏe nhất" in second["text"]
    assert "SKU hoặc tên hai mẫu" not in second["text"]
    assert "RTX 5090" in second["text"]


def test_unscoped_strongest_request_clarifies_category_not_sku() -> None:
    result = ask("máy nào khỏe nhất")

    assert result["answer_type"] == "clarify"
    assert "laptop hay điện thoại" in result["text"]
    assert "SKU" not in result["text"]


def test_value_for_money_precedence_beats_generic_best_wording() -> None:
    result = ask("Có laptop nào hiệu năng trên giá tốt nhất?")

    assert result["answer_type"] == "value_ranking"
    assert result["tools_used"] == ["catalog_value_ranking"]
    assert result["conversation_state"]["goal"] == "performance_per_price"


def test_gaming_strongest_uses_catalog_ranking_with_gaming_profile() -> None:
    result = ask("laptop chơi game khỏe nhất")

    assert result["answer_type"] == "catalog_ranking"
    assert result["tools_used"] == ["catalog_capability_ranking"]
    assert result["active_context"]["category"] == "Laptop"
    assert result["conversation_state"]["goal"] == "max_performance"
    assert result["conversation_state"]["use_case"] == "gaming"
    assert result["active_context"]["preferences"].get("gaming") == 1.0
    assert {product["category"] for product in result["products"]} == {"Laptop"}
    assert "cho game" in result["text"]


def test_lowest_price_superlative_keeps_brand_and_category_scope() -> None:
    result = ask("iPhone nào rẻ nhất trong catalog?")

    assert result["answer_type"] == "catalog_ranking"
    assert result["tools_used"] == ["catalog_capability_ranking"]
    assert result["active_context"]["category"] == "Mobile Phone"
    assert {product["brand"] for product in result["products"]} == {"Apple"}
    assert result["conversation_state"]["goal"] == "lowest_price"
    prices = [product["price_value"] for product in result["products"]]
    assert prices == sorted(prices)


def test_highest_price_superlative_keeps_phone_category_scope() -> None:
    result = ask("điện thoại đắt nhất shop là máy nào")

    assert result["answer_type"] == "catalog_ranking"
    assert result["tools_used"] == ["catalog_capability_ranking"]
    assert result["active_context"]["category"] == "Mobile Phone"
    assert {product["category"] for product in result["products"]} == {"Mobile Phone"}
    assert result["conversation_state"]["goal"] == "highest_price"
    prices = [product["price_value"] for product in result["products"]]
    assert prices == sorted(prices, reverse=True)
