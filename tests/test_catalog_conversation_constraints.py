from fastapi.testclient import TestClient

from backend.services.catalog import _parse_price_intent, _price_value, get_catalog
from backend.api.main import app
from backend.services.value_engine import ValueScoringEngine
from scripts.crawl_fptshop_catalog import _evidence_specs, _heading_specs


def test_laptop_budget_does_not_mix_phone_results() -> None:
    catalog = get_catalog()
    constraints = catalog.resolve_constraints("Tư vấn laptop gần 20 triệu")

    products = catalog.search(
        "Tư vấn laptop gần 20 triệu",
        category=constraints.category,
        price_intent=constraints.price_intent,
        limit=4,
    )

    assert products
    assert all(product.category == "Laptop" for product in products)
    assert all(abs(_price_value(product.price) - 20_000_000) <= 2_400_000 for product in products)


def test_follow_up_inherits_laptop_budget_and_filters_discrete_gpu() -> None:
    catalog = get_catalog()
    constraints = catalog.resolve_constraints(
        "Có máy nào có card rời không",
        history=["Tư vấn laptop gần 20 triệu"],
    )

    products = catalog.search(
        "Có máy nào có card rời không",
        category=constraints.category,
        price_intent=constraints.price_intent,
        discrete_gpu=constraints.discrete_gpu,
        limit=4,
    )

    assert constraints.category == "Laptop"
    assert constraints.price_intent is not None
    assert constraints.price_intent.target == 20_000_000
    assert constraints.discrete_gpu is True
    assert products
    assert all(product.category == "Laptop" for product in products)
    assert all(catalog._has_discrete_gpu(product) for product in products)


def test_new_category_does_not_inherit_stale_budget() -> None:
    catalog = get_catalog()
    constraints = catalog.resolve_constraints(
        "Tư vấn điện thoại cho tôi",
        history=["Tư vấn laptop gần 20 triệu"],
        context_product_codes=["00921510"],
    )

    assert constraints.category == "Mobile Phone"
    assert constraints.price_intent is None


def test_gaming_laptop_query_implies_discrete_gpu() -> None:
    catalog = get_catalog()
    constraints = catalog.resolve_constraints("Giá 20 triệu chơi game laptop")

    assert constraints.category == "Laptop"
    assert constraints.discrete_gpu is True


def test_sku_is_never_parsed_as_price() -> None:
    assert _parse_price_intent("00928595") is None
    assert _parse_price_intent(
        "Hãy tư vấn chi tiết sản phẩm mã 00928595."
    ) is None


def test_shared_unit_price_range_is_parsed_correctly() -> None:
    intent = _parse_price_intent(
        "Có máy nào cũng tầm giá 23 24 triệu không"
    )

    assert intent is not None
    assert intent.mode == "range"
    assert intent.minimum == 23_000_000
    assert intent.maximum == 24_000_000

    hyphenated = _parse_price_intent("laptop giá 23-24 triệu")
    assert hyphenated is not None
    assert hyphenated.minimum == 23_000_000
    assert hyphenated.maximum == 24_000_000


def test_date_is_not_parsed_as_price_range() -> None:
    assert _parse_price_intent("giá cập nhật ngày 2026-06-19") is None


def test_laptop_heading_enriches_sparse_json_ld_specs() -> None:
    specs = _heading_specs(
        '<h1>Laptop HP X Ultra 5/AI/16GB/512GB/14" FHD/Win11</h1>',
        "Laptop",
    )

    assert specs == {
        "RAM": "16GB",
        "Ổ cứng SSD": "512GB",
        "Kích thước màn hình": "14 inch",
        "Độ phân giải": "FHD",
        "Hệ điều hành": "Win11",
    }


def test_evidence_specs_promote_only_unique_sourced_facts() -> None:
    specs = _evidence_specs(
        [
            "Máy có trọng lượng khoảng 1.5 kg và bảo hành 24 tháng.",
            "Khung nhôm đạt tiêu chuẩn MIL-STD-810H, hỗ trợ nâng cấp RAM.",
            "Kết nối gồm USB-C, HDMI 2.1 và Thunderbolt 4.",
        ]
    )

    assert specs["Trọng lượng"] == "1.5 kg"
    assert specs["Bảo hành"] == "24 tháng"
    assert specs["Vật liệu"].lower() == "nhôm"
    assert "MIL-STD-810H" in specs["Tiêu chuẩn độ bền"]
    assert "USB-C" in specs["Cổng kết nối"]


def test_exact_sku_conversation_keeps_context_across_follow_ups() -> None:
    client = TestClient(app)
    history: list[dict] = []

    def ask(message: str) -> dict:
        response = client.post(
            "/api/chat",
            json={"message": message, "history": history[-12:]},
        )
        assert response.status_code == 200
        payload = response.json()
        history.append(
            {"role": "user", "text": message, "product_codes": []}
        )
        history.append(
            {
                "role": "assistant",
                "text": payload["text"],
                "product_codes": [
                    product["code"] for product in payload["products"]
                ],
            }
        )
        return payload

    first = ask("Hãy tư vấn chi tiết sản phẩm mã 00928595.")
    assert [product["code"] for product in first["products"]] == ["00928595"]
    assert "CPU Core Ultra 5" in first["text"]

    detail = ask("Nói kĩ hơn cấu hình máy cho tôi đi")
    assert [product["code"] for product in detail["products"]] == ["00928595"]
    assert "00928595" in detail["text"]
    assert "928.595" not in detail["text"]
    assert "Nokia" not in detail["text"]

    correction = ask("đang nói về mã 00928595 mà")
    assert [product["code"] for product in correction["products"]] == ["00928595"]

    explicit_detail = ask("nói kĩ hơn về máy 00928595")
    assert [product["code"] for product in explicit_detail["products"]] == ["00928595"]
    assert "RAM" in explicit_detail["text"]
    assert "chưa" in explicit_detail["text"].lower()

    alternatives = ask("Có máy nào cũng tầm giá 23 24 triệu không")
    assert alternatives["products"]
    assert all(
        product["category"] == "Laptop"
        for product in alternatives["products"]
    )
    assert all(
        23_000_000 <= product["price_value"] <= 24_000_000
        for product in alternatives["products"]
    )
    assert all(
        product["code"] != "00928595"
        for product in alternatives["products"]
    )


def test_same_budget_brand_comparison_returns_both_brands_and_a_decision() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "Cùng tầm giá nên chọn dell hay macbook",
            "history": [
                {
                    "role": "user",
                    "text": "Hãy tư vấn chi tiết sản phẩm mã 00928595.",
                    "product_codes": [],
                },
                {
                    "role": "assistant",
                    "text": "Đang tư vấn HP 00928595.",
                    "product_codes": ["00928595"],
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert {product["brand"] for product in payload["products"]} == {
        "Dell",
        "Apple",
    }
    assert "Kết luận" in payload["text"]
    assert any(
        phrase in payload["text"]
        for phrase in ("mình nghiêng về Dell", "mình nghiêng về Apple")
    )
    assert payload["ai_mode"] == "deterministic_advisor"


def test_performance_value_intent_inherits_structured_budget_and_picks_winner() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "Máy nào có tỉ lệ hiệu năng trên giá thành cao nhất?",
            "history": [],
            "conversation_state": {
                "category": "Laptop",
                "budget_target": 20_000_000,
                "budget_minimum": None,
                "budget_maximum": None,
                "goal": None,
                "use_case": None,
                "active_product_code": None,
                "compared_codes": [],
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tools_used"] == ["catalog_value_ranking"]
    assert payload["products"][0]["code"] == "00921510"
    assert "mình chốt" in payload["text"]
    assert "Vì sao mình chọn" in payload["text"]
    assert "Điểm cần chấp nhận" in payload["text"]
    assert "Bạn dùng máy chủ yếu" in payload["text"]
    assert payload["conversation_state"]["budget_target"] == 20_000_000
    assert (
        payload["conversation_state"]["goal"]
        == "performance_per_price"
    )


def test_value_follow_up_changes_profile_without_losing_budget_or_goal() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "Tôi không chơi game, chỉ làm văn phòng",
            "history": [],
            "conversation_state": {
                "category": "Laptop",
                "budget_target": 20_000_000,
                "budget_minimum": None,
                "budget_maximum": None,
                "goal": "performance_per_price",
                "use_case": None,
                "active_product_code": "00921510",
                "compared_codes": ["00921510"],
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_state"]["budget_target"] == 20_000_000
    assert payload["conversation_state"]["goal"] == "performance_per_price"
    assert payload["conversation_state"]["use_case"] == "office"
    assert "văn phòng và học tập" in payload["text"]
    assert payload["products"][0]["code"] != "00921510"
    assert (
        payload["conversation_state"]["active_product_code"]
        == payload["products"][0]["code"]
    )
    assert "chưa đủ dữ liệu GPU" not in payload["text"]


def test_value_engine_penalizes_sparse_products_and_exposes_confidence() -> None:
    catalog = get_catalog()
    products = [
        catalog.get("00921510"),
        catalog.get("00927423"),
        catalog.get("00921548"),
    ]
    rankings = ValueScoringEngine().rank(
        [product for product in products if product is not None],
        profile="overall",
    )

    assert rankings[0].product.code == "00921510"
    assert all(0 < ranking.confidence <= 0.95 for ranking in rankings)
    assert rankings[0].reasons
