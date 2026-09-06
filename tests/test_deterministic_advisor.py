import asyncio
from datetime import UTC, datetime
from pathlib import Path

from backend.services.advisor import AdvisoryResult, CatalogAdvisor
from backend.services.ai_service import AIService
from backend.services.catalog import CatalogProduct, CatalogService
from backend.services.decision_engine import DecisionPacket


def _catalog(tmp_path: Path) -> CatalogService:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "\n".join(
            [
                "Product Code,Product,Brand,Price,LLM_Context",
                'BF018999,Mobile Phone,Realme,13.252.300 VNĐ,"Sản phẩm Mobile Phone thương hiệu Realme, mã sản phẩm BF018999. Giá bán hiện tại là 13.252.300 VNĐ. Cấu hình kỹ thuật bao gồm: Chip Snapdragon 7s, RAM 8GB, Bộ nhớ trong 64GB."',
                'ALT00001,Mobile Phone,Realme,9.900.000 VNĐ,"Sản phẩm Mobile Phone thương hiệu Realme, mã sản phẩm ALT00001. Giá bán hiện tại là 9.900.000 VNĐ. Cấu hình kỹ thuật bao gồm: Chip Snapdragon 7s, RAM 8GB, Bộ nhớ trong 128GB."',
                'ALT00002,Mobile Phone,Samsung,8.500.000 VNĐ,"Sản phẩm Mobile Phone thương hiệu Samsung, mã sản phẩm ALT00002. Giá bán hiện tại là 8.500.000 VNĐ. Cấu hình kỹ thuật bao gồm: Chip Snapdragon 7s, RAM 8GB, Bộ nhớ trong 64GB."',
                'LAP00001,Laptop,Dell,12.000.000 VNĐ,"Sản phẩm Laptop thương hiệu Dell, mã sản phẩm LAP00001. Giá bán hiện tại là 12.000.000 VNĐ. Cấu hình kỹ thuật bao gồm: Core i5, RAM 8GB, Ổ cứng SSD 512GB."',
            ]
        ),
        encoding="utf-8-sig",
    )
    return CatalogService(catalog_path=path)


def test_detail_request_stays_on_exact_sku_without_unasked_alternatives(tmp_path):
    catalog = _catalog(tmp_path)
    product = catalog.get("BF018999")
    result = CatalogAdvisor(catalog).answer(
        "Hãy tư vấn chi tiết sản phẩm BF018999.",
        product,
    )

    assert "BF018999" in result.text
    assert "bộ nhớ 64GB" in result.text
    assert "ALT00001" not in result.text
    assert "ALT00002" not in result.text
    assert result.product_codes == ("BF018999",)


def test_price_objection_explains_value_without_forcing_comparison(tmp_path):
    catalog = _catalog(tmp_path)
    product = catalog.get("BF018999")
    result = CatalogAdvisor(catalog).answer("Sao máy đó giá cao vậy?", product)

    assert "BF018999" in result.text
    assert "sản phẩm" in result.text
    assert "ALT00001" not in result.text
    assert result.product_codes == ("BF018999",)


def test_alternative_request_is_the_only_mode_that_lists_other_skus(tmp_path):
    catalog = _catalog(tmp_path)
    product = catalog.get("BF018999")
    result = CatalogAdvisor(catalog).answer(
        "Có mẫu nào rẻ hơn để thay thế BF018999 không?",
        product,
    )

    assert "ALT00001" in result.text or "ALT00002" in result.text
    assert len(result.product_codes) > 1


def test_ambiguous_price_objection_asks_for_product_instead_of_random_results(tmp_path):
    catalog = _catalog(tmp_path)
    service = AIService()

    answer = asyncio.run(service.answer("giá đắt quá", catalog, []))

    assert "mã SKU hoặc tên sản phẩm" in answer.text
    assert answer.product_codes == []
    assert answer.mode == "deterministic_advisor"


def test_follow_up_price_objection_uses_conversation_product(tmp_path):
    catalog = _catalog(tmp_path)
    service = AIService()

    answer = asyncio.run(
        service.answer(
            "giá đắt quá",
            catalog,
            [],
            context_product_codes=["BF018999"],
        )
    )

    assert "BF018999" in answer.text
    assert answer.product_codes[0] == "BF018999"
    assert "LAP00001" not in answer.text


def test_follow_up_alternative_request_returns_comparables_only_when_asked(tmp_path):
    catalog = _catalog(tmp_path)
    service = AIService()

    answer = asyncio.run(
        service.answer(
            "Có mẫu nào rẻ hơn không?",
            catalog,
            [],
            context_product_codes=["BF018999"],
        )
    )

    assert answer.mode == "deterministic_advisor"
    assert answer.product_codes[0] == "BF018999"
    assert len(answer.product_codes) > 1
    assert "ALT00001" in answer.text or "ALT00002" in answer.text


def test_exact_sku_detail_returns_only_the_requested_product(tmp_path):
    catalog = _catalog(tmp_path)
    service = AIService()

    answer = asyncio.run(
        service.answer(
            "Hãy tư vấn chi tiết sản phẩm mã BF018999.",
            catalog,
            [catalog.get("BF018999")],
        )
    )

    assert answer.product_codes == ["BF018999"]
    assert "ALT00001" not in answer.text
    assert answer.verification["reasoning"] == (
        "Deterministic exact-SKU product consultation."
    )


def test_policy_question_uses_local_policy_path_without_loading_llm(tmp_path):
    catalog = _catalog(tmp_path)
    service = AIService()

    def fail_if_called():
        raise AssertionError("External workflow should not be loaded for policy lookup")

    service.get_workflow = fail_if_called
    answer = asyncio.run(service.answer("Apple bảo hành bao lâu?", catalog, []))

    assert answer.mode == "deterministic_policy"
    assert "Apple_Service_Warranty.pdf" in answer.text
    assert "một (1) năm" in answer.text


def test_freshly_fetched_price_does_not_warn_about_stale_offer_metadata() -> None:
    product = CatalogProduct(
        code="TEST0001",
        category="Laptop",
        brand="Dell",
        price="20.000.000 VNĐ",
        context="",
        specs=(),
        fetched_at=datetime.now(UTC).isoformat(),
        price_valid_until="2026-06-19",
    )

    assert CatalogAdvisor._price_validity_warning(product) is None


def test_open_question_does_not_load_heavy_workflow_by_default(
    tmp_path, monkeypatch
) -> None:
    catalog = _catalog(tmp_path)
    service = AIService()
    monkeypatch.delenv("ENABLE_EXTERNAL_AI_WORKFLOW", raising=False)

    def fail_if_called():
        raise AssertionError("Heavy workflow must be opt-in")

    service.get_workflow = fail_if_called
    answer = asyncio.run(
        service.answer(
            "Theo bạn tôi nên bắt đầu chọn máy từ đâu?",
            catalog,
            [],
        )
    )

    assert answer.mode == "catalog_fallback"
    assert answer.status == "degraded"


def test_llm_decision_phrasing_falls_back_when_it_invents_a_number(
    tmp_path, monkeypatch
) -> None:
    catalog = _catalog(tmp_path)
    product = catalog.get("BF018999")
    assert product is not None
    service = AIService()
    baseline = AdvisoryResult(
        text=f"Mình chọn SKU {product.code}, giá {product.price}.",
        product_codes=(product.code,),
    )
    packet = DecisionPacket(
        answer_type="comparison",
        products=(product,),
        scores=(),
        recommendation_code=product.code,
        facts=("RAM 8GB",),
        warnings=(),
    )

    async def unsafe_phrase(advisory, _packet):
        return AdvisoryResult(
            text=(
                f"Mình chọn SKU {product.code}, giá {product.price}, "
                "với RAM 32GB cực mạnh."
            ),
            product_codes=advisory.product_codes,
        )

    monkeypatch.setenv("ENABLE_LLM_DECISION_PHRASING", "true")
    monkeypatch.setattr(service, "_phrase_locked_decision", unsafe_phrase)
    answer = asyncio.run(
        service._decision_answer(
            baseline,
            "comparison",
            None,
            catalog,
            tool="test_engine",
            packet=packet,
            recommendation_code=product.code,
            scores=[],
        )
    )

    assert answer.text == baseline.text
    assert answer.decision_trace["phrasing"]["mode"] == "deterministic"
    assert "32gb" in answer.decision_trace["phrasing"]["fallback_reason"].lower()
