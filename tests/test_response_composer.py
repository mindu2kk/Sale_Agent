from __future__ import annotations

import pytest

from backend.agent.evidence import build_evidence_ledger
from backend.agent.product_facts import normalize_product
from backend.agent.response_composer import ResponseDraftInput, compose_response
from backend.agent.state import ProductConstraints
from backend.agent.verifier import AdvisorResponseContract, verify_response
from backend.services.catalog import CatalogProduct, get_catalog


def _product(
    code: str,
    name: str,
    brand: str,
    specs: tuple[str, ...],
    *,
    price: str = "19.490.000 VND",
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

ACER = _product(
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


def _facts(*products: CatalogProduct):
    return tuple(normalize_product(product) for product in products)


def test_focused_product_detail_uses_selected_product_facts_and_sets_focus() -> None:
    facts = _facts(HP)
    ledger = build_evidence_ledger(list(facts))

    response = compose_response(
        ResponseDraftInput(
            response_mode="focused_product_detail",
            products=facts,
            evidence_ledger=ledger,
            focused_product_code=HP.code,
        )
    )

    assert response.answer_mode == "focused_product_detail"
    assert response.related_product_codes == (HP.code,)
    assert "HP 14-ep1179TU" in response.answer_text
    assert "MSI" not in response.answer_text
    assert "41Wh" in response.answer_text
    assert any(action.type == "SET_FOCUSED_PRODUCT" for action in response.ui_actions)


def test_focused_product_detail_analyzes_real_ideapad_across_multiple_criteria() -> None:
    catalog = get_catalog()
    product = catalog.get("00929021")
    assert product is not None
    facts = _facts(product)
    ledger = build_evidence_ledger(list(facts))

    response = compose_response(
        ResponseDraftInput(
            response_mode="focused_product_detail",
            products=facts,
            evidence_ledger=ledger,
            focused_product_code=product.code,
        )
    )

    assert response.answer_mode == "focused_product_detail"
    assert "Phân tích theo tiêu chí" in response.answer_text
    assert "- Hiệu năng:" in response.answer_text
    assert "- Màn hình:" in response.answer_text
    assert "- Di động và pin:" in response.answer_text
    assert "- Bộ nhớ/lưu trữ:" in response.answer_text
    assert "1.43kg" in response.answer_text
    assert "chưa có trọng lượng" not in response.answer_text


def test_missing_field_response_discloses_missing_field_without_broad_search() -> None:
    facts = _facts(DELL_I7)
    ledger = build_evidence_ledger(list(facts), requested_fields=("weight_kg",))

    response = compose_response(
        ResponseDraftInput(
            response_mode="missing_field",
            products=facts,
            evidence_ledger=ledger,
            missing_fields=("weight_kg",),
            focused_product_code=DELL_I7.code,
        )
    )

    assert response.answer_mode == "missing_field"
    assert response.related_product_codes == (DELL_I7.code,)
    assert response.missing_fields == ("weight_kg",)
    assert "chưa có dữ liệu trọng lượng" in response.answer_text
    assert "Acer" not in response.answer_text

    verifier_result = verify_response(
        AdvisorResponseContract(
            answer_text=response.answer_text,
            related_product_codes=response.related_product_codes,
            answer_mode=response.answer_mode,
            missing_fields=response.missing_fields,
        ),
        list(facts),
        ledger,
        asked_field="weight_kg",
        focused_product_code=DELL_I7.code,
    )
    assert verifier_result.passed is True


def test_filtered_search_response_keeps_text_and_related_products_consistent() -> None:
    facts = _facts(DELL_I7)
    ledger = build_evidence_ledger(list(facts))
    constraints = ProductConstraints(brand="Dell", cpu_tier="i7", max_price=30_000_000)

    response = compose_response(
        ResponseDraftInput(
            response_mode="filtered_search_result",
            products=facts,
            evidence_ledger=ledger,
            constraints=constraints,
        )
    )

    assert response.related_product_codes == (DELL_I7.code,)
    assert "Laptop Dell 15 DC15250 i7-1355U" in response.answer_text
    assert "HP 14-ep1179TU" not in response.answer_text
    assert any(action.type == "SHOW_RELATED_PRODUCTS" for action in response.ui_actions)

    verifier_result = verify_response(
        AdvisorResponseContract(
            answer_text=response.answer_text,
            related_product_codes=response.related_product_codes,
            answer_mode=response.answer_mode,
        ),
        list(facts),
        ledger,
        constraints=constraints,
    )
    assert verifier_result.passed is True


def test_no_result_response_has_no_related_product_cards() -> None:
    response = compose_response(
        ResponseDraftInput(
            response_mode="no_result",
            products=(),
            constraints=ProductConstraints(
                brand="Dell",
                cpu_tier="i7",
                gpu_type="dedicated",
                max_price=30_000_000,
            ),
            alternative_brands=("MSI", "Asus", "Acer"),
        )
    )

    assert response.answer_mode == "no_result"
    assert response.related_product_codes == ()
    assert "Dell" in response.answer_text
    assert "card rời" in response.answer_text
    assert "MSI/Asus/Acer" in response.answer_text


def test_comparison_response_includes_only_compared_products() -> None:
    facts = _facts(DELL_I7, ACER)
    response = compose_response(
        ResponseDraftInput(
            response_mode="comparison",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts)),
        )
    )

    assert response.answer_mode == "comparison"
    assert response.related_product_codes == (DELL_I7.code, ACER.code)
    assert "Laptop Dell 15 DC15250" in response.answer_text
    assert "Acer Aspire Gaming" in response.answer_text
    assert "HP 14-ep1179TU" not in response.answer_text
    assert any(action.type == "OFFER_COMPARE" for action in response.ui_actions)


def test_correction_response_locks_corrected_product_focus() -> None:
    facts = _facts(HP)
    response = compose_response(
        ResponseDraftInput(
            response_mode="correction",
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts)),
            focused_product_code=HP.code,
        )
    )

    assert response.answer_mode == "correction"
    assert response.related_product_codes == (HP.code,)
    assert "chỉnh lại đúng mẫu" in response.answer_text
    assert "không đổi sang danh sách sản phẩm khác" in response.answer_text
    assert any(
        action.type == "SET_FOCUSED_PRODUCT" and action.product_codes == (HP.code,)
        for action in response.ui_actions
    )


@pytest.mark.parametrize(
    "mode",
    ("tradeoff", "hardware_explanation"),
)
def test_remaining_response_modes_return_grounded_related_products(mode: str) -> None:
    facts = _facts(ACER)
    response = compose_response(
        ResponseDraftInput(
            response_mode=mode,
            products=facts,
            evidence_ledger=build_evidence_ledger(list(facts)),
        )
    )

    assert response.answer_mode == mode
    assert response.related_product_codes == (ACER.code,)
    assert "Acer Aspire Gaming" in response.answer_text
