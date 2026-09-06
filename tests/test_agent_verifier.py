from __future__ import annotations

from backend.agent.evidence import build_evidence_ledger
from backend.agent.product_facts import normalize_product
from backend.agent.state import ProductConstraints
from backend.agent.verifier import AdvisorResponseContract, verify_response
from backend.services.catalog import CatalogProduct


def _product(
    code: str,
    name: str,
    brand: str,
    specs: tuple[str, ...],
    *,
    price: str = "19.490.000 VNĐ",
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
        "Card đồ hoạ Intel Graphics",
        "RAM 16GB",
        "Dung lượng Pin 41Wh",
    ),
)

DELL_I7 = _product(
    "00927992",
    "Laptop Dell 15 DC15250 i7-1355U (DC5I7952W1)",
    "Dell",
    (
        "CPU Core i7",
        "Card đồ hoạ Intel UHD Graphics",
        "RAM 16GB",
        "Ổ cứng SSD 512GB",
    ),
    price="23.790.000 VNĐ",
)

DELL_I5 = _product(
    "00927402",
    "Dell 15 DC15250 i5-1334U (71092479)",
    "Dell",
    (
        "CPU Core i5",
        "Card đồ hoạ Intel UHD Graphics",
        "RAM 16GB",
    ),
)


def test_battery_fact_blocks_contradictory_missing_pin_answer() -> None:
    facts = normalize_product(HP)
    ledger = build_evidence_ledger([facts])
    response = AdvisorResponseContract(
        answer_text="Catalog hiện chưa có dữ liệu pin của mẫu HP 14-ep1179TU Core 5 120U (C89ZSPA).",
        related_product_codes=(HP.code,),
        answer_mode="missing_field",
        missing_fields=("battery_wh",),
    )

    result = verify_response(response, [facts], ledger)

    assert result.passed is False
    assert any(failure.code == "contradictory_missing_battery" for failure in result.failures)


def test_missing_weight_for_focused_product_forces_missing_field_mode() -> None:
    facts = normalize_product(DELL_I7)
    ledger = build_evidence_ledger([facts], requested_fields=("weight_kg",))
    response = AdvisorResponseContract(
        answer_text="Mình đang xem đúng mẫu Laptop Dell 15 DC15250 i7-1355U (DC5I7952W1).",
        related_product_codes=(DELL_I7.code,),
        answer_mode="focused_product_detail",
        missing_fields=(),
    )

    result = verify_response(
        response,
        [facts],
        ledger,
        asked_field="weight_kg",
        focused_product_code=DELL_I7.code,
    )

    assert result.passed is False
    assert result.forced_response_mode == "missing_field"
    assert any(failure.code == "missing_field_not_disclosed" for failure in result.failures)


def test_focused_field_question_cannot_return_broad_search_products() -> None:
    hp_facts = normalize_product(HP)
    dell_facts = normalize_product(DELL_I7)
    ledger = build_evidence_ledger([hp_facts, dell_facts], requested_fields=("weight_kg",))
    response = AdvisorResponseContract(
        answer_text="Mình tìm thêm Dell 15.",
        related_product_codes=(DELL_I7.code,),
        answer_mode="filtered_search_result",
        missing_fields=("weight_kg",),
    )

    result = verify_response(
        response,
        [hp_facts, dell_facts],
        ledger,
        asked_field="weight_kg",
        focused_product_code=HP.code,
    )

    assert result.passed is False
    assert any(failure.code == "focused_field_broad_search" for failure in result.failures)


def test_dell_i7_constraints_reject_i5_related_product() -> None:
    facts = normalize_product(DELL_I5)
    ledger = build_evidence_ledger([facts])
    response = AdvisorResponseContract(
        answer_text="Dell 15 DC15250 i5-1334U (71092479)",
        related_product_codes=(DELL_I5.code,),
        answer_mode="filtered_search_result",
    )

    result = verify_response(
        response,
        [facts],
        ledger,
        constraints=ProductConstraints(brand="Dell", cpu_tier="i7", max_price=30_000_000),
    )

    assert result.passed is False
    assert any(failure.code == "constraint_mismatch_cpu_tier" for failure in result.failures)


def test_durability_winner_claim_is_blocked_without_durability_evidence() -> None:
    hp_facts = normalize_product(HP)
    dell_facts = normalize_product(DELL_I7)
    ledger = build_evidence_ledger([hp_facts, dell_facts])
    response = AdvisorResponseContract(
        answer_text="HP 14-ep1179TU là mẫu bền nhất trong nhóm này.",
        related_product_codes=(HP.code,),
        answer_mode="strong_claim_ranking",
    )

    result = verify_response(response, [hp_facts, dell_facts], ledger)

    assert result.passed is False
    assert "durability_best" in result.blocked_claims
    assert any(failure.code == "unsupported_durability_winner" for failure in result.failures)


def test_answer_text_and_related_products_must_match() -> None:
    hp_facts = normalize_product(HP)
    dell_facts = normalize_product(DELL_I7)
    ledger = build_evidence_ledger([hp_facts, dell_facts])
    response = AdvisorResponseContract(
        answer_text="Mình khuyên HP 14-ep1179TU Core 5 120U (C89ZSPA).",
        related_product_codes=(DELL_I7.code,),
        answer_mode="filtered_search_result",
    )

    result = verify_response(response, [hp_facts, dell_facts], ledger)

    assert result.passed is False
    assert any(failure.code == "answer_cards_mismatch" for failure in result.failures)


def test_valid_filtered_search_response_passes() -> None:
    facts = normalize_product(DELL_I7)
    ledger = build_evidence_ledger(
        [facts],
        constraints_checked={"brand": "Dell", "cpu_tier": "i7", "max_price": 30_000_000},
    )
    response = AdvisorResponseContract(
        answer_text="Laptop Dell 15 DC15250 i7-1355U (DC5I7952W1) phù hợp bộ lọc Dell i7 dưới 30 triệu.",
        related_product_codes=(DELL_I7.code,),
        answer_mode="filtered_search_result",
    )

    result = verify_response(
        response,
        [facts],
        ledger,
        constraints=ProductConstraints(brand="Dell", cpu_tier="i7", max_price=30_000_000),
    )

    assert result.passed is True
