from __future__ import annotations

import pytest

from backend.agent.product_facts import normalize_product
from backend.agent.spec_parser import infer_cpu_tier, infer_gpu_type
from backend.services.catalog import CatalogProduct, get_catalog


def _product(specs: tuple[str, ...], *, name: str = "Test Laptop") -> CatalogProduct:
    return CatalogProduct(
        code="TEST0001",
        category="Laptop",
        brand="AURA",
        price="19.490.000 VNĐ",
        context="",
        specs=specs,
        title=name,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("Core i7", "i7"),
        ("Intel Core i5-1334U", "i5"),
        ("Core 7 150U", "Core 7"),
        ("Core Ultra 7 256V", "Core Ultra 7"),
        ("Ryzen 7 7730U", "Ryzen 7"),
        ("Ryzen AI 5 330", "Ryzen AI 5"),
    ),
)
def test_cpu_tier_keeps_intel_core_and_ryzen_families_distinct(raw: str, expected: str) -> None:
    assert infer_cpu_tier(raw) == expected


def test_core_7_and_core_ultra_7_are_not_i7() -> None:
    assert infer_cpu_tier("Core 7 150U") != "i7"
    assert infer_cpu_tier("Core Ultra 7 256V") != "i7"
    assert infer_cpu_tier("Ryzen 7 7730U") != "i7"


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("Nvidia GeForce RTX 4050 6GB", "dedicated"),
        ("Nvidia GeForce MX570A 2GB", "dedicated"),
        ("Radeon RX 6500M", "dedicated"),
        ("Intel UHD Graphics", "integrated"),
        ("Intel Iris Xe Graphics", "integrated"),
        ("Intel Graphics", "integrated"),
        ("AMD Radeon Graphics", "integrated"),
    ),
)
def test_gpu_type_detection(raw: str, expected: str) -> None:
    assert infer_gpu_type(raw) == expected


def test_normalizes_numeric_facts_from_specs() -> None:
    facts = normalize_product(
        _product(
            (
                "CPU Core 5 120U",
                "Card đồ hoạ Intel Graphics",
                "RAM 16GB",
                "Ổ cứng SSD 512GB",
                "Kích thước màn hình 14 inch",
                "Tần số quét 120Hz",
                "Dung lượng Pin 41Wh",
                "Trọng lượng 1,53kg",
                "Hệ điều hành Win11",
            )
        )
    )

    assert facts.price_value == 19_490_000
    assert facts.cpu_raw == "Core 5 120U"
    assert facts.cpu_tier == "Core 5"
    assert facts.gpu_raw == "Intel Graphics"
    assert facts.gpu_type == "integrated"
    assert facts.ram_gb == 16
    assert facts.storage_gb == 512
    assert facts.screen_inches == 14
    assert facts.refresh_hz == 120
    assert facts.battery_wh == 41
    assert facts.weight_kg == 1.53
    assert facts.os == "Win11"
    assert facts.evidence_map["battery_wh"].source_text == "Dung lượng Pin 41Wh"


def test_real_catalog_hp_pin_is_not_marked_missing() -> None:
    catalog = get_catalog()
    product = catalog.get("00927778")
    assert product is not None

    facts = normalize_product(product)
    assert facts.battery_wh == 41
    assert facts.has_field("battery_wh") is True


def test_real_catalog_dell_mx_gpu_is_dedicated() -> None:
    catalog = get_catalog()
    product = catalog.get("00923725")
    assert product is not None

    facts = normalize_product(product)
    assert facts.gpu_type == "dedicated"
    assert facts.cpu_tier == "Core 7"


def test_real_catalog_ideapad_weight_can_come_from_context_highlights() -> None:
    catalog = get_catalog()
    product = catalog.get("00929021")
    assert product is not None

    facts = normalize_product(product)
    assert facts.weight_kg == 1.43
    assert "1.43 kg" in facts.evidence_map["weight_kg"].source_text
