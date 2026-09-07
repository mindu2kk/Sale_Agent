from __future__ import annotations

from backend.agent.state import ProductConstraints
from backend.agent.tools import get_product_field, search_products
from backend.services.catalog import CatalogService, get_catalog


def _catalog_without_weight(tmp_path) -> CatalogService:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "Product Code,Product,Brand,Name,Price,LLM_Context\n"
        "TEST0001,Laptop,Dell,Dell test model,20.000.000 VNĐ,"
        "Sản phẩm Laptop Dell test model có CPU Core i7 và RAM 16GB.\n",
        encoding="utf-8",
    )
    return CatalogService(path)


def test_search_products_filters_brand_cpu_and_budget_with_schema() -> None:
    catalog = get_catalog()
    result = search_products(
        catalog,
        ProductConstraints(
            category="Laptop",
            brand="Dell",
            cpu_tier="i7",
            max_price=30_000_000,
        ),
        limit=20,
    )

    assert result.products
    assert result.no_result_reason is None
    assert result.applied_filters.brand == "Dell"
    assert all(facts.brand == "Dell" for facts in result.facts)
    assert all(facts.cpu_tier == "i7" for facts in result.facts)
    assert all(facts.price_value is not None and facts.price_value <= 30_000_000 for facts in result.facts)


def test_search_products_dedicated_gpu_does_not_return_integrated_intel_graphics() -> None:
    catalog = get_catalog()
    result = search_products(
        catalog,
        ProductConstraints(
            category="Laptop",
            brand="Dell",
            max_price=30_000_000,
            gpu_type="dedicated",
        ),
        limit=20,
    )

    assert result.products
    assert all(facts.brand == "Dell" for facts in result.facts)
    assert all(facts.gpu_type == "dedicated" for facts in result.facts)
    assert all("Intel UHD" not in (facts.gpu_raw or "") for facts in result.facts)


def test_search_products_returns_no_result_instead_of_wrong_cpu_family() -> None:
    catalog = get_catalog()
    result = search_products(
        catalog,
        ProductConstraints(
            category="Laptop",
            brand="Dell",
            cpu_tier="i9",
            max_price=10_000_000,
        ),
    )

    assert result.products == ()
    assert result.facts == ()
    assert result.rejected_count > 0
    assert result.no_result_reason is not None
    assert "i9" in result.no_result_reason


def test_get_product_field_returns_known_battery_fact() -> None:
    catalog = get_catalog()
    field = get_product_field(catalog, "00927778", "battery_wh")

    assert field.missing is False
    assert field.value == 41
    assert field.source_text is not None
    assert "41" in field.source_text


def test_get_product_field_returns_missing_weight_without_searching(tmp_path) -> None:
    catalog = _catalog_without_weight(tmp_path)
    field = get_product_field(catalog, "TEST0001", "weight_kg")

    assert field.code == "TEST0001"
    assert field.field == "weight_kg"
    assert field.value is None
    assert field.missing is True
