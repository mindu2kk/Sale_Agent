"""Deterministic tool layer for the agentic sales advisor."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.agent.product_facts import NormalizedProductFacts, normalize_product
from backend.agent.search_filters import matches_constraints, no_result_reason
from backend.agent.state import ProductConstraints
from backend.services.catalog import CatalogProduct, CatalogService


@dataclass(frozen=True)
class ProductSearchResult:
    products: tuple[CatalogProduct, ...]
    facts: tuple[NormalizedProductFacts, ...]
    applied_filters: ProductConstraints
    rejected_count: int
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    no_result_reason: str | None = None


@dataclass(frozen=True)
class FieldResult:
    code: str
    field: str
    value: object | None
    source_text: str | None = None
    missing: bool = False


def search_products(
    catalog: CatalogService,
    filters: ProductConstraints,
    *,
    limit: int = 12,
    exclude_codes: set[str] | None = None,
) -> ProductSearchResult:
    excluded = {code.casefold() for code in (exclude_codes or set())}
    matched_products: list[CatalogProduct] = []
    matched_facts: list[NormalizedProductFacts] = []
    rejected_count = 0
    rejected_reasons: dict[str, int] = {}

    for product in catalog.products:
        if product.code.casefold() in excluded:
            rejected_count += 1
            rejected_reasons["excluded"] = rejected_reasons.get("excluded", 0) + 1
            continue
        facts = normalize_product(product)
        matched, reason = matches_constraints(facts, filters)
        if not matched:
            rejected_count += 1
            if reason:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            continue
        matched_products.append(product)
        matched_facts.append(facts)

    ordered = sorted(
        zip(matched_products, matched_facts),
        key=lambda item: (
            item[1].price_value if item[1].price_value is not None else 10**12,
            item[0].code,
        ),
    )
    limited = ordered[:limit]
    products = tuple(product for product, _ in limited)
    facts = tuple(item_facts for _, item_facts in limited)
    return ProductSearchResult(
        products=products,
        facts=facts,
        applied_filters=filters,
        rejected_count=rejected_count,
        rejected_reasons=rejected_reasons,
        no_result_reason=(
            no_result_reason(filters, rejected_reasons) if not products else None
        ),
    )


def get_product_by_code(catalog: CatalogService, code: str) -> CatalogProduct | None:
    return catalog.get(code)


def get_product_field(
    catalog: CatalogService,
    code: str,
    field: str,
) -> FieldResult:
    product = catalog.get(code)
    if product is None:
        return FieldResult(code=code, field=field, value=None, missing=True)
    facts = normalize_product(product)
    value = getattr(facts, field, None)
    evidence = facts.evidence_map.get(field)
    return FieldResult(
        code=code,
        field=field,
        value=value,
        source_text=evidence.source_text if evidence else None,
        missing=value is None,
    )


def normalize_product_tool(product: CatalogProduct) -> NormalizedProductFacts:
    return normalize_product(product)
