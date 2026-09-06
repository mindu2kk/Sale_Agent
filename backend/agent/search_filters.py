"""Schema-first product filtering for deterministic agent tools."""

from __future__ import annotations

from backend.agent.product_facts import NormalizedProductFacts
from backend.agent.state import ProductConstraints


def screen_inches_match(actual: float | None, expected: float | None) -> bool:
    if expected is None:
        return True
    if actual is None:
        return False
    if float(expected).is_integer():
        return int(actual) == int(expected)
    return abs(actual - expected) <= 0.11


def matches_constraints(
    facts: NormalizedProductFacts,
    constraints: ProductConstraints,
) -> tuple[bool, str | None]:
    if constraints.category and facts.category != constraints.category:
        return False, "category"
    if constraints.brand and facts.brand != constraints.brand:
        return False, "brand"
    if constraints.min_price is not None:
        if facts.price_value is None or facts.price_value < constraints.min_price:
            return False, "min_price"
    if constraints.max_price is not None:
        if facts.price_value is None or facts.price_value > constraints.max_price:
            return False, "max_price"
    if constraints.cpu_tier is not None:
        if facts.cpu_tier != constraints.cpu_tier:
            return False, "cpu_tier"
    if constraints.gpu_type is not None:
        if facts.gpu_type != constraints.gpu_type:
            return False, "gpu_type"
    if constraints.ram_gb is not None:
        if facts.ram_gb != constraints.ram_gb:
            return False, "ram_gb"
    if constraints.storage_gb is not None:
        if facts.storage_gb != constraints.storage_gb:
            return False, "storage_gb"
    if constraints.screen_inches is not None:
        if not screen_inches_match(facts.screen_inches, constraints.screen_inches):
            return False, "screen_inches"
    return True, None


def no_result_reason(
    constraints: ProductConstraints,
    rejected_reasons: dict[str, int],
) -> str | None:
    if not rejected_reasons:
        return "catalog_empty"
    ordered = sorted(rejected_reasons.items(), key=lambda item: item[1], reverse=True)
    top_reason = ordered[0][0]
    labels = {
        "category": "category",
        "brand": "brand",
        "min_price": "minimum price",
        "max_price": "maximum price",
        "cpu_tier": "CPU tier",
        "gpu_type": "GPU type",
        "ram_gb": "RAM",
        "storage_gb": "storage",
        "screen_inches": "screen size",
    }
    requested = []
    if constraints.brand:
        requested.append(constraints.brand)
    if constraints.cpu_tier:
        requested.append(constraints.cpu_tier)
    if constraints.gpu_type:
        requested.append(f"{constraints.gpu_type} GPU")
    if constraints.ram_gb:
        requested.append(f"RAM {constraints.ram_gb}GB")
    if constraints.storage_gb:
        requested.append(f"SSD {constraints.storage_gb}GB")
    if constraints.screen_inches:
        requested.append(f"{constraints.screen_inches:g} inch")
    if constraints.max_price:
        requested.append(f"under {constraints.max_price}")
    summary = " ".join(requested) if requested else "requested filters"
    return f"No products matched {summary}; most candidates were rejected by {labels.get(top_reason, top_reason)}."
