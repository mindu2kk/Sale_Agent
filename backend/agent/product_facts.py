"""Normalized product facts used by deterministic agent guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.agent.spec_parser import (
    infer_cpu_tier,
    infer_gpu_type,
    parse_capacity_gb,
    parse_float_with_unit,
    parse_int_with_unit,
    parse_specs,
)
from backend.services.catalog import CatalogProduct, _price_value


@dataclass(frozen=True)
class FactEvidence:
    field: str
    value: object
    source_text: str


@dataclass(frozen=True)
class NormalizedProductFacts:
    code: str
    name: str
    brand: str
    category: str
    price_value: int | None
    cpu_raw: str | None = None
    cpu_tier: str | None = None
    gpu_raw: str | None = None
    gpu_type: str | None = None
    ram_gb: int | None = None
    storage_gb: int | None = None
    screen_inches: float | None = None
    refresh_hz: int | None = None
    battery_wh: float | None = None
    weight_kg: float | None = None
    os: str | None = None
    evidence_map: dict[str, FactEvidence] = field(default_factory=dict)

    def has_field(self, field_name: str) -> bool:
        return getattr(self, field_name, None) is not None


def normalize_product(product: CatalogProduct) -> NormalizedProductFacts:
    parsed_specs = parse_specs(product.specs)
    evidence: dict[str, FactEvidence] = {}

    def raw(field: str) -> str | None:
        item = parsed_specs.get(field)
        return item.value if item else None

    def add(field: str, value: object, source_field: str | None = None) -> None:
        if value is None:
            return
        item = parsed_specs.get(source_field or field)
        source_text = item.source_text if item else product.context
        evidence[field] = FactEvidence(field=field, value=value, source_text=source_text)

    cpu_raw = raw("cpu")
    cpu_tier = infer_cpu_tier(cpu_raw or product.name)
    gpu_raw = raw("gpu")
    gpu_type = infer_gpu_type(gpu_raw)
    ram_gb = parse_capacity_gb(raw("ram") or "")
    storage_gb = parse_capacity_gb(raw("storage") or "")
    screen_inches = parse_float_with_unit(raw("screen") or "", "inch")
    refresh_hz = parse_int_with_unit(raw("refresh_rate") or "", "hz")
    battery_wh = parse_float_with_unit(raw("battery") or "", "wh")
    weight_source = raw("weight")
    weight_kg = parse_float_with_unit(weight_source or "", "kg")
    if weight_kg is None:
        weight_source = _extract_weight_source(product.context)
        weight_kg = parse_float_with_unit(weight_source or "", "kg")
    os_value = raw("os")
    price = _price_value(product.price) if product.price else None

    add("price_value", price)
    add("cpu_raw", cpu_raw, "cpu")
    add("cpu_tier", cpu_tier, "cpu")
    add("gpu_raw", gpu_raw, "gpu")
    add("gpu_type", gpu_type, "gpu")
    add("ram_gb", ram_gb, "ram")
    add("storage_gb", storage_gb, "storage")
    add("screen_inches", screen_inches, "screen")
    add("refresh_hz", refresh_hz, "refresh_rate")
    add("battery_wh", battery_wh, "battery")
    add("weight_kg", weight_kg, "weight")
    if weight_kg is not None and "weight_kg" not in evidence:
        evidence["weight_kg"] = FactEvidence(
            field="weight_kg",
            value=weight_kg,
            source_text=weight_source or product.context,
        )
    add("os", os_value, "os")

    return NormalizedProductFacts(
        code=product.code,
        name=product.name,
        brand=product.brand,
        category=product.category,
        price_value=price,
        cpu_raw=cpu_raw,
        cpu_tier=cpu_tier,
        gpu_raw=gpu_raw,
        gpu_type=gpu_type,
        ram_gb=ram_gb,
        storage_gb=storage_gb,
        screen_inches=screen_inches,
        refresh_hz=refresh_hz,
        battery_wh=battery_wh,
        weight_kg=weight_kg,
        os=os_value,
        evidence_map=evidence,
    )


def _extract_weight_source(context: str) -> str | None:
    if not context:
        return None
    compact = " ".join(context.split())
    match = re.search(
        r"[^.。!?]{0,80}\b\d+(?:[.,]\d+)?\s*kg\b[^.。!?]{0,80}",
        compact,
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else None


def normalize_products(products: list[CatalogProduct]) -> list[NormalizedProductFacts]:
    return [normalize_product(product) for product in products]
