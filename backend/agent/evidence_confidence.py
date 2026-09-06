"""Evidence confidence labels for advisor statements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.agent.product_facts import NormalizedProductFacts


ConfidenceLevel = Literal[
    "confirmed_by_catalog",
    "inferred_from_specs",
    "missing_data",
    "unsupported",
]


@dataclass(frozen=True)
class EvidenceConfidence:
    claim: str
    level: ConfidenceLevel
    fields: tuple[str, ...] = ()


CONFIRMED_FIELDS = {
    "price_value",
    "cpu_tier",
    "gpu_type",
    "ram_gb",
    "storage_gb",
    "screen_inches",
    "battery_wh",
    "weight_kg",
}


def confidence_for_field(product: NormalizedProductFacts, field_name: str) -> EvidenceConfidence:
    if getattr(product, field_name, None) is not None:
        return EvidenceConfidence(field_name, "confirmed_by_catalog", (field_name,))
    return EvidenceConfidence(field_name, "missing_data", (field_name,))


def build_response_confidence(
    products: tuple[NormalizedProductFacts, ...],
    requested_attributes: tuple[str, ...] = (),
) -> tuple[EvidenceConfidence, ...]:
    items: list[EvidenceConfidence] = []
    for attribute in requested_attributes:
        if any(getattr(product, attribute, None) is not None for product in products):
            items.append(EvidenceConfidence(attribute, "confirmed_by_catalog", (attribute,)))
        else:
            items.append(EvidenceConfidence(attribute, "missing_data", (attribute,)))
    if products and any(product.gpu_type or product.cpu_tier or product.ram_gb for product in products):
        items.append(
            EvidenceConfidence(
                "fit_assessment",
                "inferred_from_specs",
                ("cpu_tier", "gpu_type", "ram_gb", "storage_gb"),
            )
        )
    return tuple(items)


def confidence_summary(confidence: tuple[EvidenceConfidence, ...]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {
        "confirmed_by_catalog": [],
        "inferred_from_specs": [],
        "missing_data": [],
        "unsupported": [],
    }
    for item in confidence:
        summary[item.level].append(item.claim)
    return summary
