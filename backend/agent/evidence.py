"""Evidence ledger for deterministic sales-advisor guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.agent.product_facts import FactEvidence, NormalizedProductFacts


@dataclass(frozen=True)
class EvidenceLedger:
    products_used: tuple[str, ...] = ()
    facts_used: tuple[FactEvidence, ...] = ()
    missing_fields: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    constraints_checked: dict[str, object] = field(default_factory=dict)

    def has_fact(self, product_code: str, field_name: str) -> bool:
        return any(
            fact.field == field_name
            and self.constraints_checked.get("fact_product:" + field_name, product_code) == product_code
            for fact in self.facts_used
        )


def build_evidence_ledger(
    products: list[NormalizedProductFacts] | tuple[NormalizedProductFacts, ...],
    *,
    requested_fields: tuple[str, ...] = (),
    constraints_checked: dict[str, object] | None = None,
) -> EvidenceLedger:
    facts: list[FactEvidence] = []
    missing: list[str] = []
    checked = dict(constraints_checked or {})

    for product in products:
        for field_name, evidence in product.evidence_map.items():
            facts.append(evidence)
            checked.setdefault("fact_product:" + field_name, product.code)
        for field_name in requested_fields:
            if getattr(product, field_name, None) is None:
                missing.append(field_name)

    return EvidenceLedger(
        products_used=tuple(product.code for product in products),
        facts_used=tuple(facts),
        missing_fields=tuple(dict.fromkeys(missing)),
        constraints_checked=checked,
    )
