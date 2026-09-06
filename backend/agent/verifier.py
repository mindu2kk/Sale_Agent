"""Business guardrails for grounded advisor responses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.agent.evidence import EvidenceLedger
from backend.agent.product_facts import NormalizedProductFacts
from backend.agent.search_filters import screen_inches_match
from backend.agent.spec_parser import normalize_text
from backend.agent.state import ProductConstraints


@dataclass(frozen=True)
class AdvisorResponseContract:
    answer_text: str
    related_product_codes: tuple[str, ...]
    answer_mode: str
    missing_fields: tuple[str, ...] = ()
    displayed_attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationFailure:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    failures: tuple[VerificationFailure, ...] = ()
    forced_response_mode: str | None = None
    blocked_claims: tuple[str, ...] = ()


def verify_response(
    response: AdvisorResponseContract,
    products: list[NormalizedProductFacts] | tuple[NormalizedProductFacts, ...],
    ledger: EvidenceLedger,
    *,
    constraints: ProductConstraints | None = None,
    asked_field: str | None = None,
    focused_product_code: str | None = None,
    requested_attributes: tuple[str, ...] = (),
) -> VerifierResult:
    failures: list[VerificationFailure] = []
    forced_mode: str | None = None
    blocked_claims: list[str] = []
    product_by_code = {product.code: product for product in products}
    related_codes = set(response.related_product_codes)

    unknown_codes = related_codes - set(product_by_code)
    if unknown_codes:
        failures.append(
            VerificationFailure(
                code="related_product_not_in_evidence",
                severity="blocker",
                message="Related products include codes outside the evidence set.",
            )
        )

    diacritic_failure = _vietnamese_diacritic_failure(response.answer_text)
    if diacritic_failure:
        failures.append(
            VerificationFailure(
                code="VIETNAMESE_DIACRITICS_REQUIRED",
                severity="blocker",
                message=diacritic_failure,
            )
        )

    unsupported_claims = _unsupported_advisory_claims(response.answer_text, ledger)
    for claim in unsupported_claims:
        blocked_claims.append(claim)
        failures.append(
            VerificationFailure(
                code="UNSUPPORTED_ADVISORY_CLAIM",
                severity="blocker",
                message=f"Unsupported advisory claim: {claim}.",
            )
        )

    if focused_product_code and asked_field:
        if response.related_product_codes != (focused_product_code,):
            failures.append(
                VerificationFailure(
                    code="focused_field_broad_search",
                    severity="blocker",
                    message="Focused product field question returned another candidate set.",
                )
            )
        focused = product_by_code.get(focused_product_code)
        if focused is not None and getattr(focused, asked_field, None) is None:
            forced_mode = "missing_field"
            if asked_field not in response.missing_fields:
                failures.append(
                    VerificationFailure(
                        code="missing_field_not_disclosed",
                        severity="blocker",
                        message=f"Missing field {asked_field} must be disclosed.",
                    )
                )

    if constraints is not None:
        for product in products:
            if product.code not in related_codes:
                continue
            failures.extend(_verify_constraints(product, constraints))

    missing_display = _missing_requested_attributes(
        response,
        products,
        requested_attributes,
    )
    if missing_display:
        failures.append(
            VerificationFailure(
                code="requested_attributes_not_displayed",
                severity="blocker",
                message="Requested product attributes must appear in the answer/card display contract.",
            )
        )

    text_codes = _codes_and_known_names_in_text(response.answer_text, products)
    if text_codes and related_codes and text_codes != related_codes:
        failures.append(
            VerificationFailure(
                code="answer_cards_mismatch",
                severity="blocker",
                message="Products mentioned in text differ from related product cards.",
            )
        )

    if _claims_missing_battery(response.answer_text):
        battery_products = [
            product for product in products if product.code in related_codes and product.battery_wh is not None
        ]
        if battery_products:
            failures.append(
                VerificationFailure(
                    code="contradictory_missing_battery",
                    severity="blocker",
                    message="Answer says battery is missing although battery evidence exists.",
                )
            )

    if _claims_durability_winner(response.answer_text):
        has_durability_evidence = any(
            fact.field in {"durability", "material", "warranty", "certification"}
            for fact in ledger.facts_used
        )
        if not has_durability_evidence:
            blocked_claims.append("durability_best")
            failures.append(
                VerificationFailure(
                    code="unsupported_durability_winner",
                    severity="blocker",
                    message="Cannot claim a most-durable winner without durability evidence.",
                )
            )

    return VerifierResult(
        passed=not failures,
        failures=tuple(failures),
        forced_response_mode=forced_mode,
        blocked_claims=tuple(blocked_claims),
    )


def _verify_constraints(
    product: NormalizedProductFacts,
    constraints: ProductConstraints,
) -> list[VerificationFailure]:
    failures: list[VerificationFailure] = []
    checks = (
        ("brand", constraints.brand, product.brand),
        ("category", constraints.category, product.category),
        ("cpu_tier", constraints.cpu_tier, product.cpu_tier),
        ("gpu_type", constraints.gpu_type, product.gpu_type),
        ("ram_gb", constraints.ram_gb, product.ram_gb),
        ("storage_gb", constraints.storage_gb, product.storage_gb),
    )
    for field_name, expected, actual in checks:
        if expected is not None and actual != expected:
            failures.append(
                VerificationFailure(
                    code=f"constraint_mismatch_{field_name}",
                    severity="blocker",
                    message=f"{product.code} does not satisfy {field_name}={expected}.",
                )
            )
    if not screen_inches_match(product.screen_inches, constraints.screen_inches):
        failures.append(
            VerificationFailure(
                code="constraint_mismatch_screen_inches",
                severity="blocker",
                message=f"{product.code} does not satisfy screen_inches={constraints.screen_inches}.",
            )
        )
    if constraints.min_price is not None:
        if product.price_value is None or product.price_value < constraints.min_price:
            failures.append(
                VerificationFailure(
                    code="constraint_mismatch_min_price",
                    severity="blocker",
                    message=f"{product.code} is below the requested minimum price.",
                )
            )
    if constraints.max_price is not None:
        if product.price_value is None or product.price_value > constraints.max_price:
            failures.append(
                VerificationFailure(
                    code="constraint_mismatch_max_price",
                    severity="blocker",
                    message=f"{product.code} is above the requested maximum price.",
                )
            )
    return failures


def _missing_requested_attributes(
    response: AdvisorResponseContract,
    products: list[NormalizedProductFacts] | tuple[NormalizedProductFacts, ...],
    requested_attributes: tuple[str, ...],
) -> tuple[str, ...]:
    if not requested_attributes or not response.related_product_codes:
        return ()
    displayed = set(response.displayed_attributes)
    normalized_text = normalize_text(response.answer_text)
    missing: list[str] = []
    for attribute in requested_attributes:
        if attribute in displayed:
            continue
        related_products = [
            product for product in products if product.code in response.related_product_codes
        ]
        if any(_attribute_value_in_text(product, attribute, normalized_text) for product in related_products):
            continue
        missing.append(attribute)
    return tuple(missing)


def _attribute_value_in_text(
    product: NormalizedProductFacts,
    attribute: str,
    normalized_text: str,
) -> bool:
    value = getattr(product, attribute, None)
    if value is None:
        return False
    if attribute == "ram_gb":
        return f"ram {value}gb" in normalized_text or f"{value}gb ram" in normalized_text
    if attribute == "storage_gb":
        return (
            f"ssd {value}gb" in normalized_text
            or f"{value}gb ssd" in normalized_text
            or f"storage {value}gb" in normalized_text
        )
    if attribute == "gpu_type":
        return "card roi" in normalized_text or "gpu" in normalized_text or str(value).casefold() in normalized_text
    if attribute == "cpu_tier":
        return normalize_text(str(value)) in normalized_text
    if attribute == "price_value":
        return str(value) in normalized_text.replace(".", "")
    if attribute == "screen_inches":
        return f"{value:g} inch" in normalized_text or f"{value:g}inch" in normalized_text
    return normalize_text(str(value)) in normalized_text


def _codes_and_known_names_in_text(
    answer_text: str,
    products: list[NormalizedProductFacts] | tuple[NormalizedProductFacts, ...],
) -> set[str]:
    normalized_text = normalize_text(answer_text)
    codes: set[str] = set(re.findall(r"\b0\d{7}\b", answer_text))
    for product in products:
        product_name = normalize_text(product.name)
        if product_name and product_name in normalized_text:
            codes.add(product.code)
    return codes


def _claims_missing_battery(answer_text: str) -> bool:
    normalized = normalize_text(answer_text)
    return "chua co du lieu pin" in normalized or "thieu du lieu pin" in normalized


def _claims_durability_winner(answer_text: str) -> bool:
    normalized = normalize_text(answer_text)
    return any(
        phrase in normalized
        for phrase in (
            "ben nhat",
            "ben bi nhat",
            "do ben tot nhat",
            "mau ben nhat",
            "chon ve do ben",
        )
    )


def _vietnamese_diacritic_failure(answer_text: str) -> str | None:
    common_bad_phrases = (
        "Minh tim thay",
        "Minh chua thay",
        "Gia ",
        "Neu muon",
        "man hinh",
        "bo loc",
        "card roi",
        "duoi ",
        "catalog hien",
        " VND",
    )
    if any(phrase in answer_text for phrase in common_bad_phrases):
        return "Answer contains common Vietnamese-without-diacritics phrases."
    return None


def _unsupported_advisory_claims(answer_text: str, ledger: EvidenceLedger) -> tuple[str, ...]:
    normalized = normalize_text(answer_text)
    unsupported: list[str] = []
    has_battery = any(fact.field == "battery_wh" for fact in ledger.facts_used)
    has_durability = any(
        fact.field in {"durability", "material", "warranty", "certification"}
        for fact in ledger.facts_used
    )
    if any(term in normalized for term in ("pin trau", "pin tot", "thoi luong pin tot")) and not has_battery:
        unsupported.append("battery_quality")
    if any(term in normalized for term in ("ben hon", "ben nhat", "do ben tot")) and not has_durability:
        unsupported.append("durability")
    return tuple(dict.fromkeys(unsupported))
