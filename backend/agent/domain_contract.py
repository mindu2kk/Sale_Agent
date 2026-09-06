"""AURA sales-advisor domain contract.

This module is the system contract for grounded commerce answers. It keeps the
business invariants explicit so fixes become spec/eval-driven instead of
single-failure patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.agent.intent_router import IntentRoute
from backend.agent.product_facts import NormalizedProductFacts
from backend.agent.response_composer import AdvisorResponse
from backend.agent.search_filters import screen_inches_match
from backend.agent.state import ProductConstraints, QueryFrame


ContractViolationCode = Literal[
    "REQUESTED_ATTRIBUTE_NOT_VISIBLE",
    "NEW_CONSTRAINT_DID_NOT_SEARCH",
    "FOCUSED_PRODUCT_HIJACK",
    "QUERY_CONTINUATION_NOT_EXCLUDING_PREVIOUS",
    "PRODUCT_DOES_NOT_MATCH_FILTER",
    "MISSING_FIELD_HALLUCINATION",
    "UNSUPPORTED_STRONG_CLAIM",
    "TEXT_CARD_MISMATCH",
    "FIELD_QUESTION_DUMPED_FULL_PRODUCT",
    "MISSING_RESPONSE_MODE",
]


@dataclass(frozen=True)
class DomainRule:
    rule_id: str
    summary: str


@dataclass(frozen=True)
class ContractViolation:
    code: ContractViolationCode
    rule_id: str
    message: str


@dataclass(frozen=True)
class ContractCheckResult:
    passed: bool
    violations: tuple[ContractViolation, ...] = ()


DOMAIN_RULES: tuple[DomainRule, ...] = (
    DomainRule("R1", "Requested attributes must be visible in answer and product-card display specs."),
    DomainRule("R2", "Fresh brand/category/CPU/GPU/RAM/SSD/budget constraints must override focused product."),
    DomainRule("R3", "Focused product may be used only for pronoun follow-ups without new constraints."),
    DomainRule("R4", "Query continuations inherit last QueryFrame and exclude previously shown products."),
    DomainRule("R5", "Dell i7 means brand Dell and CPU tier i7; Core 7/Ryzen 7/Core Ultra 7 are not i7."),
    DomainRule("R6", "Dedicated GPU requests require gpu_type=dedicated; Intel Graphics/UHD/Iris Xe are not card roi."),
    DomainRule("R7", "Missing fields must be disclosed; do not hallucinate or broad-search a replacement."),
    DomainRule("R8", "Strong claims need corresponding evidence; otherwise do not choose a winner."),
    DomainRule("R9", "Answer text and product cards must reference the same product codes."),
    DomainRule("R10", "Every answer must declare a response_mode."),
)


def check_domain_contract(
    *,
    route: IntentRoute,
    query_frame: QueryFrame,
    response: AdvisorResponse,
    products: tuple[NormalizedProductFacts, ...],
    focused_product_code: str | None = None,
    exclude_codes: set[str] | None = None,
) -> ContractCheckResult:
    violations: list[ContractViolation] = []

    if not response.answer_mode:
        violations.append(_violation("MISSING_RESPONSE_MODE", "R10", "Response mode is required."))

    missing_attributes = [
        attribute
        for attribute in query_frame.requested_attributes
        if attribute not in response.displayed_attributes
        and any(getattr(product, attribute, None) is not None for product in products)
    ]
    if missing_attributes:
        violations.append(
            _violation(
                "REQUESTED_ATTRIBUTE_NOT_VISIBLE",
                "R1",
                "Requested attributes were not exposed in the response/card display contract.",
            )
        )

    if route.has_new_constraints and focused_product_code and response.related_product_codes == (focused_product_code,):
        focused = next((product for product in products if product.code == focused_product_code), None)
        if focused and not product_matches_constraints(focused, query_frame.constraints):
            violations.append(
                _violation(
                    "FOCUSED_PRODUCT_HIJACK",
                    "R2",
                    "A fresh constrained search reused a focused product that does not satisfy the frame.",
                )
            )

    if route.constraints.get("exclude_previous") is True:
        excluded = exclude_codes or set()
        if excluded & set(response.related_product_codes):
            violations.append(
                _violation(
                    "QUERY_CONTINUATION_NOT_EXCLUDING_PREVIOUS",
                    "R4",
                    "A continuation returned a product that should have been excluded.",
                )
            )

    if route.intent != "comparison":
        for product in products:
            if product.code not in response.related_product_codes:
                continue
            if not product_matches_constraints(product, query_frame.constraints):
                violations.append(
                    _violation(
                        "PRODUCT_DOES_NOT_MATCH_FILTER",
                        "R5",
                        f"{product.code} does not satisfy the active query frame.",
                    )
                )

    if _mentions_unknown_product(response.answer_text, response.related_product_codes, products):
        violations.append(
            _violation(
                "TEXT_CARD_MISMATCH",
                "R9",
                "Answer text mentions a product outside related_product_codes.",
            )
        )

    return ContractCheckResult(passed=not violations, violations=tuple(violations))


def product_matches_constraints(
    product: NormalizedProductFacts,
    constraints: ProductConstraints,
) -> bool:
    checks = (
        ("category", constraints.category, product.category),
        ("brand", constraints.brand, product.brand),
        ("cpu_tier", constraints.cpu_tier, product.cpu_tier),
        ("gpu_type", constraints.gpu_type, product.gpu_type),
        ("ram_gb", constraints.ram_gb, product.ram_gb),
        ("storage_gb", constraints.storage_gb, product.storage_gb),
    )
    for _, expected, actual in checks:
        if expected is not None and actual != expected:
            return False
    if not screen_inches_match(product.screen_inches, constraints.screen_inches):
        return False
    if constraints.min_price is not None and (
        product.price_value is None or product.price_value < constraints.min_price
    ):
        return False
    if constraints.max_price is not None and (
        product.price_value is None or product.price_value > constraints.max_price
    ):
        return False
    return True


def _mentions_unknown_product(
    answer_text: str,
    related_codes: tuple[str, ...],
    products: tuple[NormalizedProductFacts, ...],
) -> bool:
    related = set(related_codes)
    for product in products:
        if product.code in related:
            continue
        if product.code in answer_text or product.name in answer_text:
            return True
    return False


def _violation(
    code: ContractViolationCode,
    rule_id: str,
    message: str,
) -> ContractViolation:
    return ContractViolation(code=code, rule_id=rule_id, message=message)
