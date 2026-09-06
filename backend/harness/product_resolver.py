from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from backend.services.catalog import CatalogProduct
from backend.services.conversation import CandidateRef, DecisionContext, normalize_text


class ProductResolution(BaseModel):
    resolved: bool
    code: str | None = None
    name: str | None = None
    source: Literal[
        "exact_code",
        "exact_name",
        "previous_candidate_name",
        "previous_candidate_brand",
        "ordinal",
        "focused_product",
        "correction",
        "unresolved",
    ] = "unresolved"
    confidence: float = 0.0
    reason: str | None = None
    ambiguous_candidates: list[CandidateRef] = Field(default_factory=list)


_FOLLOW_UP_TERMS = (
    "phan tich",
    "noi ro",
    "noi ky",
    "noi ki",
    "chi tiet",
    "ki di",
    "ky di",
    "may nay",
    "con nay",
    "mau nay",
    "on khong",
    "dang mua khong",
    "ban dang noi",
    "bao kg",
    "trong luong",
    "pin",
    "bao hanh",
    "ben khong",
)
_CORRECTION_TERMS = ("co ma", "khong phai", "y toi la", "y minh la")
_SELECTION_TERMS = ("di", "lay", "chon", "muon con", "mau nay", "con nay")
_ORDINAL_MAP = {
    "1": 0,
    "mot": 0,
    "dau": 0,
    "2": 1,
    "hai": 1,
    "3": 2,
    "ba": 2,
}


def _contains_phrase(normalized: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", normalized) is not None


def _code_tokens(text: str) -> list[str]:
    return re.findall(r"\b[A-Z0-9]{5,}\b", text.upper())


def _match_exact_code(query: str, context: DecisionContext) -> ProductResolution | None:
    known: dict[str, CandidateRef] = {
        candidate.code.upper(): candidate for candidate in context.last_shown_candidates
    }
    if context.focused_product_code and context.focused_product_name:
        known.setdefault(
            context.focused_product_code.upper(),
            CandidateRef(
                code=context.focused_product_code,
                name=context.focused_product_name,
                brand=None,
                category=context.last_category,
            ),
        )
    for token in _code_tokens(query):
        candidate = known.get(token)
        if candidate is not None:
            return ProductResolution(
                resolved=True,
                code=candidate.code,
                name=candidate.name,
                source="exact_code",
                confidence=0.99,
                reason=f"Matched exact previous code {token}.",
            )
    return None


def _extract_ordinal(normalized: str, candidates: list[CandidateRef]) -> ProductResolution | None:
    if not candidates:
        return None
    if "cuoi" in normalized:
        candidate = candidates[-1]
        return ProductResolution(
            resolved=True,
            code=candidate.code,
            name=candidate.name,
            source="ordinal",
            confidence=0.95,
            reason="Resolved from last-position ordinal reference.",
        )
    for token, index in _ORDINAL_MAP.items():
        if re.search(rf"\b((con|may|mau)\s+(thu\s+)?{token}|thu\s+{token}|{token}\s+(dau|cuoi))\b", normalized):
            if 0 <= index < len(candidates):
                candidate = candidates[index]
                return ProductResolution(
                    resolved=True,
                    code=candidate.code,
                    name=candidate.name,
                    source="ordinal",
                    confidence=0.95,
                    reason=f"Resolved from ordinal {token}.",
                )
    return None


def _candidate_name_match_score(normalized_query: str, candidate: CandidateRef) -> float:
    normalized_name = normalize_text(candidate.name)
    if not normalized_name:
        return 0.0
    if normalized_name in normalized_query:
        return 1.0
    if normalized_query in normalized_name and len(normalized_query) >= 6:
        return 0.96

    query_tokens = set(normalized_query.split())
    name_tokens = set(normalized_name.split())
    shared = query_tokens & name_tokens
    if not shared:
        return 0.0
    token_score = len(shared) / max(1, min(len(name_tokens), len(query_tokens)))
    if candidate.brand and normalize_text(candidate.brand) in normalized_query:
        token_score += 0.15
    model_tokens = [token for token in name_tokens if any(ch.isdigit() for ch in token)]
    if model_tokens and any(token in query_tokens for token in model_tokens):
        token_score += 0.2
    return min(token_score, 0.99)


def _match_by_name_or_brand(
    normalized: str, candidates: list[CandidateRef], *, correction: bool = False
) -> ProductResolution | None:
    if not candidates:
        return None

    scored = sorted(
        (
            (_candidate_name_match_score(normalized, candidate), candidate)
            for candidate in candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    top_score, top_candidate = scored[0]
    if top_score >= 0.75:
        return ProductResolution(
            resolved=True,
            code=top_candidate.code,
            name=top_candidate.name,
            source="correction" if correction else "previous_candidate_name",
            confidence=top_score,
            reason="Resolved from previous candidate name overlap.",
        )

    brands = {}
    for candidate in candidates:
        if candidate.brand:
            brands.setdefault(normalize_text(candidate.brand), []).append(candidate)
    brand_hits = [
        candidate_list
        for brand, candidate_list in brands.items()
        if brand and brand in normalized
    ]
    if not brand_hits:
        return None
    matched = brand_hits[0]
    if len(matched) == 1:
        candidate = matched[0]
        return ProductResolution(
            resolved=True,
            code=candidate.code,
            name=candidate.name,
            source="correction" if correction else "previous_candidate_brand",
            confidence=0.88,
            reason="Resolved from a unique brand inside previous candidates.",
        )
    return ProductResolution(
        resolved=False,
        source="unresolved",
        confidence=0.0,
        reason="Brand reference is ambiguous within previous candidates.",
        ambiguous_candidates=matched,
    )


def _focused_follow_up(normalized: str, context: DecisionContext) -> ProductResolution | None:
    if not context.focused_product_code or not context.focused_product_name:
        return None
    if any(term in normalized for term in _FOLLOW_UP_TERMS):
        return ProductResolution(
            resolved=True,
            code=context.focused_product_code,
            name=context.focused_product_name,
            source="focused_product",
            confidence=0.92,
            reason="Follow-up detail request kept the focused product.",
        )
    return None


def resolve_product_reference(
    user_query: str,
    context: DecisionContext,
    current_candidates: list[CatalogProduct] | None = None,
) -> ProductResolution:
    from backend.agent.product_resolver import (
        resolve_product_reference as resolve_agent_product_reference,
    )
    from backend.agent.state import AgentState

    agent_result = resolve_agent_product_reference(
        user_query,
        AgentState.from_decision_context(context),
        current_candidates=current_candidates,
    )
    source_map = {
        "exact_code": "exact_code",
        "exact_name": "previous_candidate_name",
        "previous_candidate_name": "previous_candidate_name",
        "ordinal_selection": "ordinal",
        "focused_product": "focused_product",
        "brand_within_last_candidates": "previous_candidate_brand",
        "correction": "correction",
        "ambiguous": "unresolved",
        "unresolved": "unresolved",
    }
    converted_ambiguous = [
        CandidateRef(
            code=candidate.code,
            name=candidate.name,
            brand=candidate.brand,
            category=candidate.category,
            price=candidate.price_value,
            specs_summary=", ".join(candidate.summary_specs),
        )
        for candidate in agent_result.ambiguous_candidates
    ]
    if (
        agent_result.resolved
        or agent_result.source in {"ambiguous", "correction"}
    ):
        return ProductResolution(
            resolved=agent_result.resolved,
            code=agent_result.code,
            name=agent_result.name,
            source=source_map.get(agent_result.source, "unresolved"),
            confidence=agent_result.confidence,
            reason=agent_result.reason,
            ambiguous_candidates=converted_ambiguous,
        )

    del current_candidates
    normalized = normalize_text(user_query)
    candidates = list(context.last_shown_candidates)

    exact = _match_exact_code(user_query, context)
    if exact is not None:
        return exact

    ordinal = _extract_ordinal(normalized, candidates)
    if ordinal is not None:
        return ordinal

    is_correction = any(_contains_phrase(normalized, term) for term in _CORRECTION_TERMS)
    if is_correction:
        match = _match_by_name_or_brand(normalized, candidates, correction=True)
        if match is not None:
            return match
        if context.focused_product_code and any(
            _contains_phrase(normalized, term) for term in ("khong phai", "co ma")
        ):
            return ProductResolution(
                resolved=True,
                code=context.focused_product_code,
                name=context.focused_product_name,
                source="correction",
                confidence=0.7,
                reason="Correction fell back to the focused product.",
            )

    match = _match_by_name_or_brand(normalized, candidates)
    if match is not None:
        return match

    focus = _focused_follow_up(normalized, context)
    if focus is not None:
        return focus

    if context.focused_product_code and any(term in normalized for term in _SELECTION_TERMS):
        return ProductResolution(
            resolved=True,
            code=context.focused_product_code,
            name=context.focused_product_name,
            source="focused_product",
            confidence=0.6,
            reason="Selection-like follow-up reused the focused product.",
        )

    return ProductResolution(
        resolved=False,
        source="unresolved",
        confidence=0.0,
        reason="No grounded product reference matched previous candidates or focus.",
    )
