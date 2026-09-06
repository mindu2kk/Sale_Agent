"""Resolve product references against current agent state."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from backend.agent.intent_router import extract_constraints
from backend.agent.spec_parser import normalize_text
from backend.agent.state import AgentState, CandidateRef
from backend.services.catalog import CatalogProduct


ResolutionSource = Literal[
    "exact_code",
    "exact_name",
    "previous_candidate_name",
    "ordinal_selection",
    "focused_product",
    "brand_within_last_candidates",
    "correction",
    "ambiguous",
    "unresolved",
]


@dataclass(frozen=True)
class ProductResolution:
    resolved: bool
    code: str | None = None
    name: str | None = None
    source: ResolutionSource = "unresolved"
    confidence: float = 0.0
    reason: str | None = None
    ambiguous_candidates: tuple[CandidateRef, ...] = ()
    should_clear_focus: bool = False


CORRECTION_TERMS = ("co ma", "khong phai", "y toi la", "y minh la")
FOCUSED_REFERENCE_TERMS = (
    "may do",
    "con do",
    "mau do",
    "may nay",
    "con nay",
    "mau nay",
    "mau vua roi",
    "ban dang noi",
    "phan tich",
    "noi ro",
    "noi ky",
    "noi ki",
    "chi tiet",
    "bao kg",
    "trong luong",
    "pin",
    "bao hanh",
    "ben khong",
    "on khong",
)
SELECTION_TERMS = ("di", "lay", "chon", "mua", "con nay", "mau nay")
ORDINALS = {
    "1": 0,
    "mot": 0,
    "dau": 0,
    "2": 1,
    "hai": 1,
    "3": 2,
    "ba": 2,
}


def resolve_product_reference(
    user_query: str,
    state: AgentState,
    current_candidates: list[CatalogProduct] | None = None,
) -> ProductResolution:
    candidates = list(state.last_shown_candidates)
    if current_candidates:
        candidates.extend(CandidateRef.from_product(product) for product in current_candidates)
    candidates = _dedupe_candidates(candidates)
    normalized = normalize_text(user_query)

    correction = _is_correction(normalized)
    exact = _match_exact_code_or_model(user_query, normalized, state, candidates)
    if exact is not None:
        return _as_correction(exact) if correction else exact

    ordinal = _match_ordinal(normalized, candidates)
    if ordinal is not None:
        return _as_correction(ordinal) if correction else ordinal

    named_exact = _match_candidate_name(normalized, candidates)
    if named_exact is not None:
        return _as_correction(named_exact) if correction else named_exact

    brand_scoped = _match_brand_scoped_reference(normalized, candidates)
    if brand_scoped is not None and not correction:
        return brand_scoped
    focused_brand_scoped = _match_brand_scoped_focus(normalized, state)
    if focused_brand_scoped is not None and not correction:
        return focused_brand_scoped

    if not correction and _has_fresh_search_constraints(user_query):
        return ProductResolution(
            resolved=False,
            source="unresolved",
            confidence=0.0,
            reason="Fresh search constraints should be handled by filtered retrieval, not previous-candidate grounding.",
        )

    if correction:
        corrected = _match_name_or_brand(normalized, candidates, correction=True)
        if corrected is not None:
            return corrected
        return ProductResolution(
            resolved=False,
            source="correction",
            confidence=0.4,
            reason="Correction did not identify a replacement product.",
            should_clear_focus=True,
        )

    named = _match_name_or_brand(normalized, candidates)
    if named is not None:
        return named

    focused = _match_focused_product(normalized, state)
    if focused is not None:
        return focused

    return ProductResolution(
        resolved=False,
        source="unresolved",
        confidence=0.0,
        reason="No product reference matched state or candidates.",
    )


def _dedupe_candidates(candidates: list[CandidateRef]) -> list[CandidateRef]:
    seen: set[str] = set()
    deduped: list[CandidateRef] = []
    for candidate in candidates:
        key = candidate.code.casefold()
        if key in seen:
            continue
        deduped.append(candidate)
        seen.add(key)
    return deduped


def _contains_phrase(normalized: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", normalized) is not None


def _is_correction(normalized: str) -> bool:
    return any(_contains_phrase(normalized, term) for term in CORRECTION_TERMS)


def _has_fresh_search_constraints(raw_query: str) -> bool:
    constraints = extract_constraints(raw_query)
    return any(
        constraints.get(key) is not None
        for key in (
            "category",
            "brand",
            "cpu_tier",
            "gpu_type",
            "ram_gb",
            "storage_gb",
            "use_case",
            "min_price",
            "max_price",
            "target_price",
        )
    )


def _candidate_by_focused_product(state: AgentState) -> CandidateRef | None:
    if not state.focused_product_code or not state.focused_product_name:
        return None
    return CandidateRef(
        code=state.focused_product_code,
        name=state.focused_product_name,
        category=state.active_category,
    )


def _match_exact_code_or_model(
    raw_query: str,
    normalized: str,
    state: AgentState,
    candidates: list[CandidateRef],
) -> ProductResolution | None:
    known = {candidate.code.upper(): candidate for candidate in candidates}
    focused = _candidate_by_focused_product(state)
    if focused:
        known.setdefault(focused.code.upper(), focused)

    for token in re.findall(r"\b[A-Z0-9]{5,}\b", raw_query.upper()):
        if token in known:
            candidate = known[token]
            return _resolution(candidate, "exact_code", 0.99, f"Matched exact code {token}.")

    scored = sorted(
        (
            (_name_match_score(normalized, candidate), candidate)
            for candidate in candidates + ([focused] if focused else [])
            if candidate is not None
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if scored and scored[0][0] >= 0.92:
        candidate = scored[0][1]
        return _resolution(candidate, "exact_name", scored[0][0], "Matched exact or near-exact product name.")
    return None


def _match_ordinal(normalized: str, candidates: list[CandidateRef]) -> ProductResolution | None:
    if not candidates:
        return None
    if _contains_phrase(normalized, "mau cuoi") or _contains_phrase(normalized, "con cuoi"):
        return _resolution(candidates[-1], "ordinal_selection", 0.95, "Selected the last shown candidate.")
    for token, index in ORDINALS.items():
        if re.search(
            rf"\b((con|may|mau)\s+(thu\s+)?{token}|thu\s+{token}|{token}\s+(dau|cuoi))\b",
            normalized,
        ):
            if 0 <= index < len(candidates):
                return _resolution(candidates[index], "ordinal_selection", 0.95, f"Selected candidate #{index + 1}.")
    return None


def _match_name_or_brand(
    normalized: str,
    candidates: list[CandidateRef],
    *,
    correction: bool = False,
) -> ProductResolution | None:
    scored = sorted(
        ((_name_match_score(normalized, candidate), candidate) for candidate in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    if scored and scored[0][0] >= 0.74:
        source: ResolutionSource = "correction" if correction else "previous_candidate_name"
        return _resolution(scored[0][1], source, scored[0][0], "Matched previous candidate name.")

    brand_matches: list[CandidateRef] = []
    for candidate in candidates:
        brand = normalize_text(candidate.brand or "")
        if brand and _contains_phrase(normalized, brand):
            brand_matches.append(candidate)
    if not brand_matches:
        return None
    if len(brand_matches) == 1:
        source = "correction" if correction else "brand_within_last_candidates"
        return _resolution(brand_matches[0], source, 0.88, "Matched a unique brand in previous candidates.")
    return ProductResolution(
        resolved=False,
        source="ambiguous",
        confidence=0.0,
        reason="Brand reference matched multiple previous candidates.",
        ambiguous_candidates=tuple(brand_matches),
        should_clear_focus=correction,
    )


def _match_candidate_name(
    normalized: str,
    candidates: list[CandidateRef],
) -> ProductResolution | None:
    scored = sorted(
        ((_name_match_score(normalized, candidate), candidate) for candidate in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    if scored and scored[0][0] >= 0.74:
        return _resolution(
            scored[0][1],
            "previous_candidate_name",
            scored[0][0],
            "Matched previous candidate name.",
        )
    return None


def _match_brand_scoped_reference(
    normalized: str,
    candidates: list[CandidateRef],
) -> ProductResolution | None:
    brand_ref = re.search(r"\b(?:mau|may|con)?\s*([a-z0-9]+)\s+(?:do|nay|vua roi)\b", normalized)
    if not brand_ref:
        return None
    brand_text = brand_ref.group(1)
    brand_matches = [
        candidate
        for candidate in candidates
        if normalize_text(candidate.brand or "") == brand_text
    ]
    if len(brand_matches) != 1:
        return None
    return _resolution(
        brand_matches[0],
        "brand_within_last_candidates",
        0.9,
        "Matched a unique brand-scoped follow-up in previous candidates.",
    )


def _match_brand_scoped_focus(
    normalized: str,
    state: AgentState,
) -> ProductResolution | None:
    focused = _candidate_by_focused_product(state)
    if focused is None:
        return None
    match = re.search(r"\b(?:mau|may|con)?\s*([a-z0-9]+)\s+(?:do|nay|vua roi)\b", normalized)
    if not match:
        return None
    brand_text = match.group(1)
    focused_name = normalize_text(focused.name)
    if not _contains_phrase(focused_name, brand_text):
        return None
    return _resolution(
        focused,
        "focused_product",
        0.88,
        "Brand-scoped follow-up reused the focused product.",
    )


def _match_focused_product(normalized: str, state: AgentState) -> ProductResolution | None:
    focused = _candidate_by_focused_product(state)
    if focused is None:
        return None
    if any(_contains_phrase(normalized, term) for term in FOCUSED_REFERENCE_TERMS):
        return _resolution(focused, "focused_product", 0.92, "Pronoun/detail follow-up kept the focused product.")
    if any(_contains_phrase(normalized, term) for term in SELECTION_TERMS):
        return _resolution(focused, "focused_product", 0.6, "Selection-like follow-up reused focus.")
    return None


def _name_match_score(normalized_query: str, candidate: CandidateRef) -> float:
    normalized_name = normalize_text(candidate.name)
    if not normalized_name:
        return 0.0
    if _contains_phrase(normalized_query, normalized_name):
        return 1.0
    if normalized_query in normalized_name and len(normalized_query) >= 6:
        return 0.96
    query_tokens = set(normalized_query.split())
    name_tokens = set(normalized_name.split())
    shared = query_tokens & name_tokens
    if not shared:
        return 0.0

    score = len(shared) / max(1, min(len(query_tokens), len(name_tokens)))
    if candidate.brand and _contains_phrase(normalized_query, normalize_text(candidate.brand)):
        score += 0.12
    model_tokens = {
        token
        for token in name_tokens
        if len(token) >= 4 and any(char.isdigit() for char in token)
    }
    model_hits = model_tokens & query_tokens
    if model_hits:
        score += min(0.3, len(model_hits) * 0.12)
    return min(score, 0.99)


def _resolution(
    candidate: CandidateRef,
    source: ResolutionSource,
    confidence: float,
    reason: str,
) -> ProductResolution:
    return ProductResolution(
        resolved=True,
        code=candidate.code,
        name=candidate.name,
        source=source,
        confidence=confidence,
        reason=reason,
    )


def _as_correction(resolution: ProductResolution) -> ProductResolution:
    return ProductResolution(
        resolved=resolution.resolved,
        code=resolution.code,
        name=resolution.name,
        source="correction",
        confidence=resolution.confidence,
        reason=resolution.reason,
        ambiguous_candidates=resolution.ambiguous_candidates,
        should_clear_focus=not resolution.resolved,
    )
