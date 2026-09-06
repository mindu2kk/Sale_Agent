"""Priority-rule intent routing for the sales advisor agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from backend.agent.spec_parser import normalize_text
from backend.agent.state import AgentState
from backend.services.catalog import _detect_cpu_filters, _detect_gpu_filters, _parse_price_intent


AgentIntent = Literal[
    "correction",
    "new_filtered_search",
    "query_continuation",
    "focused_product_field_question",
    "product_selection",
    "comparison",
    "strong_claim_question",
    "product_detail",
    "broad_consulting",
    "hardware_explanation",
    "unknown",
]


@dataclass(frozen=True)
class IntentRoute:
    intent: AgentIntent
    confidence: float
    reason: str
    has_new_constraints: bool = False
    field_requested: str | None = None
    constraints: dict[str, object] = field(default_factory=dict)
    is_correction: bool = False


BRANDS = ("Dell", "HP", "Lenovo", "MSI", "Acer", "Asus", "Apple")
CORRECTION_TERMS = ("co ma", "khong phai", "y toi la", "y minh la")
PRONOUN_TERMS = ("may do", "con do", "mau do", "may nay", "con nay", "mau vua roi")
DETAIL_TERMS = ("phan tich", "chi tiet", "noi ro", "noi ky", "noi ki", "cau hinh", "thong so")
SELECTION_TERMS = (" di", "lay con", "chon", "mua con", "con thu", "mau dau", "mau cuoi")
CONTINUATION_TERMS = (
    "con may nao khong",
    "con mau nao khong",
    "con may nao nua khong",
    "con mau nao nua khong",
    "con cai nao khong",
    "mau khac khong",
    "may khac khong",
)
COMPARISON_TERMS = ("so sanh", "voi", "hay", "nen chon", "tot hon", "chon dell", "chon acer")
TOP_TWO_FOLLOWUP_TERMS = (
    "dang chu y",
    "vua hoi",
    "vua tu van",
    "ban vua tu van",
    "ban vua hoi",
    "o tren",
    "nay",
    "do",
)
STRONG_CLAIM_TERMS = (
    "ben nhat",
    "pin trau nhat",
    "pin tot nhat",
    "manh nhat",
    "khoe nhat",
    "re nhat",
    "tot nhat",
)
HARDWARE_EXPLANATION_TERMS = (
    "la gi",
    "dung de lam gi",
    "khac gi",
    "can khong",
    "co loi gi",
)


def route_intent(user_query: str, state: AgentState | None = None) -> IntentRoute:
    state = state or AgentState()
    normalized = normalize_text(user_query)
    base_constraints = extract_constraints(user_query)
    explicit_category = base_constraints.get("category") is not None
    if (
        base_constraints.get("category") is None
        and base_constraints.get("use_case") is not None
        and state.active_category
    ):
        base_constraints["category"] = state.active_category
    continuation = _is_query_continuation(normalized)
    refinement = _is_constraint_refinement(base_constraints, state)
    if explicit_category and _has_explicit_search_filters(base_constraints):
        # A self-contained category query starts a new search. It must not
        # silently retain narrower hardware filters from an earlier turn.
        refinement = False
    constraints = (
        _inherit_constraints(base_constraints, state)
        if continuation or refinement
        else base_constraints
    )
    if refinement and not continuation:
        constraints["inherits_previous"] = True
    if continuation or _asks_same_range_alternatives(normalized):
        constraints["exclude_previous"] = True
    field_requested = extract_field_question(user_query)
    correction = _has_any_phrase(normalized, CORRECTION_TERMS)
    pronoun_ref = _has_any_phrase(normalized, PRONOUN_TERMS)
    brand_scoped_ref = _is_brand_scoped_reference(normalized, base_constraints, state)
    hardware_explanation = _has_any_phrase(normalized, HARDWARE_EXPLANATION_TERMS)
    has_new_constraints = any(
        constraints.get(key) is not None
        for key in ("brand", "cpu_tier", "gpu_type", "category", "use_case")
    ) or any(
        constraints.get(key) is not None
        for key in ("min_price", "max_price", "target_price", "ram_gb", "storage_gb", "screen_inches")
    )

    if _is_top_two_followup(normalized, state):
        return IntentRoute(
            intent="comparison",
            confidence=0.93,
            reason="Top-two follow-up refers to the previous shown candidates.",
            constraints=constraints,
        )

    if correction:
        return IntentRoute(
            intent="correction",
            confidence=0.98,
            reason="Correction phrase has highest priority.",
            has_new_constraints=has_new_constraints,
            field_requested=field_requested,
            constraints=constraints,
            is_correction=True,
        )

    if hardware_explanation and not _has_shopping_scope_constraints(constraints):
        return IntentRoute(
            intent="hardware_explanation",
            confidence=0.9,
            reason="Hardware definition question without shopping filters.",
            constraints=constraints,
        )

    if (pronoun_ref or brand_scoped_ref) and field_requested:
        return IntentRoute(
            intent="focused_product_field_question",
            confidence=0.96,
            reason="Pronoun field question resolves against focused product when available.",
            field_requested=field_requested,
            constraints=constraints,
        )

    if brand_scoped_ref:
        return IntentRoute(
            intent="product_detail",
            confidence=0.9,
            reason="Brand-scoped pronoun resolves to a unique previous candidate.",
            field_requested=field_requested,
            constraints=constraints,
        )

    if _is_top_two_followup(normalized, state):
        return IntentRoute(
            intent="comparison",
            confidence=0.93,
            reason="Top-two follow-up refers to the previous shown candidates.",
            constraints=constraints,
        )

    if _is_comparison(normalized, constraints):
        return IntentRoute(
            intent="comparison",
            confidence=0.94,
            reason="Comparison wording or multiple brands detected.",
            constraints=constraints,
        )

    if re.search(r"\b0\d{7}\b", user_query):
        return IntentRoute(
            intent="product_detail",
            confidence=0.94,
            reason="Exact product code should resolve to product detail.",
            field_requested=field_requested,
            constraints=constraints,
        )

    if _is_product_selection(normalized):
        return IntentRoute(
            intent="product_selection",
            confidence=0.92,
            reason="Selection wording or ordinal points to previous candidates.",
            constraints=constraints,
        )

    if has_new_constraints:
        return IntentRoute(
            intent="query_continuation" if continuation else "new_filtered_search",
            confidence=0.96,
            reason=(
                "Continuation inherits the previous query frame."
                if continuation
                else "Fresh product constraints require a filtered search."
            ),
            has_new_constraints=True,
            field_requested=field_requested,
            constraints=constraints,
        )

    if _has_any_phrase(normalized, STRONG_CLAIM_TERMS):
        return IntentRoute(
            intent="strong_claim_question",
            confidence=0.94,
            reason="Superlative claim requires evidence-aware ranking.",
            field_requested=field_requested,
            constraints=constraints,
        )

    if _has_any_phrase(normalized, DETAIL_TERMS):
        return IntentRoute(
            intent="product_detail",
            confidence=0.86,
            reason="Detail wording without new constraints.",
            field_requested=field_requested,
            constraints=constraints,
        )

    if _is_broad_consulting(normalized):
        return IntentRoute(
            intent="broad_consulting",
            confidence=0.82,
            reason="Broad shopping request.",
            constraints=constraints,
        )

    if hardware_explanation:
        return IntentRoute(
            intent="hardware_explanation",
            confidence=0.78,
            reason="Feature explanation request.",
            constraints=constraints,
        )

    return IntentRoute(
        intent="unknown",
        confidence=0.5,
        reason="No priority rule matched.",
        field_requested=field_requested,
        constraints=constraints,
    )


def extract_constraints(user_query: str) -> dict[str, object]:
    normalized = normalize_text(user_query)
    price = _parse_price_intent(user_query)
    brands = [brand for brand in BRANDS if _contains_phrase(normalized, normalize_text(brand))]
    if _contains_phrase(normalized, "macbook") and "Apple" not in brands:
        brands.append("Apple")
    if _contains_phrase(normalized, "iphone") and "Apple" not in brands:
        brands.append("Apple")
    cpu_filters = _detect_cpu_filters(user_query)
    gpu_filters = _detect_gpu_filters(user_query)
    constraints: dict[str, object] = {
        "brand": brands[0] if len(brands) == 1 else None,
        "brands": tuple(brands),
        "cpu_tier": _cpu_filter_to_tier(cpu_filters[0]) if cpu_filters else None,
        "gpu_type": "dedicated" if _asks_dedicated_gpu(normalized, gpu_filters) else None,
        "ram_gb": _extract_capacity(normalized, ("ram",)),
        "storage_gb": _extract_capacity(normalized, ("ssd", "o cung", "bo nho trong", "storage")),
        "screen_inches": _extract_screen_inches(normalized),
        "category": _extract_category(normalized),
        "use_case": _extract_use_case(normalized),
        "min_price": price.minimum if price else None,
        "max_price": price.maximum if price else None,
        "target_price": price.target if price else None,
        "requested_attributes": extract_requested_attributes(user_query),
    }
    return constraints


def extract_field_question(user_query: str) -> str | None:
    normalized = normalize_text(user_query)
    if any(term in normalized for term in ("bao kg", "nang bao nhieu", "trong luong", "kg")):
        return "weight_kg"
    if any(term in normalized for term in ("bao nhieu inch", "may inch", "man hinh bao nhieu", "kich thuoc man hinh")):
        return "screen_inches"
    if any(term in normalized for term in ("pin", "thoi luong")):
        return "battery_wh"
    if any(term in normalized for term in ("bao hanh", "warranty")):
        return "warranty"
    if any(term in normalized for term in ("ben khong", "do ben", "ben bi")):
        return "durability"
    return None


def extract_requested_attributes(user_query: str) -> tuple[str, ...]:
    normalized = normalize_text(user_query)
    attributes: list[str] = []
    if any(term in normalized for term in ("ram", "16gb", "8gb", "32gb")):
        attributes.append("ram_gb")
    if any(term in normalized for term in ("ssd", "512gb", "1tb", "o cung", "bo nho trong")):
        attributes.append("storage_gb")
    if any(term in normalized for term in ("card roi", "gpu roi", "rtx", "gtx", "radeon rx")):
        attributes.append("gpu_type")
    if any(term in normalized for term in ("i3", "i5", "i7", "i9", "ryzen", "core ultra", "cpu")):
        attributes.append("cpu_tier")
    if any(term in normalized for term in ("duoi", "tren", "tam", "khoang", "trieu", "gia")):
        attributes.append("price_value")
    field = extract_field_question(user_query)
    if field:
        attributes.append(field)
    return tuple(dict.fromkeys(attributes))


def _cpu_filter_to_tier(cpu_filter: str) -> str:
    return {
        "intel_core_i9": "i9",
        "intel_core_i7": "i7",
        "intel_core_i5": "i5",
        "intel_core_i3": "i3",
        "intel_core_ultra": "Core Ultra",
        "intel_core_7": "Core 7",
        "intel_core_5": "Core 5",
        "amd_ryzen_7": "Ryzen 7",
        "amd_ryzen_5": "Ryzen 5",
    }.get(cpu_filter, cpu_filter)


def _asks_dedicated_gpu(normalized: str, gpu_filters: tuple[str, ...]) -> bool:
    if any(term in normalized for term in ("khong can card roi", "card tich hop")):
        return False
    return bool(gpu_filters and not set(gpu_filters).issubset({"intel_graphics", "intel_arc"})) or any(
        term in normalized for term in ("card roi", "gpu roi", "do hoa roi")
    )


def _extract_capacity(normalized: str, labels: tuple[str, ...]) -> int | None:
    for label in labels:
        match = re.search(rf"\b{re.escape(label)}\s*(\d+)\s*(gb|tb)\b", normalized)
        if match:
            value = int(match.group(1))
            return value * 1024 if match.group(2) == "tb" else value
        match = re.search(rf"\b(\d+)\s*(gb|tb)\s*{re.escape(label)}\b", normalized)
        if match:
            value = int(match.group(1))
            return value * 1024 if match.group(2) == "tb" else value
    return None


def _extract_screen_inches(normalized: str) -> float | None:
    match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(?:inch|inches)\b", normalized)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _inherit_constraints(
    constraints: dict[str, object],
    state: AgentState,
) -> dict[str, object]:
    frame = state.query_frame
    previous = frame.constraints if frame else state.last_constraints
    if previous is None:
        return constraints
    inherited = dict(constraints)
    for key in (
        "category",
        "brand",
        "min_price",
        "max_price",
        "cpu_tier",
        "gpu_type",
        "ram_gb",
        "storage_gb",
        "screen_inches",
        "use_case",
    ):
        if inherited.get(key) is None:
            inherited[key] = getattr(previous, key)
    if not inherited.get("requested_attributes") and frame is not None:
        inherited["requested_attributes"] = frame.requested_attributes
    return inherited


def _is_constraint_refinement(
    constraints: dict[str, object],
    state: AgentState,
) -> bool:
    """Treat added filters as refinements of the previous shopping frame."""

    frame = state.query_frame
    previous = frame.constraints if frame else state.last_constraints
    if previous is None:
        return False
    if getattr(previous, "category", None) is None and state.active_category is None:
        return False

    has_previous_filter = any(
        getattr(previous, key) is not None
        for key in (
            "category",
            "brand",
            "min_price",
            "max_price",
            "cpu_tier",
            "gpu_type",
            "ram_gb",
            "storage_gb",
            "screen_inches",
            "use_case",
        )
    )
    has_current_filter = any(
        constraints.get(key) is not None
        for key in (
            "brand",
            "category",
            "min_price",
            "max_price",
            "target_price",
            "cpu_tier",
            "gpu_type",
            "ram_gb",
            "storage_gb",
            "screen_inches",
            "use_case",
        )
    )
    if not has_previous_filter or not has_current_filter:
        return False

    current_category = constraints.get("category")
    previous_category = getattr(previous, "category", None) or state.active_category
    if (
        current_category is not None
        and previous_category is not None
        and current_category != previous_category
    ):
        return False

    return True


def _has_explicit_search_filters(constraints: dict[str, object]) -> bool:
    """Return whether a category query includes its own narrowing filters."""

    return any(
        constraints.get(key) is not None
        for key in (
            "brand",
            "min_price",
            "max_price",
            "target_price",
            "cpu_tier",
            "gpu_type",
            "ram_gb",
            "storage_gb",
            "screen_inches",
            "use_case",
        )
    )


def _is_brand_scoped_reference(
    normalized: str,
    constraints: dict[str, object],
    state: AgentState,
) -> bool:
    brand = constraints.get("brand")
    if not isinstance(brand, str):
        return False
    brand_text = normalize_text(brand)
    has_reference = bool(
        re.search(rf"\b(mau|may|con)\s+{re.escape(brand_text)}\s+(do|nay|vua roi)\b", normalized)
        or re.search(rf"\b(mau|may|con)\s+{re.escape(brand_text)}\s+ban\s+dang\s+noi\b", normalized)
        or re.search(rf"\b{re.escape(brand_text)}\s+(do|nay|vua roi)\b", normalized)
        or re.search(rf"\b{re.escape(brand_text)}\s+ban\s+dang\s+noi\b", normalized)
    )
    if not has_reference:
        return False
    matches = [
        candidate
        for candidate in state.last_shown_candidates
        if normalize_text(candidate.brand or "") == brand_text
    ]
    focused_name = normalize_text(state.focused_product_name or "")
    focused_matches_brand = bool(focused_name and _contains_phrase(focused_name, brand_text))
    return len(matches) == 1 or focused_matches_brand


def _has_shopping_scope_constraints(constraints: dict[str, object]) -> bool:
    return any(
        constraints.get(key) is not None
        for key in (
            "brand",
            "category",
            "cpu_tier",
            "min_price",
            "max_price",
            "target_price",
            "use_case",
        )
    )


def _extract_category(normalized: str) -> str | None:
    if _contains_phrase(normalized, "macbook"):
        return "Laptop"
    if _contains_phrase(normalized, "iphone"):
        return "Mobile Phone"
    if any(term in normalized for term in ("laptop", "may tinh xach tay", "notebook")):
        return "Laptop"
    if any(term in normalized for term in ("dien thoai", "smartphone", "phone", "iphone")):
        return "Mobile Phone"
    return None


def _extract_use_case(normalized: str) -> str | None:
    gaming_negated = any(term in normalized for term in ("khong choi game", "khong game", "khong gaming"))
    if any(term in normalized for term in ("van phong", "office", "hoc tap")):
        return "office"
    if gaming_negated:
        return None
    if any(term in normalized for term in ("choi game", "gaming", "game")):
        return "gaming"
    if any(term in normalized for term in ("do hoa", "render", "edit video")):
        return "creative"
    if any(term in normalized for term in ("mong nhe", "di chuyen", "gon nhe")):
        return "portability"
    return None


def _is_product_selection(normalized: str) -> bool:
    return _has_any_phrase(normalized, SELECTION_TERMS) or re.search(
        r"\b(con|may|mau)\s+thu\s+(1|2|3|mot|hai|ba)\b",
        normalized,
    ) is not None


def _is_top_two_followup(normalized: str, state: AgentState) -> bool:
    if len(state.last_shown_candidates) < 2:
        return False
    asks_for_two = re.search(
        r"\b(2|hai)\s+(mau|may|con|laptop|san pham)\b",
        normalized,
    ) is not None
    if not asks_for_two:
        return False
    return _has_any_phrase(normalized, TOP_TWO_FOLLOWUP_TERMS)


def _is_comparison(normalized: str, constraints: dict[str, object]) -> bool:
    brands = constraints.get("brands")
    explicit_comparison = _has_any_phrase(
        normalized,
        ("so sanh", "nen chon", "tot hon"),
    )
    brand_pair_comparison = isinstance(brands, tuple) and len(brands) >= 2 and _has_any_phrase(
        normalized,
        ("voi", "hay", "chon"),
    )
    return explicit_comparison or brand_pair_comparison


def _is_broad_consulting(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "tu van",
            "goi y",
            "nen mua",
            "may nao phu hop",
            "chon may",
        )
    )


def _is_query_continuation(normalized: str) -> bool:
    if _has_any_phrase(normalized, CONTINUATION_TERMS):
        return True
    return bool(
        re.search(r"\bcon\s+(may|mau|cai)(\s+[a-z0-9]+)?\s+nao\s+(khac|nua)\s+khong\b", normalized)
        or re.search(r"\b(may|mau|con)(\s+[a-z0-9]+)?\s+khac\s+khong\b", normalized)
    )


def _asks_same_range_alternatives(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "cung tam gia",
            "tam gia do",
            "cung muc gia",
            "muc gia do",
            "may nao cung tam",
        )
    )


def _contains_phrase(normalized: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", normalized) is not None


def _has_any_phrase(normalized: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(normalized, phrase) for phrase in phrases)
