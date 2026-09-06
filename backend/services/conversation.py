"""Stateful dialogue planning for catalog-grounded shopping advice."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.services.catalog import (
    CatalogService,
    PriceIntent,
    _detect_cpu_filters,
    _detect_gpu_filters,
)
from backend.harness.types import FreshnessState


SESSION_TTL_SECONDS = 5 * 60


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD",
        value.casefold().replace("đ", "d").replace("Ä‘", "d"),
    )
    ascii_value = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


def _contains_phrase(normalized: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", normalized) is not None


PreferenceName = Literal[
    "performance",
    "durability",
    "battery",
    "display",
    "portability",
    "gaming",
    "value",
]

DialogueAct = Literal[
    "product_detail",
    "select_previous_candidate",
    "product_detail_followup",
    "exact_product_detail",
    "product_correction",
    "focused_product_analysis",
    "brand_comparison",
    "refine_preferences",
    "price_causality",
    "price_objection",
    "cheaper_alternatives",
    "value_ranking",
    "catalog_ranking",
    "policy",
    "reject_candidate",
    "catalog_search",
    "clarify",
    "general_explanation",
]


class CandidateRef(BaseModel):
    code: str
    name: str
    brand: str | None = None
    category: str | None = None
    price: int | str | None = None
    specs_summary: str | None = None


class DecisionContext(BaseModel):
    """Backward-compatible client-owned state for a five-minute chat session."""
    model_config = {"extra": "forbid"}

    category: str | None = None
    budget_target: int | None = None
    budget_minimum: int | None = None
    budget_maximum: int | None = None
    goal: str | None = None
    use_case: str | None = None
    active_product_code: str | None = None
    compared_codes: list[str] = Field(default_factory=list, max_length=8)

    compared_brands: list[str] = Field(default_factory=list, max_length=4)
    candidate_codes: list[str] = Field(default_factory=list, max_length=12)
    last_shown_candidates: list[CandidateRef] = Field(default_factory=list, max_length=12)
    focused_product_code: str | None = None
    focused_product_name: str | None = None
    last_user_selected_product_code: str | None = None
    last_category: str | None = None
    last_sales_intent: str | None = None
    preferences: dict[str, float] = Field(default_factory=dict)
    rejected_codes: dict[str, str] = Field(default_factory=dict)
    last_intent: str | None = None
    last_recommendation_code: str | None = None
    topic_id: str | None = None
    updated_at: str | None = None
    state_version: int = 1
    catalog_revision: str | None = None
    context_compacted_at: str | None = None
    unresolved_questions: list[str] = Field(default_factory=list)
    confirmed_constraints: list[str] = Field(default_factory=list)
    last_query_frame: dict[str, object] = Field(default_factory=dict)
    freshness: FreshnessState = "unknown"

    def detect_category_drift(self, new_category: str | None) -> bool:
        if not new_category or not self.category:
            return False
        return new_category != self.category

    def check_freshness(self, current_revision: str, valid_codes: set[str]) -> dict:
        report = {
            "was_stale": False,
            "reason": "",
            "removed_candidate_ids": [],
            "removed_compared_ids": [],
            "old_catalog_revision": self.catalog_revision,
            "new_catalog_revision": current_revision
        }
        if self.catalog_revision and self.catalog_revision != current_revision:
            self.freshness = "stale"
            report["was_stale"] = True
            report["reason"] = "Catalog revision changed"

        self.catalog_revision = current_revision

        new_candidates = []
        for code in self.candidate_codes:
            if code in valid_codes:
                new_candidates.append(code)
            else:
                report["removed_candidate_ids"].append(code)
        self.candidate_codes = new_candidates
        self.last_shown_candidates = [
            candidate
            for candidate in self.last_shown_candidates
            if candidate.code in valid_codes
        ]

        new_compared = []
        for code in self.compared_codes:
            if code in valid_codes:
                new_compared.append(code)
            else:
                report["removed_compared_ids"].append(code)
        self.compared_codes = new_compared

        if report["removed_candidate_ids"] or report["removed_compared_ids"]:
            self.freshness = "stale"
            report["was_stale"] = True
            report["reason"] = report["reason"] or "Products removed from catalog"
        if self.focused_product_code and self.focused_product_code not in valid_codes:
            self.focused_product_code = None
            self.focused_product_name = None
        if self.active_product_code and self.active_product_code not in valid_codes:
            self.active_product_code = None

        if not report["was_stale"]:
            self.freshness = "fresh"

        return report

    def compact(self) -> dict:
        report = {
            "changed": False,
            "trimmed_candidates": 0,
            "trimmed_compared": 0,
            "trimmed_rejected": 0,
            "dropped_fields": [],
            "reason": ""
        }
        if len(self.candidate_codes) > 12:
            report["trimmed_candidates"] = len(self.candidate_codes) - 12
            self.candidate_codes = self.candidate_codes[:12]
            report["changed"] = True
        if len(self.last_shown_candidates) > 12:
            self.last_shown_candidates = self.last_shown_candidates[:12]
            report["changed"] = True
        if len(self.compared_codes) > 8:
            report["trimmed_compared"] = len(self.compared_codes) - 8
            self.compared_codes = self.compared_codes[:8]
            report["changed"] = True
        if len(self.rejected_codes) > 10:
            report["trimmed_rejected"] = len(self.rejected_codes) - 10
            self.rejected_codes = dict(list(self.rejected_codes.items())[-10:])
            report["changed"] = True

        if report["changed"]:
            report["reason"] = "State exceeded working memory budget"
        return report

    def is_expired(self, now: datetime | None = None) -> bool:
        if not self.updated_at:
            return False
        try:
            updated = datetime.fromisoformat(self.updated_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        current = now or datetime.now(UTC)
        return (current - updated).total_seconds() > SESSION_TTL_SECONDS

    def budget_intent(self) -> PriceIntent | None:
        if self.budget_target:
            return PriceIntent(mode="target", target=self.budget_target)
        if self.budget_minimum is not None and self.budget_maximum is not None:
            return PriceIntent(
                mode="range",
                minimum=self.budget_minimum,
                maximum=self.budget_maximum,
            )
        if self.budget_maximum is not None:
            return PriceIntent(mode="max", maximum=self.budget_maximum)
        if self.budget_minimum is not None:
            return PriceIntent(mode="min", minimum=self.budget_minimum)
        return None


@dataclass(frozen=True)
class ConversationPlan:
    dialogue_act: DialogueAct
    confidence: float
    category: str | None
    price_intent: PriceIntent | None
    brands: tuple[str, ...] = ()
    product_codes: tuple[str, ...] = ()
    preferences: dict[str, float] = field(default_factory=dict)
    goal: str | None = None
    use_case: str | None = None
    response_strategy: str = "deterministic"
    reason: str = ""
    state_delta: dict[str, object] = field(default_factory=dict)
    starts_new_topic: bool = False
    clarification_question: str | None = None


class ConversationPlanner:
    """Resolve the dialogue act before any catalog retrieval is allowed."""

    POLICY_TERMS = (
        "bao hanh",
        "doi tra",
        "hoan tien",
        "chinh sach",
        "warranty",
        "return",
        "refund",
        "exchange",
    )
    DETAIL_TERMS = (
        "tu van chi tiet",
        "noi ki",
        "noi ky",
        "ki hon",
        "ky hon",
        "cau hinh",
        "thong so",
        "noi ro",
        "phan tich",
    )
    CORRECTION_TERMS = (
        "co ma",
        "khong phai",
        "y toi la",
        "y minh la",
    )
    GENERAL_EXPLANATION_TERMS = (
        "dung de lam gi",
        "khac gi",
        "khac nhau",
        "can khong",
        "can thiet khong",
        "anh huong gi",
        "co loi gi",
        "giup gi",
        "co on khong",
        "co tot khong",
        "phan biet",
        "co nong khong",
        "co trau khong",
    )
    PRICE_CAUSALITY_PATTERNS = (
        r"\btai sao\b.*\b(re hon|dat hon|mac hon|gia cao hon|chenh gia)\b",
        r"\bvi sao\b.*\b(re hon|dat hon|mac hon|gia cao hon|chenh gia)\b",
        r"\b(re hon|dat hon|mac hon)\b.*\b(tai sao|vi sao)\b",
        r"\bchenh lech gia\b",
    )
    ALTERNATIVE_PATTERNS = (
        r"\b(tim|goi y|co|con|mau|san pham|lua chon)\b.*\bre hon\b",
        r"\b(thay the|lua chon khac|mau khac|gia tot hon|tiet kiem hon)\b",
    )
    PRICE_OBJECTION_TERMS = (
        "gia dat",
        "dat qua",
        "mac qua",
        "gia cao vay",
        "gia cao qua",
        "khong dang tien",
    )
    VALUE_TERMS = (
        "hieu nang tren gia",
        "ti le hieu nang",
        "ty le hieu nang",
        "performance per price",
        "dang tien nhat",
        "gia tri cao nhat",
        "may nao dang tien hon",
    )
    STRONGEST_PATTERNS = (
        r"\b(khoe|manh|hieu nang cao|cau hinh cao)\s+nhat\b",
        r"\bmay nao\s+(khoe|manh)\s+nhat\b",
        r"\btop\s+(hieu nang|performance)\b",
    )
    PREMIUM_PATTERNS = (
        r"\b(xin|cao cap|toan dien)\s+nhat\b",
        r"\b(tot nhat|flagship)\b.*\b(shop|cua hang|catalog)?\b",
        r"\b(best|flagship)\s+(laptop|phone|smartphone)\b",
    )
    LOWEST_PRICE_PATTERNS = (
        r"\b(re|gia thap)\s+nhat\b",
        r"\bthap gia nhat\b",
    )
    HIGHEST_PRICE_PATTERNS = (
        r"\b(dat|gia cao)\s+nhat\b",
        r"\bcao gia nhat\b",
    )
    COMPARISON_TERMS = (
        "hay",
        "so sanh",
        "nen chon",
        "nen mua",
        "tot hon",
        "may nao hon",
        "may nao dang tien",
    )
    REJECTION_TERMS = (
        "khong chon mau nay",
        "khong thich mau nay",
        "loai mau nay",
        "bo mau nay",
        "doi mau khac",
        "khong lay con nay",
    )

    def __init__(self, catalog: CatalogService) -> None:
        self.catalog = catalog

    def plan(self, message: str, state: DecisionContext | None) -> ConversationPlan:
        context = state or DecisionContext()
        if context.is_expired():
            context = DecisionContext()

        normalized = normalize_text(message)
        explicit_category = self.catalog._detect_category(message)
        category_switched = context.detect_category_drift(explicit_category)

        if category_switched:
            context.candidate_codes.clear()
            context.compared_codes.clear()
            # Note: We do not completely reset DecisionContext because
            # "Không clear budget/user constraints nếu vẫn còn phù hợp"
            context.category = explicit_category
            context.freshness = "stale"

        category = explicit_category or context.category
        price_intent = self.catalog.analyze_query(message).price_intent
        if price_intent is None and self._is_continuation(normalized):
            price_intent = context.budget_intent()
        if (
            price_intent is None
            and context.active_product_code
            and self.catalog._asks_for_same_price(message)
        ):
            active_product = self.catalog.get(context.active_product_code)
            if active_product is not None:
                from backend.services.catalog import _price_value

                price_intent = PriceIntent(
                    mode="target",
                    target=_price_value(active_product.price),
                )

        explicit_brands = self.catalog._detect_brands(message)
        if category is None and len(explicit_brands) >= 2:
            common_categories: set[str] | None = None
            for brand in explicit_brands:
                brand_categories = {
                    product.category
                    for product in self.catalog.products
                    if product.brand == brand
                }
                common_categories = (
                    brand_categories
                    if common_categories is None
                    else common_categories & brand_categories
                )
            if common_categories and len(common_categories) == 1:
                category = next(iter(common_categories))
        brands = explicit_brands
        cpu_filters = _detect_cpu_filters(message)
        gpu_filters = _detect_gpu_filters(message)
        sku_codes = tuple(
            code
            for code in re.findall(r"\b[A-Z0-9]{8}\b", message.upper())
            if self.catalog.get(code) is not None
        )
        named_products = self.catalog.resolve_products(message, limit=8)
        explicit_codes = tuple(
            dict.fromkeys(
                (*sku_codes, *(product.code for product in named_products))
            )
        )
        if (
            explicit_category is None
            and named_products
            and len({product.category for product in named_products}) == 1
        ):
            category = named_products[0].category
        detected_preferences = self._detect_preferences(normalized)
        detected_use_case = self.catalog._detect_use_case(message)
        if (
            len(named_products) >= 2
            and not self._has_explicit_gaming_need(normalized)
        ):
            detected_preferences.pop("gaming", None)
            if detected_use_case == "gaming":
                detected_use_case = None
        preferences = dict(context.preferences)
        preferences.update(detected_preferences)
        use_case = detected_use_case or context.use_case
        goal = self.catalog._detect_goal(message) or context.goal
        starts_new_topic = category_switched or self._starts_new_topic(
            explicit_brands,
            context,
            explicit_codes,
        )
        if starts_new_topic:
            preferences = detected_preferences
            use_case = detected_use_case
            goal = self.catalog._detect_goal(message)
            brands = explicit_brands
            if explicit_codes:
                product = self.catalog.get(explicit_codes[0])
                if product is not None:
                    category = product.category

        if any(term in normalized for term in self.POLICY_TERMS):
            return self._plan(
                "policy", 0.99, category, price_intent, brands, explicit_codes,
                preferences, goal, use_case, "Local policy retrieval.",
            )

        if any(re.search(pattern, normalized) for pattern in self.PRICE_CAUSALITY_PATTERNS):
            codes = explicit_codes or tuple(context.compared_codes or context.candidate_codes)
            if len(codes) < 2 and len(brands) < 2:
                return self._plan(
                    "clarify", 0.99, category, price_intent, brands, codes,
                    preferences, goal, use_case,
                    "Price causality requires two identifiable options.",
                    clarification_question=(
                        "Bạn muốn mình giải thích chênh lệch giá giữa hai mẫu hoặc hai hãng nào?"
                    ),
                )
            return self._plan(
                "price_causality", 0.98, category, price_intent, brands, codes,
                preferences, goal, use_case,
                "Causal price question takes precedence over cheaper alternatives.",
            )

        if any(term in normalized for term in self.REJECTION_TERMS):
            rejected_code = (
                context.last_recommendation_code
                or context.active_product_code
                or next(iter(context.candidate_codes), None)
            )
            if rejected_code:
                if price_intent is None:
                    rejected_product = self.catalog.get(rejected_code)
                    if rejected_product is not None:
                        from backend.services.catalog import _price_value

                        price_intent = PriceIntent(
                            mode="target",
                            target=_price_value(rejected_product.price),
                        )
                return self._plan(
                    "reject_candidate", 0.98, category, price_intent, brands,
                    (rejected_code,), preferences, goal, use_case,
                    "Exclude the rejected recommendation and retrieve a replacement.",
                )

        if any(re.search(pattern, normalized) for pattern in self.ALTERNATIVE_PATTERNS):
            codes = explicit_codes or self._active_codes(context)
            return self._plan(
                "cheaper_alternatives", 0.96, category, price_intent, brands,
                codes, preferences, goal, use_case, "Explicit request for alternatives.",
            )

        if explicit_codes and (
            any(term in normalized for term in self.DETAIL_TERMS)
            or len(explicit_codes) == 1
        ):
            if len(explicit_codes) == 1:
                product = self.catalog.get(explicit_codes[0])
                if product is not None:
                    category = product.category
            return self._plan(
                "exact_product_detail", 0.99, category, price_intent, brands,
                explicit_codes, preferences, goal, use_case, "Explicit SKU detail.",
                starts_new_topic=starts_new_topic,
            )

        if any(term in normalized for term in self.PRICE_OBJECTION_TERMS):
            return self._plan(
                "price_objection", 0.96, category, price_intent, brands,
                explicit_codes or self._active_codes(context), preferences, goal,
                use_case, "Price objection about the active product.",
            )

        ranking_goal = self._detect_catalog_ranking_goal(normalized)
        if ranking_goal is not None:
            if category is None:
                return self._plan(
                    "clarify", 0.99, category, price_intent, explicit_brands,
                    (), preferences, ranking_goal, use_case,
                    "A global catalog ranking requires a product category.",
                    clarification_question=(
                        "Bạn muốn mình tìm mẫu mạnh nhất trong nhóm laptop hay điện thoại? "
                        "Nếu là laptop, bạn có thể nói thêm ưu tiên chơi game, đồ họa hay công việc."
                    ),
                )
            ranking_preferences = dict(preferences)
            if ranking_goal in {"max_performance", "best_overall"}:
                ranking_preferences["performance"] = 1.0
            return self._plan(
                "catalog_ranking", 0.98, category, price_intent, explicit_brands,
                (), ranking_preferences, ranking_goal, use_case,
                "Global superlative request requires deterministic catalog-wide ranking.",
                starts_new_topic=starts_new_topic,
            )

        if any(term in normalized for term in self.VALUE_TERMS):
            current_codes = tuple(context.compared_codes or context.candidate_codes)
            if len(current_codes) >= 2:
                return self._plan(
                    "refine_preferences", 0.95, category, price_intent, brands,
                    current_codes, {**preferences, "value": 1.0}, goal, use_case,
                    "Value follow-up reranks the current decision set.",
                )
            return self._plan(
                "value_ranking", 0.96, category, price_intent, brands, (),
                {**preferences, "value": 1.0}, "performance_per_price", use_case,
                "Value ranking within inherited constraints.",
            )

        has_comparison_context = (
            len(context.compared_codes) >= 2
            or len(context.compared_brands) >= 2
        )
        has_new_preferences = bool(self._detect_preferences(normalized)) or bool(
            self.catalog._detect_use_case(message)
        )
        if context.goal == "performance_per_price" and has_new_preferences:
            return self._plan(
                "value_ranking", 0.96, category, price_intent, brands, (),
                preferences, "performance_per_price", use_case,
                "A use-case follow-up reruns value ranking inside the saved budget.",
            )
        if has_comparison_context and (
            has_new_preferences
            or normalized.startswith("neu ")
            or "thi sao" in normalized
        ):
            return self._plan(
                "refine_preferences", 0.97, category, price_intent, brands,
                tuple(context.compared_codes or context.candidate_codes),
                preferences, goal, use_case,
                "Follow-up preferences must preserve the current comparison set.",
            )

        if self._is_new_filtered_search(
            normalized,
            context=context,
            explicit_category=explicit_category,
            price_intent=price_intent,
            explicit_brands=explicit_brands,
            cpu_filters=cpu_filters,
            gpu_filters=gpu_filters,
            detected_use_case=detected_use_case,
        ):
            return self._plan(
                "catalog_search",
                0.95,
                category,
                price_intent,
                explicit_brands,
                (),
                detected_preferences or preferences,
                self.catalog._detect_goal(message),
                detected_use_case or use_case,
                "Fresh search filters override the previously focused product.",
                starts_new_topic=bool(
                    explicit_category
                    or explicit_brands
                    or price_intent
                    or cpu_filters
                    or gpu_filters
                    or detected_use_case
                ),
            )

        if len(explicit_brands) >= 2 or (
            has_comparison_context
            and any(term in normalized for term in self.COMPARISON_TERMS)
        ):
            return self._plan(
                "brand_comparison", 0.97, category, price_intent,
                explicit_brands or tuple(context.compared_brands),
                explicit_codes or tuple(context.compared_codes),
                preferences, goal, use_case,
                "Compare the named or active brands without broadening the set.",
                starts_new_topic=starts_new_topic,
            )

        if (
            context.active_product_code
            and any(term in normalized for term in self.DETAIL_TERMS)
        ):
            return self._plan(
                "product_detail_followup", 0.97, category, price_intent, brands,
                (context.active_product_code,), preferences, goal, use_case,
                "Detail follow-up stays on the active SKU.",
            )

        from backend.harness.product_resolver import resolve_product_reference

        product_resolution = resolve_product_reference(message, context)
        if product_resolution.resolved and product_resolution.code:
            resolved_product = self.catalog.get(product_resolution.code)
            if resolved_product is not None:
                category = resolved_product.category
            if product_resolution.source == "focused_product":
                act = "focused_product_analysis"
            elif product_resolution.source == "correction":
                act = "product_correction"
            elif product_resolution.source in {"exact_code", "exact_name"}:
                act = "exact_product_detail"
            elif product_resolution.source == "ordinal":
                act = "select_previous_candidate"
            elif any(term in normalized for term in self.DETAIL_TERMS):
                act = "product_detail_followup"
            else:
                act = "select_previous_candidate"
            return self._plan(
                act,
                max(0.9, product_resolution.confidence),
                category,
                price_intent,
                brands,
                (product_resolution.code,),
                preferences,
                goal,
                use_case,
                product_resolution.reason or "Resolved product reference from previous candidates.",
                starts_new_topic=False,
            )
        if product_resolution.ambiguous_candidates:
            return self._plan(
                "clarify",
                0.99,
                category,
                price_intent,
                brands,
                (),
                preferences,
                goal,
                use_case,
                "Product reference matched multiple previous candidates from the same brand.",
                clarification_question=(
                    "Anh/chị muốn mẫu nào trong các máy vừa xem: "
                    + "; ".join(
                        candidate.name
                        for candidate in product_resolution.ambiguous_candidates[:4]
                    )
                    + "?"
                ),
            )

        if any(term in normalized for term in self.GENERAL_EXPLANATION_TERMS):
            return self._plan(
                "general_explanation", 0.95, category, price_intent, brands,
                explicit_codes or self._active_codes(context), preferences, goal, use_case,
                "Domain knowledge or feature explanation.",
                starts_new_topic=starts_new_topic,
            )

        if self._is_vague_shopping_request(
            normalized,
            category=category,
            price_intent=price_intent,
            brands=explicit_brands,
            explicit_codes=explicit_codes,
            context=context,
        ):
            return self._plan(
                "clarify", 0.99, category, price_intent, explicit_brands,
                explicit_codes, preferences, goal, use_case,
                "The request has no safe retrieval constraints.",
                clarification_question=(
                    "Bạn cần laptop hay điện thoại, ngân sách khoảng bao nhiêu và dùng chính cho việc gì?"
                ),
            )

        return self._plan(
            "catalog_search", 0.82, category, price_intent, explicit_brands,
            explicit_codes, preferences, goal, use_case, "New catalog search.",
            starts_new_topic=starts_new_topic,
        )

    @staticmethod
    def _active_codes(context: DecisionContext) -> tuple[str, ...]:
        if context.active_product_code:
            return (context.active_product_code,)
        return tuple(context.compared_codes or context.candidate_codes)

    @staticmethod
    def _is_continuation(normalized: str) -> bool:
        return any(
            term in normalized
            for term in (
                "uu tien",
                "cung tam",
                "tam gia",
                "may nao",
                "mau nao",
                "dang tien",
                "thi sao",
                "re hon",
                "dat hon",
                "noi ki",
                "noi ky",
                "choi game",
                "van phong",
                "lap trinh",
                "do ben",
                "pin",
                "man hinh",
                "xin nhat",
                "khoe nhat",
                "manh nhat",
                "tot nhat",
            )
        )

    @staticmethod
    def _detect_preferences(normalized: str) -> dict[str, float]:
        detected: dict[str, float] = {}
        groups = {
            "performance": ("hieu nang", "manh", "khoe", "toc do", "cpu", "gpu"),
            "durability": ("do ben", "ben bi", "ben", "chac chan"),
            "battery": ("pin", "thoi luong"),
            "display": ("man hinh", "hien thi", "oled"),
            "portability": ("mong nhe", "nhe", "di chuyen", "gon"),
            "gaming": ("choi game", "gaming", "game", "card roi"),
            "value": ("dang tien", "gia tri", "hieu nang tren gia"),
        }
        for name, terms in groups.items():
            if any(term in normalized for term in terms):
                detected[name] = 1.0
        return detected

    @classmethod
    def _detect_catalog_ranking_goal(cls, normalized: str) -> str | None:
        if any(term in normalized for term in cls.VALUE_TERMS):
            return None
        pattern_groups = (
            ("max_performance", cls.STRONGEST_PATTERNS),
            ("best_overall", cls.PREMIUM_PATTERNS),
            ("lowest_price", cls.LOWEST_PRICE_PATTERNS),
            ("highest_price", cls.HIGHEST_PRICE_PATTERNS),
        )
        for goal, patterns in pattern_groups:
            if any(re.search(pattern, normalized) for pattern in patterns):
                return goal
        return None

    @staticmethod
    def _has_explicit_gaming_need(normalized: str) -> bool:
        return any(
            term in normalized
            for term in (
                "choi game",
                "uu tien gaming",
                "nhu cau gaming",
                "dung de gaming",
                "phuc vu gaming",
                "neu gaming",
            )
        )

    @staticmethod
    def _starts_new_topic(
        explicit_brands: tuple[str, ...],
        context: DecisionContext,
        explicit_codes: tuple[str, ...],
    ) -> bool:
        if explicit_codes:
            old_codes = set(context.compared_codes or context.candidate_codes)
            return bool(old_codes and not set(explicit_codes).issubset(old_codes))
        if len(explicit_brands) >= 2 and context.compared_brands:
            return set(explicit_brands) != set(context.compared_brands)
        return False

    @staticmethod
    def _is_vague_shopping_request(
        normalized: str,
        *,
        category: str | None,
        price_intent: PriceIntent | None,
        brands: tuple[str, ...],
        explicit_codes: tuple[str, ...],
        context: DecisionContext,
    ) -> bool:
        if category or price_intent or brands or explicit_codes:
            return False
        if context.active_product_code or context.candidate_codes:
            return False
        return any(
            term in normalized
            for term in (
                "tu van may",
                "chon may",
                "mua may",
                "may nao tot",
                "goi y cho toi",
                "bat dau chon may",
            )
        )

    @staticmethod
    def _has_focus_follow_up_cue(normalized: str) -> bool:
        return any(
            term in normalized
            for term in (
                "may do",
                "con do",
                "mau vua roi",
                "con nay",
                "may nay",
                "ban dang noi",
                "phan tich",
                "noi ro hon",
                "chi tiet",
                "bao kg",
                "trong luong",
                "pin",
                "bao hanh",
                "ben khong",
            )
        )

    @classmethod
    def _is_new_filtered_search(
        cls,
        normalized: str,
        *,
        context: DecisionContext,
        explicit_category: str | None,
        price_intent: PriceIntent | None,
        explicit_brands: tuple[str, ...],
        cpu_filters: tuple[str, ...],
        gpu_filters: tuple[str, ...],
        detected_use_case: str | None,
    ) -> bool:
        if not (context.active_product_code or context.focused_product_code):
            return False
        if cls._has_focus_follow_up_cue(normalized):
            return False
        if any(_contains_phrase(normalized, term) for term in cls.CORRECTION_TERMS):
            return False
        if any(term in normalized for term in cls.POLICY_TERMS):
            return False
        if len(explicit_brands) >= 2 and any(
            term in normalized for term in cls.COMPARISON_TERMS
        ):
            return False
        search_signals = (
            bool(explicit_category)
            or price_intent is not None
            or bool(explicit_brands)
            or bool(cpu_filters)
            or bool(gpu_filters)
            or detected_use_case is not None
            or any(
                term in normalized
                for term in (
                    "co may nao",
                    "co mau nao",
                    "tim",
                    "goi y",
                    "duoi",
                    "tren",
                    "toi da",
                    "tam",
                    "khoang",
                    "re hon",
                    "card roi",
                    "rtx",
                    "gtx",
                )
            )
        )
        return search_signals

    @staticmethod
    def _plan(
        dialogue_act: DialogueAct,
        confidence: float,
        category: str | None,
        price_intent: PriceIntent | None,
        brands: tuple[str, ...],
        product_codes: tuple[str, ...],
        preferences: dict[str, float],
        goal: str | None,
        use_case: str | None,
        reason: str,
        *,
        starts_new_topic: bool = False,
        clarification_question: str | None = None,
    ) -> ConversationPlan:
        state_delta: dict[str, object] = {
            "category": category,
            "goal": goal,
            "use_case": use_case,
            "preferences": preferences,
        }
        if price_intent is not None:
            state_delta["price_intent"] = {
                "mode": price_intent.mode,
                "target": price_intent.target,
                "minimum": price_intent.minimum,
                "maximum": price_intent.maximum,
            }
        if brands:
            state_delta["compared_brands"] = list(brands)
        if product_codes:
            state_delta["candidate_codes"] = list(product_codes)
        return ConversationPlan(
            dialogue_act=dialogue_act,
            confidence=confidence,
            category=category,
            price_intent=price_intent,
            brands=brands,
            product_codes=product_codes,
            preferences=preferences,
            goal=goal,
            use_case=use_case,
            reason=reason,
            state_delta=state_delta,
            starts_new_topic=starts_new_topic,
            clarification_question=clarification_question,
        )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
