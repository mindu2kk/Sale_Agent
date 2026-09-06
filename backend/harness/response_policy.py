from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from backend.services.catalog import CatalogProduct
from backend.services.conversation import normalize_text
from backend.harness.types import (
    AnswerMode,
    BeliefState,
    ConversationPlan,
    EvidenceRef,
    StrictnessMode,
)


DETAIL_INTENTS = {
    "product_detail",
    "select_previous_candidate",
    "product_detail_followup",
    "exact_product_detail",
    "product_correction",
    "focused_product_analysis",
}


class SalesResponsePolicy(BaseModel):
    mode: AnswerMode
    strictness: StrictnessMode
    intent: str
    template_key: Optional[str] = None
    failure_reason: Optional[str] = None
    requested_product_code: Optional[str] = None
    allowed_claims: List[str] = Field(default_factory=list)
    forbidden_claims: List[str] = Field(default_factory=list)
    price_warning_scope: Literal["none", "light", "prominent"] = "none"
    should_show_candidates: bool = True
    should_show_specs: bool = True
    should_show_tradeoffs: bool = False
    should_ask_clarification: bool = False
    hard_fail_allowed: bool = False
    follow_up_style: str = "generic"


def infer_sales_response_policy(
    user_query: str,
    plan: ConversationPlan,
    context: BeliefState,
    candidates: List[CatalogProduct],
    evidence_refs: List[EvidenceRef],
    failure_reason: Optional[str] = None,
    ai_available: bool = True,
) -> SalesResponsePolicy:
    del context
    query_lower = user_query.lower()
    normalized_query = normalize_text(user_query)
    normalized_tokens = set(normalized_query.split())
    dialogue_act = getattr(plan, "dialogue_act", "")

    intent = "unknown_consultative"
    template_key = "consultative"

    import re

    requested_product_code = None
    code_match = re.search(r"\b([A-Z0-9]{5,})\b", user_query.upper())
    if code_match:
        requested_product_code = code_match.group(1)

    if dialogue_act in DETAIL_INTENTS:
        intent = dialogue_act
        template_key = "product_detail"
        if getattr(plan, "product_codes", None):
            requested_product_code = plan.product_codes[0]
    elif (
        "chi tiet" in normalized_query
        or "san pham nay" in normalized_query
        or "ma" in normalized_tokens
    ) or any(
        k in query_lower for k in ["chi tiáº¿t", "mã", "sản phẩm này", "sáº£n pháº©m nÃ y"]
    ):
        intent = "product_code_detail"
        template_key = "product_detail"
    elif any(k in normalized_query for k in ["ben", "chong soc", "chuan quan doi", "ben bi"]) or any(
        k in query_lower for k in ["bá»n", "chống sốc", "chuẩn quân đội", "bền bỉ"]
    ):
        intent = "durability_consulting"
        template_key = "durability_consulting"
    elif any(k in normalized_query for k in ["manh", "chay do hoa", "game", "lap trinh", "cau hinh"]) or any(
        k in query_lower for k in ["máº¡nh", "cháº¡y Ä‘á»“ há»a", "game", "láº­p trÃ¬nh", "cấu hình"]
    ):
        intent = "performance_consulting"
        template_key = "performance_consulting"
    elif any(k in normalized_query for k in ["ngan sach", "trieu", "duoi", "tam", "re hon", "gia"]):
        intent = "price_query" if any(k in normalized_query for k in ["gia", "re", "khuyen mai"]) else "budget_consulting"
        template_key = "budget_consulting"
    elif any(k in normalized_query for k in ["so sanh", "voi", "chon ho", "nao hon"]) or any(
        k in query_lower for k in ["so sÃ¡nh", "so sánh", "với", "vá»›i", "chọn hộ"]
    ):
        intent = "comparison"
        template_key = "comparison"
    elif any(k in normalized_query for k in ["phu kien", "dien thoai"]) or any(
        k in query_lower for k in ["điện thoại", "Ä‘iá»‡n thoáº¡i", "phụ kiện", "phá»¥ kiá»‡n"]
    ):
        intent = "accessory_bundle" if "phu kien" in normalized_query else "phone_consulting"
        template_key = "accessory_bundle" if "phu kien" in normalized_query else "phone_consulting"
    elif any(k in query_lower for k in ["rtx", "ram", "ssd", "core", "cpu", "khác nhau"]):
        intent = "hardware_explanation"
        template_key = "hardware_explanation"
    elif any(k in normalized_query for k in ["dung hang ngay", "mang di", "nong", "pin"]):
        intent = "experience_question"
        template_key = "experience_question"

    if template_key != "product_detail" and (
        any(k in normalized_query for k in ["tot nhat", "re nhat", "manh nhat", "ben nhat", "dang mua nhat"])
        or any(k in query_lower for k in ["tá»‘t nháº¥t", "ráº» nháº¥t", "máº¡nh nháº¥t", "bá»n nháº¥t", "đáng mua nhất"])
    ):
        intent = "strong_superlative"
        template_key = "strong_superlative"

    if not ai_available:
        intent = "ai_unavailable"
        template_key = "product_detail" if len(candidates) == 1 else ("ai_unavailable_with_catalog_facts" if candidates else "hard_fail")

    refined_failure_reason = failure_reason
    if failure_reason == "stale_evidence":
        stale_fields = []
        for ev in evidence_refs:
            if isinstance(ev, dict):
                is_stale = ev.get("freshness") == "stale"
                field = ev.get("field")
            else:
                is_stale = getattr(ev, "freshness", "unknown") == "stale"
                field = getattr(ev, "field", None)
            if is_stale and field:
                stale_fields.append(field)
        if "price" in stale_fields or "promotion" in stale_fields:
            refined_failure_reason = "stale_price"
        elif "spec" in stale_fields:
            refined_failure_reason = "stale_spec"
        elif "stock" in stale_fields:
            refined_failure_reason = "stale_stock"

    price_warning_scope = "none"
    if refined_failure_reason in {"stale_price", "stale_promotion"} and (
        intent in {"budget_consulting", "price_query"} or any(k in query_lower for k in ["giá", "ngân sách", "khuyến mãi"])
    ):
        price_warning_scope = "light"

    follow_up_style = "generic"
    if template_key == "product_detail":
        follow_up_style = "product_detail"
    elif template_key == "durability_consulting":
        follow_up_style = "durability"
    elif template_key == "performance_consulting":
        follow_up_style = "performance"
    elif template_key in {"budget_consulting", "price_query"}:
        follow_up_style = "budget"
    elif template_key == "comparison":
        follow_up_style = "comparison"
    elif template_key == "hardware_explanation":
        follow_up_style = "hardware"

    hard_fail_allowed = False
    if not candidates and intent not in {"hardware_explanation", "experience_question", "durability_consulting"}:
        hard_fail_allowed = True
    if failure_reason in {"candidate_not_contained", "category_policy_mismatch"}:
        hard_fail_allowed = True
        template_key = "hard_fail"

    if not candidates and intent == "strong_superlative":
        template_key = "strong_superlative"

    return SalesResponsePolicy(
        mode="consultative",
        strictness="medium",
        intent=intent,
        template_key=template_key,
        failure_reason=refined_failure_reason,
        requested_product_code=requested_product_code,
        price_warning_scope=price_warning_scope,
        should_show_candidates=bool(candidates),
        should_show_specs=True,
        should_show_tradeoffs=intent == "comparison" or "so sánh" in query_lower,
        should_ask_clarification=hard_fail_allowed and not candidates,
        hard_fail_allowed=hard_fail_allowed,
        follow_up_style=follow_up_style,
    )
