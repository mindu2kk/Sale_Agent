import re
from typing import Set

from backend.harness.types import (
    DecisionGateResult,
    SafeDegradeAction,
    AskClarificationAction,
    BeliefState,
    ConversationPlan,
    EvidenceRef
)
from backend.services.ai_service import AIAnswer
from backend.services.catalog import CatalogProduct

def parse_price(price_str: str | None) -> float | None:
    if not price_str:
        return None
    clean = re.sub(r'[^\d]', '', str(price_str))
    try:
        return float(clean)
    except ValueError:
        return None

def get_product_evidence_fields(product: CatalogProduct, evidence_refs: list[EvidenceRef]) -> set[str]:
    fields = set()
    for ev in evidence_refs:
        if getattr(ev, "product_code", getattr(ev, "productId", None)) == product.code:
            if hasattr(ev, "field"):
                fields.add(ev.field)

    if getattr(product, "price", None):
        fields.add("price")

    ctx = getattr(product, "context", "").lower()
    if "bảo hành" in ctx or "policy" in ctx:
        fields.update(["policy", "warranty"])
    if "bền" in ctx or "chuẩn quân đội" in ctx or "chống sốc" in ctx or "durability" in ctx:
        fields.update(["durability", "material", "certification"])
    if "pin" in ctx or "mah" in ctx or "battery" in ctx:
        fields.add("battery")
    if "gpu" in ctx or "rtx" in ctx or "nvidia" in ctx or "amd" in ctx or "card" in ctx:
        fields.add("gpu")
    if "ssd" in ctx or "hdd" in ctx or "storage" in ctx:
        fields.add("ssd")
    if "ram" in ctx or "gb" in ctx:
        fields.add("ram")
    if "cpu" in ctx or "intel" in ctx or "core" in ctx or "ryzen" in ctx:
        fields.add("cpu")
    if "màn hình" in ctx or "display" in ctx or "inch" in ctx:
        fields.add("display")

    return fields

def evaluate_decision_gate(
    answer: AIAnswer,
    plan: ConversationPlan,
    candidates: list[CatalogProduct],
    evidence_refs: list[EvidenceRef],
    context: BeliefState
) -> DecisionGateResult:
    plan_act = getattr(plan, "dialogue_act", getattr(plan, "intent", None))
    is_recommendation = plan_act in ["brand_comparison", "catalog_ranking", "value_ranking", "compare_products"]

    answer_text_lower = answer.text.lower()
    strong_keywords = ["tốt nhất", "bền nhất", "mạnh nhất", "xịn nhất", "đáng mua nhất", "pin trâu nhất", "chắc chắn vượt trội", "rẻ nhất"]
    has_strong_claim = any(kw in answer_text_lower for kw in strong_keywords)

    winner_codes = answer.product_codes or []
    if not winner_codes and candidates:
        winner_codes = [candidates[0].code]

    winner_products = [c for c in candidates if c.code in winner_codes]
    other_products = [c for c in candidates if c.code not in winner_codes]

    is_choosing_winner = len(winner_products) == 1 and len(candidates) > 1

    if has_strong_claim:
        answer_mode = "strong_claim"
    elif is_choosing_winner:
        answer_mode = "recommendation"
    elif is_recommendation:
        answer_mode = "comparison"
    else:
        answer_mode = "consultative"

    # If it's just a factual/comparison/consultative answer and doesn't choose a winner or make strong claims, pass it automatically
    if answer_mode not in ["recommendation", "strong_claim"]:
        return DecisionGateResult(
            allowed=True, abstained=False, required_user_criterion=False,
            has_verifiable_advantage=False, confidence_ok=True, margin_ok=True,
            evidence_ok=True, differentiating_reasons_ok=True, traceEvent="decision_gate_passed"
        )

    # Is there a clear winner?
    # If the answer explicitly recommends one product or only outputs one product code
    # If it outputs multiple product codes, we assume it's discussing trade-offs unless it explicitly claims one is better.
    # We will assume that if confidence is used, it's attempting a recommendation.
    # If it doesn't try to pick a winner (e.g. just listing both), it should pass if it doesn't violate rules.
    # To simplify, we treat any recommendation plan as attempting to pick a winner if confidence >= 0.8
    # However, if it's explicitly a trade-off, confidence might be lower.

    confidence = getattr(answer, "confidence", 1.0)
    confidence_ok = confidence >= 0.8

    if not confidence_ok:
        return DecisionGateResult(
            allowed=False, abstained=True, reason="Confidence is too low for a firm recommendation.",
            required_user_criterion=False, has_verifiable_advantage=False,
            confidence_ok=False, margin_ok=False, evidence_ok=False, differentiating_reasons_ok=False,
            recommendedRecovery=SafeDegradeAction(message="Tôi không có đủ cơ sở để gợi ý một sản phẩm duy nhất. Mỗi sản phẩm có ưu nhược điểm riêng, bạn có thể cân nhắc theo nhu cầu cụ thể."),
            traceEvent="decision_gate_low_confidence"
        )

    # Check for criterion
    plan_goal = getattr(plan, "goal", getattr(plan, "objective", None))
    plan_use_case = getattr(plan, "use_case", None)
    context_preferences = getattr(context, "preferences", {})

    has_criterion = bool(plan_goal or plan_use_case or context_preferences)

    # If it's returning multiple products without a criterion, it's just a trade-off.
    # But since we already filtered out non-winner modes, we only reach here if it's a strong claim on multiple products.

    # Let's check Price Advantage (Rule B)
    has_verifiable_advantage = False
    margin_ok = False
    fresh_price = True

    winner_prices = [parse_price(p.price) for p in winner_products if parse_price(p.price) is not None]
    other_prices = [parse_price(p.price) for p in other_products if parse_price(p.price) is not None]

    is_cheaper = False
    if winner_prices and other_prices:
        min_winner_price = min(winner_prices)
        min_other_price = min(other_prices)

        if min_winner_price < min_other_price:
            is_cheaper = True

        # Check freshness
        stale_evs = [ev for ev in evidence_refs if getattr(ev, "freshness", "unknown") == "stale"]
        if stale_evs:
            fresh_price = False

        # Margin: at least 3% cheaper
        if min_winner_price < min_other_price * 0.97:
            margin_ok = True

        if fresh_price and margin_ok:
            has_verifiable_advantage = True

    # Let's check Differentiating Evidence (Rule A)
    evidence_ok = False
    differentiating_reasons_ok = False

    if has_criterion:
        # Map criterion to required evidence fields
        required_fields = set()
        goal_lower = str(plan_goal).lower()
        use_case_lower = str(plan_use_case).lower()

        if "gaming" in use_case_lower or "gaming" in context_preferences:
            required_fields.update(["gpu", "cpu", "ram"])
        if "pin" in goal_lower or "battery" in goal_lower or "battery" in context_preferences:
            required_fields.add("battery")
        if "bền" in goal_lower or "durability" in goal_lower or "durability" in context_preferences:
            required_fields.update(["durability", "material"])
        if "giá" in goal_lower or "rẻ" in goal_lower or "price" in goal_lower or "price" in context_preferences or "value" in context_preferences:
            required_fields.add("price")
        if "văn phòng" in use_case_lower or "học tập" in use_case_lower:
            required_fields.update(["ram", "storage", "weight", "display", "price"])

        if not required_fields:
            # Fallback for generic criterion
            required_fields.add("price")

        # Check if winner has the required evidence
        winner_evidence_fields = set()
        for wp in winner_products:
            winner_evidence_fields.update(get_product_evidence_fields(wp, evidence_refs))

        if any(rf in winner_evidence_fields for rf in required_fields):
            evidence_ok = True

        # Differentiating reason: Winner must have evidence that differs or answer text must mention it
        # A simple check: if evidence is ok, we assume there is a differentiating reason if we have > 1 candidates.
        # If there's only 1 candidate, we can't differentiate, so it must be based on absolute merit (but rule says abstain if no basis).
        if len(candidates) > 1:
            differentiating_reasons_ok = evidence_ok
        else:
            if "hơn" in answer_text_lower or "nhất" in answer_text_lower:
                differentiating_reasons_ok = False # Hallucinated comparison
            else:
                differentiating_reasons_ok = evidence_ok

    # Rule evaluation
    allowed_by_a = confidence_ok and has_criterion and evidence_ok and differentiating_reasons_ok
    allowed_by_b = confidence_ok and has_verifiable_advantage and margin_ok

    if allowed_by_a or allowed_by_b:
        return DecisionGateResult(
            allowed=True, abstained=False, reason=None,
            required_user_criterion=has_criterion, has_verifiable_advantage=has_verifiable_advantage,
            confidence_ok=confidence_ok, margin_ok=margin_ok, evidence_ok=evidence_ok,
            differentiating_reasons_ok=differentiating_reasons_ok,
            traceEvent="decision_gate_passed"
        )
    else:
        # Determine specific failure reason
        trace_event = "decision_gate_abstained"
        reason = "Recommendation rules not met."

        if is_cheaper and not margin_ok:
            trace_event = "decision_gate_insufficient_margin"
            reason = "Price advantage exists but margin is too small."
            recovery = SafeDegradeAction(message="Mức giá chênh lệch không đáng kể, bạn có muốn xem xét thêm về cấu hình hay thiết kế không?")
        elif not has_criterion and not has_verifiable_advantage:
            trace_event = "decision_gate_missing_criterion"
            reason = "Missing user criterion and no clear price advantage."
            recovery = AskClarificationAction(question="Bạn muốn ưu tiên tiêu chí nào? (giá rẻ, pin lâu, chơi game...)")
        elif has_criterion and not evidence_ok:
            trace_event = "decision_gate_insufficient_evidence"
            reason = "Winner lacks evidence for the user criterion."
            recovery = SafeDegradeAction(message="Tôi không tìm thấy đủ thông tin (cấu hình/tính năng) để xác nhận sản phẩm này phù hợp với tiêu chí của bạn.")
        elif has_criterion and not differentiating_reasons_ok:
            trace_event = "decision_gate_no_differentiating_reason"
            reason = "Winner lacks a differentiating reason or hallucinates a comparison."
            recovery = SafeDegradeAction(message="Mỗi sản phẩm đều có điểm mạnh riêng, tôi chưa thấy lý do rõ ràng để ưu tiên hoàn toàn một sản phẩm so với các lựa chọn khác.")
        elif not has_criterion:
            trace_event = "decision_gate_missing_criterion"
            reason = "Missing user criterion and no clear price advantage."
            recovery = AskClarificationAction(question="Bạn muốn ưu tiên tiêu chí nào? (giá rẻ, pin lâu, chơi game...)")
        else:
            trace_event = "decision_gate_abstained"
            recovery = SafeDegradeAction(message="Mình chưa đủ căn cứ chọn ưu tiên hoàn toàn một sản phẩm. Bạn muốn ưu tiên tiêu chí nào?")

        return DecisionGateResult(
            allowed=False, abstained=True, reason=reason,
            required_user_criterion=has_criterion, has_verifiable_advantage=has_verifiable_advantage,
            confidence_ok=confidence_ok, margin_ok=margin_ok, evidence_ok=evidence_ok,
            differentiating_reasons_ok=differentiating_reasons_ok,
            recommendedRecovery=recovery,
            traceEvent=trace_event
        )
