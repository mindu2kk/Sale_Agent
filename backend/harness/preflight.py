from typing import Set, Dict, Optional
from backend.services.conversation import ConversationPlan, DecisionContext
from backend.harness.types import ExecutionBudget, BudgetUsed, PreflightResult

def run_preflight(
    plan: ConversationPlan,
    context: DecisionContext,
    budget: ExecutionBudget,
    catalog_valid_codes: Set[str],
    catalog_code_to_category: Dict[str, str],
    budget_used: Optional[BudgetUsed] = None,
) -> PreflightResult:
    # 6. Execution Budget Guard
    if budget_used:
        if budget_used.phase_events >= budget.max_phase_events:
            return PreflightResult(
                passed=False,
                decision="safe_degrade",
                reason="Event budget exhausted in preflight",
                trace_event="preflight_budget_blocked",
                blocked_guard="execution_budget",
                recovery_action="safe_degrade"
            )
        if budget_used.elapsed_ms >= budget.max_elapsed_ms:
            return PreflightResult(
                passed=False,
                decision="safe_degrade",
                reason="Latency budget exhausted in preflight",
                trace_event="preflight_budget_blocked",
                blocked_guard="execution_budget",
                recovery_action="safe_degrade"
            )

    # 2. Guard category compatibility
    if plan.dialogue_act == "catalog_ranking" and not plan.category:
        return PreflightResult(
            passed=False,
            decision="ask_clarification",
            reason="Ranking objective requires an explicit category",
            trace_event="preflight_category_mismatch",
            blocked_guard="category_compatibility",
            recovery_action="ask_clarification",
            clarification_message="Bạn muốn tìm máy thuộc nhóm nào (laptop hay điện thoại)?"
        )

    if plan.category and context.category and plan.category != context.category:
        # Category drift should have been handled. If it still conflicts here, block it.
        # Drift is usually handled by clearing state context, but if the preflight detects a conflict
        # between plan.category and what's expected without drift detection resolving it:
        pass # Normally drift is OK, let it pass since planner handled it.

    if len(plan.product_codes) > 1:
        cats = {catalog_code_to_category.get(code) for code in plan.product_codes if catalog_code_to_category.get(code)}
        if len(cats) > 1:
            return PreflightResult(
                passed=False,
                decision="rejected",
                reason="Cross-category comparison is not supported yet.",
                trace_event="preflight_category_mismatch",
                blocked_guard="category_compatibility",
                recovery_action="ask_clarification",
                clarification_message="Tôi chưa hỗ trợ so sánh sản phẩm thuộc các nhóm khác nhau (như laptop với điện thoại). Bạn hãy chọn các sản phẩm cùng loại nhé."
            )

    # 3. Guard product/SKU constraints
    invalid_codes = [c for c in plan.product_codes if c not in catalog_valid_codes]
    if invalid_codes:
        return PreflightResult(
            passed=False,
            decision="ask_clarification",
            reason=f"Unknown product codes: {invalid_codes}",
            trace_event="preflight_unknown_product",
            blocked_guard="product_constraints",
            recovery_action="ask_clarification",
            clarification_message="Mã sản phẩm bạn đề cập không tồn tại hoặc đã hết hàng. Bạn có mã nào khác không?"
        )

    # 4. Guard budget constraints
    if plan.price_intent:
        pi = plan.price_intent
        if pi.minimum is not None and pi.maximum is not None and pi.minimum > pi.maximum:
            return PreflightResult(
                passed=False,
                decision="ask_clarification",
                reason="Invalid budget range (min > max)",
                trace_event="preflight_invalid_budget",
                blocked_guard="budget_constraints",
                recovery_action="ask_clarification",
                clarification_message="Ngân sách bạn nhập (tối thiểu lớn hơn tối đa) không hợp lệ. Bạn có thể nói rõ lại khoảng giá mong muốn không?"
            )
        if (pi.minimum is not None and pi.minimum < 0) or (pi.maximum is not None and pi.maximum < 0) or (pi.target is not None and pi.target < 0):
            return PreflightResult(
                passed=False,
                decision="safe_degrade",
                reason="Negative budget",
                trace_event="preflight_invalid_budget",
                blocked_guard="budget_constraints",
                recovery_action="safe_degrade",
                clarification_message="Tôi không thể xử lý ngân sách âm."
            )

    # 5. Guard skill/objective validity
    if plan.dialogue_act == "catalog_ranking":
        supported_objectives = {"best_overall", "max_performance", "lowest_price", "highest_price"}
        if plan.goal not in supported_objectives:
            return PreflightResult(
                passed=False,
                decision="ask_clarification",
                reason=f"Unsupported ranking objective: {plan.goal}",
                trace_event="preflight_invalid_skill",
                blocked_guard="skill_validity",
                recovery_action="ask_clarification",
                clarification_message="Tiêu chí đánh giá này hiện chưa được hỗ trợ. Bạn muốn tìm máy mạnh nhất, rẻ nhất, hay tốt nhất toàn diện?"
            )

    # Stub for missing or inactive skill logic. If the act is totally unknown, reject.
    supported_acts = {
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
    }
    if plan.dialogue_act not in supported_acts:
        return PreflightResult(
            passed=False,
            decision="rejected",
            reason=f"Unsupported dialogue act: {plan.dialogue_act}",
            trace_event="preflight_invalid_skill",
            blocked_guard="skill_validity",
            recovery_action="abandon",
            clarification_message="Kỹ năng này chưa được kích hoạt."
        )

    return PreflightResult(
        passed=True,
        decision="approved",
        trace_event="preflight_passed"
    )
