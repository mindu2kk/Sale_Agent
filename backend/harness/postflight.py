from backend.harness.types import (
    VerificationResult,
    VerificationFailure,
    SafeDegradeAction,
    BeliefState,
    ConversationPlan,
    EvidenceRef
)
from backend.services.ai_service import AIAnswer
from backend.services.catalog import CatalogProduct
from backend.harness.epistemic import evaluate_decision_gate
from backend.harness.fallback import build_verified_fallback_response

def run_postflight_verification(
    answer: AIAnswer,
    candidate_set: list[CatalogProduct],
    evidence_refs: list[EvidenceRef],
    context: BeliefState,
    plan: ConversationPlan,
    catalog_revision: str,
    user_query: str = ""
) -> VerificationResult:
    failures = []

    if getattr(plan, "dialogue_act", None) == "general_explanation":
        return VerificationResult(passed=True)

    # 1. Candidate Containment Check
    candidate_codes = {p.code for p in candidate_set}
    answer_codes = set(answer.product_codes or [])

    escaped_codes = answer_codes - candidate_codes
    if escaped_codes:
        failures.append(VerificationFailure(
            code="candidate_not_contained",
            severity="blocker",
            message=f"Answer mentions unverified candidates: {', '.join(escaped_codes)}"
        ))

    # 2. Freshness Check
    stale_price_evidences = [
        ev for ev in evidence_refs
        if getattr(ev, "freshness", "unknown") == "stale"
    ]
    if stale_price_evidences:
        failures.append(VerificationFailure(
            code="stale_evidence",
            severity="warning",
            message="Evidence for price, promotion, or stock is stale."
        ))

    # Catalog revision check
    if context.catalog_revision != catalog_revision and catalog_revision:
        failures.append(VerificationFailure(
            code="catalog_revision_mismatch",
            severity="warning",
            message="Catalog revision changed during turn."
        ))

    # 3. Category/Policy Compatibility Check
    answer_text_lower = answer.text.lower()
    plan_category = getattr(plan, "category", getattr(plan, "expected_category", None))
    if plan_category == "phone":
        if "gpu" in answer_text_lower or "ssd" in answer_text_lower:
            failures.append(VerificationFailure(
                code="category_policy_mismatch",
                severity="blocker",
                message="Laptop-specific components (GPU/SSD) mentioned in phone answer."
            ))

    # 4. Claim-Evidence Keyword Heuristic Check
    evidence_fields = set()
    for ev in evidence_refs:
        if hasattr(ev, "field"):
            evidence_fields.add(ev.field)

    for p in candidate_set:
        if getattr(p, "price", None):
            evidence_fields.add("price")
        ctx = getattr(p, "context", "").lower()
        if "bảo hành" in ctx or "policy" in ctx:
            evidence_fields.update(["policy", "warranty"])
        if "bền" in ctx or "chuẩn quân đội" in ctx or "chống sốc" in ctx or "durability" in ctx:
            evidence_fields.update(["durability", "material", "certification"])
        if "pin" in ctx or "mah" in ctx or "battery" in ctx:
            evidence_fields.add("battery")
        if "gpu" in ctx or "rtx" in ctx or "nvidia" in ctx or "amd" in ctx or "card" in ctx:
            evidence_fields.add("gpu")
        if "ssd" in ctx or "hdd" in ctx or "storage" in ctx:
            evidence_fields.add("ssd")
        if "ram" in ctx or "gb" in ctx:
            evidence_fields.add("ram")
        if "cpu" in ctx or "intel" in ctx or "core" in ctx or "ryzen" in ctx:
            evidence_fields.add("cpu")
        if "màn hình" in ctx or "display" in ctx or "inch" in ctx:
            evidence_fields.add("display")

    claim_mapping = {
        ("rẻ nhất", "giá tốt nhất"): ["price"],
        ("bền", "bền bỉ", "chống sốc"): ["durability", "material", "warranty", "certification"],
        ("mạnh nhất", "khỏe nhất"): ["performance", "cpu", "gpu", "ram", "spec"],
        ("bảo hành", "đổi trả", "chính sách"): ["policy", "warranty"],
        ("pin tốt", "pin trâu"): ["battery"],
        ("gpu",): ["gpu", "spec"],
        ("ssd",): ["ssd", "storage", "spec"],
        ("ram",): ["ram", "memory", "spec"],
        ("cpu",): ["cpu", "processor", "spec"],
        ("display", "màn hình"): ["display", "screen", "spec"]
    }

    for keywords, req_fields in claim_mapping.items():
        if any(kw in answer_text_lower for kw in keywords):
            if not any(rf in evidence_fields for rf in req_fields):
                failures.append(VerificationFailure(
                    code="unsupported_claim",
                    severity="warning",  # Downgrade to warning
                    message=f"Claim related to '{keywords[0]}' lacks required evidence ({', '.join(req_fields)})."
                ))

    # 5. Epistemic Decision Gate (Advanced)
    decision_gate = evaluate_decision_gate(answer, plan, candidate_set, evidence_refs, context)
    if decision_gate.abstained:
        failures.append(VerificationFailure(
            code=decision_gate.trace_event or "weak_recommendation",
            severity="warning",  # Downgrade to warning
            message=decision_gate.reason or "Recommendation gate abstained."
        ))

    if failures:
        blockers = [f for f in failures if f.severity == "blocker"]
        warnings = [f for f in failures if f.severity == "warning"]

        if blockers:
            first_blocker_code = blockers[0].code
            ai_available = answer.mode != "catalog_fallback"
            recovery_msg = build_verified_fallback_response(
                reason=first_blocker_code,
                candidates=candidate_set,
                evidence_refs=evidence_refs,
                plan=plan,
                context=context,
                mode="hard_fail",
                user_query=user_query,
                ai_available=ai_available
            )
            return VerificationResult(
                passed=False,
                failures=failures,
                recovery_action=SafeDegradeAction(message=recovery_msg)
            )
        elif warnings:
            # Downgraded to soft fallback
            first_warning_code = warnings[0].code
            if "decision_gate_missing_criterion" in first_warning_code:
                mode = "clarification_with_facts"
            else:
                mode = "tradeoff_answer"

            ai_available = answer.mode != "catalog_fallback"
            fallback_msg = build_verified_fallback_response(
                reason=first_warning_code,
                candidates=candidate_set,
                evidence_refs=evidence_refs,
                plan=plan,
                context=context,
                mode=mode,
                user_query=user_query,
                ai_available=ai_available
            )
            return VerificationResult(
                passed=False,
                failures=failures,
                recovery_action=SafeDegradeAction(message=fallback_msg)
            )

    return VerificationResult(passed=True)
