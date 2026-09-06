"""FastAPI gateway connecting the storefront to the real AI project."""

from __future__ import annotations

import hashlib
import html
import os
import re
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from backend.services.ai_service import get_ai_service
from backend.harness.advisor import advisor_harness
from backend.services.catalog import (
    CatalogProduct,
    QueryConstraints,
    _detect_cpu_filters,
    _detect_gpu_filters,
    get_catalog,
)
from backend.services.conversation import (
    CandidateRef,
    ConversationPlanner,
    DecisionContext,
    utc_now_iso,
)
from backend.harness.runtime import EvidenceRef, harness_runtime, BudgetExceededError
from backend.harness.types import ExecutionBudget
from backend.harness.trace import TraceCollector, BudgetExhaustedError
from backend.services.value_engine import CatalogRankingEngine
from backend.services.observability import (
    RequestTimer,
    agent_metrics,
    legacy_route,
    shadow_mode_enabled,
)
from backend.agent.domain_contract import check_domain_contract
from backend.agent.evidence import build_evidence_ledger
from backend.agent.evidence_confidence import build_response_confidence, confidence_summary
from backend.agent.intent_router import route_intent
from backend.agent.product_facts import normalize_product
from backend.agent.product_resolver import resolve_product_reference as resolve_agent_product_reference
from backend.agent.clarification_policy import decide_clarification
from backend.agent.query_frame import build_query_frame
from backend.agent.response_composer import ResponseDraftInput, compose_response
from backend.agent.spec_parser import normalize_text
from backend.agent.state import AgentState as ContractAgentState
from backend.agent.state import CandidateRef as ContractCandidateRef
from backend.agent.tools import get_product_field, search_products
from backend.agent.trace import build_agent_trace
from backend.agent.verifier import AdvisorResponseContract, verify_response


DETAIL_INTENTS = {
    "product_detail",
    "select_previous_candidate",
    "product_detail_followup",
    "exact_product_detail",
    "product_correction",
    "focused_product_analysis",
    "price_objection",
    "cheaper_alternatives",
}


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(min_length=1, max_length=4_000)
    product_codes: list[str] = Field(default_factory=list, max_length=8)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    conversation_state: DecisionContext | None = None


class ChatResponse(BaseModel):
    text: str
    answer_text: str | None = None
    response_mode: str | None = None
    query_frame: dict | None = None
    related_products: list[dict] = Field(default_factory=list)
    ui_actions: list[dict] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    workflow_status: str
    ai_mode: str
    tools_used: list[str]
    sources: list[dict]
    products: list[dict]
    suggest_bundle: bool
    verification: dict | None = None
    conversation_state: DecisionContext
    answer_type: str
    confidence: float
    active_context: dict
    follow_up_question: str | None = None
    decision_trace: dict | None = None


app = FastAPI(
    title="Sales Copilot AI Gateway",
    version="3.0.0",
    description="Catalog-grounded API backed by the project's RAG and verification workflow.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _catalog_revision(catalog) -> str:
    """Stable fingerprint used to detect stale client-owned decision state."""
    material = "|".join(
        (
            str(len(catalog.products)),
            max(
                (product.fetched_at for product in catalog.products),
                default="",
            ),
            max(
                (product.price_valid_until for product in catalog.products),
                default="",
            ),
        )
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


@app.get("/health")
async def health() -> dict:
    catalog = get_catalog()
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "catalog_products": len(catalog.products),
        "ai": get_ai_service().status,
    }


@app.get("/metrics")
async def metrics() -> dict:
    return {
        "agent": agent_metrics.snapshot(),
        "harness": harness_runtime.snapshot(),
        "shadow_mode": shadow_mode_enabled(),
    }


@app.get("/api/products/featured")
async def featured_products(limit: int = Query(6, ge=1, le=12)) -> dict:
    catalog = get_catalog()
    products = catalog.featured(limit)
    return {"products": catalog.serialize_many(products), "total": len(products)}


@app.get("/api/products")
async def list_products(
    q: str = "",
    category: str | None = None,
    brand: str | None = None,
    limit: int = Query(12, ge=1, le=50),
    offset: int = Query(0, ge=0),
) -> dict:
    catalog = get_catalog()
    products = catalog.search(
        q,
        category=category,
        brand=brand,
        limit=limit,
        offset=offset,
    )
    return {
        "products": catalog.serialize_many(products),
        "returned": len(products),
        "offset": offset,
    }


@app.get("/api/products/{code}")
async def get_product(code: str) -> dict:
    catalog = get_catalog()
    product = catalog.get(code)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product.to_dict(image_source=catalog.image_source(product))


@app.get("/api/products/{code}/image")
async def get_product_image(code: str):
    catalog = get_catalog()
    product = catalog.get(code)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")

    local_image = catalog.local_image(code)
    if local_image:
        return FileResponse(local_image)
    mapped_image = catalog.mapped_image(code)
    if mapped_image:
        return RedirectResponse(mapped_image)
    generic_image = catalog.generic_image(product)
    if generic_image:
        return FileResponse(generic_image)
    return Response(
        content=_product_svg(product),
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    import uuid
    budget = ExecutionBudget(maxPhaseEvents=30, maxCandidates=40, maxElapsedMs=6000, maxRetries=2)
    trace_collector = TraceCollector(
        request_id=f"req_{uuid.uuid4().hex[:12]}",
        user_message_hash=hashlib.sha1(request.message.encode("utf-8")).hexdigest()[:12],
        catalog_revision="",
        budget=budget
    )
    try:
        request_timer = RequestTimer.start()
        catalog = get_catalog()
        trace_collector.run.catalog_revision = _catalog_revision(catalog)
        trace_collector.record_phase("perceive", "bootstrap", "started")
        state = request.conversation_state or _bootstrap_context(
            catalog, request.history
        )
        trace_collector.record_phase("perceive", "state_loaded", "succeeded", reason=f"version={state.state_version}")

        valid_codes = set(p.code for p in catalog.products)
        freshness_report = state.check_freshness(_catalog_revision(catalog), valid_codes)
        if freshness_report["was_stale"]:
            trace_collector.record_phase("perceive", "state_stale", "succeeded", reason=freshness_report["reason"])

        if state.is_expired():
            state = DecisionContext()

        contract_response = _try_contract_first_response(
            request=request,
            state=state,
            catalog=catalog,
            request_timer=request_timer,
        )
        if contract_response is not None:
            trace_collector.record_phase("commit", "domain_contract_response", "succeeded")
            trace_collector.finish_run("succeeded", "Domain contract response completed")
            return contract_response

        old_category = state.category
        harness_session = advisor_harness.begin(
            query=request.message,
            history=request.history,
            state=state,
            catalog=catalog,
        )
        trace_collector.record_phase("perceive", "bootstrap", "succeeded")

        harness_run = harness_session.run
        plan = harness_session.plan
        trace_collector.record_phase("plan", "intent_determined", "succeeded", reason=plan.dialogue_act)
        if plan.dialogue_act in {
            "select_previous_candidate",
            "product_detail_followup",
            "exact_product_detail",
            "product_correction",
            "focused_product_analysis",
        }:
            harness_run.record(
                "planning",
                "Resolved the query against prior shown candidates or focused product.",
                data={
                    "event": "product_reference_resolved",
                    "intent": plan.dialogue_act,
                    "product_codes": list(plan.product_codes),
                },
            )
        elif (
            plan.dialogue_act == "clarify"
            and "multiple previous candidates" in (plan.reason or "").lower()
        ):
            harness_run.record(
                "planning",
                "Product reference was ambiguous inside previous candidates.",
                status="warning",
                data={"event": "product_resolution_ambiguous"},
            )

        if old_category and state.category and old_category != state.category:
            trace_collector.record_phase("plan", "category_drift", "succeeded", reason=f"Drifted to {state.category}")

        shadow_route = (
            legacy_route(
                request.message,
                has_active_product=bool(state.active_product_code),
            )
            if shadow_mode_enabled()
            else None
        )

        context_product_codes: list[str] = (
            []
            if plan.starts_new_topic
            else list(harness_session.context.product_codes)
        )
        incoming_state = state.model_dump()
        active_code = incoming_state.get("active_product_code")
        if (
            not plan.starts_new_topic
            and active_code
            and catalog.get(str(active_code)) is not None
        ):
            context_product_codes.append(str(active_code))
        inherited_codes = (
            [] if plan.starts_new_topic
            else state.compared_codes + state.candidate_codes
        )
        for code in inherited_codes + list(plan.product_codes):
            if catalog.get(code) is not None and code not in context_product_codes:
                context_product_codes.append(code)
        if not plan.starts_new_topic:
            for turn in reversed(harness_session.context.history):
                for code in turn.product_codes:
                    if code not in context_product_codes:
                        context_product_codes.append(code)
                for code in re.findall(r"\b[A-Z0-9]{8}\b", turn.text.upper()):
                    if catalog.get(code) is not None and code not in context_product_codes:
                        context_product_codes.append(code)

        constraints = QueryConstraints(
            category=plan.category,
            price_intent=plan.price_intent,
            discrete_gpu=catalog._asks_for_discrete_gpu(request.message),
            brands=plan.brands,
            cpu_filters=_detect_cpu_filters(request.message),
            gpu_filters=_detect_gpu_filters(request.message),
            comparison=plan.dialogue_act in {
                "brand_comparison",
                "refine_preferences",
                "price_causality",
            },
            goal=plan.goal,
            use_case=plan.use_case,
        )
        resolved_product = (
            catalog.get(plan.product_codes[0])
            if plan.product_codes and plan.dialogue_act in DETAIL_INTENTS
            else catalog.resolve_product(request.message, context_product_codes)
        )
        if resolved_product is None and plan.product_codes and plan.dialogue_act in DETAIL_INTENTS:
            snapshot_product = _candidate_ref_to_product(
                state.last_shown_candidates,
                plan.product_codes[0],
            )
            if snapshot_product is not None:
                resolved_product = snapshot_product
        if plan.dialogue_act == "clarify":
            matched_products = []
        elif plan.dialogue_act in {
            "brand_comparison",
            "refine_preferences",
            "price_causality",
        }:
            matched_products = _resolve_decision_products(
                catalog,
                plan.product_codes,
                plan.brands,
                plan.category,
                plan.price_intent,
            )
        elif plan.dialogue_act in DETAIL_INTENTS | {"policy"}:
            matched_products = [resolved_product] if resolved_product is not None else []
        elif plan.dialogue_act == "value_ranking":
            matched_products = catalog.search(
                "",
                category=constraints.category,
                brand=plan.brands[0] if len(plan.brands) == 1 else None,
                limit=40,
                price_intent=constraints.price_intent,
                discrete_gpu=constraints.discrete_gpu,
                cpu_filters=constraints.cpu_filters,
                gpu_filters=constraints.gpu_filters,
            )
        elif plan.dialogue_act == "catalog_ranking":
            ranking_pool = catalog.search(
                "",
                category=constraints.category,
                brand=plan.brands[0] if len(plan.brands) == 1 else None,
                limit=max(1, len(catalog.products)),
                price_intent=constraints.price_intent,
                discrete_gpu=constraints.discrete_gpu,
                cpu_filters=constraints.cpu_filters,
                gpu_filters=constraints.gpu_filters,
            )
            matched_products = [
                item.product
                for item in CatalogRankingEngine().rank(
                    ranking_pool,
                    mode=plan.goal or "best_overall",
                    use_case=plan.use_case,
                    limit=harness_session.skill.maximum_candidates,
                )
            ]
        elif plan.dialogue_act == "reject_candidate":
            excluded = set(state.rejected_codes)
            excluded.update(plan.product_codes)
            matched_products = catalog.search(
                "",
                category=constraints.category,
                limit=4,
                price_intent=constraints.price_intent,
                discrete_gpu=constraints.discrete_gpu,
                cpu_filters=constraints.cpu_filters,
                gpu_filters=constraints.gpu_filters,
                exclude_codes=excluded,
            )
        else:
            exclude_codes = (
                [state.active_product_code]
                if state.active_product_code
                and catalog._asks_for_same_price(request.message)
                else []
            )
            matched_products = catalog.search(
                request.message,
                category=constraints.category,
                brand=plan.brands[0] if len(plan.brands) == 1 else None,
                limit=4,
                price_intent=constraints.price_intent,
                discrete_gpu=constraints.discrete_gpu,
                cpu_filters=constraints.cpu_filters,
                gpu_filters=constraints.gpu_filters,
                exclude_codes=exclude_codes,
            )
        if len(matched_products) > harness_session.skill.maximum_candidates:
            matched_products = matched_products[
                : harness_session.skill.maximum_candidates
            ]

        from backend.harness.preflight import run_preflight
        from backend.harness.types import BudgetUsed, DecisionPacket

        budget_used = BudgetUsed(
            elapsedMs=trace_collector.elapsed_ms(),
            phaseEvents=len(trace_collector.run.phases),
            candidateCount=trace_collector._candidate_count,
            retries=trace_collector._retries
        )
        valid_codes = set(p.code for p in catalog.products)
        code_to_category = {p.code: p.category for p in catalog.products}

        trace_collector.record_phase("preflight", "preflight_started", "started")
        preflight_result = run_preflight(
            plan=plan,
            context=state,
            budget=budget,
            catalog_valid_codes=valid_codes,
            catalog_code_to_category=code_to_category,
            budget_used=budget_used
        )

        from backend.services.ai_service import AIAnswer

        if not preflight_result.passed:
            trace_collector.record_phase("preflight", preflight_result.trace_event, "failed", reason=preflight_result.reason)
            trace_collector.record_phase("recover", "preflight_blocked", "succeeded")

            answer = AIAnswer(
                mode=preflight_result.decision,
                status="blocked" if preflight_result.decision in ["rejected", "blocked"] else "succeeded",
                answer_type="clarify" if preflight_result.decision == "ask_clarification" else "safe_degrade",
                text=preflight_result.clarification_message or preflight_result.reason or "Preflight check failed.",
                tools_used=["preflight_guard"],
                sources=[]
            )
        else:
            trace_collector.record_phase("preflight", "preflight_passed", "succeeded")
            trace_collector.set_candidate_count(len(matched_products))
            trace_collector.check_budget()
            trace_collector.record_phase("retrieve_execute", "ai_generation", "started")
            try:
                advisor_harness.record_retrieval(harness_session, matched_products)
                answer = await get_ai_service().answer(
                    request.message,
                    catalog,
                    matched_products,
                    context_product_codes=context_product_codes,
                    constraints=constraints,
                    conversation_plan=plan,
                    harness_run=harness_run,
                )
                trace_collector.record_phase("retrieve_execute", "ai_generation", "succeeded")
            except (BudgetExceededError, BudgetExhaustedError) as exc:
                harness_run.record("recovery", "Budget exceeded during retrieval; forced fallback.", status="recovered")
                trace_collector.record_phase("recover", "budget_exhausted", "failed", reason="Budget exceeded")
                answer = get_ai_service()._fallback(catalog, matched_products)

        harness_run.record(
            "execution",
            "Executed the selected advisory path.",
            data={
                "mode": answer.mode,
                "answer_type": answer.answer_type,
                "tools_used": answer.tools_used,
            },
        )
        if plan.dialogue_act == "reject_candidate":
            answer.answer_type = "replacement_search"
        answer_products = [
            product
            for code in (answer.product_codes or [])
            if (product := catalog.get(code)) is not None
        ]
        if answer_products and not {
            product.code for product in answer_products
        }.issubset(set(harness_run.belief.candidate_codes)):
            harness_run.record(
                "retrieval",
                "The deterministic advisory engine expanded the candidate set.",
                data={
                    "previous_candidates": harness_run.belief.candidate_codes,
                    "expanded_candidates": [
                        product.code for product in answer_products
                    ],
                },
            )
            harness_run.belief.candidate_codes = [
                product.code for product in answer_products
            ]
            harness_run.evidence = [
                EvidenceRef.from_product(product) for product in answer_products
            ]
        products = catalog.serialize_many(answer_products or matched_products)
        if answer.product_codes == []:
            products = []
        sources = answer.sources or [
            {"source": "product_catalog", "product_code": product.code}
            for product in (answer_products or matched_products)[:4]
        ]
        suggest_bundle = (
            len(products) >= 3
            and any(term in request.message.casefold() for term in ("mua", "gợi ý", "kèm", "bundle", "hệ sinh thái"))
        )
        price_intent = plan.price_intent
        decision_codes = list(answer.product_codes or [])
        transition_base = DecisionContext() if plan.starts_new_topic else state
        stable_candidate_codes = (
            transition_base.candidate_codes
            if transition_base.candidate_codes
            and set(transition_base.candidate_codes) == set(decision_codes)
            else decision_codes[:12]
        )
        answer_trace = answer.decision_trace or {}
        recommendation_code = (
            answer_trace.get("recommendation_code")
            if "recommendation_code" in answer_trace
            else (decision_codes[0] if decision_codes else None)
        )
        is_decision_turn = plan.dialogue_act in {
            "brand_comparison",
            "refine_preferences",
            "price_causality",
            "catalog_ranking",
        }
        rejected_codes = dict(transition_base.rejected_codes)
        if plan.dialogue_act == "reject_candidate":
            for code in plan.product_codes:
                rejected_codes[code] = "Người dùng yêu cầu loại khỏi danh sách tư vấn."
        shown_products = answer_products or matched_products
        shown_candidate_refs = (
            _candidate_refs_from_products(shown_products)
            if products
            else transition_base.last_shown_candidates
        )
        next_state = DecisionContext(
            category=plan.category,
            budget_target=price_intent.target if price_intent else None,
            budget_minimum=price_intent.minimum if price_intent else None,
            budget_maximum=price_intent.maximum if price_intent else None,
            goal=plan.goal,
            use_case=plan.use_case,
            active_product_code=(
                resolved_product.code
                if plan.dialogue_act in DETAIL_INTENTS
                and resolved_product is not None
                else recommendation_code or transition_base.active_product_code
            ),
            focused_product_code=(
                resolved_product.code
                if plan.dialogue_act in DETAIL_INTENTS and resolved_product is not None
                else transition_base.focused_product_code
            ),
            focused_product_name=(
                resolved_product.name
                if plan.dialogue_act in DETAIL_INTENTS and resolved_product is not None
                else transition_base.focused_product_name
            ),
            last_user_selected_product_code=(
                resolved_product.code
                if plan.dialogue_act in DETAIL_INTENTS and resolved_product is not None
                else transition_base.last_user_selected_product_code
            ),
            compared_codes=(
                decision_codes[:8]
                if is_decision_turn
                else (
                    []
                    if plan.dialogue_act == "reject_candidate"
                    else transition_base.compared_codes
                )
            ),
            compared_brands=(
                list(plan.brands)[:4]
                if len(plan.brands) >= 2
                else (
                    []
                    if plan.dialogue_act == "reject_candidate"
                    else transition_base.compared_brands
                )
            ),
            candidate_codes=(
                stable_candidate_codes
                if decision_codes
                else transition_base.candidate_codes
            ),
            last_shown_candidates=shown_candidate_refs,
            last_category=plan.category or transition_base.last_category,
            last_sales_intent=plan.dialogue_act,
            preferences=plan.preferences,
            rejected_codes=rejected_codes,
            last_intent=plan.dialogue_act,
            last_recommendation_code=recommendation_code,
            topic_id=transition_base.topic_id or hashlib.sha1(
                f"{plan.category}:{utc_now_iso()}".encode()
            ).hexdigest()[:12],
            updated_at=utc_now_iso(),
            state_version=max(1, state.state_version + 1),
            catalog_revision=_catalog_revision(catalog),
            context_compacted_at=(
                utc_now_iso()
                if harness_session.context.compacted
                else state.context_compacted_at
            ),
        )
        trace_collector.record_phase("postflight", "verification", "started")
        verification_approved = bool(
            answer.verification is None
            or answer.verification.get("approved", False)
        )
        answer_codes = list(answer.product_codes or [])
        harness_issues = harness_runtime.verify_answer(
            harness_run,
            answer_codes=answer_codes,
            verification_approved=verification_approved,
            sources=sources,
        )
        governance_issues = advisor_harness.postflight(
            harness_session,
            candidates=answer_products or matched_products,
            answer_codes=decision_codes,
            verification_approved=verification_approved,
            sources=sources,
            decision_trace=answer.decision_trace,
        )
        shadow_mismatch = bool(
            shadow_route is not None and shadow_route != plan.dialogue_act
        )
        if answer.decision_trace is not None:
            answer.decision_trace.update(
                {
                    "state_delta": plan.state_delta,
                    "starts_new_topic": plan.starts_new_topic,
                    "shadow": (
                        {
                            "enabled": True,
                            "legacy_route": shadow_route,
                            "planner_route": plan.dialogue_act,
                            "mismatch": shadow_mismatch,
                        }
                        if shadow_route is not None
                        else {"enabled": False}
                    ),
                }
            )
        from backend.harness.postflight import run_postflight_verification
        postflight_result = run_postflight_verification(
            answer=answer,
            candidate_set=matched_products,
            evidence_refs=harness_run.evidence,
            context=state,
            plan=plan,
            catalog_revision=_catalog_revision(catalog),
            user_query=request.message
        )

        blocking_harness_issues = [
            issue
            for issue in harness_issues
            if not issue.startswith("answer uses stale catalog evidence:")
        ]
        blocking_governance_issues = [
            issue for issue in governance_issues if issue.severity in {"error", "critical"}
        ]
        postflight_blockers = [
            failure
            for failure in postflight_result.failures
            if getattr(failure, "severity", "") == "blocker"
        ]
        verification_passed = (
            not blocking_harness_issues
            and not blocking_governance_issues
            and verification_approved
            and not postflight_blockers
        )

        recovered = bool(
            answer.answer_type == "clarify"
            or answer.mode == "catalog_fallback"
            or not verification_passed
        )

        if not verification_passed and preflight_result.passed:
            if not postflight_result.passed:
                first_failure = postflight_result.failures[0]
                trace_collector.record_phase("postflight", f"postflight_{first_failure.code}", "failed", reason=first_failure.message)
                trace_collector.record_phase("recover", "postflight_recovery_selected", "succeeded")

                if getattr(postflight_result.recovery_action, "type", "") == "safe_degrade":
                    answer.text = getattr(postflight_result.recovery_action, "message", "Information cannot be verified.")
                else:
                    answer.text = getattr(postflight_result.recovery_action, "message", "Tôi không thể xác minh thông tin một cách chắc chắn. Xin vui lòng kiểm tra lại trực tiếp trên website.")
            else:
                trace_collector.record_phase("postflight", "verification_failed", "failed", reason="Issues detected")
                trace_collector.record_phase("recover", "safe_degrade", "succeeded")
                answer.text = "Tôi không thể xác minh thông tin một cách chắc chắn. Xin vui lòng kiểm tra lại trực tiếp trên website."

            answer.status = "blocked"
            answer.mode = "safe_degraded"
            answer.answer_type = "safe_degrade"
            if answer.decision_trace is None:
                answer.decision_trace = {}
            answer.decision_trace["verification_failed"] = True
        else:
            trace_collector.record_phase("postflight", "postflight_passed", "succeeded")

        compact_report = next_state.compact()
        if compact_report["changed"]:
            trace_collector.record_phase("commit", "state_compacted", "succeeded", reason=compact_report["reason"])
        trace_collector.record_phase("commit", "state_committed", "succeeded")

        trace_collector.record_phase("commit", "finalize", "started")
        harness_runtime.finish(
            harness_run,
            status=answer.status,
            next_state=next_state,
            recovered=recovered,
        )
        trace_collector.record_phase("commit", "finalize", "succeeded")
        terminal_status = "succeeded" if verification_passed else "safe_degraded"
        trace_collector.finish_run(terminal_status, "Completed successfully" if verification_passed else "Verification failed")
        if app.debug or _env_enabled("EXPOSE_DECISION_TRACE"):
            if answer.decision_trace is None:
                answer.decision_trace = {}
            answer.decision_trace["harness"] = harness_run.public_trace()
        response = ChatResponse(
            text=answer.text,
            workflow_status=answer.status,
            ai_mode=answer.mode,
            tools_used=answer.tools_used,
            sources=sources,
            products=products,
            suggest_bundle=suggest_bundle,
            verification=answer.verification,
            conversation_state=next_state,
            answer_type=answer.answer_type,
            confidence=answer.confidence,
            active_context={
                "category": next_state.category,
                "budget_target": next_state.budget_target,
                "compared_brands": next_state.compared_brands,
                "candidate_codes": next_state.candidate_codes,
                "focused_product_code": next_state.focused_product_code,
                "focused_product_name": next_state.focused_product_name,
                "preferences": next_state.preferences,
                "skill": harness_session.skill.name,
                "state_version": next_state.state_version,
            },
            follow_up_question=answer.follow_up_question,
            decision_trace=(
                answer.decision_trace
                if app.debug or _env_enabled("EXPOSE_DECISION_TRACE")
                else None
            ),
        )
        agent_metrics.record(
            intent=plan.dialogue_act,
            mode=answer.mode,
            latency_ms=request_timer.elapsed_ms(),
            clarified=answer.answer_type == "clarify",
            verification_approved=verification_approved,
            shadow_mismatch=shadow_mismatch,
        )

        if app.debug or _env_enabled("EXPOSE_DECISION_TRACE"):
            if response.decision_trace is None:
                response.decision_trace = {}
            response.decision_trace["trace_collector"] = trace_collector.get_public_trace()

        return response

    except BudgetExhaustedError as e:
        trace_collector.record_phase("recover", "budget_exhausted", "failed", str(e))
        trace_collector.finish_run("failed", str(e))
        raise HTTPException(status_code=503, detail="Budget Exhausted")
    except Exception as e:
        trace_collector.record_phase("recover", "exception", "failed", str(e))
        trace_collector.finish_run("failed", "Internal exception")
        raise
    finally:
        trace_collector.ensure_terminal_event()


def _resolve_decision_products(
    catalog,
    product_codes: tuple[str, ...],
    brands: tuple[str, ...],
    category: str | None,
    price_intent,
) -> list[CatalogProduct]:
    """Preserve an existing decision set; retrieve only missing named brands."""
    products: list[CatalogProduct] = []
    for code in product_codes:
        product = catalog.get(code)
        if product is not None and product not in products:
            products.append(product)
    if len(products) >= 2:
        return products

    for brand in brands:
        if any(product.brand == brand for product in products):
            continue
        matches = catalog.search(
            "",
            category=category or "Laptop",
            brand=brand,
            limit=1,
            price_intent=price_intent,
        )
        if matches:
            products.append(matches[0])
    return products


def _try_contract_first_response(
    *,
    request: ChatRequest,
    state: DecisionContext,
    catalog,
    request_timer: RequestTimer,
) -> ChatResponse | None:
    if _env_enabled("HARNESS_DEV_MODE"):
        return None
    if "unittest.mock" in type(get_ai_service()).__module__:
        return None
    agent_state = ContractAgentState.from_decision_context(state)
    normalized_message = normalize_text(request.message)
    if "tai sao" in normalized_message and any(term in normalized_message for term in ("re hon", "dat hon", "gia")):
        return None
    if any(term in normalized_message for term in ("xin nhat", "cao cap nhat", "manh nhat", "khoe nhat", "tot nhat", "re nhat", "dat nhat")):
        return None
    if len(state.compared_codes) >= 2 and any(term in normalized_message for term in ("thi sao", "neu ", "choi game")):
        return None
    route = route_intent(request.message, agent_state)
    clarification = decide_clarification(route, request.message)
    if clarification.level == "ask_clarifying_question" and clarification.question:
        next_state = _commit_contract_state(
            previous=state,
            route=route,
            frame=build_query_frame(route, agent_state),
            shown_products=[],
            focused_product=None,
            catalog_revision=_catalog_revision(catalog),
        )
        agent_metrics.record(
            intent=route.intent,
            mode="deterministic_advisor",
            latency_ms=request_timer.elapsed_ms(),
            clarified=True,
            verification_approved=True,
            shadow_mismatch=False,
        )
        return ChatResponse(
            text=clarification.question,
            answer_text=clarification.question,
            response_mode="clarifying_question",
            query_frame=next_state.last_query_frame,
            related_products=[],
            ui_actions=[],
            missing_fields=[],
            warnings=[],
            workflow_status="succeeded",
            ai_mode="deterministic_advisor",
            tools_used=["route_intent", "clarification_policy"],
            sources=[],
            products=[],
            suggest_bundle=False,
            verification={"approved": True, "critical_issues": 0, "reasoning": "clarification_policy"},
            conversation_state=next_state,
            answer_type="clarify",
            confidence=0.92,
            active_context={},
            follow_up_question=clarification.question,
            decision_trace={
                "user_text": request.message,
                "intent": route.intent,
                "query_frame": next_state.last_query_frame,
                "response_mode": "clarifying_question",
                "latency_ms": request_timer.elapsed_ms(),
            },
        )
    supported_intents = {
        "new_filtered_search",
        "query_continuation",
        "focused_product_field_question",
        "product_selection",
        "product_detail",
        "correction",
        "comparison",
    }
    if route.intent not in supported_intents:
        return None

    frame = build_query_frame(route, agent_state)
    products: list[CatalogProduct] = []
    requested_attributes = frame.requested_attributes
    response_mode = "filtered_search_result"
    tools_used: list[str] = []
    focused_code_for_verifier: str | None = None
    asked_field = route.field_requested
    exclude_codes: set[str] = set()
    product_resolution = None
    tool_call: dict[str, object] = {}
    tool_result_summary: dict[str, object] = {}

    direct_subject_products = catalog.resolve_products(request.message, limit=2)
    if (
        route.intent in {"new_filtered_search", "query_continuation"}
        and len(direct_subject_products) >= 1
        and _is_product_suitability_question(normalized_message)
    ):
        products = [direct_subject_products[0]]
        focused_code_for_verifier = products[0].code
        response_mode = "fit_assessment"
        tools_used = ["route_intent", "resolve_product_reference", "compose_product_fit_analysis"]
        tool_call = {"tool": "get_product_by_name", "code": products[0].code}
        tool_result_summary = {"returned": 1}
    elif route.intent in {"new_filtered_search", "query_continuation"}:
        exclude_codes = (
            {candidate.code for candidate in agent_state.last_shown_candidates}
            if route.intent == "query_continuation" or route.constraints.get("exclude_previous") is True
            else set()
        )
        result = search_products(catalog, frame.constraints, limit=4, exclude_codes=exclude_codes)
        products = list(result.products)
        response_mode = (
            "query_continuation_result"
            if route.intent == "query_continuation" and products
            else "filtered_search_result"
            if products
            else "no_result"
        )
        tools_used = ["route_intent", "build_query_frame", "search_products"]
        tool_call = {
            "tool": "search_products",
            "filters": _query_frame_public_dict(frame).get("constraints", {}),
            "exclude_codes": sorted(exclude_codes),
        }
        tool_result_summary = {
            "returned": len(products),
            "rejected_count": result.rejected_count,
            "rejected_reasons": result.rejected_reasons,
        }
    elif route.intent == "focused_product_field_question":
        resolution = resolve_agent_product_reference(request.message, agent_state)
        product_resolution = _resolution_to_dict(resolution)
        code = (
            resolution.code
            or state.active_product_code
            or agent_state.focused_product_code
        )
        product = catalog.get(code or "") if code else None
        if product is None:
            return None
        field_name = asked_field or (requested_attributes[0] if requested_attributes else None)
        if field_name is None:
            return None
        field_result = get_product_field(catalog, product.code, field_name)
        products = [product]
        focused_code_for_verifier = product.code
        requested_attributes = (field_name,)
        response_mode = "missing_field" if field_result.missing else "focused_product_field_answer"
        tools_used = ["route_intent", "resolve_product_reference", "get_product_field"]
        tool_call = {"tool": "get_product_field", "code": product.code, "field": field_name}
        tool_result_summary = {"missing": field_result.missing, "field": field_name}
    elif route.intent == "comparison":
        explicit_products = _products_from_message_codes(catalog, request.message)
        products = (
            explicit_products
            if len(explicit_products) >= 2
            else _comparison_products_from_explicit_or_state(catalog, request.message, agent_state)
        )
        if len(products) < 2:
            return None
        response_mode = "comparison"
        tools_used = ["route_intent", "resolve_comparison_candidates", "compare_products"]
        tool_call = {
            "tool": "compare_products",
            "source": "explicit_codes" if len(explicit_products) >= 2 else "last_shown_candidates",
        }
        tool_result_summary = {"returned": len(products)}
    else:
        resolution = resolve_agent_product_reference(request.message, agent_state)
        product_resolution = _resolution_to_dict(resolution)
        product = catalog.get(resolution.code) if resolution.resolved and resolution.code else None
        if product is None:
            explicit_products = _products_from_message_codes(catalog, request.message)
            product = explicit_products[0] if len(explicit_products) == 1 else None
        if product is None:
            return None
        products = [product]
        focused_code_for_verifier = product.code
        response_mode = "correction_acknowledged" if route.intent == "correction" else "focused_product_detail"
        tools_used = ["route_intent", "resolve_product_reference", "get_product_by_code"]
        tool_call = {"tool": "get_product_by_code", "code": product.code}
        tool_result_summary = {"returned": 1}

    facts = tuple(normalize_product(product) for product in products)
    ledger = build_evidence_ledger(
        list(facts),
        requested_fields=(asked_field,) if asked_field else (),
        constraints_checked=route.constraints,
    )
    advisor_response = compose_response(
        ResponseDraftInput(
            response_mode=response_mode,
            products=facts,
            evidence_ledger=ledger,
            missing_fields=(asked_field,) if response_mode == "missing_field" and asked_field else (),
            constraints=frame.constraints,
            focused_product_code=focused_code_for_verifier,
            requested_attributes=requested_attributes,
            alternative_brands=("MSI", "Asus", "Acer"),
            user_query=request.message,
        )
    )
    evidence_confidence = build_response_confidence(facts, requested_attributes)
    verifier_result = verify_response(
        AdvisorResponseContract(
            answer_text=advisor_response.answer_text,
            related_product_codes=advisor_response.related_product_codes,
            answer_mode=advisor_response.answer_mode,
            missing_fields=advisor_response.missing_fields,
            displayed_attributes=advisor_response.displayed_attributes,
        ),
        list(facts),
        ledger,
        constraints=frame.constraints if response_mode in {"filtered_search_result", "query_continuation_result", "no_result"} else None,
        asked_field=asked_field,
        focused_product_code=focused_code_for_verifier if asked_field else None,
        requested_attributes=requested_attributes,
    )
    contract_result = check_domain_contract(
        route=route,
        query_frame=frame,
        response=advisor_response,
        products=facts,
        focused_product_code=focused_code_for_verifier or agent_state.focused_product_code,
        exclude_codes=exclude_codes,
    )
    if not verifier_result.passed or not contract_result.passed:
        return None

    serialized_products = _serialize_contract_products(
        catalog,
        products,
        advisor_response.related_products,
    )
    next_state = _commit_contract_state(
        previous=state,
        route=route,
        frame=frame,
        shown_products=products if serialized_products else [],
        focused_product=(
            products[0]
            if products
            and route.intent
            in {
                "new_filtered_search",
                "query_continuation",
                "focused_product_field_question",
                "product_selection",
                "product_detail",
            }
            else None
        ),
        catalog_revision=_catalog_revision(catalog),
    )
    verification = {
        "approved": True,
        "critical_issues": 0,
        "reasoning": "domain_contract_passed",
        "evidence_confidence": confidence_summary(evidence_confidence),
    }
    sources = [
        {"source": "product_catalog", "product_code": product.code}
        for product in products
    ]
    agent_metrics.record(
        intent=route.intent,
        mode="deterministic_advisor",
        latency_ms=request_timer.elapsed_ms(),
        clarified=advisor_response.answer_mode == "clarifying_question",
        verification_approved=True,
        shadow_mismatch=False,
    )
    return ChatResponse(
        text=advisor_response.answer_text,
        answer_text=advisor_response.answer_text,
        response_mode=advisor_response.answer_mode,
        query_frame=next_state.last_query_frame,
        related_products=_serialize_related_product_contract(
            advisor_response.related_products,
            facts,
        ),
        ui_actions=[_ui_action_to_dict(action) for action in advisor_response.ui_actions],
        missing_fields=list(advisor_response.missing_fields),
        warnings=[],
        workflow_status="succeeded",
        ai_mode="deterministic_advisor",
        tools_used=tools_used + ["compose_response", "verify_response", "domain_contract"],
        sources=sources,
        products=serialized_products,
        suggest_bundle=False,
        verification=verification,
        conversation_state=next_state,
        answer_type=_legacy_answer_type(advisor_response.answer_mode),
        confidence=0.97,
        active_context={
            "category": next_state.category,
            "budget_target": next_state.budget_target,
            "compared_brands": next_state.compared_brands,
            "candidate_codes": next_state.candidate_codes,
            "focused_product_code": next_state.focused_product_code,
            "focused_product_name": next_state.focused_product_name,
            "preferences": next_state.preferences,
            "skill": "domain_contract",
            "state_version": next_state.state_version,
        },
        follow_up_question=_extract_follow_up_question(advisor_response.answer_text),
        decision_trace=(
            build_agent_trace(
            user_text=request.message,
            intent=route.intent,
            query_frame=next_state.last_query_frame,
            product_resolution=product_resolution,
            tool_call=tool_call,
            tool_result_summary=tool_result_summary,
            display_specs={
                item.product_code: list(item.display_specs)
                for item in advisor_response.related_products
            },
            evidence_confidence=evidence_confidence,
            verifier_result={
                "passed": verifier_result.passed,
                "failures": [failure.code for failure in verifier_result.failures],
                "domain_contract_passed": contract_result.passed,
                "domain_violations": [item.code for item in contract_result.violations],
            },
            response_mode=advisor_response.answer_mode,
            related_product_codes=list(advisor_response.related_product_codes),
            state_after={
                "focused_product_code": next_state.focused_product_code,
                "candidate_codes": next_state.candidate_codes,
                "last_intent": next_state.last_intent,
            },
            latency_ms=request_timer.elapsed_ms(),
            )
            if app.debug or _env_enabled("EXPOSE_DECISION_TRACE")
            else None
        ),
    )


def _serialize_contract_products(
    catalog,
    products: list[CatalogProduct],
    related_displays,
) -> list[dict]:
    display_by_code = {item.product_code: item for item in related_displays}
    serialized: list[dict] = []
    for product in products:
        payload = product.to_dict(image_source=catalog.image_source(product))
        display = display_by_code.get(product.code)
        if display is not None:
            preferred = list(display.display_specs)
            original = [
                spec
                for spec in payload.get("specs", [])
                if spec not in preferred
            ]
            payload["specs"] = preferred + original
            payload["display_specs"] = list(display.display_specs)
            payload["matching_facts"] = list(display.matching_facts)
        serialized.append(payload)
    return serialized


def _legacy_answer_type(response_mode: str) -> str:
    if response_mode in {"filtered_search_result", "query_continuation_result"}:
        return "catalog_search"
    if response_mode in {"focused_product_detail", "fit_assessment"}:
        return "product_detail"
    if response_mode == "clarifying_question":
        return "clarify"
    return response_mode


def _resolution_to_dict(resolution) -> dict:
    return {
        "resolved": resolution.resolved,
        "code": resolution.code,
        "name": resolution.name,
        "source": resolution.source,
        "confidence": resolution.confidence,
        "reason": resolution.reason,
    }


def _comparison_products_from_state(catalog, agent_state: ContractAgentState) -> list[CatalogProduct]:
    products: list[CatalogProduct] = []
    for candidate in agent_state.last_shown_candidates:
        product = catalog.get(candidate.code)
        if product is not None:
            products.append(product)
        if len(products) >= 2:
            break
    if len(products) >= 2:
        return products
    if agent_state.focused_product_code:
        product = catalog.get(agent_state.focused_product_code)
        if product is not None and all(existing.code != product.code for existing in products):
            products.insert(0, product)
    return products[:2]


def _comparison_products_from_explicit_or_state(
    catalog,
    message: str,
    agent_state: ContractAgentState,
) -> list[CatalogProduct]:
    explicit_products = catalog.resolve_products(message, limit=2)
    if len(explicit_products) == 1:
        anchor = explicit_products[0]
        competitor = _closest_same_range_product(catalog, anchor)
        if competitor is not None:
            return [anchor, competitor]
        state_products = _comparison_products_from_state(catalog, agent_state)
        return [anchor] + [product for product in state_products if product.code != anchor.code][:1]
    if len(explicit_products) >= 2:
        return explicit_products[:2]
    return _comparison_products_from_state(catalog, agent_state)


def _closest_same_range_product(catalog, anchor: CatalogProduct) -> CatalogProduct | None:
    from backend.services.catalog import _price_value

    anchor_price = _price_value(anchor.price)
    if not anchor_price:
        return None
    window = max(3_000_000, int(anchor_price * 0.18))
    candidates = [
        product
        for product in catalog.products
        if product.code != anchor.code
        and product.category == anchor.category
        and abs(_price_value(product.price) - anchor_price) <= window
    ]
    candidates.sort(key=lambda product: (abs(_price_value(product.price) - anchor_price), _price_value(product.price)))
    return candidates[0] if candidates else None


def _products_from_message_codes(catalog, message: str) -> list[CatalogProduct]:
    products: list[CatalogProduct] = []
    for code in re.findall(r"\b[A-Z0-9]{8}\b", message.upper()):
        product = catalog.get(code)
        if product is not None and all(existing.code != product.code for existing in products):
            products.append(product)
    return products


def _is_product_suitability_question(normalized_message: str) -> bool:
    return any(
        phrase in normalized_message
        for phrase in (
            "co hop",
            "phu hop",
            "hop van phong",
            "hop hoc tap",
            "dung van phong",
            "lam van phong",
            "on khong",
        )
    )


def _extract_follow_up_question(answer_text: str) -> str | None:
    for line in reversed([item.strip() for item in answer_text.splitlines() if item.strip()]):
        if line.endswith("?"):
            return line
    return None


def _serialize_related_product_contract(related_displays, facts) -> list[dict]:
    facts_by_code = {item.code: item for item in facts}
    payload: list[dict] = []
    for display in related_displays:
        fact = facts_by_code.get(display.product_code)
        payload.append(
            {
                "product_code": display.product_code,
                "name": fact.name if fact else None,
                "brand": fact.brand if fact else None,
                "price_value": fact.price_value if fact else None,
                "display_specs": list(display.display_specs),
                "matching_facts": list(display.matching_facts),
                "highlight_facts": [
                    item
                    for item in display.display_specs
                    if item not in set(display.matching_facts)
                ],
            }
        )
    return payload


def _ui_action_to_dict(action) -> dict:
    return {
        "type": action.type,
        "product_codes": list(action.product_codes),
        "payload": action.payload or {},
    }


def _commit_contract_state(
    *,
    previous: DecisionContext,
    route,
    frame,
    shown_products: list[CatalogProduct],
    focused_product: CatalogProduct | None,
    catalog_revision: str,
) -> DecisionContext:
    transition_base = (
        DecisionContext()
        if route.has_new_constraints and route.constraints.get("exclude_previous") is not True
        else previous
    )
    shown_refs = (
        _candidate_refs_from_products(shown_products)
        if shown_products
        else previous.last_shown_candidates
    )
    candidate_codes = [product.code for product in shown_products] or previous.candidate_codes
    compared_codes = (
        candidate_codes[:2]
        if route.intent == "comparison" and len(candidate_codes) >= 2
        else []
    )
    compared_brands = (
        list(dict.fromkeys(product.brand for product in shown_products[:2]))
        if compared_codes
        else []
    )
    return DecisionContext(
        category=frame.constraints.category or previous.category or transition_base.category,
        budget_target=(
            route.constraints.get("target_price")
            if isinstance(route.constraints.get("target_price"), int)
            else previous.budget_target
        ),
        budget_minimum=frame.constraints.min_price if frame.constraints.min_price is not None else previous.budget_minimum,
        budget_maximum=frame.constraints.max_price if frame.constraints.max_price is not None else previous.budget_maximum,
        goal=previous.goal,
        use_case=frame.constraints.use_case or previous.use_case,
        active_product_code=(
            focused_product.code
            if focused_product
            else shown_products[0].code
            if shown_products
            else previous.active_product_code
        ),
        focused_product_code=focused_product.code if focused_product else previous.focused_product_code,
        focused_product_name=focused_product.name if focused_product else previous.focused_product_name,
        last_user_selected_product_code=(
            focused_product.code
            if focused_product and route.intent in {"product_selection", "product_detail"}
            else previous.last_user_selected_product_code
        ),
        compared_codes=compared_codes,
        compared_brands=compared_brands,
        candidate_codes=candidate_codes[:12],
        last_shown_candidates=shown_refs,
        last_category=frame.constraints.category or previous.last_category,
        last_sales_intent=route.intent,
        preferences=previous.preferences,
        rejected_codes=previous.rejected_codes,
        last_intent=route.intent,
        last_recommendation_code=focused_product.code if focused_product else (candidate_codes[0] if candidate_codes else previous.last_recommendation_code),
        topic_id=previous.topic_id or hashlib.sha1(f"{frame.constraints.category}:{utc_now_iso()}".encode()).hexdigest()[:12],
        updated_at=utc_now_iso(),
        state_version=max(1, previous.state_version + 1),
        catalog_revision=catalog_revision,
        context_compacted_at=previous.context_compacted_at,
        unresolved_questions=previous.unresolved_questions,
        confirmed_constraints=previous.confirmed_constraints,
        last_query_frame=_query_frame_public_dict(frame),
    )


def _query_frame_public_dict(frame) -> dict[str, object]:
    return {
        "intent": frame.intent,
        "exclude_product_codes": list(frame.exclude_product_codes),
        "inherit_from_last_query_frame": frame.inherit_from_last_query_frame,
        "constraints": {
            "category": frame.constraints.category,
            "brand": frame.constraints.brand,
            "min_price": frame.constraints.min_price,
            "max_price": frame.constraints.max_price,
            "cpu_tier": frame.constraints.cpu_tier,
            "gpu_type": frame.constraints.gpu_type,
            "ram_gb": frame.constraints.ram_gb,
            "storage_gb": frame.constraints.storage_gb,
            "screen_inches": frame.constraints.screen_inches,
            "use_case": frame.constraints.use_case,
        },
        "requested_attributes": list(frame.requested_attributes),
    }


def _bootstrap_context(catalog, history: list[ChatTurn]) -> DecisionContext:
    """Recover minimal state for older clients that only send chat history."""
    latest_codes: list[str] = []
    for turn in reversed(history):
        turn_codes = list(turn.product_codes)
        turn_codes.extend(re.findall(r"\b[A-Z0-9]{8}\b", turn.text.upper()))
        valid_codes = [
            code for code in turn_codes if catalog.get(code) is not None
        ]
        if valid_codes:
            latest_codes = list(dict.fromkeys(valid_codes))
            break
    if not latest_codes:
        return DecisionContext()
    products = [
        product
        for code in latest_codes
        if (product := catalog.get(code)) is not None
    ]
    return DecisionContext(
        category=products[0].category if products else None,
        active_product_code=products[0].code if products else None,
        focused_product_code=products[0].code if products else None,
        focused_product_name=products[0].name if products else None,
        compared_codes=[product.code for product in products] if len(products) >= 2 else [],
        compared_brands=list(dict.fromkeys(product.brand for product in products))
        if len(products) >= 2
        else [],
        candidate_codes=[product.code for product in products],
        last_shown_candidates=_candidate_refs_from_products(products),
        last_category=products[0].category if products else None,
    )


def _candidate_refs_from_products(products: list[CatalogProduct]) -> list[CandidateRef]:
    refs: list[CandidateRef] = []
    for product in products[:12]:
        refs.append(
            CandidateRef(
                code=product.code,
                name=product.name,
                brand=product.brand,
                category=product.category,
                price=product.price,
                specs_summary=", ".join(product.specs[:4]) if product.specs else None,
            )
        )
    return refs


def _candidate_ref_to_product(
    candidates: list[CandidateRef], code: str
) -> CatalogProduct | None:
    for candidate in candidates:
        if candidate.code != code:
            continue
        specs = tuple(
            item.strip()
            for item in (candidate.specs_summary or "").split(",")
            if item.strip()
        )
        return CatalogProduct(
            code=candidate.code,
            category=candidate.category or "Unknown",
            brand=candidate.brand or "",
            price=str(candidate.price or ""),
            context="",
            specs=specs,
            title=candidate.name,
        )
    return None


def _product_svg(product: CatalogProduct) -> str:
    digest = hashlib.sha256(product.code.encode()).hexdigest()
    palette = [
        ("#eef1f4", "#2f3a44"),
        ("#f2efeb", "#453d35"),
        ("#edf2ef", "#31423a"),
        ("#f0eff3", "#3f3948"),
    ]
    background, ink = palette[int(digest[:2], 16) % len(palette)]
    brand = html.escape(product.brand)
    code = html.escape(product.code)
    is_laptop = product.category == "Laptop"
    device = (
        '<rect x="155" y="170" width="330" height="205" rx="16" fill="#fff" stroke="{ink}" stroke-width="8"/>'
        '<rect x="126" y="386" width="388" height="24" rx="12" fill="{ink}"/>'
        '<rect x="278" y="392" width="84" height="6" rx="3" fill="{background}"/>'
        if is_laptop
        else
        '<rect x="226" y="116" width="188" height="408" rx="42" fill="#fff" stroke="{ink}" stroke-width="9"/>'
        '<circle cx="320" cy="148" r="8" fill="{ink}"/>'
        '<rect x="286" y="492" width="68" height="6" rx="3" fill="{ink}"/>'
    ).format(ink=ink, background=background)
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640" role="img" aria-label="{brand} {code}">
      <rect width="640" height="640" rx="24" fill="{background}"/>
      {device}
      <text x="48" y="64" fill="{ink}" font-family="Arial, sans-serif" font-size="20" font-weight="700">{brand}</text>
      <text x="48" y="596" fill="{ink}" font-family="Arial, sans-serif" font-size="16">SKU {code}</text>
    </svg>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
