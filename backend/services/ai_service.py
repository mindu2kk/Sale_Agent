"""Lazy integration with the project's RAG, research agent, and verification workflow."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass

from dotenv import load_dotenv

from backend.services.advisor import CatalogAdvisor
from backend.services.catalog import CatalogProduct, CatalogService, QueryConstraints
from backend.services.conversation import ConversationPlan
from backend.services.decision_engine import (
    DecisionPacket,
    DecisionPacketVerifier,
    PriceCausalityExplainer,
    ProductComparisonEngine,
)
from backend.services.policy_service import get_policy_knowledge_base
from backend.services.value_engine import CatalogRankingEngine, ValueScoringEngine
from backend.harness.runtime import BudgetExceededError

logger = logging.getLogger(__name__)
load_dotenv()

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


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD", value.casefold().replace("đ", "d")
    )
    ascii_value = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


@dataclass
class AIAnswer:
    text: str
    status: str
    tools_used: list[str]
    sources: list[dict]
    mode: str
    verification: dict | None = None
    product_codes: list[str] | None = None
    answer_type: str = "catalog_search"
    confidence: float = 0.8
    follow_up_question: str | None = None
    decision_trace: dict | None = None


class AIService:
    def __init__(self) -> None:
        self._workflow = None
        self._phrasing_llm = None
        self._lock = threading.Lock()
        self._error: str | None = None

    @property
    def status(self) -> dict[str, str | bool | None]:
        return {
            "configured": self._external_workflow_enabled()
            and bool(os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")),
            "external_workflow_enabled": self._external_workflow_enabled(),
            "loaded": self._workflow is not None,
            "error": self._error,
        }

    @staticmethod
    def _external_workflow_enabled() -> bool:
        return os.getenv("ENABLE_EXTERNAL_AI_WORKFLOW", "false").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _patch_react_agent(self) -> None:
        import llama_index.core.agent as agent_module

        if hasattr(agent_module.ReActAgent, "from_tools"):
            return

        from llama_index.core.agent.workflow import ReActAgent as NewReActAgent

        class LegacySource:
            def __init__(self, tool_name: str) -> None:
                self.tool_name = tool_name
                self.raw_input = ""
                self.raw_output = ""

        class LegacyResponse:
            def __init__(self, output) -> None:
                self.output = output
                self.sources = [
                    LegacySource(
                        getattr(call, "tool_name", None)
                        or getattr(call, "name", None)
                        or "unknown"
                    )
                    for call in (getattr(output, "tool_calls", None) or [])
                ]

            def __str__(self) -> str:
                return str(getattr(self.output, "response", self.output))

        class CompatReActAgent:
            def __init__(self, tools, llm=None, verbose=False, max_iterations=10, context="", **_):
                self.inner = NewReActAgent(
                    tools=tools,
                    llm=llm,
                    verbose=verbose,
                    max_iterations=max_iterations,
                    context=context,
                )

            @classmethod
            def from_tools(cls, tools, **kwargs):
                return cls(tools, **kwargs)

            def chat(self, message: str):
                async def run():
                    return await self.inner.run(message)

                return LegacyResponse(asyncio.run(run()))

        agent_module.ReActAgent = CompatReActAgent

    def _build_llm(self):
        google_key = os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

        if google_key:
            from llama_index.llms.google_genai import GoogleGenAI

            return GoogleGenAI(model=model, api_key=google_key), model
        if openai_key:
            from llama_index.llms.openai import OpenAI

            return OpenAI(model=model, api_key=openai_key, temperature=0.1), model
        raise RuntimeError("No GOOGLE_API_KEY or OPENAI_API_KEY configured")

    def _get_phrasing_llm(self):
        if self._phrasing_llm is not None:
            return self._phrasing_llm
        with self._lock:
            if self._phrasing_llm is None:
                self._phrasing_llm, _ = self._build_llm()
        return self._phrasing_llm

    @staticmethod
    def _decision_phrasing_enabled() -> bool:
        return os.getenv(
            "ENABLE_LLM_DECISION_PHRASING", "false"
        ).strip().casefold() in {"1", "true", "yes", "on"}

    def _embedding_candidates(self) -> list[str]:
        configured = os.getenv("EMBED_MODEL")
        candidates = [
            configured,
            "BAAI/bge-m3",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-small",
        ]
        deduped: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _configure_embeddings(self):
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        from backend.retrieval.hybrid_retriever import HybridRetriever

        for model_name in self._embedding_candidates():
            try:
                Settings.embed_model = HuggingFaceEmbedding(model_name=model_name, device="cpu")
                retriever = HybridRetriever(
                    docstore_path="./chroma_db/docstore.json",
                    chroma_path="./chroma_db",
                    embed_model_name=model_name,
                )
                logger.info("Using embedding model: %s", model_name)
                return retriever
            except Exception as exc:
                logger.warning("Embedding model failed (%s): %s", model_name, exc)
                gc.collect()

        logger.warning("All embedding models failed, continuing with BM25-only retrieval.")
        Settings.embed_model = None
        return HybridRetriever(
            docstore_path="./chroma_db/docstore.json",
            chroma_path="./chroma_db",
            embed_model_name=self._embedding_candidates()[-1],
        )

    def _build_workflow(self):
        self._patch_react_agent()
        llm, model_name = self._build_llm()

        from llama_index.core import Settings

        from backend.workflows.research_agent.sales_research_agent import SalesResearchAgent
        from backend.services.catalog import get_catalog
        from backend.services.grounded_rag import GroundedRAGPipeline
        from backend.retrieval.relevance_checker import RelevanceChecker
        from backend.verification.agent.verification_agent import VerificationAgent
        from backend.verification.config.config import get_config
        from backend.verification.workflow.workflow import VerificationWorkflow

        Settings.llm = llm
        retriever = self._configure_embeddings()
        rag_pipeline = GroundedRAGPipeline(
            catalog=get_catalog(),
            policy_retriever=retriever,
            checker=RelevanceChecker(llm=llm),
        )

        config = get_config()
        config.llm_model_name = model_name
        config.relevance_min_coverage = 0.3

        research_agent = SalesResearchAgent(
            llm=llm,
            rag_pipeline=rag_pipeline,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )
        verification_agent = VerificationAgent(
            llm=llm,
            rag_pipeline=rag_pipeline,
            config=config,
        )
        return VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )

    def get_workflow(self):
        if self._workflow is not None:
            return self._workflow
        with self._lock:
            if self._workflow is None:
                try:
                    logger.info("Loading AI workflow and retrieval stack...")
                    self._workflow = self._build_workflow()
                    self._error = None
                except Exception as exc:
                    self._error = str(exc)
                    logger.exception("AI workflow initialization failed")
                    raise
        return self._workflow

    async def answer(
        self,
        message: str,
        catalog: CatalogService,
        matched_products: list[CatalogProduct],
        context_product_codes: list[str] | None = None,
        constraints: QueryConstraints | None = None,
        conversation_plan: ConversationPlan | None = None,
        harness_run: Any = None,
    ) -> AIAnswer:
        if harness_run:
            try:
                harness_run.enforce_budget()
            except BudgetExceededError as e:
                logger.warning("Harness budget exceeded before AI generation: %s", e)
                return self._fallback(catalog, matched_products)

        context_product_codes = context_product_codes or []
        advisor = CatalogAdvisor(catalog)
        normalized_message = _normalize_text(message)
        dialogue_act = (
            conversation_plan.dialogue_act if conversation_plan is not None else None
        )
        has_fresh_search_constraints = bool(
            dialogue_act == "catalog_search"
            and constraints is not None
            and (
                constraints.price_intent is not None
                or constraints.discrete_gpu
                or bool(constraints.brands)
                or bool(constraints.cpu_filters)
                or bool(constraints.gpu_filters)
                or constraints.use_case is not None
                or constraints.category is not None
            )
        )
        context_product = (
            matched_products[0]
            if matched_products and dialogue_act in DETAIL_INTENTS
            else (
                None
                if has_fresh_search_constraints
                else catalog.resolve_product(message, context_product_codes)
            )
        )
        is_policy_query = any(
            term in normalized_message
            for term in (
                "bao hanh",
                "doi tra",
                "hoan tien",
                "doi may",
                "chinh sach",
                "warranty",
                "return",
                "refund",
                "exchange",
                "loi nha san xuat",
                "doi trong bao lau",
                "co duoc tra",
            )
        )
        if dialogue_act == "clarify":
            question = (
                conversation_plan.clarification_question
                if conversation_plan
                else None
            ) or "Bạn có thể cho mình thêm loại máy, ngân sách hoặc mẫu đang cân nhắc không?"
            return AIAnswer(
                text=question,
                status="needs_context",
                tools_used=[],
                sources=[],
                mode="deterministic_advisor",
                verification={
                    "approved": True,
                    "critical_issues": 0,
                    "reasoning": "Low-information request was clarified before retrieval.",
                },
                product_codes=[],
                answer_type="clarify",
                confidence=conversation_plan.confidence if conversation_plan else 0.99,
                follow_up_question=question,
                decision_trace={
                    "intent": "clarify",
                    "intent_reason": conversation_plan.reason if conversation_plan else "",
                    "state_delta": conversation_plan.state_delta if conversation_plan else {},
                    "candidate_codes": [],
                },
            )
        if dialogue_act == "policy" or is_policy_query:
            policy_kb = get_policy_knowledge_base()
            policy_answer = policy_kb.answer(
                message,
                brand=context_product.brand if context_product else None,
            )
            return AIAnswer(
                text=policy_answer.text,
                status="approved" if policy_answer.sources else "needs_context",
                tools_used=["local_policy_search"],
                sources=list(policy_answer.sources),
                mode="deterministic_policy",
                verification={
                    "approved": bool(policy_answer.sources),
                    "critical_issues": 0 if policy_answer.sources else 1,
                    "reasoning": "Extractive answer from local policy PDFs; no external LLM used.",
                },
                product_codes=[context_product.code] if context_product else [],
                answer_type="policy",
                confidence=conversation_plan.confidence if conversation_plan else 0.95,
            )

        if dialogue_act == "price_causality":
            advisory, packet = PriceCausalityExplainer().explain(matched_products)
            return await self._decision_answer(
                advisory,
                packet.answer_type,
                conversation_plan,
                catalog,
                tool="price_causality_explainer",
                packet=packet,
                recommendation_code=packet.recommendation_code,
                scores=[],
            )

        if dialogue_act in {"brand_comparison", "refine_preferences"}:
            engine = ProductComparisonEngine()
            advisory, packet = engine.compare(
                matched_products,
                preferences=(
                    conversation_plan.preferences if conversation_plan else {}
                ),
                use_case=(
                    conversation_plan.use_case if conversation_plan else None
                ),
                budget=(
                    conversation_plan.price_intent.target
                    if conversation_plan
                    and conversation_plan.price_intent
                    and conversation_plan.price_intent.mode == "target"
                    else None
                ),
            )
            return await self._decision_answer(
                advisory,
                packet.answer_type,
                conversation_plan,
                catalog,
                tool="preference_reranker",
                packet=packet,
                recommendation_code=packet.recommendation_code,
                scores=[
                    {
                        "product_code": score.product.code,
                        "score": score.score,
                        "confidence": score.confidence,
                        "reasons": list(score.reasons),
                        "cautions": list(score.cautions),
                    }
                    for score in packet.scores
                ],
            )

        if dialogue_act == "reject_candidate":
            return self._replacement_answer(
                catalog,
                matched_products,
                rejected_product=context_product,
                constraints=constraints,
            )

        if (
            context_product is not None
            and not has_fresh_search_constraints
            and (
            dialogue_act in DETAIL_INTENTS
            or
            re.search(r"\b[A-Z0-9]{8}\b", message.upper())
            or advisor.is_price_objection(message)
            or advisor.is_detail_request(message)
            or advisor.is_alternative_request(message)
        )):
            fact_answer = advisor.answer_missing_field_or_known_fact(
                message,
                context_product,
            )
            if fact_answer is not None:
                return AIAnswer(
                    text=fact_answer.text,
                    status="approved",
                    tools_used=["catalog_advisor"],
                    sources=[
                        {
                            "source": "product_catalog",
                            "product_code": context_product.code,
                            "source_url": context_product.source_url,
                            "fetched_at": context_product.fetched_at,
                            "price_valid_until": context_product.price_valid_until,
                        }
                    ],
                    mode="deterministic_advisor",
                    verification={
                        "approved": True,
                        "critical_issues": 0,
                        "reasoning": "Focused product fact answer stayed inside catalog evidence only.",
                    },
                    product_codes=[context_product.code],
                    answer_type="product_detail",
                    confidence=conversation_plan.confidence if conversation_plan else 0.95,
                )
            advisory_intent = (
                {
                    "product_detail": "detail",
                    "select_previous_candidate": "detail",
                    "product_detail_followup": "detail",
                    "exact_product_detail": "detail",
                    "product_correction": "detail",
                    "focused_product_analysis": "detail",
                    "price_objection": "price_objection",
                    "cheaper_alternatives": "alternatives",
                }.get(dialogue_act)
                or advisor.classify_intent(message)
            )
            advisory = advisor.answer(
                message,
                context_product,
                force_price_analysis=advisory_intent == "price_objection",
                force_intent=advisory_intent,
            )
            return AIAnswer(
                text=advisory.text,
                status="approved",
                tools_used=["catalog_advisor"],
                sources=[
                    {
                        "source": "product_catalog",
                        "product_code": code,
                        "source_url": item.source_url if item else "",
                        "fetched_at": item.fetched_at if item else "",
                        "price_valid_until": item.price_valid_until if item else "",
                    }
                    for code in advisory.product_codes
                    for item in [catalog.get(code)]
                ],
                mode="deterministic_advisor",
                verification={
                    "approved": True,
                    "critical_issues": 0,
                    "reasoning": {
                        "detail": "Deterministic exact-SKU product consultation.",
                        "price_objection": "Deterministic price-position and value explanation.",
                        "alternatives": "Deterministic comparable-product ranking.",
                    }[advisory_intent],
                },
                product_codes=list(advisory.product_codes),
                answer_type=(
                    "product_detail"
                    if advisory_intent == "detail"
                    else (dialogue_act or advisory_intent)
                ),
                confidence=conversation_plan.confidence if conversation_plan else 0.95,
            )

        if dialogue_act == "price_objection" or advisor.is_price_objection(message):
            return AIAnswer(
                text=(
                    "Dạ, mình hiểu bạn đang thấy mức giá chưa thuyết phục. "
                    "Bạn cho mình mã SKU hoặc tên sản phẩm đang nói tới nhé; "
                    "mình sẽ phân tích điểm đáng tiền, điểm chưa hợp lý và đưa mẫu rẻ hơn để so trực tiếp."
                ),
                status="needs_context",
                tools_used=[],
                sources=[],
                mode="deterministic_advisor",
                verification={
                    "approved": True,
                    "critical_issues": 0,
                    "reasoning": "Ambiguous price objection requires the referenced product.",
                },
                product_codes=[],
                answer_type="clarify",
                confidence=0.98,
                follow_up_question="Bạn đang nói tới SKU hoặc mẫu máy nào?",
            )

        if (
            constraints is not None
            and constraints.comparison
            and len(matched_products) >= 2
        ):
            advisory = advisor.compare_products(
                matched_products,
                reference_price=(
                    constraints.price_intent.target
                    if constraints.price_intent is not None
                    else None
                ),
            )
            return AIAnswer(
                text=advisory.text,
                status="approved",
                tools_used=["catalog_comparison"],
                sources=[
                    {
                        "source": "product_catalog",
                        "product_code": code,
                        "source_url": item.source_url if item else "",
                    }
                    for code in advisory.product_codes
                    for item in [catalog.get(code)]
                ],
                mode="deterministic_advisor",
                verification={
                    "approved": True,
                    "critical_issues": 0,
                    "reasoning": "Deterministic same-budget product comparison.",
                },
                product_codes=list(advisory.product_codes),
                answer_type="comparison",
                confidence=0.9,
            )

        if (
            constraints is not None
            and constraints.goal == "performance_per_price"
        ):
            engine = ValueScoringEngine()
            profile = constraints.use_case or "overall"
            rankings = engine.rank(matched_products, profile=profile)
            office_rankings = engine.rank(matched_products, profile="office")
            advisory = advisor.performance_value_answer(
                rankings,
                budget=(
                    constraints.price_intent.target
                    if constraints.price_intent is not None
                    else None
                ),
                profile=constraints.use_case,
                office_alternative=office_rankings[0] if office_rankings else None,
            )
            return AIAnswer(
                text=advisory.text,
                status="approved",
                tools_used=["catalog_value_ranking"],
                sources=[
                    {
                        "source": "product_catalog",
                        "product_code": code,
                        "source_url": item.source_url if item else "",
                    }
                    for code in advisory.product_codes
                    for item in [catalog.get(code)]
                ],
                mode="deterministic_advisor",
                verification={
                    "approved": True,
                    "critical_issues": 0,
                    "reasoning": (
                        "Deterministic performance-per-price ranking with "
                        "local normalized component tiers and confidence guard."
                    ),
                },
                product_codes=list(advisory.product_codes),
                answer_type="value_ranking",
                confidence=conversation_plan.confidence if conversation_plan else 0.9,
            )

        if dialogue_act == "catalog_ranking":
            goal = conversation_plan.goal or "best_overall"
            rankings = CatalogRankingEngine().rank(
                matched_products,
                mode=goal,
                use_case=constraints.use_case if constraints else None,
                limit=5,
            )
            advisory = advisor.catalog_ranking_answer(
                rankings,
                goal=goal,
                use_case=constraints.use_case if constraints else None,
            )
            score_margin = (
                rankings[0].score - rankings[1].score
                if len(rankings) > 1
                else 100.0
            )
            decisive = bool(
                rankings
                and (
                    goal in {"lowest_price", "highest_price"}
                    or (
                        rankings[0].confidence >= 0.7
                        and score_margin >= 4.0
                    )
                )
            )
            winner_code = rankings[0].product.code if decisive else None
            return AIAnswer(
                text=advisory.text,
                status="approved" if rankings else "needs_context",
                tools_used=["catalog_capability_ranking"],
                sources=[
                    {
                        "source": "product_catalog",
                        "product_code": code,
                        "source_url": item.source_url if item else "",
                        "fetched_at": item.fetched_at if item else "",
                    }
                    for code in advisory.product_codes
                    for item in [catalog.get(code)]
                ],
                mode="deterministic_advisor",
                verification={
                    "approved": bool(rankings),
                    "critical_issues": 0 if rankings else 1,
                    "reasoning": (
                        "Catalog-wide deterministic capability ranking with "
                        "versioned component tiers and evidence completeness."
                    ),
                },
                product_codes=list(advisory.product_codes),
                answer_type="catalog_ranking",
                confidence=rankings[0].confidence if rankings else 0.0,
                decision_trace={
                    "intent": "catalog_ranking",
                    "ranking_goal": goal,
                    "recommendation_code": winner_code,
                    "score_margin": round(score_margin, 2),
                    "abstained": not decisive,
                    "benchmark_version": "2026-06-internal-relative-v1",
                    "scores": [
                        {
                            "product_code": item.product.code,
                            "score": item.score,
                            "confidence": item.confidence,
                            "reasons": list(item.reasons),
                            "cautions": list(item.cautions),
                        }
                        for item in rankings
                    ],
                },
            )

        if dialogue_act == "general_explanation":
            return await self._general_explanation_answer(message)

        if self._should_use_direct_catalog(message) or (
            constraints is not None
            and (
                constraints.category is not None
                or constraints.price_intent is not None
                or constraints.discrete_gpu
            )
        ):
            return self._direct_catalog_answer(
                message,
                catalog,
                matched_products,
                constraints=constraints,
            )

        if not self._external_workflow_enabled():
            logger.info(
                "External AI workflow is disabled; using the fast grounded fallback."
            )
            return self._fallback(catalog, matched_products)

        try:
            if harness_run:
                harness_run.record("execution", "Delegating to LLM verification workflow.", data={"mode": "verification_workflow"})
                harness_run.enforce_budget()

            async def execute_external_workflow():
                workflow = await asyncio.to_thread(self.get_workflow)
                return await workflow.execute_workflow(
                    message,
                    customer_context={
                        "matched_product_codes": [
                            product.code for product in matched_products
                        ],
                    },
                )

            state = await asyncio.wait_for(
                execute_external_workflow(),
                timeout=float(os.getenv("AI_WORKFLOW_TIMEOUT_SECONDS", "6")),
            )
            workflow_status = state.get("workflow_status", "unknown")
            text = (
                state.get("final_response") or state.get("draft_response") or ""
                if workflow_status == "approved"
                else state.get("draft_response") or state.get("final_response") or ""
            )
            text = re.sub(r"^\s*assistant:\s*", "", text, flags=re.IGNORECASE)
            if not text:
                raise RuntimeError("AI workflow returned an empty response")

            verification_result = state.get("verification_result")
            verification = None
            if verification_result is not None:
                criteria = getattr(verification_result, "criteria", None)
                verification = {
                    "approved": bool(getattr(verification_result, "is_approved", False)),
                    "critical_issues": int(getattr(criteria, "critical_issues_count", 0) or 0),
                    "reasoning": str(
                        getattr(verification_result, "verification_reasoning", "")
                    )[:1200],
                }

            return AIAnswer(
                text=text,
                status=workflow_status,
                tools_used=list(state.get("tools_used", [])),
                sources=list(state.get("research_sources", [])),
                mode="verification_workflow",
                verification=verification,
                product_codes=[product.code for product in matched_products],
                answer_type=dialogue_act or "open_question",
                confidence=conversation_plan.confidence if conversation_plan else 0.7,
            )
        except BudgetExceededError as exc:
            logger.warning("Harness budget exceeded during LLM workflow: %s", exc)
            if harness_run:
                harness_run.record("recovery", "Budget exceeded; forced deterministic fallback.", status="recovered")
            return self._fallback(catalog, matched_products)
        except Exception as exc:
            logger.warning("Using catalog-grounded fallback: %s", exc)
            return self._fallback(catalog, matched_products)

    async def _general_explanation_answer(self, query: str) -> AIAnswer:
        prompt = (
            "Bạn là chuyên viên tư vấn bán hàng điện tử. Khách hàng hỏi câu hỏi kiến thức chung:\n"
            f'"{query}"\n\n'
            "Quy tắc:\n"
            "1. Chỉ giải thích trong phạm vi máy tính, điện thoại, linh kiện điện tử.\n"
            "2. Trả lời tự nhiên, thân thiện, súc tích (100-200 từ).\n"
            "3. Nếu câu hỏi ngoài lề (y tế, chính trị, v.v.), hãy từ chối khéo léo.\n"
            "Chỉ trả về trực tiếp nội dung câu trả lời, không cần mào đầu, không dùng định dạng JSON."
        )
        try:
            llm = await asyncio.to_thread(self._get_phrasing_llm)
            response = await asyncio.wait_for(
                llm.acomplete(prompt),
                timeout=float(os.getenv("LLM_WORKFLOW_TIMEOUT_SECONDS", "10")),
            )
            text = str(response).strip()
            if not text:
                text = "Dạ, em hiểu câu hỏi của anh chị. Tuy nhiên, hiện tại em chỉ hỗ trợ giải đáp về các sản phẩm công nghệ ạ."
        except Exception as e:
            logger.error(f"Error in _general_explanation_answer: {e}")
            text = "Dạ, em hiểu câu hỏi của anh chị. Tuy nhiên, hiện tại em chỉ hỗ trợ giải đáp về các sản phẩm công nghệ ạ."
        return AIAnswer(
            text=text,
            status="approved",
            tools_used=["general_knowledge_llm"],
            sources=[],
            mode="general_knowledge",
            product_codes=[],
            answer_type="general_explanation",
            confidence=0.9,
            verification={"approved": True, "critical_issues": 0, "reasoning": "General domain knowledge query."}
        )

    async def _decision_answer(
        self,
        advisory,
        answer_type: str,
        plan: ConversationPlan | None,
        catalog: CatalogService,
        *,
        tool: str,
        packet: DecisionPacket,
        recommendation_code: str | None,
        scores: list[dict],
    ) -> AIAnswer:
        packet_verification = DecisionPacketVerifier().verify(packet, advisory)
        if not packet_verification.approved:
            return AIAnswer(
                text=(
                    "Mình chưa thể đưa ra kết luận an toàn vì dữ liệu so sánh vừa tạo "
                    "không vượt qua bước kiểm tra tính nhất quán. Bạn hãy thử lại với "
                    "đúng hai SKU cần so sánh."
                ),
                status="needs_context",
                tools_used=[tool, "decision_packet_verifier"],
                sources=[],
                mode="deterministic_advisor",
                verification={
                    "approved": False,
                    "critical_issues": len(packet_verification.issues),
                    "reasoning": list(packet_verification.issues),
                },
                product_codes=[],
                answer_type="clarify",
                confidence=1.0,
                follow_up_question="Bạn gửi giúp mình đúng hai SKU cần so sánh nhé.",
                decision_trace={
                    "intent": plan.dialogue_act if plan else answer_type,
                    "packet_verification": {
                        "approved": False,
                        "issues": list(packet_verification.issues),
                    },
                },
            )
        phrased_advisory = advisory
        phrasing_mode = "deterministic"
        phrasing_issue: str | None = None
        if self._decision_phrasing_enabled():
            try:
                phrased_advisory = await self._phrase_locked_decision(
                    advisory,
                    packet,
                )
                phrased_verification = DecisionPacketVerifier().verify(
                    packet,
                    phrased_advisory,
                    baseline_text=advisory.text,
                )
                if not phrased_verification.approved:
                    phrasing_issue = "; ".join(phrased_verification.issues)
                    phrased_advisory = advisory
                else:
                    phrasing_mode = "llm_grounded_rewrite"
            except Exception as exc:
                phrasing_issue = str(exc)
                phrased_advisory = advisory
        return AIAnswer(
            text=phrased_advisory.text,
            status="approved",
            tools_used=[
                tool,
                "decision_packet_verifier",
                *(
                    ["grounded_llm_phraser"]
                    if phrasing_mode == "llm_grounded_rewrite"
                    else []
                ),
            ],
            sources=[
                {
                    "source": "product_catalog",
                    "product_code": code,
                    "source_url": item.source_url if item else "",
                    "fetched_at": item.fetched_at if item else "",
                }
                for code in advisory.product_codes
                for item in [catalog.get(code)]
            ],
            mode="deterministic_advisor",
            verification={
                "approved": True,
                "critical_issues": 0,
                "reasoning": (
                    "DecisionPacket passed deterministic candidate, SKU, price, "
                    "and recommendation consistency checks."
                ),
            },
            product_codes=list(advisory.product_codes),
            answer_type=answer_type,
            confidence=plan.confidence if plan else 0.9,
            decision_trace={
                "intent": plan.dialogue_act if plan else answer_type,
                "intent_reason": plan.reason if plan else "",
                "state_delta": plan.state_delta if plan else {},
                "candidate_codes": list(advisory.product_codes),
                "recommendation_code": recommendation_code,
                "scores": scores,
                "decision_confidence": packet.decision_confidence,
                "abstained": packet.abstained,
                "packet_verification": {
                    "approved": True,
                    "issues": [],
                },
                "phrasing": {
                    "mode": phrasing_mode,
                    "fallback_reason": phrasing_issue,
                },
            },
        )

    async def _phrase_locked_decision(
        self,
        advisory,
        packet: DecisionPacket,
    ):
        allowed = [
            {
                "code": product.code,
                "name": product.name,
                "brand": product.brand,
                "price": product.price,
                "specs": list(product.specs),
            }
            for product in packet.products
        ]
        prompt = (
            "Bạn là chuyên viên tư vấn bán hàng. Chỉ viết lại câu trả lời cho tự nhiên, "
            "thân thiện và dễ quyết định hơn. Không thêm, bỏ, suy diễn hoặc thay đổi SKU, "
            "giá, số liệu, sản phẩm, kết luận hay cảnh báo. Phải nhắc nguyên văn SKU được "
            "khuyến nghị. Trả về JSON duy nhất dạng {\"text\":\"...\"}.\n\n"
            f"LOCKED_PRODUCTS={json.dumps(allowed, ensure_ascii=False)}\n"
            f"LOCKED_RECOMMENDATION={packet.recommendation_code}\n"
            f"LOCKED_FACTS={json.dumps(packet.facts, ensure_ascii=False)}\n"
            f"LOCKED_WARNINGS={json.dumps(packet.warnings, ensure_ascii=False)}\n"
            f"BASELINE_ANSWER={advisory.text}"
        )
        llm = await asyncio.to_thread(self._get_phrasing_llm)
        response = await asyncio.wait_for(
            llm.acomplete(prompt),
            timeout=float(os.getenv("LLM_PHRASING_TIMEOUT_SECONDS", "4")),
        )
        raw = str(response).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        payload = json.loads(raw)
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM phrasing returned invalid JSON text")
        return type(advisory)(
            text=text.strip(),
            product_codes=advisory.product_codes,
        )

    def _fallback(
        self,
        catalog: CatalogService,
        products: list[CatalogProduct],
    ) -> AIAnswer:
        del catalog
        if len(products) == 1:
            product = products[0]
            specs = list(product.specs)
            lines = [
                (
                    "Mình chưa phân tích sâu bằng AI được, nhưng dựa trên dữ liệu catalog hiện có, "
                    f"{product.name} ({product.code}) có giá {product.price}."
                )
            ]
            detail_specs = []
            for label in (
                "CPU",
                "Card đồ hoạ",
                "GPU",
                "RAM",
                "Ổ cứng SSD",
                "Kích thước màn hình",
                "Độ phân giải",
            ):
                detail_specs.extend(
                    spec for spec in specs if spec.lower().startswith(label.lower())
                )
            if detail_specs:
                lines.append("")
                lines.extend(f"- {spec}" for spec in dict.fromkeys(detail_specs))
            lines.append("")
            if any(token in " ".join(specs).lower() for token in ("rtx", "gtx", "radeon", "arc graphics")):
                lines.append("Mẫu này hợp hơn nếu anh/chị cần hiệu năng tốt hơn cho đồ họa hoặc game tầm vừa.")
            else:
                lines.append("Mẫu này hợp hơn với học tập, văn phòng, làm việc cơ bản, họp online và di chuyển nhẹ.")
                lines.append("Nếu anh/chị cần chơi game hoặc đồ họa nặng thì GPU tích hợp sẽ là điểm cần cân nhắc.")
            return AIAnswer(
                text="\n".join(lines),
                status="degraded",
                tools_used=["catalog_search"],
                sources=[{"source": "product_catalog", "product_code": product.code}],
                mode="catalog_fallback",
                verification=None,
                product_codes=[product.code],
                answer_type="product_detail",
            )
        if not products:
            return AIAnswer(
                text=(
                    "Dạ, hệ thống chưa tìm thấy sản phẩm phù hợp trong catalog nội bộ cho câu hỏi này. "
                    "Bạn có thể cung cấp thêm mã SKU, mức giá hoặc loại máy mong muốn."
                ),
                status="degraded",
                tools_used=["catalog_search"],
                sources=[],
                mode="catalog_fallback",
                verification=None,
                product_codes=[],
            )

        lines = [
            "Mình chưa phân tích sâu bằng AI được, nhưng dựa trên dữ liệu catalog hiện có, mình thấy các mẫu sau đáng cân nhắc:",
        ]
        for product in products[:3]:
            specs = ", ".join(product.specs[:3])
            detail = f" - {specs}" if specs else ""
            lines.append(f"- {product.name}: {product.price}{detail}.")
        lines.append("Thông tin trên được lấy trực tiếp từ product_catalog_clean.csv.")

        return AIAnswer(
            text="\n".join(lines),
            status="degraded",
            tools_used=["catalog_search"],
            sources=[
                {"source": "product_catalog", "product_code": product.code}
                for product in products[:3]
            ],
            mode="catalog_fallback",
            verification=None,
            product_codes=[product.code for product in products[:3]],
        )

    def _should_use_direct_catalog(self, message: str) -> bool:
        normalized = _normalize_text(message)
        if any(
            term in normalized
            for term in ("bao hanh", "doi tra", "chinh sach", "warranty", "return", "exchange")
        ):
            return False
        if re.search(r"\b[A-Z0-9]{8}\b", message.upper()):
            return True
        from backend.services.catalog import _parse_price_intent

        if _parse_price_intent(message) is not None:
            return True
        return any(
            token in normalized
            for token in (
                "ngan sach",
                "budget",
                "tam",
                "khoang",
                "tu van",
                "goi y",
                "tim laptop",
                "tim dien thoai",
                "laptop",
                "dien thoai",
            )
        )

    def _direct_catalog_answer(
        self,
        message: str,
        catalog: CatalogService,
        products: list[CatalogProduct],
        constraints: QueryConstraints | None = None,
    ) -> AIAnswer:
        if not products:
            return AIAnswer(
                text=(
                    "Dạ, hiện catalog nội bộ chưa có mẫu phù hợp với yêu cầu này. "
                    "Bạn có thể gửi thêm mã SKU, mức giá hoặc loại máy để mình lọc sát hơn."
                ),
                status="approved",
                tools_used=["catalog_search"],
                sources=[],
                mode="catalog_fallback",
                verification={
                    "approved": True,
                    "critical_issues": 0,
                    "reasoning": "Direct catalog lookup with no matching products.",
                },
                product_codes=[],
            )

        analysis = catalog.analyze_query(message)
        price_intent = (
            constraints.price_intent
            if constraints is not None and constraints.price_intent is not None
            else analysis.price_intent
        )
        highest_price = max((int(re.sub(r"\D", "", item.price) or "0") for item in catalog.products), default=0)

        def format_vnd(amount: int) -> str:
            return f"{amount:,}".replace(",", ".") + " VNĐ"

        exact_sku = re.search(r"\b[A-Z0-9]{8}\b", message.upper())
        lines: list[str] = []
        if exact_sku:
            lines.append("Dạ, đây là thông tin mình đọc trực tiếp từ catalog:")
        elif (
            constraints is not None
            and constraints.discrete_gpu
            and price_intent
            and price_intent.target
        ):
            lines.append(
                f"Dạ, trong tầm {format_vnd(price_intent.target)}, đây là các laptop có card đồ họa rời gần nhất:"
            )
        elif price_intent and price_intent.target:
            target = price_intent.target
            nearest_gap = min(
                abs(int(re.sub(r"\D", "", item.price) or "0") - target)
                for item in products
            )
            if nearest_gap > max(1_500_000, int(target * 0.12)):
                lines.append(f"Dạ, hiện catalog chưa có mẫu ở mốc khoảng {format_vnd(target)}.")
                if highest_price:
                    lines.append(
                        f"Mức giá cao nhất đang có là {format_vnd(highest_price)}, nên đây là các mẫu gần nhất để bạn tham khảo:"
                    )
            else:
                lines.append(f"Dạ, trong tầm {format_vnd(target)}, đây là các mẫu gần nhất:")
        elif (
            price_intent
            and price_intent.mode == "range"
            and price_intent.minimum
            and price_intent.maximum
        ):
            lines.append(
                f"Dạ, trong khoảng {format_vnd(price_intent.minimum)} đến "
                f"{format_vnd(price_intent.maximum)}, đây là các mẫu phù hợp:"
            )
        elif price_intent and price_intent.maximum:
            lines.append(
                f"Dạ, trong ngân sách dưới {format_vnd(price_intent.maximum)}, mình gợi ý các mẫu sau:"
            )
        elif price_intent and price_intent.minimum:
            lines.append(
                f"Dạ, từ {format_vnd(price_intent.minimum)} trở lên, đây là các mẫu đáng chú ý nhất:"
            )
        elif constraints is not None and constraints.discrete_gpu:
            lines.append(
                "Dạ, có. Đây là các laptop có card đồ họa rời phù hợp với ngữ cảnh bạn đang tìm:"
            )
        else:
            lines.append("Dạ, đây là các sản phẩm phù hợp nhất mình tìm được:")

        for product in products[:3]:
            specs = ", ".join(product.specs[:3])
            detail = f" Cấu hình nổi bật: {specs}." if specs else ""
            lines.append(f"- {product.name} — {product.price}; SKU {product.code}.{detail}")

        lines.append("Mình có thể lọc tiếp theo đúng hãng, RAM, SSD hoặc loại máy bạn muốn.")
        return AIAnswer(
            text="\n".join(lines),
            status="approved",
            tools_used=["catalog_search"],
            sources=[
                {"source": "product_catalog", "product_code": product.code}
                for product in products[:3]
            ],
            mode="catalog_fallback",
            verification={
                "approved": True,
                "critical_issues": 0,
                "reasoning": "Direct catalog response for structured product lookup.",
            },
            product_codes=[product.code for product in products[:3]],
        )

    def _replacement_answer(
        self,
        catalog: CatalogService,
        products: list[CatalogProduct],
        *,
        rejected_product: CatalogProduct | None,
        constraints: QueryConstraints | None,
    ) -> AIAnswer:
        rejected_name = (
            f"**{rejected_product.name}** (SKU {rejected_product.code})"
            if rejected_product
            else "mẫu vừa rồi"
        )
        if not products:
            return AIAnswer(
                text=(
                    f"Được, mình đã loại {rejected_name} khỏi danh sách.\n\n"
                    "Hiện catalog chưa có mẫu thay thế đủ sát các điều kiện đang giữ. "
                    "Bạn muốn nới ngân sách, đổi hãng hay ưu tiên lại hiệu năng/pin/màn hình?"
                ),
                status="needs_context",
                tools_used=["catalog_search"],
                sources=[],
                mode="catalog_fallback",
                verification={
                    "approved": True,
                    "critical_issues": 0,
                    "reasoning": "Rejected product excluded; no safe replacement found.",
                },
                product_codes=[],
                answer_type="replacement_search",
                confidence=0.98,
                follow_up_question=(
                    "Bạn muốn nới ngân sách, đổi hãng hay ưu tiên lại tiêu chí nào?"
                ),
            )

        budget = constraints.price_intent.target if constraints and constraints.price_intent else None
        lines = [
            f"Được, mình đã loại {rejected_name} khỏi danh sách.",
            "",
            (
                f"Giữ ngân sách quanh **{self._format_vnd(budget)}**, "
                "mình thấy ba hướng thay thế đáng cân nhắc:"
                if budget
                else "Mình thấy ba hướng thay thế đáng cân nhắc:"
            ),
        ]
        for index, product in enumerate(products[:3], start=1):
            specs = list(product.specs[:3])
            strength = specs[0] if specs else "cấu hình phù hợp nhu cầu phổ thông"
            supporting = f"; {specs[1]}" if len(specs) > 1 else ""
            fit = (
                "phù hợp nếu bạn muốn ưu tiên hiệu năng"
                if any(key in strength.casefold() for key in ("cpu", "chip", "card", "gpu"))
                else "phù hợp nếu bạn muốn một lựa chọn cân bằng, dễ chốt"
            )
            lines.extend(
                [
                    "",
                    f"**{index}. {product.name}** — {product.price}",
                    f"- {strength}{supporting}.",
                    f"- SKU `{product.code}`; {fit}.",
                ]
            )
        lines.extend(
            [
                "",
                "Nếu phải chốt nhanh, mình sẽ so hai mẫu đầu theo đúng nhu cầu sử dụng "
                "của bạn thay vì chỉ chọn mẫu có giá gần nhất.",
            ]
        )
        return AIAnswer(
            text="\n".join(lines),
            status="approved",
            tools_used=["catalog_search", "replacement_formatter"],
            sources=[
                {
                    "source": "product_catalog",
                    "product_code": product.code,
                    "source_url": product.source_url,
                }
                for product in products[:3]
            ],
            mode="deterministic_advisor",
            verification={
                "approved": True,
                "critical_issues": 0,
                "reasoning": "Rejected SKU excluded and replacements formatted by use-case.",
            },
            product_codes=[product.code for product in products[:3]],
            answer_type="replacement_search",
            confidence=0.96,
            follow_up_question="Bạn dùng máy chủ yếu cho công việc gì để mình chốt hai mẫu đầu?",
        )

    @staticmethod
    def _format_vnd(amount: int) -> str:
        return f"{amount:,}".replace(",", ".") + " VNĐ"


_ai_service = AIService()


def get_ai_service() -> AIService:
    return _ai_service
