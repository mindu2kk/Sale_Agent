from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from llama_index.core.agent import ReActAgent

from backend.workflows.research_agent.tools import build_internal_db_tool, build_tavily_tool
from backend.workflows.research_agent.prompts import AGENT_SYSTEM_PROMPT, build_correction_context
from backend.services.advisor import CatalogAdvisor
from backend.services.catalog import CatalogService

if TYPE_CHECKING:
    from backend.verification.models.verification import VerificationResult

logger = logging.getLogger(__name__)

# Workflow status literals matching WorkflowState in verification/models/state.py
WorkflowStatus = Literal[
    "initialized",
    "researching",
    "verifying",
    "correcting",
    "approved",
    "escalated",
    "failed",
]


@dataclass
class AgentResult:
    """
    Result returned by SalesResearchAgent.run().

    Core fields (always populated):
        objection_text  – the original customer objection
        draft_response  – the generated draft answer
        tools_used      – list of tool names called during research

    Optional fields for binary verification workflow integration:
        verification_result – VerificationResult from the Verification Agent
                              (None until verification has been executed)
        workflow_status     – current status in the LangGraph WorkflowState
                              (None when used outside the workflow)
        retry_count         – how many correction retries have been attempted
        correction_feedback – structured correction instructions injected on retry
    """

    # --- core fields (required) ---
    objection_text: str
    draft_response: str
    tools_used: list[str] = field(default_factory=list)

    # --- binary verification workflow integration (optional) ---
    verification_result: Optional["VerificationResult"] = field(default=None)
    workflow_status: Optional[WorkflowStatus] = field(default=None)
    retry_count: int = field(default=0)
    correction_feedback: Optional[str] = field(default=None)


class SalesResearchAgent:
    def __init__(self, llm, rag_pipeline, tavily_api_key: str | None = None) -> None:
        if llm is None:
            raise ValueError("llm cannot be None")
        if rag_pipeline is None:
            raise ValueError("rag_pipeline cannot be None")

        self._rag_pipeline = rag_pipeline
        tools = [build_internal_db_tool(rag_pipeline)]

        tavily_tool = build_tavily_tool(tavily_api_key)
        if tavily_tool is not None:
            tools.append(tavily_tool)

        if not tools:
            raise ValueError("tools list cannot be empty")

        self._agent = ReActAgent.from_tools(
            tools,
            llm=llm,
            verbose=False,
            max_iterations=2,
            context=AGENT_SYSTEM_PROMPT,
        )

    def run(
        self,
        objection: str,
        correction_feedback: Optional[str] = None,
        verification_issues: Optional[list] = None,
    ) -> AgentResult:
        """
        Run the Sales Research Agent for a given customer objection.

        Args:
            objection: The customer objection text to respond to.
            correction_feedback: Optional structured correction instructions from the
                Self-Correction Node when retrying after a failed verification.
                When provided, the feedback is prepended to the query so the agent
                knows exactly what to fix.
            verification_issues: Optional list of PriceIssue / PolicyIssue /
                RelevanceIssue objects with granular issue details.  Used together
                with ``correction_feedback`` to give the agent precise instructions.

        Returns:
            AgentResult with the draft response and tool usage metadata.
        """
        try:
            direct_catalog_answer = self._build_catalog_answer(objection)
            if direct_catalog_answer:
                return AgentResult(
                    objection_text=objection,
                    draft_response=direct_catalog_answer,
                    tools_used=["internal_db_search"],
                )

            grounded_context = self._build_grounded_context(objection)

            # Build the query: prepend correction context when retrying
            if correction_feedback:
                correction_ctx = build_correction_context(
                    correction_feedback=correction_feedback,
                    verification_issues=verification_issues,
                )
                query = (
                    f"{grounded_context}\n\n{correction_ctx}\n\n---\n\n"
                    f"CÂU HỎI GỐC CỦA KHÁCH HÀNG:\n{objection}"
                )
                logger.info(
                    "Running agent with correction feedback for objection: %s", objection
                )
            else:
                query = f"{grounded_context}\n\nCÂU HỎI KHÁCH HÀNG:\n{objection}"
                logger.info("Running agent for objection: %s", objection)

            response = self._agent.chat(query)
            draft_response = str(response).strip()
            if not draft_response:
                draft_response = self._build_grounded_fallback(objection)

            tools_used = ["internal_db_search"]
            tools_used.extend(s.tool_name for s in response.sources)
            tools_used = list(dict.fromkeys(tools_used))

            for source in response.sources:
                logger.info(
                    "Tool called: %s | Input: %s",
                    source.tool_name,
                    source.raw_input,
                )
                logger.info("Observation: %s", source.raw_output)

            return AgentResult(
                objection_text=objection,
                draft_response=draft_response,
                tools_used=tools_used,
            )
        except Exception as exc:
            logger.error("Agent error: %s", exc, exc_info=True)

            # Check if this is an API quota error
            exc_str = str(exc).lower()
            is_quota_error = any(keyword in exc_str for keyword in [
                "quota", "429", "resource_exhausted", "rate limit"
            ])

            if is_quota_error:
                # For quota errors, raise the exception so the workflow can handle it properly
                # instead of returning a misleading escalation message
                logger.error("API quota exhausted - raising exception for proper error handling")
                raise RuntimeError(
                    "API quota exhausted. Please wait for quota reset or add OPENAI_API_KEY to .env"
                ) from exc

            # For other errors, keep the response grounded in internal data.
            return AgentResult(
                objection_text=objection,
                draft_response=self._build_grounded_fallback(objection),
                tools_used=["internal_db_search"],
            )

    def _build_grounded_context(self, query: str) -> str:
        """Retrieve internal evidence before generation so grounding is deterministic."""
        try:
            result = self._rag_pipeline.query(query)
        except Exception as exc:
            logger.warning("Pre-retrieval failed: %s", exc)
            return (
                "DỮ LIỆU NỘI BỘ: Không thể truy xuất ở bước chuẩn bị. "
                "Hãy gọi internal_db_search trước khi trả lời."
            )

        if isinstance(result, str):
            return f"DỮ LIỆU NỘI BỘ ĐÃ TRUY XUẤT:\n{result}"

        evidence = []
        for item in result[:8]:
            metadata = item.node.metadata
            source = metadata.get("source_type", "internal")
            code = metadata.get("product_code", "")
            label = f"[{source}{f' | SKU {code}' if code else ''}]"
            evidence.append(f"{label} {item.node.text}")

        if not evidence:
            return (
                "DỮ LIỆU NỘI BỘ: Không tìm thấy bằng chứng phù hợp. "
                "Không được bịa đặt thông tin."
            )

        return (
            "DỮ LIỆU NỘI BỘ ĐÃ TRUY XUẤT (NGUỒN CHÂN LÝ, PHẢI ƯU TIÊN):\n"
            + "\n".join(evidence)
            + "\n\nHãy trả lời dựa trên dữ liệu trên. Không được nói rằng không tìm thấy "
            "nếu SKU hoặc thông tin đã xuất hiện trong bằng chứng."
        )

    def _build_grounded_fallback(self, query: str) -> str:
        """Build a concise answer directly from internal evidence."""
        try:
            result = self._rag_pipeline.query(query)
        except Exception:
            result = []
        if isinstance(result, list) and result:
            primary = result[0].node
            code = primary.metadata.get("product_code")
            source = primary.metadata.get("source_type", "internal")
            citation = f" (Nguồn: {source}{f', SKU {code}' if code else ''})"
            return f"Dạ, {primary.text}{citation}"
        return (
            "Dạ, hệ thống chưa tìm thấy dữ liệu nội bộ phù hợp cho câu hỏi này. "
            "Vui lòng cung cấp mã SKU, thương hiệu hoặc cấu hình cụ thể hơn."
        )

    def _build_catalog_answer(self, query: str) -> Optional[str]:
        if not self._should_use_catalog_answer(query):
            return None

        catalog = getattr(self._rag_pipeline, "catalog", None)
        if not isinstance(catalog, CatalogService):
            return None

        advisor = CatalogAdvisor(catalog)
        product = catalog.resolve_product(query)
        if product is not None:
            return advisor.answer(query, product).text
        if advisor.is_price_objection(query):
            return (
                "Dạ, mình hiểu bạn đang thấy giá chưa thuyết phục. "
                "Bạn cho mình mã SKU hoặc tên sản phẩm đang nói tới nhé; "
                "mình sẽ phân tích lý do định giá và so với các mẫu rẻ hơn, thay vì gợi ý ngẫu nhiên."
            )

        analysis = catalog.analyze_query(query)
        products = catalog.search(query, limit=4)
        if not products:
            return (
                "Dạ, hiện catalog nội bộ chưa có sản phẩm phù hợp với yêu cầu này. "
                "Bạn có thể gửi thêm mã SKU, mức giá hoặc loại máy để mình lọc chính xác hơn."
            )

        top_prices = [int(re.sub(r"\D", "", product.price) or "0") for product in products]
        highest_price = max(
            int(re.sub(r"\D", "", product.price) or "0")
            for product in getattr(catalog, "products", [])
        ) if getattr(catalog, "products", None) else 0

        def format_vnd(amount: int) -> str:
            return f"{amount:,}".replace(",", ".") + " VNĐ"

        exact_sku = re.search(r"\b[A-Z0-9]{8}\b", query.upper())
        lines: list[str] = []
        price_intent = analysis.price_intent
        if exact_sku:
            lines.append("Dạ, đây là thông tin chi tiết mình đọc trực tiếp từ catalog nội bộ:")
        elif price_intent and price_intent.mode == "target" and price_intent.target:
            target = price_intent.target
            nearest_gap = min(abs(price - target) for price in top_prices)
            if nearest_gap > max(1_500_000, int(target * 0.12)):
                lines.append(
                    f"Dạ, catalog hiện chưa có mẫu chạm mốc khoảng {format_vnd(target)}."
                )
                if highest_price:
                    lines.append(
                        f"Mức giá cao nhất đang có là {format_vnd(highest_price)}, nên mình gợi ý các mẫu gần nhất để bạn cân nhắc:"
                    )
            else:
                lines.append(
                    f"Dạ, trong tầm {format_vnd(target)} mình thấy các mẫu phù hợp nhất là:"
                )
        elif price_intent and price_intent.mode == "max" and price_intent.maximum:
            lines.append(
                f"Dạ, trong ngân sách dưới {format_vnd(price_intent.maximum)}, mình gợi ý các mẫu nổi bật sau:"
            )
        elif price_intent and price_intent.mode == "min" and price_intent.minimum:
            lines.append(
                f"Dạ, từ {format_vnd(price_intent.minimum)} trở lên, đây là các mẫu đáng chú ý nhất trong catalog:"
            )
        elif price_intent and price_intent.mode == "range" and price_intent.minimum and price_intent.maximum:
            lines.append(
                f"Dạ, trong khoảng {format_vnd(price_intent.minimum)} đến {format_vnd(price_intent.maximum)}, mình chọn ra các mẫu gần nhất như sau:"
            )
        else:
            lines.append("Dạ, đây là các sản phẩm phù hợp nhất mình tìm được trong catalog nội bộ:")

        for product in products[:3]:
            specs = ", ".join(product.specs[:3]) if getattr(product, "specs", None) else ""
            spec_text = f" Cấu hình nổi bật: {specs}." if specs else ""
            lines.append(
                f"- {product.brand} {'Laptop' if product.category == 'Laptop' else 'Điện thoại'} · SKU {product.code}: {product.price}.{spec_text}"
            )

        lines.append("Nếu muốn, mình có thể lọc tiếp theo đúng hãng, loại máy hoặc mức RAM/SSD.")
        return "\n".join(lines)

    def _should_use_catalog_answer(self, query: str) -> bool:
        lowered = query.casefold()
        policy_terms = ("bảo hành", "đổi trả", "chính sách", "warranty", "return", "exchange")
        if any(term in lowered for term in policy_terms):
            return False
        if re.search(r"\b[A-Z0-9]{8}\b", query.upper()):
            return True
        product_terms = (
            "giá",
            "triệu",
            "vnđ",
            "vnd",
            "ngân sách",
            "tầm",
            "khoảng",
            "tư vấn",
            "gợi ý",
            "laptop",
            "điện thoại",
            "mobile phone",
        )
        return any(term in lowered for term in product_terms)
