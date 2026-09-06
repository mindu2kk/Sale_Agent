"""
RelevanceChecker: Classifies incoming queries before triggering the RAG pipeline.

Uses an LLM to categorize a query as CAN_ANSWER, PARTIAL, or NO_MATCH,
allowing the system to skip retrieval for irrelevant questions and reduce API costs.
"""

from typing import Literal

RelevanceLabel = Literal["CAN_ANSWER", "PARTIAL", "NO_MATCH"]

_SYSTEM_PROMPT_TEMPLATE = """Bạn là bộ phân loại câu hỏi cho hệ thống tư vấn bán hàng điện tử.
Phân loại câu hỏi sau vào đúng một trong ba nhãn:
- CAN_ANSWER: câu hỏi liên quan trực tiếp đến sản phẩm, mã SKU, giá, hoặc chính sách bảo hành/đổi trả
- PARTIAL: câu hỏi liên quan một phần (thương hiệu chung, câu hỏi mơ hồ có thể liên quan)
- NO_MATCH: câu hỏi hoàn toàn không liên quan đến sản phẩm hoặc chính sách

Chỉ trả về đúng một nhãn, không giải thích thêm.

Câu hỏi: {query}
Nhãn:"""


class RelevanceChecker:
    """Classifies a user query into one of three relevance labels using an LLM.

    Labels:
        - CAN_ANSWER: query is directly related to products, SKUs, prices, or policies.
        - PARTIAL: query is partially related (e.g., general brand questions).
        - NO_MATCH: query is completely unrelated to products or policies.

    Args:
        llm: Any LlamaIndex-compatible LLM instance (e.g., OpenAI, Ollama).
    """

    def __init__(self, llm) -> None:
        """Initialize the RelevanceChecker with a LlamaIndex-compatible LLM.

        Args:
            llm: A LlamaIndex-compatible LLM instance used to classify queries.
        """
        self.llm = llm
        self._prompt_template = _SYSTEM_PROMPT_TEMPLATE

    def check(self, query: str) -> RelevanceLabel:
        """Classify a query string into a relevance label.

        Sends the query to the LLM using a fixed system prompt, parses the
        output, and returns one of the three valid labels. If the LLM returns
        an unrecognized value, defaults to PARTIAL.

        Args:
            query: The user's input query string.

        Returns:
            One of "CAN_ANSWER", "PARTIAL", or "NO_MATCH".

        Raises:
            Exception: Propagates any exception raised by the LLM client
                       (e.g., timeout, network error) without swallowing it.
        """
        if query.strip() == "":
            return "NO_MATCH"

        prompt = self._prompt_template.format(query=query)
        response = self.llm.complete(prompt)
        raw = response.text.strip().rstrip(".!?").upper()

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if raw in valid_labels:
            return raw  # type: ignore[return-value]
        return "PARTIAL"
