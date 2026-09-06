"""
RAGPipeline: Integrates RelevanceChecker and HybridRetriever into a single query pipeline.

Routes queries based on relevance classification:
- NO_MATCH  → returns a default response without calling the retriever
- CAN_ANSWER / PARTIAL → calls HybridRetriever and returns retrieved nodes
"""

from __future__ import annotations

import logging
from typing import Union

from llama_index.core.schema import NodeWithScore

from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.relevance_checker import RelevanceChecker

logger = logging.getLogger(__name__)

_DEFAULT_NO_MATCH_RESPONSE = (
    "Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi hỗ trợ của hệ thống. "
    "Vui lòng hỏi về sản phẩm hoặc chính sách bảo hành/đổi trả."
)


class RAGPipeline:
    """Orchestrates relevance checking and hybrid retrieval for user queries.

    Args:
        retriever: A HybridRetriever instance used to fetch relevant nodes.
        checker: A RelevanceChecker instance used to classify the query.
    """

    def __init__(self, retriever: HybridRetriever, checker: RelevanceChecker) -> None:
        self.retriever = retriever
        self.checker = checker

    def query(self, user_query: str) -> Union[str, list[NodeWithScore]]:
        """Classify the query and route to retrieval or default response.

        1. Calls checker.check() to get a relevance label.
        2. Logs the label.
        3. If NO_MATCH → returns the default response string.
        4. If CAN_ANSWER or PARTIAL → calls retriever.retrieve() and returns nodes.

        Args:
            user_query: The user's input query string.

        Returns:
            A default response string for NO_MATCH queries, or a list of
            NodeWithScore for CAN_ANSWER / PARTIAL queries.
        """
        label = self.checker.check(user_query)
        logger.info(f"[RelevanceChecker] label={label} query={user_query!r}")

        if label == "NO_MATCH":
            return _DEFAULT_NO_MATCH_RESPONSE

        # CAN_ANSWER or PARTIAL
        return self.retriever.retrieve(user_query)
