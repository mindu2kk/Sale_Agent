"""Unified RAG adapter: live CSV catalog for products + Chroma for policies."""

from __future__ import annotations

import logging

from llama_index.core.schema import NodeWithScore, TextNode

from backend.services.catalog import CatalogService

logger = logging.getLogger(__name__)


class GroundedRAGPipeline:
    def __init__(self, catalog: CatalogService, policy_retriever, checker=None) -> None:
        self.catalog = catalog
        self.policy_retriever = policy_retriever
        # Compatibility for verification components built against RAGPipeline.retriever.
        self.retriever = policy_retriever
        self.checker = checker

    def query(self, user_query: str) -> list[NodeWithScore] | str:
        if not user_query.strip():
            return []

        catalog_products = self.catalog.search(user_query, limit=8)
        product_nodes = [
            NodeWithScore(
                node=TextNode(
                    text=product.context,
                    metadata={
                        "source_type": "product_catalog",
                        "product_code": product.code,
                        "brand": product.brand,
                        "category": product.category,
                        "price": product.price,
                    },
                ),
                score=1.0 - index * 0.05,
            )
            for index, product in enumerate(catalog_products)
        ]

        policy_terms = (
            "bảo hành",
            "đổi trả",
            "chính sách",
            "warranty",
            "return",
            "exchange",
        )
        should_search_policy = not product_nodes or any(
            term in user_query.casefold() for term in policy_terms
        )
        policy_nodes: list[NodeWithScore] = []
        if should_search_policy:
            try:
                policy_nodes = [
                    node
                    for node in self.policy_retriever.retrieve(user_query, top_k=6)
                    if self._is_policy_node(node)
                ]
            except Exception as exc:
                logger.warning("Policy retriever failed: %s", exc)

        results = product_nodes + policy_nodes
        if results:
            return results
        return (
            "Hệ thống chưa tìm thấy thông tin phù hợp trong catalog sản phẩm "
            "hoặc tài liệu chính sách nội bộ."
        )

    @staticmethod
    def _is_policy_node(node: NodeWithScore) -> bool:
        metadata = node.node.metadata or {}
        if metadata.get("source_type") == "policy_pdf":
            return True
        file_name = str(metadata.get("file_name", "")).casefold()
        return any(
            token in file_name
            for token in ("policy", "warranty", "return", "exchange")
        )
