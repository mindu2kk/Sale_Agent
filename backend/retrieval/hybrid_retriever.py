"""
HybridRetriever: Combines BM25 (keyword) and ChromaDB vector search,
merges results via Reciprocal Rank Fusion (RRF), and applies SKU boost.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import logging

# ChromaDB
import chromadb

# LlamaIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# BM25
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(value: str) -> list[str]:
    decomposed = unicodedata.normalize(
        "NFD", value.casefold().replace("đ", "d")
    )
    ascii_value = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return re.findall(r"[a-z0-9]+", ascii_value)

class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search with RRF fusion."""

    def __init__(
        self,
        docstore_path: str = "./chroma_db/docstore.json",
        chroma_path: str = "./chroma_db",
        collection_name: str = "sales_copilot_vdb",
        embed_model_name: str = "BAAI/bge-m3",
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        """
        Initialize HybridRetriever.

        Args:
            docstore_path: Path to the docstore JSON file used to build BM25 index.
            chroma_path: Path to the ChromaDB persistent storage directory.
            collection_name: Name of the ChromaDB collection to connect to.
            embed_model_name: HuggingFace model name for query embedding.
            bm25_top_k: Number of top results to retrieve from BM25.
            vector_top_k: Number of top results to retrieve from vector search.
            rrf_k: RRF constant k (default 60 per literature standard).

        Raises:
            FileNotFoundError: If docstore_path does not exist or cannot be read.
            ValueError: If the ChromaDB collection does not exist.
        """
        self.docstore_path = docstore_path
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.rrf_k = rrf_k

        # Populated during _init_bm25 and _init_vector
        self._nodes: list[Any] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []
        self._bm25_retriever: Any = None
        self._vector_retriever: Any = None

        self._init_bm25()
        self._init_vector()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_bm25(self) -> None:
        """Load nodes from docstore and build BM25 index."""
        if not os.path.exists(self.docstore_path):
            raise FileNotFoundError(
                f"Docstore file not found at '{self.docstore_path}'. "
                "Please run the ingestion pipeline first to generate the docstore."
            )

        with open(self.docstore_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docstore_data: dict = data.get("docstore/data", {})

        nodes: list[TextNode] = []
        for node_id, node_entry in docstore_data.items():
            node_data: dict = node_entry.get("__data__", {})
            node = TextNode(**node_data)
            metadata = node.metadata or {}
            file_name = str(metadata.get("file_name", "")).casefold()
            if (
                not metadata.get("source_type")
                and any(term in file_name for term in ("policy", "warranty", "return", "exchange"))
            ):
                metadata["source_type"] = "policy_pdf"
                node.metadata = metadata
            nodes.append(node)

        self._nodes = nodes
        # Tokenize each node's text by splitting on whitespace
        self._bm25_corpus = [_tokenize(node.text) for node in self._nodes]

        # Build BM25Okapi index; handle empty corpus gracefully
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)
        else:
            # BM25Okapi requires at least one document; use a dummy token list
            # so the object is valid but will return zero scores for any query.
            self._bm25 = BM25Okapi([[""]])

    def _init_vector(self) -> None:
        """Connect to ChromaDB and initialise the vector retriever."""
        try:
            db = chromadb.PersistentClient(path=self.chroma_path)

            existing = [c.name for c in db.list_collections()]
            if self.collection_name not in existing:
                raise ValueError(
                    f"ChromaDB collection '{self.collection_name}' does not exist at "
                    f"'{self.chroma_path}'. Please run the ingestion pipeline first."
                )

            chroma_collection = db.get_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            embed_model = HuggingFaceEmbedding(
                model_name=self.embed_model_name,
                device="cpu",
            )
            self._vector_retriever = index.as_retriever(
                similarity_top_k=self.vector_top_k,
                embed_model=embed_model,
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(
                "Vector retriever unavailable, falling back to BM25 only: %s",
                exc,
            )
            self._vector_retriever = None

    def _vector_search(self, query: str) -> list[NodeWithScore]:
        """Return top ``self.vector_top_k`` nodes from vector search.

        Args:
            query: The search query string.

        Returns:
            List of NodeWithScore from the vector retriever.
        """
        if self._vector_retriever is None:
            return []
        return self._vector_retriever.retrieve(query)

    def _bm25_search(self, query: str) -> list[NodeWithScore]:
        """Return top ``self.bm25_top_k`` nodes scored by BM25.

        Args:
            query: The search query string.

        Returns:
            List of NodeWithScore sorted by BM25 score descending,
            length <= bm25_top_k. Returns an empty list when the corpus
            is empty.
        """
        if not self._nodes:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Pair each node with its score, sort descending, take top-k
        scored = sorted(
            zip(self._nodes, scores),
            key=lambda x: x[1],
            reverse=True,
        )[: self.bm25_top_k]

        return [NodeWithScore(node=node, score=float(score)) for node, score in scored]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[NodeWithScore]:
        """
        Retrieve the most relevant nodes for a query.

        Runs BM25 and vector search in parallel, merges results with RRF,
        applies SKU exact-match boost, and returns the top_k nodes.

        Args:
            query: The search query string.
            top_k: Maximum number of nodes to return (default 10).

        Returns:
            List of NodeWithScore sorted by descending fusion score,
            length <= top_k.
        """
        # Task 3.1 — parallel execution via ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_bm25 = executor.submit(self._bm25_search, query)
            future_vector = executor.submit(self._vector_search, query)
            bm25_results: list[NodeWithScore] = future_bm25.result()
            vector_results: list[NodeWithScore] = future_vector.result()

        # Task 3.2 — RRF fusion
        # source_type priority for tie-breaking (Task 3.6)
        source_priority = {"product_catalog": 1, "policy_pdf": 0}

        fusion_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        for result_list in (bm25_results, vector_results):
            for rank, nws in enumerate(result_list, start=1):
                node_id = nws.node.node_id
                fusion_scores[node_id] = fusion_scores.get(node_id, 0.0) + 1.0 / (self.rrf_k + rank)
                if node_id not in node_map:
                    node_map[node_id] = nws

        # Task 3.3 — sort descending by fusion_score; tie-break by source_type priority (Task 3.6)
        def sort_key(node_id: str):
            score = fusion_scores[node_id]
            priority = source_priority.get(
                node_map[node_id].node.metadata.get("source_type", ""), 0
            )
            return (score, priority)

        sorted_ids = sorted(fusion_scores.keys(), key=sort_key, reverse=True)

        results = [
            NodeWithScore(node=node_map[nid].node, score=fusion_scores[nid])
            for nid in sorted_ids
        ]

        # Task 3.5 — SKU boost: preserve punctuation such as hyphens in product codes.
        query_upper = query.upper()
        boost_index: int | None = None
        for i, nws in enumerate(results):
            product_code = nws.node.metadata.get("product_code", "")
            normalized_code = str(product_code).strip().upper()
            if normalized_code and re.search(
                rf"(?<![A-Z0-9]){re.escape(normalized_code)}(?![A-Z0-9])",
                query_upper,
            ):
                boost_index = i
                break

        if boost_index is not None and boost_index != 0:
            boosted = results.pop(boost_index)
            results.insert(0, boosted)

        return results[:top_k]
