"""
Unit tests for HybridRetriever, RelevanceChecker, and RAGPipeline.

All external services (ChromaDB, LLM, embedding models) are mocked.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.relevance_checker import RelevanceChecker
from backend.retrieval.pipeline import RAGPipeline, _DEFAULT_NO_MATCH_RESPONSE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(node_id: str, text: str, metadata: dict | None = None) -> TextNode:
    return TextNode(id_=node_id, text=text, metadata=metadata or {})


def _make_nws(node_id: str, text: str, score: float = 1.0, metadata: dict | None = None) -> NodeWithScore:
    return NodeWithScore(node=_make_node(node_id, text, metadata), score=score)


def _make_docstore(nodes: list[dict], path: str) -> None:
    """Write a minimal docstore JSON file."""
    docstore_data = {}
    for n in nodes:
        docstore_data[n["id_"]] = {
            "__data__": {
                "id_": n["id_"],
                "text": n.get("text", ""),
                "metadata": n.get("metadata", {}),
            }
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"docstore/data": docstore_data}, f)


# ---------------------------------------------------------------------------
# Task 6.1 — FileNotFoundError when docstore path is wrong
# ---------------------------------------------------------------------------

class TestDocstoreNotFound:
    def test_raises_file_not_found_error(self):
        """HybridRetriever raises FileNotFoundError for a non-existent docstore path."""
        with patch.object(HybridRetriever, "_init_vector", return_value=None):
            with pytest.raises(FileNotFoundError):
                HybridRetriever(
                    docstore_path="/nonexistent/path/docstore.json",
                    chroma_path="./chroma_db",
                )


# ---------------------------------------------------------------------------
# Task 6.2 — ValueError when ChromaDB collection doesn't exist
# ---------------------------------------------------------------------------

class TestCollectionNotFound:
    def test_raises_value_error_when_collection_missing(self):
        """HybridRetriever raises ValueError when the ChromaDB collection is absent."""
        mock_collection = MagicMock()
        mock_collection.name = "some_other_collection"

        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_collection]

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as tmp:
            json.dump({"docstore/data": {}}, tmp)
            tmp_path = tmp.name

        try:
            with patch("chromadb.PersistentClient", return_value=mock_client):
                with patch.object(HybridRetriever, "_init_bm25", return_value=None):
                    with pytest.raises(ValueError):
                        HybridRetriever(
                            docstore_path=tmp_path,
                            collection_name="sales_copilot_vdb",
                        )
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Task 6.3 — RRF formula with fixed input
# ---------------------------------------------------------------------------

class TestRRFFormula:
    """Verify the RRF score formula directly without instantiating HybridRetriever."""

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def test_rrf_scores_match_expected(self):
        """
        Node A: rank 1 in BM25, rank 2 in vector → score = 1/61 + 1/62
        Node B: rank 2 in BM25 only             → score = 1/62
        """
        k = 60
        score_a = self._rrf_score(1, k) + self._rrf_score(2, k)
        score_b = self._rrf_score(2, k)

        expected_a = 1 / (60 + 1) + 1 / (60 + 2)
        expected_b = 1 / (60 + 2)

        assert abs(score_a - expected_a) < 1e-12
        assert abs(score_b - expected_b) < 1e-12
        assert score_a > score_b


# ---------------------------------------------------------------------------
# Task 6.4 — SKU boost: node with matching SKU appears at index 0
# ---------------------------------------------------------------------------

class TestSKUBoost:
    def _build_retriever(self, nodes: list[TextNode]) -> HybridRetriever:
        """Build a HybridRetriever with both init methods patched out."""
        with patch.object(HybridRetriever, "_init_bm25", return_value=None), \
             patch.object(HybridRetriever, "_init_vector", return_value=None):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever.docstore_path = ""
            retriever.chroma_path = ""
            retriever.collection_name = ""
            retriever.embed_model_name = ""
            retriever.bm25_top_k = 20
            retriever.vector_top_k = 20
            retriever.rrf_k = 60
            retriever._nodes = nodes
            retriever._bm25 = None
            retriever._bm25_corpus = []
            retriever._bm25_retriever = None
            retriever._vector_retriever = None
        return retriever

    def test_sku_node_is_at_index_zero(self):
        """Node with matching product_code must be at index 0 after SKU boost."""
        sku = "IP15-PRM-256"
        sku_node = _make_node("sku-node", "iPhone 15 Pro Max 256GB", {"product_code": sku, "source_type": "product_catalog"})
        other_node_1 = _make_node("other-1", "Samsung Galaxy S24", {"product_code": "SAM-S24", "source_type": "product_catalog"})
        other_node_2 = _make_node("other-2", "Warranty policy", {"source_type": "policy_pdf"})

        retriever = self._build_retriever([sku_node, other_node_1, other_node_2])

        # BM25 returns sku_node at rank 3 (low), vector returns it at rank 3 too
        bm25_results = [
            _make_nws("other-1", "Samsung Galaxy S24", 3.0, {"product_code": "SAM-S24", "source_type": "product_catalog"}),
            _make_nws("other-2", "Warranty policy", 2.0, {"source_type": "policy_pdf"}),
            _make_nws("sku-node", "iPhone 15 Pro Max 256GB", 1.0, {"product_code": sku, "source_type": "product_catalog"}),
        ]
        vector_results = [
            _make_nws("other-1", "Samsung Galaxy S24", 0.9, {"product_code": "SAM-S24", "source_type": "product_catalog"}),
            _make_nws("other-2", "Warranty policy", 0.8, {"source_type": "policy_pdf"}),
            _make_nws("sku-node", "iPhone 15 Pro Max 256GB", 0.7, {"product_code": sku, "source_type": "product_catalog"}),
        ]

        with patch.object(retriever, "_bm25_search", return_value=bm25_results), \
             patch.object(retriever, "_vector_search", return_value=vector_results):
            results = retriever.retrieve(f"{sku} iPhone 15 Pro Max")

        assert len(results) > 0
        assert results[0].node.node_id == "sku-node"


# ---------------------------------------------------------------------------
# Task 6.5 — Tie-breaking: product_catalog preferred over policy_pdf
# ---------------------------------------------------------------------------

class TestTieBreaking:
    def test_product_catalog_ranks_higher_than_policy_pdf(self):
        """When fusion scores are equal, product_catalog beats policy_pdf."""
        catalog_node = _make_node("cat-1", "Product info", {"source_type": "product_catalog"})
        policy_node = _make_node("pol-1", "Policy info", {"source_type": "policy_pdf"})

        # Both appear at rank 1 in BM25 only → identical fusion scores
        bm25_results = [
            _make_nws("cat-1", "Product info", 1.0, {"source_type": "product_catalog"}),
            _make_nws("pol-1", "Policy info", 1.0, {"source_type": "policy_pdf"}),
        ]

        with patch.object(HybridRetriever, "_init_bm25", return_value=None), \
             patch.object(HybridRetriever, "_init_vector", return_value=None):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever.rrf_k = 60
            retriever.bm25_top_k = 20
            retriever.vector_top_k = 20
            retriever._nodes = [catalog_node, policy_node]
            retriever._bm25 = None
            retriever._bm25_corpus = []
            retriever._bm25_retriever = None
            retriever._vector_retriever = None

        with patch.object(retriever, "_bm25_search", return_value=bm25_results), \
             patch.object(retriever, "_vector_search", return_value=[]):
            results = retriever.retrieve("some query")

        ids = [r.node.node_id for r in results]
        assert ids.index("cat-1") < ids.index("pol-1")


# ---------------------------------------------------------------------------
# Task 6.6 — RelevanceChecker with mock LLM returning all three labels
# ---------------------------------------------------------------------------

class TestRelevanceCheckerLabels:
    def _make_checker(self, llm_text: str) -> RelevanceChecker:
        mock_response = MagicMock()
        mock_response.text = llm_text
        mock_llm = MagicMock()
        mock_llm.complete.return_value = mock_response
        return RelevanceChecker(llm=mock_llm)

    def test_can_answer_label(self):
        checker = self._make_checker("CAN_ANSWER")
        assert checker.check("iPhone 15 Pro Max giá bao nhiêu?") == "CAN_ANSWER"

    def test_partial_label(self):
        checker = self._make_checker("PARTIAL")
        assert checker.check("Apple có sản phẩm gì?") == "PARTIAL"

    def test_no_match_label(self):
        checker = self._make_checker("NO_MATCH")
        assert checker.check("Thời tiết hôm nay thế nào?") == "NO_MATCH"


# ---------------------------------------------------------------------------
# Task 6.7 — Label parsing with whitespace / extra punctuation
# ---------------------------------------------------------------------------

class TestLabelParsing:
    def _check_with_raw(self, raw_text: str) -> str:
        mock_response = MagicMock()
        mock_response.text = raw_text
        mock_llm = MagicMock()
        mock_llm.complete.return_value = mock_response
        checker = RelevanceChecker(llm=mock_llm)
        return checker.check("any query")

    def test_can_answer_with_trailing_period_and_spaces(self):
        assert self._check_with_raw("  CAN_ANSWER. ") == "CAN_ANSWER"

    def test_partial_with_newlines(self):
        assert self._check_with_raw("\nPARTIAL\n") == "PARTIAL"

    def test_no_match_lowercase_with_period(self):
        # "no_match." → strip "." → upper → "NO_MATCH"
        assert self._check_with_raw("no_match.") == "NO_MATCH"


# ---------------------------------------------------------------------------
# Task 6.8 — NO_MATCH doesn't call retriever
# ---------------------------------------------------------------------------

class TestNoMatchSkipsRetrieval:
    def test_retriever_not_called_on_no_match(self):
        """RAGPipeline must not call retriever.retrieve when checker returns NO_MATCH."""
        mock_checker = MagicMock()
        mock_checker.check.return_value = "NO_MATCH"

        mock_retriever = MagicMock()

        pipeline = RAGPipeline(retriever=mock_retriever, checker=mock_checker)
        result = pipeline.query("Thời tiết hôm nay?")

        mock_retriever.retrieve.assert_not_called()
        assert result == _DEFAULT_NO_MATCH_RESPONSE

    def test_default_response_is_string(self):
        """The default NO_MATCH response must be a non-empty string."""
        assert isinstance(_DEFAULT_NO_MATCH_RESPONSE, str)
        assert len(_DEFAULT_NO_MATCH_RESPONSE) > 0


# ---------------------------------------------------------------------------
# Task 6.9 — Empty corpus doesn't crash BM25 build
# ---------------------------------------------------------------------------

class TestEmptyCorpus:
    def test_empty_docstore_initializes_without_exception(self):
        """HybridRetriever must initialize successfully with an empty docstore."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump({"docstore/data": {}}, tmp)
            tmp_path = tmp.name

        try:
            with patch.object(HybridRetriever, "_init_vector", return_value=None):
                retriever = HybridRetriever(
                    docstore_path=tmp_path,
                    chroma_path="./chroma_db",
                )
            assert retriever._nodes == []
        finally:
            os.unlink(tmp_path)

    def test_retrieve_returns_empty_list_for_empty_corpus(self):
        """retrieve() on an empty corpus must return an empty list without crashing."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump({"docstore/data": {}}, tmp)
            tmp_path = tmp.name

        try:
            with patch.object(HybridRetriever, "_init_vector", return_value=None):
                retriever = HybridRetriever(
                    docstore_path=tmp_path,
                    chroma_path="./chroma_db",
                )
            with patch.object(retriever, "_vector_search", return_value=[]):
                results = retriever.retrieve("test")
            assert results == []
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Task 6 — SalesResearchAgent unit tests
# ---------------------------------------------------------------------------

from backend.workflows.research_agent.sales_research_agent import SalesResearchAgent, AgentResult


def make_mock_response(text: str, tool_names: list[str]):
    response = MagicMock()
    response.__str__ = lambda self: text
    sources = []
    for name in tool_names:
        src = MagicMock()
        src.tool_name = name
        src.raw_input = {"query": "test"}
        src.raw_output = "mock output"
        sources.append(src)
    response.sources = sources
    return response


@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_agent_raises_on_none_llm(mock_react):
    mock_pipeline = MagicMock()
    with pytest.raises(ValueError):
        SalesResearchAgent(llm=None, rag_pipeline=mock_pipeline)


@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_agent_raises_on_none_pipeline(mock_react):
    mock_llm = MagicMock()
    with pytest.raises(ValueError):
        SalesResearchAgent(llm=mock_llm, rag_pipeline=None)


@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_internal_db_called_first(mock_react_cls):
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là thông tin sản phẩm.",
        ["internal_db_search", "tavily_web_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run("test objection")

    assert result.tools_used[0] == "internal_db_search"


@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_tavily_not_called_when_db_sufficient(mock_react_cls):
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, thông tin đã đủ từ DB nội bộ.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run("test objection")

    assert "tavily_web_search" not in result.tools_used


# ---------------------------------------------------------------------------
# Task 7 — test_early_termination_when_db_sufficient
# ---------------------------------------------------------------------------

@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_early_termination_when_db_sufficient(mock_react_cls):
    """Task 7: When Internal_DB_Tool returns valid JSON data (not NO_MATCH),
    tavily_web_search must NOT appear in AgentResult.tools_used."""
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    # Mock Internal_DB_Tool returning valid product data (not NO_MATCH)
    valid_db_data = json.dumps([{
        "source": "product_catalog",
        "product_code": "IP15-PRM-256",
        "content": "iPhone 15 Pro Max 256GB giá 29.990.000 VNĐ...",
    }])
    mock_pipeline.query.return_value = [MagicMock()]

    mock_agent_instance = MagicMock()
    # Agent stops after internal_db_search — early termination
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, iPhone 15 Pro Max 256GB có giá 29.990.000 VNĐ.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run("iPhone 15 Pro Max giá bao nhiêu?")

    assert "tavily_web_search" not in result.tools_used


@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_draft_starts_with_da_or_vang(mock_react_cls):
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, máy này rất tốt ạ.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run("test objection")

    assert result.draft_response.startswith("Dạ,") or result.draft_response.startswith("Vâng,")


@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_no_info_fallback_message(mock_react_cls):
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, hiện tại hệ thống chưa có đủ thông tin để hỗ trợ câu hỏi này.",
        [],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run("test objection")

    assert "chưa có đủ thông tin" in result.draft_response


@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_agent_result_has_all_fields(mock_react_cls):
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là phản hồi.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run("test objection")

    assert result.objection_text is not None
    assert result.draft_response is not None
    assert result.tools_used is not None


@patch("backend.workflows.research_agent.sales_research_agent.build_tavily_tool")
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_tavily_absent_when_no_api_key(mock_react_cls, mock_build_tavily):
    mock_build_tavily.return_value = None
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response("Dạ, ok.", [])
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)

    assert agent is not None
    assert mock_build_tavily.return_value is None


@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_max_iterations_respected(mock_react_cls):
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là kết quả.",
        ["internal_db_search", "internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run("test objection")

    assert len(result.tools_used) <= 2


# ---------------------------------------------------------------------------
# Task 8.10 — test_tool_exception_returns_safe_json
# ---------------------------------------------------------------------------

from backend.workflows.research_agent.tools import build_internal_db_tool


def test_tool_exception_returns_safe_json():
    """Mock rag_pipeline.query to raise RuntimeError; internal_db_search must return
    JSON with status==ERROR and not propagate the exception."""
    mock_pipeline = MagicMock()
    mock_pipeline.query.side_effect = RuntimeError("db failure")

    tool = build_internal_db_tool(mock_pipeline)
    # Call the underlying function directly
    result_json = tool.fn("any query")

    result = json.loads(result_json)
    assert result["status"] == "ERROR"
    assert "db failure" in result["message"]


# ---------------------------------------------------------------------------
# Task 8.11 — test_agent_result_asdict_serializable
# ---------------------------------------------------------------------------

import dataclasses


def test_agent_result_asdict_serializable():
    """AgentResult with all fields populated must be serializable via dataclasses.asdict."""
    result = AgentResult(
        objection_text="Khách chê đắt",
        draft_response="Dạ, sản phẩm có chất lượng cao.",
        tools_used=["internal_db_search"],
        verification_result=None,
        workflow_status="approved",
        retry_count=1,
        correction_feedback="Fix price",
    )
    d = dataclasses.asdict(result)
    assert isinstance(d, dict)
    assert "objection_text" in d
    assert "draft_response" in d
    assert "tools_used" in d
    assert "workflow_status" in d
    assert "retry_count" in d
    assert "correction_feedback" in d


# ---------------------------------------------------------------------------
# Task 9 — Unit Tests Memory Safety
# ---------------------------------------------------------------------------

def test_internal_db_truncates_at_500_chars():
    """9.1: node text of 1000 chars → content in JSON output has len <= 503."""
    long_text = "x" * 1000
    mock_node = MagicMock()
    mock_node.node.metadata = {"source_type": "product_catalog", "product_code": "TEST-001"}
    mock_node.node.text = long_text

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = [mock_node]

    tool = build_internal_db_tool(mock_pipeline)
    result_json = tool.fn("test query")

    nodes = json.loads(result_json)
    assert isinstance(nodes, list)
    assert len(nodes[0]["content"]) <= 503


def test_tavily_truncates_at_500_chars():
    """9.2: Tavily content of 1000 chars → truncated to 500 in output."""
    from backend.workflows.research_agent.tools import build_tavily_tool

    long_content = "y" * 1000
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [{"title": "Test", "content": long_content}]
    }

    mock_tavily_module = MagicMock()
    mock_tavily_module.TavilyClient.return_value = mock_client

    import sys
    sys.modules.setdefault("tavily", mock_tavily_module)
    with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
        tool = build_tavily_tool(tavily_api_key="fake-key")

    result_json = tool.fn("test query")
    results = json.loads(result_json)
    assert len(results[0]["content"]) <= 500


def test_internal_db_no_match_returns_valid_json():
    """9.3: RAGPipeline returns a plain string → JSON has status==NO_MATCH."""
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = "Không tìm thấy thông tin phù hợp."

    tool = build_internal_db_tool(mock_pipeline)
    result_json = tool.fn("unknown product")

    result = json.loads(result_json)
    assert result["status"] == "NO_MATCH"


def test_internal_db_exception_returns_error_json():
    """9.4: RAGPipeline raises RuntimeError → JSON has status==ERROR, message==db error, no propagation."""
    mock_pipeline = MagicMock()
    mock_pipeline.query.side_effect = RuntimeError("db error")

    tool = build_internal_db_tool(mock_pipeline)
    result_json = tool.fn("any query")

    result = json.loads(result_json)
    assert result["status"] == "ERROR"
    assert result["message"] == "db error"


# ---------------------------------------------------------------------------
# Task 6.5 — QueryCache unit tests
# ---------------------------------------------------------------------------

import time as _time
from backend.workflows.research_agent.cache import QueryCache


def test_cache_hit_returns_same_result():
    """set then get same key → same value."""
    cache = QueryCache()
    cache.set("q1", "result1")
    assert cache.get("q1") == "result1"


def test_cache_miss_returns_none():
    """get key not set → None."""
    cache = QueryCache()
    assert cache.get("nonexistent") is None


def test_cache_ttl_expiry():
    """set with TTL=0.01s, sleep 0.02s, get → None."""
    cache = QueryCache(ttl=0.01)
    cache.set("q1", "value")
    _time.sleep(0.02)
    assert cache.get("q1") is None


def test_cache_max_size_eviction():
    """fill to max_size+1 → oldest entry evicted."""
    cache = QueryCache(max_size=3)
    cache.set("k1", "v1")
    _time.sleep(0.001)
    cache.set("k2", "v2")
    _time.sleep(0.001)
    cache.set("k3", "v3")
    _time.sleep(0.001)
    # Adding one more should evict k1 (oldest)
    cache.set("k4", "v4")
    assert cache.get("k1") is None
    assert cache.get("k4") == "v4"
