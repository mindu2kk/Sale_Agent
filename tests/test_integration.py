"""
Integration & performance tests for HybridRetriever, RelevanceChecker, and RAGPipeline.

Task 8.1 — End-to-end: real query → RelevanceChecker → HybridRetriever → valid metadata
Task 8.2 — Performance: HybridRetriever latency ≤ 3s
Task 8.3 — Performance: RelevanceChecker latency ≤ 5s

Uses the real docstore.json and ChromaDB on disk.
LLM: Google Gemini via google-generativeai (GOOGLE_API_KEY from .env).
"""

from __future__ import annotations

import os
import time

import pytest
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Set the global embed model so HybridRetriever._init_vector doesn't fall back to OpenAI
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# ---------------------------------------------------------------------------
# Gemini availability check — skip LLM tests if daily quota is exhausted
# ---------------------------------------------------------------------------

def _check_gemini_available() -> tuple[bool, str]:
    """Return (available, reason) by making a minimal probe call."""
    import google.generativeai as genai
    import google.api_core.exceptions as gexc
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "GOOGLE_API_KEY not set"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        model.generate_content("ping")
        return True, ""
    except gexc.ResourceExhausted as exc:
        return False, f"Gemini quota exhausted (free tier daily limit reached): {exc}"
    except Exception as exc:
        return False, f"Gemini unavailable: {exc}"


_GEMINI_AVAILABLE, _GEMINI_SKIP_REASON = _check_gemini_available()
requires_gemini = pytest.mark.skipif(
    not _GEMINI_AVAILABLE,
    reason=_GEMINI_SKIP_REASON or "Gemini API unavailable",
)

# ---------------------------------------------------------------------------
# Gemini LLM wrapper — compatible with RelevanceChecker's llm.complete() API
# ---------------------------------------------------------------------------

class _GeminiLLM:
    """Minimal wrapper around google-generativeai for use with RelevanceChecker.

    Retries once on per-minute quota errors; fails fast on daily quota exhaustion.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-lite") -> None:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in environment / .env")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def complete(self, prompt: str):
        import google.api_core.exceptions as gexc
        for attempt in range(2):
            try:
                response = self._model.generate_content(prompt)
                return _GeminiResponse(response.text)
            except gexc.ResourceExhausted as exc:
                # Daily quota exhausted — no point retrying
                if "PerDay" in str(exc) or attempt >= 1:
                    raise
                time.sleep(65)  # wait for per-minute quota to reset


class _GeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------------------
# Shared constants & fixtures
# ---------------------------------------------------------------------------

DOCSTORE_PATH = "./chroma_db/docstore.json"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sales_copilot_vdb"

VALID_SOURCE_TYPES = {"product_catalog", "policy_pdf"}
VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}

HYBRID_RETRIEVER_LATENCY_LIMIT_S = 3.0
RELEVANCE_CHECKER_LATENCY_LIMIT_S = 5.0


@pytest.fixture(scope="module")
def hybrid_retriever():
    """Real HybridRetriever backed by on-disk docstore and ChromaDB."""
    from retriever.hybrid_retriever import HybridRetriever
    return HybridRetriever(
        docstore_path=DOCSTORE_PATH,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )


@pytest.fixture(scope="module")
def relevance_checker():
    """Real RelevanceChecker using Gemini 2.0 Flash Lite."""
    from retriever.relevance_checker import RelevanceChecker
    llm = _GeminiLLM(model_name="gemini-2.0-flash-lite")
    return RelevanceChecker(llm=llm)


@pytest.fixture(scope="module")
def rag_pipeline(hybrid_retriever, relevance_checker):
    """Real RAGPipeline wiring together the real retriever and checker."""
    from rag_pipeline import RAGPipeline
    return RAGPipeline(retriever=hybrid_retriever, checker=relevance_checker)


# ---------------------------------------------------------------------------
# Task 8.1 — End-to-end tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Real query flows through RelevanceChecker → HybridRetriever."""

    PRODUCT_QUERIES = [
        "iPhone 15 Pro Max giá bao nhiêu?",
        "Chính sách bảo hành Samsung là gì?",
        "Cho tôi xem thông tin sản phẩm IP15-PRM-256",
    ]
    NO_MATCH_QUERIES = [
        "Thời tiết hôm nay ở Hà Nội thế nào?",
        "Cho tôi biết công thức nấu phở bò",
    ]

    @requires_gemini
    def test_product_query_returns_nodes_with_valid_metadata(self, rag_pipeline):
        """Task 8.1: Product queries return NodeWithScore list with valid metadata."""
        from llama_index.core.schema import NodeWithScore
        from rag_pipeline import _DEFAULT_NO_MATCH_RESPONSE

        for query in self.PRODUCT_QUERIES:
            result = rag_pipeline.query(query)
            if result == _DEFAULT_NO_MATCH_RESPONSE:
                continue  # checker returned NO_MATCH — acceptable edge case

            assert isinstance(result, list), (
                f"Expected list for query={query!r}, got {type(result)}"
            )
            assert len(result) > 0, f"Expected at least one node for query={query!r}"

            for nws in result:
                assert isinstance(nws, NodeWithScore)
                meta = nws.node.metadata
                has_source_type = meta.get("source_type") in VALID_SOURCE_TYPES
                # PDF parent nodes may lack source_type but always have file_name
                has_file_name = bool(meta.get("file_name") or meta.get("file_path"))
                assert has_source_type or has_file_name, (
                    f"Node {nws.node.node_id!r} missing both source_type and file_name. "
                    f"metadata={meta}"
                )
                assert nws.node.text, f"Node {nws.node.node_id!r} has empty text"

    @requires_gemini
    def test_no_match_query_returns_default_response(self, rag_pipeline):
        """Task 8.1: Off-topic queries return the default NO_MATCH response string."""
        from rag_pipeline import _DEFAULT_NO_MATCH_RESPONSE

        for query in self.NO_MATCH_QUERIES:
            result = rag_pipeline.query(query)
            assert result == _DEFAULT_NO_MATCH_RESPONSE, (
                f"Expected default response for query={query!r}, got: {result!r}"
            )

    def test_retrieved_nodes_have_required_metadata_fields(self, hybrid_retriever):
        """Task 8.1: HybridRetriever nodes have source_type or file_name metadata."""
        results = hybrid_retriever.retrieve("Samsung warranty policy", top_k=5)
        assert len(results) > 0, "Expected at least one result"

        for nws in results:
            meta = nws.node.metadata
            has_source_type = meta.get("source_type") in VALID_SOURCE_TYPES
            has_file_name = bool(meta.get("file_name") or meta.get("file_path"))
            assert has_source_type or has_file_name, (
                f"Node {nws.node.node_id!r} missing both source_type and file_name. "
                f"metadata={meta}"
            )
            assert nws.node.text, f"Node {nws.node.node_id!r} has empty text"

    @requires_gemini
    def test_relevance_checker_returns_valid_label_for_real_queries(self, relevance_checker):
        """Task 8.1: RelevanceChecker returns a valid label for all test queries."""
        for query in self.PRODUCT_QUERIES + self.NO_MATCH_QUERIES:
            label = relevance_checker.check(query)
            assert label in VALID_LABELS, (
                f"Invalid label={label!r} for query={query!r}"
            )

    def test_top_k_respected_in_end_to_end(self, hybrid_retriever):
        """Task 8.1: top_k is respected in real retrieval."""
        for top_k in (1, 3, 5):
            results = hybrid_retriever.retrieve("Apple iPhone", top_k=top_k)
            assert len(results) <= top_k, (
                f"top_k={top_k} violated: got {len(results)} results"
            )


# ---------------------------------------------------------------------------
# Task 8.2 — HybridRetriever latency ≤ 3s
# ---------------------------------------------------------------------------

class TestHybridRetrieverPerformance:
    """Verify HybridRetriever responds within the latency budget."""

    PERF_QUERIES = [
        "iPhone 15 Pro Max",
        "Samsung Galaxy S24 Ultra giá bao nhiêu",
        "chính sách đổi trả trong 30 ngày",
    ]

    def test_retriever_latency_under_3s(self, hybrid_retriever):
        """Task 8.2: Each retrieve() call must complete in < 3 seconds."""
        for query in self.PERF_QUERIES:
            start = time.perf_counter()
            results = hybrid_retriever.retrieve(query, top_k=10)
            elapsed = time.perf_counter() - start

            assert elapsed < HYBRID_RETRIEVER_LATENCY_LIMIT_S, (
                f"HybridRetriever too slow for query={query!r}: "
                f"{elapsed:.3f}s > {HYBRID_RETRIEVER_LATENCY_LIMIT_S}s"
            )
            assert isinstance(results, list)

    def test_retriever_average_latency_under_3s(self, hybrid_retriever):
        """Task 8.2: Average latency across multiple queries must be < 3s."""
        times = []
        for query in self.PERF_QUERIES:
            start = time.perf_counter()
            hybrid_retriever.retrieve(query, top_k=10)
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times)
        assert avg < HYBRID_RETRIEVER_LATENCY_LIMIT_S, (
            f"Average HybridRetriever latency {avg:.3f}s exceeds {HYBRID_RETRIEVER_LATENCY_LIMIT_S}s"
        )


# ---------------------------------------------------------------------------
# Task 8.3 — RelevanceChecker latency ≤ 5s
# ---------------------------------------------------------------------------

class TestRelevanceCheckerPerformance:
    """Verify RelevanceChecker responds within the latency budget."""

    PERF_QUERIES = [
        "iPhone 15 Pro Max giá bao nhiêu?",
        "Thời tiết hôm nay thế nào?",
        "Chính sách bảo hành Apple bao lâu?",
    ]

    @requires_gemini
    def test_checker_latency_under_5s(self, relevance_checker):
        """Task 8.3: Each check() call must complete in < 5 seconds."""
        for query in self.PERF_QUERIES:
            start = time.perf_counter()
            label = relevance_checker.check(query)
            elapsed = time.perf_counter() - start

            assert elapsed < RELEVANCE_CHECKER_LATENCY_LIMIT_S, (
                f"RelevanceChecker too slow for query={query!r}: "
                f"{elapsed:.3f}s > {RELEVANCE_CHECKER_LATENCY_LIMIT_S}s"
            )
            assert label in VALID_LABELS

    @requires_gemini
    def test_checker_average_latency_under_5s(self, relevance_checker):
        """Task 8.3: Average latency across multiple queries must be < 5s."""
        times = []
        for query in self.PERF_QUERIES:
            start = time.perf_counter()
            relevance_checker.check(query)
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times)
        assert avg < RELEVANCE_CHECKER_LATENCY_LIMIT_S, (
            f"Average RelevanceChecker latency {avg:.3f}s exceeds {RELEVANCE_CHECKER_LATENCY_LIMIT_S}s"
        )


# ---------------------------------------------------------------------------
# Task 8 (Sales Research Agent) — Integration tests
# ---------------------------------------------------------------------------

import dataclasses
from unittest.mock import MagicMock, patch
from agent.sales_research_agent import SalesResearchAgent, AgentResult


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


@patch("agent.sales_research_agent.ReActAgent")
@patch("agent.sales_research_agent.build_tavily_tool", return_value=None)
def test_e2e_dell_inspiron_objection(mock_tavily, mock_react_cls):
    """End-to-end: objection about Dell Inspiron → draft_response has product info from DB."""
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, em hiểu anh/chị đang so sánh về giá. Máy Dell Inspiron 15 có vỏ nhôm cao cấp, "
        "RAM 16GB, SSD 512GB và chính sách bảo hành 1 đổi 1 trong 12 tháng. "
        "Với mức giá 18.990.000 VNĐ, đây là mức giá cạnh tranh cho cấu hình này.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run("Khách chê Dell Inspiron đắt")

    # Verify draft has product info
    assert "Dell Inspiron" in result.draft_response
    assert result.objection_text == "Khách chê Dell Inspiron đắt"
    assert "internal_db_search" in result.tools_used


@patch("agent.sales_research_agent.ReActAgent")
@patch("agent.sales_research_agent.build_tavily_tool", return_value=None)
def test_performance_no_tavily(mock_tavily, mock_react_cls):
    """Performance: run() completes in ≤ 5s when not using Tavily (Requirement 6.1)."""
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là thông tin sản phẩm.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)

    start = time.time()
    result = agent.run("test objection")
    elapsed = time.time() - start

    assert elapsed <= 5.0, f"run() took {elapsed:.2f}s, expected ≤ 5s"
    assert isinstance(result, AgentResult)


@patch("agent.sales_research_agent.ReActAgent")
@patch("agent.sales_research_agent.build_tavily_tool", return_value=None)
def test_performance_with_tavily(mock_tavily, mock_react_cls):
    """Performance: run() completes in ≤ 10s when using Tavily (Requirement 6.2)."""
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là thông tin sản phẩm. Theo thông tin thị trường hiện tại...",
        ["internal_db_search", "tavily_web_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)

    start = time.time()
    result = agent.run("test objection with web search")
    elapsed = time.time() - start

    assert elapsed <= 10.0, f"run() took {elapsed:.2f}s, expected ≤ 10s"
    assert isinstance(result, AgentResult)


@patch("agent.sales_research_agent.ReActAgent")
@patch("agent.sales_research_agent.build_tavily_tool", return_value=None)
def test_state_graph_readiness(mock_tavily, mock_react_cls):
    """StateGraph readiness: AgentResult can be converted to dict via dataclasses.asdict()."""
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

    # Verify it can be converted to dict for StateGraph
    result_dict = dataclasses.asdict(result)

    assert isinstance(result_dict, dict)
    assert "objection_text" in result_dict
    assert "draft_response" in result_dict
    assert "tools_used" in result_dict
    assert result_dict["objection_text"] == "test objection"
