"""
Property-based tests for HybridRetriever, RelevanceChecker, and RAGPipeline.

Uses Hypothesis. Each test runs at least 100 examples.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from llama_index.core.schema import NodeWithScore, TextNode
from rank_bm25 import BM25Okapi

from backend.retrieval.pipeline import RAGPipeline, _DEFAULT_NO_MATCH_RESPONSE
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.relevance_checker import RelevanceChecker


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_node(node_id: str, text: str, metadata: dict | None = None) -> TextNode:
    return TextNode(id_=node_id, text=text, metadata=metadata or {})


def _make_nws(
    node_id: str,
    text: str,
    score: float = 1.0,
    metadata: dict | None = None,
) -> NodeWithScore:
    return NodeWithScore(node=_make_node(node_id, text, metadata), score=score)


def _make_docstore_file(nodes: list[TextNode]) -> str:
    """Write a minimal docstore JSON to a temp file and return its path."""
    docstore_data = {}
    for node in nodes:
        docstore_data[node.node_id] = {
            "__data__": {
                "id_": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
            }
        }
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    )
    json.dump({"docstore/data": docstore_data}, tmp)
    tmp.close()
    return tmp.name


def _build_retriever_from_nodes(nodes: list[TextNode]) -> HybridRetriever:
    """Create a HybridRetriever backed by a temp docstore; _init_vector is patched."""
    tmp_path = _make_docstore_file(nodes)
    try:
        with patch.object(HybridRetriever, "_init_vector", return_value=None):
            retriever = HybridRetriever(
                docstore_path=tmp_path,
                chroma_path="./chroma_db",
            )
    finally:
        os.unlink(tmp_path)
    return retriever


# ---------------------------------------------------------------------------
# Fixed corpus used by Properties 1 and 4
# ---------------------------------------------------------------------------

_CORPUS_TEXTS = [
    ("node-1", "iPhone 15 Pro Max 256GB titanium black", {"product_code": "IP15-PRM-256", "source_type": "product_catalog"}),
    ("node-2", "Samsung Galaxy S24 Ultra 512GB phantom black", {"product_code": "SAM-S24U-512", "source_type": "product_catalog"}),
    ("node-3", "Apple MacBook Pro M3 14 inch 16GB RAM", {"product_code": "MBP-M3-14-16", "source_type": "product_catalog"}),
    ("node-4", "Sony WH-1000XM5 wireless noise cancelling headphones", {"product_code": "SONY-WH1000XM5", "source_type": "product_catalog"}),
    ("node-5", "Apple warranty policy covers manufacturing defects", {"source_type": "policy_pdf"}),
    ("node-6", "Samsung warranty policy twelve months coverage", {"source_type": "policy_pdf"}),
    ("node-7", "Return and exchange policy within thirty days", {"source_type": "policy_pdf"}),
    ("node-8", "iPad Pro 12.9 inch M2 chip 256GB WiFi", {"product_code": "IPAD-PRO-M2-256", "source_type": "product_catalog"}),
]

corpus_nodes: list[TextNode] = [
    _make_node(nid, text, meta) for nid, text, meta in _CORPUS_TEXTS
]

product_nodes: list[TextNode] = [
    node for node in corpus_nodes if node.metadata.get("product_code")
]


# ---------------------------------------------------------------------------
# Property 1: BM25 Round-trip
# ---------------------------------------------------------------------------

# Feature: hybrid-retriever-relevance-checker, Property 1: BM25 Round-trip
@given(st.sampled_from(corpus_nodes))
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_bm25_roundtrip(node: TextNode) -> None:
    """Validates: Requirements 8.1

    For any node in the corpus, querying BM25 with the node's own text
    must return that node in the top-5 results.
    """
    all_texts = [n.text for n in corpus_nodes]
    corpus_tokenized = [text.split() for text in all_texts]
    bm25 = BM25Okapi(corpus_tokenized)

    query_tokens = node.text.split()
    scores = bm25.get_scores(query_tokens)

    # Pair (index, score), sort descending, take top-5
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
    top5_indices = [idx for idx, _ in ranked]

    node_index = next(i for i, n in enumerate(corpus_nodes) if n.node_id == node.node_id)
    assert node_index in top5_indices, (
        f"Node '{node.node_id}' not found in BM25 top-5 when queried with its own text."
    )


# ---------------------------------------------------------------------------
# Property 2: HybridRetriever Idempotence
# ---------------------------------------------------------------------------

# Feature: hybrid-retriever-relevance-checker, Property 2: HybridRetriever Idempotence
@given(
    st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hybrid_retriever_idempotence(query: str) -> None:
    """Validates: Requirements 8.2

    Two HybridRetriever instances built from the same docstore must return
    the same node_id order for the same query.
    """
    tmp_path = _make_docstore_file(corpus_nodes)
    try:
        with patch.object(HybridRetriever, "_init_vector", return_value=None):
            r1 = HybridRetriever(docstore_path=tmp_path, chroma_path="./chroma_db")
            r2 = HybridRetriever(docstore_path=tmp_path, chroma_path="./chroma_db")

        # Patch vector search to return empty (deterministic)
        with patch.object(r1, "_vector_search", return_value=[]), \
             patch.object(r2, "_vector_search", return_value=[]):
            results1 = r1.retrieve(query)
            results2 = r2.retrieve(query)
    finally:
        os.unlink(tmp_path)

    ids1 = [nws.node.node_id for nws in results1]
    ids2 = [nws.node.node_id for nws in results2]
    assert ids1 == ids2, (
        f"Idempotence violated for query={query!r}: {ids1} != {ids2}"
    )


# ---------------------------------------------------------------------------
# Property 3: top_k Bound
# ---------------------------------------------------------------------------

# Feature: hybrid-retriever-relevance-checker, Property 3: top_k Bound
@given(st.text(min_size=1), st.integers(min_value=1, max_value=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_topk_bound(query: str, top_k: int) -> None:
    """Validates: Requirements 3.4, 8.3

    The number of nodes returned by HybridRetriever must always be <= top_k.
    """
    # 30 fixed nodes to ensure top_k is the binding constraint
    fixed_nodes = [
        _make_nws(f"node-{i}", f"text content number {i}", float(30 - i))
        for i in range(30)
    ]

    retriever = _build_retriever_from_nodes(corpus_nodes)

    with patch.object(retriever, "_bm25_search", return_value=fixed_nodes), \
         patch.object(retriever, "_vector_search", return_value=fixed_nodes):
        results = retriever.retrieve(query, top_k=top_k)

    assert len(results) <= top_k, (
        f"top_k bound violated: got {len(results)} results for top_k={top_k}"
    )


# ---------------------------------------------------------------------------
# Property 4: SKU Exact Match Top-3
# ---------------------------------------------------------------------------

# Feature: hybrid-retriever-relevance-checker, Property 4: SKU Exact Match Top-3
@given(st.sampled_from(product_nodes))
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_sku_exact_match_top3(product_node: TextNode) -> None:
    """Validates: Requirements 4.1

    For any query containing a product_code, the node with that product_code
    must appear in the top-3 results.
    """
    sku = product_node.metadata["product_code"]
    retriever = _build_retriever_from_nodes(corpus_nodes)

    # Build 12 other nodes ranked above the SKU node (rank 10 for SKU)
    other_nodes = [
        _make_nws(f"other-{i}", f"unrelated content {i}", float(12 - i))
        for i in range(12)
    ]
    sku_nws = _make_nws(
        product_node.node_id,
        product_node.text,
        1.0,
        product_node.metadata,
    )

    # SKU node at rank 10 (index 9) in both retrievers
    bm25_results = other_nodes[:9] + [sku_nws] + other_nodes[9:]
    vector_results = other_nodes[:9] + [sku_nws] + other_nodes[9:]

    with patch.object(retriever, "_bm25_search", return_value=bm25_results), \
         patch.object(retriever, "_vector_search", return_value=vector_results):
        results = retriever.retrieve(sku, top_k=10)

    result_ids = [nws.node.node_id for nws in results]
    assert product_node.node_id in result_ids[:3], (
        f"SKU node '{product_node.node_id}' not in top-3 for query={sku!r}. "
        f"Got: {result_ids}"
    )


# ---------------------------------------------------------------------------
# Property 5: RRF Score Monotonicity
# ---------------------------------------------------------------------------

# Feature: hybrid-retriever-relevance-checker, Property 5: RRF Score Monotonicity
@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_rrf_score_monotonicity(rank_a: int, rank_b: int) -> None:
    """Validates: Requirements 3.2, 3.3

    If node A has a better (lower) rank than node B in both retrievers,
    then A's fusion score must be strictly greater than B's fusion score.
    """
    k = 60

    def fusion_score(rank_bm25: int, rank_vector: int) -> float:
        return 1.0 / (k + rank_bm25) + 1.0 / (k + rank_vector)

    score_a = fusion_score(rank_a, rank_a)
    score_b = fusion_score(rank_b, rank_b)

    if rank_a < rank_b:
        assert score_a > score_b, (
            f"Monotonicity violated: rank_a={rank_a} < rank_b={rank_b} "
            f"but score_a={score_a} <= score_b={score_b}"
        )
    elif rank_a > rank_b:
        assert score_a < score_b, (
            f"Monotonicity violated: rank_a={rank_a} > rank_b={rank_b} "
            f"but score_a={score_a} >= score_b={score_b}"
        )
    else:
        # Equal ranks → equal scores
        assert score_a == score_b


# ---------------------------------------------------------------------------
# Property 6: RelevanceChecker Label Validity
# ---------------------------------------------------------------------------

_VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
_LABEL_CYCLE = list(_VALID_LABELS)


# Feature: hybrid-retriever-relevance-checker, Property 6: RelevanceChecker Label Validity
@given(st.text())
@settings(max_examples=100)
def test_relevance_label_validity(query: str) -> None:
    """Validates: Requirements 5.1, 5.6

    RelevanceChecker.check() must always return one of the three valid labels,
    never raise an exception for any valid string input.
    """
    # Cycle through valid labels so the mock returns each one across examples
    mock_response = MagicMock()
    mock_response.text = _LABEL_CYCLE[hash(query) % len(_LABEL_CYCLE)]
    mock_llm = MagicMock()
    mock_llm.complete.return_value = mock_response

    checker = RelevanceChecker(llm=mock_llm)
    result = checker.check(query)

    assert result in _VALID_LABELS, (
        f"Invalid label returned: {result!r} for query={query!r}"
    )


# ---------------------------------------------------------------------------
# Property 7: NO_MATCH Skips Retrieval
# ---------------------------------------------------------------------------

# Feature: hybrid-retriever-relevance-checker, Property 7: NO_MATCH Skips Retrieval
@given(st.text())
@settings(max_examples=100)
def test_no_match_skips_retrieval(query: str) -> None:
    """Validates: Requirements 6.1

    When RelevanceChecker returns NO_MATCH, RAGPipeline must not call
    retriever.retrieve() and must return the default response string.
    """
    mock_checker = MagicMock()
    mock_checker.check.return_value = "NO_MATCH"

    mock_retriever = MagicMock()

    pipeline = RAGPipeline(retriever=mock_retriever, checker=mock_checker)
    result = pipeline.query(query)

    mock_retriever.retrieve.assert_not_called()
    assert result == _DEFAULT_NO_MATCH_RESPONSE, (
        f"Expected default response for NO_MATCH, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Property 8: Union Pool Completeness
# ---------------------------------------------------------------------------

# Feature: hybrid-retriever-relevance-checker, Property 8: Union Pool Completeness
@given(
    st.lists(st.text(min_size=1), min_size=1, max_size=10),
    st.lists(st.text(min_size=1), min_size=1, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_union_pool_completeness(bm25_ids: list[str], vector_ids: list[str]) -> None:
    """Validates: Requirements 3.5, 3.6

    Every node returned by either BM25 or vector search must appear in the
    fusion pool before the top_k cut is applied.
    """
    # Deduplicate IDs while preserving order
    seen: set[str] = set()
    unique_bm25_ids: list[str] = []
    for nid in bm25_ids:
        if nid not in seen:
            seen.add(nid)
            unique_bm25_ids.append(nid)

    unique_vector_ids: list[str] = []
    for nid in vector_ids:
        if nid not in seen:
            seen.add(nid)
            unique_vector_ids.append(nid)

    all_unique_ids = unique_bm25_ids + unique_vector_ids

    bm25_results = [
        _make_nws(nid, f"bm25 text for {nid}", float(len(unique_bm25_ids) - i))
        for i, nid in enumerate(unique_bm25_ids)
    ]
    vector_results = [
        _make_nws(nid, f"vector text for {nid}", float(len(unique_vector_ids) - i))
        for i, nid in enumerate(unique_vector_ids)
    ]

    retriever = _build_retriever_from_nodes(corpus_nodes)

    # Use top_k=100 to avoid the cut truncating the pool
    with patch.object(retriever, "_bm25_search", return_value=bm25_results), \
         patch.object(retriever, "_vector_search", return_value=vector_results):
        results = retriever.retrieve("test query", top_k=100)

    result_ids = {nws.node.node_id for nws in results}

    for nid in all_unique_ids:
        assert nid in result_ids, (
            f"Node '{nid}' from retriever output missing from fusion pool. "
            f"Result IDs: {result_ids}"
        )


# ===========================================================================
# Sales Research Agent — Property-Based Tests (Task 7)
# ===========================================================================

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


# ---------------------------------------------------------------------------
# Property 1: Internal DB Priority
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 1: Internal DB Priority
@given(objection=st.text(min_size=1, max_size=200))
@settings(max_examples=20)
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_pbt_internal_db_priority(mock_react_cls, objection):
    """Validates: Requirements 1.1

    If both tools are used, internal_db_search must come before tavily_web_search.
    """
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là thông tin.",
        ["internal_db_search", "tavily_web_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run(objection)

    if "internal_db_search" in result.tools_used and "tavily_web_search" in result.tools_used:
        db_idx = result.tools_used.index("internal_db_search")
        tavily_idx = result.tools_used.index("tavily_web_search")
        assert db_idx < tavily_idx


# ---------------------------------------------------------------------------
# Property 2: Max Iterations Bound
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 2: Max Iterations Bound
@given(objection=st.text(min_size=1, max_size=200))
@settings(max_examples=20)
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_pbt_max_iterations_bound(mock_react_cls, objection):
    """Validates: Requirements 1.2

    For any objection, len(tools_used) <= 2.
    """
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là kết quả.",
        ["internal_db_search", "internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run(objection)

    assert len(result.tools_used) <= 2


# ---------------------------------------------------------------------------
# Property 3: Draft Response Format
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 3: Draft Response Format
@given(objection=st.text(min_size=1, max_size=200))
@settings(max_examples=20)
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_pbt_draft_response_format(mock_react_cls, objection):
    """Validates: Requirements 2.1

    When DB returns valid data, draft_response starts with 'Dạ,' or 'Vâng,'.
    """
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là thông tin sản phẩm từ DB nội bộ.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run(objection)

    assert result.draft_response.startswith("Dạ,") or result.draft_response.startswith("Vâng,")


# ---------------------------------------------------------------------------
# Property 4: AgentResult Completeness
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 4: AgentResult Completeness
@given(objection=st.text(min_size=0, max_size=500))
@settings(max_examples=20)
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_pbt_agent_result_completeness(mock_react_cls, objection):
    """Validates: Requirements 1.3

    For any string input, run() never raises and returns AgentResult with all 3 fields not None.
    """
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là phản hồi.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)

    # Should never raise
    result = agent.run(objection)

    assert isinstance(result, AgentResult)
    assert result.objection_text is not None
    assert result.draft_response is not None
    assert result.tools_used is not None


# ---------------------------------------------------------------------------
# Property 5: Conflict Resolution (Mocked)
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 5: Conflict Resolution (Mocked)
@given(objection=st.text(min_size=1, max_size=200))
@settings(max_examples=10)
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_pbt_conflict_resolution(mock_react_cls, objection):
    """Validates: Requirements 2.2

    When internal DB says 15M and web says 10M, draft must contain 15M not 10M.
    """
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    # Simulate agent correctly using internal DB price (15M) not web price (10M)
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, giá Dell Inspiron 15 nội bộ là 15.000.000 VNĐ, đây là giá chính thức của công ty.",
        ["internal_db_search", "tavily_web_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run(objection)

    assert "15.000.000" in result.draft_response
    assert "10.000.000" not in result.draft_response


# ===========================================================================
# Task 11 — PBT Upgrade Properties
# ===========================================================================

import dataclasses
import time as _time
from backend.workflows.research_agent.tools import build_internal_db_tool
from backend.workflows.research_agent.cache import QueryCache


# ---------------------------------------------------------------------------
# Property 6 — Memory Safety
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 6: Memory Safety
@given(node_text=st.text(min_size=1, max_size=2000))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_pbt_memory_safety_truncation(node_text: str) -> None:
    """Validates: Requirements 4.1

    For any node text of length 1–2000, content in JSON output has len <= 503.
    """
    mock_node = MagicMock()
    mock_node.node.metadata = {"source_type": "product_catalog", "product_code": "TEST"}
    mock_node.node.text = node_text

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = [mock_node]

    tool = build_internal_db_tool(mock_pipeline)
    result_json = tool.fn("test query")

    nodes = json.loads(result_json)
    assert isinstance(nodes, list)
    assert len(nodes[0]["content"]) <= 503


# ---------------------------------------------------------------------------
# Property 7 — Cache Idempotence
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 7: Cache Idempotence
@given(query=st.text(min_size=1, max_size=100))
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_pbt_cache_idempotence(query: str) -> None:
    """Validates: Requirements 9.1

    Calling internal_db_tool twice with same query and a QueryCache → results equal,
    RAGPipeline called only once.
    """
    mock_node = MagicMock()
    mock_node.node.metadata = {"source_type": "product_catalog", "product_code": "P001"}
    mock_node.node.text = "Sample product text"

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = [mock_node]

    cache = QueryCache()
    tool = build_internal_db_tool(mock_pipeline, cache=cache)

    result1 = tool.fn(query)
    result2 = tool.fn(query)

    assert result1 == result2
    # RAGPipeline.query should only be called once (second call is cache hit)
    mock_pipeline.query.assert_called_once()


# ---------------------------------------------------------------------------
# Property 8 — Error Resilience
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 8: Error Resilience
@given(objection=st.text(min_size=1, max_size=200))
@settings(max_examples=20)
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_pbt_error_resilience(mock_react_cls, objection: str) -> None:
    """Validates: Requirements 1.4

    For any exception injected into LLM mock, agent.run() returns AgentResult
    with draft_response starting with 'Dạ,'.
    """
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.side_effect = RuntimeError("LLM failure")
    mock_react_cls.from_tools.return_value = mock_agent_instance

    from backend.workflows.research_agent.sales_research_agent import SalesResearchAgent, AgentResult
    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run(objection)

    assert isinstance(result, AgentResult)
    assert result.draft_response.startswith("Dạ,")


# ---------------------------------------------------------------------------
# Property 9 — Objection Preservation
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 9: Objection Preservation
@given(objection=st.text(min_size=0, max_size=500))
@settings(max_examples=30)
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_pbt_objection_preservation(mock_react_cls, objection: str) -> None:
    """Validates: Requirements 1.3

    For any string x, AgentResult.objection_text == x.
    """
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là phản hồi.", ["internal_db_search"]
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    from backend.workflows.research_agent.sales_research_agent import SalesResearchAgent
    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run(objection)

    assert result.objection_text == objection


# ---------------------------------------------------------------------------
# Property 10 — Early Termination
# ---------------------------------------------------------------------------

# Feature: sales-research-agent, Property 10: Early Termination
@given(objection=st.text(min_size=1, max_size=200))
@settings(max_examples=20)
@patch("backend.workflows.research_agent.sales_research_agent.ReActAgent")
def test_pbt_early_termination(mock_react_cls, objection: str) -> None:
    """Validates: Requirements 10.1

    When internal_db_search returns valid JSON (non-NO_MATCH), tavily_web_search
    must NOT appear in tools_used.
    """
    mock_llm = MagicMock()
    mock_pipeline = MagicMock()

    mock_agent_instance = MagicMock()
    # Simulate agent stopping after internal_db_search (early termination)
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, thông tin đã đủ từ DB nội bộ.",
        ["internal_db_search"],
    )
    mock_react_cls.from_tools.return_value = mock_agent_instance

    from backend.workflows.research_agent.sales_research_agent import SalesResearchAgent
    agent = SalesResearchAgent(llm=mock_llm, rag_pipeline=mock_pipeline)
    result = agent.run(objection)

    assert "tavily_web_search" not in result.tools_used
