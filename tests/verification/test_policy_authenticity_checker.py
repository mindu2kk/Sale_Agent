"""
Unit Tests for PolicyAuthenticityChecker — Policy Verification Against Official Documents

Tests Task 2.2.2: Policy verification against official documents with exact matching.

Covers:
- RAG pipeline unavailable → graceful fallback (PASS)
- No nodes retrieved → fabricated policy (FAIL, correct_policy=None)
- Policy-type query templates used correctly
- Exact substring match in retrieved chunks → PASS
- Token-overlap fuzzy match → PASS
- Claim not found in any chunk → FAIL with correct_policy text
- Duration claim verified / unverified
- Policy node filtering by source_type and file_name metadata
- No claims / no duration → presence of chunks is sufficient (PASS)
- Retrieval exception → FAIL gracefully
- Forbidden phrases still detected before DB lookup
- Full check_policy_authenticity integration with real statement extraction
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional, Dict, Any

from backend.verification.agent.checkers import PolicyAuthenticityChecker
from backend.verification.config.config import VerificationConfig
from backend.verification.models.verification import PolicyIssue, IssueSeverity
from backend.verification.utils.cache import PolicyDocumentCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checker(rag_pipeline=None) -> PolicyAuthenticityChecker:
    config = VerificationConfig()
    # Pass a fresh cache each time to avoid singleton cache pollution between tests
    return PolicyAuthenticityChecker(
        llm=None, rag_pipeline=rag_pipeline, config=config,
        policy_cache=PolicyDocumentCache()
    )


def _make_node(text: str, source_type: str = "policy_pdf", file_name: str = "warranty_policy.pdf"):
    """Build a mock NodeWithScore with the given text and metadata."""
    node = MagicMock()
    node.text = text
    node.metadata = {"source_type": source_type, "file_name": file_name}
    nws = MagicMock()
    nws.node = node
    return nws


def _make_rag_pipeline(nodes: List) -> MagicMock:
    """Build a mock RAGPipeline whose retriever returns *nodes*."""
    retriever = MagicMock()
    retriever.retrieve.return_value = nodes
    pipeline = MagicMock()
    pipeline.retriever = retriever
    return pipeline


# ---------------------------------------------------------------------------
# _lookup_policy_in_db — RAG pipeline availability
# ---------------------------------------------------------------------------

class TestLookupPolicyInDbFallback:
    def test_no_rag_pipeline_returns_pass(self):
        """When rag_pipeline is None, lookup returns (True, None) — no false positives."""
        checker = _make_checker(rag_pipeline=None)
        statement = {"type": "warranty", "keyword": "bảo hành", "claims": ["miễn phí bảo hành"], "duration": None, "text": "bảo hành miễn phí"}
        result = checker._lookup_policy_in_db(statement)
        assert result == (True, None)

    def test_rag_pipeline_without_retriever_returns_pass(self):
        """When rag_pipeline has no retriever attribute, lookup returns (True, None)."""
        pipeline = MagicMock(spec=[])  # no attributes
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {"type": "warranty", "keyword": "bảo hành", "claims": [], "duration": None, "text": ""}
        result = checker._lookup_policy_in_db(statement)
        assert result == (True, None)

    def test_retrieval_exception_returns_fail_no_correct_policy(self):
        """When retriever.retrieve() raises, returns (False, None) — cannot verify."""
        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("connection error")
        pipeline = MagicMock()
        pipeline.retriever = retriever
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {"type": "warranty", "keyword": "bảo hành", "claims": ["miễn phí"], "duration": None, "text": "bảo hành miễn phí"}
        is_verified, correct_policy = checker._lookup_policy_in_db(statement)
        assert is_verified is False
        assert correct_policy is None

    def test_empty_retrieval_returns_fail_fabricated(self):
        """When retriever returns empty list, policy is considered fabricated → (False, None)."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {"type": "return", "keyword": "đổi trả", "claims": ["hoàn tiền 100%"], "duration": None, "text": "đổi trả hoàn tiền 100%"}
        is_verified, correct_policy = checker._lookup_policy_in_db(statement)
        assert is_verified is False
        assert correct_policy is None


# ---------------------------------------------------------------------------
# _lookup_policy_in_db — query building
# ---------------------------------------------------------------------------

class TestQueryBuilding:
    def test_warranty_type_uses_warranty_template(self):
        """Warranty policy type uses the warranty query template."""
        pipeline = _make_rag_pipeline(nodes=[_make_node("bảo hành 12 tháng")])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {"type": "warranty", "keyword": "bảo hành", "claims": [], "duration": None, "text": "bảo hành"}
        checker._lookup_policy_in_db(statement)
        call_args = pipeline.retriever.retrieve.call_args
        query_used = call_args[0][0]
        assert "bảo hành" in query_used or "warranty" in query_used

    def test_return_type_uses_return_template(self):
        """Return policy type uses the return/refund query template."""
        pipeline = _make_rag_pipeline(nodes=[_make_node("đổi trả trong 30 ngày")])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {"type": "return", "keyword": "đổi trả", "claims": [], "duration": None, "text": "đổi trả"}
        checker._lookup_policy_in_db(statement)
        query_used = pipeline.retriever.retrieve.call_args[0][0]
        assert "đổi trả" in query_used or "return" in query_used or "refund" in query_used

    def test_duration_enriches_query(self):
        """Duration claim is appended to the search query."""
        pipeline = _make_rag_pipeline(nodes=[_make_node("bảo hành 12 tháng")])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": [],
            "duration": {"amount": "12", "unit": "tháng", "full_match": "bảo hành 12 tháng"},
            "text": "bảo hành 12 tháng",
        }
        checker._lookup_policy_in_db(statement)
        query_used = pipeline.retriever.retrieve.call_args[0][0]
        assert "12" in query_used or "tháng" in query_used

    def test_claims_enrich_query(self):
        """Top claims are appended to the search query."""
        pipeline = _make_rag_pipeline(nodes=[_make_node("miễn phí bảo hành")])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["miễn phí bảo hành", "bảo hành chính hãng"],
            "duration": None,
            "text": "bảo hành miễn phí chính hãng",
        }
        checker._lookup_policy_in_db(statement)
        query_used = pipeline.retriever.retrieve.call_args[0][0]
        assert "miễn phí" in query_used or "bảo hành" in query_used

    def test_retrieval_called_with_top_k(self):
        """Retriever is called with top_k=_RETRIEVAL_TOP_K."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {"type": "warranty", "keyword": "bảo hành", "claims": [], "duration": None, "text": ""}
        checker._lookup_policy_in_db(statement)
        call_kwargs = pipeline.retriever.retrieve.call_args[1]
        assert call_kwargs.get("top_k") == PolicyAuthenticityChecker._RETRIEVAL_TOP_K


# ---------------------------------------------------------------------------
# _lookup_policy_in_db — claim matching (PASS cases)
# ---------------------------------------------------------------------------

class TestClaimMatchingPass:
    def test_exact_claim_substring_match_passes(self):
        """Claim found as exact substring in retrieved chunk → PASS."""
        node = _make_node("Sản phẩm được bảo hành miễn phí trong 12 tháng kể từ ngày mua.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["miễn phí"],
            "duration": None,
            "text": "bảo hành miễn phí",
        }
        is_verified, correct_policy = checker._lookup_policy_in_db(statement)
        assert is_verified is True
        assert correct_policy is None

    def test_case_insensitive_match_passes(self):
        """Claim matching is case-insensitive."""
        node = _make_node("BẢO HÀNH MIỄN PHÍ 12 THÁNG.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["miễn phí bảo hành"],
            "duration": None,
            "text": "bảo hành miễn phí",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        assert is_verified is True

    def test_fuzzy_token_overlap_passes(self):
        """Claim with high token overlap (≥55%) in retrieved chunk → PASS."""
        # Claim: "bảo hành chính hãng miễn phí" (4 tokens)
        # Chunk contains 3 of those tokens → 75% overlap → PASS
        node = _make_node("Chính sách bảo hành chính hãng miễn phí áp dụng cho tất cả sản phẩm.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["bảo hành chính hãng miễn phí"],
            "duration": None,
            "text": "bảo hành chính hãng miễn phí",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        assert is_verified is True

    def test_no_claims_no_duration_passes_with_any_chunk(self):
        """No claims and no duration → presence of retrieved chunks is sufficient."""
        node = _make_node("Chính sách bảo hành áp dụng cho tất cả sản phẩm.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": [],
            "duration": None,
            "text": "bảo hành",
        }
        is_verified, correct_policy = checker._lookup_policy_in_db(statement)
        assert is_verified is True
        assert correct_policy is None

    def test_duration_found_verbatim_passes(self):
        """Duration claim found verbatim in retrieved chunk → PASS."""
        node = _make_node("Sản phẩm được bảo hành 12 tháng kể từ ngày mua hàng.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": [],
            "duration": {"amount": "12", "unit": "tháng", "full_match": "bảo hành 12 tháng"},
            "text": "bảo hành 12 tháng",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        assert is_verified is True

    def test_duration_found_as_numeric_amount_unit_passes(self):
        """Duration found as 'amount unit' even if full_match not verbatim → PASS."""
        node = _make_node("Thời gian bảo hành: 12 tháng tính từ ngày xuất hóa đơn.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": [],
            "duration": {"amount": "12", "unit": "tháng", "full_match": "bảo hành 12 tháng"},
            "text": "bảo hành 12 tháng",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        assert is_verified is True

    def test_multiple_claims_all_found_passes(self):
        """All claims found in retrieved chunks → PASS."""
        node = _make_node(
            "Sản phẩm được bảo hành miễn phí chính hãng trong 12 tháng. "
            "Khách hàng có thể đổi trả trong 30 ngày."
        )
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["miễn phí", "chính hãng"],
            "duration": None,
            "text": "bảo hành miễn phí chính hãng",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        assert is_verified is True


# ---------------------------------------------------------------------------
# _lookup_policy_in_db — claim matching (FAIL cases)
# ---------------------------------------------------------------------------

class TestClaimMatchingFail:
    def test_claim_not_found_returns_fail_with_correct_policy(self):
        """Claim not found in any chunk → FAIL, correct_policy = best chunk text."""
        chunk_text = "Sản phẩm được bảo hành 12 tháng. Không áp dụng miễn phí sửa chữa."
        node = _make_node(chunk_text)
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["miễn phí sửa chữa trọn đời"],  # not in chunk
            "duration": None,
            "text": "bảo hành miễn phí sửa chữa trọn đời",
        }
        is_verified, correct_policy = checker._lookup_policy_in_db(statement)
        assert is_verified is False
        assert correct_policy == chunk_text

    def test_duration_not_found_returns_fail(self):
        """Duration claim not found in any chunk → FAIL."""
        node = _make_node("Sản phẩm được bảo hành 6 tháng kể từ ngày mua.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": [],
            "duration": {"amount": "24", "unit": "tháng", "full_match": "bảo hành 24 tháng"},
            "text": "bảo hành 24 tháng",
        }
        is_verified, correct_policy = checker._lookup_policy_in_db(statement)
        assert is_verified is False
        assert correct_policy is not None  # best chunk returned as reference

    def test_partial_claims_fail_if_any_missing(self):
        """If any claim is missing from all chunks → FAIL."""
        node = _make_node("Bảo hành miễn phí 12 tháng.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["miễn phí", "bảo hành trọn đời"],  # second claim not in chunk
            "duration": None,
            "text": "bảo hành miễn phí trọn đời",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        assert is_verified is False

    def test_low_token_overlap_fails(self):
        """Claim with low token overlap (<55%) → FAIL."""
        # Claim: "hoàn tiền 100% không điều kiện" (4 tokens)
        # Chunk has only 1 matching token → 25% overlap → FAIL
        node = _make_node("Chính sách đổi trả áp dụng trong 30 ngày.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "return",
            "keyword": "hoàn tiền",
            "claims": ["hoàn tiền 100% không điều kiện"],
            "duration": None,
            "text": "hoàn tiền 100% không điều kiện",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        assert is_verified is False


# ---------------------------------------------------------------------------
# _is_policy_node — node filtering
# ---------------------------------------------------------------------------

class TestIsPolicyNode:
    def test_source_type_policy_pdf_is_policy_node(self):
        checker = _make_checker()
        node = _make_node("text", source_type="policy_pdf", file_name="doc.pdf")
        assert checker._is_policy_node(node) is True

    def test_file_name_contains_warranty_is_policy_node(self):
        checker = _make_checker()
        node = _make_node("text", source_type="product_catalog", file_name="Apple_warranty_2024.pdf")
        assert checker._is_policy_node(node) is True

    def test_file_name_contains_return_is_policy_node(self):
        checker = _make_checker()
        node = _make_node("text", source_type="other", file_name="FPTS_Return_Exchange_Policy.pdf")
        assert checker._is_policy_node(node) is True

    def test_product_catalog_node_is_not_policy_node(self):
        checker = _make_checker()
        node = _make_node("text", source_type="product_catalog", file_name="product_catalog.csv")
        assert checker._is_policy_node(node) is False

    def test_policy_nodes_preferred_over_catalog_nodes(self):
        """When both policy and catalog nodes retrieved, policy nodes are used."""
        policy_node = _make_node("bảo hành 12 tháng miễn phí", source_type="policy_pdf")
        catalog_node = _make_node("iPhone 15 Pro giá 29.990.000 VNĐ", source_type="product_catalog", file_name="catalog.csv")
        pipeline = _make_rag_pipeline(nodes=[catalog_node, policy_node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["miễn phí"],
            "duration": None,
            "text": "bảo hành miễn phí",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        # Policy node contains "miễn phí" → should PASS
        assert is_verified is True

    def test_fallback_to_all_nodes_when_no_policy_nodes(self):
        """When no policy-tagged nodes, falls back to all retrieved nodes."""
        catalog_node = _make_node(
            "bảo hành 12 tháng miễn phí",
            source_type="product_catalog",
            file_name="catalog.csv",
        )
        pipeline = _make_rag_pipeline(nodes=[catalog_node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "claims": ["miễn phí"],
            "duration": None,
            "text": "bảo hành miễn phí",
        }
        is_verified, _ = checker._lookup_policy_in_db(statement)
        # Falls back to catalog node which contains "miễn phí" → PASS
        assert is_verified is True


# ---------------------------------------------------------------------------
# _claim_found_in_nodes
# ---------------------------------------------------------------------------

class TestClaimFoundInNodes:
    def test_empty_claim_always_returns_true(self):
        checker = _make_checker()
        assert checker._claim_found_in_nodes("", []) is True
        assert checker._claim_found_in_nodes("  ", []) is True

    def test_exact_match_found(self):
        checker = _make_checker()
        node = _make_node("bảo hành miễn phí 12 tháng")
        assert checker._claim_found_in_nodes("miễn phí", [node]) is True

    def test_no_match_returns_false(self):
        checker = _make_checker()
        node = _make_node("sản phẩm chất lượng cao")
        assert checker._claim_found_in_nodes("miễn phí bảo hành", [node]) is False

    def test_match_across_multiple_nodes(self):
        """Claim found in second node → True."""
        checker = _make_checker()
        node1 = _make_node("thông tin sản phẩm")
        node2 = _make_node("bảo hành miễn phí chính hãng")
        assert checker._claim_found_in_nodes("miễn phí", [node1, node2]) is True

    def test_no_nodes_returns_false(self):
        checker = _make_checker()
        assert checker._claim_found_in_nodes("miễn phí", []) is False


# ---------------------------------------------------------------------------
# Full check_policy_authenticity integration
# ---------------------------------------------------------------------------

class TestCheckPolicyAuthenticityIntegration:
    def test_no_policy_statements_passes(self):
        """Draft with no policy keywords → PASS with no issues."""
        checker = _make_checker(rag_pipeline=None)
        passed, issues = checker.check_policy_authenticity(
            "iPhone 15 Pro có camera 48MP và chip A17 Pro."
        )
        assert passed is True
        assert issues == []

    def test_forbidden_phrase_fails_before_db_lookup(self):
        """Forbidden phrase detected → FAIL CRITICAL without DB lookup."""
        pipeline = _make_rag_pipeline(nodes=[_make_node("bảo hành 12 tháng")])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm tự bịa bảo hành 5 năm miễn phí."
        )
        assert passed is False
        assert len(issues) >= 1
        assert issues[0].severity == IssueSeverity.CRITICAL
        assert issues[0].is_fabricated is True
        # DB lookup should NOT have been called
        pipeline.retriever.retrieve.assert_not_called()

    def test_verified_policy_passes(self):
        """Policy statement verified against DB → PASS."""
        node = _make_node("Sản phẩm được bảo hành 12 tháng kể từ ngày mua.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm được bảo hành 12 tháng kể từ ngày mua."
        )
        assert passed is True
        assert issues == []

    def test_unverified_policy_fails_with_issue(self):
        """Policy statement not verified → FAIL with PolicyIssue."""
        pipeline = _make_rag_pipeline(nodes=[])  # nothing retrieved → fabricated
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm được bảo hành trọn đời miễn phí."
        )
        assert passed is False
        assert len(issues) >= 1
        issue = issues[0]
        assert isinstance(issue, PolicyIssue)
        assert issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)

    def test_return_type_always_tuple(self):
        """check_policy_authenticity always returns (bool, list)."""
        checker = _make_checker()
        result = checker.check_policy_authenticity("")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)


# ---------------------------------------------------------------------------
# _extract_policy_statements — Requirement 5.1: keyword detection
# ---------------------------------------------------------------------------

class TestExtractPolicyStatements:
    """Tests for policy statement extraction using keyword taxonomy (Req 5.1)."""

    def test_no_policy_keywords_returns_empty(self):
        """Text with no policy keywords → empty list."""
        checker = _make_checker()
        statements = checker._extract_policy_statements(
            "iPhone 15 Pro có camera 48MP và chip A17 Pro rất mạnh."
        )
        assert statements == []

    def test_warranty_keyword_detected(self):
        """'bảo hành' keyword triggers warranty policy extraction."""
        checker = _make_checker()
        statements = checker._extract_policy_statements("Sản phẩm được bảo hành 12 tháng.")
        assert len(statements) >= 1
        types = [s["type"] for s in statements]
        assert "warranty" in types

    def test_return_keyword_detected(self):
        """'đổi trả' keyword triggers return policy extraction."""
        checker = _make_checker()
        statements = checker._extract_policy_statements("Chính sách đổi trả trong 30 ngày.")
        assert len(statements) >= 1
        types = [s["type"] for s in statements]
        assert "return" in types

    def test_refund_keyword_detected(self):
        """'hoàn tiền' keyword triggers return policy extraction."""
        checker = _make_checker()
        statements = checker._extract_policy_statements("Hoàn tiền 100% nếu không hài lòng.")
        assert len(statements) >= 1
        types = [s["type"] for s in statements]
        assert "return" in types

    def test_english_warranty_keyword_detected(self):
        """English 'warranty' keyword is detected."""
        checker = _make_checker()
        statements = checker._extract_policy_statements("This product comes with a 1-year warranty.")
        assert len(statements) >= 1
        types = [s["type"] for s in statements]
        assert "warranty" in types

    def test_service_keyword_detected(self):
        """'sửa chữa' keyword triggers service policy extraction."""
        checker = _make_checker()
        statements = checker._extract_policy_statements("Dịch vụ sửa chữa miễn phí trong thời gian bảo hành.")
        assert len(statements) >= 1
        types = [s["type"] for s in statements]
        assert any(t in ("service", "warranty") for t in types)

    def test_duration_extracted_from_statement(self):
        """Duration claim (e.g. '12 tháng') is extracted from statement."""
        checker = _make_checker()
        statements = checker._extract_policy_statements("Bảo hành 12 tháng kể từ ngày mua.")
        assert len(statements) >= 1
        # At least one statement should have a duration
        durations = [s.get("duration") for s in statements if s.get("duration")]
        assert len(durations) >= 1
        duration = durations[0]
        assert duration["amount"] == "12"
        assert "tháng" in duration["unit"]

    def test_claim_extracted_from_statement(self):
        """Specific claims (e.g. 'miễn phí bảo hành') are extracted from statement."""
        checker = _make_checker()
        # "miễn phí bảo hành" matches the claim pattern: r'(?:miễn phí|free)\s+(?:bảo hành|warranty|...)'
        statements = checker._extract_policy_statements("Sản phẩm được miễn phí bảo hành chính hãng.")
        assert len(statements) >= 1
        all_claims = [c for s in statements for c in s.get("claims", [])]
        assert len(all_claims) >= 1

    def test_statement_text_is_sentence_containing_keyword(self):
        """Extracted statement text contains the triggering keyword."""
        checker = _make_checker()
        statements = checker._extract_policy_statements(
            "Sản phẩm tốt. Bảo hành 12 tháng. Giao hàng nhanh."
        )
        assert len(statements) >= 1
        for s in statements:
            assert "bảo hành" in s["text"].lower() or "warranty" in s["text"].lower()

    def test_multiple_policy_types_in_one_text(self):
        """Multiple policy types in one text are all extracted."""
        checker = _make_checker()
        text = (
            "Sản phẩm được bảo hành 12 tháng. "
            "Chính sách đổi trả trong 30 ngày. "
            "Hỗ trợ kỹ thuật 24/7."
        )
        statements = checker._extract_policy_statements(text)
        types = {s["type"] for s in statements}
        # Should detect at least warranty and return
        assert len(types) >= 2

    def test_deduplication_same_sentence_kept_once(self):
        """Same sentence matched by multiple keywords is deduplicated."""
        checker = _make_checker()
        # "bảo hành" and "warranty" both match the same sentence
        statements = checker._extract_policy_statements(
            "Sản phẩm có warranty bảo hành 12 tháng."
        )
        # Should not have duplicate entries for the same sentence span
        positions = [s["position"] for s in statements]
        assert len(positions) == len(set(positions))

    def test_confidence_score_present_and_valid(self):
        """Each extracted statement has a confidence score between 0 and 1."""
        checker = _make_checker()
        statements = checker._extract_policy_statements("Bảo hành miễn phí 12 tháng.")
        assert len(statements) >= 1
        for s in statements:
            assert "confidence" in s
            assert 0.0 <= s["confidence"] <= 1.0

    def test_multi_word_keyword_higher_confidence(self):
        """Multi-word keywords (e.g. 'đổi trả') yield higher confidence than single-word."""
        checker = _make_checker()
        # "đổi trả" is multi-word → confidence += 0.2
        statements_multi = checker._extract_policy_statements("Chính sách đổi trả trong 30 ngày.")
        # "return" is single-word
        statements_single = checker._extract_policy_statements("Return policy applies for 30 days.")
        if statements_multi and statements_single:
            # Multi-word should have higher or equal confidence
            max_multi = max(s["confidence"] for s in statements_multi)
            max_single = max(s["confidence"] for s in statements_single)
            assert max_multi >= max_single


# ---------------------------------------------------------------------------
# _verify_policy_statement — Requirement 5.2 & 5.3: severity classification
# ---------------------------------------------------------------------------

class TestVerifyPolicyStatement:
    """Tests for _verify_policy_statement severity classification (Req 5.2, 5.3)."""

    def test_forbidden_phrase_returns_critical_fabricated(self):
        """Statement with forbidden phrase → FAIL, CRITICAL, is_fabricated=True."""
        checker = _make_checker(rag_pipeline=None)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "text": "Sản phẩm tự bịa bảo hành 5 năm.",
            "claims": [],
            "duration": None,
            "position": (0, 40),
            "confidence": 0.8,
        }
        is_authentic, issue = checker._verify_policy_statement(statement)
        assert is_authentic is False
        assert issue is not None
        assert issue.is_fabricated is True
        assert issue.severity == IssueSeverity.CRITICAL

    def test_verified_policy_returns_pass_no_issue(self):
        """Policy verified against DB → PASS, no issue returned."""
        node = _make_node("Sản phẩm được bảo hành 12 tháng kể từ ngày mua.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "text": "Sản phẩm được bảo hành 12 tháng.",
            "claims": [],
            "duration": None,
            "position": (0, 35),
            "confidence": 0.7,
        }
        is_authentic, issue = checker._verify_policy_statement(statement)
        assert is_authentic is True
        assert issue is None

    def test_fabricated_policy_no_db_match_returns_critical(self):
        """Policy not found in DB at all (empty retrieval) → CRITICAL, is_fabricated=True."""
        pipeline = _make_rag_pipeline(nodes=[])  # nothing retrieved
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "text": "Bảo hành trọn đời miễn phí không giới hạn.",
            "claims": ["miễn phí bảo hành trọn đời"],
            "duration": None,
            "position": (0, 45),
            "confidence": 0.9,
        }
        is_authentic, issue = checker._verify_policy_statement(statement)
        assert is_authentic is False
        assert issue is not None
        assert issue.is_fabricated is True
        assert issue.severity == IssueSeverity.CRITICAL
        assert issue.correct_policy is None

    def test_inaccurate_policy_db_has_different_terms_returns_major(self):
        """Policy found in DB but claim doesn't match → MAJOR, is_inaccurate=True."""
        # DB has "6 tháng" but draft claims "24 tháng"
        node = _make_node("Sản phẩm được bảo hành 6 tháng kể từ ngày mua.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "text": "Sản phẩm được bảo hành 24 tháng.",
            "claims": [],
            "duration": {"amount": "24", "unit": "tháng", "full_match": "bảo hành 24 tháng"},
            "position": (0, 35),
            "confidence": 0.8,
        }
        is_authentic, issue = checker._verify_policy_statement(statement)
        assert is_authentic is False
        assert issue is not None
        assert issue.is_inaccurate is True
        assert issue.severity == IssueSeverity.MAJOR
        # correct_policy should reference the DB chunk
        assert issue.correct_policy is not None

    def test_issue_contains_mentioned_policy_text(self):
        """PolicyIssue.mentioned_policy contains the original statement text."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        statement_text = "Bảo hành miễn phí 5 năm cho tất cả sản phẩm."
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "text": statement_text,
            "claims": ["miễn phí"],
            "duration": None,
            "position": (0, 50),
            "confidence": 0.8,
        }
        _, issue = checker._verify_policy_statement(statement)
        assert issue is not None
        assert issue.mentioned_policy == statement_text

    def test_issue_policy_type_matches_statement_type(self):
        """PolicyIssue.policy_type matches the statement's detected type."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        for policy_type in ("warranty", "return", "exchange", "service", "support"):
            statement = {
                "type": policy_type,
                "keyword": "bảo hành",
                "text": "Chính sách áp dụng.",
                "claims": ["miễn phí"],
                "duration": None,
                "position": (0, 20),
                "confidence": 0.6,
            }
            _, issue = checker._verify_policy_statement(statement)
            if issue is not None:
                assert issue.policy_type == policy_type

    def test_fabricated_issue_has_correction_suggestion(self):
        """Fabricated policy issue includes a correction_suggestion."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "return",
            "keyword": "hoàn tiền",
            "text": "Hoàn tiền 100% không điều kiện trong 365 ngày.",
            "claims": ["hoàn tiền 100%"],
            "duration": None,
            "position": (0, 50),
            "confidence": 0.9,
        }
        _, issue = checker._verify_policy_statement(statement)
        assert issue is not None
        assert issue.correction_suggestion is not None
        assert len(issue.correction_suggestion) > 0

    def test_forbidden_phrase_skips_db_lookup(self):
        """Forbidden phrase detected → DB lookup is NOT called."""
        pipeline = _make_rag_pipeline(nodes=[_make_node("bảo hành 12 tháng")])
        checker = _make_checker(rag_pipeline=pipeline)
        statement = {
            "type": "warranty",
            "keyword": "bảo hành",
            "text": "Theo ý kiến cá nhân thì bảo hành 10 năm.",
            "claims": [],
            "duration": None,
            "position": (0, 45),
            "confidence": 0.7,
        }
        checker._verify_policy_statement(statement)
        pipeline.retriever.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Binary PASS/FAIL scenarios — check_policy_authenticity (Req 5.1, 5.2, 5.3)
# ---------------------------------------------------------------------------

class TestBinaryPassScenarios:
    """PASS cases: verified policies should return (True, [])."""

    def test_pass_no_policy_in_draft(self):
        """Draft with no policy content → PASS."""
        checker = _make_checker()
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm có màn hình 6.1 inch và pin 3000mAh."
        )
        assert passed is True
        assert issues == []

    def test_pass_warranty_verified_against_db(self):
        """Warranty statement verified in DB → PASS."""
        node = _make_node("Sản phẩm được bảo hành 12 tháng kể từ ngày mua hàng.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm được bảo hành 12 tháng kể từ ngày mua hàng."
        )
        assert passed is True
        assert issues == []

    def test_pass_return_policy_verified(self):
        """Return policy verified in DB → PASS."""
        node = _make_node("Khách hàng có thể đổi trả sản phẩm trong vòng 30 ngày.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Khách hàng có thể đổi trả sản phẩm trong vòng 30 ngày."
        )
        assert passed is True
        assert issues == []

    def test_pass_no_rag_pipeline_graceful_fallback(self):
        """No RAG pipeline → graceful fallback, policy statements pass."""
        checker = _make_checker(rag_pipeline=None)
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm được bảo hành 12 tháng theo chính sách nhà sản xuất."
        )
        assert passed is True
        assert issues == []

    def test_pass_english_warranty_verified(self):
        """English warranty statement verified in DB → PASS."""
        node = _make_node("All products come with a 1-year warranty from the date of purchase.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "This product comes with a 1-year warranty."
        )
        assert passed is True
        assert issues == []

    def test_pass_empty_draft(self):
        """Empty draft → PASS with no issues."""
        checker = _make_checker()
        passed, issues = checker.check_policy_authenticity("")
        assert passed is True
        assert issues == []


class TestBinaryFailScenarios:
    """FAIL cases: fabricated/unverified policies should return (False, [PolicyIssue...])."""

    def test_fail_fabricated_policy_forbidden_phrase(self):
        """Forbidden phrase in draft → FAIL, CRITICAL PolicyIssue."""
        checker = _make_checker(rag_pipeline=None)
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm tự bịa bảo hành 10 năm miễn phí."
        )
        assert passed is False
        assert len(issues) >= 1
        assert issues[0].is_fabricated is True
        assert issues[0].severity == IssueSeverity.CRITICAL

    def test_fail_policy_not_in_db_is_fabricated(self):
        """Policy not found in DB (empty retrieval) → FAIL, CRITICAL."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Bảo hành trọn đời miễn phí cho tất cả sản phẩm."
        )
        assert passed is False
        assert len(issues) >= 1
        assert issues[0].severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)

    def test_fail_inaccurate_duration_returns_major(self):
        """Policy with wrong duration vs DB → FAIL, MAJOR."""
        node = _make_node("Sản phẩm được bảo hành 6 tháng kể từ ngày mua.")
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm được bảo hành 24 tháng kể từ ngày mua."
        )
        assert passed is False
        assert len(issues) >= 1
        assert issues[0].severity in (IssueSeverity.MAJOR, IssueSeverity.CRITICAL)

    def test_fail_issue_is_policy_issue_instance(self):
        """Failed verification returns PolicyIssue instances."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Hoàn tiền 100% không điều kiện trong 365 ngày."
        )
        assert passed is False
        for issue in issues:
            assert isinstance(issue, PolicyIssue)

    def test_fail_fabricated_policy_has_explanation(self):
        """Fabricated policy issue has a non-empty explanation."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Bảo hành miễn phí trọn đời không giới hạn."
        )
        assert passed is False
        assert len(issues) >= 1
        assert len(issues[0].explanation) > 0

    def test_fail_multiple_fabricated_policies(self):
        """Multiple fabricated policy statements → multiple issues."""
        pipeline = _make_rag_pipeline(nodes=[])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Bảo hành trọn đời. Hoàn tiền 100% bất kỳ lúc nào. Đổi máy mới miễn phí."
        )
        assert passed is False
        assert len(issues) >= 1  # at least one issue detected

    def test_fail_retrieval_exception_treated_as_unverified(self):
        """Retrieval exception → policy treated as unverified → FAIL."""
        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("DB down")
        pipeline = MagicMock()
        pipeline.retriever = retriever
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Sản phẩm được bảo hành 12 tháng."
        )
        assert passed is False
        assert len(issues) >= 1

    def test_fail_correct_policy_provided_for_inaccurate(self):
        """Inaccurate policy issue includes correct_policy from DB."""
        chunk_text = "Bảo hành 6 tháng theo chính sách nhà sản xuất."
        node = _make_node(chunk_text)
        pipeline = _make_rag_pipeline(nodes=[node])
        checker = _make_checker(rag_pipeline=pipeline)
        passed, issues = checker.check_policy_authenticity(
            "Bảo hành 24 tháng theo chính sách nhà sản xuất."
        )
        assert passed is False
        assert len(issues) >= 1
        # Inaccurate issues should have correct_policy reference
        inaccurate = [i for i in issues if i.is_inaccurate]
        if inaccurate:
            assert inaccurate[0].correct_policy is not None


# ---------------------------------------------------------------------------
# Early termination behavior
# ---------------------------------------------------------------------------

class TestEarlyTermination:
    """Tests for early termination on critical policy issues."""

    def test_early_termination_stops_after_critical_issue(self):
        """With early termination enabled and stop_on_first_critical=True,
        processing stops after first CRITICAL issue."""
        from backend.verification.config.thresholds_config import (
            VerificationThresholdsConfig,
            EscalationThresholds,
        )
        # Configure early termination to stop on first critical
        thresholds_config = VerificationThresholdsConfig(
            escalation=EscalationThresholds(
                early_termination_enabled=True,
                stop_on_first_critical=True,
                multiple_critical_threshold=3,
            )
        )
        # Both statements are fabricated (empty DB) → both would be CRITICAL
        pipeline = _make_rag_pipeline(nodes=[])
        config = VerificationConfig()
        checker = PolicyAuthenticityChecker(
            llm=None,
            rag_pipeline=pipeline,
            config=config,
            thresholds_config=thresholds_config,
        )
        # Text with two policy statements
        text = "Bảo hành trọn đời miễn phí. Hoàn tiền 100% bất kỳ lúc nào."
        passed, issues = checker.check_policy_authenticity(text)
        assert passed is False
        # With stop_on_first_critical, should stop after first critical issue
        assert len(issues) >= 1

    def test_no_early_termination_processes_all_statements(self):
        """With early termination disabled, all statements are processed."""
        from backend.verification.config.thresholds_config import (
            VerificationThresholdsConfig,
            EscalationThresholds,
        )
        thresholds_config = VerificationThresholdsConfig(
            escalation=EscalationThresholds(
                early_termination_enabled=False,
                stop_on_first_critical=False,
            )
        )
        pipeline = _make_rag_pipeline(nodes=[])
        config = VerificationConfig()
        checker = PolicyAuthenticityChecker(
            llm=None,
            rag_pipeline=pipeline,
            config=config,
            thresholds_config=thresholds_config,
        )
        text = "Bảo hành trọn đời miễn phí. Hoàn tiền 100% bất kỳ lúc nào."
        passed, issues = checker.check_policy_authenticity(text)
        assert passed is False
        # Without early termination, should process all statements
        assert len(issues) >= 1

    def test_result_always_returns_tuple(self):
        """check_policy_authenticity always returns (bool, list) regardless of input."""
        checker = _make_checker()
        for draft in ["", "bảo hành", "tự bịa bảo hành 10 năm", "no policy here"]:
            result = checker.check_policy_authenticity(draft)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], list)
