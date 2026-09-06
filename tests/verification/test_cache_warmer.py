"""
Tests for async cache warming strategies (Task 5.1.4).

Covers:
- warm_prompt_cache: pre-renders prompt templates into PromptTemplateCache
- warm_product_price_cache: pre-loads product queries into ProductPriceLookupCache
- warm_policy_cache: pre-loads policy queries into PolicyDocumentCache
- warm_all_caches: concurrent composite warmer
- WarmingResult / AllCachesWarmingResult dataclasses
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.verification.utils.cache import (
    PolicyDocumentCache,
    ProductPriceLookupCache,
)
from backend.verification.utils.cache_warmer import (
    AllCachesWarmingResult,
    WarmingResult,
    _POLICY_WARM_QUERIES,
    _PRODUCT_WARM_QUERIES,
    _PROMPT_WARM_PATTERNS,
    warm_all_caches,
    warm_policy_cache,
    warm_product_price_cache,
    warm_prompt_cache,
)
from backend.verification.utils.product_matcher import ProductMatch, ProductMatcher
from backend.verification.utils.prompt_cache import PromptTemplateCache


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_prompt_manager(fail_templates: Optional[list] = None) -> MagicMock:
    """Return a mock CachedPromptTemplates that renders successfully by default."""
    fail_templates = fail_templates or []
    manager = MagicMock()

    def _render(name: str, **kwargs: Any) -> str:
        if name in fail_templates:
            raise ValueError(f"Simulated render failure for {name}")
        return f"rendered:{name}"

    manager.render.side_effect = _render
    return manager


def _make_product_matcher(found: bool = True) -> MagicMock:
    """Return a mock ProductMatcher."""
    matcher = MagicMock(spec=ProductMatcher)
    if found:
        match = ProductMatch(
            product_name="iPhone 15",
            brand="Apple",
            sku="APL-IP15-128",
            price_raw="29.990.000 VNĐ",
            price_vnd=29_990_000.0,
            match_score=1.0,
            match_type="exact_name",
        )
        matcher.find_product.return_value = match
    else:
        matcher.find_product.return_value = None
    return matcher


def _make_rag_pipeline(has_retriever: bool = True, returns_nodes: bool = True) -> MagicMock:
    """Return a mock RAG pipeline."""
    pipeline = MagicMock()
    if not has_retriever:
        del pipeline.retriever
        return pipeline

    node = MagicMock()
    node.get_content.return_value = "Bảo hành 12 tháng chính hãng Apple."
    pipeline.retriever.retrieve.return_value = [node] if returns_nodes else []
    return pipeline


# ---------------------------------------------------------------------------
# WarmingResult tests
# ---------------------------------------------------------------------------

class TestWarmingResult:
    def test_success_rate_all_warmed(self):
        r = WarmingResult("test", entries_warmed=10, entries_failed=0, duration_seconds=0.1)
        assert r.success_rate == 1.0

    def test_success_rate_all_failed(self):
        r = WarmingResult("test", entries_warmed=0, entries_failed=5, duration_seconds=0.1)
        assert r.success_rate == 0.0

    def test_success_rate_partial(self):
        r = WarmingResult("test", entries_warmed=3, entries_failed=1, duration_seconds=0.1)
        assert r.success_rate == pytest.approx(0.75)

    def test_success_rate_empty(self):
        r = WarmingResult("test", entries_warmed=0, entries_failed=0, duration_seconds=0.0)
        assert r.success_rate == 0.0

    def test_str_representation(self):
        r = WarmingResult("my_cache", entries_warmed=5, entries_failed=1, duration_seconds=1.23)
        s = str(r)
        assert "my_cache" in s
        assert "5 warmed" in s
        assert "1 failed" in s


# ---------------------------------------------------------------------------
# warm_prompt_cache tests
# ---------------------------------------------------------------------------

class TestWarmPromptCache:
    @pytest.mark.asyncio
    async def test_warms_all_default_patterns(self):
        manager = _make_prompt_manager()
        result = await warm_prompt_cache(manager)

        assert result.cache_name == "prompt_template_cache"
        assert result.entries_warmed == len(_PROMPT_WARM_PATTERNS)
        assert result.entries_failed == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_render_called_for_each_pattern(self):
        manager = _make_prompt_manager()
        await warm_prompt_cache(manager)

        assert manager.render.call_count == len(_PROMPT_WARM_PATTERNS)

    @pytest.mark.asyncio
    async def test_partial_failure_counted(self):
        failing = [_PROMPT_WARM_PATTERNS[0][0]]  # first template fails
        manager = _make_prompt_manager(fail_templates=failing)

        result = await warm_prompt_cache(manager)

        assert result.entries_failed == 1
        assert result.entries_warmed == len(_PROMPT_WARM_PATTERNS) - 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_custom_patterns_used(self):
        manager = _make_prompt_manager()
        custom = [("price_accuracy_check", {"objection_text": "x", "draft_response": "y",
                                             "db_data": "z", "price_tolerance": "1",
                                             "critical_threshold": "30"})]
        result = await warm_prompt_cache(manager, patterns=custom)

        assert result.entries_warmed == 1
        assert manager.render.call_count == 1

    @pytest.mark.asyncio
    async def test_duration_is_positive(self):
        manager = _make_prompt_manager()
        result = await warm_prompt_cache(manager)
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# warm_product_price_cache tests
# ---------------------------------------------------------------------------

class TestWarmProductPriceCache:
    @pytest.mark.asyncio
    async def test_warms_all_default_queries(self):
        matcher = _make_product_matcher(found=True)
        cache = ProductPriceLookupCache()

        result = await warm_product_price_cache(matcher, price_cache=cache)

        assert result.cache_name == "product_price_cache"
        assert result.entries_warmed == len(_PRODUCT_WARM_QUERIES)
        assert result.entries_failed == 0

    @pytest.mark.asyncio
    async def test_cache_populated_after_warming(self):
        matcher = _make_product_matcher(found=True)
        cache = ProductPriceLookupCache()

        await warm_product_price_cache(matcher, price_cache=cache)

        for query in _PRODUCT_WARM_QUERIES:
            cached = cache.get(query)
            assert cached is not None, f"Expected cache hit for '{query}'"
            found, product_match = cached
            assert found is True
            assert product_match is not None

    @pytest.mark.asyncio
    async def test_negative_cache_when_product_not_found(self):
        matcher = _make_product_matcher(found=False)
        cache = ProductPriceLookupCache()

        result = await warm_product_price_cache(matcher, price_cache=cache)

        assert result.entries_warmed == len(_PRODUCT_WARM_QUERIES)
        for query in _PRODUCT_WARM_QUERIES:
            cached = cache.get(query)
            assert cached is not None
            found, product_match = cached
            assert found is False
            assert product_match is None

    @pytest.mark.asyncio
    async def test_already_cached_entries_skipped(self):
        matcher = _make_product_matcher(found=True)
        cache = ProductPriceLookupCache()

        # Pre-populate one entry
        first_query = _PRODUCT_WARM_QUERIES[0]
        cache.put(first_query, None)  # negative cache

        await warm_product_price_cache(matcher, price_cache=cache)

        # find_product should NOT have been called for the pre-cached query
        # (total calls = total queries - 1 pre-cached)
        assert matcher.find_product.call_count == len(_PRODUCT_WARM_QUERIES) - 1

    @pytest.mark.asyncio
    async def test_custom_queries_used(self):
        matcher = _make_product_matcher(found=True)
        cache = ProductPriceLookupCache()
        custom = ["Samsung Galaxy S23"]

        result = await warm_product_price_cache(matcher, price_cache=cache, queries=custom)

        assert result.entries_warmed == 1
        assert matcher.find_product.call_count == 1

    @pytest.mark.asyncio
    async def test_matcher_exception_counted_as_failure(self):
        matcher = MagicMock(spec=ProductMatcher)
        matcher.find_product.side_effect = RuntimeError("catalog unavailable")
        cache = ProductPriceLookupCache()

        result = await warm_product_price_cache(matcher, price_cache=cache)

        assert result.entries_failed == len(_PRODUCT_WARM_QUERIES)
        assert result.entries_warmed == 0
        assert len(result.errors) == len(_PRODUCT_WARM_QUERIES)

    @pytest.mark.asyncio
    async def test_uses_global_cache_when_none_provided(self):
        matcher = _make_product_matcher(found=True)
        # Should not raise even without explicit cache
        result = await warm_product_price_cache(matcher)
        assert result.entries_warmed > 0


# ---------------------------------------------------------------------------
# warm_policy_cache tests
# ---------------------------------------------------------------------------

class TestWarmPolicyCache:
    @pytest.mark.asyncio
    async def test_warms_all_default_queries(self):
        pipeline = _make_rag_pipeline()
        cache = PolicyDocumentCache()

        result = await warm_policy_cache(pipeline, policy_cache=cache)

        assert result.cache_name == "policy_document_cache"
        assert result.entries_warmed == len(_POLICY_WARM_QUERIES)
        assert result.entries_failed == 0

    @pytest.mark.asyncio
    async def test_cache_populated_with_rag_content(self):
        pipeline = _make_rag_pipeline(returns_nodes=True)
        cache = PolicyDocumentCache()

        await warm_policy_cache(pipeline, policy_cache=cache)

        for query in _POLICY_WARM_QUERIES:
            cached = cache.get(query)
            assert cached is not None
            is_verified, content = cached
            assert is_verified is True
            assert content is not None

    @pytest.mark.asyncio
    async def test_empty_retrieval_stores_false_sentinel(self):
        pipeline = _make_rag_pipeline(returns_nodes=False)
        cache = PolicyDocumentCache()

        await warm_policy_cache(pipeline, policy_cache=cache)

        for query in _POLICY_WARM_QUERIES:
            cached = cache.get(query)
            assert cached is not None
            is_verified, content = cached
            assert is_verified is False

    @pytest.mark.asyncio
    async def test_no_rag_pipeline_stores_sentinel(self):
        cache = PolicyDocumentCache()

        result = await warm_policy_cache(None, policy_cache=cache)

        assert result.entries_warmed == len(_POLICY_WARM_QUERIES)
        for query in _POLICY_WARM_QUERIES:
            cached = cache.get(query)
            assert cached is not None
            is_verified, _ = cached
            assert is_verified is True  # neutral sentinel

    @pytest.mark.asyncio
    async def test_pipeline_without_retriever_stores_sentinel(self):
        pipeline = _make_rag_pipeline(has_retriever=False)
        cache = PolicyDocumentCache()

        result = await warm_policy_cache(pipeline, policy_cache=cache)

        assert result.entries_warmed == len(_POLICY_WARM_QUERIES)

    @pytest.mark.asyncio
    async def test_already_cached_entries_skipped(self):
        pipeline = _make_rag_pipeline()
        cache = PolicyDocumentCache()

        first_query = _POLICY_WARM_QUERIES[0]
        cache.put(first_query, (True, "pre-cached content"))

        await warm_policy_cache(pipeline, policy_cache=cache)

        # retrieve should NOT have been called for the pre-cached query
        assert pipeline.retriever.retrieve.call_count == len(_POLICY_WARM_QUERIES) - 1

    @pytest.mark.asyncio
    async def test_retrieval_exception_counted_as_failure(self):
        pipeline = _make_rag_pipeline()
        pipeline.retriever.retrieve.side_effect = ConnectionError("DB unavailable")
        cache = PolicyDocumentCache()

        result = await warm_policy_cache(pipeline, policy_cache=cache)

        assert result.entries_failed == len(_POLICY_WARM_QUERIES)
        assert result.entries_warmed == 0

    @pytest.mark.asyncio
    async def test_custom_queries_used(self):
        pipeline = _make_rag_pipeline()
        cache = PolicyDocumentCache()
        custom = ["chính sách bảo hành Apple"]

        result = await warm_policy_cache(pipeline, policy_cache=cache, queries=custom)

        assert result.entries_warmed == 1
        assert pipeline.retriever.retrieve.call_count == 1


# ---------------------------------------------------------------------------
# warm_all_caches tests
# ---------------------------------------------------------------------------

class TestWarmAllCaches:
    @pytest.mark.asyncio
    async def test_returns_all_caches_warming_result(self):
        manager = _make_prompt_manager()
        matcher = _make_product_matcher()
        pipeline = _make_rag_pipeline()
        price_cache = ProductPriceLookupCache()
        policy_cache = PolicyDocumentCache()

        result = await warm_all_caches(
            prompt_manager=manager,
            product_matcher=matcher,
            rag_pipeline=pipeline,
            price_cache=price_cache,
            policy_cache=policy_cache,
        )

        assert isinstance(result, AllCachesWarmingResult)
        assert isinstance(result.prompt, WarmingResult)
        assert isinstance(result.product_price, WarmingResult)
        assert isinstance(result.policy, WarmingResult)

    @pytest.mark.asyncio
    async def test_total_warmed_aggregated(self):
        manager = _make_prompt_manager()
        matcher = _make_product_matcher()
        pipeline = _make_rag_pipeline()
        price_cache = ProductPriceLookupCache()
        policy_cache = PolicyDocumentCache()

        result = await warm_all_caches(
            prompt_manager=manager,
            product_matcher=matcher,
            rag_pipeline=pipeline,
            price_cache=price_cache,
            policy_cache=policy_cache,
        )

        expected = (
            len(_PROMPT_WARM_PATTERNS)
            + len(_PRODUCT_WARM_QUERIES)
            + len(_POLICY_WARM_QUERIES)
        )
        assert result.total_warmed == expected
        assert result.total_failed == 0

    @pytest.mark.asyncio
    async def test_total_failed_aggregated(self):
        manager = _make_prompt_manager(fail_templates=[_PROMPT_WARM_PATTERNS[0][0]])
        matcher = MagicMock(spec=ProductMatcher)
        matcher.find_product.side_effect = RuntimeError("fail")
        pipeline = _make_rag_pipeline()
        pipeline.retriever.retrieve.side_effect = ConnectionError("fail")
        price_cache = ProductPriceLookupCache()
        policy_cache = PolicyDocumentCache()

        result = await warm_all_caches(
            prompt_manager=manager,
            product_matcher=matcher,
            rag_pipeline=pipeline,
            price_cache=price_cache,
            policy_cache=policy_cache,
        )

        assert result.total_failed > 0

    @pytest.mark.asyncio
    async def test_duration_is_positive(self):
        manager = _make_prompt_manager()
        matcher = _make_product_matcher()
        pipeline = _make_rag_pipeline()

        result = await warm_all_caches(
            prompt_manager=manager,
            product_matcher=matcher,
            rag_pipeline=pipeline,
        )

        assert result.total_duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_str_representation(self):
        manager = _make_prompt_manager()
        matcher = _make_product_matcher()
        pipeline = _make_rag_pipeline()

        result = await warm_all_caches(
            prompt_manager=manager,
            product_matcher=matcher,
            rag_pipeline=pipeline,
        )

        s = str(result)
        assert "total_warmed" in s
        assert "total_failed" in s
        assert "duration" in s

    @pytest.mark.asyncio
    async def test_all_three_strategies_run_concurrently(self):
        """Verify all three warming functions are invoked (not just one)."""
        manager = _make_prompt_manager()
        matcher = _make_product_matcher()
        pipeline = _make_rag_pipeline()
        price_cache = ProductPriceLookupCache()
        policy_cache = PolicyDocumentCache()

        result = await warm_all_caches(
            prompt_manager=manager,
            product_matcher=matcher,
            rag_pipeline=pipeline,
            price_cache=price_cache,
            policy_cache=policy_cache,
        )

        # Each sub-result should have been populated
        assert result.prompt.entries_warmed + result.prompt.entries_failed > 0
        assert result.product_price.entries_warmed + result.product_price.entries_failed > 0
        assert result.policy.entries_warmed + result.policy.entries_failed > 0
