"""
Async Cache Warming Strategies cho Verification Agent

Pre-populates caches with common verification patterns to reduce cold-start
latency on the first real request. Three warming strategies are provided:

- warm_prompt_cache        : Pre-renders all known prompt templates with
                             representative variable sets.
- warm_product_price_cache : Pre-loads common product queries into the
                             ProductPriceLookupCache.
- warm_policy_cache        : Pre-loads common policy search queries into the
                             PolicyDocumentCache.
- warm_all_caches          : Runs all three strategies concurrently.

All functions are async-safe and designed to be called at application startup
(e.g. inside a FastAPI lifespan handler or a LangGraph workflow __init__).

Supports Requirement 9.4 (caching for performance) and Task 5.1.4.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .cache import (
    PolicyDocumentCache,
    ProductPriceLookupCache,
    get_policy_document_cache,
    get_product_price_cache,
)
from .product_matcher import ProductMatcher
from .prompt_cache import PromptTemplateCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common verification patterns used for warming
# ---------------------------------------------------------------------------

# Representative prompt template names and their minimal variable sets.
# These cover the most frequently rendered templates in the verification flow.
_PROMPT_WARM_PATTERNS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "price_accuracy_check",
        {
            "objection_text": "Giá sản phẩm này có đắt không?",
            "draft_response": "Sản phẩm có giá 29.990.000 VNĐ.",
            "db_data": "iPhone 15: 29.990.000 VNĐ",
            "price_tolerance": "1",
            "critical_threshold": "30",
        },
    ),
    (
        "policy_authenticity_check",
        {
            "draft_response": "Sản phẩm được bảo hành 12 tháng chính hãng.",
            "policy_documents": "Bảo hành 12 tháng tại trung tâm bảo hành Apple.",
            "forbidden_phrases": "tự bịa, không có trong hệ thống",
        },
    ),
    (
        "topic_relevance_check",
        {
            "objection_text": "Tại sao giá iPhone 15 lại cao như vậy?",
            "draft_response": "iPhone 15 có nhiều tính năng cao cấp xứng đáng với mức giá.",
            "relevance_threshold": "0.7",
            "empathy_phrases": "Tôi hiểu, Cảm ơn bạn đã chia sẻ",
        },
    ),
    (
        "correction_feedback",
        {
            "objection_text": "Giá sản phẩm này có đắt không?",
            "failed_draft": "Sản phẩm có giá 35.000.000 VNĐ.",
            "verification_issues": "Price deviation 16.7% exceeds tolerance 1%",
        },
    ),
]

# Common product queries that appear frequently in sales conversations.
_PRODUCT_WARM_QUERIES: List[str] = [
    "iPhone 15",
    "iPhone 15 Pro",
    "iPhone 14",
    "Samsung Galaxy S23",
    "Samsung Galaxy S24",
    "MacBook Air",
    "MacBook Pro",
    "iPad Pro",
    "Samsung Galaxy Tab",
    "Apple Watch",
]

# Common policy search queries used by PolicyAuthenticityChecker.
_POLICY_WARM_QUERIES: List[str] = [
    "chính sách bảo hành warranty policy terms",
    "chính sách đổi trả hoàn tiền return refund policy",
    "chính sách đổi máy thay thế sản phẩm exchange replacement policy",
    "dịch vụ sửa chữa bảo trì service repair policy",
    "hỗ trợ khách hàng customer support policy",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WarmingResult:
    """Summary of a single cache warming operation."""

    cache_name: str
    entries_warmed: int
    entries_failed: int
    duration_seconds: float
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.entries_warmed + self.entries_failed
        return self.entries_warmed / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"WarmingResult({self.cache_name}: "
            f"{self.entries_warmed} warmed, "
            f"{self.entries_failed} failed, "
            f"{self.duration_seconds:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Prompt cache warming
# ---------------------------------------------------------------------------

async def warm_prompt_cache(
    prompt_manager: Any,
    cache: Optional[PromptTemplateCache] = None,
    patterns: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> WarmingResult:
    """
    Pre-render common prompt templates into the PromptTemplateCache.

    Args:
        prompt_manager: A ``CachedPromptTemplates`` or ``PromptTemplateManager``
                        instance. Must expose ``render(name, **vars)`` and
                        ``get_template(name)``.
        cache:          Optional explicit cache to warm. If None, the cache
                        embedded in *prompt_manager* is used (when available).
        patterns:       Override the default warm patterns. Each entry is a
                        (template_name, variables_dict) tuple.

    Returns:
        WarmingResult with warming statistics.
    """
    start = time.monotonic()
    warm_patterns = patterns or _PROMPT_WARM_PATTERNS
    warmed = 0
    failed = 0
    errors: List[str] = []

    for template_name, variables in warm_patterns:
        try:
            # render() will populate the cache as a side-effect when using
            # CachedPromptTemplates; for plain PromptTemplateManager it just
            # validates the template exists.
            await asyncio.to_thread(prompt_manager.render, template_name, **variables)
            warmed += 1
            logger.debug("Warmed prompt template: %s", template_name)
        except Exception as exc:
            failed += 1
            msg = f"Failed to warm prompt '{template_name}': {exc}"
            errors.append(msg)
            logger.warning(msg)

    duration = time.monotonic() - start
    result = WarmingResult(
        cache_name="prompt_template_cache",
        entries_warmed=warmed,
        entries_failed=failed,
        duration_seconds=duration,
        errors=errors,
    )
    logger.info("Prompt cache warming complete: %s", result)
    return result


# ---------------------------------------------------------------------------
# Product price cache warming
# ---------------------------------------------------------------------------

async def warm_product_price_cache(
    product_matcher: ProductMatcher,
    price_cache: Optional[ProductPriceLookupCache] = None,
    queries: Optional[List[str]] = None,
) -> WarmingResult:
    """
    Pre-load common product queries into the ProductPriceLookupCache.

    Args:
        product_matcher: A ``ProductMatcher`` instance backed by the catalog CSV.
        price_cache:     Cache to populate. Defaults to the global singleton.
        queries:         Override the default product query list.

    Returns:
        WarmingResult with warming statistics.
    """
    start = time.monotonic()
    cache = price_cache or get_product_price_cache()
    warm_queries = queries or _PRODUCT_WARM_QUERIES
    warmed = 0
    failed = 0
    errors: List[str] = []

    async def _warm_one(query: str) -> None:
        nonlocal warmed, failed
        try:
            # Skip if already cached
            if cache.get(query) is not None:
                warmed += 1
                return
            match = await asyncio.to_thread(product_matcher.find_product, query)
            cache.put(query, match)
            warmed += 1
            logger.debug("Warmed product price cache: %s → %s", query, match)
        except Exception as exc:
            failed += 1
            msg = f"Failed to warm product query '{query}': {exc}"
            errors.append(msg)
            logger.warning(msg)

    await asyncio.gather(*[_warm_one(q) for q in warm_queries])

    duration = time.monotonic() - start
    result = WarmingResult(
        cache_name="product_price_cache",
        entries_warmed=warmed,
        entries_failed=failed,
        duration_seconds=duration,
        errors=errors,
    )
    logger.info("Product price cache warming complete: %s", result)
    return result


# ---------------------------------------------------------------------------
# Policy document cache warming
# ---------------------------------------------------------------------------

async def warm_policy_cache(
    rag_pipeline: Any,
    policy_cache: Optional[PolicyDocumentCache] = None,
    queries: Optional[List[str]] = None,
) -> WarmingResult:
    """
    Pre-load common policy search queries into the PolicyDocumentCache.

    The warming stores a sentinel ``(True, None)`` entry for each query so
    that the first real lookup finds a cache hit and skips the RAG retrieval.
    When a live RAG pipeline is available, the actual retrieval result is
    stored instead.

    Args:
        rag_pipeline: The RAG pipeline instance (may be None in tests).
        policy_cache: Cache to populate. Defaults to the global singleton.
        queries:      Override the default policy query list.

    Returns:
        WarmingResult with warming statistics.
    """
    start = time.monotonic()
    cache = policy_cache or get_policy_document_cache()
    warm_queries = queries or _POLICY_WARM_QUERIES
    warmed = 0
    failed = 0
    errors: List[str] = []

    async def _warm_one(query: str) -> None:
        nonlocal warmed, failed
        try:
            # Skip if already cached
            if cache.get(query) is not None:
                warmed += 1
                return

            if rag_pipeline is not None and hasattr(rag_pipeline, "retriever"):
                # Perform real retrieval and cache the result
                nodes = await asyncio.to_thread(
                    rag_pipeline.retriever.retrieve, query
                )
                if nodes:
                    top_text = nodes[0].get_content() if hasattr(nodes[0], "get_content") else str(nodes[0])
                    cache.put(query, (True, top_text))
                else:
                    cache.put(query, (False, None))
            else:
                # No live pipeline — store a neutral sentinel so the key is
                # present and the checker falls through to its own logic.
                cache.put(query, (True, None))

            warmed += 1
            logger.debug("Warmed policy cache: %s", query)
        except Exception as exc:
            failed += 1
            msg = f"Failed to warm policy query '{query}': {exc}"
            errors.append(msg)
            logger.warning(msg)

    await asyncio.gather(*[_warm_one(q) for q in warm_queries])

    duration = time.monotonic() - start
    result = WarmingResult(
        cache_name="policy_document_cache",
        entries_warmed=warmed,
        entries_failed=failed,
        duration_seconds=duration,
        errors=errors,
    )
    logger.info("Policy cache warming complete: %s", result)
    return result


# ---------------------------------------------------------------------------
# Composite warmer
# ---------------------------------------------------------------------------

@dataclass
class AllCachesWarmingResult:
    """Aggregated result from warming all caches."""

    prompt: WarmingResult
    product_price: WarmingResult
    policy: WarmingResult
    total_duration_seconds: float

    @property
    def total_warmed(self) -> int:
        return self.prompt.entries_warmed + self.product_price.entries_warmed + self.policy.entries_warmed

    @property
    def total_failed(self) -> int:
        return self.prompt.entries_failed + self.product_price.entries_failed + self.policy.entries_failed

    def __str__(self) -> str:
        return (
            f"AllCachesWarmingResult("
            f"total_warmed={self.total_warmed}, "
            f"total_failed={self.total_failed}, "
            f"duration={self.total_duration_seconds:.2f}s)"
        )


async def warm_all_caches(
    prompt_manager: Any,
    product_matcher: ProductMatcher,
    rag_pipeline: Any,
    price_cache: Optional[ProductPriceLookupCache] = None,
    policy_cache: Optional[PolicyDocumentCache] = None,
    prompt_cache: Optional[PromptTemplateCache] = None,
) -> AllCachesWarmingResult:
    """
    Run all three cache warming strategies concurrently.

    Args:
        prompt_manager:  CachedPromptTemplates or PromptTemplateManager instance.
        product_matcher: ProductMatcher backed by the catalog CSV.
        rag_pipeline:    RAG pipeline for policy retrieval (may be None).
        price_cache:     Optional explicit ProductPriceLookupCache.
        policy_cache:    Optional explicit PolicyDocumentCache.
        prompt_cache:    Optional explicit PromptTemplateCache (passed to
                         warm_prompt_cache when prompt_manager lacks its own).

    Returns:
        AllCachesWarmingResult aggregating all three WarmingResult objects.
    """
    start = time.monotonic()

    prompt_result, price_result, policy_result = await asyncio.gather(
        warm_prompt_cache(prompt_manager, cache=prompt_cache),
        warm_product_price_cache(product_matcher, price_cache=price_cache),
        warm_policy_cache(rag_pipeline, policy_cache=policy_cache),
    )

    total_duration = time.monotonic() - start
    result = AllCachesWarmingResult(
        prompt=prompt_result,
        product_price=price_result,
        policy=policy_result,
        total_duration_seconds=total_duration,
    )
    logger.info("All caches warmed: %s", result)
    return result
