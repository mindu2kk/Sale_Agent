"""
Tests for ProductPriceLookupCache (task 5.1.2)

Covers:
- Basic get/put/miss behaviour
- TTL expiration
- Negative caching (None result for "not found")
- Cache key normalisation (casing / whitespace)
- PriceAccuracyChecker uses cache on second call
- Cache stats reporting
"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from backend.verification.utils.cache import (
    ProductPriceLookupCache,
    get_product_price_cache,
    configure_product_price_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_product_match(name="iPhone 15", brand="Apple", sku="APL-001",
                        price_raw="29.990.000 VNĐ", price_vnd=29_990_000.0,
                        match_score=0.95, match_type="fuzzy_name"):
    """Return a minimal mock that looks like a ProductMatch."""
    m = MagicMock()
    m.product_name = name
    m.brand = brand
    m.sku = sku
    m.price_raw = price_raw
    m.price_vnd = price_vnd
    m.match_score = match_score
    m.match_type = match_type
    m.display_name = f"{brand} {name}"
    return m


# ---------------------------------------------------------------------------
# 1. Basic get / put / miss
# ---------------------------------------------------------------------------

class TestProductPriceLookupCacheBasic:

    def test_miss_returns_none(self):
        cache = ProductPriceLookupCache()
        assert cache.get("iphone 15") is None

    def test_put_and_get_found(self):
        cache = ProductPriceLookupCache()
        pm = _make_product_match()
        cache.put("iphone 15", pm)
        result = cache.get("iphone 15")
        assert result is not None
        found, product = result
        assert found is True
        assert product is pm

    def test_put_none_negative_cache(self):
        """Storing None should create a negative cache entry (found=False)."""
        cache = ProductPriceLookupCache()
        cache.put("unknown product xyz", None)
        result = cache.get("unknown product xyz")
        assert result is not None
        found, product = result
        assert found is False
        assert product is None

    def test_invalidate_removes_entry(self):
        cache = ProductPriceLookupCache()
        pm = _make_product_match()
        cache.put("iphone 15", pm)
        cache.invalidate("iphone 15")
        assert cache.get("iphone 15") is None

    def test_clear_removes_all(self):
        cache = ProductPriceLookupCache()
        cache.put("iphone 15", _make_product_match())
        cache.put("samsung s24", _make_product_match("Galaxy S24", "Samsung"))
        cache.clear()
        assert cache.get("iphone 15") is None
        assert cache.get("samsung s24") is None


# ---------------------------------------------------------------------------
# 2. TTL expiration
# ---------------------------------------------------------------------------

class TestProductPriceLookupCacheTTL:

    def test_entry_expires_after_ttl(self):
        cache = ProductPriceLookupCache(ttl_seconds=1.0)
        pm = _make_product_match()
        cache.put("iphone 15", pm)

        # Advance time past TTL using mock
        with patch("time.time", return_value=time.time() + 2.0):
            assert cache.get("iphone 15") is None

    def test_entry_valid_before_ttl(self):
        cache = ProductPriceLookupCache(ttl_seconds=300.0)
        pm = _make_product_match()
        cache.put("iphone 15", pm)

        # Advance time but still within TTL
        with patch("time.time", return_value=time.time() + 100.0):
            result = cache.get("iphone 15")
        assert result is not None
        found, product = result
        assert found is True


# ---------------------------------------------------------------------------
# 3. Negative caching
# ---------------------------------------------------------------------------

class TestNegativeCaching:

    def test_negative_cache_distinguishable_from_miss(self):
        cache = ProductPriceLookupCache()
        # Miss → None
        assert cache.get("totally unknown") is None
        # Store negative result
        cache.put("totally unknown", None)
        # Now it's a hit with found=False
        result = cache.get("totally unknown")
        assert result is not None
        found, product = result
        assert found is False
        assert product is None

    def test_negative_cache_can_be_overwritten(self):
        cache = ProductPriceLookupCache()
        cache.put("iphone 15", None)
        pm = _make_product_match()
        cache.put("iphone 15", pm)
        result = cache.get("iphone 15")
        assert result is not None
        found, product = result
        assert found is True
        assert product is pm


# ---------------------------------------------------------------------------
# 4. Cache key normalisation
# ---------------------------------------------------------------------------

class TestCacheKeyNormalisation:

    def test_case_insensitive(self):
        cache = ProductPriceLookupCache()
        pm = _make_product_match()
        cache.put("iPhone 15", pm)
        # Different casing should hit the same entry
        result = cache.get("iphone 15")
        assert result is not None
        assert result[0] is True

    def test_leading_trailing_whitespace(self):
        cache = ProductPriceLookupCache()
        pm = _make_product_match()
        cache.put("  iphone 15  ", pm)
        result = cache.get("iphone 15")
        assert result is not None
        assert result[0] is True

    def test_mixed_case_and_whitespace(self):
        cache = ProductPriceLookupCache()
        pm = _make_product_match()
        cache.put("  IPHONE 15  ", pm)
        result = cache.get("iphone 15")
        assert result is not None


# ---------------------------------------------------------------------------
# 5. PriceAccuracyChecker uses cache on second call
# ---------------------------------------------------------------------------

class TestPriceAccuracyCheckerCacheIntegration:
    """Verify that _find_product_from_context only calls ProductMatcher once
    for the same query when a cache is provided."""

    def _make_checker(self, price_cache):
        """Build a PriceAccuracyChecker with a mocked ProductMatcher."""
        from backend.verification.agent.checkers import PriceAccuracyChecker
        from backend.verification.config import VerificationConfig

        config = VerificationConfig()
        checker = PriceAccuracyChecker(
            llm=None,
            rag_pipeline=None,
            config=config,
            catalog_path="nonexistent_path.csv",  # triggers FileNotFoundError → None
            price_cache=price_cache,
        )
        # Replace the None matcher with a mock
        mock_matcher = MagicMock()
        pm = _make_product_match()
        mock_matcher.find_product.return_value = pm
        mock_matcher.threshold = 0.6
        checker._product_matcher = mock_matcher
        return checker, mock_matcher, pm

    def test_second_call_uses_cache(self):
        cache = ProductPriceLookupCache()
        checker, mock_matcher, pm = self._make_checker(cache)

        # First call — should hit ProductMatcher
        result1 = checker._find_product_from_context("iphone 15")
        assert result1 is pm
        assert mock_matcher.find_product.call_count == 1

        # Second call with same query — should use cache, NOT call matcher again
        result2 = checker._find_product_from_context("iphone 15")
        assert result2 is pm
        assert mock_matcher.find_product.call_count == 1  # still 1

    def test_negative_result_cached(self):
        cache = ProductPriceLookupCache()
        checker, mock_matcher, _ = self._make_checker(cache)
        mock_matcher.find_product.return_value = None
        mock_matcher.find_all.return_value = []

        result1 = checker._find_product_from_context("totally unknown product")
        assert result1 is None

        # Second call — matcher should NOT be called again
        call_count_after_first = mock_matcher.find_product.call_count
        result2 = checker._find_product_from_context("totally unknown product")
        assert result2 is None
        assert mock_matcher.find_product.call_count == call_count_after_first

    def test_different_queries_call_matcher_separately(self):
        cache = ProductPriceLookupCache()
        checker, mock_matcher, pm = self._make_checker(cache)

        checker._find_product_from_context("iphone 15")
        checker._find_product_from_context("samsung galaxy s24")
        # Two distinct queries → matcher called at least twice
        assert mock_matcher.find_product.call_count >= 2


# ---------------------------------------------------------------------------
# 6. Cache stats reporting
# ---------------------------------------------------------------------------

class TestCacheStats:

    def test_stats_initial(self):
        cache = ProductPriceLookupCache()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    def test_stats_after_put_and_get(self):
        cache = ProductPriceLookupCache()
        pm = _make_product_match()
        cache.put("iphone 15", pm)

        cache.get("iphone 15")   # hit
        cache.get("missing key")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_stats_hit_rate(self):
        cache = ProductPriceLookupCache()
        pm = _make_product_match()
        cache.put("iphone 15", pm)
        cache.get("iphone 15")  # hit
        cache.get("iphone 15")  # hit
        cache.get("missing")    # miss

        stats = cache.get_stats()
        assert abs(stats["hit_rate"] - 2 / 3) < 0.01


# ---------------------------------------------------------------------------
# 7. Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:

    def test_get_product_price_cache_returns_same_instance(self):
        c1 = get_product_price_cache()
        c2 = get_product_price_cache()
        assert c1 is c2

    def test_configure_replaces_singleton(self):
        original = get_product_price_cache()
        new_cache = configure_product_price_cache(max_size=10, ttl_seconds=60.0)
        assert new_cache is not original
        assert get_product_price_cache() is new_cache
        # Restore default for other tests
        configure_product_price_cache()
