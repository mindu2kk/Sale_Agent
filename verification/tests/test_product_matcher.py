"""
Unit tests for ProductMatcher

Tests cover:
- Exact SKU matching
- Exact name matching
- Fuzzy name matching (token-based)
- No-match / below-threshold cases
- Price parsing
- Edge cases (empty query, missing catalog)
"""

import pytest
from verification.utils.product_matcher import (
    ProductMatcher,
    ProductMatch,
    _normalize,
    _token_set_ratio,
    _parse_price_vnd,
)

# ---------------------------------------------------------------------------
# Sample catalog (in-memory, no CSV needed)
# ---------------------------------------------------------------------------

SAMPLE_CATALOG = [
    {
        "Product Code": "IPHONE15PRO",
        "Product": "Mobile Phone",
        "Brand": "Apple",
        "Price": "29.990.000 VNĐ",
    },
    {
        "Product Code": "SAMSUNG_S23",
        "Product": "Mobile Phone",
        "Brand": "Samsung",
        "Price": "15.000.000 VNĐ",
    },
    {
        "Product Code": "MACBOOK_AIR",
        "Product": "Laptop",
        "Brand": "Apple",
        "Price": "28.000.000 VNĐ",
    },
    {
        "Product Code": "DELL_XPS15",
        "Product": "Laptop",
        "Brand": "Dell",
        "Price": "35.000.000 VNĐ",
    },
    {
        "Product Code": "SONY_WH1000",
        "Product": "Headphones",
        "Brand": "Sony",
        "Price": "8.500.000 VNĐ",
    },
]


@pytest.fixture
def matcher():
    return ProductMatcher(catalog_source=SAMPLE_CATALOG, threshold=0.6)


# ---------------------------------------------------------------------------
# _parse_price_vnd
# ---------------------------------------------------------------------------

class TestParsePriceVnd:
    def test_dot_separated_vnd(self):
        assert _parse_price_vnd("29.990.000 VNĐ") == 29990000.0

    def test_plain_digits(self):
        assert _parse_price_vnd("15000000") == 15000000.0

    def test_empty_string(self):
        assert _parse_price_vnd("") == 0.0

    def test_non_numeric(self):
        assert _parse_price_vnd("N/A") == 0.0


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("iPhone 15 Pro") == "iphone 15 pro"

    def test_strips_special_chars(self):
        # Special chars are replaced by spaces then collapsed
        assert _normalize("Apple® MacBook™") == "apple macbook"

    def test_collapses_spaces(self):
        result = _normalize("  hello   world  ")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# _token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical_strings(self):
        assert _token_set_ratio("apple iphone", "apple iphone") == pytest.approx(1.0, abs=0.01)

    def test_subset_query(self):
        # "iPhone 15" is a subset of "Apple iPhone 15 128GB"
        score = _token_set_ratio("iphone 15", "apple iphone 15 128gb")
        assert score >= 0.6

    def test_completely_different(self):
        score = _token_set_ratio("laptop dell", "sony headphones")
        assert score < 0.3

    def test_empty_strings(self):
        assert _token_set_ratio("", "apple") == 0.0
        assert _token_set_ratio("apple", "") == 0.0


# ---------------------------------------------------------------------------
# ProductMatcher.find_product
# ---------------------------------------------------------------------------

class TestFindProductExactSku:
    def test_exact_sku_match(self, matcher):
        result = matcher.find_product("IPHONE15PRO")
        assert result is not None
        assert result.match_type == "exact_sku"
        assert result.match_score == 1.0
        assert result.sku == "IPHONE15PRO"

    def test_exact_sku_case_insensitive(self, matcher):
        result = matcher.find_product("iphone15pro")
        assert result is not None
        assert result.match_type == "exact_sku"

    def test_sku_not_found_falls_through(self, matcher):
        # "UNKNOWN_SKU" should not match by SKU; may or may not match fuzzy
        result = matcher.find_product("UNKNOWN_SKU_XYZ_999")
        # Either None or a low-confidence fuzzy match — just ensure no crash
        if result is not None:
            assert result.match_score >= matcher.threshold


class TestFindProductExactName:
    def test_exact_brand_product_match(self, matcher):
        result = matcher.find_product("Apple Laptop")
        assert result is not None
        assert result.match_type in ("exact_name", "fuzzy_name")
        assert result.brand == "Apple"

    def test_exact_product_type_match(self, matcher):
        result = matcher.find_product("Headphones")
        assert result is not None
        assert result.product_name == "Headphones"


class TestFindProductFuzzy:
    def test_fuzzy_brand_product(self, matcher):
        # "Samsung phone" should fuzzy-match Samsung Mobile Phone
        result = matcher.find_product("Samsung phone")
        assert result is not None
        assert result.brand == "Samsung"
        assert result.match_type == "fuzzy_name"

    def test_fuzzy_partial_name(self, matcher):
        # "Dell laptop" should match Dell Laptop
        result = matcher.find_product("Dell laptop")
        assert result is not None
        assert result.brand == "Dell"

    def test_fuzzy_apple_mobile(self, matcher):
        # "Apple mobile" should match Apple Mobile Phone
        result = matcher.find_product("Apple mobile")
        assert result is not None
        assert result.brand == "Apple"
        assert result.product_name == "Mobile Phone"

    def test_below_threshold_returns_none(self):
        # Use a very high threshold so nothing matches
        strict_matcher = ProductMatcher(catalog_source=SAMPLE_CATALOG, threshold=0.99)
        result = strict_matcher.find_product("completely unrelated query xyz")
        assert result is None

    def test_empty_query_returns_none(self, matcher):
        assert matcher.find_product("") is None
        assert matcher.find_product("   ") is None


# ---------------------------------------------------------------------------
# ProductMatch properties
# ---------------------------------------------------------------------------

class TestProductMatchProperties:
    def test_display_name(self, matcher):
        result = matcher.find_product("IPHONE15PRO")
        assert result is not None
        assert result.display_name == "Apple Mobile Phone"

    def test_price_vnd_parsed(self, matcher):
        result = matcher.find_product("IPHONE15PRO")
        assert result is not None
        assert result.price_vnd == 29990000.0

    def test_price_raw_preserved(self, matcher):
        result = matcher.find_product("IPHONE15PRO")
        assert result is not None
        assert "29.990.000" in result.price_raw


# ---------------------------------------------------------------------------
# ProductMatcher.find_all
# ---------------------------------------------------------------------------

class TestFindAll:
    def test_returns_multiple_results(self, matcher):
        # "Apple" should match both Apple products
        results = matcher.find_all("Apple", top_k=5)
        assert len(results) >= 1
        brands = {r.brand for r in results}
        assert "Apple" in brands

    def test_respects_top_k(self, matcher):
        results = matcher.find_all("Mobile Phone", top_k=2)
        assert len(results) <= 2

    def test_sorted_by_score(self, matcher):
        results = matcher.find_all("Samsung Mobile Phone", top_k=5)
        scores = [r.match_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self, matcher):
        assert matcher.find_all("") == []


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestCsvLoading:
    def test_loads_real_catalog(self):
        """Smoke test: real CSV loads without error."""
        try:
            m = ProductMatcher("data/product_catalog_clean.csv")
            assert len(m._products) > 0
        except FileNotFoundError:
            pytest.skip("product_catalog_clean.csv not available in test environment")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ProductMatcher("data/nonexistent_catalog.csv")
