"""
Unit Tests for PriceExtractor

Tests Requirements 4.1: Price extraction from draft_response using regex/NLP

Covers:
- Vietnamese price formats: VND, đồng, triệu, nghìn, tỷ, đ suffix
- English price formats: $X, X USD, X dollars
- Normalization to VND
- Product context extraction
- Edge cases: empty text, no prices, multiple prices, price ranges
- Determinism
"""

import pytest
from verification.utils.price_extractor import (
    PriceExtractor,
    ExtractedPrice,
    extract_prices_detailed,
    _clean_number,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return PriceExtractor(usd_to_vnd_rate=25_000.0)


# ---------------------------------------------------------------------------
# _clean_number helper tests
# ---------------------------------------------------------------------------

class TestCleanNumber:
    def test_plain_integer(self):
        assert _clean_number("25000000") == 25_000_000.0

    def test_comma_thousands(self):
        assert _clean_number("25,000,000") == 25_000_000.0

    def test_dot_thousands(self):
        assert _clean_number("25.000.000") == 25_000_000.0

    def test_mixed_separators_us_style(self):
        # 1,234.56 → US decimal
        assert _clean_number("1,234.56") == pytest.approx(1234.56)

    def test_simple_decimal(self):
        assert _clean_number("25.5") == pytest.approx(25.5)

    def test_single_digit(self):
        assert _clean_number("5") == 5.0


# ---------------------------------------------------------------------------
# Vietnamese VND explicit formats
# ---------------------------------------------------------------------------

class TestVietnameseVNDExplicit:
    def test_comma_separated_vnd(self, extractor):
        prices = extractor.extract("Giá iPhone là 25,000,000 VND")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(25_000_000.0)
        assert prices[0].currency == "VND"
        assert "25,000,000 VND" in prices[0].original_text

    def test_dot_separated_vnd(self, extractor):
        prices = extractor.extract("Sản phẩm giá 25.000.000 VND")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(25_000_000.0)

    def test_vnd_lowercase(self, extractor):
        prices = extractor.extract("giá 10.000.000 vnđ")
        assert len(prices) == 1
        assert prices[0].currency == "VND"

    def test_dong_suffix(self, extractor):
        prices = extractor.extract("Giá 25.000.000đ rất hợp lý")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(25_000_000.0)
        assert prices[0].currency == "VND"

    def test_dong_word(self, extractor):
        prices = extractor.extract("Chỉ 500,000 đồng thôi")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(500_000.0)


# ---------------------------------------------------------------------------
# Vietnamese shorthand formats
# ---------------------------------------------------------------------------

class TestVietnameseShorthand:
    def test_trieu(self, extractor):
        prices = extractor.extract("iPhone 15 giá 25 triệu")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(25_000_000.0)
        assert prices[0].currency == "VND"

    def test_trieu_decimal(self, extractor):
        prices = extractor.extract("Giá 25.5 triệu")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(25_500_000.0)

    def test_nghin(self, extractor):
        prices = extractor.extract("Phụ kiện giá 500 nghìn")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(500_000.0)

    def test_ngan(self, extractor):
        prices = extractor.extract("Chỉ 200 ngàn")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(200_000.0)

    def test_ty(self, extractor):
        prices = extractor.extract("Xe hơi giá 2 tỷ")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(2_000_000_000.0)


# ---------------------------------------------------------------------------
# English USD formats
# ---------------------------------------------------------------------------

class TestEnglishUSD:
    def test_dollar_sign_prefix(self, extractor):
        prices = extractor.extract("Price is $500")
        assert len(prices) == 1
        assert prices[0].currency == "USD"
        assert prices[0].amount_vnd == pytest.approx(500 * 25_000.0)

    def test_usd_suffix(self, extractor):
        prices = extractor.extract("Costs 500 USD")
        assert len(prices) == 1
        assert prices[0].currency == "USD"
        assert prices[0].amount_vnd == pytest.approx(500 * 25_000.0)

    def test_dollars_word(self, extractor):
        prices = extractor.extract("Only 200 dollars")
        assert len(prices) == 1
        assert prices[0].currency == "USD"

    def test_dollar_with_thousands(self, extractor):
        prices = extractor.extract("MacBook costs $1,299")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(1299 * 25_000.0)

    def test_custom_exchange_rate(self):
        extractor = PriceExtractor(usd_to_vnd_rate=24_000.0)
        prices = extractor.extract("$100")
        assert prices[0].amount_vnd == pytest.approx(100 * 24_000.0)


# ---------------------------------------------------------------------------
# Price ranges
# ---------------------------------------------------------------------------

class TestPriceRanges:
    def test_trieu_range(self, extractor):
        prices = extractor.extract("Giá từ 25-30 triệu")
        # Should return 2 prices (lower and upper bound)
        assert len(prices) == 2
        amounts = sorted(p.amount_vnd for p in prices)
        assert amounts[0] == pytest.approx(25_000_000.0)
        assert amounts[1] == pytest.approx(30_000_000.0)
        assert all(p.currency == "VND" for p in prices)

    def test_usd_range(self, extractor):
        prices = extractor.extract("Price range $400-$500")
        assert len(prices) == 2
        amounts = sorted(p.amount_vnd for p in prices)
        assert amounts[0] == pytest.approx(400 * 25_000.0)
        assert amounts[1] == pytest.approx(500 * 25_000.0)
        assert all(p.currency == "USD" for p in prices)

    def test_nghin_range(self, extractor):
        prices = extractor.extract("Từ 500-800 nghìn")
        assert len(prices) == 2
        amounts = sorted(p.amount_vnd for p in prices)
        assert amounts[0] == pytest.approx(500_000.0)
        assert amounts[1] == pytest.approx(800_000.0)


# ---------------------------------------------------------------------------
# Multiple prices in one text
# ---------------------------------------------------------------------------

class TestMultiplePrices:
    def test_two_products(self, extractor):
        text = "iPhone 15 giá 25 triệu, Samsung S24 giá 22 triệu"
        prices = extractor.extract(text)
        assert len(prices) == 2
        amounts = sorted(p.amount_vnd for p in prices)
        assert amounts[0] == pytest.approx(22_000_000.0)
        assert amounts[1] == pytest.approx(25_000_000.0)

    def test_mixed_currencies(self, extractor):
        text = "Giá VND: 25,000,000 VND hoặc $1,000 USD"
        prices = extractor.extract(text)
        currencies = {p.currency for p in prices}
        assert "VND" in currencies
        assert "USD" in currencies


# ---------------------------------------------------------------------------
# Product context
# ---------------------------------------------------------------------------

class TestProductContext:
    def test_context_contains_surrounding_text(self, extractor):
        text = "iPhone 15 Pro Max có giá 34,990,000 VND tại cửa hàng"
        prices = extractor.extract(text)
        assert len(prices) == 1
        ctx = prices[0].product_context
        assert "iPhone" in ctx or "34,990,000" in ctx

    def test_context_at_start_of_text(self, extractor):
        text = "25 triệu là giá iPhone"
        prices = extractor.extract(text)
        assert len(prices) == 1
        # Should not raise; context may be short
        assert isinstance(prices[0].product_context, str)

    def test_context_at_end_of_text(self, extractor):
        text = "Sản phẩm này có giá 25 triệu"
        prices = extractor.extract(text)
        assert len(prices) == 1
        assert isinstance(prices[0].product_context, str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self, extractor):
        assert extractor.extract("") == []

    def test_whitespace_only(self, extractor):
        assert extractor.extract("   ") == []

    def test_no_prices(self, extractor):
        assert extractor.extract("Sản phẩm này rất tốt, chất lượng cao") == []

    def test_number_without_currency(self, extractor):
        # Plain numbers without currency markers should NOT be extracted
        prices = extractor.extract("Có 5 sản phẩm trong kho")
        assert len(prices) == 0

    def test_zero_amount(self, extractor):
        # "0 VND" is technically valid
        prices = extractor.extract("Miễn phí: 0 VND")
        assert len(prices) == 1
        assert prices[0].amount_vnd == 0.0

    def test_large_amount(self, extractor):
        prices = extractor.extract("Biệt thự giá 50 tỷ")
        assert len(prices) == 1
        assert prices[0].amount_vnd == pytest.approx(50_000_000_000.0)

    def test_position_fields(self, extractor):
        text = "Giá 25 triệu nhé"
        prices = extractor.extract(text)
        assert len(prices) == 1
        p = prices[0]
        assert p.start >= 0
        assert p.end > p.start
        assert text[p.start:p.end] == p.original_text


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self, extractor):
        text = "iPhone 15 giá 25 triệu, Samsung S24 giá 22,000,000 VND"
        result1 = extractor.extract(text)
        result2 = extractor.extract(text)
        assert len(result1) == len(result2)
        for p1, p2 in zip(result1, result2):
            assert p1.original_text == p2.original_text
            assert p1.amount_vnd == p2.amount_vnd
            assert p1.currency == p2.currency

    def test_order_is_by_position(self, extractor):
        text = "Giá A: 10 triệu. Giá B: 20 triệu. Giá C: 30 triệu."
        prices = extractor.extract(text)
        assert len(prices) == 3
        starts = [p.start for p in prices]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

class TestExtractPricesDetailed:
    def test_returns_list(self):
        result = extract_prices_detailed("Giá 25 triệu")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_custom_rate(self):
        result = extract_prices_detailed("$100", usd_to_vnd_rate=24_000.0)
        assert result[0].amount_vnd == pytest.approx(100 * 24_000.0)

    def test_empty_returns_empty(self):
        assert extract_prices_detailed("") == []
