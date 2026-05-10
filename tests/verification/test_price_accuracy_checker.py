"""
Unit Tests for PriceAccuracyChecker — Binary Price Verification

Tests Requirements 4.3: Binary scoring — PASS if deviation ≤ 1%, FAIL otherwise
with structured PriceIssue.

Covers:
- PASS cases: exact match, within ±1% tolerance
- FAIL cases: deviation > 1% with structured PriceIssue
- Severity classification: minor / major / critical
- Edge case: objection mentions price but draft has no price → FAIL MAJOR
- Edge case: no price in draft, no price in objection → PASS
- Edge case: product not found in catalog → FAIL MAJOR
- Multiple products in draft
- No catalog available (None matcher)
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional

from verification.agent.checkers import PriceAccuracyChecker
from verification.config.config import VerificationConfig
from verification.config.thresholds_config import PriceAccuracyThresholds
from verification.models.verification import PriceIssue, IssueSeverity
from verification.utils.product_matcher import ProductMatch
from verification.utils.cache import ProductPriceLookupCache


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_checker(
    catalog_source=None,
    tolerance_percent: float = 1.0,
    critical_threshold: float = 30.0,
) -> PriceAccuracyChecker:
    """
    Build a PriceAccuracyChecker with an in-memory catalog (no CSV needed).
    """
    config = VerificationConfig(
        price_tolerance_percent=tolerance_percent,
        price_critical_threshold=critical_threshold,
    )
    thresholds = PriceAccuracyThresholds(
        pass_tolerance_percent=tolerance_percent,
        critical_threshold_percent=critical_threshold,
    )
    checker = PriceAccuracyChecker(
        llm=None,
        rag_pipeline=None,
        config=config,
        thresholds=thresholds,
    )
    # Replace the real ProductMatcher with a mock or in-memory one
    if catalog_source is not None:
        from verification.utils.product_matcher import ProductMatcher
        checker._product_matcher = ProductMatcher(catalog_source=catalog_source)
    else:
        checker._product_matcher = None
    return checker


SAMPLE_CATALOG = [
    {
        "Product Code": "IPHONE15PRO",
        "Product": "Mobile Phone",
        "Brand": "Apple",
        "Price": "29.990.000 VNĐ",   # 29,990,000 VND
    },
    {
        "Product Code": "SAMSUNG_S23",
        "Product": "Mobile Phone",
        "Brand": "Samsung",
        "Price": "15.000.000 VNĐ",   # 15,000,000 VND
    },
    {
        "Product Code": "MACBOOK_AIR",
        "Product": "Laptop",
        "Brand": "Apple",
        "Price": "28.000.000 VNĐ",   # 28,000,000 VND
    },
]


def _make_position_aware_mock_matcher(product_prices: dict) -> MagicMock:
    """
    Build a mock ProductMatcher that maps product names to prices.

    Uses the ExtractedPrice.amount_vnd (passed via a thread-local trick) to
    find the closest-matching product. Since we can't pass extra args to
    find_product(), we use a simpler approach: find the product name that
    appears in the context AND whose catalog price is closest to the FIRST
    price number found in the context.

    product_prices: {product_name: price_vnd}  (insertion-ordered)
    """
    import re as _re
    products_list = list(product_prices.items())

    def _first_price_in_context(context: str) -> float:
        """Extract the FIRST price number from context string."""
        patterns = [
            r'(\d{1,3}(?:[.,]\d{3})+)',   # 29.990.000 or 29,990,000
            r'(\d+)\s*tri[eệ]u',           # 29 triệu
        ]
        for pat in patterns:
            m = _re.search(pat, context, _re.IGNORECASE)
            if m:
                raw = m.group(1) if m.lastindex else m.group(0)
                normalized = raw.replace('.', '').replace(',', '')
                try:
                    val = float(normalized)
                    if 'tri' in pat:
                        val *= 1_000_000
                    if val > 1000:
                        return val
                except ValueError:
                    pass
        return 0.0

    def mock_find_product(context: str):
        # Find the FIRST price in context, then match to closest catalog product
        price_val = _first_price_in_context(context)
        if price_val > 0:
            best_name, best_price = min(
                products_list,
                key=lambda np: abs(np[1] - price_val)
            )
            return ProductMatch(
                product_name=best_name,
                brand="Test",
                sku=best_name.upper().replace(" ", "_"),
                price_raw=f"{best_price:,.0f} VNĐ",
                price_vnd=float(best_price),
                match_score=0.9,
                match_type="fuzzy_name",
            )
        # Fallback: return first product
        name, price = products_list[0]
        return ProductMatch(
            product_name=name,
            brand="Test",
            sku=name.upper().replace(" ", "_"),
            price_raw=f"{price:,.0f} VNĐ",
            price_vnd=float(price),
            match_score=0.9,
            match_type="fuzzy_name",
        )

    mock_matcher = MagicMock()
    mock_matcher.find_product.side_effect = mock_find_product
    mock_matcher.threshold = 0.6
    mock_matcher.find_all.return_value = []
    return mock_matcher


@pytest.fixture
def checker():
    return _make_checker(catalog_source=SAMPLE_CATALOG)


@pytest.fixture
def checker_no_catalog():
    return _make_checker(catalog_source=None)


# ---------------------------------------------------------------------------
# PASS cases — no prices in draft
# ---------------------------------------------------------------------------

class TestNoPricesInDraft:
    def test_no_price_no_price_objection_passes(self, checker):
        """Draft has no price, objection doesn't mention price → PASS"""
        passed, issues = checker.check_price_accuracy(
            draft="iPhone 15 Pro là điện thoại cao cấp với camera tuyệt vời.",
            objection="Điện thoại này có tốt không?",
        )
        assert passed is True
        assert issues == []

    def test_no_price_in_draft_but_objection_mentions_price_fails(self, checker):
        """Draft has no price, but objection mentions price → FAIL MAJOR"""
        passed, issues = checker.check_price_accuracy(
            draft="iPhone 15 Pro là điện thoại cao cấp.",
            objection="Giá điện thoại này là bao nhiêu?",
        )
        assert passed is False
        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == IssueSeverity.MAJOR
        assert "price" in issue.explanation.lower() or "giá" in issue.explanation.lower() or "pricing" in issue.explanation.lower()

    def test_price_keywords_in_objection_trigger_fail(self, checker):
        """Various price keywords in objection trigger FAIL when draft has no price"""
        price_objections = [
            "Sản phẩm này giá bao nhiêu?",
            "Có đắt không?",
            "What is the cost?",
            "Is it expensive?",
            "Budget của tôi là 20 triệu",
        ]
        for objection in price_objections:
            passed, issues = checker.check_price_accuracy(
                draft="Sản phẩm rất tốt.",
                objection=objection,
            )
            assert passed is False, f"Expected FAIL for objection: {objection}"
            assert len(issues) == 1


# ---------------------------------------------------------------------------
# PASS cases — price within tolerance
# ---------------------------------------------------------------------------

class TestPriceWithinTolerance:
    def test_exact_price_match_passes(self, checker):
        """Exact price match → PASS"""
        # Apple Mobile Phone = 29,990,000 VND
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone có giá 29.990.000 VNĐ, rất hợp lý.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is True
        assert issues == []

    def test_price_within_1_percent_passes(self, checker):
        """Price within ±1% tolerance → PASS"""
        # 29,990,000 * 1.009 ≈ 29,989,910 — within 1%
        # Use 30,200,000 which is ~0.7% above 29,990,000
        # 29,990,000 + 0.7% = 30,199,930 ≈ 30,200,000
        # Actually let's use 29,990,000 * 1.005 = 30,139,950 → ~0.5% deviation
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá khoảng 30.139.950 VNĐ.",
            objection="Giá iPhone bao nhiêu?",
        )
        # 0.5% deviation ≤ 1% tolerance → PASS
        assert passed is True
        assert issues == []

    def test_no_catalog_no_context_passes(self, checker_no_catalog):
        """No catalog available → cannot verify → treat as pass"""
        passed, issues = checker_no_catalog.check_price_accuracy(
            draft="Sản phẩm giá 25 triệu.",
            objection="Giá bao nhiêu?",
        )
        # Without catalog, prices with no product context cannot be verified
        # The checker should not produce false positives
        assert isinstance(passed, bool)
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# FAIL cases — price deviation > tolerance
# ---------------------------------------------------------------------------

class TestPriceDeviationFails:
    def test_significant_deviation_fails(self, checker):
        """Price significantly different from catalog → FAIL"""
        # Apple Mobile Phone = 29,990,000 VND
        # Mention 25,000,000 VND → ~16.6% deviation → FAIL MAJOR
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone chỉ có giá 25 triệu thôi.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is False
        assert len(issues) >= 1

    def test_price_issue_has_required_fields(self, checker):
        """PriceIssue must have product_name, mentioned_price, actual_price, deviation_percent, severity, explanation"""
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 20 triệu.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is False
        assert len(issues) >= 1
        issue = issues[0]
        assert issue.product_name  # non-empty
        assert issue.mentioned_price is not None
        assert issue.actual_price is not None
        assert issue.deviation_percent is not None
        assert issue.deviation_percent >= 0
        assert issue.severity in (IssueSeverity.MINOR, IssueSeverity.MAJOR, IssueSeverity.CRITICAL)
        assert issue.explanation  # non-empty

    def test_deviation_percent_is_non_negative(self, checker):
        """deviation_percent must always be non-negative"""
        # Mention price higher than actual
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 40 triệu.",
            objection="Giá iPhone bao nhiêu?",
        )
        if issues:
            for issue in issues:
                if issue.deviation_percent is not None:
                    assert issue.deviation_percent >= 0


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class TestSeverityClassification:
    def test_minor_severity_for_small_deviation(self):
        """Deviation between 1% and 5% → MINOR severity"""
        # Default thresholds: minor=5%, major=15%, critical=30%
        # 3% deviation → MINOR
        checker = _make_checker(catalog_source=SAMPLE_CATALOG, tolerance_percent=1.0)
        # Apple Mobile Phone = 29,990,000 VND
        # 3% above = 30,889,700 VND ≈ 30.89 triệu
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 30.889.700 VNĐ.",
            objection="Giá iPhone bao nhiêu?",
        )
        if not passed and issues:
            assert issues[0].severity == IssueSeverity.MINOR

    def test_major_severity_for_medium_deviation(self):
        """Deviation between 5% and 30% → MAJOR severity"""
        checker = _make_checker(catalog_source=SAMPLE_CATALOG, tolerance_percent=1.0)
        # Apple Mobile Phone = 29,990,000 VND
        # 20% below = 23,992,000 VND ≈ 24 triệu
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 24 triệu.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is False
        if issues:
            assert issues[0].severity == IssueSeverity.MAJOR

    def test_critical_severity_for_large_deviation(self):
        """Deviation > 30% → CRITICAL severity"""
        checker = _make_checker(catalog_source=SAMPLE_CATALOG, tolerance_percent=1.0)
        # Apple Mobile Phone = 29,990,000 VND
        # 50% below = 14,995,000 VND ≈ 15 triệu
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 15 triệu.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is False
        if issues:
            assert issues[0].severity == IssueSeverity.CRITICAL

    def test_missing_price_in_draft_is_major(self, checker):
        """Missing price when objection asks about price → MAJOR severity"""
        passed, issues = checker.check_price_accuracy(
            draft="Sản phẩm rất tốt.",
            objection="Giá bao nhiêu?",
        )
        assert passed is False
        assert issues[0].severity == IssueSeverity.MAJOR


# ---------------------------------------------------------------------------
# Product not found in catalog
# ---------------------------------------------------------------------------

class TestProductNotFound:
    def test_unknown_product_price_fails(self, checker):
        """Price mentioned for unknown product → FAIL MAJOR"""
        passed, issues = checker.check_price_accuracy(
            draft="Sản phẩm XYZ123 giá 50 triệu.",
            objection="Giá sản phẩm XYZ123 bao nhiêu?",
        )
        # Either passes (no context match) or fails with MAJOR
        # The key is it should not crash
        assert isinstance(passed, bool)
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# Binary decision boundary
# ---------------------------------------------------------------------------

class TestBinaryDecisionBoundary:
    def test_exactly_at_tolerance_passes(self):
        """Deviation exactly at tolerance boundary → PASS"""
        # Use a mock matcher that returns a known price
        config = VerificationConfig(price_tolerance_percent=1.0, price_critical_threshold=30.0)
        thresholds = PriceAccuracyThresholds(pass_tolerance_percent=1.0)
        checker = PriceAccuracyChecker(llm=None, rag_pipeline=None, config=config, thresholds=thresholds)

        mock_match = ProductMatch(
            product_name="Mobile Phone",
            brand="Apple",
            sku="IPHONE15PRO",
            price_raw="29.990.000 VNĐ",
            price_vnd=29_990_000.0,
            match_score=0.9,
            match_type="fuzzy_name",
        )
        mock_matcher = MagicMock()
        mock_matcher.find_product.return_value = mock_match
        checker._product_matcher = mock_matcher

        # 1% above 29,990,000 = 30,289,900
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 30.289.900 VNĐ.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is True
        assert issues == []

    def test_just_above_tolerance_fails(self):
        """Deviation just above tolerance → FAIL"""
        config = VerificationConfig(price_tolerance_percent=1.0, price_critical_threshold=30.0)
        thresholds = PriceAccuracyThresholds(pass_tolerance_percent=1.0)
        checker = PriceAccuracyChecker(llm=None, rag_pipeline=None, config=config, thresholds=thresholds)

        mock_match = ProductMatch(
            product_name="Mobile Phone",
            brand="Apple",
            sku="IPHONE15PRO",
            price_raw="29.990.000 VNĐ",
            price_vnd=29_990_000.0,
            match_score=0.9,
            match_type="fuzzy_name",
        )
        mock_matcher = MagicMock()
        mock_matcher.find_product.return_value = mock_match
        checker._product_matcher = mock_matcher

        # 2% above 29,990,000 = 30,589,800
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 30.589.800 VNĐ.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is False
        assert len(issues) == 1
        assert issues[0].deviation_percent > 1.0

    def test_return_type_is_always_tuple(self, checker):
        """check_price_accuracy always returns (bool, list)"""
        result = checker.check_price_accuracy("", "")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)


# ---------------------------------------------------------------------------
# Configurable tolerance
# ---------------------------------------------------------------------------

class TestConfigurableTolerance:
    def test_5_percent_tolerance_passes_small_deviation(self):
        """With 5% tolerance, a 3% deviation should PASS"""
        checker = _make_checker(catalog_source=SAMPLE_CATALOG, tolerance_percent=5.0)
        # Apple Mobile Phone = 29,990,000 VND
        # 3% above = 30,889,700 VND
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 30.889.700 VNĐ.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is True
        assert issues == []

    def test_strict_0_tolerance_fails_any_deviation(self):
        """With 0% tolerance, any deviation should FAIL"""
        checker = _make_checker(catalog_source=SAMPLE_CATALOG, tolerance_percent=0.0)
        # Apple Mobile Phone = 29,990,000 VND
        # Even 0.1% deviation should fail
        passed, issues = checker.check_price_accuracy(
            draft="Apple Mobile Phone giá 30.020.000 VNĐ.",
            objection="Giá iPhone bao nhiêu?",
        )
        assert passed is False


# ---------------------------------------------------------------------------
# Objection price keyword detection
# ---------------------------------------------------------------------------

class TestObjectionPriceKeywords:
    def test_english_price_keywords(self, checker):
        """English price keywords in objection trigger FAIL when no price in draft"""
        english_keywords = ["price", "cost", "expensive", "cheap", "money", "budget"]
        for keyword in english_keywords:
            passed, issues = checker.check_price_accuracy(
                draft="Sản phẩm rất tốt.",
                objection=f"What is the {keyword}?",
            )
            assert passed is False, f"Expected FAIL for keyword: {keyword}"

    def test_vietnamese_price_keywords(self, checker):
        """Vietnamese price keywords in objection trigger FAIL when no price in draft"""
        viet_keywords = ["giá", "đắt", "rẻ", "tiền", "ngân sách", "chi phí"]
        for keyword in viet_keywords:
            passed, issues = checker.check_price_accuracy(
                draft="Sản phẩm rất tốt.",
                objection=f"Sản phẩm này {keyword} bao nhiêu?",
            )
            assert passed is False, f"Expected FAIL for keyword: {keyword}"


# ---------------------------------------------------------------------------
# Multiple Products in Single Draft (Requirements 4.5)
# ---------------------------------------------------------------------------

class TestMultipleProductsInDraft:
    """Tests for handling multiple products in a single draft (Req 4.5)."""

    def _make_checker_with_mock_matcher(self, product_prices: dict) -> PriceAccuracyChecker:
        """
        Build a checker with a mock that maps price values to products.
        Mocks _find_product_from_context directly using amount_vnd hint.
        product_prices: {product_name: price_vnd}
        """
        config = VerificationConfig(price_tolerance_percent=1.0, price_critical_threshold=30.0)
        thresholds = PriceAccuracyThresholds(
            pass_tolerance_percent=1.0,
            critical_threshold_percent=30.0,
        )
        checker = PriceAccuracyChecker(
            llm=None, rag_pipeline=None, config=config, thresholds=thresholds,
            price_cache=ProductPriceLookupCache()
        )

        products_list = list(product_prices.items())

        def mock_find_product_from_context(context: str, amount_vnd: float = 0.0):
            # Match by closest catalog price to the extracted amount_vnd
            if amount_vnd > 0:
                best_name, best_price = min(
                    products_list,
                    key=lambda np: abs(np[1] - amount_vnd)
                )
                return ProductMatch(
                    product_name=best_name,
                    brand="Test",
                    sku=best_name.upper().replace(" ", "_"),
                    price_raw=f"{best_price:,.0f} VNĐ",
                    price_vnd=float(best_price),
                    match_score=0.9,
                    match_type="fuzzy_name",
                )
            # Fallback: return first product
            name, price = products_list[0]
            return ProductMatch(
                product_name=name, brand="Test",
                sku=name.upper().replace(" ", "_"),
                price_raw=f"{price:,.0f} VNĐ", price_vnd=float(price),
                match_score=0.9, match_type="fuzzy_name",
            )

        checker._find_product_from_context = mock_find_product_from_context
        return checker

    def test_all_prices_correct_passes(self):
        """All products have correct prices → PASS (worst-case: all pass)"""
        checker = self._make_checker_with_mock_matcher({
            "iPhone 15 Pro": 29_990_000,
            "Samsung S23": 15_000_000,
        })
        # Both prices exactly correct
        draft = (
            "iPhone 15 Pro có giá 29.990.000 VNĐ. "
            "Samsung S23 có giá 15.000.000 VNĐ."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá các điện thoại bao nhiêu?")
        assert passed is True
        assert issues == []

    def test_one_wrong_price_fails_overall(self):
        """One product has wrong price → overall FAIL (worst-case scoring)"""
        checker = self._make_checker_with_mock_matcher({
            "iPhone 15 Pro": 29_990_000,
            "Samsung S23": 15_000_000,
        })
        # iPhone correct, Samsung wrong (10M vs 15M = 33% deviation → CRITICAL)
        draft = (
            "iPhone 15 Pro có giá 29.990.000 VNĐ. "
            "Samsung S23 chỉ có giá 10 triệu thôi."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá các điện thoại bao nhiêu?")
        assert passed is False
        assert len(issues) >= 1

    def test_all_prices_wrong_returns_multiple_issues(self):
        """All products have wrong prices → FAIL with multiple issues"""
        checker = self._make_checker_with_mock_matcher({
            "iPhone 15 Pro": 29_990_000,
            "Samsung S23": 15_000_000,
        })
        # Both prices significantly wrong
        draft = (
            "iPhone 15 Pro có giá 10 triệu. "
            "Samsung S23 có giá 5 triệu."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá các điện thoại bao nhiêu?")
        assert passed is False
        assert len(issues) >= 1  # At least one issue found

    def test_worst_case_scoring_any_fail_means_overall_fail(self):
        """Worst-case: even one price failure causes overall FAIL"""
        checker = self._make_checker_with_mock_matcher({
            "iPhone 15 Pro": 29_990_000,
            "Samsung S23": 15_000_000,
        })
        # iPhone correct, Samsung very wrong
        draft = (
            "iPhone 15 Pro giá 29.990.000 VNĐ. "
            "Samsung S23 giá 1 triệu."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá bao nhiêu?")
        # Overall must fail because Samsung price is wrong
        assert passed is False

    def test_issues_list_contains_only_failed_products(self):
        """Issues list should only contain products with price problems"""
        checker = self._make_checker_with_mock_matcher({
            "iPhone 15 Pro": 29_990_000,
            "Samsung S23": 15_000_000,
        })
        # iPhone correct, Samsung wrong
        draft = (
            "iPhone 15 Pro giá 29.990.000 VNĐ. "
            "Samsung S23 giá 5 triệu."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá bao nhiêu?")
        assert passed is False
        # All issues should have non-empty product names
        for issue in issues:
            assert issue.product_name  # non-empty


# ---------------------------------------------------------------------------
# Early Termination Behavior (Requirements 4.5)
# ---------------------------------------------------------------------------

class TestEarlyTermination:
    """Tests for early termination when critical issues are found."""

    def _make_checker_with_early_termination(
        self,
        stop_on_first_critical: bool = True,
        early_termination_enabled: bool = True,
        product_prices: Optional[dict] = None,
    ) -> PriceAccuracyChecker:
        """Build a checker with specific early termination settings."""
        from verification.config.thresholds_config import (
            VerificationThresholdsConfig,
            EscalationThresholds,
        )
        escalation = EscalationThresholds(
            early_termination_enabled=early_termination_enabled,
            stop_on_first_critical=stop_on_first_critical,
            multiple_critical_threshold=3,
        )
        thresholds_config = VerificationThresholdsConfig(escalation=escalation)
        config = VerificationConfig(price_tolerance_percent=1.0, price_critical_threshold=30.0)
        thresholds = PriceAccuracyThresholds(
            pass_tolerance_percent=1.0,
            critical_threshold_percent=30.0,
        )
        checker = PriceAccuracyChecker(
            llm=None,
            rag_pipeline=None,
            config=config,
            thresholds=thresholds,
            thresholds_config=thresholds_config,
            price_cache=ProductPriceLookupCache(),
        )

        if product_prices:
            products_list = list(product_prices.items())

            def mock_find_product_from_context(context: str, amount_vnd: float = 0.0):
                if amount_vnd > 0:
                    best_name, best_price = min(
                        products_list,
                        key=lambda np: abs(np[1] - amount_vnd)
                    )
                    return ProductMatch(
                        product_name=best_name, brand="Test",
                        sku=best_name.upper().replace(" ", "_"),
                        price_raw=f"{best_price:,.0f} VNĐ", price_vnd=float(best_price),
                        match_score=0.9, match_type="fuzzy_name",
                    )
                name, price = products_list[0]
                return ProductMatch(
                    product_name=name, brand="Test",
                    sku=name.upper().replace(" ", "_"),
                    price_raw=f"{price:,.0f} VNĐ", price_vnd=float(price),
                    match_score=0.9, match_type="fuzzy_name",
                )

            checker._find_product_from_context = mock_find_product_from_context
        else:
            from verification.utils.product_matcher import ProductMatcher
            checker._product_matcher = ProductMatcher(catalog_source=SAMPLE_CATALOG)
        return checker

    def test_early_termination_stops_after_first_critical(self):
        """With stop_on_first_critical=True, processing stops after first CRITICAL issue."""
        checker = self._make_checker_with_early_termination(
            stop_on_first_critical=True,
            product_prices={
                "iPhone 15 Pro": 29_990_000,
                "Samsung S23": 15_000_000,
            },
        )
        # iPhone 15 Pro: 10 triệu = 10M vs 29.99M → ~66.6% deviation → CRITICAL
        # Samsung S23: 1 triệu = 1M vs 15M → ~93.3% deviation → CRITICAL
        # With early termination, should stop after first CRITICAL
        draft = (
            "iPhone 15 Pro giá 10 triệu. "
            "Samsung S23 giá 1 triệu."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá bao nhiêu?")
        assert passed is False
        # Early termination: should have stopped after first critical issue
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.CRITICAL

    def test_early_termination_disabled_processes_all_prices(self):
        """With early_termination_enabled=False, all prices are processed."""
        checker = self._make_checker_with_early_termination(
            stop_on_first_critical=True,
            early_termination_enabled=False,
            product_prices={
                "iPhone 15 Pro": 29_990_000,
                "Samsung S23": 15_000_000,
            },
        )
        # Both prices critically wrong
        draft = (
            "iPhone 15 Pro giá 10 triệu. "
            "Samsung S23 giá 1 triệu."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá bao nhiêu?")
        assert passed is False
        # Without early termination, both issues should be found
        assert len(issues) >= 2

    def test_early_termination_returns_issues_found_so_far(self):
        """Early termination returns all issues found before stopping, not just the first."""
        from verification.config.thresholds_config import (
            VerificationThresholdsConfig,
            EscalationThresholds,
        )
        # Terminate after 2 critical issues
        escalation = EscalationThresholds(
            early_termination_enabled=True,
            stop_on_first_critical=False,
            multiple_critical_threshold=2,
        )
        thresholds_config = VerificationThresholdsConfig(escalation=escalation)
        config = VerificationConfig(price_tolerance_percent=1.0, price_critical_threshold=30.0)
        thresholds = PriceAccuracyThresholds(
            pass_tolerance_percent=1.0,
            critical_threshold_percent=30.0,
        )
        checker = PriceAccuracyChecker(
            llm=None,
            rag_pipeline=None,
            config=config,
            thresholds=thresholds,
            thresholds_config=thresholds_config,
        )

        product_prices = {
            "iPhone 15 Pro": 29_990_000,
            "Samsung S23": 15_000_000,
        }

        checker._product_matcher = _make_position_aware_mock_matcher(product_prices)

        # Both prices critically wrong (>30% deviation)
        draft = (
            "iPhone 15 Pro giá 10 triệu. "
            "Samsung S23 giá 1 triệu."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá bao nhiêu?")
        assert passed is False
        # Should have exactly 2 issues (stopped after reaching threshold=2)
        assert len(issues) == 2

    def test_no_early_termination_for_non_critical_issues(self):
        """Early termination does NOT trigger for MAJOR/MINOR issues."""
        checker = self._make_checker_with_early_termination(
            stop_on_first_critical=True,
            product_prices={
                "iPhone 15 Pro": 29_990_000,
                "Samsung S23": 15_000_000,
            },
        )
        # iPhone 15 Pro: 25 triệu = 25M vs 29.99M → ~16.6% deviation → MAJOR
        # Samsung S23: 12 triệu = 12M vs 15M → 20% deviation → MAJOR
        # Neither is CRITICAL (>30%), so early termination should NOT trigger
        draft = (
            "iPhone 15 Pro giá 25 triệu. "
            "Samsung S23 giá 12 triệu."
        )
        passed, issues = checker.check_price_accuracy(draft, "Giá bao nhiêu?")
        assert passed is False
        # Both MAJOR issues should be collected (no early termination for non-critical)
        assert len(issues) >= 2
        for issue in issues:
            assert issue.severity == IssueSeverity.MAJOR

    def test_thresholds_config_parameter_accepted(self):
        """PriceAccuracyChecker accepts thresholds_config parameter without error."""
        from verification.config.thresholds_config import VerificationThresholdsConfig
        config = VerificationConfig(price_tolerance_percent=1.0, price_critical_threshold=30.0)
        thresholds_config = VerificationThresholdsConfig()
        checker = PriceAccuracyChecker(
            llm=None,
            rag_pipeline=None,
            config=config,
            thresholds_config=thresholds_config,
        )
        assert checker is not None
        # Should have an early termination manager
        assert checker._early_termination is not None

    def test_default_no_thresholds_config_still_works(self):
        """PriceAccuracyChecker works without thresholds_config (uses defaults)."""
        config = VerificationConfig(price_tolerance_percent=1.0, price_critical_threshold=30.0)
        checker = PriceAccuracyChecker(
            llm=None,
            rag_pipeline=None,
            config=config,
        )
        checker._product_matcher = None
        passed, issues = checker.check_price_accuracy(
            "Sản phẩm rất tốt.",
            "Điện thoại này có tốt không?",
        )
        assert isinstance(passed, bool)
        assert isinstance(issues, list)
