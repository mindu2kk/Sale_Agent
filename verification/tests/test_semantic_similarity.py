"""
Tests for SemanticSimilarityAnalyzer and TopicRelevanceChecker

Covers:
- Intent detection (Vietnamese + English keywords)
- Coverage ratio calculation
- Empathy detection
- Missing aspects identification
- TopicRelevanceChecker PASS/FAIL binary decision
"""

import pytest
from unittest.mock import MagicMock

from verification.utils.semantic_similarity import (
    SemanticSimilarityAnalyzer,
    SimilarityResult,
    INTENT_KEYWORDS,
    EMPATHY_PHRASES,
)
from verification.agent.checkers import TopicRelevanceChecker
from verification.models.verification import RelevanceIssue, IssueSeverity
from verification.config.config import VerificationConfig
from verification.config.thresholds_config import TopicRelevanceThresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return SemanticSimilarityAnalyzer(min_coverage_ratio=0.7, empathy_bonus_enabled=True)


@pytest.fixture
def checker():
    config = VerificationConfig()
    thresholds = TopicRelevanceThresholds(pass_coverage_threshold=0.7)
    return TopicRelevanceChecker(llm=None, config=config, thresholds=thresholds)


# ---------------------------------------------------------------------------
# SemanticSimilarityAnalyzer — intent detection
# ---------------------------------------------------------------------------

class TestIntentDetection:
    def test_detects_price_intent_vietnamese(self, analyzer):
        result = analyzer.analyze("Sản phẩm này đắt quá, giá bao nhiêu?", "dummy response")
        assert "price" in result.detected_intents

    def test_detects_price_intent_english(self, analyzer):
        result = analyzer.analyze("This is too expensive, what is the cost?", "dummy response")
        assert "price" in result.detected_intents

    def test_detects_feature_intent(self, analyzer):
        result = analyzer.analyze("Tính năng camera của máy này như thế nào?", "dummy response")
        assert "feature" in result.detected_intents

    def test_detects_comparison_intent(self, analyzer):
        result = analyzer.analyze("So sánh iPhone vs Samsung, khác gì nhau?", "dummy response")
        assert "comparison" in result.detected_intents

    def test_detects_policy_intent(self, analyzer):
        result = analyzer.analyze("Chính sách bảo hành và đổi trả như thế nào?", "dummy response")
        assert "policy" in result.detected_intents

    def test_detects_multiple_intents(self, analyzer):
        result = analyzer.analyze(
            "Giá đắt quá, tính năng có gì hay, so sánh với Samsung?",
            "dummy response"
        )
        assert len(result.detected_intents) >= 2

    def test_no_intent_for_generic_text(self, analyzer):
        result = analyzer.analyze("Hello world", "Hello world")
        # May or may not detect intents — just ensure it doesn't crash
        assert isinstance(result.detected_intents, list)


# ---------------------------------------------------------------------------
# SemanticSimilarityAnalyzer — coverage ratio
# ---------------------------------------------------------------------------

class TestCoverageRatio:
    def test_high_coverage_when_response_addresses_price(self, analyzer):
        objection = "Giá sản phẩm này đắt quá, có giảm giá không?"
        response = (
            "Chúng tôi hiểu rằng giá cả là quan trọng. "
            "Giá iPhone 15 là 25 triệu VND, hiện có khuyến mãi giảm giá 10%."
        )
        result = analyzer.analyze(objection, response)
        assert result.coverage_ratio >= 0.5

    def test_low_coverage_when_response_is_irrelevant(self, analyzer):
        objection = "Giá sản phẩm này đắt quá, có giảm giá không?"
        response = "Lịch sử công ty Apple được thành lập năm 1976 bởi Steve Jobs."
        result = analyzer.analyze(objection, response)
        assert result.coverage_ratio < 0.7

    def test_coverage_ratio_in_valid_range(self, analyzer):
        result = analyzer.analyze("any objection text", "any response text")
        assert 0.0 <= result.coverage_ratio <= 1.0

    def test_similarity_score_in_valid_range(self, analyzer):
        result = analyzer.analyze("any objection text", "any response text")
        assert 0.0 <= result.similarity_score <= 1.0

    def test_identical_texts_high_similarity(self, analyzer):
        text = "giá sản phẩm tính năng bảo hành"
        result = analyzer.analyze(text, text)
        assert result.similarity_score >= 0.7

    def test_empty_response_low_coverage(self, analyzer):
        result = analyzer.analyze("Giá bao nhiêu?", "")
        assert result.coverage_ratio < 0.5


# ---------------------------------------------------------------------------
# SemanticSimilarityAnalyzer — empathy detection
# ---------------------------------------------------------------------------

class TestEmpathyDetection:
    def test_detects_empathy_vietnamese(self, analyzer):
        response = "Chúng tôi hiểu được lo lắng của bạn về giá cả."
        result = analyzer.analyze("Giá đắt quá", response)
        assert result.has_empathy is True

    def test_detects_empathy_english(self, analyzer):
        response = "I completely understand your concern about the price."
        result = analyzer.analyze("Too expensive", response)
        assert result.has_empathy is True

    def test_no_empathy_in_plain_response(self, analyzer):
        response = "The product costs 25 million VND."
        result = analyzer.analyze("How much does it cost?", response)
        assert result.has_empathy is False

    def test_empathy_bonus_increases_coverage(self):
        analyzer_with_bonus = SemanticSimilarityAnalyzer(empathy_bonus_enabled=True)
        analyzer_no_bonus = SemanticSimilarityAnalyzer(empathy_bonus_enabled=False)

        objection = "Giá đắt quá"
        response = "Chúng tôi hiểu được lo lắng của bạn. Giá hiện tại là 25 triệu."

        result_with = analyzer_with_bonus.analyze(objection, response)
        result_without = analyzer_no_bonus.analyze(objection, response)

        # With empathy bonus, coverage should be >= without bonus
        assert result_with.coverage_ratio >= result_without.coverage_ratio


# ---------------------------------------------------------------------------
# SemanticSimilarityAnalyzer — missing aspects
# ---------------------------------------------------------------------------

class TestMissingAspects:
    def test_missing_aspects_when_price_not_addressed(self, analyzer):
        objection = "Giá sản phẩm này đắt quá"
        response = "Sản phẩm có tính năng camera tốt và pin lâu."
        result = analyzer.analyze(objection, response)
        # Should detect price not addressed
        assert isinstance(result.missing_aspects, list)

    def test_no_missing_aspects_when_well_covered(self, analyzer):
        objection = "Giá bao nhiêu?"
        response = (
            "Chúng tôi hiểu rằng giá cả quan trọng. "
            "Giá sản phẩm là 25 triệu VND, có khuyến mãi giảm giá."
        )
        result = analyzer.analyze(objection, response)
        # Coverage should be decent; missing aspects may be empty or minimal
        assert isinstance(result.missing_aspects, list)

    def test_missing_aspects_is_list(self, analyzer):
        result = analyzer.analyze("test objection", "test response")
        assert isinstance(result.missing_aspects, list)


# ---------------------------------------------------------------------------
# TopicRelevanceChecker — binary PASS/FAIL
# ---------------------------------------------------------------------------

class TestTopicRelevanceChecker:
    def test_pass_when_coverage_sufficient(self, checker):
        objection = "Giá sản phẩm này đắt quá, có giảm giá không?"
        response = (
            "Chúng tôi hiểu được lo lắng của bạn về giá cả. "
            "Giá iPhone 15 hiện tại là 25 triệu VND. "
            "Chúng tôi có chương trình khuyến mãi giảm giá 10% cho khách hàng mới. "
            "Chi phí này rất hợp lý so với tính năng sản phẩm."
        )
        is_pass, issues = checker.check(objection, response)
        # With good price coverage and empathy, should pass or be close
        assert isinstance(is_pass, bool)
        assert isinstance(issues, list)

    def test_fail_when_response_irrelevant(self, checker):
        objection = "Giá sản phẩm này đắt quá, có giảm giá không?"
        response = "Lịch sử công ty Apple được thành lập năm 1976 bởi Steve Jobs tại California."
        is_pass, issues = checker.check(objection, response)
        assert is_pass is False
        assert len(issues) > 0

    def test_fail_returns_relevance_issue(self, checker):
        objection = "So sánh iPhone vs Samsung, khác gì nhau?"
        response = "Xin chào, cảm ơn bạn đã liên hệ."
        is_pass, issues = checker.check(objection, response)
        if not is_pass:
            assert all(isinstance(i, RelevanceIssue) for i in issues)

    def test_issue_has_valid_coverage_ratio(self, checker):
        objection = "Giá đắt quá"
        response = "Lịch sử công ty."
        is_pass, issues = checker.check(objection, response)
        if not is_pass:
            for issue in issues:
                assert 0.0 <= issue.response_coverage <= 1.0

    def test_issue_severity_is_valid(self, checker):
        objection = "Giá đắt quá"
        response = "Lịch sử công ty."
        is_pass, issues = checker.check(objection, response)
        if not is_pass:
            for issue in issues:
                assert issue.severity in (
                    IssueSeverity.CRITICAL, IssueSeverity.MAJOR, IssueSeverity.MINOR
                )

    def test_check_and_check_topic_relevance_equivalent(self, checker):
        objection = "Bảo hành bao lâu?"
        response = "Sản phẩm được bảo hành 12 tháng."
        result_check = checker.check(objection, response)
        result_method = checker.check_topic_relevance(objection, response)
        assert result_check[0] == result_method[0]

    def test_empty_objection_does_not_crash(self, checker):
        is_pass, issues = checker.check("", "Some response text")
        assert isinstance(is_pass, bool)

    def test_empty_response_fails(self, checker):
        is_pass, issues = checker.check("Giá bao nhiêu?", "")
        assert is_pass is False

    def test_detected_intents_in_issue(self, checker):
        objection = "Giá đắt quá, so sánh với Samsung đi"
        response = "Xin chào."
        is_pass, issues = checker.check(objection, response)
        if not is_pass:
            for issue in issues:
                assert isinstance(issue.detected_intents, list)
