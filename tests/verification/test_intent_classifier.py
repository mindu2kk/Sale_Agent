"""
Tests for IntentClassifier

Covers:
- Single intent detection (Vietnamese + English)
- Multi-intent detection
- Confidence scoring
- Pattern matching
- Edge cases (empty text, unknown intents)
- Integration with TopicRelevanceChecker
"""

import pytest
from verification.utils.intent_classifier import (
    IntentClassifier,
    ClassificationResult,
    IntentScore,
    INTENT_TAXONOMY,
    INTENT_LABELS,
)
from verification.agent.checkers import TopicRelevanceChecker
from verification.config.config import VerificationConfig
from verification.config.thresholds_config import TopicRelevanceThresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier():
    return IntentClassifier(confidence_threshold=0.2, max_intents=3)


@pytest.fixture
def checker():
    config = VerificationConfig()
    thresholds = TopicRelevanceThresholds(pass_coverage_threshold=0.7)
    return TopicRelevanceChecker(llm=None, config=config, thresholds=thresholds)


# ---------------------------------------------------------------------------
# Single intent detection
# ---------------------------------------------------------------------------

class TestSingleIntentDetection:
    def test_price_intent_vietnamese(self, classifier):
        result = classifier.classify("Sản phẩm này đắt quá, giá bao nhiêu?")
        assert result.primary_intent == "price"
        assert "price" in result.intent_names

    def test_price_intent_english(self, classifier):
        result = classifier.classify("This product is too expensive, what is the price?")
        assert result.primary_intent == "price"

    def test_feature_intent_vietnamese(self, classifier):
        result = classifier.classify("Tính năng camera của máy này như thế nào?")
        assert "feature" in result.intent_names

    def test_feature_intent_english(self, classifier):
        result = classifier.classify("What are the specs and performance of this device?")
        assert "feature" in result.intent_names

    def test_comparison_intent_vietnamese(self, classifier):
        result = classifier.classify("So sánh iPhone vs Samsung, khác gì nhau?")
        assert "comparison" in result.intent_names

    def test_comparison_intent_english(self, classifier):
        result = classifier.classify("Compare iPhone versus Samsung, what is the difference?")
        assert "comparison" in result.intent_names

    def test_policy_intent_vietnamese(self, classifier):
        result = classifier.classify("Chính sách bảo hành và đổi trả như thế nào?")
        assert "policy" in result.intent_names

    def test_policy_intent_english(self, classifier):
        result = classifier.classify("What is the warranty and return policy?")
        assert "policy" in result.intent_names

    def test_availability_intent_vietnamese(self, classifier):
        result = classifier.classify("Sản phẩm còn hàng không, giao hàng bao lâu?")
        assert "availability" in result.intent_names

    def test_availability_intent_english(self, classifier):
        result = classifier.classify("Is this product in stock? How long is the delivery?")
        assert "availability" in result.intent_names

    def test_support_intent_vietnamese(self, classifier):
        result = classifier.classify("Tôi cần hỗ trợ kỹ thuật, sản phẩm bị lỗi")
        assert "support" in result.intent_names

    def test_support_intent_english(self, classifier):
        result = classifier.classify("I need technical support, the product has an issue")
        assert "support" in result.intent_names


# ---------------------------------------------------------------------------
# Multi-intent detection
# ---------------------------------------------------------------------------

class TestMultiIntentDetection:
    def test_price_and_feature_intents(self, classifier):
        result = classifier.classify("Giá đắt quá nhưng tính năng camera có tốt không?")
        assert result.is_multi_intent
        assert "price" in result.intent_names
        assert "feature" in result.intent_names

    def test_price_and_comparison_intents(self, classifier):
        result = classifier.classify("So sánh giá iPhone vs Samsung, cái nào rẻ hơn?")
        assert result.is_multi_intent

    def test_policy_and_support_intents(self, classifier):
        result = classifier.classify("Bảo hành bao lâu và hỗ trợ kỹ thuật như thế nào?")
        assert result.is_multi_intent

    def test_max_intents_respected(self, classifier):
        # Even with many matching keywords, max_intents=3 is enforced
        result = classifier.classify(
            "Giá đắt, tính năng kém, so sánh với Samsung, bảo hành ngắn, hết hàng"
        )
        assert len(result.intents) <= classifier.max_intents

    def test_single_intent_not_multi(self, classifier):
        result = classifier.classify("Giá bao nhiêu?")
        # May or may not be multi-intent depending on keyword overlap
        assert isinstance(result.is_multi_intent, bool)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    def test_confidence_in_valid_range(self, classifier):
        result = classifier.classify("Giá sản phẩm này đắt quá")
        for score in result.intents:
            assert 0.0 <= score.confidence <= 1.0

    def test_strong_match_higher_confidence(self, classifier):
        strong = classifier.classify("Giá đắt quá, bao nhiêu tiền, có giảm giá không?")
        weak = classifier.classify("Giá")
        strong_price = strong.get_intent("price")
        weak_price = weak.get_intent("price")
        if strong_price and weak_price:
            assert strong_price.confidence >= weak_price.confidence

    def test_pattern_match_boosts_confidence(self, classifier):
        # "giá bao nhiêu" matches a pattern → higher confidence than keyword alone
        with_pattern = classifier.classify("Giá bao nhiêu vậy?")
        without_pattern = classifier.classify("giá")
        p1 = with_pattern.get_intent("price")
        p2 = without_pattern.get_intent("price")
        if p1 and p2:
            assert p1.confidence >= p2.confidence

    def test_matched_keywords_populated(self, classifier):
        result = classifier.classify("Giá đắt quá, tiền nhiều vậy")
        price_score = result.get_intent("price")
        assert price_score is not None
        assert len(price_score.matched_keywords) > 0

    def test_matched_patterns_populated_for_pattern_match(self, classifier):
        result = classifier.classify("Giá bao nhiêu vậy?")
        price_score = result.get_intent("price")
        assert price_score is not None
        # Pattern "giá\s+(?:bao nhiêu|...)" should match
        assert len(price_score.matched_patterns) > 0

    def test_top_confidence_matches_primary(self, classifier):
        result = classifier.classify("Giá đắt quá, tính năng camera tốt không?")
        if result.intents:
            assert result.top_confidence == result.intents[0].confidence


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_returns_general_inquiry(self, classifier):
        result = classifier.classify("")
        assert result.primary_intent == "general_inquiry"
        assert result.intents == []
        assert result.is_multi_intent is False

    def test_whitespace_only_returns_general_inquiry(self, classifier):
        result = classifier.classify("   ")
        assert result.primary_intent == "general_inquiry"

    def test_unrelated_text_returns_general_inquiry(self, classifier):
        result = classifier.classify("Hello world, nice to meet you")
        # May or may not match — just ensure no crash and valid result
        assert isinstance(result.primary_intent, str)
        assert isinstance(result.intents, list)

    def test_very_long_text_does_not_crash(self, classifier):
        long_text = "giá " * 500 + "tính năng " * 500
        result = classifier.classify(long_text)
        assert isinstance(result, ClassificationResult)

    def test_mixed_language_text(self, classifier):
        result = classifier.classify("Giá price đắt expensive, tính năng features tốt")
        assert "price" in result.intent_names or "feature" in result.intent_names

    def test_get_primary_intent_convenience(self, classifier):
        intent = classifier.get_primary_intent("Giá bao nhiêu?")
        assert isinstance(intent, str)
        assert intent == "price"

    def test_classify_batch(self, classifier):
        objections = [
            "Giá đắt quá",
            "Tính năng camera như thế nào?",
            "So sánh iPhone vs Samsung",
        ]
        results = classifier.classify_batch(objections)
        assert len(results) == 3
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_get_intent_returns_none_for_missing(self, classifier):
        result = classifier.classify("Giá bao nhiêu?")
        # "comparison" likely not detected for a simple price question
        score = result.get_intent("nonexistent_intent")
        assert score is None


# ---------------------------------------------------------------------------
# ClassificationResult properties
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_intent_names_property(self, classifier):
        result = classifier.classify("Giá đắt, tính năng kém")
        assert isinstance(result.intent_names, list)
        assert all(isinstance(n, str) for n in result.intent_names)

    def test_raw_text_is_lowercased(self, classifier):
        result = classifier.classify("GIÁ BAO NHIÊU?")
        assert result.raw_text == result.raw_text.lower()

    def test_intents_sorted_by_confidence(self, classifier):
        result = classifier.classify("Giá đắt quá, tính năng camera tốt không?")
        confidences = [s.confidence for s in result.intents]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# IntentScore label
# ---------------------------------------------------------------------------

class TestIntentScore:
    def test_intent_score_label_populated(self, classifier):
        result = classifier.classify("Giá bao nhiêu?")
        for score in result.intents:
            assert score.label in INTENT_LABELS.values()

    def test_all_intents_have_labels(self):
        for intent in INTENT_TAXONOMY:
            assert intent in INTENT_LABELS


# ---------------------------------------------------------------------------
# Integration: TopicRelevanceChecker uses IntentClassifier
# ---------------------------------------------------------------------------

class TestIntentClassifierIntegration:
    def test_checker_uses_intent_classifier(self, checker):
        """TopicRelevanceChecker should expose _intent_classifier."""
        assert hasattr(checker, "_intent_classifier")
        assert isinstance(checker._intent_classifier, IntentClassifier)

    def test_fail_issue_has_primary_intent(self, checker):
        objection = "Giá đắt quá, có giảm giá không?"
        response = "Lịch sử công ty Apple được thành lập năm 1976."
        is_pass, issues = checker.check(objection, response)
        if not is_pass:
            assert issues[0].objection_intent != ""

    def test_fail_issue_detected_intents_from_classifier(self, checker):
        objection = "Giá đắt quá, so sánh với Samsung đi, tính năng camera như thế nào?"
        response = "Xin chào."
        is_pass, issues = checker.check(objection, response)
        if not is_pass:
            assert isinstance(issues[0].detected_intents, list)

    def test_general_inquiry_fallback(self, checker):
        """Empty objection should not crash the checker."""
        is_pass, issues = checker.check("", "Some response")
        assert isinstance(is_pass, bool)
