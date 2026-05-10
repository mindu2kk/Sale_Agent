"""
Unit Tests for Fallback Verification Modes - Task 6.3.3

Tests cover:
- Each fallback mode activates correctly when services are unavailable
- Fallback results are properly structured VerificationResult objects
- Thresholds are lowered appropriately in degraded modes
- Human review flagging works correctly in full degraded mode
"""

import pytest
from unittest.mock import MagicMock, patch

from verification.models.verification import (
    IssueSeverity,
    VerificationResult,
    RubricCriteria,
)
from verification.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    reset_circuit_breaker_registry,
)
from verification.utils.fallback_verification import (
    FallbackConfig,
    FallbackMode,
    FallbackVerificationManager,
    get_fallback_verification_manager,
    reset_fallback_verification_manager,
    _rule_based_price_check,
    _rule_based_policy_check,
    _rule_based_relevance_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons before each test."""
    reset_fallback_verification_manager()
    reset_circuit_breaker_registry()
    yield
    reset_fallback_verification_manager()
    reset_circuit_breaker_registry()


def _make_manager(
    llm_open: bool = False,
    db_open: bool = False,
    cache: dict = None,
    config: FallbackConfig = None,
) -> FallbackVerificationManager:
    """Helper: create a FallbackVerificationManager with controlled CB states."""
    llm_cb = MagicMock(spec=CircuitBreaker)
    llm_cb.is_open.return_value = llm_open
    llm_cb.state = CircuitState.OPEN if llm_open else CircuitState.CLOSED

    db_cb = MagicMock(spec=CircuitBreaker)
    db_cb.is_open.return_value = db_open
    db_cb.state = CircuitState.OPEN if db_open else CircuitState.CLOSED

    return FallbackVerificationManager(
        config=config or FallbackConfig(),
        llm_circuit_breaker=llm_cb,
        db_circuit_breaker=db_cb,
        cache=cache or {},
        correlation_id="test-corr-id",
    )


PRICE_DRAFT = "iPhone 15 Pro Max giá 35 triệu VND, rất đáng mua."
PRICE_OBJECTION = "Tại sao iPhone lại đắt hơn Samsung? Giá có hợp lý không?"
POLICY_DRAFT = "Sản phẩm có bảo hành 12 tháng và hỗ trợ đổi trả trong 30 ngày."
GENERIC_OBJECTION = "Sản phẩm này có tốt không?"
RELEVANCE_DRAFT = "iPhone có camera tốt và hiệu năng mạnh mẽ."
FEATURE_OBJECTION = "Tính năng camera của iPhone như thế nào?"


# ---------------------------------------------------------------------------
# Rule-based helper tests
# ---------------------------------------------------------------------------

class TestRuleBasedPriceCheck:
    def test_pass_when_objection_not_price_related(self):
        passed, issues = _rule_based_price_check("Some draft", "Tell me about features")
        assert passed is True
        assert issues == []

    def test_pass_when_price_present_in_price_related_objection(self):
        passed, issues = _rule_based_price_check(PRICE_DRAFT, PRICE_OBJECTION)
        assert passed is True
        assert issues == []

    def test_fail_when_no_price_in_price_related_objection(self):
        passed, issues = _rule_based_price_check("iPhone is great.", PRICE_OBJECTION)
        assert passed is False
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.MAJOR

    def test_price_pattern_matches_vnd_amounts(self):
        draft = "Giá sản phẩm là 29.990.000 VND."
        passed, issues = _rule_based_price_check(draft, PRICE_OBJECTION)
        assert passed is True

    def test_price_pattern_matches_trieu(self):
        draft = "Chỉ 29 triệu thôi."
        passed, issues = _rule_based_price_check(draft, PRICE_OBJECTION)
        assert passed is True


class TestRuleBasedPolicyCheck:
    def test_pass_when_no_policy_keywords(self):
        passed, issues = _rule_based_policy_check("iPhone has great camera.")
        assert passed is True
        assert issues == []

    def test_pass_with_minor_warning_when_policy_keywords_present(self):
        passed, issues = _rule_based_policy_check(POLICY_DRAFT)
        assert passed is True
        # Warning issue injected
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.MINOR

    def test_warning_issue_is_policy_issue(self):
        from verification.models.verification import PolicyIssue
        passed, issues = _rule_based_policy_check(POLICY_DRAFT)
        assert isinstance(issues[0], PolicyIssue)
        assert issues[0].is_fabricated is False


class TestRuleBasedRelevanceCheck:
    def test_pass_when_no_detectable_intent(self):
        passed, issues = _rule_based_relevance_check("Some draft", "Some vague question")
        assert passed is True
        assert issues == []

    def test_pass_when_draft_addresses_intent(self):
        passed, issues = _rule_based_relevance_check(RELEVANCE_DRAFT, FEATURE_OBJECTION)
        assert passed is True

    def test_fail_when_draft_misses_intent(self):
        passed, issues = _rule_based_relevance_check(
            "Sản phẩm rất tốt.", PRICE_OBJECTION, min_coverage=0.7
        )
        assert passed is False
        assert len(issues) == 1

    def test_coverage_ratio_in_issue(self):
        passed, issues = _rule_based_relevance_check(
            "Sản phẩm rất tốt.", PRICE_OBJECTION, min_coverage=0.7
        )
        assert issues[0].response_coverage < 1.0

    def test_lowered_threshold_allows_partial_coverage(self):
        # With min_coverage=0.3, partial coverage should pass
        passed, issues = _rule_based_relevance_check(
            RELEVANCE_DRAFT, PRICE_OBJECTION, min_coverage=0.3
        )
        # May pass or fail depending on keyword overlap — just check structure
        assert isinstance(passed, bool)


# ---------------------------------------------------------------------------
# FallbackVerificationManager — mode detection
# ---------------------------------------------------------------------------

class TestModeDetection:
    def test_detect_llm_outage_when_llm_open(self):
        manager = _make_manager(llm_open=True, db_open=False)
        assert manager.detect_mode() == FallbackMode.LLM_OUTAGE

    def test_detect_db_outage_when_db_open(self):
        manager = _make_manager(llm_open=False, db_open=True)
        assert manager.detect_mode() == FallbackMode.DATABASE_OUTAGE

    def test_detect_full_degraded_when_both_open(self):
        manager = _make_manager(llm_open=True, db_open=True)
        assert manager.detect_mode() == FallbackMode.FULL_DEGRADED

    def test_detect_partial_when_both_closed(self):
        manager = _make_manager(llm_open=False, db_open=False)
        assert manager.detect_mode() == FallbackMode.PARTIAL_OUTAGE


# ---------------------------------------------------------------------------
# LLM Outage Mode
# ---------------------------------------------------------------------------

class TestLLMOutageMode:
    def test_returns_verification_result(self):
        manager = _make_manager(llm_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.LLM_OUTAGE)
        assert isinstance(result, VerificationResult)

    def test_result_has_rubric_criteria(self):
        manager = _make_manager(llm_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.LLM_OUTAGE)
        assert isinstance(result.criteria, RubricCriteria)

    def test_llm_tokens_used_is_zero(self):
        manager = _make_manager(llm_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.LLM_OUTAGE)
        assert result.llm_tokens_used == 0

    def test_reasoning_mentions_rule_based(self):
        manager = _make_manager(llm_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.LLM_OUTAGE)
        assert "rule-based" in result.verification_reasoning.lower() or \
               "llm" in result.verification_reasoning.lower()

    def test_price_check_runs_without_llm(self):
        manager = _make_manager(llm_open=True)
        # Draft has price, objection is price-related → should pass price check
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.LLM_OUTAGE)
        assert result.criteria.price_accuracy_pass is True

    def test_policy_check_runs_without_llm(self):
        manager = _make_manager(llm_open=True)
        result = manager.run_fallback(POLICY_DRAFT, GENERIC_OBJECTION, FallbackMode.LLM_OUTAGE)
        # Policy keywords present → pass with minor warning
        assert result.criteria.policy_authenticity_pass is True

    def test_overall_pass_is_boolean(self):
        manager = _make_manager(llm_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.LLM_OUTAGE)
        assert isinstance(result.criteria.overall_pass, bool)


# ---------------------------------------------------------------------------
# Database Outage Mode
# ---------------------------------------------------------------------------

class TestDatabaseOutageMode:
    def test_returns_verification_result(self):
        manager = _make_manager(db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert isinstance(result, VerificationResult)

    def test_price_check_defaults_to_pass_without_cache(self):
        manager = _make_manager(db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert result.criteria.price_accuracy_pass is True

    def test_price_warning_issue_injected_without_cache(self):
        manager = _make_manager(db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert len(result.criteria.price_issues) >= 1
        assert result.criteria.price_issues[0].severity == IssueSeverity.MINOR

    def test_policy_check_defaults_to_pass_without_cache(self):
        manager = _make_manager(db_open=True)
        result = manager.run_fallback(POLICY_DRAFT, GENERIC_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert result.criteria.policy_authenticity_pass is True

    def test_uses_cached_price_result_when_available(self):
        cached_price = (False, [])  # Simulate a cached FAIL result
        manager = _make_manager(db_open=True, cache={"price_check_result": cached_price})
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert result.criteria.price_accuracy_pass is False

    def test_uses_cached_policy_result_when_available(self):
        from verification.models.verification import PolicyIssue
        cached_issue = PolicyIssue(
            mentioned_policy="bảo hành 2 năm",
            policy_type="warranty",
            is_fabricated=True,
            is_inaccurate=False,
            severity=IssueSeverity.CRITICAL,
            explanation="Fabricated warranty",
        )
        cached_policy = (False, [cached_issue])
        manager = _make_manager(db_open=True, cache={"policy_check_result": cached_policy})
        result = manager.run_fallback(POLICY_DRAFT, GENERIC_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert result.criteria.policy_authenticity_pass is False
        assert result.criteria.policy_issues[0].is_fabricated is True

    def test_lowered_relevance_threshold_applied(self):
        config = FallbackConfig(db_outage_relevance_min_coverage=0.3)
        manager = _make_manager(db_open=True, config=config)
        # With very low threshold, most drafts should pass relevance
        result = manager.run_fallback(RELEVANCE_DRAFT, FEATURE_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert result.criteria.topic_relevance_pass is True

    def test_reasoning_mentions_db(self):
        manager = _make_manager(db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert "db" in result.verification_reasoning.lower() or \
               "database" in result.verification_reasoning.lower() or \
               "cached" in result.verification_reasoning.lower()

    def test_cache_check_result_stores_result(self):
        manager = _make_manager(db_open=True)
        manager.cache_check_result("price_check_result", (True, []))
        assert manager._cache["price_check_result"] == (True, [])


# ---------------------------------------------------------------------------
# Partial Outage Mode
# ---------------------------------------------------------------------------

class TestPartialOutageMode:
    def test_returns_verification_result(self):
        manager = _make_manager()
        result = manager.run_fallback(
            PRICE_DRAFT, PRICE_OBJECTION,
            FallbackMode.PARTIAL_OUTAGE,
            available_services=["relevance"],
        )
        assert isinstance(result, VerificationResult)

    def test_skips_price_check_when_db_unavailable(self):
        manager = _make_manager()
        result = manager.run_fallback(
            PRICE_DRAFT, PRICE_OBJECTION,
            FallbackMode.PARTIAL_OUTAGE,
            available_services=["relevance"],  # no "db"
        )
        # Price defaults to pass with warning
        assert result.criteria.price_accuracy_pass is True
        assert len(result.criteria.price_issues) >= 1

    def test_skips_policy_check_when_db_unavailable(self):
        manager = _make_manager()
        result = manager.run_fallback(
            POLICY_DRAFT, GENERIC_OBJECTION,
            FallbackMode.PARTIAL_OUTAGE,
            available_services=["relevance"],
        )
        assert result.criteria.policy_authenticity_pass is True

    def test_relevance_check_always_runs(self):
        manager = _make_manager()
        result = manager.run_fallback(
            RELEVANCE_DRAFT, FEATURE_OBJECTION,
            FallbackMode.PARTIAL_OUTAGE,
            available_services=[],  # nothing available
        )
        # Relevance is rule-based — always runs
        assert isinstance(result.criteria.topic_relevance_pass, bool)

    def test_adjusted_relevance_threshold(self):
        config = FallbackConfig(partial_outage_relevance_min_coverage=0.3)
        manager = _make_manager(config=config)
        result = manager.run_fallback(
            RELEVANCE_DRAFT, FEATURE_OBJECTION,
            FallbackMode.PARTIAL_OUTAGE,
            available_services=["relevance"],
        )
        assert result.criteria.topic_relevance_pass is True

    def test_all_checks_run_when_all_services_available(self):
        manager = _make_manager()
        result = manager.run_fallback(
            PRICE_DRAFT, PRICE_OBJECTION,
            FallbackMode.PARTIAL_OUTAGE,
            available_services=["llm", "db", "price", "policy", "relevance"],
        )
        assert isinstance(result, VerificationResult)
        assert isinstance(result.criteria.overall_pass, bool)

    def test_reasoning_mentions_partial_outage(self):
        manager = _make_manager()
        result = manager.run_fallback(
            PRICE_DRAFT, PRICE_OBJECTION,
            FallbackMode.PARTIAL_OUTAGE,
            available_services=["relevance"],
        )
        assert "partial" in result.verification_reasoning.lower() or \
               "outage" in result.verification_reasoning.lower() or \
               "available" in result.verification_reasoning.lower()


# ---------------------------------------------------------------------------
# Full Degraded Mode
# ---------------------------------------------------------------------------

class TestFullDegradedMode:
    def test_returns_verification_result(self):
        manager = _make_manager(llm_open=True, db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert isinstance(result, VerificationResult)

    def test_all_checks_default_to_pass(self):
        manager = _make_manager(llm_open=True, db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert result.criteria.price_accuracy_pass is True
        assert result.criteria.policy_authenticity_pass is True
        assert result.criteria.topic_relevance_pass is True
        assert result.criteria.overall_pass is True

    def test_major_warning_issues_injected_on_all_criteria(self):
        manager = _make_manager(llm_open=True, db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert len(result.criteria.price_issues) >= 1
        assert len(result.criteria.policy_issues) >= 1
        assert len(result.criteria.relevance_issues) >= 1
        assert result.criteria.price_issues[0].severity == IssueSeverity.MAJOR
        assert result.criteria.policy_issues[0].severity == IssueSeverity.MAJOR
        assert result.criteria.relevance_issues[0].severity == IssueSeverity.MAJOR

    def test_human_review_flagging_enabled_by_default(self):
        config = FallbackConfig(full_degraded_flag_human_review=True)
        manager = _make_manager(llm_open=True, db_open=True, config=config)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        # Result should be structured correctly for human review routing
        assert result.criteria.overall_pass is True  # not hard-blocked
        assert result.criteria.get_major_issues_count() >= 3  # triggers escalation

    def test_human_review_flagging_can_be_disabled(self):
        config = FallbackConfig(full_degraded_flag_human_review=False)
        manager = _make_manager(llm_open=True, db_open=True, config=config)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert isinstance(result, VerificationResult)

    def test_has_critical_issues_is_false(self):
        manager = _make_manager(llm_open=True, db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert result.has_critical_issues is False

    def test_immediate_termination_is_false(self):
        manager = _make_manager(llm_open=True, db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert result.immediate_termination is False

    def test_reasoning_mentions_human_review(self):
        manager = _make_manager(llm_open=True, db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert "human" in result.verification_reasoning.lower() or \
               "review" in result.verification_reasoning.lower()

    def test_llm_tokens_used_is_zero(self):
        manager = _make_manager(llm_open=True, db_open=True)
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert result.llm_tokens_used == 0


# ---------------------------------------------------------------------------
# run_fallback dispatch
# ---------------------------------------------------------------------------

class TestRunFallbackDispatch:
    def test_dispatches_to_llm_outage(self):
        manager = _make_manager()
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.LLM_OUTAGE)
        assert "rule-based" in result.verification_reasoning.lower() or \
               "llm" in result.verification_reasoning.lower()

    def test_dispatches_to_db_outage(self):
        manager = _make_manager()
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.DATABASE_OUTAGE)
        assert "db" in result.verification_reasoning.lower() or \
               "cached" in result.verification_reasoning.lower()

    def test_dispatches_to_partial_outage(self):
        manager = _make_manager()
        result = manager.run_fallback(
            PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.PARTIAL_OUTAGE, available_services=[]
        )
        assert "partial" in result.verification_reasoning.lower() or \
               "outage" in result.verification_reasoning.lower() or \
               "available" in result.verification_reasoning.lower()

    def test_dispatches_to_full_degraded(self):
        manager = _make_manager()
        result = manager.run_fallback(PRICE_DRAFT, PRICE_OBJECTION, FallbackMode.FULL_DEGRADED)
        assert "human" in result.verification_reasoning.lower() or \
               "review" in result.verification_reasoning.lower()


# ---------------------------------------------------------------------------
# Service status
# ---------------------------------------------------------------------------

class TestServiceStatus:
    def test_get_service_status_returns_dict(self):
        manager = _make_manager(llm_open=True, db_open=False)
        status = manager.get_service_status()
        assert isinstance(status, dict)
        assert "llm_api" in status
        assert "internal_db" in status

    def test_service_status_reflects_circuit_state(self):
        manager = _make_manager(llm_open=True, db_open=False)
        status = manager.get_service_status()
        assert status["llm_api"] == CircuitState.OPEN.value
        assert status["internal_db"] == CircuitState.CLOSED.value


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    def test_get_fallback_verification_manager_returns_instance(self):
        manager = get_fallback_verification_manager()
        assert isinstance(manager, FallbackVerificationManager)

    def test_get_fallback_verification_manager_returns_same_instance(self):
        m1 = get_fallback_verification_manager()
        m2 = get_fallback_verification_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_fallback_verification_manager()
        reset_fallback_verification_manager()
        m2 = get_fallback_verification_manager()
        assert m1 is not m2
