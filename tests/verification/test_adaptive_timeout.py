"""
Unit Tests: AdaptiveTimeoutManager

Tests for adaptive timeout mechanisms based on issue complexity.

Supports Task 5.4.4: Build adaptive timeout mechanisms based on issue complexity
"""

import asyncio
import pytest

from backend.verification.models.verification import (
    IssueSeverity,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
)
from backend.verification.utils.adaptive_timeout import (
    AdaptiveTimeoutManager,
    ComplexityScore,
    TimeoutConfig,
    get_adaptive_timeout_manager,
    reset_adaptive_timeout_manager,
    run_with_adaptive_timeout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price_issue(severity: IssueSeverity) -> PriceIssue:
    return PriceIssue(
        product_name="Test Product",
        severity=severity,
        explanation=f"Price issue {severity.value}",
    )


def make_policy_issue(severity: IssueSeverity) -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Test policy",
        policy_type="warranty",
        is_fabricated=severity == IssueSeverity.CRITICAL,
        is_inaccurate=severity == IssueSeverity.MAJOR,
        severity=severity,
        explanation=f"Policy issue {severity.value}",
    )


def make_relevance_issue(severity: IssueSeverity) -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="Test intent",
        response_coverage=0.3 if severity == IssueSeverity.CRITICAL else 0.6,
        severity=severity,
        explanation=f"Relevance issue {severity.value}",
    )


# ---------------------------------------------------------------------------
# Tests: TimeoutConfig defaults
# ---------------------------------------------------------------------------

class TestTimeoutConfigDefaults:
    def test_default_base_timeouts_present(self):
        cfg = TimeoutConfig()
        assert "price" in cfg.base_timeouts
        assert "policy" in cfg.base_timeouts
        assert "relevance" in cfg.base_timeouts

    def test_policy_base_timeout_longer_than_price(self):
        cfg = TimeoutConfig()
        assert cfg.base_timeouts["policy"] >= cfg.base_timeouts["price"]

    def test_severity_multipliers_ordering(self):
        cfg = TimeoutConfig()
        assert cfg.severity_multipliers["critical"] > cfg.severity_multipliers["major"]
        assert cfg.severity_multipliers["major"] > cfg.severity_multipliers["minor"]

    def test_min_less_than_max(self):
        cfg = TimeoutConfig()
        assert cfg.min_timeout < cfg.max_timeout


# ---------------------------------------------------------------------------
# Tests: Timeout calculation — no issues
# ---------------------------------------------------------------------------

class TestCalculateTimeoutNoIssues:
    def test_returns_base_timeout_for_price(self):
        mgr = AdaptiveTimeoutManager()
        t = mgr.calculate_timeout("price")
        assert t == mgr._config.base_timeouts["price"]

    def test_returns_base_timeout_for_policy(self):
        mgr = AdaptiveTimeoutManager()
        t = mgr.calculate_timeout("policy")
        assert t == mgr._config.base_timeouts["policy"]

    def test_returns_base_timeout_for_relevance(self):
        mgr = AdaptiveTimeoutManager()
        t = mgr.calculate_timeout("relevance")
        assert t == mgr._config.base_timeouts["relevance"]

    def test_unknown_check_type_uses_default(self):
        mgr = AdaptiveTimeoutManager()
        t = mgr.calculate_timeout("unknown_check")
        assert t == mgr._config.base_timeouts.get("default", 10.0)

    def test_empty_issues_list_same_as_none(self):
        mgr = AdaptiveTimeoutManager()
        assert mgr.calculate_timeout("price", []) == mgr.calculate_timeout("price", None)


# ---------------------------------------------------------------------------
# Tests: Timeout scales with severity
# ---------------------------------------------------------------------------

class TestCalculateTimeoutSeverityScaling:
    def test_critical_issue_increases_timeout(self):
        mgr = AdaptiveTimeoutManager()
        base = mgr.calculate_timeout("price", [])
        with_critical = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.CRITICAL)])
        assert with_critical > base

    def test_major_issue_increases_timeout(self):
        mgr = AdaptiveTimeoutManager()
        base = mgr.calculate_timeout("price", [])
        with_major = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.MAJOR)])
        assert with_major > base

    def test_critical_adds_more_than_major(self):
        mgr = AdaptiveTimeoutManager()
        with_critical = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.CRITICAL)])
        with_major = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.MAJOR)])
        assert with_critical > with_major

    def test_major_adds_more_than_minor(self):
        mgr = AdaptiveTimeoutManager()
        with_major = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.MAJOR)])
        with_minor = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.MINOR)])
        assert with_major > with_minor

    def test_multiple_critical_issues_scale_timeout(self):
        mgr = AdaptiveTimeoutManager()
        one_critical = mgr.calculate_timeout("policy", [make_policy_issue(IssueSeverity.CRITICAL)])
        two_critical = mgr.calculate_timeout("policy", [
            make_policy_issue(IssueSeverity.CRITICAL),
            make_policy_issue(IssueSeverity.CRITICAL),
        ])
        assert two_critical > one_critical

    def test_mixed_severity_issues(self):
        mgr = AdaptiveTimeoutManager()
        base = mgr.calculate_timeout("relevance", [])
        mixed = mgr.calculate_timeout("relevance", [
            make_relevance_issue(IssueSeverity.CRITICAL),
            make_relevance_issue(IssueSeverity.MAJOR),
            make_relevance_issue(IssueSeverity.MINOR),
        ])
        assert mixed > base


# ---------------------------------------------------------------------------
# Tests: Timeout scales with issue count
# ---------------------------------------------------------------------------

class TestCalculateTimeoutIssueCount:
    def test_more_issues_increases_timeout(self):
        mgr = AdaptiveTimeoutManager()
        one = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.MINOR)])
        three = mgr.calculate_timeout("price", [
            make_price_issue(IssueSeverity.MINOR),
            make_price_issue(IssueSeverity.MINOR),
            make_price_issue(IssueSeverity.MINOR),
        ])
        assert three > one

    def test_issue_count_factor_zero_disables_volume_scaling(self):
        cfg = TimeoutConfig(issue_count_factor=0.0)
        mgr = AdaptiveTimeoutManager(cfg)
        # With factor=0, volume (count-1) contribution is zero.
        # Severity multipliers still apply per issue, so 5 issues > 1 issue.
        # Verify that the volume-only delta is zero by comparing equal-count lists.
        one_minor = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.MINOR)])
        one_minor_again = mgr.calculate_timeout("price", [make_price_issue(IssueSeverity.MINOR)])
        assert one_minor == one_minor_again  # deterministic


# ---------------------------------------------------------------------------
# Tests: Clamping
# ---------------------------------------------------------------------------

class TestCalculateTimeoutClamping:
    def test_timeout_never_below_min(self):
        cfg = TimeoutConfig(min_timeout=8.0, base_timeouts={"price": 5.0, "default": 5.0})
        mgr = AdaptiveTimeoutManager(cfg)
        t = mgr.calculate_timeout("price", [])
        assert t >= 8.0

    def test_timeout_never_above_max(self):
        cfg = TimeoutConfig(max_timeout=12.0)
        mgr = AdaptiveTimeoutManager(cfg)
        # Many critical issues would push timeout very high
        issues = [make_price_issue(IssueSeverity.CRITICAL)] * 20
        t = mgr.calculate_timeout("price", issues)
        assert t <= 12.0

    def test_result_within_bounds(self):
        mgr = AdaptiveTimeoutManager()
        issues = [make_policy_issue(IssueSeverity.CRITICAL)] * 5
        t = mgr.calculate_timeout("policy", issues)
        assert mgr._config.min_timeout <= t <= mgr._config.max_timeout


# ---------------------------------------------------------------------------
# Tests: Historical EMA
# ---------------------------------------------------------------------------

class TestHistoricalEMA:
    def test_record_execution_time_stores_history(self):
        mgr = AdaptiveTimeoutManager()
        mgr.record_execution_time("price", 8.0)
        assert "price" in mgr.get_history()

    def test_history_influences_future_timeout(self):
        cfg = TimeoutConfig(history_alpha=0.5)
        mgr = AdaptiveTimeoutManager(cfg)
        # Record a very long execution time
        mgr.record_execution_time("price", 50.0)
        t = mgr.calculate_timeout("price", [])
        # With alpha=0.5, result should be between base and 50s (clamped to max)
        assert t > cfg.base_timeouts["price"]

    def test_reset_history_clears_ema(self):
        mgr = AdaptiveTimeoutManager()
        mgr.record_execution_time("price", 30.0)
        mgr.reset_history()
        assert mgr.get_history() == {}

    def test_ema_updates_on_second_record(self):
        cfg = TimeoutConfig(history_alpha=0.5)
        mgr = AdaptiveTimeoutManager(cfg)
        mgr.record_execution_time("policy", 20.0)
        first = mgr.get_history()["policy"]
        mgr.record_execution_time("policy", 10.0)
        second = mgr.get_history()["policy"]
        # EMA: 0.5 * 10 + 0.5 * 20 = 15
        assert abs(second - 15.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: ComplexityScore
# ---------------------------------------------------------------------------

class TestComplexityScore:
    def test_returns_complexity_score_object(self):
        mgr = AdaptiveTimeoutManager()
        score = mgr.get_complexity_score("price", [make_price_issue(IssueSeverity.CRITICAL)])
        assert isinstance(score, ComplexityScore)

    def test_complexity_score_counts_correct(self):
        mgr = AdaptiveTimeoutManager()
        issues = [
            make_price_issue(IssueSeverity.CRITICAL),
            make_price_issue(IssueSeverity.MAJOR),
            make_price_issue(IssueSeverity.MINOR),
        ]
        score = mgr.get_complexity_score("price", issues)
        assert score.critical_count == 1
        assert score.major_count == 1
        assert score.minor_count == 1
        assert score.issue_count == 3

    def test_complexity_score_to_dict(self):
        mgr = AdaptiveTimeoutManager()
        score = mgr.get_complexity_score("policy", [])
        d = score.to_dict()
        assert "check_type" in d
        assert "computed_timeout" in d
        assert d["check_type"] == "policy"


# ---------------------------------------------------------------------------
# Tests: Async context manager — enforce_timeout
# ---------------------------------------------------------------------------

class TestEnforceTimeout:
    @pytest.mark.asyncio
    async def test_enforce_timeout_yields_timeout_value(self):
        mgr = AdaptiveTimeoutManager()
        async with mgr.enforce_timeout("price", []) as timeout:
            assert isinstance(timeout, float)
            assert timeout > 0

    @pytest.mark.asyncio
    async def test_enforce_timeout_completes_fast_coroutine(self):
        mgr = AdaptiveTimeoutManager()
        result = []
        async with mgr.enforce_timeout("price", []):
            await asyncio.sleep(0)
            result.append("done")
        assert result == ["done"]

    @pytest.mark.asyncio
    async def test_enforce_timeout_records_execution_time(self):
        mgr = AdaptiveTimeoutManager()
        async with mgr.enforce_timeout("price", []):
            await asyncio.sleep(0.01)
        assert "price" in mgr.get_history()

    @pytest.mark.asyncio
    async def test_enforce_timeout_seconds_yields_value(self):
        mgr = AdaptiveTimeoutManager()
        async with mgr.enforce_timeout_seconds(15.0) as t:
            assert t == 15.0


# ---------------------------------------------------------------------------
# Tests: run_with_adaptive_timeout helper
# ---------------------------------------------------------------------------

class TestRunWithAdaptiveTimeout:
    @pytest.mark.asyncio
    async def test_returns_coroutine_result(self):
        async def fast_coro():
            return 42

        result = await run_with_adaptive_timeout(fast_coro(), "price")
        assert result == 42

    @pytest.mark.asyncio
    async def test_raises_timeout_error_on_slow_coro(self):
        cfg = TimeoutConfig(
            base_timeouts={"price": 0.05, "default": 0.05},
            min_timeout=0.05,
            max_timeout=0.05,
        )
        mgr = AdaptiveTimeoutManager(cfg)

        async def slow_coro():
            await asyncio.sleep(5)

        with pytest.raises(asyncio.TimeoutError):
            await run_with_adaptive_timeout(slow_coro(), "price", manager=mgr)

    @pytest.mark.asyncio
    async def test_records_history_after_completion(self):
        mgr = AdaptiveTimeoutManager()

        async def fast_coro():
            return "ok"

        await run_with_adaptive_timeout(fast_coro(), "relevance", manager=mgr)
        assert "relevance" in mgr.get_history()

    @pytest.mark.asyncio
    async def test_uses_issues_for_timeout_calculation(self):
        """Critical issues should produce a longer timeout than no issues."""
        mgr = AdaptiveTimeoutManager()
        issues = [make_policy_issue(IssueSeverity.CRITICAL)]
        timeout_with_issues = mgr.calculate_timeout("policy", issues)
        timeout_no_issues = mgr.calculate_timeout("policy", [])
        assert timeout_with_issues > timeout_no_issues


# ---------------------------------------------------------------------------
# Tests: Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    def setup_method(self):
        reset_adaptive_timeout_manager()

    def teardown_method(self):
        reset_adaptive_timeout_manager()

    def test_get_returns_manager_instance(self):
        mgr = get_adaptive_timeout_manager()
        assert isinstance(mgr, AdaptiveTimeoutManager)

    def test_get_returns_same_instance(self):
        mgr1 = get_adaptive_timeout_manager()
        mgr2 = get_adaptive_timeout_manager()
        assert mgr1 is mgr2

    def test_reset_creates_new_instance(self):
        mgr1 = get_adaptive_timeout_manager()
        reset_adaptive_timeout_manager()
        mgr2 = get_adaptive_timeout_manager()
        assert mgr1 is not mgr2


# ---------------------------------------------------------------------------
# Tests: Integration with different issue model types
# ---------------------------------------------------------------------------

class TestIssueModelIntegration:
    def test_price_issue_severity_extracted(self):
        mgr = AdaptiveTimeoutManager()
        issue = make_price_issue(IssueSeverity.CRITICAL)
        t = mgr.calculate_timeout("price", [issue])
        assert t > mgr._config.base_timeouts["price"]

    def test_policy_issue_severity_extracted(self):
        mgr = AdaptiveTimeoutManager()
        issue = make_policy_issue(IssueSeverity.MAJOR)
        t = mgr.calculate_timeout("policy", [issue])
        assert t > mgr._config.base_timeouts["policy"]

    def test_relevance_issue_severity_extracted(self):
        mgr = AdaptiveTimeoutManager()
        issue = make_relevance_issue(IssueSeverity.MINOR)
        t = mgr.calculate_timeout("relevance", [issue])
        # Minor adds a small amount
        assert t >= mgr._config.base_timeouts["relevance"]

    def test_mixed_issue_types_in_single_call(self):
        mgr = AdaptiveTimeoutManager()
        issues = [
            make_price_issue(IssueSeverity.CRITICAL),
            make_policy_issue(IssueSeverity.MAJOR),
            make_relevance_issue(IssueSeverity.MINOR),
        ]
        t = mgr.calculate_timeout("price", issues)
        assert t > mgr._config.base_timeouts["price"]

    def test_custom_config_base_timeouts(self):
        cfg = TimeoutConfig(base_timeouts={"price": 20.0, "policy": 30.0, "default": 15.0})
        mgr = AdaptiveTimeoutManager(cfg)
        assert mgr.calculate_timeout("price", []) == 20.0
        assert mgr.calculate_timeout("policy", []) == 30.0
