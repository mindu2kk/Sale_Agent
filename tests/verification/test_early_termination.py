"""
Unit Tests: Early Termination Manager

Tests for configurable early termination rules during parallel verification.

Supports Task 1.3.2: Implement configurable early termination rules cho critical issues
"""

import pytest
from verification.config.thresholds_config import (
    EscalationThresholds,
    IssueSeverity,
    VerificationThresholdsConfig,
    get_default_thresholds_config,
)
from verification.utils.early_termination import (
    EarlyTerminationManager,
    TerminationResult,
    create_early_termination_manager,
)


# ---------------------------------------------------------------------------
# Minimal stub issue objects for testing
# ---------------------------------------------------------------------------

class _Issue:
    """Minimal stub with a .severity attribute."""
    def __init__(self, severity: IssueSeverity):
        self.severity = severity


def _critical() -> _Issue:
    return _Issue(IssueSeverity.CRITICAL)


def _major() -> _Issue:
    return _Issue(IssueSeverity.MAJOR)


def _minor() -> _Issue:
    return _Issue(IssueSeverity.MINOR)


# ---------------------------------------------------------------------------
# Tests: EarlyTerminationManager basics
# ---------------------------------------------------------------------------

class TestEarlyTerminationManagerDefaults:
    """Test default configuration behaviour."""

    def test_enabled_by_default(self):
        mgr = EarlyTerminationManager()
        assert mgr.enabled is True

    def test_stop_on_first_critical_false_by_default(self):
        mgr = EarlyTerminationManager()
        assert mgr.stop_on_first_critical is False

    def test_multiple_critical_threshold_default(self):
        mgr = EarlyTerminationManager()
        assert mgr.multiple_critical_threshold == 3

    def test_no_termination_with_zero_issues(self):
        mgr = EarlyTerminationManager()
        result = mgr.should_terminate([])
        assert result.should_terminate is False
        assert result.critical_count == 0

    def test_no_termination_with_only_minor_issues(self):
        mgr = EarlyTerminationManager()
        result = mgr.should_terminate([_minor(), _minor(), _minor()])
        assert result.should_terminate is False

    def test_no_termination_below_threshold(self):
        mgr = EarlyTerminationManager()
        # 2 critical < threshold of 3
        result = mgr.should_terminate([_critical(), _critical()])
        assert result.should_terminate is False

    def test_termination_at_threshold(self):
        mgr = EarlyTerminationManager()
        # 3 critical == threshold of 3
        result = mgr.should_terminate([_critical(), _critical(), _critical()])
        assert result.should_terminate is True

    def test_termination_above_threshold(self):
        mgr = EarlyTerminationManager()
        issues = [_critical()] * 5
        result = mgr.should_terminate(issues)
        assert result.should_terminate is True
        assert result.critical_count == 5

    def test_result_bool_conversion(self):
        mgr = EarlyTerminationManager()
        result = mgr.should_terminate([_critical()] * 3)
        assert bool(result) is True

        result_no = mgr.should_terminate([_critical()])
        assert bool(result_no) is False


class TestStopOnFirstCritical:
    """Test stop_on_first_critical mode."""

    def _make_mgr(self) -> EarlyTerminationManager:
        config = get_default_thresholds_config()
        config.escalation.stop_on_first_critical = True
        return EarlyTerminationManager(config)

    def test_terminates_on_single_critical(self):
        mgr = self._make_mgr()
        result = mgr.should_terminate([_critical()])
        assert result.should_terminate is True

    def test_no_termination_without_critical(self):
        mgr = self._make_mgr()
        result = mgr.should_terminate([_major(), _major()])
        assert result.should_terminate is False

    def test_reason_mentions_first_critical(self):
        mgr = self._make_mgr()
        result = mgr.should_terminate([_critical()])
        assert "first critical" in result.reason.lower()


class TestEarlyTerminationDisabled:
    """Test behaviour when early termination is disabled (dev/test env)."""

    def _make_mgr(self) -> EarlyTerminationManager:
        config = get_default_thresholds_config()
        config.escalation.early_termination_enabled = False
        return EarlyTerminationManager(config)

    def test_never_terminates_even_with_many_criticals(self):
        mgr = self._make_mgr()
        result = mgr.should_terminate([_critical()] * 10)
        assert result.should_terminate is False

    def test_enabled_property_is_false(self):
        mgr = self._make_mgr()
        assert mgr.enabled is False


class TestIssueSeverityDirectInput:
    """Test that bare IssueSeverity values are accepted."""

    def test_severity_enum_values_counted(self):
        mgr = EarlyTerminationManager()
        issues = [IssueSeverity.CRITICAL, IssueSeverity.CRITICAL, IssueSeverity.CRITICAL]
        result = mgr.should_terminate(issues)
        assert result.should_terminate is True
        assert result.critical_count == 3

    def test_mixed_severity_enum_values(self):
        mgr = EarlyTerminationManager()
        issues = [IssueSeverity.CRITICAL, IssueSeverity.MAJOR, IssueSeverity.MINOR]
        result = mgr.should_terminate(issues)
        assert result.critical_count == 1
        assert result.total_count == 3


class TestShouldTerminateForCount:
    """Test the count-based convenience method."""

    def test_terminates_at_threshold(self):
        mgr = EarlyTerminationManager()
        result = mgr.should_terminate_for_count(3)
        assert result.should_terminate is True

    def test_no_termination_below_threshold(self):
        mgr = EarlyTerminationManager()
        result = mgr.should_terminate_for_count(2)
        assert result.should_terminate is False


# ---------------------------------------------------------------------------
# Tests: Environment-specific factory
# ---------------------------------------------------------------------------

class TestCreateEarlyTerminationManagerFactory:
    """Test the factory helper with environment overrides."""

    def test_development_disables_early_termination(self):
        mgr = create_early_termination_manager(environment="development")
        assert mgr.enabled is False
        # Even with many criticals, should not terminate
        result = mgr.should_terminate([_critical()] * 10)
        assert result.should_terminate is False

    def test_production_uses_strict_settings(self):
        mgr = create_early_termination_manager(environment="production")
        # Production: max_critical_before_escalation=1, but early_termination_enabled
        # stays True (not overridden in production env overrides)
        assert mgr.enabled is True

    def test_testing_disables_early_termination(self):
        mgr = create_early_termination_manager(environment="testing")
        assert mgr.enabled is False

    def test_no_environment_uses_defaults(self):
        mgr = create_early_termination_manager()
        assert mgr.enabled is True
        assert mgr.multiple_critical_threshold == 3

    def test_custom_config_respected(self):
        config = get_default_thresholds_config()
        config.escalation.multiple_critical_threshold = 2
        mgr = create_early_termination_manager(config=config)
        assert mgr.multiple_critical_threshold == 2
        result = mgr.should_terminate([_critical(), _critical()])
        assert result.should_terminate is True


# ---------------------------------------------------------------------------
# Tests: TerminationResult
# ---------------------------------------------------------------------------

class TestTerminationResult:
    """Test TerminationResult dataclass."""

    def test_fields_populated(self):
        r = TerminationResult(
            should_terminate=True,
            reason="test reason",
            critical_count=3,
            total_count=5,
        )
        assert r.should_terminate is True
        assert r.reason == "test reason"
        assert r.critical_count == 3
        assert r.total_count == 5

    def test_bool_false(self):
        r = TerminationResult(
            should_terminate=False,
            reason="no termination",
            critical_count=0,
            total_count=2,
        )
        assert bool(r) is False


# ---------------------------------------------------------------------------
# Task 5.4.1: CriticalIssueDetector tests
# ---------------------------------------------------------------------------

from verification.utils.early_termination import (
    CriticalIssueDetector,
    TerminationDecision,
    should_terminate_immediately,
)
from verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity as ModelIssueSeverity,
)


def _make_price_issue(severity: ModelIssueSeverity) -> PriceIssue:
    return PriceIssue(
        product_name="Test Product",
        severity=severity,
        explanation="Test price issue",
    )


def _make_policy_issue(severity: ModelIssueSeverity) -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Test policy",
        policy_type="warranty",
        is_fabricated=(severity == ModelIssueSeverity.CRITICAL),
        is_inaccurate=False,
        severity=severity,
        explanation="Test policy issue",
    )


def _make_relevance_issue(severity: ModelIssueSeverity) -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="Test intent",
        response_coverage=0.1 if severity == ModelIssueSeverity.CRITICAL else 0.6,
        severity=severity,
        explanation="Test relevance issue",
    )


def _make_verification_result(
    price_issues=None,
    policy_issues=None,
    relevance_issues=None,
) -> VerificationResult:
    price_issues = price_issues or []
    policy_issues = policy_issues or []
    relevance_issues = relevance_issues or []

    price_pass = len(price_issues) == 0
    policy_pass = len(policy_issues) == 0
    relevance_pass = len(relevance_issues) == 0

    criteria = RubricCriteria(
        price_accuracy_pass=price_pass,
        price_issues=price_issues,
        policy_authenticity_pass=policy_pass,
        policy_issues=policy_issues,
        topic_relevance_pass=relevance_pass,
        relevance_issues=relevance_issues,
    )
    return VerificationResult(
        criteria=criteria,
        verification_reasoning="Test verification result",
        execution_time_seconds=0.1,
        llm_tokens_used=100,
    )


class TestCriticalIssueDetectorCheckIssues:
    """Test CriticalIssueDetector.check_issues() with raw issue lists."""

    def test_single_critical_price_issue_triggers_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            price_issues=[_make_price_issue(ModelIssueSeverity.CRITICAL)]
        )
        assert decision.should_terminate is True
        assert decision.critical_price_count == 1
        assert decision.critical_policy_count == 0
        assert decision.critical_relevance_count == 0

    def test_single_critical_policy_issue_triggers_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            policy_issues=[_make_policy_issue(ModelIssueSeverity.CRITICAL)]
        )
        assert decision.should_terminate is True
        assert decision.critical_policy_count == 1

    def test_single_critical_relevance_issue_triggers_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            relevance_issues=[_make_relevance_issue(ModelIssueSeverity.CRITICAL)]
        )
        assert decision.should_terminate is True
        assert decision.critical_relevance_count == 1

    def test_multiple_minor_issues_do_not_trigger_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            price_issues=[_make_price_issue(ModelIssueSeverity.MINOR)] * 3,
            policy_issues=[_make_policy_issue(ModelIssueSeverity.MINOR)] * 2,
            relevance_issues=[_make_relevance_issue(ModelIssueSeverity.MINOR)] * 2,
        )
        assert decision.should_terminate is False
        assert decision.total_critical_count == 0

    def test_multiple_major_issues_do_not_trigger_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            price_issues=[_make_price_issue(ModelIssueSeverity.MAJOR)] * 5,
            policy_issues=[_make_policy_issue(ModelIssueSeverity.MAJOR)] * 3,
        )
        assert decision.should_terminate is False

    def test_empty_issues_no_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues()
        assert decision.should_terminate is False
        assert decision.total_critical_count == 0

    def test_mixed_severities_with_one_critical_triggers_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            price_issues=[
                _make_price_issue(ModelIssueSeverity.MINOR),
                _make_price_issue(ModelIssueSeverity.MAJOR),
                _make_price_issue(ModelIssueSeverity.CRITICAL),  # this one triggers
            ]
        )
        assert decision.should_terminate is True
        assert decision.critical_price_count == 1

    def test_termination_decision_bool_conversion(self):
        detector = CriticalIssueDetector()
        yes = detector.check_issues(price_issues=[_make_price_issue(ModelIssueSeverity.CRITICAL)])
        no = detector.check_issues(price_issues=[_make_price_issue(ModelIssueSeverity.MINOR)])
        assert bool(yes) is True
        assert bool(no) is False

    def test_reason_populated_on_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            price_issues=[_make_price_issue(ModelIssueSeverity.CRITICAL)]
        )
        assert "critical" in decision.reason.lower()
        assert len(decision.reason) > 10

    def test_reason_populated_on_no_termination(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            price_issues=[_make_price_issue(ModelIssueSeverity.MINOR)]
        )
        assert len(decision.reason) > 0

    def test_critical_issues_list_populated(self):
        detector = CriticalIssueDetector()
        critical_issue = _make_price_issue(ModelIssueSeverity.CRITICAL)
        decision = detector.check_issues(price_issues=[critical_issue])
        assert len(decision.critical_issues) == 1
        assert decision.critical_issues[0] is critical_issue

    def test_total_critical_count_property(self):
        detector = CriticalIssueDetector()
        decision = detector.check_issues(
            price_issues=[_make_price_issue(ModelIssueSeverity.CRITICAL)],
            policy_issues=[_make_policy_issue(ModelIssueSeverity.CRITICAL)],
            relevance_issues=[_make_relevance_issue(ModelIssueSeverity.CRITICAL)],
        )
        assert decision.total_critical_count == 3


class TestCriticalIssueDetectorCheckVerificationResult:
    """Test CriticalIssueDetector.check_verification_result()."""

    def test_critical_price_issue_in_result_triggers_termination(self):
        detector = CriticalIssueDetector()
        result = _make_verification_result(
            price_issues=[_make_price_issue(ModelIssueSeverity.CRITICAL)]
        )
        decision = detector.check_verification_result(result)
        assert decision.should_terminate is True
        assert decision.critical_price_count == 1

    def test_critical_policy_issue_in_result_triggers_termination(self):
        detector = CriticalIssueDetector()
        result = _make_verification_result(
            policy_issues=[_make_policy_issue(ModelIssueSeverity.CRITICAL)]
        )
        decision = detector.check_verification_result(result)
        assert decision.should_terminate is True
        assert decision.critical_policy_count == 1

    def test_critical_relevance_issue_in_result_triggers_termination(self):
        detector = CriticalIssueDetector()
        result = _make_verification_result(
            relevance_issues=[_make_relevance_issue(ModelIssueSeverity.CRITICAL)]
        )
        decision = detector.check_verification_result(result)
        assert decision.should_terminate is True
        assert decision.critical_relevance_count == 1

    def test_only_minor_issues_no_termination(self):
        detector = CriticalIssueDetector()
        result = _make_verification_result(
            price_issues=[_make_price_issue(ModelIssueSeverity.MINOR)],
            policy_issues=[_make_policy_issue(ModelIssueSeverity.MINOR)],
            relevance_issues=[_make_relevance_issue(ModelIssueSeverity.MINOR)],
        )
        decision = detector.check_verification_result(result)
        assert decision.should_terminate is False

    def test_approved_result_no_termination(self):
        detector = CriticalIssueDetector()
        result = _make_verification_result()  # no issues → approved
        decision = detector.check_verification_result(result)
        assert decision.should_terminate is False


class TestShouldTerminateImmediatelyFunction:
    """Test the module-level should_terminate_immediately() convenience function."""

    def test_critical_price_issue_triggers(self):
        result = _make_verification_result(
            price_issues=[_make_price_issue(ModelIssueSeverity.CRITICAL)]
        )
        decision = should_terminate_immediately(result)
        assert decision.should_terminate is True

    def test_critical_policy_issue_triggers(self):
        result = _make_verification_result(
            policy_issues=[_make_policy_issue(ModelIssueSeverity.CRITICAL)]
        )
        decision = should_terminate_immediately(result)
        assert decision.should_terminate is True

    def test_critical_relevance_issue_triggers(self):
        result = _make_verification_result(
            relevance_issues=[_make_relevance_issue(ModelIssueSeverity.CRITICAL)]
        )
        decision = should_terminate_immediately(result)
        assert decision.should_terminate is True

    def test_multiple_minor_issues_no_termination(self):
        result = _make_verification_result(
            price_issues=[_make_price_issue(ModelIssueSeverity.MINOR)] * 5,
            policy_issues=[_make_policy_issue(ModelIssueSeverity.MINOR)] * 3,
            relevance_issues=[_make_relevance_issue(ModelIssueSeverity.MINOR)] * 2,
        )
        decision = should_terminate_immediately(result)
        assert decision.should_terminate is False

    def test_no_issues_no_termination(self):
        result = _make_verification_result()
        decision = should_terminate_immediately(result)
        assert decision.should_terminate is False


class TestVerificationResultImmediateTerminationFlag:
    """Test that VerificationResult has has_critical_issues and immediate_termination fields."""

    def test_default_flags_are_false(self):
        result = _make_verification_result()
        assert result.has_critical_issues is False
        assert result.immediate_termination is False

    def test_flags_can_be_set_to_true(self):
        result = _make_verification_result()
        result.has_critical_issues = True
        result.immediate_termination = True
        assert result.has_critical_issues is True
        assert result.immediate_termination is True


class TestRoutingWithCriticalIssueDetector:
    """Test that routing correctly escalates when immediate_termination=True."""

    def _make_state(self, verification_result, retry_count=0, max_retries=3):
        return {
            "verification_result": verification_result,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "workflow_status": "verifying",
        }

    def _make_router(self):
        from verification.workflow.routing import WorkflowRouter
        from verification.config.config import VerificationConfig
        config = VerificationConfig()
        return WorkflowRouter(config)

    def test_routes_to_escalation_when_immediate_termination_true(self):
        router = self._make_router()
        result = _make_verification_result(
            price_issues=[_make_price_issue(ModelIssueSeverity.CRITICAL)]
        )
        result.immediate_termination = True
        state = self._make_state(result)
        decision = router.route_after_verification(state)
        assert decision == "escalation"

    def test_routes_to_escalation_for_critical_price_issue(self):
        router = self._make_router()
        result = _make_verification_result(
            price_issues=[_make_price_issue(ModelIssueSeverity.CRITICAL)]
        )
        state = self._make_state(result)
        decision = router.route_after_verification(state)
        assert decision == "escalation"

    def test_routes_to_escalation_for_critical_policy_issue(self):
        router = self._make_router()
        result = _make_verification_result(
            policy_issues=[_make_policy_issue(ModelIssueSeverity.CRITICAL)]
        )
        state = self._make_state(result)
        decision = router.route_after_verification(state)
        assert decision == "escalation"

    def test_routes_to_escalation_for_critical_relevance_issue(self):
        router = self._make_router()
        result = _make_verification_result(
            relevance_issues=[_make_relevance_issue(ModelIssueSeverity.CRITICAL)]
        )
        state = self._make_state(result)
        decision = router.route_after_verification(state)
        assert decision == "escalation"

    def test_routes_to_correction_for_minor_issues(self):
        router = self._make_router()
        result = _make_verification_result(
            price_issues=[_make_price_issue(ModelIssueSeverity.MINOR)]
        )
        state = self._make_state(result)
        decision = router.route_after_verification(state)
        assert decision == "correction"

    def test_routes_to_approved_when_no_issues(self):
        router = self._make_router()
        result = _make_verification_result()  # all pass
        state = self._make_state(result)
        decision = router.route_after_verification(state)
        assert decision == "approved"
