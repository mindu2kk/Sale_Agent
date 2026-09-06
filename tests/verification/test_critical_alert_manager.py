"""
Tests for CriticalAlertManager - Task 6.2.4

Covers:
- Alert triggered when VerificationResult has critical issues
- Alert includes full issue context (price deviation, fabricated policy, etc.)
- Alert deduplication within time window
- Alert history tracking
- Integration with error_rate_tracker threshold breaches
- No alert when only minor/major issues (no critical)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import List
from unittest.mock import MagicMock

import pytest

from backend.verification.models.verification import (
    IssueSeverity,
    PolicyIssue,
    PriceIssue,
    RelevanceIssue,
    RubricCriteria,
    VerificationResult,
)
from backend.verification.utils.critical_alert_manager import (
    AlertSeverity,
    CriticalAlert,
    CriticalAlertManager,
    get_critical_alert_manager,
    reset_critical_alert_manager,
)
from backend.verification.utils.error_rate_tracker import ErrorRateTracker


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_verification_result(
    price_issues: list | None = None,
    policy_issues: list | None = None,
    relevance_issues: list | None = None,
    price_pass: bool = True,
    policy_pass: bool = True,
    relevance_pass: bool = True,
) -> VerificationResult:
    criteria = RubricCriteria(
        price_accuracy_pass=price_pass,
        policy_authenticity_pass=policy_pass,
        topic_relevance_pass=relevance_pass,
        price_issues=price_issues or [],
        policy_issues=policy_issues or [],
        relevance_issues=relevance_issues or [],
    )
    return VerificationResult(
        criteria=criteria,
        verification_reasoning="Test verification reasoning for unit tests",
        execution_time_seconds=0.5,
        llm_tokens_used=100,
    )


def _critical_price_issue() -> PriceIssue:
    return PriceIssue(
        product_name="iPhone 15 Pro Max",
        product_sku="IP15PM-256",
        mentioned_price="45,000,000 VND",
        actual_price="29,990,000 VND",
        deviation_percent=50.0,
        severity=IssueSeverity.CRITICAL,
        explanation="Price deviation 50% — critical threshold exceeded",
        correction_suggestion="Update price to 29,990,000 VND",
    )


def _major_price_issue() -> PriceIssue:
    return PriceIssue(
        product_name="Samsung Galaxy S24",
        mentioned_price="22,000,000 VND",
        actual_price="19,000,000 VND",
        deviation_percent=15.8,
        severity=IssueSeverity.MAJOR,
        explanation="Price deviation 15.8% — major threshold",
    )


def _minor_price_issue() -> PriceIssue:
    return PriceIssue(
        product_name="AirPods Pro",
        mentioned_price="6,100,000 VND",
        actual_price="6,000,000 VND",
        deviation_percent=1.7,
        severity=IssueSeverity.MINOR,
        explanation="Minor price deviation 1.7%",
    )


def _critical_policy_issue(fabricated: bool = True) -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Bảo hành 5 năm cho tất cả sản phẩm",
        policy_type="warranty",
        is_fabricated=fabricated,
        is_inaccurate=not fabricated,
        severity=IssueSeverity.CRITICAL,
        explanation="Fabricated warranty policy — no such policy exists",
        source_document="warranty_policy_2024.pdf",
        correction_suggestion="Remove fabricated warranty claim",
    )


def _major_policy_issue() -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Đổi trả trong 30 ngày",
        policy_type="return",
        is_fabricated=False,
        is_inaccurate=True,
        severity=IssueSeverity.MAJOR,
        explanation="Return policy inaccurate — actual is 14 days",
        correct_policy="Đổi trả trong 14 ngày",
    )


def _critical_relevance_issue() -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="So sánh giá iPhone vs Samsung",
        response_coverage=0.1,
        missing_aspects=["price comparison", "feature comparison"],
        severity=IssueSeverity.CRITICAL,
        explanation="Response coverage only 10% — severely off-topic",
    )


def _minor_relevance_issue() -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="Hỏi về camera",
        response_coverage=0.75,
        missing_aspects=["night mode"],
        severity=IssueSeverity.MINOR,
        explanation="Minor coverage gap",
    )


@pytest.fixture
def manager() -> CriticalAlertManager:
    """Fresh CriticalAlertManager with very short dedup window for testing."""
    return CriticalAlertManager(dedup_window_seconds=1.0, error_rate_threshold=0.5)


@pytest.fixture
def manager_with_file(tmp_path) -> CriticalAlertManager:
    """CriticalAlertManager with file-based alert log."""
    log_path = str(tmp_path / "alerts.jsonl")
    return CriticalAlertManager(
        dedup_window_seconds=1.0,
        error_rate_threshold=0.5,
        alert_log_path=log_path,
    )


# ---------------------------------------------------------------------------
# 1. Alert triggered when VerificationResult has critical issues
# ---------------------------------------------------------------------------

class TestAlertTriggeredOnCriticalIssues:

    def test_critical_price_issue_fires_alert(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()],
            price_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="corr_001", workflow_id="wf_001")
        assert len(alerts) == 1
        assert alerts[0].alert_severity == AlertSeverity.CRITICAL

    def test_critical_policy_issue_fires_alert(self, manager):
        result = _make_verification_result(
            policy_issues=[_critical_policy_issue()],
            policy_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="corr_002", workflow_id="wf_002")
        assert len(alerts) == 1
        assert alerts[0].alert_severity == AlertSeverity.CRITICAL

    def test_critical_relevance_issue_fires_alert(self, manager):
        result = _make_verification_result(
            relevance_issues=[_critical_relevance_issue()],
            relevance_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="corr_003", workflow_id="wf_003")
        assert len(alerts) == 1
        assert alerts[0].alert_severity == AlertSeverity.CRITICAL

    def test_multiple_critical_issues_fire_multiple_alerts(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()],
            policy_issues=[_critical_policy_issue()],
            price_pass=False,
            policy_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="corr_004", workflow_id="wf_004")
        assert len(alerts) == 2
        assert all(a.alert_severity == AlertSeverity.CRITICAL for a in alerts)

    def test_no_alert_when_all_pass(self, manager):
        result = _make_verification_result()
        alerts = manager.check_and_alert(result, correlation_id="corr_005", workflow_id="wf_005")
        assert alerts == []


# ---------------------------------------------------------------------------
# 2. Alert includes full issue context
# ---------------------------------------------------------------------------

class TestAlertIssueContext:

    def test_price_alert_context_contains_deviation(self, manager):
        issue = _critical_price_issue()
        result = _make_verification_result(price_issues=[issue], price_pass=False)
        alerts = manager.check_and_alert(result, correlation_id="corr_010", workflow_id="wf_010")
        assert len(alerts) == 1
        ctx = alerts[0].issue_context
        assert ctx["issue_type"] == "price"
        assert ctx["product_name"] == "iPhone 15 Pro Max"
        assert ctx["deviation_percent"] == 50.0
        assert ctx["mentioned_price"] == "45,000,000 VND"
        assert ctx["actual_price"] == "29,990,000 VND"
        assert ctx["product_sku"] == "IP15PM-256"
        assert ctx["severity"] == "critical"
        assert "explanation" in ctx
        assert "correction_suggestion" in ctx

    def test_policy_alert_context_contains_fabrication_flag(self, manager):
        issue = _critical_policy_issue(fabricated=True)
        result = _make_verification_result(policy_issues=[issue], policy_pass=False)
        alerts = manager.check_and_alert(result, correlation_id="corr_011", workflow_id="wf_011")
        assert len(alerts) == 1
        ctx = alerts[0].issue_context
        assert ctx["issue_type"] == "policy"
        assert ctx["is_fabricated"] is True
        assert ctx["policy_type"] == "warranty"
        assert ctx["source_document"] == "warranty_policy_2024.pdf"
        assert ctx["severity"] == "critical"

    def test_relevance_alert_context_contains_coverage(self, manager):
        issue = _critical_relevance_issue()
        result = _make_verification_result(relevance_issues=[issue], relevance_pass=False)
        alerts = manager.check_and_alert(result, correlation_id="corr_012", workflow_id="wf_012")
        assert len(alerts) == 1
        ctx = alerts[0].issue_context
        assert ctx["issue_type"] == "relevance"
        assert ctx["response_coverage"] == 0.1
        assert "price comparison" in ctx["missing_aspects"]
        assert ctx["severity"] == "critical"

    def test_alert_contains_correlation_id(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        alerts = manager.check_and_alert(
            result, correlation_id="corr_xyz_123", workflow_id="wf_abc"
        )
        assert alerts[0].correlation_id == "corr_xyz_123"
        assert alerts[0].workflow_id == "wf_abc"

    def test_alert_message_is_descriptive(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        alerts = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert "iPhone 15 Pro Max" in alerts[0].message
        assert "50.0" in alerts[0].message


# ---------------------------------------------------------------------------
# 3. Alert deduplication within time window
# ---------------------------------------------------------------------------

class TestAlertDeduplication:

    def test_same_issue_deduplicated_within_window(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        # First call fires
        alerts1 = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert len(alerts1) == 1

        # Second call within dedup window — should be suppressed
        alerts2 = manager.check_and_alert(result, correlation_id="c2", workflow_id="w2")
        assert len(alerts2) == 0

    def test_same_issue_fires_again_after_window_expires(self, manager):
        """dedup_window_seconds=1.0 so wait 1.1s for window to expire."""
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        alerts1 = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert len(alerts1) == 1

        time.sleep(1.1)

        alerts2 = manager.check_and_alert(result, correlation_id="c2", workflow_id="w2")
        assert len(alerts2) == 1

    def test_different_issues_not_deduplicated(self, manager):
        result_price = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        result_policy = _make_verification_result(
            policy_issues=[_critical_policy_issue()], policy_pass=False
        )
        alerts1 = manager.check_and_alert(result_price, correlation_id="c1", workflow_id="w1")
        alerts2 = manager.check_and_alert(result_policy, correlation_id="c2", workflow_id="w2")
        assert len(alerts1) == 1
        assert len(alerts2) == 1

    def test_dedup_does_not_affect_history_count(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        manager.check_and_alert(result, correlation_id="c2", workflow_id="w2")  # deduplicated
        # Only 1 alert in history
        assert manager.get_alert_count() == 1


# ---------------------------------------------------------------------------
# 4. Alert history tracking
# ---------------------------------------------------------------------------

class TestAlertHistoryTracking:

    def test_history_grows_with_each_unique_alert(self, manager):
        result_price = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        result_policy = _make_verification_result(
            policy_issues=[_critical_policy_issue()], policy_pass=False
        )
        manager.check_and_alert(result_price, correlation_id="c1", workflow_id="w1")
        manager.check_and_alert(result_policy, correlation_id="c2", workflow_id="w2")
        assert manager.get_alert_count() == 2

    def test_get_alert_history_returns_most_recent_first(self, manager):
        result_price = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        result_policy = _make_verification_result(
            policy_issues=[_critical_policy_issue()], policy_pass=False
        )
        manager.check_and_alert(result_price, correlation_id="c1", workflow_id="w1")
        time.sleep(0.01)
        manager.check_and_alert(result_policy, correlation_id="c2", workflow_id="w2")

        history = manager.get_alert_history()
        assert len(history) == 2
        # Most recent first
        assert history[0].timestamp >= history[1].timestamp

    def test_history_limit_parameter(self, manager):
        result_price = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        result_policy = _make_verification_result(
            policy_issues=[_critical_policy_issue()], policy_pass=False
        )
        manager.check_and_alert(result_price, correlation_id="c1", workflow_id="w1")
        manager.check_and_alert(result_policy, correlation_id="c2", workflow_id="w2")

        history = manager.get_alert_history(limit=1)
        assert len(history) == 1

    def test_history_severity_filter(self, manager):
        result_price = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        manager.check_and_alert(result_price, correlation_id="c1", workflow_id="w1")

        critical_history = manager.get_alert_history(severity_filter=AlertSeverity.CRITICAL)
        high_history = manager.get_alert_history(severity_filter=AlertSeverity.HIGH)
        assert len(critical_history) == 1
        assert len(high_history) == 0

    def test_clear_history(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert manager.get_alert_count() == 1

        manager.clear_history()
        assert manager.get_alert_count() == 0

    def test_alert_has_timestamp(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        before = time.time()
        alerts = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        after = time.time()
        assert before <= alerts[0].timestamp <= after


# ---------------------------------------------------------------------------
# 5. Integration with error_rate_tracker threshold breaches
# ---------------------------------------------------------------------------

class TestErrorRateThresholdIntegration:

    def test_alert_fired_when_error_rate_exceeds_threshold(self, manager):
        tracker = ErrorRateTracker(window_seconds=60)
        mgr = CriticalAlertManager(
            dedup_window_seconds=1.0,
            error_rate_threshold=0.5,
            error_rate_tracker=tracker,
        )
        # Record 6 errors and 4 successes → 60% error rate
        for _ in range(6):
            tracker.record_error("verification", "TimeoutError")
        for _ in range(4):
            tracker.record_success("verification")

        alert = mgr.check_error_rate_threshold(
            "verification", correlation_id="c1", workflow_id="w1"
        )
        assert alert is not None
        assert alert.alert_severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)
        assert alert.issue_context["component"] == "verification"
        assert alert.issue_context["error_rate"] >= 0.5

    def test_no_alert_when_error_rate_below_threshold(self, manager):
        tracker = ErrorRateTracker(window_seconds=60)
        mgr = CriticalAlertManager(
            dedup_window_seconds=1.0,
            error_rate_threshold=0.5,
            error_rate_tracker=tracker,
        )
        # 2 errors, 8 successes → 20% error rate
        for _ in range(2):
            tracker.record_error("verification", "TimeoutError")
        for _ in range(8):
            tracker.record_success("verification")

        alert = mgr.check_error_rate_threshold(
            "verification", correlation_id="c1", workflow_id="w1"
        )
        assert alert is None

    def test_critical_severity_when_error_rate_very_high(self, manager):
        tracker = ErrorRateTracker(window_seconds=60)
        mgr = CriticalAlertManager(
            dedup_window_seconds=1.0,
            error_rate_threshold=0.5,
            error_rate_tracker=tracker,
        )
        # 9 errors, 1 success → 90% error rate
        for _ in range(9):
            tracker.record_error("research", "ConnectionError")
        tracker.record_success("research")

        alert = mgr.check_error_rate_threshold(
            "research", correlation_id="c1", workflow_id="w1"
        )
        assert alert is not None
        assert alert.alert_severity == AlertSeverity.CRITICAL

    def test_error_rate_alert_context_contains_stats(self, manager):
        tracker = ErrorRateTracker(window_seconds=60)
        mgr = CriticalAlertManager(
            dedup_window_seconds=1.0,
            error_rate_threshold=0.5,
            error_rate_tracker=tracker,
        )
        for _ in range(7):
            tracker.record_error("correction", "ValidationError")
        for _ in range(3):
            tracker.record_success("correction")

        alert = mgr.check_error_rate_threshold(
            "correction", correlation_id="c1", workflow_id="w1"
        )
        assert alert is not None
        ctx = alert.issue_context
        assert "component_stats" in ctx
        assert ctx["threshold"] == 0.5

    def test_pre_computed_error_rate_used_when_provided(self, manager):
        """check_error_rate_threshold accepts a pre-computed rate."""
        alert = manager.check_error_rate_threshold(
            "verification",
            error_rate=0.75,
            correlation_id="c1",
            workflow_id="w1",
        )
        assert alert is not None
        assert alert.issue_context["error_rate"] == 0.75


# ---------------------------------------------------------------------------
# 6. No alert when only minor/major issues (no critical)
# ---------------------------------------------------------------------------

class TestNoAlertForNonCriticalIssues:

    def test_no_alert_for_minor_price_issue(self, manager):
        result = _make_verification_result(
            price_issues=[_minor_price_issue()],
            price_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert alerts == []

    def test_no_alert_for_major_price_issue(self, manager):
        result = _make_verification_result(
            price_issues=[_major_price_issue()],
            price_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert alerts == []

    def test_no_alert_for_major_policy_issue(self, manager):
        result = _make_verification_result(
            policy_issues=[_major_policy_issue()],
            policy_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert alerts == []

    def test_no_alert_for_minor_relevance_issue(self, manager):
        result = _make_verification_result(
            relevance_issues=[_minor_relevance_issue()],
            relevance_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert alerts == []

    def test_mixed_critical_and_minor_only_critical_fires(self, manager):
        result = _make_verification_result(
            price_issues=[_critical_price_issue(), _minor_price_issue()],
            price_pass=False,
        )
        alerts = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        # Only the critical issue fires an alert
        assert len(alerts) == 1
        assert alerts[0].alert_severity == AlertSeverity.CRITICAL


# ---------------------------------------------------------------------------
# 7. Callback handler tests
# ---------------------------------------------------------------------------

class TestCallbackHandlers:

    def test_callback_invoked_on_alert(self, manager):
        received: List[CriticalAlert] = []
        manager.register_callback(received.append)

        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert len(received) == 1
        assert received[0].alert_severity == AlertSeverity.CRITICAL

    def test_callback_not_invoked_for_deduplicated_alert(self, manager):
        received: List[CriticalAlert] = []
        manager.register_callback(received.append)

        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        manager.check_and_alert(result, correlation_id="c2", workflow_id="w2")  # deduped
        assert len(received) == 1

    def test_failing_callback_does_not_raise(self, manager):
        def bad_callback(alert):
            raise RuntimeError("callback failure")

        manager.register_callback(bad_callback)
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        # Should not raise
        alerts = manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert len(alerts) == 1

    def test_multiple_callbacks_all_invoked(self, manager):
        received_a: List[CriticalAlert] = []
        received_b: List[CriticalAlert] = []
        manager.register_callback(received_a.append)
        manager.register_callback(received_b.append)

        result = _make_verification_result(
            policy_issues=[_critical_policy_issue()], policy_pass=False
        )
        manager.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        assert len(received_a) == 1
        assert len(received_b) == 1


# ---------------------------------------------------------------------------
# 8. File-based alert log tests
# ---------------------------------------------------------------------------

class TestFileAlertLog:

    def test_alert_written_to_file(self, manager_with_file):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        manager_with_file.check_and_alert(result, correlation_id="c1", workflow_id="w1")

        log_path = manager_with_file.alert_log_path
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["alert_severity"] == "CRITICAL"
        assert "issue_context" in data
        assert data["correlation_id"] == "c1"

    def test_multiple_alerts_appended_to_file(self, manager_with_file):
        result_price = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        result_policy = _make_verification_result(
            policy_issues=[_critical_policy_issue()], policy_pass=False
        )
        manager_with_file.check_and_alert(result_price, correlation_id="c1", workflow_id="w1")
        manager_with_file.check_and_alert(result_policy, correlation_id="c2", workflow_id="w2")

        lines = manager_with_file.alert_log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_deduplicated_alert_not_written_to_file(self, manager_with_file):
        result = _make_verification_result(
            price_issues=[_critical_price_issue()], price_pass=False
        )
        manager_with_file.check_and_alert(result, correlation_id="c1", workflow_id="w1")
        manager_with_file.check_and_alert(result, correlation_id="c2", workflow_id="w2")  # deduped

        lines = manager_with_file.alert_log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# 9. Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:

    def setup_method(self):
        reset_critical_alert_manager()

    def teardown_method(self):
        reset_critical_alert_manager()

    def test_get_critical_alert_manager_returns_singleton(self):
        m1 = get_critical_alert_manager()
        m2 = get_critical_alert_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_critical_alert_manager()
        reset_critical_alert_manager()
        m2 = get_critical_alert_manager()
        assert m1 is not m2
