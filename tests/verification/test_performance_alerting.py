"""
Tests for Task 5.3.5: Performance alerting thresholds with early termination benefits.

Covers:
- PerformanceAlertThresholds default values
- PerformanceAlert.to_dict() structure
- PerformanceAlertingSystem.check_latency() fires warning/critical alerts
- PerformanceAlertingSystem.check_throughput() fires on low WPS and high fail rate
- PerformanceAlertingSystem.check_resources() fires on memory/CPU thresholds
- PerformanceAlertingSystem.check_tokens() fires on token count and cost thresholds
- should_recommend_early_termination() returns True only on critical alerts
- get_early_termination_reason() aggregates messages from critical alerts
- get_alert_summary() returns correct counts and structure
- alert_callback is invoked for every alert
- reset() clears all alerts
- create_performance_alerting_system() returns environment-specific thresholds
"""

import time
import pytest
from unittest.mock import MagicMock

from verification.utils.performance import AsyncStepLatencyTracker, StepLatencyRecord, ThroughputMonitor
from verification.utils.resource_monitor import ResourceUsageReport
from verification.utils.token_tracker import LLMTokenTracker
from verification.utils.performance_alerting import (
    AlertSeverity,
    AlertType,
    PerformanceAlert,
    PerformanceAlertThresholds,
    PerformanceAlertingSystem,
    create_performance_alerting_system,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system(**threshold_overrides) -> PerformanceAlertingSystem:
    t = PerformanceAlertThresholds(**threshold_overrides)
    return PerformanceAlertingSystem(thresholds=t)


def _make_tracker_with_step(step_name: str, duration_ms: float) -> AsyncStepLatencyTracker:
    tracker = AsyncStepLatencyTracker()
    tracker._records[step_name] = StepLatencyRecord(
        step_name=step_name,
        start_time=time.time(),
        end_time=time.time() + duration_ms / 1000,
        duration_ms=duration_ms,
    )
    return tracker


def _make_resource_report(peak_memory_mb: float = 0.0, peak_cpu: float = 0.0) -> ResourceUsageReport:
    return ResourceUsageReport(
        duration_seconds=1.0,
        peak_memory_rss_mb=peak_memory_mb,
        avg_memory_rss_mb=peak_memory_mb * 0.8,
        peak_cpu_percent=peak_cpu,
        avg_cpu_percent=peak_cpu * 0.8,
        peak_async_tasks=1,
        avg_async_tasks=1.0,
        peak_thread_count=2,
        sample_count=5,
    )


def _make_token_tracker(input_tokens: int = 0, output_tokens: int = 0, model: str = "gpt-4o-mini") -> LLMTokenTracker:
    tracker = LLMTokenTracker()
    if input_tokens or output_tokens:
        tracker.record(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
    return tracker


# ---------------------------------------------------------------------------
# PerformanceAlert.to_dict()
# ---------------------------------------------------------------------------

class TestPerformanceAlertToDict:
    def test_to_dict_has_all_required_fields(self):
        alert = PerformanceAlert(
            alert_type=AlertType.LATENCY,
            severity=AlertSeverity.WARNING,
            threshold=3000.0,
            actual_value=4000.0,
            message="Latency exceeded",
        )
        d = alert.to_dict()
        for key in ("alert_type", "severity", "threshold", "actual_value",
                    "message", "recommend_early_termination", "timestamp", "metadata"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_alert_type_is_string(self):
        alert = PerformanceAlert(
            alert_type=AlertType.TOKEN_COST,
            severity=AlertSeverity.CRITICAL,
            threshold=0.10,
            actual_value=0.25,
            message="Cost exceeded",
        )
        assert isinstance(alert.to_dict()["alert_type"], str)

    def test_to_dict_recommend_early_termination_default_false(self):
        alert = PerformanceAlert(
            alert_type=AlertType.RESOURCE,
            severity=AlertSeverity.WARNING,
            threshold=512.0,
            actual_value=600.0,
            message="Memory high",
        )
        assert alert.to_dict()["recommend_early_termination"] is False

    def test_to_dict_recommend_early_termination_true_when_set(self):
        alert = PerformanceAlert(
            alert_type=AlertType.LATENCY,
            severity=AlertSeverity.CRITICAL,
            threshold=8000.0,
            actual_value=10000.0,
            message="Critical latency",
            recommend_early_termination=True,
        )
        assert alert.to_dict()["recommend_early_termination"] is True


# ---------------------------------------------------------------------------
# check_latency()
# ---------------------------------------------------------------------------

class TestCheckLatency:
    def test_no_alert_below_warning_threshold(self):
        system = _make_system(warning_latency_ms=3000.0)
        tracker = _make_tracker_with_step("price_check", duration_ms=100.0)
        alerts = system.check_latency(tracker)
        assert alerts == []

    def test_warning_alert_at_warning_threshold(self):
        system = _make_system(warning_latency_ms=3000.0, critical_latency_ms=8000.0)
        tracker = _make_tracker_with_step("price_check", duration_ms=3500.0)
        alerts = system.check_latency(tracker)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].alert_type == AlertType.LATENCY

    def test_critical_alert_at_critical_threshold(self):
        system = _make_system(warning_latency_ms=3000.0, critical_latency_ms=8000.0)
        tracker = _make_tracker_with_step("price_check", duration_ms=9000.0)
        alerts = system.check_latency(tracker)
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_critical_latency_recommends_early_termination(self):
        system = _make_system(
            critical_latency_ms=8000.0,
            enable_early_termination_recommendations=True,
        )
        tracker = _make_tracker_with_step("price_check", duration_ms=9000.0)
        system.check_latency(tracker)
        assert system.should_recommend_early_termination()

    def test_warning_latency_does_not_recommend_early_termination(self):
        system = _make_system(
            warning_latency_ms=3000.0,
            critical_latency_ms=8000.0,
            enable_early_termination_recommendations=True,
        )
        tracker = _make_tracker_with_step("price_check", duration_ms=4000.0)
        system.check_latency(tracker)
        assert not system.should_recommend_early_termination()

    def test_aggregate_total_latency_warning(self):
        system = _make_system(
            warning_latency_ms=100.0,
            critical_latency_ms=999_999.0,
            total_latency_warning_ms=500.0,
            total_latency_critical_ms=999_999.0,
        )
        tracker = AsyncStepLatencyTracker()
        tracker._records["price_check"] = StepLatencyRecord(
            step_name="price_check", start_time=0.0, end_time=0.3, duration_ms=300.0
        )
        tracker._records["policy_check"] = StepLatencyRecord(
            step_name="policy_check", start_time=0.0, end_time=0.3, duration_ms=300.0
        )
        alerts = system.check_latency(tracker)
        agg_alerts = [a for a in alerts if a.metadata.get("aggregate")]
        assert len(agg_alerts) == 1
        assert agg_alerts[0].severity == AlertSeverity.WARNING

    def test_aggregate_total_latency_critical(self):
        system = _make_system(
            warning_latency_ms=100.0,
            critical_latency_ms=999_999.0,
            total_latency_warning_ms=500.0,
            total_latency_critical_ms=1000.0,
        )
        tracker = AsyncStepLatencyTracker()
        tracker._records["price_check"] = StepLatencyRecord(
            step_name="price_check", start_time=0.0, end_time=0.6, duration_ms=600.0
        )
        tracker._records["policy_check"] = StepLatencyRecord(
            step_name="policy_check", start_time=0.0, end_time=0.6, duration_ms=600.0
        )
        alerts = system.check_latency(tracker)
        agg_alerts = [a for a in alerts if a.metadata.get("aggregate")]
        assert any(a.severity == AlertSeverity.CRITICAL for a in agg_alerts)

    def test_step_name_in_alert_metadata(self):
        system = _make_system(warning_latency_ms=100.0)
        tracker = _make_tracker_with_step("policy_check", duration_ms=200.0)
        alerts = system.check_latency(tracker)
        step_alerts = [a for a in alerts if not a.metadata.get("aggregate")]
        assert step_alerts[0].metadata["step_name"] == "policy_check"

    def test_multiple_steps_multiple_alerts(self):
        system = _make_system(warning_latency_ms=100.0, critical_latency_ms=999_999.0,
                               total_latency_warning_ms=999_999.0, total_latency_critical_ms=999_999.0)
        tracker = AsyncStepLatencyTracker()
        for name in ("price_check", "policy_check", "relevance_check"):
            tracker._records[name] = StepLatencyRecord(
                step_name=name, start_time=0.0, end_time=0.2, duration_ms=200.0
            )
        alerts = system.check_latency(tracker)
        step_alerts = [a for a in alerts if not a.metadata.get("aggregate")]
        assert len(step_alerts) == 3


# ---------------------------------------------------------------------------
# check_throughput()
# ---------------------------------------------------------------------------

class TestCheckThroughput:
    def _make_monitor_with_completions(self, passes: int, fails: int) -> ThroughputMonitor:
        m = ThroughputMonitor()
        for _ in range(passes):
            m.workflow_completed(passed=True)
        for _ in range(fails):
            m.workflow_completed(passed=False)
        return m

    def test_no_alert_when_no_completions(self):
        system = _make_system()
        m = ThroughputMonitor()
        alerts = system.check_throughput(m)
        assert alerts == []

    def test_no_alert_on_healthy_throughput(self):
        system = _make_system(max_fail_rate_warning=0.5, max_fail_rate_critical=0.8)
        m = self._make_monitor_with_completions(passes=9, fails=1)
        alerts = system.check_throughput(m, window_seconds=60.0)
        fail_alerts = [a for a in alerts if a.alert_type == AlertType.THROUGHPUT
                       and "fail rate" in a.message]
        assert fail_alerts == []

    def test_warning_on_high_fail_rate(self):
        system = _make_system(max_fail_rate_warning=0.4, max_fail_rate_critical=0.8)
        m = self._make_monitor_with_completions(passes=5, fails=5)
        alerts = system.check_throughput(m, window_seconds=60.0)
        fail_alerts = [a for a in alerts if "fail rate" in a.message]
        assert any(a.severity == AlertSeverity.WARNING for a in fail_alerts)

    def test_critical_on_very_high_fail_rate(self):
        system = _make_system(max_fail_rate_warning=0.4, max_fail_rate_critical=0.7)
        m = self._make_monitor_with_completions(passes=2, fails=8)
        alerts = system.check_throughput(m, window_seconds=60.0)
        fail_alerts = [a for a in alerts if "fail rate" in a.message]
        assert any(a.severity == AlertSeverity.CRITICAL for a in fail_alerts)

    def test_critical_fail_rate_recommends_early_termination(self):
        system = _make_system(
            max_fail_rate_critical=0.7,
            enable_early_termination_recommendations=True,
        )
        m = self._make_monitor_with_completions(passes=1, fails=9)
        system.check_throughput(m, window_seconds=60.0)
        assert system.should_recommend_early_termination()

    def test_fail_rate_metadata_has_counts(self):
        system = _make_system(max_fail_rate_warning=0.3)
        m = self._make_monitor_with_completions(passes=6, fails=4)
        alerts = system.check_throughput(m, window_seconds=60.0)
        fail_alerts = [a for a in alerts if "fail rate" in a.message]
        assert fail_alerts[0].metadata["fail_count"] == 4
        assert fail_alerts[0].metadata["total"] == 10


# ---------------------------------------------------------------------------
# check_resources()
# ---------------------------------------------------------------------------

class TestCheckResources:
    def test_no_alert_below_thresholds(self):
        system = _make_system(max_memory_mb_warning=512.0, max_cpu_percent_warning=70.0)
        report = _make_resource_report(peak_memory_mb=100.0, peak_cpu=20.0)
        alerts = system.check_resources(report)
        assert alerts == []

    def test_memory_warning_alert(self):
        system = _make_system(max_memory_mb_warning=512.0, max_memory_mb_critical=1024.0)
        report = _make_resource_report(peak_memory_mb=600.0)
        alerts = system.check_resources(report)
        mem_alerts = [a for a in alerts if a.metadata.get("resource") == "memory_rss_mb"]
        assert any(a.severity == AlertSeverity.WARNING for a in mem_alerts)

    def test_memory_critical_alert(self):
        system = _make_system(max_memory_mb_warning=512.0, max_memory_mb_critical=1024.0)
        report = _make_resource_report(peak_memory_mb=1100.0)
        alerts = system.check_resources(report)
        mem_alerts = [a for a in alerts if a.metadata.get("resource") == "memory_rss_mb"]
        assert any(a.severity == AlertSeverity.CRITICAL for a in mem_alerts)

    def test_memory_critical_recommends_early_termination(self):
        system = _make_system(
            max_memory_mb_critical=512.0,
            enable_early_termination_recommendations=True,
        )
        report = _make_resource_report(peak_memory_mb=600.0)
        system.check_resources(report)
        assert system.should_recommend_early_termination()

    def test_cpu_warning_alert(self):
        system = _make_system(max_cpu_percent_warning=70.0, max_cpu_percent_critical=90.0)
        report = _make_resource_report(peak_cpu=75.0)
        alerts = system.check_resources(report)
        cpu_alerts = [a for a in alerts if a.metadata.get("resource") == "cpu_percent"]
        assert any(a.severity == AlertSeverity.WARNING for a in cpu_alerts)

    def test_cpu_critical_alert(self):
        system = _make_system(max_cpu_percent_warning=70.0, max_cpu_percent_critical=90.0)
        report = _make_resource_report(peak_cpu=95.0)
        alerts = system.check_resources(report)
        cpu_alerts = [a for a in alerts if a.metadata.get("resource") == "cpu_percent"]
        assert any(a.severity == AlertSeverity.CRITICAL for a in cpu_alerts)

    def test_both_memory_and_cpu_alerts_fired(self):
        system = _make_system(
            max_memory_mb_warning=100.0, max_memory_mb_critical=999_999.0,
            max_cpu_percent_warning=10.0, max_cpu_percent_critical=999.0,
        )
        report = _make_resource_report(peak_memory_mb=200.0, peak_cpu=50.0)
        alerts = system.check_resources(report)
        resources = {a.metadata.get("resource") for a in alerts}
        assert "memory_rss_mb" in resources
        assert "cpu_percent" in resources


# ---------------------------------------------------------------------------
# check_tokens()
# ---------------------------------------------------------------------------

class TestCheckTokens:
    def test_no_alert_below_thresholds(self):
        system = _make_system(
            max_tokens_per_workflow_warning=20_000,
            max_cost_per_workflow_warning=0.05,
        )
        tracker = _make_token_tracker(input_tokens=100, output_tokens=50)
        alerts = system.check_tokens(tracker)
        assert alerts == []

    def test_token_warning_alert(self):
        system = _make_system(
            max_tokens_per_workflow_warning=100,
            max_tokens_per_workflow_critical=999_999,
        )
        tracker = _make_token_tracker(input_tokens=80, output_tokens=50)
        alerts = system.check_tokens(tracker)
        token_alerts = [a for a in alerts if a.metadata.get("metric") == "total_tokens"]
        assert any(a.severity == AlertSeverity.WARNING for a in token_alerts)

    def test_token_critical_alert(self):
        system = _make_system(
            max_tokens_per_workflow_warning=100,
            max_tokens_per_workflow_critical=200,
        )
        tracker = _make_token_tracker(input_tokens=150, output_tokens=100)
        alerts = system.check_tokens(tracker)
        token_alerts = [a for a in alerts if a.metadata.get("metric") == "total_tokens"]
        assert any(a.severity == AlertSeverity.CRITICAL for a in token_alerts)

    def test_token_critical_recommends_early_termination(self):
        system = _make_system(
            max_tokens_per_workflow_critical=100,
            enable_early_termination_recommendations=True,
        )
        tracker = _make_token_tracker(input_tokens=80, output_tokens=50)
        system.check_tokens(tracker)
        assert system.should_recommend_early_termination()

    def test_cost_warning_alert(self):
        # gpt-4 is expensive: $0.03/1k input, $0.06/1k output
        system = _make_system(
            max_cost_per_workflow_warning=0.001,
            max_cost_per_workflow_critical=999.0,
        )
        tracker = _make_token_tracker(input_tokens=100, output_tokens=100, model="gpt-4")
        alerts = system.check_tokens(tracker)
        cost_alerts = [a for a in alerts if a.metadata.get("metric") == "cost_usd"]
        assert any(a.severity == AlertSeverity.WARNING for a in cost_alerts)

    def test_cost_critical_alert(self):
        system = _make_system(
            max_cost_per_workflow_warning=0.001,
            max_cost_per_workflow_critical=0.005,
        )
        tracker = _make_token_tracker(input_tokens=100, output_tokens=100, model="gpt-4")
        alerts = system.check_tokens(tracker)
        cost_alerts = [a for a in alerts if a.metadata.get("metric") == "cost_usd"]
        assert any(a.severity == AlertSeverity.CRITICAL for a in cost_alerts)

    def test_no_alert_when_no_calls_recorded(self):
        system = _make_system()
        tracker = LLMTokenTracker()
        alerts = system.check_tokens(tracker)
        assert alerts == []


# ---------------------------------------------------------------------------
# Early termination integration
# ---------------------------------------------------------------------------

class TestEarlyTerminationIntegration:
    def test_no_early_termination_when_no_alerts(self):
        system = _make_system()
        assert not system.should_recommend_early_termination()

    def test_no_early_termination_on_warning_only(self):
        system = _make_system(
            warning_latency_ms=100.0,
            critical_latency_ms=999_999.0,
            enable_early_termination_recommendations=True,
        )
        tracker = _make_tracker_with_step("price_check", duration_ms=200.0)
        system.check_latency(tracker)
        assert not system.should_recommend_early_termination()

    def test_early_termination_on_critical_latency(self):
        system = _make_system(
            critical_latency_ms=500.0,
            enable_early_termination_recommendations=True,
        )
        tracker = _make_tracker_with_step("price_check", duration_ms=600.0)
        system.check_latency(tracker)
        assert system.should_recommend_early_termination()

    def test_early_termination_disabled_even_on_critical(self):
        system = _make_system(
            critical_latency_ms=500.0,
            enable_early_termination_recommendations=False,
        )
        tracker = _make_tracker_with_step("price_check", duration_ms=600.0)
        system.check_latency(tracker)
        assert not system.should_recommend_early_termination()

    def test_get_early_termination_reason_none_when_no_critical(self):
        system = _make_system()
        assert system.get_early_termination_reason() is None

    def test_get_early_termination_reason_contains_message(self):
        system = _make_system(
            critical_latency_ms=500.0,
            enable_early_termination_recommendations=True,
        )
        tracker = _make_tracker_with_step("price_check", duration_ms=600.0)
        system.check_latency(tracker)
        reason = system.get_early_termination_reason()
        assert reason is not None
        assert "price_check" in reason or "latency" in reason.lower()

    def test_get_early_termination_reason_aggregates_multiple_messages(self):
        system = _make_system(
            critical_latency_ms=500.0,
            max_memory_mb_critical=100.0,
            enable_early_termination_recommendations=True,
        )
        tracker = _make_tracker_with_step("price_check", duration_ms=600.0)
        system.check_latency(tracker)
        system.check_resources(_make_resource_report(peak_memory_mb=200.0))
        reason = system.get_early_termination_reason()
        assert ";" in reason  # multiple reasons joined


# ---------------------------------------------------------------------------
# get_alert_summary()
# ---------------------------------------------------------------------------

class TestGetAlertSummary:
    def test_summary_has_required_keys(self):
        system = _make_system()
        summary = system.get_alert_summary()
        for key in ("total_alerts", "critical_count", "warning_count",
                    "recommend_early_termination", "early_termination_reason", "alerts"):
            assert key in summary, f"Missing key: {key}"

    def test_summary_counts_correct(self):
        system = _make_system(
            warning_latency_ms=100.0,
            critical_latency_ms=500.0,
            total_latency_warning_ms=999_999.0,
            total_latency_critical_ms=999_999.0,
        )
        tracker = AsyncStepLatencyTracker()
        tracker._records["price_check"] = StepLatencyRecord(
            step_name="price_check", start_time=0.0, end_time=0.2, duration_ms=200.0
        )
        tracker._records["policy_check"] = StepLatencyRecord(
            step_name="policy_check", start_time=0.0, end_time=0.6, duration_ms=600.0
        )
        system.check_latency(tracker)
        summary = system.get_alert_summary()
        assert summary["warning_count"] == 1
        assert summary["critical_count"] == 1
        assert summary["total_alerts"] == 2

    def test_summary_alerts_list_contains_dicts(self):
        system = _make_system(warning_latency_ms=100.0, total_latency_warning_ms=999_999.0,
                               total_latency_critical_ms=999_999.0)
        tracker = _make_tracker_with_step("price_check", duration_ms=200.0)
        system.check_latency(tracker)
        summary = system.get_alert_summary()
        assert isinstance(summary["alerts"], list)
        assert isinstance(summary["alerts"][0], dict)

    def test_summary_empty_when_no_alerts(self):
        system = _make_system()
        summary = system.get_alert_summary()
        assert summary["total_alerts"] == 0
        assert summary["alerts"] == []
        assert not summary["recommend_early_termination"]


# ---------------------------------------------------------------------------
# alert_callback
# ---------------------------------------------------------------------------

class TestAlertCallback:
    def test_callback_invoked_for_each_alert(self):
        fired = []
        t = PerformanceAlertThresholds(
            warning_latency_ms=100.0,
            critical_latency_ms=999_999.0,
            total_latency_warning_ms=999_999.0,
            total_latency_critical_ms=999_999.0,
            alert_callback=fired.append,
        )
        system = PerformanceAlertingSystem(thresholds=t)
        tracker = AsyncStepLatencyTracker()
        for name in ("price_check", "policy_check"):
            tracker._records[name] = StepLatencyRecord(
                step_name=name, start_time=0.0, end_time=0.2, duration_ms=200.0
            )
        system.check_latency(tracker)
        assert len(fired) == 2

    def test_callback_receives_performance_alert(self):
        received = []
        t = PerformanceAlertThresholds(
            warning_latency_ms=100.0,
            total_latency_warning_ms=999_999.0,
            total_latency_critical_ms=999_999.0,
            alert_callback=received.append,
        )
        system = PerformanceAlertingSystem(thresholds=t)
        tracker = _make_tracker_with_step("price_check", duration_ms=200.0)
        system.check_latency(tracker)
        assert isinstance(received[0], PerformanceAlert)

    def test_callback_exception_does_not_break_alerting(self):
        def bad_callback(alert):
            raise RuntimeError("callback error")

        t = PerformanceAlertThresholds(
            warning_latency_ms=100.0,
            total_latency_warning_ms=999_999.0,
            total_latency_critical_ms=999_999.0,
            alert_callback=bad_callback,
        )
        system = PerformanceAlertingSystem(thresholds=t)
        tracker = _make_tracker_with_step("price_check", duration_ms=200.0)
        # Should not raise
        system.check_latency(tracker)
        assert system.has_alerts()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_alerts(self):
        system = _make_system(warning_latency_ms=100.0, total_latency_warning_ms=999_999.0,
                               total_latency_critical_ms=999_999.0)
        tracker = _make_tracker_with_step("price_check", duration_ms=200.0)
        system.check_latency(tracker)
        assert system.has_alerts()
        system.reset()
        assert not system.has_alerts()

    def test_reset_clears_early_termination_recommendation(self):
        system = _make_system(
            critical_latency_ms=100.0,
            enable_early_termination_recommendations=True,
        )
        tracker = _make_tracker_with_step("price_check", duration_ms=200.0)
        system.check_latency(tracker)
        assert system.should_recommend_early_termination()
        system.reset()
        assert not system.should_recommend_early_termination()


# ---------------------------------------------------------------------------
# create_performance_alerting_system() factory
# ---------------------------------------------------------------------------

class TestCreatePerformanceAlertingSystem:
    def test_default_environment_returns_system(self):
        system = create_performance_alerting_system()
        assert isinstance(system, PerformanceAlertingSystem)

    def test_production_has_stricter_latency_threshold(self):
        prod = create_performance_alerting_system(environment="production")
        dev = create_performance_alerting_system(environment="development")
        assert prod._thresholds.critical_latency_ms < dev._thresholds.critical_latency_ms

    def test_production_enables_early_termination(self):
        prod = create_performance_alerting_system(environment="production")
        assert prod._thresholds.enable_early_termination_recommendations is True

    def test_development_disables_early_termination(self):
        dev = create_performance_alerting_system(environment="development")
        assert dev._thresholds.enable_early_termination_recommendations is False

    def test_testing_disables_early_termination(self):
        test = create_performance_alerting_system(environment="testing")
        assert test._thresholds.enable_early_termination_recommendations is False

    def test_callback_passed_to_system(self):
        fired = []
        system = create_performance_alerting_system(
            environment="production",
            alert_callback=fired.append,
        )
        # Trigger a critical latency alert (production threshold is 5000ms)
        tracker = _make_tracker_with_step("price_check", duration_ms=6000.0)
        system.check_latency(tracker)
        assert len(fired) > 0

    def test_unknown_environment_uses_defaults(self):
        system = create_performance_alerting_system(environment="unknown_env")
        assert isinstance(system, PerformanceAlertingSystem)
        # Should use default thresholds
        assert system._thresholds.critical_latency_ms == PerformanceAlertThresholds().critical_latency_ms
