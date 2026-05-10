"""
Tests for LLM Token Usage Tracking with Cost Optimization Alerts - Task 5.3.4

Covers:
- Per-call token recording and cost estimation
- Aggregate summary correctness
- Per-call and cumulative alert thresholds
- Alert callback invocation
- Context-manager (sync + async) API
- Integration helpers for ExecutionStep and WorkflowMetrics
- Model pricing lookup (exact, prefix, fallback)
- Reset / lifecycle
"""

import asyncio
import pytest
from typing import List

from verification.utils.token_tracker import (
    DEFAULT_MODEL_PRICING,
    CostAlert,
    LLMTokenTracker,
    TokenAlertThresholds,
    TokenUsageRecord,
    estimate_cost,
    get_token_tracker,
    reset_token_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(**threshold_kwargs) -> LLMTokenTracker:
    thresholds = TokenAlertThresholds(**threshold_kwargs) if threshold_kwargs else None
    return LLMTokenTracker(thresholds=thresholds)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost(1000, 500, model="gpt-4o")
        in_rate, out_rate = DEFAULT_MODEL_PRICING["gpt-4o"]
        expected = (1000 * in_rate + 500 * out_rate) / 1000.0
        assert abs(cost - expected) < 1e-9

    def test_prefix_match(self):
        # "gpt-4o-2024-05-13" should match "gpt-4o" prefix
        cost_prefix = estimate_cost(100, 50, model="gpt-4o-2024-05-13")
        cost_exact = estimate_cost(100, 50, model="gpt-4o")
        assert abs(cost_prefix - cost_exact) < 1e-9

    def test_unknown_model_uses_default(self):
        cost = estimate_cost(1000, 1000, model="unknown-model-xyz")
        in_rate, out_rate = DEFAULT_MODEL_PRICING["default"]
        expected = (1000 * in_rate + 1000 * out_rate) / 1000.0
        assert abs(cost - expected) < 1e-9

    def test_zero_tokens(self):
        assert estimate_cost(0, 0) == 0.0


# ---------------------------------------------------------------------------
# LLMTokenTracker.record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_creates_entry(self):
        tracker = _make_tracker()
        rec = tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        assert isinstance(rec, TokenUsageRecord)
        assert rec.input_tokens == 100
        assert rec.output_tokens == 50
        assert rec.total_tokens == 150
        assert rec.cost_usd > 0
        assert len(tracker) == 1

    def test_record_node_name_and_correlation(self):
        tracker = _make_tracker()
        rec = tracker.record(
            "gpt-4o-mini",
            input_tokens=200,
            output_tokens=80,
            node_name="verification",
            correlation_id="corr-123",
        )
        assert rec.node_name == "verification"
        assert rec.correlation_id == "corr-123"

    def test_multiple_records_accumulate(self):
        tracker = _make_tracker()
        tracker.record("gpt-4o", 100, 50)
        tracker.record("gpt-4o", 200, 100)
        assert len(tracker) == 2

    def test_to_dict_has_required_keys(self):
        tracker = _make_tracker()
        rec = tracker.record("gpt-4o", 100, 50, node_name="price_check")
        d = rec.to_dict()
        for key in ("timestamp", "model", "input_tokens", "output_tokens",
                    "total_tokens", "cost_usd", "node_name", "correlation_id"):
            assert key in d


# ---------------------------------------------------------------------------
# TokenUsageSummary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self):
        tracker = _make_tracker()
        s = tracker.summary()
        assert s.total_tokens == 0
        assert s.total_cost_usd == 0.0
        assert s.call_count == 0
        assert s.avg_tokens_per_call == 0.0
        assert s.avg_cost_per_call == 0.0

    def test_summary_aggregates_correctly(self):
        tracker = _make_tracker()
        tracker.record("gpt-4o", 100, 50)
        tracker.record("gpt-4o", 200, 100)
        s = tracker.summary()
        assert s.total_input_tokens == 300
        assert s.total_output_tokens == 150
        assert s.total_tokens == 450
        assert s.call_count == 2
        assert s.avg_tokens_per_call == 225.0

    def test_summary_to_dict(self):
        tracker = _make_tracker()
        tracker.record("gpt-4o", 100, 50)
        d = tracker.summary().to_dict()
        for key in ("total_input_tokens", "total_output_tokens", "total_tokens",
                    "total_cost_usd", "call_count", "avg_tokens_per_call", "avg_cost_per_call"):
            assert key in d


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_no_alerts_below_threshold(self):
        tracker = _make_tracker(
            max_per_call_cost_usd=10.0,
            max_per_call_tokens=100_000,
            max_total_cost_usd=100.0,
            max_total_tokens=1_000_000,
        )
        tracker.record("gpt-4o", 100, 50)
        assert not tracker.has_alerts()

    def test_per_call_cost_alert(self):
        tracker = _make_tracker(max_per_call_cost_usd=0.0001)
        tracker.record("gpt-4o", 1000, 500)  # will exceed $0.0001
        alerts = [a for a in tracker.alerts if a.alert_type == "per_call_cost"]
        assert len(alerts) >= 1
        assert alerts[0].actual_value > alerts[0].threshold

    def test_per_call_token_alert(self):
        tracker = _make_tracker(max_per_call_tokens=10)
        tracker.record("gpt-4o", 100, 50)
        alerts = [a for a in tracker.alerts if a.alert_type == "per_call_tokens"]
        assert len(alerts) >= 1

    def test_cumulative_cost_alert(self):
        tracker = _make_tracker(max_total_cost_usd=0.0001)
        tracker.record("gpt-4o", 1000, 500)
        alerts = [a for a in tracker.alerts if a.alert_type == "total_cost"]
        assert len(alerts) >= 1

    def test_cumulative_token_alert(self):
        tracker = _make_tracker(max_total_tokens=50)
        tracker.record("gpt-4o", 100, 50)
        alerts = [a for a in tracker.alerts if a.alert_type == "total_tokens"]
        assert len(alerts) >= 1

    def test_alert_callback_invoked(self):
        fired: List[CostAlert] = []
        thresholds = TokenAlertThresholds(
            max_per_call_cost_usd=0.0001,
            alert_callback=fired.append,
        )
        tracker = LLMTokenTracker(thresholds=thresholds)
        tracker.record("gpt-4o", 1000, 500)
        assert len(fired) >= 1
        assert isinstance(fired[0], CostAlert)

    def test_alert_callback_exception_does_not_propagate(self):
        def bad_callback(alert):
            raise RuntimeError("callback error")

        thresholds = TokenAlertThresholds(
            max_per_call_cost_usd=0.0001,
            alert_callback=bad_callback,
        )
        tracker = LLMTokenTracker(thresholds=thresholds)
        # Should not raise
        tracker.record("gpt-4o", 1000, 500)

    def test_alert_to_dict(self):
        tracker = _make_tracker(max_per_call_cost_usd=0.0001)
        tracker.record("gpt-4o", 1000, 500)
        alert = tracker.alerts[0]
        d = alert.to_dict()
        for key in ("alert_type", "threshold", "actual_value", "message", "timestamp"):
            assert key in d


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

class TestContextManagers:
    def test_sync_track_call(self):
        tracker = _make_tracker()
        with tracker.track_call("gpt-4o", node_name="price_check") as ctx:
            ctx.input_tokens = 300
            ctx.output_tokens = 150
        assert len(tracker) == 1
        rec = tracker.summary().records[0]
        assert rec.input_tokens == 300
        assert rec.output_tokens == 150
        assert rec.node_name == "price_check"

    def test_sync_track_call_exception_still_records(self):
        tracker = _make_tracker()
        try:
            with tracker.track_call("gpt-4o") as ctx:
                ctx.input_tokens = 100
                ctx.output_tokens = 50
                raise ValueError("simulated error")
        except ValueError:
            pass
        # Record should still be committed
        assert len(tracker) == 1

    def test_async_track_call(self):
        async def _run():
            tracker = _make_tracker()
            async with tracker.async_track_call("gpt-4o-mini", node_name="policy_check") as ctx:
                ctx.input_tokens = 400
                ctx.output_tokens = 200
            assert len(tracker) == 1
            rec = tracker.summary().records[0]
            assert rec.input_tokens == 400
            assert rec.node_name == "policy_check"

        asyncio.run(_run())

    def test_async_track_call_exception_still_records(self):
        async def _run():
            tracker = _make_tracker()
            try:
                async with tracker.async_track_call("gpt-4o") as ctx:
                    ctx.input_tokens = 100
                    ctx.output_tokens = 50
                    raise RuntimeError("async error")
            except RuntimeError:
                pass
            assert len(tracker) == 1

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

class TestIntegrationHelpers:
    def test_get_metrics_for_execution_step_empty(self):
        tracker = _make_tracker()
        m = tracker.get_metrics_for_execution_step()
        assert m["llm_total_tokens"] == 0
        assert m["llm_call_count"] == 0
        assert "llm_last_call_model" not in m

    def test_get_metrics_for_execution_step_with_records(self):
        tracker = _make_tracker()
        tracker.record("gpt-4o", 100, 50, node_name="verification")
        m = tracker.get_metrics_for_execution_step()
        assert m["llm_total_input_tokens"] == 100
        assert m["llm_total_output_tokens"] == 50
        assert m["llm_total_tokens"] == 150
        assert m["llm_call_count"] == 1
        assert m["llm_last_call_model"] == "gpt-4o"
        assert m["llm_last_call_input_tokens"] == 100

    def test_get_workflow_metrics_fields(self):
        tracker = _make_tracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("gpt-4o", 500, 250)
        fields = tracker.get_workflow_metrics_fields()
        assert fields["llm_tokens_used"] == 2250
        assert fields["llm_tokens_input"] == 1500
        assert fields["llm_tokens_output"] == 750
        assert fields["cost_estimate"] > 0


# ---------------------------------------------------------------------------
# Reset / lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_reset_clears_records_and_alerts(self):
        tracker = _make_tracker(max_per_call_cost_usd=0.0001)
        tracker.record("gpt-4o", 1000, 500)
        assert len(tracker) == 1
        assert tracker.has_alerts()
        tracker.reset()
        assert len(tracker) == 0
        assert not tracker.has_alerts()

    def test_singleton_get_and_reset(self):
        reset_token_tracker()
        t1 = get_token_tracker()
        t2 = get_token_tracker()
        assert t1 is t2
        reset_token_tracker()
        t3 = get_token_tracker()
        assert t3 is not t1
