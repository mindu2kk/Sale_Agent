"""
Tests for Task 5.3.1: Latency tracking for each async verification step.

Covers:
- AsyncStepLatencyTracker context manager records step_name, start_time,
  end_time, duration_ms for each async verification step.
- Latency metrics are stored in VerificationResult.step_latencies.
- Latency metrics are propagated into ExecutionStep.metrics via workflow node.
- Aggregate latency summary (total_ms, max_ms, min_ms, avg_ms, step_count).
- Error paths: failed steps are still recorded with success=False.
"""

import asyncio
import time
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from backend.verification.utils.performance import AsyncStepLatencyTracker, StepLatencyRecord
from backend.verification.agent.verification_agent import VerificationAgent
from backend.verification.workflow.workflow import VerificationWorkflow
from backend.verification.config.config import VerificationConfig
from backend.verification.models.verification import (
    VerificationResult, RubricCriteria,
)
from backend.verification.models.execution import ExecutionStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    defaults = dict(
        price_tolerance_percent=1.0, price_critical_threshold=30.0,
        max_retries=3, parallel_verification=True, early_termination=False,
        async_timeout_seconds=10, enable_caching=False, llm_model_name="gpt-4",
    )
    defaults.update(overrides)
    return VerificationConfig(**defaults)


def _make_criteria(price_pass=True, policy_pass=True, relevance_pass=True):
    return RubricCriteria(
        price_accuracy_pass=price_pass,
        policy_authenticity_pass=policy_pass,
        topic_relevance_pass=relevance_pass,
        price_issues=[], policy_issues=[], relevance_issues=[],
    )


def _make_vr(criteria=None, step_latencies=None):
    if criteria is None:
        criteria = _make_criteria()
    vr = VerificationResult(
        criteria=criteria,
        verification_reasoning="Verification completed successfully",
        execution_time_seconds=0.5,
        llm_tokens_used=100,
    )
    vr.step_latencies = step_latencies
    return vr


def _make_state(**overrides):
    base = {
        "objection_text": "San pham co bao hanh khong?",
        "draft_response": "San pham duoc bao hanh 12 thang.",
        "tools_used": [], "research_reasoning": "", "research_sources": [],
        "verification_result": None, "correction_feedback": None,
        "retry_count": 0, "max_retries": 3, "final_response": "",
        "workflow_status": "researching", "execution_log": [],
        "start_time": datetime.now().isoformat(), "end_time": None,
        "config": {}, "customer_context": {}, "resource_usage": {},
        "error_log": [], "workflow_id": "wf_test", "correlation_id": "corr_test",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Unit tests: AsyncStepLatencyTracker
# ---------------------------------------------------------------------------

class TestAsyncStepLatencyTrackerContextManager:
    """Tests for the async context manager interface."""

    @pytest.mark.asyncio
    async def test_records_step_name(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        record = tracker.get_record("price_check")
        assert record is not None
        assert record.step_name == "price_check"

    @pytest.mark.asyncio
    async def test_records_duration_ms_positive(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("policy_check"):
            await asyncio.sleep(0.01)
        record = tracker.get_record("policy_check")
        assert record.duration_ms >= 10.0  # at least 10ms

    @pytest.mark.asyncio
    async def test_records_start_and_end_time(self):
        tracker = AsyncStepLatencyTracker()
        before = time.time()
        async with tracker.track("relevance_check"):
            pass
        after = time.time()
        record = tracker.get_record("relevance_check")
        assert before <= record.start_time <= record.end_time <= after

    @pytest.mark.asyncio
    async def test_end_time_after_start_time(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            await asyncio.sleep(0.005)
        record = tracker.get_record("price_check")
        assert record.end_time >= record.start_time

    @pytest.mark.asyncio
    async def test_success_true_on_normal_completion(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        assert tracker.get_record("price_check").success is True

    @pytest.mark.asyncio
    async def test_success_false_on_exception(self):
        tracker = AsyncStepLatencyTracker()
        with pytest.raises(ValueError):
            async with tracker.track("price_check"):
                raise ValueError("test error")
        record = tracker.get_record("price_check")
        assert record.success is False
        assert "test error" in record.error

    @pytest.mark.asyncio
    async def test_error_message_stored_on_exception(self):
        tracker = AsyncStepLatencyTracker()
        with pytest.raises(RuntimeError):
            async with tracker.track("policy_check"):
                raise RuntimeError("connection failed")
        assert tracker.get_record("policy_check").error == "connection failed"

    @pytest.mark.asyncio
    async def test_duration_ms_matches_elapsed_time(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            await asyncio.sleep(0.05)
        record = tracker.get_record("price_check")
        # duration_ms should be approximately 50ms (allow generous tolerance for CI)
        assert 30.0 <= record.duration_ms <= 500.0

    @pytest.mark.asyncio
    async def test_multiple_steps_tracked_independently(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            await asyncio.sleep(0.01)
        async with tracker.track("policy_check"):
            await asyncio.sleep(0.02)
        async with tracker.track("relevance_check"):
            pass

        assert tracker.get_record("price_check") is not None
        assert tracker.get_record("policy_check") is not None
        assert tracker.get_record("relevance_check") is not None

    @pytest.mark.asyncio
    async def test_get_all_records_returns_all_steps(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        async with tracker.track("policy_check"):
            pass
        records = tracker.get_all_records()
        names = [r.step_name for r in records]
        assert "price_check" in names
        assert "policy_check" in names

    @pytest.mark.asyncio
    async def test_unknown_step_returns_none(self):
        tracker = AsyncStepLatencyTracker()
        assert tracker.get_record("nonexistent") is None

    @pytest.mark.asyncio
    async def test_reset_clears_all_records(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        tracker.reset()
        assert tracker.get_record("price_check") is None
        assert tracker.get_all_records() == []


class TestAsyncStepLatencyTrackerMetrics:
    """Tests for get_all_metrics() output format."""

    @pytest.mark.asyncio
    async def test_metrics_contains_per_step_latency_keys(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        async with tracker.track("policy_check"):
            pass
        metrics = tracker.get_all_metrics()
        assert "step_latency_price_check_ms" in metrics
        assert "step_latency_policy_check_ms" in metrics

    @pytest.mark.asyncio
    async def test_metrics_per_step_latency_is_positive(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            await asyncio.sleep(0.01)
        metrics = tracker.get_all_metrics()
        assert metrics["step_latency_price_check_ms"] > 0

    @pytest.mark.asyncio
    async def test_metrics_contains_step_latencies_dict(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        metrics = tracker.get_all_metrics()
        assert "step_latencies" in metrics
        assert "price_check" in metrics["step_latencies"]

    @pytest.mark.asyncio
    async def test_step_latencies_dict_has_required_fields(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        detail = tracker.get_all_metrics()["step_latencies"]["price_check"]
        assert "step_name" in detail
        assert "start_time" in detail
        assert "end_time" in detail
        assert "duration_ms" in detail
        assert "success" in detail

    @pytest.mark.asyncio
    async def test_aggregate_latency_present(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        metrics = tracker.get_all_metrics()
        assert "aggregate_latency" in metrics

    @pytest.mark.asyncio
    async def test_aggregate_latency_fields(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        agg = tracker.get_all_metrics()["aggregate_latency"]
        assert "total_ms" in agg
        assert "max_ms" in agg
        assert "min_ms" in agg
        assert "avg_ms" in agg
        assert "step_count" in agg

    @pytest.mark.asyncio
    async def test_aggregate_step_count_correct(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        async with tracker.track("policy_check"):
            pass
        async with tracker.track("relevance_check"):
            pass
        agg = tracker.get_all_metrics()["aggregate_latency"]
        assert agg["step_count"] == 3

    @pytest.mark.asyncio
    async def test_aggregate_total_ms_is_sum_of_steps(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            pass
        async with tracker.track("policy_check"):
            pass
        metrics = tracker.get_all_metrics()
        agg = metrics["aggregate_latency"]
        expected_total = (
            metrics["step_latency_price_check_ms"]
            + metrics["step_latency_policy_check_ms"]
        )
        assert abs(agg["total_ms"] - expected_total) < 0.01

    @pytest.mark.asyncio
    async def test_empty_tracker_aggregate_zeros(self):
        tracker = AsyncStepLatencyTracker()
        agg = tracker.get_all_metrics()["aggregate_latency"]
        assert agg["step_count"] == 0
        assert agg["total_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_aggregate_max_gte_min(self):
        tracker = AsyncStepLatencyTracker()
        async with tracker.track("price_check"):
            await asyncio.sleep(0.01)
        async with tracker.track("policy_check"):
            pass
        agg = tracker.get_all_metrics()["aggregate_latency"]
        assert agg["max_ms"] >= agg["min_ms"]


class TestStepLatencyRecordToDict:
    """Tests for StepLatencyRecord.to_dict()."""

    def test_to_dict_contains_all_fields(self):
        record = StepLatencyRecord(
            step_name="price_check",
            start_time=1000.0,
            end_time=1000.1,
            duration_ms=100.0,
            success=True,
            error=None,
        )
        d = record.to_dict()
        assert d["step_name"] == "price_check"
        assert d["start_time"] == 1000.0
        assert d["end_time"] == 1000.1
        assert d["duration_ms"] == 100.0
        assert d["success"] is True
        assert d["error"] is None

    def test_to_dict_rounds_duration_ms(self):
        record = StepLatencyRecord(
            step_name="price_check",
            start_time=0.0,
            end_time=0.1,
            duration_ms=100.123456789,
        )
        d = record.to_dict()
        assert d["duration_ms"] == round(100.123456789, 3)


# ---------------------------------------------------------------------------
# Integration tests: VerificationResult.step_latencies
# ---------------------------------------------------------------------------

class TestVerificationResultStepLatencies:
    """Tests that VerificationResult stores step latency data."""

    def test_step_latencies_field_defaults_to_none(self):
        vr = _make_vr()
        assert vr.step_latencies is None

    def test_step_latencies_can_be_set(self):
        vr = _make_vr()
        latencies = {"step_latency_price_check_ms": 50.0}
        vr.step_latencies = latencies
        assert vr.step_latencies == latencies

    def test_step_latencies_accepts_tracker_output(self):
        """VerificationResult.step_latencies accepts the dict from get_all_metrics()."""
        tracker = AsyncStepLatencyTracker()
        # Simulate synchronous record insertion for testing
        tracker._records["price_check"] = StepLatencyRecord(
            step_name="price_check",
            start_time=1000.0,
            end_time=1000.05,
            duration_ms=50.0,
        )
        vr = _make_vr()
        vr.step_latencies = tracker.get_all_metrics()
        assert "step_latency_price_check_ms" in vr.step_latencies
        assert "aggregate_latency" in vr.step_latencies


# ---------------------------------------------------------------------------
# Integration tests: ExecutionStep.metrics includes step latencies
# ---------------------------------------------------------------------------

class TestWorkflowNodeLatencyPropagation:
    """Tests that the verification workflow node propagates step latencies into ExecutionStep.metrics."""

    def _make_workflow_with_latencies(self, latencies=None):
        config = _make_config()
        ra = MagicMock()
        va = MagicMock()
        vr = _make_vr(step_latencies=latencies)
        va.verify_draft = AsyncMock(return_value=vr)
        wf = VerificationWorkflow(research_agent=ra, verification_agent=va, config=config)
        return wf

    def test_step_latencies_in_execution_metrics_when_present(self):
        latencies = {
            "step_latency_price_check_ms": 45.2,
            "step_latency_policy_check_ms": 38.7,
            "step_latency_relevance_check_ms": 62.1,
            "step_latencies": {},
            "aggregate_latency": {"total_ms": 146.0, "step_count": 3,
                                   "max_ms": 62.1, "min_ms": 38.7, "avg_ms": 48.7},
        }
        wf = self._make_workflow_with_latencies(latencies)
        result = wf._execute_verification_node(_make_state())
        metrics = result["execution_log"][0].metrics
        assert "step_latency_price_check_ms" in metrics
        assert metrics["step_latency_price_check_ms"] == 45.2
        assert "step_latency_policy_check_ms" in metrics
        assert "step_latency_relevance_check_ms" in metrics

    def test_aggregate_latency_in_execution_metrics(self):
        latencies = {
            "aggregate_latency": {"total_ms": 100.0, "step_count": 3,
                                   "max_ms": 50.0, "min_ms": 20.0, "avg_ms": 33.3},
        }
        wf = self._make_workflow_with_latencies(latencies)
        result = wf._execute_verification_node(_make_state())
        metrics = result["execution_log"][0].metrics
        assert "aggregate_latency" in metrics
        assert metrics["aggregate_latency"]["step_count"] == 3

    def test_no_latency_data_does_not_break_execution(self):
        """When step_latencies is None, execution still succeeds."""
        wf = self._make_workflow_with_latencies(latencies=None)
        result = wf._execute_verification_node(_make_state())
        assert result["execution_log"][0].status == ExecutionStatus.SUCCESS

    def test_existing_metrics_preserved_alongside_latencies(self):
        latencies = {"step_latency_price_check_ms": 30.0}
        wf = self._make_workflow_with_latencies(latencies)
        result = wf._execute_verification_node(_make_state())
        metrics = result["execution_log"][0].metrics
        # Core metrics still present
        assert "overall_pass" in metrics
        assert "critical_issues" in metrics
        assert "tokens_used" in metrics
        # Latency metrics also present
        assert "step_latency_price_check_ms" in metrics


# ---------------------------------------------------------------------------
# Integration tests: VerificationAgent async methods produce latency data
# ---------------------------------------------------------------------------

class TestVerificationAgentLatencyIntegration:
    """Tests that VerificationAgent._verify_parallel_simple and _verify_sequential
    produce VerificationResult with step_latencies populated."""

    def _make_agent(self, parallel=True, early_termination=False):
        config = _make_config(
            parallel_verification=parallel,
            early_termination=early_termination,
        )
        llm = MagicMock()
        rag = MagicMock()
        agent = VerificationAgent(llm=llm, rag_pipeline=rag, config=config)

        # Mock the three checker methods to return immediately
        agent.price_checker.check_price_accuracy = MagicMock(return_value=(True, []))
        agent.policy_checker.check_policy_authenticity = MagicMock(return_value=(True, []))
        agent.relevance_checker.check_topic_relevance = MagicMock(return_value=(True, []))

        return agent

    def _make_agent_state(self):
        return {
            "draft_response": "San pham duoc bao hanh 12 thang.",
            "objection_text": "San pham co bao hanh khong?",
        }

    @pytest.mark.asyncio
    async def test_parallel_simple_produces_step_latencies(self):
        agent = self._make_agent(parallel=True, early_termination=False)
        state = self._make_agent_state()
        result = await agent._verify_parallel_simple(state)
        assert result.step_latencies is not None

    @pytest.mark.asyncio
    async def test_parallel_simple_latencies_has_all_three_steps(self):
        agent = self._make_agent(parallel=True, early_termination=False)
        state = self._make_agent_state()
        result = await agent._verify_parallel_simple(state)
        latencies = result.step_latencies
        assert "step_latency_price_check_ms" in latencies
        assert "step_latency_policy_check_ms" in latencies
        assert "step_latency_relevance_check_ms" in latencies

    @pytest.mark.asyncio
    async def test_sequential_produces_step_latencies(self):
        agent = self._make_agent(parallel=False, early_termination=False)
        state = self._make_agent_state()
        result = await agent._verify_sequential(state)
        assert result.step_latencies is not None

    @pytest.mark.asyncio
    async def test_sequential_latencies_has_all_three_steps(self):
        agent = self._make_agent(parallel=False, early_termination=False)
        state = self._make_agent_state()
        result = await agent._verify_sequential(state)
        latencies = result.step_latencies
        assert "step_latency_price_check_ms" in latencies
        assert "step_latency_policy_check_ms" in latencies
        assert "step_latency_relevance_check_ms" in latencies

    @pytest.mark.asyncio
    async def test_parallel_with_early_termination_produces_step_latencies(self):
        agent = self._make_agent(parallel=True, early_termination=True)
        state = self._make_agent_state()
        result = await agent._verify_parallel(state)
        assert result.step_latencies is not None

    @pytest.mark.asyncio
    async def test_latencies_aggregate_step_count_correct(self):
        agent = self._make_agent(parallel=False, early_termination=False)
        state = self._make_agent_state()
        result = await agent._verify_sequential(state)
        agg = result.step_latencies.get("aggregate_latency", {})
        assert agg.get("step_count") == 3

    @pytest.mark.asyncio
    async def test_latencies_duration_ms_non_negative(self):
        agent = self._make_agent(parallel=False, early_termination=False)
        state = self._make_agent_state()
        result = await agent._verify_sequential(state)
        for key, val in result.step_latencies.items():
            if key.startswith("step_latency_") and key.endswith("_ms"):
                assert val >= 0.0
