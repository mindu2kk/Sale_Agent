"""
Tests for MetricsCollector - Binary Decision Tracking

Covers:
- Binary decision tracking (PASS/FAIL per checker)
- Aggregation correctness (pass_rate calculation)
- Correlation ID linking
- Metrics export format (WorkflowMetrics compatibility)

Requirements: 7.1, 7.2, 7.3
"""

import pytest
from verification.utils.metrics_collector import MetricsCollector
from verification.models.execution import WorkflowMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collector():
    """Fresh MetricsCollector for each test."""
    return MetricsCollector()


# ---------------------------------------------------------------------------
# 1. Binary decision tracking (PASS/FAIL per checker)
# ---------------------------------------------------------------------------

class TestBinaryDecisionTracking:
    """Tests for per-checker binary PASS/FAIL recording."""

    def test_record_price_pass(self, collector):
        collector.record_checker_decision("wf1", "corr1", "price", passed=True, execution_time=0.5)
        stats = collector.get_checker_stats("price")
        assert stats["total_runs"] == 1
        assert stats["pass_count"] == 1
        assert stats["fail_count"] == 0

    def test_record_price_fail(self, collector):
        collector.record_checker_decision("wf1", "corr1", "price", passed=False, execution_time=0.3)
        stats = collector.get_checker_stats("price")
        assert stats["total_runs"] == 1
        assert stats["pass_count"] == 0
        assert stats["fail_count"] == 1

    def test_record_policy_pass(self, collector):
        collector.record_checker_decision("wf1", "corr1", "policy", passed=True)
        stats = collector.get_checker_stats("policy")
        assert stats["pass_count"] == 1

    def test_record_relevance_fail(self, collector):
        collector.record_checker_decision("wf1", "corr1", "relevance", passed=False)
        stats = collector.get_checker_stats("relevance")
        assert stats["fail_count"] == 1

    def test_multiple_decisions_same_checker(self, collector):
        for passed in [True, True, False, True, False]:
            collector.record_checker_decision("wf1", "corr1", "price", passed=passed)
        stats = collector.get_checker_stats("price")
        assert stats["total_runs"] == 5
        assert stats["pass_count"] == 3
        assert stats["fail_count"] == 2

    def test_all_three_checkers_independent(self, collector):
        collector.record_checker_decision("wf1", "corr1", "price", passed=True)
        collector.record_checker_decision("wf1", "corr1", "policy", passed=False)
        collector.record_checker_decision("wf1", "corr1", "relevance", passed=True)

        assert collector.get_checker_stats("price")["pass_count"] == 1
        assert collector.get_checker_stats("policy")["fail_count"] == 1
        assert collector.get_checker_stats("relevance")["pass_count"] == 1

    def test_unknown_checker_raises(self, collector):
        with pytest.raises(ValueError, match="Unknown checker"):
            collector.record_checker_decision("wf1", "corr1", "unknown_checker", passed=True)

    def test_checker_name_case_insensitive(self, collector):
        collector.record_checker_decision("wf1", "corr1", "PRICE", passed=True)
        stats = collector.get_checker_stats("price")
        assert stats["total_runs"] == 1

    def test_workflow_decision_pass(self, collector):
        collector.record_workflow_decision(
            "wf1", "corr1", passed=True, final_status="approved",
            total_execution_time=3.0, retry_count=0, escalated=False
        )
        summary = collector.get_workflow_summary()
        assert summary["total_workflows"] == 1
        assert summary["pass_count"] == 1
        assert summary["fail_count"] == 0

    def test_workflow_decision_fail_escalated(self, collector):
        collector.record_workflow_decision(
            "wf1", "corr1", passed=False, final_status="escalated",
            total_execution_time=5.0, retry_count=3, escalated=True
        )
        summary = collector.get_workflow_summary()
        assert summary["fail_count"] == 1
        assert summary["total_retries"] == 3
        assert summary["total_escalations"] == 1


# ---------------------------------------------------------------------------
# 2. Aggregation correctness (pass_rate calculation)
# ---------------------------------------------------------------------------

class TestAggregationCorrectness:
    """Tests for pass_rate and other aggregate calculations."""

    def test_pass_rate_all_pass(self, collector):
        for _ in range(4):
            collector.record_checker_decision("wf", "c", "price", passed=True)
        assert collector.get_checker_stats("price")["pass_rate"] == 1.0

    def test_pass_rate_all_fail(self, collector):
        for _ in range(3):
            collector.record_checker_decision("wf", "c", "policy", passed=False)
        assert collector.get_checker_stats("policy")["pass_rate"] == 0.0

    def test_pass_rate_mixed(self, collector):
        for passed in [True, True, False, True]:
            collector.record_checker_decision("wf", "c", "relevance", passed=passed)
        stats = collector.get_checker_stats("relevance")
        assert abs(stats["pass_rate"] - 0.75) < 1e-9

    def test_pass_rate_zero_runs(self, collector):
        stats = collector.get_checker_stats("price")
        assert stats["pass_rate"] == 0.0
        assert stats["total_runs"] == 0

    def test_workflow_pass_rate(self, collector):
        for passed in [True, True, False]:
            collector.record_workflow_decision(
                "wf", "c", passed=passed, final_status="approved" if passed else "failed",
                total_execution_time=1.0
            )
        summary = collector.get_workflow_summary()
        assert abs(summary["pass_rate"] - 2 / 3) < 1e-9

    def test_escalation_rate(self, collector):
        collector.record_workflow_decision("wf1", "c1", passed=False, final_status="escalated",
                                           total_execution_time=1.0, escalated=True)
        collector.record_workflow_decision("wf2", "c2", passed=True, final_status="approved",
                                           total_execution_time=1.0, escalated=False)
        summary = collector.get_workflow_summary()
        assert abs(summary["escalation_rate"] - 0.5) < 1e-9

    def test_average_execution_time(self, collector):
        for t in [1.0, 2.0, 3.0]:
            collector.record_checker_decision("wf", "c", "price", passed=True, execution_time=t)
        stats = collector.get_checker_stats("price")
        assert abs(stats["average_execution_time"] - 2.0) < 1e-9

    def test_node_stats_aggregation(self, collector):
        for t in [0.5, 1.5, 1.0]:
            collector.record_node_execution("verification", t)
        node_stats = collector.get_node_stats()
        assert "verification" in node_stats
        assert node_stats["verification"]["total_executions"] == 3
        assert abs(node_stats["verification"]["average_time"] - 1.0) < 1e-9

    def test_all_checker_stats_returns_all_three(self, collector):
        all_stats = collector.get_all_checker_stats()
        assert set(all_stats.keys()) == {"price", "policy", "relevance"}

    def test_retry_count_accumulates(self, collector):
        collector.record_workflow_decision("wf1", "c1", passed=False, final_status="escalated",
                                           total_execution_time=1.0, retry_count=2)
        collector.record_workflow_decision("wf2", "c2", passed=False, final_status="escalated",
                                           total_execution_time=1.0, retry_count=3)
        summary = collector.get_workflow_summary()
        assert summary["total_retries"] == 5


# ---------------------------------------------------------------------------
# 3. Correlation ID linking
# ---------------------------------------------------------------------------

class TestCorrelationIDLinking:
    """Tests for linking metrics to workflow runs via correlation IDs."""

    def test_decisions_linked_to_correlation_id(self, collector):
        collector.record_checker_decision("wf1", "corr-A", "price", passed=True)
        collector.record_checker_decision("wf1", "corr-A", "policy", passed=False)
        collector.record_checker_decision("wf2", "corr-B", "price", passed=True)

        linked = collector.get_decisions_by_correlation_id("corr-A")
        assert linked["correlation_id"] == "corr-A"
        assert len(linked["checker_decisions"]) == 2

    def test_workflow_decision_linked_to_correlation_id(self, collector):
        collector.record_workflow_decision("wf1", "corr-A", passed=True,
                                           final_status="approved", total_execution_time=2.0)
        collector.record_workflow_decision("wf2", "corr-B", passed=False,
                                           final_status="failed", total_execution_time=1.0)

        linked = collector.get_decisions_by_correlation_id("corr-A")
        assert len(linked["workflow_decisions"]) == 1
        assert linked["workflow_decisions"][0]["final_status"] == "approved"

    def test_unknown_correlation_id_returns_empty(self, collector):
        linked = collector.get_decisions_by_correlation_id("nonexistent")
        assert linked["checker_decisions"] == []
        assert linked["workflow_decisions"] == []

    def test_multiple_workflows_same_correlation_id(self, collector):
        # Edge case: same correlation ID reused across two workflow runs
        for wf_id in ["wf1", "wf2"]:
            collector.record_checker_decision(wf_id, "shared-corr", "price", passed=True)

        linked = collector.get_decisions_by_correlation_id("shared-corr")
        assert len(linked["checker_decisions"]) == 2

    def test_checker_decision_contains_workflow_id(self, collector):
        collector.record_checker_decision("wf-XYZ", "corr-1", "relevance", passed=False)
        linked = collector.get_decisions_by_correlation_id("corr-1")
        assert linked["checker_decisions"][0]["workflow_id"] == "wf-XYZ"


# ---------------------------------------------------------------------------
# 4. Metrics export format
# ---------------------------------------------------------------------------

class TestMetricsExportFormat:
    """Tests for export_metrics() and to_workflow_metrics() output format."""

    def _populate(self, collector):
        """Helper: add some representative data."""
        for passed in [True, False, True]:
            collector.record_checker_decision("wf1", "c1", "price", passed=passed, execution_time=0.5)
        for passed in [True, True]:
            collector.record_checker_decision("wf1", "c1", "policy", passed=passed, execution_time=0.3)
        for passed in [False]:
            collector.record_checker_decision("wf1", "c1", "relevance", passed=passed, execution_time=0.4)
        collector.record_node_execution("research", 1.0)
        collector.record_node_execution("verification", 1.2)
        collector.record_workflow_decision("wf1", "c1", passed=False, final_status="escalated",
                                           total_execution_time=3.0, retry_count=1, escalated=True)

    def test_export_contains_required_keys(self, collector):
        self._populate(collector)
        data = collector.export_metrics()
        required = [
            "timestamp", "total_workflows", "workflow_pass_count", "workflow_fail_count",
            "workflow_pass_rate", "total_retries", "total_escalations", "escalation_rate",
            "checker_stats", "node_stats", "total_execution_time", "total_steps",
            "nodes_executed", "verification_pass_rate",
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_export_checker_stats_structure(self, collector):
        self._populate(collector)
        data = collector.export_metrics()
        for checker in ("price", "policy", "relevance"):
            cs = data["checker_stats"][checker]
            assert "total_runs" in cs
            assert "pass_count" in cs
            assert "fail_count" in cs
            assert "pass_rate" in cs
            assert "average_execution_time" in cs

    def test_export_pass_rate_values_correct(self, collector):
        self._populate(collector)
        data = collector.export_metrics()
        price_stats = data["checker_stats"]["price"]
        assert price_stats["total_runs"] == 3
        assert price_stats["pass_count"] == 2
        assert abs(price_stats["pass_rate"] - 2 / 3) < 1e-9

    def test_export_nodes_executed_list(self, collector):
        self._populate(collector)
        data = collector.export_metrics()
        assert "research" in data["nodes_executed"]
        assert "verification" in data["nodes_executed"]

    def test_to_workflow_metrics_returns_pydantic_model(self, collector):
        self._populate(collector)
        metrics = collector.to_workflow_metrics()
        assert isinstance(metrics, WorkflowMetrics)

    def test_to_workflow_metrics_fields_populated(self, collector):
        self._populate(collector)
        metrics = collector.to_workflow_metrics()
        assert metrics.total_retries == 1
        assert "research" in metrics.nodes_executed
        assert metrics.escalation_rate == 1.0  # 1 escalation / 1 workflow

    def test_export_empty_collector(self, collector):
        data = collector.export_metrics()
        assert data["total_workflows"] == 0
        assert data["workflow_pass_rate"] == 0.0
        for checker in ("price", "policy", "relevance"):
            assert data["checker_stats"][checker]["total_runs"] == 0

    def test_reset_clears_all_data(self, collector):
        self._populate(collector)
        collector.reset()
        data = collector.export_metrics()
        assert data["total_workflows"] == 0
        assert data["checker_stats"]["price"]["total_runs"] == 0

    def test_thread_safety(self, collector):
        """Concurrent writes should not corrupt counters."""
        import threading

        def add_decisions():
            for _ in range(50):
                collector.record_checker_decision("wf", "c", "price", passed=True)

        threads = [threading.Thread(target=add_decisions) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = collector.get_checker_stats("price")
        assert stats["total_runs"] == 200
        assert stats["pass_count"] == 200
