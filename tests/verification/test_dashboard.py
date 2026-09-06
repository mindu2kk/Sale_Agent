"""
Tests for DashboardDataCollector and related dashboard models.

Covers:
- WorkflowStatusSummary creation from workflow state
- DashboardSnapshot aggregation across multiple workflows
- Issue classification breakdown accuracy
- JSON export format

Requirements: 7.1, 7.2, 7.3, 7.5
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

from backend.verification.models.execution import WorkflowMetrics
from backend.verification.models.verification import (
    IssueSeverity,
    PolicyIssue,
    PriceIssue,
    RelevanceIssue,
    RubricCriteria,
    VerificationResult,
)
from backend.verification.utils.dashboard import (
    DashboardDataCollector,
    DashboardSnapshot,
    IssueBreakdown,
    IssueSeverityCounts,
    IssueTypeCounts,
    WorkflowStatusSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workflow_state(
    workflow_id: str = "wf_test_001",
    status: str = "approved",
    retry_count: int = 0,
    verification_result: Any = None,
    start_offset_seconds: float = 0.0,
    duration_seconds: float = 8.0,
) -> Dict[str, Any]:
    """Build a minimal WorkflowState dict for testing."""
    start = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc) + timedelta(
        seconds=start_offset_seconds
    )
    end = start + timedelta(seconds=duration_seconds)
    return {
        "workflow_id": workflow_id,
        "correlation_id": f"corr_{workflow_id}",
        "workflow_status": status,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "retry_count": retry_count,
        "verification_result": verification_result,
        "resource_usage": {
            "llm_tokens_total": 1000,
            "llm_cost_usd": 0.01,
        },
    }


def _make_verification_result(
    price_pass: bool = True,
    policy_pass: bool = True,
    relevance_pass: bool = True,
    price_issues=None,
    policy_issues=None,
    relevance_issues=None,
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
        verification_reasoning="Test verification",
        execution_time_seconds=2.5,
        llm_tokens_used=500,
    )


def _make_metrics(
    total_time: float = 8.0,
    llm_tokens: int = 3500,
    cost: float = 0.0175,
    verification_avg: float = 2.5,
) -> WorkflowMetrics:
    return WorkflowMetrics(
        total_execution_time=total_time,
        total_retries=0,
        total_steps=3,
        successful_steps=3,
        failed_steps=0,
        timeout_steps=0,
        nodes_executed=["research", "verification"],
        node_average_times={"verification": verification_avg},
        critical_issues_found=0,
        major_issues_found=0,
        minor_issues_found=0,
        total_issues_found=0,
        llm_tokens_used=llm_tokens,
        llm_tokens_input=2000,
        llm_tokens_output=1500,
        cost_estimate=cost,
        cache_hits=3,
        cache_misses=1,
        db_queries_count=5,
        external_api_calls=2,
        verification_pass_rate=1.0,
        escalation_rate=0.0,
    )


# ---------------------------------------------------------------------------
# WorkflowStatusSummary tests
# ---------------------------------------------------------------------------

class TestWorkflowStatusSummary:
    def test_basic_creation_from_approved_state(self):
        state = _make_workflow_state(status="approved")
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)

        assert summary.workflow_id == "wf_test_001"
        assert summary.status == "approved"
        assert summary.final_outcome == "approved"
        assert summary.retry_count == 0

    def test_duration_calculated_from_timestamps(self):
        state = _make_workflow_state(duration_seconds=10.0)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)

        assert summary.duration_seconds is not None
        assert abs(summary.duration_seconds - 10.0) < 0.01

    def test_final_outcome_escalated(self):
        state = _make_workflow_state(status="escalated")
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)
        assert summary.final_outcome == "escalated"

    def test_final_outcome_failed(self):
        state = _make_workflow_state(status="failed")
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)
        assert summary.final_outcome == "failed"

    def test_final_outcome_unknown_for_running(self):
        state = _make_workflow_state(status="running")
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)
        assert summary.final_outcome == "unknown"

    def test_retry_count_preserved(self):
        state = _make_workflow_state(retry_count=2)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)
        assert summary.retry_count == 2

    def test_performance_metrics_from_workflow_metrics(self):
        state = _make_workflow_state()
        metrics = _make_metrics(llm_tokens=4000, cost=0.02, verification_avg=3.0)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state, metrics)

        assert summary.total_llm_tokens == 4000
        assert abs(summary.cost_estimate_usd - 0.02) < 1e-6
        assert summary.avg_verification_time == 3.0

    def test_performance_metrics_fallback_to_resource_usage(self):
        state = _make_workflow_state()
        state["resource_usage"] = {"llm_tokens_total": 2500, "llm_cost_usd": 0.025}
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state, metrics=None)

        assert summary.total_llm_tokens == 2500
        assert abs(summary.cost_estimate_usd - 0.025) < 1e-6

    def test_no_end_time_gives_none_duration(self):
        state = _make_workflow_state()
        state["end_time"] = None
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)
        assert summary.duration_seconds is None

    def test_correlation_id_captured(self):
        state = _make_workflow_state(workflow_id="wf_abc")
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)
        assert summary.correlation_id == "corr_wf_abc"


# ---------------------------------------------------------------------------
# Issue classification breakdown tests
# ---------------------------------------------------------------------------

class TestIssueClassificationBreakdown:
    def test_no_verification_result_gives_zero_counts(self):
        state = _make_workflow_state(verification_result=None)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)

        assert summary.issue_breakdown.by_type.total == 0
        assert summary.issue_breakdown.by_severity.total == 0

    def test_price_issue_counted_correctly(self):
        price_issue = PriceIssue(
            product_name="iPhone 15",
            mentioned_price="35,000,000 VND",
            actual_price="29,990,000 VND",
            deviation_percent=16.7,
            severity=IssueSeverity.MAJOR,
            explanation="Price deviation detected",
        )
        vr = _make_verification_result(price_pass=False, price_issues=[price_issue])
        state = _make_workflow_state(verification_result=vr)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)

        assert summary.issue_breakdown.by_type.price == 1
        assert summary.issue_breakdown.by_type.policy == 0
        assert summary.issue_breakdown.by_type.relevance == 0
        assert summary.issue_breakdown.by_severity.major == 1
        assert summary.issue_breakdown.by_severity.critical == 0

    def test_policy_issue_critical_severity(self):
        policy_issue = PolicyIssue(
            mentioned_policy="Bảo hành 5 năm",
            policy_type="warranty",
            is_fabricated=True,
            is_inaccurate=True,
            severity=IssueSeverity.CRITICAL,
            explanation="Fabricated warranty policy",
        )
        vr = _make_verification_result(policy_pass=False, policy_issues=[policy_issue])
        state = _make_workflow_state(verification_result=vr)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)

        assert summary.issue_breakdown.by_type.policy == 1
        assert summary.issue_breakdown.by_severity.critical == 1

    def test_relevance_issue_minor_severity(self):
        relevance_issue = RelevanceIssue(
            objection_intent="Price comparison",
            response_coverage=0.85,
            severity=IssueSeverity.MINOR,
            explanation="Slightly off-topic",
        )
        vr = _make_verification_result(relevance_pass=False, relevance_issues=[relevance_issue])
        state = _make_workflow_state(verification_result=vr)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)

        assert summary.issue_breakdown.by_type.relevance == 1
        assert summary.issue_breakdown.by_severity.minor == 1

    def test_multiple_issues_across_types(self):
        price_issue = PriceIssue(
            product_name="Samsung S24",
            severity=IssueSeverity.MAJOR,
            explanation="Price wrong",
        )
        policy_issue = PolicyIssue(
            mentioned_policy="Return 60 days",
            policy_type="return",
            is_fabricated=False,
            is_inaccurate=True,
            severity=IssueSeverity.CRITICAL,
            explanation="Policy inaccurate",
        )
        relevance_issue = RelevanceIssue(
            objection_intent="Feature comparison",
            response_coverage=0.5,
            severity=IssueSeverity.MINOR,
            explanation="Missing aspects",
        )
        vr = _make_verification_result(
            price_pass=False,
            policy_pass=False,
            relevance_pass=False,
            price_issues=[price_issue],
            policy_issues=[policy_issue],
            relevance_issues=[relevance_issue],
        )
        state = _make_workflow_state(verification_result=vr)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)

        assert summary.issue_breakdown.by_type.price == 1
        assert summary.issue_breakdown.by_type.policy == 1
        assert summary.issue_breakdown.by_type.relevance == 1
        assert summary.issue_breakdown.by_type.total == 3
        assert summary.issue_breakdown.by_severity.critical == 1
        assert summary.issue_breakdown.by_severity.major == 1
        assert summary.issue_breakdown.by_severity.minor == 1
        assert summary.issue_breakdown.by_severity.total == 3

    def test_issue_breakdown_from_dict_verification_result(self):
        """Verify breakdown works when verification_result is a plain dict."""
        vr_dict = {
            "criteria": {
                "price_issues": [{"severity": "major"}],
                "policy_issues": [],
                "relevance_issues": [{"severity": "critical"}],
            }
        }
        state = _make_workflow_state(verification_result=vr_dict)
        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(state)

        assert summary.issue_breakdown.by_type.price == 1
        assert summary.issue_breakdown.by_type.relevance == 1
        assert summary.issue_breakdown.by_severity.major == 1
        assert summary.issue_breakdown.by_severity.critical == 1


# ---------------------------------------------------------------------------
# DashboardSnapshot aggregation tests
# ---------------------------------------------------------------------------

class TestDashboardSnapshot:
    def test_empty_summaries_returns_zero_snapshot(self):
        collector = DashboardDataCollector()
        snapshot = collector.build_dashboard_snapshot([])

        assert snapshot.total_workflows == 0
        assert snapshot.approval_rate == 0.0
        assert snapshot.escalation_rate == 0.0

    def test_approval_rate_calculation(self):
        collector = DashboardDataCollector()
        summaries = [
            collector.collect_workflow_summary(_make_workflow_state(f"wf_{i}", status="approved"))
            for i in range(7)
        ] + [
            collector.collect_workflow_summary(_make_workflow_state(f"wf_e{i}", status="escalated"))
            for i in range(2)
        ] + [
            collector.collect_workflow_summary(_make_workflow_state("wf_f0", status="failed"))
        ]

        snapshot = collector.build_dashboard_snapshot(summaries)

        assert snapshot.total_workflows == 10
        assert snapshot.approved_count == 7
        assert snapshot.escalated_count == 2
        assert snapshot.failed_count == 1
        assert abs(snapshot.approval_rate - 0.7) < 1e-6
        assert abs(snapshot.escalation_rate - 0.2) < 1e-6

    def test_issue_breakdown_aggregated_across_workflows(self):
        collector = DashboardDataCollector()

        price_issue = PriceIssue(
            product_name="iPhone",
            severity=IssueSeverity.MAJOR,
            explanation="Price wrong",
        )
        vr1 = _make_verification_result(price_pass=False, price_issues=[price_issue])
        vr2 = _make_verification_result(price_pass=False, price_issues=[price_issue])

        s1 = collector.collect_workflow_summary(_make_workflow_state("wf_1", verification_result=vr1))
        s2 = collector.collect_workflow_summary(_make_workflow_state("wf_2", verification_result=vr2))
        s3 = collector.collect_workflow_summary(_make_workflow_state("wf_3"))  # no issues

        snapshot = collector.build_dashboard_snapshot([s1, s2, s3])

        assert snapshot.issue_classification_breakdown.by_type.price == 2
        assert snapshot.issue_classification_breakdown.by_severity.major == 2
        assert snapshot.issue_classification_breakdown.by_type.policy == 0

    def test_avg_retries_calculation(self):
        collector = DashboardDataCollector()
        summaries = [
            collector.collect_workflow_summary(_make_workflow_state("wf_1", retry_count=0)),
            collector.collect_workflow_summary(_make_workflow_state("wf_2", retry_count=2)),
            collector.collect_workflow_summary(_make_workflow_state("wf_3", retry_count=1)),
        ]
        snapshot = collector.build_dashboard_snapshot(summaries)
        assert abs(snapshot.avg_retries - 1.0) < 1e-6

    def test_avg_duration_calculation(self):
        collector = DashboardDataCollector()
        summaries = [
            collector.collect_workflow_summary(_make_workflow_state("wf_1", duration_seconds=6.0)),
            collector.collect_workflow_summary(_make_workflow_state("wf_2", duration_seconds=10.0)),
            collector.collect_workflow_summary(_make_workflow_state("wf_3", duration_seconds=8.0)),
        ]
        snapshot = collector.build_dashboard_snapshot(summaries)
        assert abs(snapshot.avg_duration_seconds - 8.0) < 0.01

    def test_top_failure_reasons_sorted_by_frequency(self):
        collector = DashboardDataCollector()

        price_issue = PriceIssue(
            product_name="iPhone",
            severity=IssueSeverity.MAJOR,
            explanation="Price wrong",
        )
        policy_issue = PolicyIssue(
            mentioned_policy="Warranty",
            policy_type="warranty",
            is_fabricated=True,
            is_inaccurate=True,
            severity=IssueSeverity.CRITICAL,
            explanation="Fabricated",
        )

        # 3 price issues, 1 policy issue
        vr_price = _make_verification_result(price_pass=False, price_issues=[price_issue])
        vr_policy = _make_verification_result(policy_pass=False, policy_issues=[policy_issue])

        summaries = [
            collector.collect_workflow_summary(_make_workflow_state(f"wf_p{i}", verification_result=vr_price))
            for i in range(3)
        ] + [
            collector.collect_workflow_summary(_make_workflow_state("wf_pol", verification_result=vr_policy))
        ]

        snapshot = collector.build_dashboard_snapshot(summaries)

        # price_issues should appear first (3 occurrences > 1)
        assert len(snapshot.top_failure_reasons) > 0
        assert "price_issues" in snapshot.top_failure_reasons[0]

    def test_snapshot_has_timestamp(self):
        collector = DashboardDataCollector()
        snapshot = collector.build_dashboard_snapshot([])
        assert snapshot.snapshot_timestamp != ""
        # Should be parseable as ISO datetime
        datetime.fromisoformat(snapshot.snapshot_timestamp.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------

class TestExportToDict:
    def test_export_returns_dict_with_required_keys(self):
        collector = DashboardDataCollector()
        collector.collect_workflow_summary(_make_workflow_state("wf_1"))
        data = collector.export_to_dict()

        assert "export_timestamp" in data
        assert "summaries" in data
        assert "snapshot" in data

    def test_export_summaries_list_length(self):
        collector = DashboardDataCollector()
        for i in range(3):
            collector.collect_workflow_summary(_make_workflow_state(f"wf_{i}"))
        data = collector.export_to_dict()
        assert len(data["summaries"]) == 3

    def test_export_snapshot_contains_total_workflows(self):
        collector = DashboardDataCollector()
        for i in range(5):
            collector.collect_workflow_summary(_make_workflow_state(f"wf_{i}"))
        data = collector.export_to_dict()
        assert data["snapshot"]["total_workflows"] == 5

    def test_export_is_json_serializable(self):
        import json

        collector = DashboardDataCollector()
        price_issue = PriceIssue(
            product_name="Samsung",
            severity=IssueSeverity.MINOR,
            explanation="Minor price diff",
        )
        vr = _make_verification_result(price_pass=False, price_issues=[price_issue])
        collector.collect_workflow_summary(_make_workflow_state("wf_1", verification_result=vr))

        data = collector.export_to_dict()
        # Should not raise
        serialized = json.dumps(data, default=str)
        parsed = json.loads(serialized)
        assert parsed["snapshot"]["total_workflows"] == 1

    def test_export_timestamp_is_iso_format(self):
        collector = DashboardDataCollector()
        data = collector.export_to_dict()
        ts = data["export_timestamp"]
        # Should parse without error
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_export_summary_contains_issue_breakdown(self):
        collector = DashboardDataCollector()
        price_issue = PriceIssue(
            product_name="iPhone",
            severity=IssueSeverity.CRITICAL,
            explanation="Critical price error",
        )
        vr = _make_verification_result(price_pass=False, price_issues=[price_issue])
        collector.collect_workflow_summary(_make_workflow_state("wf_1", verification_result=vr))

        data = collector.export_to_dict()
        summary_data = data["summaries"][0]
        assert "issue_breakdown" in summary_data
        assert summary_data["issue_breakdown"]["by_type"]["price"] == 1
        assert summary_data["issue_breakdown"]["by_severity"]["critical"] == 1

    def test_reset_clears_summaries(self):
        collector = DashboardDataCollector()
        collector.collect_workflow_summary(_make_workflow_state("wf_1"))
        collector.reset()
        data = collector.export_to_dict()
        assert len(data["summaries"]) == 0
        assert data["snapshot"]["total_workflows"] == 0
