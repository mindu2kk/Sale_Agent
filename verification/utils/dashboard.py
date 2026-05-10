"""
Workflow Status Dashboard Data Collector

Aggregates workflow execution data into structured dashboard-ready format:
- WorkflowStatusSummary: Per-workflow summary with issue classification breakdown
- DashboardSnapshot: Aggregated stats across multiple workflows
- DashboardDataCollector: Collects and exports dashboard data

Requirements: 7.1 (execution history), 7.2 (real-time status),
              7.3 (structured JSON for analytics), 7.5 (exportable observability data)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.execution import WorkflowMetrics
from ..models.verification import IssueSeverity


# ---------------------------------------------------------------------------
# Issue classification breakdown models
# ---------------------------------------------------------------------------

class IssueTypeCounts(BaseModel):
    """Counts of issues broken down by type (price / policy / relevance)."""
    price: int = Field(default=0, ge=0, description="Number of price accuracy issues")
    policy: int = Field(default=0, ge=0, description="Number of policy authenticity issues")
    relevance: int = Field(default=0, ge=0, description="Number of topic relevance issues")

    @property
    def total(self) -> int:
        return self.price + self.policy + self.relevance


class IssueSeverityCounts(BaseModel):
    """Counts of issues broken down by severity level."""
    critical: int = Field(default=0, ge=0, description="Number of critical issues")
    major: int = Field(default=0, ge=0, description="Number of major issues")
    minor: int = Field(default=0, ge=0, description="Number of minor issues")

    @property
    def total(self) -> int:
        return self.critical + self.major + self.minor


class IssueBreakdown(BaseModel):
    """Full issue classification breakdown by both type and severity."""
    by_type: IssueTypeCounts = Field(default_factory=IssueTypeCounts)
    by_severity: IssueSeverityCounts = Field(default_factory=IssueSeverityCounts)

    @property
    def total_issues(self) -> int:
        return self.by_severity.total


# ---------------------------------------------------------------------------
# Per-workflow summary
# ---------------------------------------------------------------------------

class WorkflowStatusSummary(BaseModel):
    """
    Dashboard-ready summary for a single workflow execution.

    **Validates: Requirements 7.1** - Complete execution history tracking
    **Validates: Requirements 7.2** - Real-time workflow status
    **Validates: Requirements 7.3** - Structured JSON format for analytics
    """

    # Identity
    workflow_id: str = Field(description="Unique workflow identifier")
    correlation_id: str = Field(default="", description="Distributed tracing correlation ID")

    # Status
    status: str = Field(description="Final workflow status (approved/escalated/failed/running)")
    final_outcome: str = Field(
        default="unknown",
        description="Final outcome: 'approved', 'escalated', 'failed', or 'unknown'",
    )

    # Timestamps
    start_time: str = Field(description="Workflow start timestamp (ISO format)")
    end_time: Optional[str] = Field(default=None, description="Workflow end timestamp (ISO format)")
    duration_seconds: Optional[float] = Field(
        default=None, ge=0.0, description="Total execution duration in seconds"
    )

    # Retry tracking
    retry_count: int = Field(default=0, ge=0, description="Number of correction retries performed")

    # Issue classification breakdown
    issue_breakdown: IssueBreakdown = Field(
        default_factory=IssueBreakdown,
        description="Issue counts by type and severity",
    )

    # Performance metrics
    avg_verification_time: Optional[float] = Field(
        default=None, ge=0.0, description="Average verification step time (seconds)"
    )
    total_llm_tokens: int = Field(default=0, ge=0, description="Total LLM tokens consumed")
    cost_estimate_usd: float = Field(default=0.0, ge=0.0, description="Estimated LLM cost (USD)")

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "wf_20240115_103000_abc123",
                "correlation_id": "corr_xyz789",
                "status": "approved",
                "final_outcome": "approved",
                "start_time": "2024-01-15T10:30:00.000Z",
                "end_time": "2024-01-15T10:30:08.500Z",
                "duration_seconds": 8.5,
                "retry_count": 1,
                "issue_breakdown": {
                    "by_type": {"price": 1, "policy": 0, "relevance": 0},
                    "by_severity": {"critical": 0, "major": 1, "minor": 0},
                },
                "avg_verification_time": 2.5,
                "total_llm_tokens": 3500,
                "cost_estimate_usd": 0.0175,
            }
        }


# ---------------------------------------------------------------------------
# Aggregated dashboard snapshot
# ---------------------------------------------------------------------------

class DashboardSnapshot(BaseModel):
    """
    Aggregated dashboard statistics across multiple workflow executions.

    **Validates: Requirements 7.2** - Real-time workflow status
    **Validates: Requirements 7.3** - Structured JSON format for analytics
    **Validates: Requirements 7.5** - Exportable observability data
    """

    # Snapshot metadata
    snapshot_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When this snapshot was generated (ISO format)",
    )

    # Workflow counts
    total_workflows: int = Field(default=0, ge=0, description="Total workflows included")
    approved_count: int = Field(default=0, ge=0, description="Workflows that were approved")
    escalated_count: int = Field(default=0, ge=0, description="Workflows that were escalated")
    failed_count: int = Field(default=0, ge=0, description="Workflows that failed")

    # Rates
    approval_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction approved (0-1)")
    escalation_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction escalated (0-1)")

    # Aggregated issue classification breakdown
    issue_classification_breakdown: IssueBreakdown = Field(
        default_factory=IssueBreakdown,
        description="Aggregated issue counts across all workflows",
    )

    # Top failure reasons (most common issue types/severities)
    top_failure_reasons: List[str] = Field(
        default_factory=list,
        description="Most common failure reasons ordered by frequency",
    )

    # Retry & duration stats
    avg_retries: float = Field(default=0.0, ge=0.0, description="Average retry count per workflow")
    avg_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Average workflow duration in seconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "snapshot_timestamp": "2024-01-15T11:00:00.000Z",
                "total_workflows": 50,
                "approved_count": 38,
                "escalated_count": 8,
                "failed_count": 4,
                "approval_rate": 0.76,
                "escalation_rate": 0.16,
                "issue_classification_breakdown": {
                    "by_type": {"price": 12, "policy": 5, "relevance": 8},
                    "by_severity": {"critical": 6, "major": 14, "minor": 5},
                },
                "top_failure_reasons": [
                    "price_issues (12 occurrences)",
                    "relevance_issues (8 occurrences)",
                    "policy_issues (5 occurrences)",
                ],
                "avg_retries": 1.2,
                "avg_duration_seconds": 9.3,
            }
        }


# ---------------------------------------------------------------------------
# DashboardDataCollector
# ---------------------------------------------------------------------------

class DashboardDataCollector:
    """
    Collects and aggregates workflow execution data into dashboard-ready format.

    Usage::

        collector = DashboardDataCollector()
        summary = collector.collect_workflow_summary(workflow_state, metrics)
        snapshot = collector.build_dashboard_snapshot([summary1, summary2, ...])
        data = collector.export_to_dict()

    **Validates: Requirements 7.1, 7.2, 7.3, 7.5**
    """

    def __init__(self) -> None:
        self._summaries: List[WorkflowStatusSummary] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_workflow_summary(
        self,
        workflow_state: Dict[str, Any],
        metrics: Optional[WorkflowMetrics] = None,
    ) -> WorkflowStatusSummary:
        """
        Build a WorkflowStatusSummary from a WorkflowState dict and optional metrics.

        Args:
            workflow_state: WorkflowState TypedDict (or plain dict with same keys)
            metrics: Optional WorkflowMetrics Pydantic model for performance data

        Returns:
            WorkflowStatusSummary ready for dashboard display
        """
        workflow_id = workflow_state.get("workflow_id", "unknown")
        correlation_id = workflow_state.get("correlation_id", "")
        status = workflow_state.get("workflow_status", "unknown")
        start_time = workflow_state.get("start_time", datetime.now(timezone.utc).isoformat())
        end_time = workflow_state.get("end_time")
        retry_count = workflow_state.get("retry_count", 0)

        # Determine final outcome
        final_outcome = self._derive_final_outcome(status)

        # Calculate duration
        duration_seconds: Optional[float] = None
        if end_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                duration_seconds = (end_dt - start_dt).total_seconds()
            except (ValueError, AttributeError):
                pass

        # Build issue breakdown from verification_result
        issue_breakdown = self._extract_issue_breakdown(workflow_state)

        # Performance metrics from WorkflowMetrics or resource_usage
        avg_verification_time: Optional[float] = None
        total_llm_tokens = 0
        cost_estimate_usd = 0.0

        if metrics is not None:
            avg_verification_time = metrics.node_average_times.get(
                "verification", metrics.average_step_time
            )
            total_llm_tokens = metrics.llm_tokens_used
            cost_estimate_usd = metrics.cost_estimate
        else:
            resource_usage = workflow_state.get("resource_usage", {})
            total_llm_tokens = resource_usage.get("llm_tokens_total", 0)
            cost_estimate_usd = resource_usage.get("llm_cost_usd", 0.0)

        summary = WorkflowStatusSummary(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            status=status,
            final_outcome=final_outcome,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            retry_count=retry_count,
            issue_breakdown=issue_breakdown,
            avg_verification_time=avg_verification_time,
            total_llm_tokens=total_llm_tokens,
            cost_estimate_usd=cost_estimate_usd,
        )

        self._summaries.append(summary)
        return summary

    def build_dashboard_snapshot(
        self, summaries: List[WorkflowStatusSummary]
    ) -> DashboardSnapshot:
        """
        Aggregate a list of WorkflowStatusSummary objects into a DashboardSnapshot.

        Args:
            summaries: List of per-workflow summaries to aggregate

        Returns:
            DashboardSnapshot with aggregated statistics
        """
        total = len(summaries)
        if total == 0:
            return DashboardSnapshot()

        approved = sum(1 for s in summaries if s.final_outcome == "approved")
        escalated = sum(1 for s in summaries if s.final_outcome == "escalated")
        failed = sum(1 for s in summaries if s.final_outcome == "failed")

        approval_rate = approved / total
        escalation_rate = escalated / total

        # Aggregate issue breakdown
        agg_by_type = IssueTypeCounts(
            price=sum(s.issue_breakdown.by_type.price for s in summaries),
            policy=sum(s.issue_breakdown.by_type.policy for s in summaries),
            relevance=sum(s.issue_breakdown.by_type.relevance for s in summaries),
        )
        agg_by_severity = IssueSeverityCounts(
            critical=sum(s.issue_breakdown.by_severity.critical for s in summaries),
            major=sum(s.issue_breakdown.by_severity.major for s in summaries),
            minor=sum(s.issue_breakdown.by_severity.minor for s in summaries),
        )
        issue_breakdown = IssueBreakdown(by_type=agg_by_type, by_severity=agg_by_severity)

        # Top failure reasons (by issue type frequency)
        top_failure_reasons = self._compute_top_failure_reasons(agg_by_type, agg_by_severity)

        # Avg retries and duration
        avg_retries = sum(s.retry_count for s in summaries) / total
        durations = [s.duration_seconds for s in summaries if s.duration_seconds is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return DashboardSnapshot(
            total_workflows=total,
            approved_count=approved,
            escalated_count=escalated,
            failed_count=failed,
            approval_rate=approval_rate,
            escalation_rate=escalation_rate,
            issue_classification_breakdown=issue_breakdown,
            top_failure_reasons=top_failure_reasons,
            avg_retries=avg_retries,
            avg_duration_seconds=avg_duration,
        )

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all collected summaries and a snapshot as a JSON-serializable dict.

        **Validates: Requirements 7.3, 7.5** - Structured JSON export

        Returns:
            Dict with 'summaries' list and 'snapshot' aggregation
        """
        snapshot = self.build_dashboard_snapshot(self._summaries)
        return {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "summaries": [s.model_dump() for s in self._summaries],
            "snapshot": snapshot.model_dump(),
        }

    def get_summaries(self) -> List[WorkflowStatusSummary]:
        """Return all collected summaries (read-only copy)."""
        return list(self._summaries)

    def reset(self) -> None:
        """Clear all collected summaries."""
        self._summaries.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_final_outcome(status: str) -> str:
        """Map workflow_status to a canonical final_outcome string."""
        mapping = {
            "approved": "approved",
            "escalated": "escalated",
            "failed": "failed",
        }
        return mapping.get(status, "unknown")

    @staticmethod
    def _extract_issue_breakdown(workflow_state: Dict[str, Any]) -> IssueBreakdown:
        """
        Extract issue counts from the verification_result inside workflow_state.

        Handles both Pydantic VerificationResult objects and plain dicts.
        """
        verification_result = workflow_state.get("verification_result")
        if verification_result is None:
            return IssueBreakdown()

        # Support both Pydantic model and plain dict
        if hasattr(verification_result, "criteria"):
            criteria = verification_result.criteria
            price_issues = getattr(criteria, "price_issues", [])
            policy_issues = getattr(criteria, "policy_issues", [])
            relevance_issues = getattr(criteria, "relevance_issues", [])
        elif isinstance(verification_result, dict):
            criteria = verification_result.get("criteria", {})
            if isinstance(criteria, dict):
                price_issues = criteria.get("price_issues", [])
                policy_issues = criteria.get("policy_issues", [])
                relevance_issues = criteria.get("relevance_issues", [])
            else:
                return IssueBreakdown()
        else:
            return IssueBreakdown()

        # Count by type
        by_type = IssueTypeCounts(
            price=len(price_issues),
            policy=len(policy_issues),
            relevance=len(relevance_issues),
        )

        # Count by severity across all issue types
        all_issues = list(price_issues) + list(policy_issues) + list(relevance_issues)
        critical = major = minor = 0
        for issue in all_issues:
            severity = (
                issue.severity if hasattr(issue, "severity")
                else issue.get("severity", "") if isinstance(issue, dict)
                else ""
            )
            # Normalise to string value
            if hasattr(severity, "value"):
                severity = severity.value
            if severity == IssueSeverity.CRITICAL or severity == "critical":
                critical += 1
            elif severity == IssueSeverity.MAJOR or severity == "major":
                major += 1
            elif severity == IssueSeverity.MINOR or severity == "minor":
                minor += 1

        by_severity = IssueSeverityCounts(critical=critical, major=major, minor=minor)
        return IssueBreakdown(by_type=by_type, by_severity=by_severity)

    @staticmethod
    def _compute_top_failure_reasons(
        by_type: IssueTypeCounts,
        by_severity: IssueSeverityCounts,
    ) -> List[str]:
        """
        Build a human-readable list of top failure reasons sorted by frequency.
        Includes both type-based and severity-based reasons.
        """
        type_reasons = [
            ("price_issues", by_type.price),
            ("policy_issues", by_type.policy),
            ("relevance_issues", by_type.relevance),
        ]
        severity_reasons = [
            ("critical_severity_issues", by_severity.critical),
            ("major_severity_issues", by_severity.major),
            ("minor_severity_issues", by_severity.minor),
        ]

        # Combine, filter zeros, sort descending
        all_reasons = type_reasons + severity_reasons
        filtered = [(label, count) for label, count in all_reasons if count > 0]
        filtered.sort(key=lambda x: x[1], reverse=True)

        return [f"{label} ({count} occurrences)" for label, count in filtered]
