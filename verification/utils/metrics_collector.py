"""
Execution Metrics Collector với Binary Decision Tracking

Tracks binary PASS/FAIL outcomes per checker throughout the workflow:
- Binary decision outcomes per checker (price_pass, policy_pass, relevance_pass)
- Overall workflow PASS/FAIL decisions with timestamps
- Aggregated metrics: total_runs, pass_count, fail_count, pass_rate per checker
- Retry counts and escalation counts
- Execution times per node and per checker
- Correlation ID linking for distributed tracing

Requirements: 7.1 (execution history tracking), 7.2 (real-time workflow status),
              7.3 (structured JSON format for analytics)
"""

import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..models.execution import WorkflowMetrics


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckerDecision:
    """A single binary decision record for one checker in one workflow run."""
    workflow_id: str
    correlation_id: str
    checker: str          # "price", "policy", "relevance"
    passed: bool
    execution_time: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class WorkflowDecision:
    """Overall PASS/FAIL decision for a complete workflow run."""
    workflow_id: str
    correlation_id: str
    passed: bool          # True = approved, False = escalated/failed
    final_status: str     # "approved", "escalated", "failed"
    total_execution_time: float
    retry_count: int
    escalated: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CheckerStats:
    """Aggregated statistics for a single checker."""
    total_runs: int = 0
    pass_count: int = 0
    fail_count: int = 0
    total_execution_time: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.pass_count / self.total_runs

    @property
    def average_execution_time(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.total_execution_time / self.total_runs


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects and aggregates execution metrics with binary decision tracking.

    Thread-safe collector that records:
    - Per-checker binary decisions (price / policy / relevance)
    - Overall workflow outcomes (approved / escalated / failed)
    - Node execution times
    - Retry and escalation counts
    - Correlation ID linking

    **Validates: Requirements 7.1** - execution history tracking
    **Validates: Requirements 7.2** - real-time workflow status
    **Validates: Requirements 7.3** - structured JSON format for analytics
    """

    CHECKERS = ("price", "policy", "relevance")

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Per-checker aggregated stats
        self._checker_stats: Dict[str, CheckerStats] = {
            c: CheckerStats() for c in self.CHECKERS
        }

        # Raw decision history (for export / audit)
        self._checker_decisions: List[CheckerDecision] = []
        self._workflow_decisions: List[WorkflowDecision] = []

        # Node execution times: {node_name: [time1, time2, ...]}
        self._node_times: Dict[str, List[float]] = {}

        # Workflow-level counters
        self._total_retries: int = 0
        self._total_escalations: int = 0
        self._total_workflows: int = 0
        self._pass_workflows: int = 0

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_checker_decision(
        self,
        workflow_id: str,
        correlation_id: str,
        checker: str,
        passed: bool,
        execution_time: float = 0.0,
    ) -> None:
        """
        Record a binary PASS/FAIL decision for a single checker.

        Args:
            workflow_id: Unique workflow identifier
            correlation_id: Distributed tracing correlation ID
            checker: One of "price", "policy", "relevance"
            passed: True = PASS, False = FAIL
            execution_time: Time taken by this checker (seconds)
        """
        checker = checker.lower()
        if checker not in self.CHECKERS:
            raise ValueError(f"Unknown checker '{checker}'. Must be one of {self.CHECKERS}")

        decision = CheckerDecision(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            checker=checker,
            passed=passed,
            execution_time=execution_time,
        )

        with self._lock:
            stats = self._checker_stats[checker]
            stats.total_runs += 1
            if passed:
                stats.pass_count += 1
            else:
                stats.fail_count += 1
            stats.total_execution_time += execution_time

            self._checker_decisions.append(decision)

    def record_workflow_decision(
        self,
        workflow_id: str,
        correlation_id: str,
        passed: bool,
        final_status: str,
        total_execution_time: float,
        retry_count: int = 0,
        escalated: bool = False,
    ) -> None:
        """
        Record the overall PASS/FAIL outcome for a completed workflow.

        Args:
            workflow_id: Unique workflow identifier
            correlation_id: Distributed tracing correlation ID
            passed: True = approved, False = failed/escalated
            final_status: "approved", "escalated", or "failed"
            total_execution_time: Total workflow execution time (seconds)
            retry_count: Number of correction retries performed
            escalated: Whether the workflow was escalated to human review
        """
        decision = WorkflowDecision(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            passed=passed,
            final_status=final_status,
            total_execution_time=total_execution_time,
            retry_count=retry_count,
            escalated=escalated,
        )

        with self._lock:
            self._workflow_decisions.append(decision)
            self._total_workflows += 1
            if passed:
                self._pass_workflows += 1
            self._total_retries += retry_count
            if escalated:
                self._total_escalations += 1

    def record_node_execution(
        self,
        node_name: str,
        execution_time: float,
    ) -> None:
        """
        Record execution time for a workflow node.

        Args:
            node_name: Name of the node (e.g. "research", "verification", "correction")
            execution_time: Execution time in seconds
        """
        with self._lock:
            if node_name not in self._node_times:
                self._node_times[node_name] = []
            self._node_times[node_name].append(execution_time)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_checker_stats(self, checker: str) -> Dict[str, Any]:
        """
        Get aggregated statistics for a specific checker.

        Returns dict with: total_runs, pass_count, fail_count, pass_rate,
        average_execution_time.
        """
        checker = checker.lower()
        if checker not in self.CHECKERS:
            raise ValueError(f"Unknown checker '{checker}'")

        with self._lock:
            stats = self._checker_stats[checker]
            return {
                "checker": checker,
                "total_runs": stats.total_runs,
                "pass_count": stats.pass_count,
                "fail_count": stats.fail_count,
                "pass_rate": stats.pass_rate,
                "average_execution_time": stats.average_execution_time,
            }

    def get_all_checker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return aggregated stats for all three checkers."""
        return {c: self.get_checker_stats(c) for c in self.CHECKERS}

    def get_workflow_summary(self) -> Dict[str, Any]:
        """
        Return high-level workflow-level summary metrics.

        **Validates: Requirements 7.2** - real-time workflow status
        """
        with self._lock:
            total = self._total_workflows
            passed = self._pass_workflows
            return {
                "total_workflows": total,
                "pass_count": passed,
                "fail_count": total - passed,
                "pass_rate": passed / total if total > 0 else 0.0,
                "total_retries": self._total_retries,
                "total_escalations": self._total_escalations,
                "escalation_rate": (
                    self._total_escalations / total if total > 0 else 0.0
                ),
            }

    def get_node_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return execution time statistics per node."""
        with self._lock:
            result = {}
            for node, times in self._node_times.items():
                if times:
                    result[node] = {
                        "total_executions": len(times),
                        "total_time": sum(times),
                        "average_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                    }
            return result

    def get_decisions_by_correlation_id(
        self, correlation_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve all decisions linked to a specific correlation ID.

        **Validates: Requirements 7.1** - execution history tracking
        """
        with self._lock:
            checker_decisions = [
                {
                    "workflow_id": d.workflow_id,
                    "checker": d.checker,
                    "passed": d.passed,
                    "execution_time": d.execution_time,
                    "timestamp": d.timestamp,
                }
                for d in self._checker_decisions
                if d.correlation_id == correlation_id
            ]
            workflow_decisions = [
                {
                    "workflow_id": d.workflow_id,
                    "passed": d.passed,
                    "final_status": d.final_status,
                    "total_execution_time": d.total_execution_time,
                    "retry_count": d.retry_count,
                    "escalated": d.escalated,
                    "timestamp": d.timestamp,
                }
                for d in self._workflow_decisions
                if d.correlation_id == correlation_id
            ]

        return {
            "correlation_id": correlation_id,
            "checker_decisions": checker_decisions,
            "workflow_decisions": workflow_decisions,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all collected metrics as a structured dictionary.

        Compatible with WorkflowMetrics Pydantic model fields.

        **Validates: Requirements 7.3** - structured JSON format for analytics
        """
        with self._lock:
            checker_stats = {
                c: {
                    "total_runs": self._checker_stats[c].total_runs,
                    "pass_count": self._checker_stats[c].pass_count,
                    "fail_count": self._checker_stats[c].fail_count,
                    "pass_rate": self._checker_stats[c].pass_rate,
                    "average_execution_time": self._checker_stats[c].average_execution_time,
                }
                for c in self.CHECKERS
            }

            total = self._total_workflows
            passed = self._pass_workflows

            node_stats = {}
            all_node_times: List[float] = []
            for node, times in self._node_times.items():
                if times:
                    node_stats[node] = {
                        "total_executions": len(times),
                        "average_time": sum(times) / len(times),
                    }
                    all_node_times.extend(times)

            total_execution_time = sum(all_node_times)
            total_steps = sum(
                s["total_executions"] for s in node_stats.values()
            )

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # Workflow-level
                "total_workflows": total,
                "workflow_pass_count": passed,
                "workflow_fail_count": total - passed,
                "workflow_pass_rate": passed / total if total > 0 else 0.0,
                "total_retries": self._total_retries,
                "total_escalations": self._total_escalations,
                "escalation_rate": (
                    self._total_escalations / total if total > 0 else 0.0
                ),
                # Per-checker binary decision stats
                "checker_stats": checker_stats,
                # Node execution stats
                "node_stats": node_stats,
                # WorkflowMetrics-compatible fields
                "total_execution_time": total_execution_time,
                "total_steps": total_steps,
                "successful_steps": total_steps,  # conservative: all recorded steps succeeded
                "failed_steps": 0,
                "timeout_steps": 0,
                "nodes_executed": list(self._node_times.keys()),
                "verification_pass_rate": (
                    self._checker_stats["price"].pass_rate
                    * self._checker_stats["policy"].pass_rate
                    * self._checker_stats["relevance"].pass_rate
                    if total > 0 else 0.0
                ),
                "escalation_rate": (
                    self._total_escalations / total if total > 0 else 0.0
                ),
            }

    def to_workflow_metrics(self) -> WorkflowMetrics:
        """
        Build a WorkflowMetrics Pydantic model from collected data.

        Returns a WorkflowMetrics instance compatible with the existing
        execution models.
        """
        data = self.export_metrics()

        total_steps = data["total_steps"]
        total_time = data["total_execution_time"]

        node_execution_counts: Dict[str, int] = {}
        node_average_times: Dict[str, float] = {}
        with self._lock:
            for node, times in self._node_times.items():
                node_execution_counts[node] = len(times)
                node_average_times[node] = sum(times) / len(times) if times else 0.0

        return WorkflowMetrics(
            total_execution_time=total_time,
            min_step_time=0.0,
            max_step_time=0.0,
            total_retries=data["total_retries"],
            total_steps=total_steps,
            successful_steps=total_steps,
            failed_steps=0,
            timeout_steps=0,
            nodes_executed=data["nodes_executed"],
            node_execution_counts=node_execution_counts,
            node_average_times=node_average_times,
            critical_issues_found=0,
            major_issues_found=0,
            minor_issues_found=0,
            total_issues_found=0,
            llm_tokens_used=0,
            llm_tokens_input=0,
            llm_tokens_output=0,
            cost_estimate=0.0,
            cache_hits=0,
            cache_misses=0,
            db_queries_count=0,
            external_api_calls=0,
            verification_pass_rate=data["verification_pass_rate"],
            escalation_rate=data["escalation_rate"],
        )

    def reset(self) -> None:
        """Reset all collected metrics (useful for testing)."""
        with self._lock:
            self._checker_stats = {c: CheckerStats() for c in self.CHECKERS}
            self._checker_decisions.clear()
            self._workflow_decisions.clear()
            self._node_times.clear()
            self._total_retries = 0
            self._total_escalations = 0
            self._total_workflows = 0
            self._pass_workflows = 0
