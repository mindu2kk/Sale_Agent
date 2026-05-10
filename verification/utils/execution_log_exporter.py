"""
Execution Log Export Functionality với Pydantic Serialization

Exports execution logs to structured JSON/JSONL formats for analytics:
- JSON export using Pydantic's .model_dump() serialization (Pydantic v2)
- JSONL (JSON Lines) export for streaming/large datasets
- Filtering by date range, workflow_id, status, node_name
- Aggregation/summary export using WorkflowMetrics
- Proper datetime serialization
- Integration with existing logs/ directory structure

Requirements: 7.3 (structured JSON format), 7.5 (exportable observability data)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.execution import ExecutionStep, WorkflowExecutionLog, WorkflowMetrics


# ---------------------------------------------------------------------------
# Filter criteria
# ---------------------------------------------------------------------------

class ExportFilter:
    """
    Filter criteria for execution log export.

    All fields are optional; unset fields are not applied.
    """

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        node_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        self.workflow_id = workflow_id
        self.status = status
        self.node_name = node_name
        self.start_date = start_date
        self.end_date = end_date

    def matches_log(self, log: WorkflowExecutionLog) -> bool:
        """Return True if the log passes all active filter criteria."""
        if self.workflow_id and log.workflow_id != self.workflow_id:
            return False

        if self.status and log.final_status != self.status:
            return False

        if self.node_name:
            if self.node_name not in log.node_history:
                return False

        if self.start_date:
            log_start = _parse_datetime(log.start_time)
            if log_start is None or log_start < self.start_date:
                return False

        if self.end_date:
            log_end = log.end_time
            if log_end is None:
                return False
            log_end_dt = _parse_datetime(log_end)
            if log_end_dt is None or log_end_dt > self.end_date:
                return False

        return True

    def matches_step(self, step: ExecutionStep) -> bool:
        """Return True if the step passes node_name and date filters."""
        if self.node_name and step.node_name != self.node_name:
            return False

        if self.status and step.status.value != self.status:
            return False

        if self.start_date or self.end_date:
            step_ts = _parse_datetime(step.timestamp)
            if step_ts is not None:
                if self.start_date and step_ts < self.start_date:
                    return False
                if self.end_date and step_ts > self.end_date:
                    return False

        return True


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse a datetime value (str or datetime) to a timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _default_serializer(obj: Any) -> Any:
    """JSON default serializer for types not handled by stdlib json."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _serialize_step(step: ExecutionStep) -> Dict[str, Any]:
    """Serialize an ExecutionStep to a JSON-compatible dict via Pydantic v2."""
    return step.model_dump(mode="json")


def _serialize_log(log: WorkflowExecutionLog) -> Dict[str, Any]:
    """Serialize a WorkflowExecutionLog to a JSON-compatible dict via Pydantic v2."""
    return log.model_dump(mode="json")


def _serialize_metrics(metrics: WorkflowMetrics) -> Dict[str, Any]:
    """Serialize a WorkflowMetrics to a JSON-compatible dict via Pydantic v2."""
    return metrics.model_dump(mode="json")


# ---------------------------------------------------------------------------
# ExecutionLogExporter
# ---------------------------------------------------------------------------

class ExecutionLogExporter:
    """
    Exports execution logs to JSON / JSONL formats with filtering and aggregation.

    Supports:
    - In-memory log registration
    - Export to JSON file (single object with metadata + records)
    - Export to JSONL file (one JSON object per line)
    - Filtering by workflow_id, status, node_name, date range
    - Summary/aggregation export using WorkflowMetrics

    **Validates: Requirements 7.3** - Structured JSON format for analytics
    **Validates: Requirements 7.5** - Exportable observability data

    Usage::

        exporter = ExecutionLogExporter(logs_dir="logs")
        exporter.register_log(workflow_execution_log)
        exporter.export_json("logs/workflow/export.json")
        exporter.export_jsonl("logs/workflow/export.jsonl")
        summary = exporter.export_summary()
    """

    DEFAULT_LOGS_DIR = Path("logs")

    def __init__(self, logs_dir: Optional[str] = None) -> None:
        self._logs: List[WorkflowExecutionLog] = []
        self._logs_dir = Path(logs_dir) if logs_dir else self.DEFAULT_LOGS_DIR

    # ------------------------------------------------------------------
    # Log registration
    # ------------------------------------------------------------------

    def register_log(self, log: WorkflowExecutionLog) -> None:
        """Register a WorkflowExecutionLog for export."""
        self._logs.append(log)

    def register_logs(self, logs: List[WorkflowExecutionLog]) -> None:
        """Register multiple WorkflowExecutionLog objects."""
        self._logs.extend(logs)

    def clear(self) -> None:
        """Remove all registered logs."""
        self._logs.clear()

    @property
    def log_count(self) -> int:
        """Number of registered logs."""
        return len(self._logs)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_logs(
        self,
        export_filter: Optional[ExportFilter] = None,
    ) -> List[WorkflowExecutionLog]:
        """
        Return logs matching the given filter.

        Args:
            export_filter: Optional ExportFilter; if None, all logs are returned.

        Returns:
            Filtered list of WorkflowExecutionLog objects.
        """
        if export_filter is None:
            return list(self._logs)
        return [log for log in self._logs if export_filter.matches_log(log)]

    def filter_steps(
        self,
        export_filter: Optional[ExportFilter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return serialized ExecutionStep records matching the filter.

        Each record includes the parent workflow_id for traceability.

        Args:
            export_filter: Optional ExportFilter.

        Returns:
            List of serialized step dicts.
        """
        results: List[Dict[str, Any]] = []
        for log in self._logs:
            for step in log.steps:
                if export_filter is None or export_filter.matches_step(step):
                    step_dict = _serialize_step(step)
                    # Always ensure workflow_id is set from the parent log
                    if not step_dict.get("workflow_id"):
                        step_dict["workflow_id"] = log.workflow_id
                    results.append(step_dict)
        return results

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def export_json(
        self,
        output_path: Optional[str] = None,
        export_filter: Optional[ExportFilter] = None,
        indent: int = 2,
    ) -> Dict[str, Any]:
        """
        Export filtered logs to a JSON file and return the export dict.

        The output structure is::

            {
                "export_timestamp": "<ISO>",
                "total_records": <int>,
                "logs": [ <WorkflowExecutionLog>, ... ]
            }

        Args:
            output_path: File path to write JSON. If None, only returns dict.
            export_filter: Optional filter criteria.
            indent: JSON indentation level (default 2).

        Returns:
            JSON-serializable dict with export data.

        **Validates: Requirements 7.3, 7.5**
        """
        filtered = self.filter_logs(export_filter)
        serialized_logs = [_serialize_log(log) for log in filtered]

        export_data: Dict[str, Any] = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": len(serialized_logs),
            "logs": serialized_logs,
        }

        if output_path:
            self._write_json(output_path, export_data, indent=indent)

        return export_data

    def export_jsonl(
        self,
        output_path: Optional[str] = None,
        export_filter: Optional[ExportFilter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export filtered logs to a JSONL file (one JSON object per line).

        Each line is a serialized WorkflowExecutionLog.

        Args:
            output_path: File path to write JSONL. If None, only returns list.
            export_filter: Optional filter criteria.

        Returns:
            List of serialized log dicts (one per line).

        **Validates: Requirements 7.3, 7.5**
        """
        filtered = self.filter_logs(export_filter)
        records = [_serialize_log(log) for log in filtered]

        if output_path:
            self._write_jsonl(output_path, records)

        return records

    def export_steps_jsonl(
        self,
        output_path: Optional[str] = None,
        export_filter: Optional[ExportFilter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export individual ExecutionStep records to JSONL format.

        Useful for fine-grained analytics on individual node executions.

        Args:
            output_path: File path to write JSONL. If None, only returns list.
            export_filter: Optional filter criteria.

        Returns:
            List of serialized step dicts.
        """
        records = self.filter_steps(export_filter)

        if output_path:
            self._write_jsonl(output_path, records)

        return records

    # ------------------------------------------------------------------
    # Summary / aggregation export
    # ------------------------------------------------------------------

    def export_summary(
        self,
        output_path: Optional[str] = None,
        export_filter: Optional[ExportFilter] = None,
    ) -> Dict[str, Any]:
        """
        Export an aggregated summary of filtered logs.

        Aggregates WorkflowMetrics fields across all matching logs and
        returns a summary dict suitable for analytics dashboards.

        Args:
            output_path: Optional file path to write JSON summary.
            export_filter: Optional filter criteria.

        Returns:
            Dict with aggregated metrics and per-workflow summaries.

        **Validates: Requirements 7.3, 7.5**
        """
        filtered = self.filter_logs(export_filter)

        if not filtered:
            summary: Dict[str, Any] = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_workflows": 0,
                "aggregated_metrics": {},
                "per_workflow": [],
            }
            if output_path:
                self._write_json(output_path, summary)
            return summary

        # Aggregate metrics
        total_execution_time = sum(log.metrics.total_execution_time for log in filtered)
        total_retries = sum(log.metrics.total_retries for log in filtered)
        total_steps = sum(log.metrics.total_steps for log in filtered)
        successful_steps = sum(log.metrics.successful_steps for log in filtered)
        failed_steps = sum(log.metrics.failed_steps for log in filtered)
        total_llm_tokens = sum(log.metrics.llm_tokens_used for log in filtered)
        total_cost = sum(log.metrics.cost_estimate for log in filtered)
        total_cache_hits = sum(log.metrics.cache_hits for log in filtered)
        total_cache_misses = sum(log.metrics.cache_misses for log in filtered)
        total_critical_issues = sum(log.metrics.critical_issues_found for log in filtered)
        total_major_issues = sum(log.metrics.major_issues_found for log in filtered)
        total_minor_issues = sum(log.metrics.minor_issues_found for log in filtered)

        n = len(filtered)
        avg_execution_time = total_execution_time / n
        cache_total = total_cache_hits + total_cache_misses
        cache_hit_rate = total_cache_hits / cache_total if cache_total > 0 else 0.0
        success_rate = successful_steps / total_steps if total_steps > 0 else 0.0

        # Status breakdown
        status_counts: Dict[str, int] = {}
        for log in filtered:
            status_counts[log.final_status] = status_counts.get(log.final_status, 0) + 1

        # Per-workflow summaries (lightweight)
        per_workflow = [
            {
                "workflow_id": log.workflow_id,
                "correlation_id": log.correlation_id,
                "final_status": log.final_status,
                "start_time": _serialize_datetime(log.start_time),
                "end_time": _serialize_datetime(log.end_time),
                "total_execution_time": log.metrics.total_execution_time,
                "total_retries": log.metrics.total_retries,
                "total_steps": log.metrics.total_steps,
                "llm_tokens_used": log.metrics.llm_tokens_used,
                "cost_estimate": log.metrics.cost_estimate,
                "nodes_executed": log.node_history,
            }
            for log in filtered
        ]

        summary = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_workflows": n,
            "status_breakdown": status_counts,
            "aggregated_metrics": {
                "total_execution_time": total_execution_time,
                "avg_execution_time": avg_execution_time,
                "total_retries": total_retries,
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "success_rate": success_rate,
                "total_llm_tokens": total_llm_tokens,
                "total_cost_usd": total_cost,
                "cache_hit_rate": cache_hit_rate,
                "total_critical_issues": total_critical_issues,
                "total_major_issues": total_major_issues,
                "total_minor_issues": total_minor_issues,
            },
            "per_workflow": per_workflow,
        }

        if output_path:
            self._write_json(output_path, summary)

        return summary

    # ------------------------------------------------------------------
    # Convenience: export to default logs/ directory
    # ------------------------------------------------------------------

    def export_to_logs_dir(
        self,
        filename: str = "execution_export.json",
        export_filter: Optional[ExportFilter] = None,
        fmt: str = "json",
    ) -> str:
        """
        Export logs to the configured logs/ directory.

        Args:
            filename: Output filename (default: execution_export.json).
            export_filter: Optional filter criteria.
            fmt: "json", "jsonl", or "summary".

        Returns:
            Absolute path of the written file.
        """
        output_path = self._logs_dir / "workflow" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "jsonl":
            self.export_jsonl(str(output_path), export_filter)
        elif fmt == "summary":
            self.export_summary(str(output_path), export_filter)
        else:
            self.export_json(str(output_path), export_filter)

        return str(output_path.resolve())

    # ------------------------------------------------------------------
    # Private I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(path: str, data: Dict[str, Any], indent: int = 2) -> None:
        """Write data as JSON to path, creating parent directories as needed."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=_default_serializer)

    @staticmethod
    def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
        """Write records as JSONL to path, creating parent directories as needed."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for record in records:
                line = json.dumps(record, ensure_ascii=False, default=_default_serializer)
                f.write(line + "\n")


# ---------------------------------------------------------------------------
# Serialization helper for datetime fields in WorkflowExecutionLog
# ---------------------------------------------------------------------------

def _serialize_datetime(value: Any) -> Optional[str]:
    """Convert datetime or str to ISO string, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
