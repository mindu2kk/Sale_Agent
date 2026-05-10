"""
Tests for ExecutionLogExporter (Task 6.1.5)

Covers:
- JSON export with Pydantic serialization
- JSONL export format
- Filtering by workflow_id, status, node_name, date range
- Summary/aggregation export
- File I/O operations

Requirements: 7.3 (structured JSON format), 7.5 (exportable observability data)
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import pytest

from verification.models.execution import (
    ExecutionStatus,
    ExecutionStep,
    WorkflowExecutionLog,
    WorkflowMetrics,
)
from verification.utils.execution_log_exporter import (
    ExecutionLogExporter,
    ExportFilter,
    _serialize_step,
    _serialize_log,
    _serialize_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    total_time: float = 5.0,
    retries: int = 0,
    steps: int = 2,
    successful: int = 2,
    failed: int = 0,
    llm_tokens: int = 1000,
    cost: float = 0.01,
    cache_hits: int = 2,
    cache_misses: int = 1,
    critical: int = 0,
    major: int = 0,
    minor: int = 0,
) -> WorkflowMetrics:
    return WorkflowMetrics(
        total_execution_time=total_time,
        total_retries=retries,
        total_steps=steps,
        successful_steps=successful,
        failed_steps=failed,
        timeout_steps=0,
        nodes_executed=["research", "verification"],
        node_execution_counts={"research": 1, "verification": 1},
        node_average_times={"research": 2.0, "verification": 3.0},
        critical_issues_found=critical,
        major_issues_found=major,
        minor_issues_found=minor,
        total_issues_found=critical + major + minor,
        llm_tokens_used=llm_tokens,
        llm_tokens_input=600,
        llm_tokens_output=400,
        cost_estimate=cost,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        db_queries_count=3,
        external_api_calls=1,
        verification_pass_rate=1.0,
        escalation_rate=0.0,
    )


def _make_step(
    node_name: str = "verification",
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    timestamp: Optional[str] = None,
) -> ExecutionStep:
    return ExecutionStep(
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        node_name=node_name,
        execution_time=1.5,
        status=status,
        input_summary="test input",
        output_summary="test output",
    )


def _make_log(
    workflow_id: str = "wf_test_001",
    final_status: str = "completed",
    start_offset_seconds: float = 0.0,
    duration_seconds: float = 5.0,
    steps: Optional[List[ExecutionStep]] = None,
    node_history: Optional[List[str]] = None,
    metrics: Optional[WorkflowMetrics] = None,
) -> WorkflowExecutionLog:
    start = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=start_offset_seconds
    )
    end = start + timedelta(seconds=duration_seconds)

    log = WorkflowExecutionLog(
        workflow_id=workflow_id,
        correlation_id=f"corr_{workflow_id}",
        start_time=start,
        end_time=end,
        steps=steps or [],
        metrics=metrics or _make_metrics(),
        final_status=final_status,
        node_history=node_history or ["research", "verification"],
    )
    return log


# ---------------------------------------------------------------------------
# Pydantic serialization tests
# ---------------------------------------------------------------------------

class TestPydanticSerialization:
    def test_serialize_step_returns_dict(self):
        step = _make_step()
        result = _serialize_step(step)
        assert isinstance(result, dict)
        assert result["node_name"] == "verification"
        assert result["status"] == "success"

    def test_serialize_step_has_all_fields(self):
        step = _make_step()
        result = _serialize_step(step)
        assert "timestamp" in result
        assert "execution_time" in result
        assert "input_summary" in result
        assert "output_summary" in result

    def test_serialize_log_returns_dict(self):
        log = _make_log()
        result = _serialize_log(log)
        assert isinstance(result, dict)
        assert result["workflow_id"] == "wf_test_001"
        assert result["final_status"] == "completed"

    def test_serialize_log_includes_metrics(self):
        log = _make_log()
        result = _serialize_log(log)
        assert "metrics" in result
        assert isinstance(result["metrics"], dict)
        assert "total_execution_time" in result["metrics"]

    def test_serialize_metrics_returns_dict(self):
        metrics = _make_metrics()
        result = _serialize_metrics(metrics)
        assert isinstance(result, dict)
        assert result["total_execution_time"] == 5.0
        assert result["llm_tokens_used"] == 1000

    def test_serialize_log_with_steps(self):
        steps = [_make_step("research"), _make_step("verification")]
        log = _make_log(steps=steps)
        result = _serialize_log(log)
        assert len(result["steps"]) == 2
        assert result["steps"][0]["node_name"] == "research"

    def test_serialized_log_is_json_serializable(self):
        log = _make_log()
        result = _serialize_log(log)
        # Should not raise
        serialized = json.dumps(result, default=str)
        parsed = json.loads(serialized)
        assert parsed["workflow_id"] == "wf_test_001"


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------

class TestJsonExport:
    def test_export_json_returns_dict(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        data = exporter.export_json()
        assert isinstance(data, dict)

    def test_export_json_has_required_keys(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        data = exporter.export_json()
        assert "export_timestamp" in data
        assert "total_records" in data
        assert "logs" in data

    def test_export_json_total_records_matches(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        data = exporter.export_json()
        assert data["total_records"] == 2
        assert len(data["logs"]) == 2

    def test_export_json_writes_file(self, tmp_path):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        output = str(tmp_path / "export.json")
        exporter.export_json(output_path=output)
        assert Path(output).exists()

    def test_export_json_file_is_valid_json(self, tmp_path):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        output = str(tmp_path / "export.json")
        exporter.export_json(output_path=output)
        with open(output, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed["total_records"] == 1

    def test_export_json_creates_parent_dirs(self, tmp_path):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        output = str(tmp_path / "nested" / "dir" / "export.json")
        exporter.export_json(output_path=output)
        assert Path(output).exists()

    def test_export_json_empty_logs(self):
        exporter = ExecutionLogExporter()
        data = exporter.export_json()
        assert data["total_records"] == 0
        assert data["logs"] == []

    def test_export_json_timestamp_is_iso(self):
        exporter = ExecutionLogExporter()
        data = exporter.export_json()
        ts = data["export_timestamp"]
        datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# JSONL export tests
# ---------------------------------------------------------------------------

class TestJsonlExport:
    def test_export_jsonl_returns_list(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        records = exporter.export_jsonl()
        assert isinstance(records, list)
        assert len(records) == 1

    def test_export_jsonl_writes_file(self, tmp_path):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        output = str(tmp_path / "export.jsonl")
        exporter.export_jsonl(output_path=output)
        assert Path(output).exists()

    def test_export_jsonl_one_line_per_record(self, tmp_path):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        output = str(tmp_path / "export.jsonl")
        exporter.export_jsonl(output_path=output)
        with open(output, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2

    def test_export_jsonl_each_line_is_valid_json(self, tmp_path):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        output = str(tmp_path / "export.jsonl")
        exporter.export_jsonl(output_path=output)
        with open(output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parsed = json.loads(line)
                    assert "workflow_id" in parsed

    def test_export_steps_jsonl(self, tmp_path):
        steps = [_make_step("research"), _make_step("verification")]
        log = _make_log(steps=steps)
        exporter = ExecutionLogExporter()
        exporter.register_log(log)
        output = str(tmp_path / "steps.jsonl")
        records = exporter.export_steps_jsonl(output_path=output)
        assert len(records) == 2
        assert Path(output).exists()

    def test_export_steps_jsonl_includes_workflow_id(self):
        steps = [_make_step("research")]
        log = _make_log("wf_abc", steps=steps)
        exporter = ExecutionLogExporter()
        exporter.register_log(log)
        records = exporter.export_steps_jsonl()
        assert records[0]["workflow_id"] == "wf_abc"


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_filter_by_workflow_id(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_001"))
        exporter.register_log(_make_log("wf_002"))
        f = ExportFilter(workflow_id="wf_001")
        filtered = exporter.filter_logs(f)
        assert len(filtered) == 1
        assert filtered[0].workflow_id == "wf_001"

    def test_filter_by_status(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1", final_status="completed"))
        exporter.register_log(_make_log("wf_2", final_status="failed"))
        exporter.register_log(_make_log("wf_3", final_status="completed"))
        f = ExportFilter(status="completed")
        filtered = exporter.filter_logs(f)
        assert len(filtered) == 2

    def test_filter_by_node_name(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1", node_history=["research", "verification"]))
        exporter.register_log(_make_log("wf_2", node_history=["research"]))
        f = ExportFilter(node_name="verification")
        filtered = exporter.filter_logs(f)
        assert len(filtered) == 1
        assert filtered[0].workflow_id == "wf_1"

    def test_filter_by_start_date(self):
        exporter = ExecutionLogExporter()
        # wf_1 starts at offset 0 (2024-01-15 10:00:00)
        exporter.register_log(_make_log("wf_1", start_offset_seconds=0))
        # wf_2 starts at offset 3600 (2024-01-15 11:00:00)
        exporter.register_log(_make_log("wf_2", start_offset_seconds=3600))

        cutoff = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        f = ExportFilter(start_date=cutoff)
        filtered = exporter.filter_logs(f)
        assert len(filtered) == 1
        assert filtered[0].workflow_id == "wf_2"

    def test_filter_by_end_date(self):
        exporter = ExecutionLogExporter()
        # wf_1: ends at 10:00:05
        exporter.register_log(_make_log("wf_1", start_offset_seconds=0, duration_seconds=5))
        # wf_2: ends at 11:00:05
        exporter.register_log(_make_log("wf_2", start_offset_seconds=3600, duration_seconds=5))

        cutoff = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        f = ExportFilter(end_date=cutoff)
        filtered = exporter.filter_logs(f)
        assert len(filtered) == 1
        assert filtered[0].workflow_id == "wf_1"

    def test_filter_no_criteria_returns_all(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        filtered = exporter.filter_logs(None)
        assert len(filtered) == 2

    def test_filter_combined_workflow_id_and_status(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1", final_status="completed"))
        exporter.register_log(_make_log("wf_1", final_status="failed"))
        exporter.register_log(_make_log("wf_2", final_status="completed"))
        f = ExportFilter(workflow_id="wf_1", status="completed")
        filtered = exporter.filter_logs(f)
        assert len(filtered) == 1

    def test_filter_steps_by_node_name(self):
        steps = [_make_step("research"), _make_step("verification"), _make_step("research")]
        log = _make_log(steps=steps)
        exporter = ExecutionLogExporter()
        exporter.register_log(log)
        f = ExportFilter(node_name="research")
        step_records = exporter.filter_steps(f)
        assert len(step_records) == 2
        assert all(r["node_name"] == "research" for r in step_records)

    def test_export_json_with_filter(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1", final_status="completed"))
        exporter.register_log(_make_log("wf_2", final_status="failed"))
        f = ExportFilter(status="failed")
        data = exporter.export_json(export_filter=f)
        assert data["total_records"] == 1
        assert data["logs"][0]["workflow_id"] == "wf_2"

    def test_export_jsonl_with_filter(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1", final_status="completed"))
        exporter.register_log(_make_log("wf_2", final_status="escalated"))
        f = ExportFilter(status="escalated")
        records = exporter.export_jsonl(export_filter=f)
        assert len(records) == 1
        assert records[0]["workflow_id"] == "wf_2"


# ---------------------------------------------------------------------------
# Summary / aggregation export tests
# ---------------------------------------------------------------------------

class TestSummaryExport:
    def test_summary_returns_dict(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        summary = exporter.export_summary()
        assert isinstance(summary, dict)

    def test_summary_has_required_keys(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        summary = exporter.export_summary()
        assert "export_timestamp" in summary
        assert "total_workflows" in summary
        assert "aggregated_metrics" in summary
        assert "per_workflow" in summary

    def test_summary_total_workflows_count(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        exporter.register_log(_make_log("wf_3"))
        summary = exporter.export_summary()
        assert summary["total_workflows"] == 3

    def test_summary_aggregates_execution_time(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log(metrics=_make_metrics(total_time=4.0)))
        exporter.register_log(_make_log(metrics=_make_metrics(total_time=6.0)))
        summary = exporter.export_summary()
        agg = summary["aggregated_metrics"]
        assert abs(agg["total_execution_time"] - 10.0) < 1e-6
        assert abs(agg["avg_execution_time"] - 5.0) < 1e-6

    def test_summary_aggregates_llm_tokens(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log(metrics=_make_metrics(llm_tokens=1000)))
        exporter.register_log(_make_log(metrics=_make_metrics(llm_tokens=2000)))
        summary = exporter.export_summary()
        assert summary["aggregated_metrics"]["total_llm_tokens"] == 3000

    def test_summary_aggregates_cost(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log(metrics=_make_metrics(cost=0.01)))
        exporter.register_log(_make_log(metrics=_make_metrics(cost=0.02)))
        summary = exporter.export_summary()
        assert abs(summary["aggregated_metrics"]["total_cost_usd"] - 0.03) < 1e-6

    def test_summary_status_breakdown(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1", final_status="completed"))
        exporter.register_log(_make_log("wf_2", final_status="completed"))
        exporter.register_log(_make_log("wf_3", final_status="failed"))
        summary = exporter.export_summary()
        assert summary["status_breakdown"]["completed"] == 2
        assert summary["status_breakdown"]["failed"] == 1

    def test_summary_per_workflow_list(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        summary = exporter.export_summary()
        assert len(summary["per_workflow"]) == 2
        ids = {w["workflow_id"] for w in summary["per_workflow"]}
        assert "wf_1" in ids
        assert "wf_2" in ids

    def test_summary_empty_logs(self):
        exporter = ExecutionLogExporter()
        summary = exporter.export_summary()
        assert summary["total_workflows"] == 0
        assert summary["aggregated_metrics"] == {}

    def test_summary_writes_file(self, tmp_path):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log())
        output = str(tmp_path / "summary.json")
        exporter.export_summary(output_path=output)
        assert Path(output).exists()
        with open(output, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed["total_workflows"] == 1

    def test_summary_with_filter(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1", final_status="completed"))
        exporter.register_log(_make_log("wf_2", final_status="failed"))
        f = ExportFilter(status="completed")
        summary = exporter.export_summary(export_filter=f)
        assert summary["total_workflows"] == 1

    def test_summary_cache_hit_rate(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log(metrics=_make_metrics(cache_hits=3, cache_misses=1)))
        summary = exporter.export_summary()
        assert abs(summary["aggregated_metrics"]["cache_hit_rate"] - 0.75) < 1e-6

    def test_summary_success_rate(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log(metrics=_make_metrics(steps=4, successful=3, failed=1)))
        summary = exporter.export_summary()
        assert abs(summary["aggregated_metrics"]["success_rate"] - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# File I/O and logs/ directory integration tests
# ---------------------------------------------------------------------------

class TestFileIO:
    def test_register_and_clear(self):
        exporter = ExecutionLogExporter()
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        assert exporter.log_count == 2
        exporter.clear()
        assert exporter.log_count == 0

    def test_register_logs_bulk(self):
        exporter = ExecutionLogExporter()
        logs = [_make_log(f"wf_{i}") for i in range(5)]
        exporter.register_logs(logs)
        assert exporter.log_count == 5

    def test_export_to_logs_dir_json(self, tmp_path):
        exporter = ExecutionLogExporter(logs_dir=str(tmp_path))
        exporter.register_log(_make_log())
        path = exporter.export_to_logs_dir("test_export.json", fmt="json")
        assert Path(path).exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_records"] == 1

    def test_export_to_logs_dir_jsonl(self, tmp_path):
        exporter = ExecutionLogExporter(logs_dir=str(tmp_path))
        exporter.register_log(_make_log("wf_1"))
        exporter.register_log(_make_log("wf_2"))
        path = exporter.export_to_logs_dir("test_export.jsonl", fmt="jsonl")
        assert Path(path).exists()
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

    def test_export_to_logs_dir_summary(self, tmp_path):
        exporter = ExecutionLogExporter(logs_dir=str(tmp_path))
        exporter.register_log(_make_log())
        path = exporter.export_to_logs_dir("summary.json", fmt="summary")
        assert Path(path).exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "aggregated_metrics" in data

    def test_export_to_logs_dir_creates_workflow_subdir(self, tmp_path):
        exporter = ExecutionLogExporter(logs_dir=str(tmp_path))
        exporter.register_log(_make_log())
        path = exporter.export_to_logs_dir("out.json")
        # Should be inside logs/workflow/
        assert "workflow" in path

    def test_default_logs_dir_is_logs(self):
        exporter = ExecutionLogExporter()
        assert exporter._logs_dir == Path("logs")

    def test_custom_logs_dir(self, tmp_path):
        exporter = ExecutionLogExporter(logs_dir=str(tmp_path))
        assert exporter._logs_dir == tmp_path
