# Execution tracking models

`execution.py` contains Pydantic models for recording verification-workflow
execution. They are application-level records; they do not persist themselves
or require LangGraph.

## Models

- `ExecutionStep` records one node execution. Its required fields are
  `node_name`, `execution_time`, `status`, `input_summary`, and
  `output_summary`. It creates an `exec_...` correlation ID by default and can
  hold error details, resource use, LLM token/cost data, and arbitrary
  `metrics`.
- `WorkflowMetrics` is an aggregate snapshot. Most counters are required when
  it is created (time, step counts, issue counts, token/cost data, cache data,
  database/API counts, and verification/escalation rates). It recalculates
  `success_rate`, `average_step_time`, and `cache_hit_rate` from the source
  counters. `efficiency_score`, `performance_grade`, and `cost_per_success`
  are read-only derived properties.
- `WorkflowExecutionLog` groups steps, errors, warnings, context, and a
  required `WorkflowMetrics` snapshot. `add_step()` assigns the step index,
  workflow ID, and parent correlation ID; it does not recompute aggregate
  metrics.
- `WorkflowTracker` keeps active `WorkflowExecutionLog` instances in memory
  and tracks completed/failed counts, current load, and slow active workflows.

## Minimal flow

Create an initial metrics snapshot before opening a workflow log, append
steps, then replace the snapshot after calculating the final aggregate:

```python
from backend.verification.models.execution import (
    ExecutionStatus,
    ExecutionStep,
    WorkflowExecutionLog,
    WorkflowMetrics,
    WorkflowTracker,
)

metrics = WorkflowMetrics(  # supply every required aggregate counter
    total_execution_time=0.0, total_retries=0, total_steps=0,
    successful_steps=0, failed_steps=0, timeout_steps=0, nodes_executed=[],
    critical_issues_found=0, major_issues_found=0, minor_issues_found=0,
    total_issues_found=0, llm_tokens_used=0, llm_tokens_input=0,
    llm_tokens_output=0, cost_estimate=0.0, cache_hits=0, cache_misses=0,
    db_queries_count=0, external_api_calls=0, verification_pass_rate=0.0,
    escalation_rate=0.0,
)
log = WorkflowExecutionLog(workflow_id="request-123", metrics=metrics)
tracker = WorkflowTracker()
tracker.start_workflow(log)

log.add_step(ExecutionStep(
    node_name="verification", execution_time=0.2,
    status=ExecutionStatus.SUCCESS, input_summary="draft", output_summary="passed",
))
log.update_status("completed")
tracker.complete_workflow(log.workflow_id, "completed")
```

For a complete, runnable simulation see
`tools/examples/execution_tracking_example.py`.

## Verification

- Model behavior is covered by `tests/verification/test_execution_models.py`.
- Serialization and workflow integrations have additional coverage in
  `tests/verification/test_pydantic_serialization.py`,
  `tests/verification/test_execution_log_exporter.py`, and workflow-node tests.

Run from the repository root:

```powershell
python -m pytest tests/verification/test_execution_models.py -q
python tools/examples/execution_tracking_example.py
```
