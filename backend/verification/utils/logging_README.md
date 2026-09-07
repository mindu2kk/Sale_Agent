# Verification logging

`logging.py` provides correlation context, structured log formatting, an
in-process workflow-status registry, and `EnhancedVerificationLogger`.
`logging_setup.py` builds and returns configured loggers from
`VerificationConfig`.

## Setup and context

```python
from backend.verification.utils.logging import (
    CorrelationIDGenerator,
    correlation_context,
    performance_tracking,
    workflow_context,
)
from backend.verification.utils.logging_setup import setup_verification_logging

configurator = setup_verification_logging("development")
logger = configurator.get_logger("backend.verification.example")
workflow_id = CorrelationIDGenerator.generate_workflow_id()

with workflow_context(workflow_id, logger):
    with correlation_context(workflow_id=workflow_id):
        with performance_tracking("verification", logger):
            logger.log_node_execution("verification", "completed", 0.2)
```

`correlation_context()` restores any prior context when it exits.
`async_correlation_context()` uses `contextvars`, so child asyncio tasks inherit
the correlation context. The status registry is process-local; use
`get_workflow_status()` while the process is running and
`cleanup_completed_workflows()` to remove old completed, failed, or escalated
entries.

## Logger API

`EnhancedVerificationLogger` exposes standard level methods (`debug`, `info`,
`warning`, `error`, `critical`) plus:

- `log_workflow_start()` / `log_workflow_end()` to update workflow status;
- `log_node_execution()`, `log_verification_start()`, and
  `log_verification_result()` for workflow events;
- `log_performance_metrics()` and `export_metrics()` for an in-memory metrics
  buffer (exporting clears that logger's buffer);
- `log_error_with_context()`, `log_retry_attempt()`, and `log_escalation()`.

With `detailed_logging=True`, `EnhancedVerificationLogger` uses
`StructuredFormatter`, which adds active correlation and workflow fields to log
records. Handler destinations and levels are determined by
`VerificationConfig` and the environment passed to
`setup_verification_logging()`.

## Verification

The public API and sync/async context restoration are covered by
`tests/verification/test_logging.py`. The end-to-end usage sample is
`tools/examples/logging_example.py`.

Run from the repository root:

```powershell
python -m pytest tests/verification/test_logging.py -q
python tools/examples/logging_example.py
```
