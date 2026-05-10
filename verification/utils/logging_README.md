# Enhanced Structured Logging với Correlation IDs

Comprehensive logging system for Verification Agent workflow observability.

## Features

### 🔗 Correlation ID Tracking
- **Automatic ID Generation**: Unique correlation IDs for distributed tracing
- **Thread-Local Context**: Isolated correlation context per thread
- **Nested Context Support**: Hierarchical correlation tracking
- **Format Validation**: Consistent ID format validation

### 📊 Structured JSON Logging
- **JSON Format**: Machine-readable structured logs
- **Rich Context**: Automatic inclusion of correlation IDs, workflow context, performance metrics
- **Exception Handling**: Comprehensive error context preservation
- **Custom Fields**: Extensible log entry fields

### 🔍 Workflow Observability
- **Real-time Monitoring**: Live workflow status tracking
- **Phase Tracking**: Detailed workflow phase transitions
- **Progress Monitoring**: Percentage-based progress tracking
- **Dashboard Data**: Ready-to-use dashboard metrics

### ⚡ Performance Monitoring
- **Execution Timing**: Automatic operation timing
- **Resource Tracking**: Memory, CPU, token usage monitoring
- **Metrics Collection**: Aggregated performance statistics
- **Export Capabilities**: Metrics export for analytics

### 🛡️ Error Handling
- **Context Preservation**: Full error context capture
- **Severity Classification**: Error severity levels
- **Recovery Tracking**: Error recovery attempt logging
- **Alerting Support**: Integration-ready error alerting

## Quick Start

### Basic Setup

```python
from verification.utils.logging_setup import setup_verification_logging
from verification.utils.logging import correlation_context, CorrelationIDGenerator

# Setup logging for your environment
configurator = setup_verification_logging("development")
logger = configurator.get_logger("verification.component")

# Generate correlation ID
correlation_id = CorrelationIDGenerator.generate_correlation_id()

# Use correlation context
with correlation_context(correlation_id=correlation_id):
    logger.info("Message with correlation tracking")
```

### Workflow Tracking

```python
from verification.utils.logging import workflow_context

workflow_id = CorrelationIDGenerator.generate_workflow_id()

with workflow_context(workflow_id, logger):
    logger.log_workflow_start(workflow_id, "Customer objection")
    
    # Your workflow logic here
    
    logger.log_workflow_end(workflow_id, "completed", execution_time)
```

### Performance Monitoring

```python
from verification.utils.logging import performance_tracking

with performance_tracking("verification_operation", logger):
    # Your operation here
    result = perform_verification()

# Metrics are automatically logged
```

## Configuration

### Environment-Specific Configuration

The logging system supports environment-specific configurations:

- **Development**: Debug level, detailed console output
- **Production**: Warning level, structured JSON output, file rotation
- **Testing**: Error level only, minimal output

### Configuration Files

- `verification/config/logging_config.yaml` - Main logging configuration
- `verification/config/environments/` - Environment-specific overrides

### Key Configuration Options

```yaml
logging:
  loggers:
    verification:
      level: "INFO"  # DEBUG, INFO, WARNING, ERROR
      handlers: ["console", "file"]
      
correlation_tracking:
  id_generation:
    format: "wf_{timestamp}_{random}"
    
workflow_observability:
  execution_tracking:
    enabled: true
    track_performance_metrics: true
```

## API Reference

### CorrelationIDGenerator

```python
# Generate correlation ID
correlation_id = CorrelationIDGenerator.generate_correlation_id()
# Returns: "wf_20240115103015_abc123"

# Generate workflow ID
workflow_id = CorrelationIDGenerator.generate_workflow_id()
# Returns: "workflow_20240115_103015_def456"

# Validate correlation ID
is_valid = CorrelationIDGenerator.validate_correlation_id(correlation_id)
```

### EnhancedVerificationLogger

```python
logger = EnhancedVerificationLogger("component.name", config)

# Basic logging với correlation context
logger.info("Message", extra_field="value")
logger.error("Error occurred", error_context={"operation": "verification"})

# Workflow-specific logging
logger.log_workflow_start(workflow_id, objection_text)
logger.log_workflow_end(workflow_id, final_status, execution_time)

# Performance logging
timer_id = logger.start_timer("operation")
# ... perform operation ...
duration = logger.end_timer(timer_id)

# Verification logging
timer_id = logger.log_verification_start(objection, draft)
logger.log_verification_result(verification_result, timer_id)

# Error logging với context
logger.log_error_with_context(exception, context_dict)
```

### Context Managers

```python
# Correlation context
with correlation_context(correlation_id="wf_123") as ctx_id:
    # All logging within this block includes correlation ID
    logger.info("Correlated message")

# Workflow context
with workflow_context(workflow_id, logger):
    # Automatic workflow start/end logging
    # Correlation ID management
    pass

# Performance tracking
with performance_tracking("operation_name", logger):
    # Automatic timing và metrics collection
    perform_operation()
```

### Workflow Status Tracking

```python
# Update workflow status
update_workflow_status(workflow_id, "running", current_node="verification", progress=75.0)

# Get workflow status
status = get_workflow_status(workflow_id)
print(f"Status: {status.status}, Progress: {status.progress_percentage}%")

# Cleanup completed workflows
cleaned_count = cleanup_completed_workflows(max_age_hours=24)
```

## Log Output Examples

### Structured JSON Log Entry

```json
{
  "timestamp": "2024-01-15T10:30:15.123456Z",
  "level": "INFO",
  "logger": "verification.workflow",
  "message": "Verification completed: PASSED",
  "correlation_id": "wf_20240115103015_abc123",
  "workflow_id": "workflow_20240115_103015_def456",
  "workflow_event": "verification_complete",
  "verification_passed": true,
  "critical_issues": 0,
  "execution_time": 2.45,
  "tokens_used": 150,
  "module": "verification_agent",
  "function": "verify_draft",
  "line": 142,
  "thread_id": 12345,
  "process_id": 67890
}
```

### Console Output (Development)

```
[2024-01-15 10:30:15] INFO     | verification.workflow | wf_abc123    | workflow_def456 | Verification completed: PASSED
[2024-01-15 10:30:16] DEBUG    | verification.agent   | wf_abc123    | workflow_def456 | Price accuracy check: 100% match
[2024-01-15 10:30:17] WARNING  | verification.retry   | wf_abc123    | workflow_def456 | Retry attempt 1/3: Policy verification failed
```

## Integration Examples

### With StateGraph Workflow

```python
class VerificationWorkflow:
    def __init__(self):
        self.logger = get_verification_logger("workflow")
    
    def execute_verification_node(self, state):
        workflow_id = state.get("workflow_id")
        
        with workflow_context(workflow_id, self.logger):
            timer_id = self.logger.log_verification_start(
                state["objection"], 
                state["draft_response"]
            )
            
            try:
                result = self.verification_agent.verify_draft(state)
                self.logger.log_verification_result(result, timer_id)
                return result
                
            except Exception as e:
                self.logger.log_error_with_context(e, {
                    "node": "verification",
                    "state_keys": list(state.keys())
                })
                raise
```

### With Async Operations

```python
async def async_verification_workflow(objection: str):
    correlation_id = CorrelationIDGenerator.generate_correlation_id()
    logger = get_verification_logger("async")
    
    with correlation_context(correlation_id=correlation_id):
        logger.info("Starting async verification workflow")
        
        # Async operations maintain correlation context
        research_task = asyncio.create_task(research_objection(objection))
        verification_task = asyncio.create_task(verify_response())
        
        results = await asyncio.gather(research_task, verification_task)
        
        logger.info("Async workflow completed", results_count=len(results))
        return results
```

## Monitoring và Analytics

### Dashboard Integration

The logging system provides ready-to-use dashboard data:

```python
from verification.utils.logging import get_observability_manager

manager = get_observability_manager(config)
dashboard_data = manager.get_workflow_dashboard_data()

# Returns:
{
    "total_active_workflows": 5,
    "workflows_by_status": {"running": 3, "verification": 2},
    "workflows_by_node": {"research": 2, "verification": 3},
    "average_progress": 67.5,
    "active_workflows": [...]
}
```

### Metrics Export

```python
# Export performance metrics
metrics = logger.export_metrics()

# Export execution data
from verification.utils.workflow_tracker import WorkflowExecutionTracker
tracker = WorkflowExecutionTracker(config, logger)
execution_data = tracker.export_execution_data("analytics/workflow_data.json")
```

### Log Analysis

Structured JSON logs can be easily analyzed:

```bash
# Count verification results by status
cat logs/verification_workflow.log | jq -r 'select(.workflow_event=="verification_complete") | .verification_passed' | sort | uniq -c

# Average execution time by workflow phase
cat logs/verification_workflow.log | jq -r 'select(.execution_time) | "\(.workflow_event) \(.execution_time)"' | awk '{sum[$1]+=$2; count[$1]++} END {for(i in sum) print i, sum[i]/count[i]}'

# Error rate by component
cat logs/verification_errors.log | jq -r '.logger' | sort | uniq -c
```

## Best Practices

### 1. Always Use Correlation Context

```python
# ✅ Good
with correlation_context():
    logger.info("Processing request")
    process_request()

# ❌ Bad
logger.info("Processing request")  # No correlation context
```

### 2. Include Relevant Context

```python
# ✅ Good
logger.log_error_with_context(error, {
    "operation": "verification",
    "input_length": len(objection),
    "retry_count": retry_count
})

# ❌ Bad
logger.error(str(error))  # No context
```

### 3. Use Performance Tracking

```python
# ✅ Good
with performance_tracking("llm_call", logger):
    result = llm.generate(prompt)

# ❌ Bad
result = llm.generate(prompt)  # No performance tracking
```

### 4. Structured Error Handling

```python
# ✅ Good
try:
    result = risky_operation()
except SpecificError as e:
    logger.log_error_with_context(e, {
        "operation": "risky_operation",
        "parameters": operation_params
    })
    # Handle error appropriately

# ❌ Bad
try:
    result = risky_operation()
except Exception as e:
    print(f"Error: {e}")  # Poor error handling
```

## Troubleshooting

### Common Issues

1. **Missing Correlation IDs**
   - Ensure you're using `correlation_context()` or `workflow_context()`
   - Check thread-local context isolation in multi-threaded scenarios

2. **Log Files Not Created**
   - Verify `logs/` directory exists và is writable
   - Check logging configuration file paths

3. **Performance Impact**
   - Adjust log levels for production (WARNING or ERROR)
   - Disable detailed logging if not needed
   - Use log rotation to manage file sizes

4. **Memory Usage**
   - Regular cleanup of completed workflows
   - Limit execution history retention
   - Monitor metrics buffer sizes

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
config = VerificationConfig(
    log_level=LogLevel.DEBUG,
    detailed_logging=True,
    performance_tracking=True
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all logging tests
pytest verification/tests/test_logging.py -v

# Run specific test categories
pytest verification/tests/test_logging.py::TestCorrelationIDGenerator -v
pytest verification/tests/test_logging.py::TestStructuredFormatter -v
pytest verification/tests/test_logging.py::TestWorkflowObservability -v
```

## Examples

See `verification/utils/logging_example.py` for comprehensive usage examples:

```bash
python -m verification.utils.logging_example
```

This will demonstrate all logging features và generate sample log files.