"""
Enhanced Structured Logging Utilities cho Verification Agent

Comprehensive logging system với:
- Structured JSON formatting cho observability
- Correlation ID tracking cho distributed tracing (thread-local + contextvars for async)
- Workflow execution monitoring
- Performance metrics integration
- Error context preservation
- Real-time status monitoring
- Exportable analytics data
"""

import logging
import logging.config
import json
import uuid
import os
import threading
import time
import contextvars
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path
from dataclasses import dataclass, asdict

from ..config import VerificationConfig, LogLevel


# Thread-local storage for correlation context (sync/thread usage)
_correlation_context = threading.local()

# ContextVar for async-safe correlation ID propagation across async tasks
# This propagates automatically through asyncio tasks created within a context
_async_correlation_var: contextvars.ContextVar[Optional["CorrelationContext"]] = contextvars.ContextVar(
    "_async_correlation_var", default=None
)

# Global workflow status tracking
_workflow_status_registry: Dict[str, Dict[str, Any]] = {}
_workflow_registry_lock = threading.Lock()


@dataclass
class CorrelationContext:
    """Correlation context for distributed tracing"""
    correlation_id: str
    workflow_id: Optional[str] = None
    parent_correlation_id: Optional[str] = None
    start_time: datetime = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)


@dataclass
class WorkflowStatus:
    """Real-time workflow status tracking"""
    workflow_id: str
    correlation_id: str
    status: str
    current_node: Optional[str] = None
    progress_percentage: float = 0.0
    start_time: datetime = None
    last_update: datetime = None
    execution_steps: List[str] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)
        if self.last_update is None:
            self.last_update = datetime.now(timezone.utc)
        if self.execution_steps is None:
            self.execution_steps = []


class CorrelationIDGenerator:
    """Enhanced correlation ID generation với custom formats"""

    @staticmethod
    def generate_correlation_id(prefix: str = "wf") -> str:
        """Generate correlation ID với timestamp và random component"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = uuid.uuid4().hex[:6]
        return f"{prefix}_{timestamp}_{random_part}"

    @staticmethod
    def generate_workflow_id(objection_hash: Optional[str] = None) -> str:
        """Generate workflow ID với optional objection hash"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if objection_hash:
            return f"workflow_{timestamp}_{objection_hash[:8]}"
        else:
            random_part = uuid.uuid4().hex[:8]
            return f"workflow_{timestamp}_{random_part}"

    @staticmethod
    def validate_correlation_id(correlation_id: str) -> bool:
        """Validate correlation ID format"""
        if not correlation_id or len(correlation_id) > 50:
            return False

        # Check format: prefix_timestamp_random
        parts = correlation_id.split('_')
        if len(parts) != 3:
            return False

        prefix, timestamp, random_part = parts

        # Validate prefix
        if not prefix.startswith('wf'):
            return False

        # Validate timestamp format
        if len(timestamp) != 14:
            return False

        # Validate random part
        if len(random_part) != 6:
            return False

        return True
    """
    JSON formatter cho structured logging
    """

class StructuredFormatter(logging.Formatter):
    """
    Enhanced JSON formatter cho structured logging với correlation tracking
    """

    def __init__(self, include_correlation_id: bool = True,
                 include_workflow_context: bool = True,
                 include_performance_metrics: bool = True):
        super().__init__()
        self.include_correlation_id = include_correlation_id
        self.include_workflow_context = include_workflow_context
        self.include_performance_metrics = include_performance_metrics

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON với enhanced context"""

        # Base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": threading.current_thread().ident,
            "process_id": os.getpid()
        }

        # Add correlation context
        if self.include_correlation_id:
            correlation_context = get_correlation_context()
            if correlation_context:
                log_entry["correlation_id"] = correlation_context.correlation_id
                if correlation_context.workflow_id:
                    log_entry["workflow_id"] = correlation_context.workflow_id
                if correlation_context.parent_correlation_id:
                    log_entry["parent_correlation_id"] = correlation_context.parent_correlation_id

        # Add workflow context
        if self.include_workflow_context and hasattr(record, 'workflow_context'):
            log_entry["workflow_context"] = record.workflow_context

        # Add performance metrics
        if self.include_performance_metrics:
            if hasattr(record, 'execution_time'):
                log_entry["execution_time"] = record.execution_time
            if hasattr(record, 'tokens_used'):
                log_entry["tokens_used"] = record.tokens_used
            if hasattr(record, 'memory_usage'):
                log_entry["memory_usage"] = record.memory_usage
            if hasattr(record, 'api_calls'):
                log_entry["api_calls"] = record.api_calls

        # Add error details if exception
        # exc_info can be True (boolean) when passed directly to LogRecord constructor,
        # or a (type, value, tb) tuple when captured by the logging framework.
        # Normalise to a proper tuple before accessing elements.
        import sys
        exc_info = record.exc_info
        if exc_info is True:
            exc_info = sys.exc_info()
        if exc_info and isinstance(exc_info, tuple) and exc_info[0] is not None:
            log_entry["exception"] = {
                "type": exc_info[0].__name__,
                "message": str(exc_info[1]),
                "traceback": self.formatException(exc_info)
            }

        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add workflow status if available
        if hasattr(record, 'workflow_status'):
            log_entry["workflow_status"] = record.workflow_status

        # Add node information if available
        if hasattr(record, 'node_name'):
            log_entry["node_name"] = record.node_name

        # Add verification result if available
        if hasattr(record, 'verification_result'):
            log_entry["verification_result"] = record.verification_result

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class EnhancedVerificationLogger:
    """
    Enhanced logger cho Verification Agent với comprehensive observability
    """

    def __init__(self, name: str, config: VerificationConfig):
        """
        Initialize enhanced verification logger

        Args:
            name: Logger name
            config: Verification configuration
        """
        self.logger = logging.getLogger(name)
        self.config = config
        self.name = name

        # Performance tracking
        self._start_times: Dict[str, float] = {}
        self._metrics_buffer: List[Dict[str, Any]] = []

        # Configure logger if not already configured
        if not self.logger.handlers:
            self._configure_logger()

    def _configure_logger(self):
        """Configure logger với enhanced structured formatting"""

        # Set log level
        log_level_value = self.config.log_level.value if hasattr(self.config.log_level, 'value') else self.config.log_level
        log_level = getattr(logging, log_level_value)
        self.logger.setLevel(log_level)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Set formatter
        if self.config.detailed_logging:
            formatter = StructuredFormatter(
                include_correlation_id=True,
                include_workflow_context=True,
                include_performance_metrics=True
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if configured
        if hasattr(self.config, 'log_file_path'):
            file_handler = logging.FileHandler(self.config.log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Prevent duplicate logs
        self.logger.propagate = False

    def _get_context_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get enhanced context information cho log entry"""

        context = {}

        # Add correlation context
        correlation_context = get_correlation_context()
        if correlation_context:
            context["correlation_id"] = correlation_context.correlation_id
            if correlation_context.workflow_id:
                context["workflow_id"] = correlation_context.workflow_id

        # Add workflow status
        if correlation_context and correlation_context.workflow_id:
            workflow_status = get_workflow_status(correlation_context.workflow_id)
            if workflow_status:
                context["workflow_status"] = workflow_status.status
                context["current_node"] = workflow_status.current_node
                context["progress_percentage"] = workflow_status.progress_percentage

        # Add custom extra fields
        if extra:
            context["extra_fields"] = extra

        return context

    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation_name}_{int(time.time() * 1000)}"
        self._start_times[timer_id] = time.time()
        return timer_id

    def end_timer(self, timer_id: str) -> float:
        """End timing and return duration"""
        if timer_id in self._start_times:
            duration = time.time() - self._start_times[timer_id]
            del self._start_times[timer_id]
            return duration
        return 0.0

    def debug(self, message: str, **kwargs):
        """Log debug message với enhanced context"""
        self.logger.debug(message, extra=self._get_context_extra(kwargs))

    def info(self, message: str, **kwargs):
        """Log info message với enhanced context"""
        self.logger.info(message, extra=self._get_context_extra(kwargs))

    def warning(self, message: str, **kwargs):
        """Log warning message với enhanced context"""
        self.logger.warning(message, extra=self._get_context_extra(kwargs))

    def error(self, message: str, **kwargs):
        """Log error message với enhanced context"""
        self.logger.error(message, extra=self._get_context_extra(kwargs))

    def critical(self, message: str, **kwargs):
        """Log critical message với enhanced context"""
        self.logger.critical(message, extra=self._get_context_extra(kwargs))

    def log_workflow_start(self, workflow_id: str, objection: str):
        """Log workflow execution start"""
        self.info(
            f"Workflow started: {workflow_id}",
            workflow_event="workflow_start",
            workflow_id=workflow_id,
            objection_length=len(objection),
            objection_preview=objection[:100] + "..." if len(objection) > 100 else objection
        )

        # Update workflow status
        update_workflow_status(workflow_id, "running", current_node="research")

    def log_workflow_end(self, workflow_id: str, final_status: str, execution_time: float):
        """Log workflow execution end"""
        self.info(
            f"Workflow completed: {workflow_id} - Status: {final_status}",
            workflow_event="workflow_end",
            workflow_id=workflow_id,
            final_status=final_status,
            total_execution_time=execution_time
        )

        # Update workflow status
        update_workflow_status(workflow_id, final_status, progress_percentage=100.0)

    def log_node_execution(self, node_name: str, status: str, execution_time: float, **kwargs):
        """Log StateGraph node execution"""
        self.info(
            f"Node {node_name} executed: {status}",
            workflow_event="node_execution",
            node_name=node_name,
            node_status=status,
            execution_time=execution_time,
            **kwargs
        )

        # Update workflow status
        correlation_context = get_correlation_context()
        if correlation_context and correlation_context.workflow_id:
            update_workflow_status(
                correlation_context.workflow_id,
                "running",
                current_node=node_name
            )

    def log_verification_start(self, objection: str, draft: str):
        """Log verification process start"""
        timer_id = self.start_timer("verification")

        self.info(
            "Verification process started",
            workflow_event="verification_start",
            objection_length=len(objection),
            draft_length=len(draft),
            timer_id=timer_id
        )

        return timer_id

    def log_verification_result(self, result, timer_id: str = None):
        """Log verification result với performance metrics"""
        execution_time = self.end_timer(timer_id) if timer_id else 0.0

        self.info(
            f"Verification completed: {'PASSED' if result.is_approved else 'FAILED'}",
            workflow_event="verification_complete",
            verification_passed=result.is_approved,
            critical_issues=getattr(result.criteria, 'critical_issues_count', 0),
            execution_time=execution_time,
            tokens_used=getattr(result, 'llm_tokens_used', 0),
            verification_reasoning=getattr(result, 'verification_reasoning', '')[:200]
        )

    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics với structured format"""
        self.info(
            "Performance metrics recorded",
            workflow_event="performance_metrics",
            **metrics
        )

        # Buffer metrics for batch export
        self._metrics_buffer.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics
        })

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error với comprehensive context"""
        correlation_context = get_correlation_context()

        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "workflow_event": "error_occurred"
        }

        if correlation_context:
            error_context["correlation_id"] = correlation_context.correlation_id
            if correlation_context.workflow_id:
                error_context["workflow_id"] = correlation_context.workflow_id

        self.error(
            f"Error occurred: {str(error)}",
            exc_info=True,
            **error_context
        )

    def log_state_transition(self, from_state: str, to_state: str, condition: str = None):
        """Log StateGraph state transitions"""
        self.debug(
            f"State transition: {from_state} -> {to_state}",
            workflow_event="state_transition",
            from_state=from_state,
            to_state=to_state,
            transition_condition=condition
        )

    def log_retry_attempt(self, attempt_number: int, max_retries: int, reason: str):
        """Log retry attempts"""
        self.warning(
            f"Retry attempt {attempt_number}/{max_retries}: {reason}",
            workflow_event="retry_attempt",
            attempt_number=attempt_number,
            max_retries=max_retries,
            retry_reason=reason
        )

    def log_escalation(self, reason: str, context: Dict[str, Any]):
        """Log human escalation events"""
        self.warning(
            f"Escalating to human: {reason}",
            workflow_event="human_escalation",
            escalation_reason=reason,
            escalation_context=context
        )

    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export buffered metrics và clear buffer"""
        metrics = self._metrics_buffer.copy()
        self._metrics_buffer.clear()
        return metrics


# Global logger registry
_loggers: Dict[str, EnhancedVerificationLogger] = {}


def setup_verification_logger(name: str, config: VerificationConfig) -> EnhancedVerificationLogger:
    """
    Setup enhanced verification logger với configuration

    Args:
        name: Logger name
        config: Verification configuration

    Returns:
        Configured EnhancedVerificationLogger instance
    """

    if name not in _loggers:
        _loggers[name] = EnhancedVerificationLogger(name, config)

    return _loggers[name]


def get_logger(name: str) -> EnhancedVerificationLogger:
    """
    Get existing logger by name

    Args:
        name: Logger name

    Returns:
        EnhancedVerificationLogger instance

    Raises:
        KeyError: If logger not found
    """

    if name not in _loggers:
        raise KeyError(f"Logger '{name}' not found. Use setup_verification_logger() first.")

    return _loggers[name]


def set_correlation_context(correlation_id: str, workflow_id: Optional[str] = None,
                          parent_correlation_id: Optional[str] = None) -> None:
    """Set correlation context for current thread AND async context (contextvars).

    Sets both thread-local (for sync/thread usage) and contextvars (for async task propagation).
    When called inside an async function, the context propagates automatically to child tasks.
    """
    ctx = CorrelationContext(
        correlation_id=correlation_id,
        workflow_id=workflow_id,
        parent_correlation_id=parent_correlation_id
    )
    # Thread-local for sync/thread usage
    _correlation_context.context = ctx
    # ContextVar for async task propagation
    _async_correlation_var.set(ctx)


def get_correlation_context() -> Optional[CorrelationContext]:
    """Get correlation context from current async context (contextvars) or thread-local fallback.

    Checks contextvars first (async-safe), then falls back to thread-local storage.
    This ensures correlation IDs propagate correctly through asyncio.gather() and
    asyncio.create_task() calls.
    """
    # Check contextvars first (async-safe, propagates through async tasks)
    async_ctx = _async_correlation_var.get(None)
    if async_ctx is not None:
        return async_ctx
    # Fallback to thread-local (sync/thread usage)
    return getattr(_correlation_context, 'context', None)


def clear_correlation_context() -> None:
    """Clear correlation context from both thread-local and async context."""
    if hasattr(_correlation_context, 'context'):
        delattr(_correlation_context, 'context')
    _async_correlation_var.set(None)


def set_async_correlation_context(correlation_id: str, workflow_id: Optional[str] = None,
                                   parent_correlation_id: Optional[str] = None) -> contextvars.Token:
    """Set correlation context using contextvars only (async-safe, returns token for restoration).

    Use this in async functions when you want to set context that propagates to child tasks
    but can be cleanly restored. Returns a token that can be used with reset_async_correlation_context().

    Args:
        correlation_id: Correlation ID for this async context
        workflow_id: Optional workflow ID
        parent_correlation_id: Optional parent correlation ID

    Returns:
        contextvars.Token for restoring previous context
    """
    ctx = CorrelationContext(
        correlation_id=correlation_id,
        workflow_id=workflow_id,
        parent_correlation_id=parent_correlation_id
    )
    return _async_correlation_var.set(ctx)


def reset_async_correlation_context(token: contextvars.Token) -> None:
    """Reset async correlation context to previous value using token.

    Args:
        token: Token returned by set_async_correlation_context()
    """
    _async_correlation_var.reset(token)


def update_workflow_status(workflow_id: str, status: str,
                         current_node: Optional[str] = None,
                         progress_percentage: Optional[float] = None) -> None:
    """Update workflow status trong global registry"""
    with _workflow_registry_lock:
        if workflow_id not in _workflow_status_registry:
            correlation_context = get_correlation_context()
            correlation_id = correlation_context.correlation_id if correlation_context else "unknown"

            _workflow_status_registry[workflow_id] = WorkflowStatus(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                status=status,
                current_node=current_node,
                progress_percentage=progress_percentage or 0.0
            )
        else:
            workflow_status = _workflow_status_registry[workflow_id]
            workflow_status.status = status
            workflow_status.last_update = datetime.now(timezone.utc)

            if current_node is not None:
                workflow_status.current_node = current_node
                workflow_status.execution_steps.append(current_node)

            if progress_percentage is not None:
                workflow_status.progress_percentage = progress_percentage


def get_workflow_status(workflow_id: str) -> Optional[WorkflowStatus]:
    """Get workflow status từ global registry"""
    with _workflow_registry_lock:
        return _workflow_status_registry.get(workflow_id)


def get_all_workflow_statuses() -> Dict[str, WorkflowStatus]:
    """Get all workflow statuses"""
    with _workflow_registry_lock:
        return _workflow_status_registry.copy()


def cleanup_completed_workflows(max_age_hours: int = 24) -> int:
    """Cleanup completed workflows older than max_age_hours"""
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    cleaned_count = 0

    with _workflow_registry_lock:
        completed_workflows = []

        for workflow_id, status in _workflow_status_registry.items():
            if (status.status in ["completed", "failed", "escalated"] and
                status.last_update < cutoff_time):
                completed_workflows.append(workflow_id)

        for workflow_id in completed_workflows:
            del _workflow_status_registry[workflow_id]
            cleaned_count += 1

    return cleaned_count


@contextmanager
def correlation_context(logger: Optional[EnhancedVerificationLogger] = None,
                       correlation_id: Optional[str] = None,
                       workflow_id: Optional[str] = None,
                       parent_correlation_id: Optional[str] = None):
    """
    Enhanced context manager cho correlation ID tracking

    Args:
        logger: EnhancedVerificationLogger instance (optional)
        correlation_id: Correlation ID (auto-generated if None)
        workflow_id: Workflow ID (optional)
        parent_correlation_id: Parent correlation ID (optional)
    """

    if correlation_id is None:
        correlation_id = CorrelationIDGenerator.generate_correlation_id()

    # Set correlation context
    old_context = get_correlation_context()
    set_correlation_context(correlation_id, workflow_id, parent_correlation_id)

    try:
        yield correlation_id
    finally:
        # Restore previous context
        if old_context:
            set_correlation_context(
                old_context.correlation_id,
                old_context.workflow_id,
                old_context.parent_correlation_id
            )
        else:
            clear_correlation_context()


@asynccontextmanager
async def async_correlation_context(correlation_id: Optional[str] = None,
                                     workflow_id: Optional[str] = None,
                                     parent_correlation_id: Optional[str] = None,
                                     logger: Optional[EnhancedVerificationLogger] = None):
    """
    Async-safe context manager for correlation ID tracking using contextvars.

    Unlike the sync version, this uses contextvars which propagate automatically
    through asyncio.gather() and asyncio.create_task() calls, making it safe
    for use in async workflow nodes.

    Args:
        correlation_id: Correlation ID (auto-generated if None)
        workflow_id: Workflow ID (optional)
        parent_correlation_id: Parent correlation ID (optional)
        logger: EnhancedVerificationLogger instance (optional)

    Yields:
        The correlation_id being used

    Example:
        async with async_correlation_context(correlation_id=state["correlation_id"],
                                              workflow_id=state["workflow_id"]) as cid:
            # All async tasks spawned here inherit the correlation context
            await asyncio.gather(check_price(), check_policy(), check_relevance())
    """
    if correlation_id is None:
        correlation_id = CorrelationIDGenerator.generate_correlation_id()

    # Use contextvars token for clean restoration
    token = set_async_correlation_context(correlation_id, workflow_id, parent_correlation_id)
    # Also set thread-local for any sync code within this context
    old_thread_ctx = getattr(_correlation_context, 'context', None)
    _correlation_context.context = CorrelationContext(
        correlation_id=correlation_id,
        workflow_id=workflow_id,
        parent_correlation_id=parent_correlation_id
    )

    try:
        yield correlation_id
    finally:
        # Restore previous async context
        _async_correlation_var.reset(token)
        # Restore thread-local
        if old_thread_ctx is not None:
            _correlation_context.context = old_thread_ctx
        elif hasattr(_correlation_context, 'context'):
            delattr(_correlation_context, 'context')


@contextmanager
def workflow_context(workflow_id: str, logger: Optional[EnhancedVerificationLogger] = None):
    """
    Enhanced context manager cho workflow tracking

    Args:
        workflow_id: Workflow ID
        logger: EnhancedVerificationLogger instance (optional)
    """

    # Generate correlation ID if not exists
    correlation_context_obj = get_correlation_context()
    if not correlation_context_obj:
        correlation_id = CorrelationIDGenerator.generate_correlation_id()
        set_correlation_context(correlation_id, workflow_id)
    else:
        # Update existing context with workflow ID
        set_correlation_context(
            correlation_context_obj.correlation_id,
            workflow_id,
            correlation_context_obj.parent_correlation_id
        )

    # Initialize workflow status
    update_workflow_status(workflow_id, "initialized")

    if logger:
        logger.log_workflow_start(workflow_id, "Workflow context established")

    try:
        yield workflow_id
    finally:
        # Finalize workflow status
        final_status = get_workflow_status(workflow_id)
        if final_status and final_status.status not in ["completed", "failed", "escalated"]:
            update_workflow_status(workflow_id, "completed")

        if logger:
            logger.log_workflow_end(workflow_id, final_status.status if final_status else "unknown", 0.0)


@contextmanager
def performance_tracking(operation_name: str, logger: EnhancedVerificationLogger):
    """
    Context manager cho performance tracking

    Args:
        operation_name: Name of operation being tracked
        logger: EnhancedVerificationLogger instance
    """

    timer_id = logger.start_timer(operation_name)
    start_time = time.time()

    try:
        yield timer_id
    finally:
        execution_time = logger.end_timer(timer_id)

        # Log performance metrics
        logger.log_performance_metrics({
            "operation": operation_name,
            "execution_time": execution_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


def configure_root_logger(config: VerificationConfig):
    """
    Configure root logger cho verification system với enhanced features

    Args:
        config: Verification configuration
    """

    # Configure root logger
    root_logger = logging.getLogger("verification")

    # Set log level
    log_level_value = config.log_level.value if hasattr(config.log_level, 'value') else config.log_level
    log_level = getattr(logging, log_level_value)
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Set formatter
    if config.detailed_logging:
        formatter = StructuredFormatter(
            include_correlation_id=True,
            include_workflow_context=True,
            include_performance_metrics=True
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Prevent duplicate logs
    root_logger.propagate = False


def load_logging_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load logging configuration từ YAML file

    Args:
        config_path: Path to logging config file

    Returns:
        Logging configuration dictionary
    """

    if config_path is None:
        config_path = "backend/verification/config/logging_config.yaml"

    config_file = Path(config_path)

    if not config_file.exists():
        # Return default configuration
        return get_default_logging_config()

    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return config_data
    except Exception as e:
        print(f"Warning: Failed to load logging config from {config_path}: {e}")
        return get_default_logging_config()


def get_default_logging_config() -> Dict[str, Any]:
    """Get default logging configuration"""

    return {
        "logging": {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured_json": {
                    "class": "backend.verification.utils.logging.StructuredFormatter"
                },
                "detailed_console": {
                    "format": "[{asctime}] {levelname:8} | {name:20} | {message}",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "detailed_console"
                }
            },
            "loggers": {
                "verification": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False
                }
            }
        }
    }


def setup_logging_from_config(config_path: Optional[str] = None) -> None:
    """
    Setup logging từ configuration file

    Args:
        config_path: Path to logging config file
    """

    config_data = load_logging_config(config_path)

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    try:
        logging.config.dictConfig(config_data["logging"])
    except Exception as e:
        print(f"Warning: Failed to configure logging: {e}")
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)'
        )


class WorkflowObservabilityManager:
    """
    Manager cho workflow observability và real-time monitoring
    """

    def __init__(self, config: VerificationConfig):
        self.config = config
        self.logger = setup_verification_logger("backend.verification.observability", config)
        self._active_workflows: Dict[str, WorkflowStatus] = {}

    def start_workflow_monitoring(self, workflow_id: str, correlation_id: str) -> None:
        """Start monitoring a workflow"""

        workflow_status = WorkflowStatus(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            status="started"
        )

        self._active_workflows[workflow_id] = workflow_status
        update_workflow_status(workflow_id, "started")

        self.logger.info(
            f"Started monitoring workflow: {workflow_id}",
            workflow_event="monitoring_started",
            workflow_id=workflow_id,
            correlation_id=correlation_id
        )

    def update_workflow_progress(self, workflow_id: str, progress: float, current_node: str = None) -> None:
        """Update workflow progress"""

        if workflow_id in self._active_workflows:
            self._active_workflows[workflow_id].progress_percentage = progress
            if current_node:
                self._active_workflows[workflow_id].current_node = current_node

            update_workflow_status(workflow_id, "running", current_node, progress)

            self.logger.debug(
                f"Workflow progress updated: {workflow_id} - {progress:.1f}%",
                workflow_event="progress_update",
                workflow_id=workflow_id,
                progress_percentage=progress,
                current_node=current_node
            )

    def finish_workflow_monitoring(self, workflow_id: str, final_status: str) -> None:
        """Finish monitoring a workflow"""

        if workflow_id in self._active_workflows:
            self._active_workflows[workflow_id].status = final_status
            self._active_workflows[workflow_id].progress_percentage = 100.0

            update_workflow_status(workflow_id, final_status, progress_percentage=100.0)

            self.logger.info(
                f"Finished monitoring workflow: {workflow_id} - Status: {final_status}",
                workflow_event="monitoring_finished",
                workflow_id=workflow_id,
                final_status=final_status
            )

            # Remove from active workflows after a delay
            # (Keep for a short time for final status queries)
            del self._active_workflows[workflow_id]

    def get_workflow_dashboard_data(self) -> Dict[str, Any]:
        """Get data for workflow status dashboard"""

        active_workflows = list(self._active_workflows.values())

        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_active_workflows": len(active_workflows),
            "workflows_by_status": {},
            "workflows_by_node": {},
            "average_progress": 0.0,
            "active_workflows": []
        }

        if active_workflows:
            # Calculate statistics
            status_counts = {}
            node_counts = {}
            total_progress = 0.0

            for workflow in active_workflows:
                # Count by status
                status_counts[workflow.status] = status_counts.get(workflow.status, 0) + 1

                # Count by current node
                if workflow.current_node:
                    node_counts[workflow.current_node] = node_counts.get(workflow.current_node, 0) + 1

                # Sum progress
                total_progress += workflow.progress_percentage

                # Add to active workflows list
                dashboard_data["active_workflows"].append({
                    "workflow_id": workflow.workflow_id,
                    "correlation_id": workflow.correlation_id,
                    "status": workflow.status,
                    "current_node": workflow.current_node,
                    "progress_percentage": workflow.progress_percentage,
                    "start_time": workflow.start_time.isoformat(),
                    "last_update": workflow.last_update.isoformat()
                })

            dashboard_data["workflows_by_status"] = status_counts
            dashboard_data["workflows_by_node"] = node_counts
            dashboard_data["average_progress"] = total_progress / len(active_workflows)

        return dashboard_data


# Export observability manager instance
_observability_manager: Optional[WorkflowObservabilityManager] = None


def get_observability_manager(config: VerificationConfig) -> WorkflowObservabilityManager:
    """Get global observability manager instance"""
    global _observability_manager

    if _observability_manager is None:
        _observability_manager = WorkflowObservabilityManager(config)

    return _observability_manager
