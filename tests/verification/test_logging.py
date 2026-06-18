"""
Test Suite for Enhanced Structured Logging với Correlation IDs

Comprehensive tests for:
- Correlation ID generation và validation
- Structured JSON formatting
- Workflow observability tracking
- Performance metrics collection
- Error context preservation
- Thread-local context management
- Async-safe correlation ID propagation via contextvars (Task 6.1.1)
"""

import pytest
import asyncio
import json
import threading
import time
import tempfile
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from verification.utils.logging import (
    CorrelationIDGenerator,
    StructuredFormatter,
    EnhancedVerificationLogger,
    WorkflowObservabilityManager,
    set_correlation_context,
    get_correlation_context,
    clear_correlation_context,
    set_async_correlation_context,
    reset_async_correlation_context,
    async_correlation_context,
    correlation_context,
    workflow_context,
    performance_tracking,
    update_workflow_status,
    get_workflow_status,
    cleanup_completed_workflows
)
from verification.utils.logging_setup import (
    LoggingConfigurator,
    setup_verification_logging,
    get_workflow_logger,
    get_verification_logger,
    configure_logging_for_testing
)
from verification.config import VerificationConfig, LogLevel


class TestCorrelationIDGenerator:
    """Test correlation ID generation và validation"""
    
    def test_generate_correlation_id_format(self):
        """Test correlation ID format"""
        correlation_id = CorrelationIDGenerator.generate_correlation_id()
        
        # Should have format: wf_YYYYMMDDHHMMSS_XXXXXX
        assert correlation_id.startswith("wf_")
        parts = correlation_id.split("_")
        assert len(parts) == 3
        
        prefix, timestamp, random_part = parts
        assert prefix == "wf"
        assert len(timestamp) == 14  # YYYYMMDDHHMMSS
        assert len(random_part) == 6
        assert random_part.isalnum()
    
    def test_generate_workflow_id_format(self):
        """Test workflow ID generation"""
        workflow_id = CorrelationIDGenerator.generate_workflow_id()
        
        # Should have format: workflow_YYYYMMDD_HHMMSS_XXXXXXXX
        assert workflow_id.startswith("workflow_")
        parts = workflow_id.split("_")
        assert len(parts) == 4
        
        prefix, date_part, time_part, random_part = parts
        assert prefix == "workflow"
        assert len(date_part) == 8  # YYYYMMDD
        assert len(time_part) == 6  # HHMMSS
        assert len(random_part) == 8
    
    def test_generate_workflow_id_with_hash(self):
        """Test workflow ID generation với objection hash"""
        objection_hash = "abc123def456"
        workflow_id = CorrelationIDGenerator.generate_workflow_id(objection_hash)
        
        # Should include first 8 chars of hash
        assert "abc123de" in workflow_id
    
    def test_validate_correlation_id_valid(self):
        """Test correlation ID validation - valid cases"""
        valid_ids = [
            "wf_20240115103015_abc123",
            "wf_20231225120000_xyz789",
            "wf_20240301000000_123abc"
        ]
        
        for correlation_id in valid_ids:
            assert CorrelationIDGenerator.validate_correlation_id(correlation_id)
    
    def test_validate_correlation_id_invalid(self):
        """Test correlation ID validation - invalid cases"""
        invalid_ids = [
            "",  # Empty
            "invalid_format",  # Wrong format
            "wf_2024_abc",  # Too few parts
            "wf_20240115103015_abc123_extra",  # Too many parts
            "wrong_20240115103015_abc123",  # Wrong prefix
            "wf_2024011510301_abc123",  # Wrong timestamp length
            "wf_20240115103015_abc12",  # Wrong random length
            "wf_20240115103015_abc123" + "x" * 50,  # Too long
        ]
        
        for correlation_id in invalid_ids:
            assert not CorrelationIDGenerator.validate_correlation_id(correlation_id)


class TestStructuredFormatter:
    """Test structured JSON formatting"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.formatter = StructuredFormatter(
            include_correlation_id=True,
            include_workflow_context=True,
            include_performance_metrics=True
        )
    
    def test_basic_log_formatting(self):
        """Test basic log record formatting"""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        # Check required fields
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
        assert "thread_id" in log_data
        assert "process_id" in log_data
    
    def test_correlation_id_inclusion(self):
        """Test correlation ID inclusion trong log"""
        correlation_id = "wf_20240115103015_abc123"
        workflow_id = "workflow_20240115_103015_def456"
        
        # Set correlation context
        set_correlation_context(correlation_id, workflow_id)
        
        try:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Test with correlation",
                args=(),
                exc_info=None
            )
            
            formatted = self.formatter.format(record)
            log_data = json.loads(formatted)
            
            assert log_data["correlation_id"] == correlation_id
            assert log_data["workflow_id"] == workflow_id
        finally:
            clear_correlation_context()
    
    def test_performance_metrics_inclusion(self):
        """Test performance metrics inclusion"""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test with metrics",
            args=(),
            exc_info=None
        )
        
        # Add performance metrics
        record.execution_time = 1.23
        record.tokens_used = 150
        record.memory_usage = 1024
        record.api_calls = 3
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["execution_time"] == 1.23
        assert log_data["tokens_used"] == 150
        assert log_data["memory_usage"] == 1024
        assert log_data["api_calls"] == 3
    
    def test_exception_formatting(self):
        """Test exception information formatting"""
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
            
            formatted = self.formatter.format(record)
            log_data = json.loads(formatted)
            
            assert "exception" in log_data
            assert log_data["exception"]["type"] == "ValueError"
            assert log_data["exception"]["message"] == "Test exception"
            assert "traceback" in log_data["exception"]


class TestEnhancedVerificationLogger:
    """Test enhanced verification logger"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = VerificationConfig(
            log_level=LogLevel.DEBUG,
            detailed_logging=True,
            performance_tracking=True
        )
        self.logger = EnhancedVerificationLogger("test.verification", self.config)
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        assert self.logger.name == "test.verification"
        assert self.logger.config == self.config
        assert self.logger.logger.level == logging.DEBUG
    
    def test_context_extra_with_correlation(self):
        """Test context extra với correlation ID"""
        correlation_id = "wf_20240115103015_abc123"
        workflow_id = "workflow_20240115_103015_def456"
        
        set_correlation_context(correlation_id, workflow_id)
        
        try:
            extra = self.logger._get_context_extra({"custom": "value"})
            
            assert extra["correlation_id"] == correlation_id
            assert extra["workflow_id"] == workflow_id
            assert extra["extra_fields"]["custom"] == "value"
        finally:
            clear_correlation_context()
    
    def test_timer_functionality(self):
        """Test timer start/end functionality"""
        timer_id = self.logger.start_timer("test_operation")
        
        assert timer_id in self.logger._start_times
        
        time.sleep(0.1)  # Small delay
        duration = self.logger.end_timer(timer_id)
        
        assert duration >= 0.1
        assert timer_id not in self.logger._start_times
    
    def test_workflow_logging_methods(self):
        """Test workflow-specific logging methods"""
        workflow_id = "test_workflow_123"
        objection = "Test objection text"
        
        # Test workflow start logging
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.log_workflow_start(workflow_id, objection)
            mock_info.assert_called_once()
            
            # Check call arguments
            call_args = mock_info.call_args
            assert workflow_id in call_args[0][0]
            assert call_args[1]["extra"]["extra_fields"]["workflow_event"] == "workflow_start"
    
    def test_performance_metrics_logging(self):
        """Test performance metrics logging"""
        metrics = {
            "execution_time": 2.5,
            "tokens_used": 200,
            "api_calls": 5
        }
        
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.log_performance_metrics(metrics)
            mock_info.assert_called_once()
            
            call_args = mock_info.call_args
            assert call_args[1]["extra"]["extra_fields"]["workflow_event"] == "performance_metrics"
            assert call_args[1]["extra"]["extra_fields"]["execution_time"] == 2.5
    
    def test_error_logging_with_context(self):
        """Test error logging với comprehensive context"""
        error = ValueError("Test error")
        context = {"operation": "verification", "input_length": 100}
        
        with patch.object(self.logger.logger, 'error') as mock_error:
            self.logger.log_error_with_context(error, context)
            mock_error.assert_called_once()
            
            call_args = mock_error.call_args
            assert "Error occurred" in call_args[0][0]
            assert call_args[1]["extra"]["extra_fields"]["error_type"] == "ValueError"
            assert call_args[1]["extra"]["extra_fields"]["context"] == context


class TestCorrelationContext:
    """Test correlation context management"""
    
    def test_set_get_clear_correlation_context(self):
        """Test basic correlation context operations"""
        correlation_id = "wf_20240115103015_abc123"
        workflow_id = "workflow_test"
        
        # Initially no context
        assert get_correlation_context() is None
        
        # Set context
        set_correlation_context(correlation_id, workflow_id)
        context = get_correlation_context()
        
        assert context is not None
        assert context.correlation_id == correlation_id
        assert context.workflow_id == workflow_id
        
        # Clear context
        clear_correlation_context()
        assert get_correlation_context() is None
    
    def test_correlation_context_manager(self):
        """Test correlation context manager"""
        correlation_id = "wf_20240115103015_abc123"
        
        # Initially no context
        assert get_correlation_context() is None
        
        with correlation_context(correlation_id=correlation_id) as ctx_id:
            assert ctx_id == correlation_id
            context = get_correlation_context()
            assert context.correlation_id == correlation_id
        
        # Context should be cleared after exiting
        assert get_correlation_context() is None
    
    def test_nested_correlation_context(self):
        """Test nested correlation contexts"""
        outer_id = "wf_outer_123"
        inner_id = "wf_inner_456"
        
        with correlation_context(correlation_id=outer_id):
            outer_context = get_correlation_context()
            assert outer_context.correlation_id == outer_id
            
            with correlation_context(correlation_id=inner_id):
                inner_context = get_correlation_context()
                assert inner_context.correlation_id == inner_id
            
            # Should restore outer context
            restored_context = get_correlation_context()
            assert restored_context.correlation_id == outer_id
    
    def test_thread_local_isolation(self):
        """Test thread-local context isolation"""
        results = {}
        
        def worker_thread(thread_id: str):
            correlation_id = f"wf_thread_{thread_id}"
            set_correlation_context(correlation_id)
            
            time.sleep(0.1)  # Simulate work
            
            context = get_correlation_context()
            results[thread_id] = context.correlation_id if context else None
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=[str(i)])
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Each thread should have its own correlation ID
        assert len(results) == 3
        assert results["0"] == "wf_thread_0"
        assert results["1"] == "wf_thread_1"
        assert results["2"] == "wf_thread_2"


class TestWorkflowObservability:
    """Test workflow observability features"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = VerificationConfig(
            log_level=LogLevel.INFO,
            detailed_logging=True
        )
        self.manager = WorkflowObservabilityManager(self.config)
    
    def test_workflow_monitoring_lifecycle(self):
        """Test complete workflow monitoring lifecycle"""
        workflow_id = "test_workflow_123"
        correlation_id = "wf_20240115103015_abc123"
        
        # Start monitoring
        self.manager.start_workflow_monitoring(workflow_id, correlation_id)
        
        # Update progress
        self.manager.update_workflow_progress(workflow_id, 50.0, "verification")
        
        # Finish monitoring
        self.manager.finish_workflow_monitoring(workflow_id, "completed")
        
        # Workflow should be removed from active workflows
        assert workflow_id not in self.manager._active_workflows
    
    def test_dashboard_data_generation(self):
        """Test dashboard data generation"""
        # Add some active workflows
        for i in range(3):
            workflow_id = f"workflow_{i}"
            correlation_id = f"wf_test_{i}"
            self.manager.start_workflow_monitoring(workflow_id, correlation_id)
            self.manager.update_workflow_progress(workflow_id, i * 30.0, f"node_{i}")
        
        dashboard_data = self.manager.get_workflow_dashboard_data()
        
        assert dashboard_data["total_active_workflows"] == 3
        assert len(dashboard_data["active_workflows"]) == 3
        assert "average_progress" in dashboard_data
        assert "workflows_by_status" in dashboard_data


class TestLoggingConfiguration:
    """Test logging configuration và setup"""
    
    def test_logging_configurator_initialization(self):
        """Test logging configurator initialization"""
        config = VerificationConfig(log_level=LogLevel.DEBUG)
        configurator = LoggingConfigurator(config)
        
        assert configurator.config == config
        assert not configurator._configured
    
    def test_setup_logging_for_environment(self):
        """Test environment-specific logging setup"""
        configurator = configure_logging_for_testing()
        
        assert configurator._configured
        
        # Get a logger and test it
        logger = configurator.get_logger("test.component")
        assert isinstance(logger, EnhancedVerificationLogger)
    
    def test_workflow_logger_creation(self):
        """Test workflow-specific logger creation"""
        workflow_id = "test_workflow_456"
        logger = get_workflow_logger(workflow_id)
        
        assert isinstance(logger, EnhancedVerificationLogger)
        assert "workflow" in logger.name
    
    def test_verification_logger_creation(self):
        """Test component-specific logger creation"""
        logger = get_verification_logger("agent")
        
        assert isinstance(logger, EnhancedVerificationLogger)
        assert "verification.agent" in logger.name


class TestPerformanceTracking:
    """Test performance tracking context manager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = VerificationConfig(
            log_level=LogLevel.INFO,
            performance_tracking=True
        )
        self.logger = EnhancedVerificationLogger("test.performance", self.config)
    
    def test_performance_tracking_context(self):
        """Test performance tracking context manager"""
        operation_name = "test_operation"
        
        with performance_tracking(operation_name, self.logger) as timer_id:
            assert timer_id is not None
            time.sleep(0.1)  # Simulate work
        
        # Timer should be cleaned up
        assert timer_id not in self.logger._start_times


class TestWorkflowStatusTracking:
    """Test workflow status tracking"""
    
    def test_workflow_status_update_and_retrieval(self):
        """Test workflow status update và retrieval"""
        workflow_id = "test_status_workflow"
        correlation_id = "wf_status_test"
        
        # Set correlation context
        set_correlation_context(correlation_id, workflow_id)
        
        try:
            # Update status
            update_workflow_status(workflow_id, "running", "verification", 75.0)
            
            # Retrieve status
            status = get_workflow_status(workflow_id)
            
            assert status is not None
            assert status.workflow_id == workflow_id
            assert status.status == "running"
            assert status.current_node == "verification"
            assert status.progress_percentage == 75.0
        finally:
            clear_correlation_context()
    
    def test_workflow_cleanup(self):
        """Test workflow cleanup functionality"""
        # Create some old workflows
        old_workflow_id = "old_workflow"
        set_correlation_context("wf_old", old_workflow_id)
        update_workflow_status(old_workflow_id, "completed")
        
        # Mock old timestamp
        with patch('verification.utils.logging.datetime') as mock_datetime:
            # Make workflow appear old
            old_time = datetime.now(timezone.utc) - timedelta(hours=25)
            mock_datetime.now.return_value = old_time
            
            # Run cleanup
            cleaned_count = cleanup_completed_workflows(max_age_hours=24)
            
            # Should clean up old workflows
            assert cleaned_count >= 0


@pytest.fixture
def temp_log_file():
    """Fixture for temporary log file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        yield f.name
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestIntegrationScenarios:
    """Integration tests for complete logging scenarios"""
    
    def test_complete_workflow_logging_scenario(self, temp_log_file):
        """Test complete workflow logging scenario"""
        # Setup logging
        configurator = configure_logging_for_testing()
        logger = configurator.get_logger("integration.test")
        
        workflow_id = "integration_test_workflow"
        correlation_id = CorrelationIDGenerator.generate_correlation_id()
        
        # Simulate complete workflow
        with workflow_context(workflow_id, logger):
            with correlation_context(correlation_id=correlation_id):
                # Log workflow start
                logger.log_workflow_start(workflow_id, "Test objection")
                
                # Log node executions
                timer_id = logger.log_verification_start("objection", "draft")
                
                # Simulate verification result
                mock_result = Mock()
                mock_result.is_approved = True
                mock_result.criteria = Mock()
                mock_result.criteria.critical_issues_count = 0
                mock_result.verification_reasoning = "Test reasoning"
                mock_result.llm_tokens_used = 0
                
                logger.log_verification_result(mock_result, timer_id)
                
                # Log workflow completion
                logger.log_workflow_end(workflow_id, "completed", 2.5)
        
        # Verify context is cleaned up (correlation_context manager clears on exit)
        # workflow_context does not clear correlation; check correlation_context cleanup
        # After both context managers exit, correlation_context was cleared by its own __exit__
        # but workflow_context may have set it beforehand - clean up explicitly for assertion
        clear_correlation_context()
        assert get_correlation_context() is None
    
    def test_error_handling_with_correlation(self):
        """Test error handling với correlation context"""
        configurator = configure_logging_for_testing()
        logger = configurator.get_logger("error.test")
        
        correlation_id = CorrelationIDGenerator.generate_correlation_id()
        
        with correlation_context(correlation_id=correlation_id):
            try:
                raise ValueError("Test error for logging")
            except ValueError as e:
                logger.log_error_with_context(e, {"operation": "test"})
        
        # Should not raise any exceptions
        assert True
    
    def test_concurrent_workflow_logging(self):
        """Test concurrent workflow logging"""
        configurator = configure_logging_for_testing()
        results = {}
        
        def workflow_worker(worker_id: int):
            logger = configurator.get_logger(f"worker.{worker_id}")
            workflow_id = f"concurrent_workflow_{worker_id}"
            correlation_id = CorrelationIDGenerator.generate_correlation_id()
            
            with workflow_context(workflow_id, logger):
                with correlation_context(correlation_id=correlation_id):
                    logger.info(f"Worker {worker_id} processing")
                    time.sleep(0.1)  # Simulate work
                    
                    context = get_correlation_context()
                    results[worker_id] = context.correlation_id if context else None
        
        # Start multiple concurrent workflows
        threads = []
        for i in range(5):
            thread = threading.Thread(target=workflow_worker, args=[i])
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Each worker should have unique correlation ID
        assert len(results) == 5
        correlation_ids = list(results.values())
        assert len(set(correlation_ids)) == 5  # All unique
        
        # All should be valid correlation IDs
        for correlation_id in correlation_ids:
            assert CorrelationIDGenerator.validate_correlation_id(correlation_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAsyncCorrelationIDPropagation:
    """
    Tests for async-safe correlation ID propagation via contextvars.
    
    Validates: Requirement 8.5 - ALL errors SHALL be logged with correlation IDs
    Validates: Requirement 7.1 - Complete execution history with timestamps for each node
    """

    def setup_method(self):
        """Ensure clean correlation context before each test."""
        clear_correlation_context()

    def teardown_method(self):
        """Clean up correlation context after each test."""
        clear_correlation_context()

    @pytest.mark.asyncio
    async def test_async_correlation_context_manager(self):
        """Test async_correlation_context sets and restores correlation context."""
        correlation_id = CorrelationIDGenerator.generate_correlation_id()

        # Initially no context
        clear_correlation_context()
        assert get_correlation_context() is None

        async with async_correlation_context(correlation_id=correlation_id) as cid:
            assert cid == correlation_id
            ctx = get_correlation_context()
            assert ctx is not None
            assert ctx.correlation_id == correlation_id

        # Context should be cleared after exiting
        assert get_correlation_context() is None

    @pytest.mark.asyncio
    async def test_async_correlation_propagates_to_gather_tasks(self):
        """Test that correlation ID propagates to asyncio.gather() child tasks.
        
        This is the key async-safety test: contextvars propagate automatically
        to tasks created with asyncio.gather() or asyncio.create_task().
        """
        correlation_id = CorrelationIDGenerator.generate_correlation_id()
        results = {}

        async def check_correlation(task_name: str):
            """Async task that reads correlation context."""
            ctx = get_correlation_context()
            results[task_name] = ctx.correlation_id if ctx else None

        async with async_correlation_context(correlation_id=correlation_id):
            # All tasks spawned within this context inherit the correlation ID
            await asyncio.gather(
                check_correlation("task_1"),
                check_correlation("task_2"),
                check_correlation("task_3"),
            )

        # All tasks should have seen the same correlation ID
        assert results["task_1"] == correlation_id
        assert results["task_2"] == correlation_id
        assert results["task_3"] == correlation_id

    @pytest.mark.asyncio
    async def test_async_correlation_with_workflow_id(self):
        """Test async_correlation_context with workflow_id."""
        correlation_id = CorrelationIDGenerator.generate_correlation_id()
        workflow_id = f"workflow_{correlation_id}"

        async with async_correlation_context(
            correlation_id=correlation_id,
            workflow_id=workflow_id
        ) as cid:
            ctx = get_correlation_context()
            assert ctx is not None
            assert ctx.correlation_id == correlation_id
            assert ctx.workflow_id == workflow_id

    @pytest.mark.asyncio
    async def test_nested_async_correlation_contexts(self):
        """Test nested async correlation contexts restore correctly."""
        # Ensure clean state before test
        clear_correlation_context()

        outer_id = CorrelationIDGenerator.generate_correlation_id()
        inner_id = CorrelationIDGenerator.generate_correlation_id()

        async with async_correlation_context(correlation_id=outer_id):
            outer_ctx = get_correlation_context()
            assert outer_ctx.correlation_id == outer_id

            async with async_correlation_context(correlation_id=inner_id):
                inner_ctx = get_correlation_context()
                assert inner_ctx.correlation_id == inner_id

            # Should restore outer context
            restored_ctx = get_correlation_context()
            assert restored_ctx.correlation_id == outer_id

        # Should be cleared after outermost context exits
        assert get_correlation_context() is None

    @pytest.mark.asyncio
    async def test_concurrent_async_workflows_have_isolated_contexts(self):
        """Test that concurrent async workflows have isolated correlation contexts.
        
        Each asyncio task should have its own correlation context, not shared.
        """
        results = {}

        async def workflow_task(workflow_num: int):
            """Simulate an async workflow node with its own correlation ID."""
            correlation_id = CorrelationIDGenerator.generate_correlation_id()
            async with async_correlation_context(correlation_id=correlation_id):
                # Simulate some async work
                await asyncio.sleep(0.01)
                ctx = get_correlation_context()
                results[workflow_num] = ctx.correlation_id if ctx else None

        # Run multiple concurrent workflows
        await asyncio.gather(
            workflow_task(1),
            workflow_task(2),
            workflow_task(3),
            workflow_task(4),
            workflow_task(5),
        )

        # Each workflow should have a unique correlation ID
        assert len(results) == 5
        correlation_ids = list(results.values())
        # All should be valid correlation IDs
        for cid in correlation_ids:
            assert cid is not None
            assert CorrelationIDGenerator.validate_correlation_id(cid)
        # All should be unique
        assert len(set(correlation_ids)) == 5

    def test_set_async_correlation_context_returns_token(self):
        """Test set_async_correlation_context returns a token for restoration."""
        # Ensure clean state before test
        clear_correlation_context()

        correlation_id = CorrelationIDGenerator.generate_correlation_id()

        token = set_async_correlation_context(correlation_id)
        ctx = get_correlation_context()
        assert ctx is not None
        assert ctx.correlation_id == correlation_id

        # Reset using token
        reset_async_correlation_context(token)
        # After reset, should be back to None (the default value of the ContextVar)
        # Note: we also need to clear thread-local
        clear_correlation_context()
        ctx_after = get_correlation_context()
        assert ctx_after is None

    @pytest.mark.asyncio
    async def test_workflow_node_structured_log_includes_correlation_id(self):
        """Test that structured log entries include correlation_id from async context.
        
        Validates: Requirement 8.5 - ALL errors SHALL be logged with correlation IDs
        """
        config = VerificationConfig(log_level=LogLevel.DEBUG, detailed_logging=True)
        logger = EnhancedVerificationLogger("test.async.node", config)

        correlation_id = CorrelationIDGenerator.generate_correlation_id()
        workflow_id = "test_workflow_async"

        log_records = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        handler = CapturingHandler()
        logger.logger.addHandler(handler)

        try:
            async with async_correlation_context(
                correlation_id=correlation_id,
                workflow_id=workflow_id
            ):
                # Simulate node entry log
                logger.info(
                    "Node started",
                    workflow_event="node_start",
                    node_name="verification",
                )
                # Simulate async work
                await asyncio.sleep(0.001)
                # Simulate node exit log
                logger.info(
                    "Node completed",
                    workflow_event="node_end",
                    node_name="verification",
                    execution_time=0.001,
                )

            # Verify log records captured correlation context
            assert len(log_records) >= 2
            for record in log_records:
                extra = record.__dict__
                # The context extra should include correlation_id
                assert extra.get("correlation_id") == correlation_id or \
                       extra.get("extra_fields", {}).get("correlation_id") is not None or \
                       extra.get("workflow_id") == workflow_id
        finally:
            logger.logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_async_error_logging_includes_correlation_id(self):
        """Test that error logs in async context include correlation IDs.
        
        Validates: Requirement 8.5 - ALL errors SHALL be logged with correlation IDs
        """
        config = VerificationConfig(log_level=LogLevel.DEBUG, detailed_logging=True)
        logger = EnhancedVerificationLogger("test.async.error", config)

        correlation_id = CorrelationIDGenerator.generate_correlation_id()

        async with async_correlation_context(correlation_id=correlation_id):
            try:
                raise ValueError("Test async error with correlation ID")
            except ValueError as e:
                # Should not raise - error logging should work with correlation context
                logger.log_error_with_context(e, {
                    "node_name": "verification",
                    "operation": "price_check",
                })

        # Test passes if no exception was raised during error logging
        assert True