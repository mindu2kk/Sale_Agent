"""
Test cases for ExecutionStep and WorkflowMetrics models

Tests the enhanced performance tracking models với comprehensive validation.
"""

import pytest
from datetime import datetime
from verification.models.execution import (
    ExecutionStep, 
    WorkflowMetrics, 
    WorkflowExecutionLog, 
    WorkflowTracker,
    ExecutionStatus
)


class TestExecutionStep:
    """Test ExecutionStep model với performance tracking"""
    
    def test_execution_step_creation(self):
        """Test basic ExecutionStep creation"""
        step = ExecutionStep(
            node_name="verification",
            execution_time=2.5,
            status=ExecutionStatus.SUCCESS,
            input_summary="test input",
            output_summary="test output"
        )
        
        assert step.node_name == "verification"
        assert step.execution_time == 2.5
        assert step.status == ExecutionStatus.SUCCESS
        assert step.is_successful()
        assert not step.is_failed()
        assert step.correlation_id.startswith("exec_")
        assert step.timestamp is not None
    
    def test_execution_step_with_llm_metrics(self):
        """Test ExecutionStep với LLM performance metrics"""
        step = ExecutionStep(
            node_name="research",
            execution_time=3.2,
            status=ExecutionStatus.SUCCESS,
            input_summary="objection analysis",
            output_summary="draft response",
            llm_tokens_input=500,
            llm_tokens_output=300,
            llm_cost_usd=0.008,
            memory_usage_mb=45.2,
            cpu_usage_percent=12.5
        )
        
        assert step.get_total_llm_tokens() == 800
        assert step.llm_cost_usd == 0.008
        assert step.memory_usage_mb == 45.2
        assert step.cpu_usage_percent == 12.5
    
    def test_execution_step_failed_status(self):
        """Test ExecutionStep với failed status"""
        step = ExecutionStep(
            node_name="verification",
            execution_time=1.5,
            status=ExecutionStatus.FAILED,
            input_summary="invalid input",
            output_summary="error occurred",
            error_details="Validation failed",
            error_type="validation_error"
        )
        
        assert step.is_failed()
        assert not step.is_successful()
        assert step.error_details == "Validation failed"
        assert step.error_type == "validation_error"
    
    def test_execution_step_add_metric(self):
        """Test adding custom metrics to ExecutionStep"""
        step = ExecutionStep(
            node_name="correction",
            execution_time=1.8,
            status=ExecutionStatus.SUCCESS,
            input_summary="correction input",
            output_summary="correction output"
        )
        
        step.add_metric("db_queries", 3)
        step.add_metric("cache_hits", 2)
        
        assert step.metrics["db_queries"] == 3
        assert step.metrics["cache_hits"] == 2


class TestWorkflowMetrics:
    """Test WorkflowMetrics model với comprehensive analytics"""
    
    def test_workflow_metrics_creation(self):
        """Test basic WorkflowMetrics creation"""
        metrics = WorkflowMetrics(
            total_execution_time=10.0,
            total_steps=3,
            successful_steps=2,
            failed_steps=1,
            timeout_steps=0,
            total_retries=1,
            nodes_executed=["research", "verification", "correction"],
            critical_issues_found=0,
            major_issues_found=1,
            minor_issues_found=2,
            total_issues_found=3,
            llm_tokens_used=1500,
            llm_tokens_input=900,
            llm_tokens_output=600,
            cost_estimate=0.025,
            cache_hits=5,
            cache_misses=2,
            db_queries_count=3,
            external_api_calls=2,
            verification_pass_rate=0.8,
            escalation_rate=0.1
        )
        
        assert metrics.total_execution_time == 10.0
        assert metrics.success_rate == pytest.approx(0.667, rel=1e-2)
        assert metrics.efficiency_score == pytest.approx(0.333, rel=1e-2)
        assert metrics.cache_hit_rate == pytest.approx(0.714, rel=1e-2)
    
    def test_performance_grade_calculation(self):
        """Test performance grade calculation"""
        # Grade A metrics
        metrics_a = WorkflowMetrics(
            total_execution_time=5.0,
            total_steps=2,
            successful_steps=2,
            failed_steps=0,
            timeout_steps=0,
            total_retries=0,
            nodes_executed=["research", "verification"],
            critical_issues_found=0,
            major_issues_found=0,
            minor_issues_found=0,
            total_issues_found=0,
            llm_tokens_used=1000,
            llm_tokens_input=600,
            llm_tokens_output=400,
            cost_estimate=0.015,
            cache_hits=10,
            cache_misses=1,
            db_queries_count=2,
            external_api_calls=1,
            verification_pass_rate=1.0,
            escalation_rate=0.0
        )
        
        assert metrics_a.performance_grade == "A"
        
        # Grade F metrics
        metrics_f = WorkflowMetrics(
            total_execution_time=20.0,
            total_steps=10,
            successful_steps=4,
            failed_steps=6,
            timeout_steps=2,
            total_retries=8,
            nodes_executed=["research", "verification", "correction"],
            critical_issues_found=3,
            major_issues_found=2,
            minor_issues_found=1,
            total_issues_found=6,
            llm_tokens_used=5000,
            llm_tokens_input=3000,
            llm_tokens_output=2000,
            cost_estimate=0.1,
            cache_hits=2,
            cache_misses=8,
            db_queries_count=15,
            external_api_calls=10,
            verification_pass_rate=0.3,
            escalation_rate=0.4
        )
        
        assert metrics_f.performance_grade == "F"
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations generation"""
        # Poor performance metrics
        metrics = WorkflowMetrics(
            total_execution_time=25.0,  # High execution time
            total_steps=3,
            successful_steps=1,
            failed_steps=2,
            timeout_steps=0,
            total_retries=5,  # High retry count
            nodes_executed=["research", "verification", "correction"],
            critical_issues_found=2,
            major_issues_found=3,
            minor_issues_found=1,
            total_issues_found=6,
            llm_tokens_used=8000,
            llm_tokens_input=5000,
            llm_tokens_output=3000,
            cost_estimate=0.2,  # High cost
            cache_hits=1,
            cache_misses=9,  # Poor cache performance
            db_queries_count=20,
            external_api_calls=15,
            verification_pass_rate=0.2,
            escalation_rate=0.6  # High escalation rate
        )
        
        recommendations = metrics.get_optimization_recommendations()
        
        # Should have multiple recommendations for poor performance
        assert len(recommendations) > 1
        assert any("caching" in rec.lower() for rec in recommendations)
        assert any("retry" in rec.lower() for rec in recommendations)
        assert any("escalation" in rec.lower() for rec in recommendations)


class TestWorkflowExecutionLog:
    """Test WorkflowExecutionLog model với real-time tracking"""
    
    def test_workflow_log_creation(self):
        """Test WorkflowExecutionLog creation"""
        metrics = WorkflowMetrics(
            total_execution_time=5.0,
            total_steps=2,
            successful_steps=2,
            failed_steps=0,
            timeout_steps=0,
            total_retries=0,
            nodes_executed=["research", "verification"],
            critical_issues_found=0,
            major_issues_found=0,
            minor_issues_found=0,
            total_issues_found=0,
            llm_tokens_used=1000,
            llm_tokens_input=600,
            llm_tokens_output=400,
            cost_estimate=0.015,
            cache_hits=5,
            cache_misses=1,
            db_queries_count=2,
            external_api_calls=1,
            verification_pass_rate=1.0,
            escalation_rate=0.0
        )
        
        log = WorkflowExecutionLog(
            workflow_id="test_workflow_123",
            metrics=metrics,
            user_id="test_user",
            session_id="test_session"
        )
        
        assert log.workflow_id == "test_workflow_123"
        assert log.user_id == "test_user"
        assert log.session_id == "test_session"
        assert log.correlation_id.startswith("wf_")
        assert log.is_running()
        assert not log.is_completed()
    
    def test_workflow_log_add_steps(self):
        """Test adding steps to workflow log"""
        metrics = WorkflowMetrics(
            total_execution_time=5.0,
            total_steps=1,
            successful_steps=1,
            failed_steps=0,
            timeout_steps=0,
            total_retries=0,
            nodes_executed=["research"],
            critical_issues_found=0,
            major_issues_found=0,
            minor_issues_found=0,
            total_issues_found=0,
            llm_tokens_used=500,
            llm_tokens_input=300,
            llm_tokens_output=200,
            cost_estimate=0.008,
            cache_hits=3,
            cache_misses=1,
            db_queries_count=1,
            external_api_calls=1,
            verification_pass_rate=1.0,
            escalation_rate=0.0
        )
        
        log = WorkflowExecutionLog(
            workflow_id="test_workflow_456",
            metrics=metrics
        )
        
        step = ExecutionStep(
            node_name="research",
            execution_time=2.5,
            status=ExecutionStatus.SUCCESS,
            input_summary="test input",
            output_summary="test output"
        )
        
        log.add_step(step)
        
        assert len(log.steps) == 1
        assert log.steps[0].step_index == 0
        assert log.steps[0].workflow_id == "test_workflow_456"
        assert log.steps[0].parent_correlation_id == log.correlation_id
        assert log.current_step_index == 0
    
    def test_workflow_log_error_tracking(self):
        """Test error and warning tracking"""
        metrics = WorkflowMetrics(
            total_execution_time=5.0,
            total_steps=1,
            successful_steps=0,
            failed_steps=1,
            timeout_steps=0,
            total_retries=1,
            nodes_executed=["verification"],
            critical_issues_found=1,
            major_issues_found=0,
            minor_issues_found=0,
            total_issues_found=1,
            llm_tokens_used=500,
            llm_tokens_input=300,
            llm_tokens_output=200,
            cost_estimate=0.008,
            cache_hits=2,
            cache_misses=2,
            db_queries_count=1,
            external_api_calls=1,
            verification_pass_rate=0.0,
            escalation_rate=1.0
        )
        
        log = WorkflowExecutionLog(
            workflow_id="test_workflow_error",
            metrics=metrics
        )
        
        log.add_error("validation_error", "Price mismatch detected", "verification")
        log.add_warning("performance_warning", "Slow response time", "research")
        
        assert len(log.errors) == 1
        assert len(log.warnings) == 1
        assert log.errors[0]["error_type"] == "validation_error"
        assert log.warnings[0]["warning_type"] == "performance_warning"


class TestWorkflowTracker:
    """Test WorkflowTracker model với real-time monitoring"""
    
    def test_workflow_tracker_creation(self):
        """Test WorkflowTracker creation"""
        tracker = WorkflowTracker()
        
        assert tracker.current_load == 0
        assert tracker.total_workflows_processed == 0
        assert tracker.success_rate == 0.0
        assert not tracker.is_overloaded()
    
    def test_workflow_tracking_lifecycle(self):
        """Test complete workflow tracking lifecycle"""
        tracker = WorkflowTracker()
        
        # Create workflow log
        metrics = WorkflowMetrics(
            total_execution_time=5.0,
            total_steps=2,
            successful_steps=2,
            failed_steps=0,
            timeout_steps=0,
            total_retries=0,
            nodes_executed=["research", "verification"],
            critical_issues_found=0,
            major_issues_found=0,
            minor_issues_found=0,
            total_issues_found=0,
            llm_tokens_used=1000,
            llm_tokens_input=600,
            llm_tokens_output=400,
            cost_estimate=0.015,
            cache_hits=5,
            cache_misses=1,
            db_queries_count=2,
            external_api_calls=1,
            verification_pass_rate=1.0,
            escalation_rate=0.0
        )
        
        log = WorkflowExecutionLog(
            workflow_id="tracked_workflow_123",
            metrics=metrics
        )
        
        # Start tracking
        tracker.start_workflow(log)
        
        assert tracker.current_load == 1
        assert tracker.peak_concurrent_workflows == 1
        assert log.workflow_id in tracker.active_workflows
        
        # Complete workflow
        completed_log = tracker.complete_workflow(log.workflow_id, "completed")
        
        assert tracker.current_load == 0
        assert tracker.total_workflows_processed == 1
        assert tracker.successful_workflows == 1
        assert tracker.success_rate == 1.0
        assert completed_log is not None
        assert completed_log.final_status == "completed"
    
    def test_workflow_tracker_overload_detection(self):
        """Test overload detection"""
        tracker = WorkflowTracker(max_concurrent_workflows=2)
        
        # Create multiple workflows
        for i in range(3):
            metrics = WorkflowMetrics(
                total_execution_time=5.0,
                total_steps=1,
                successful_steps=1,
                failed_steps=0,
                timeout_steps=0,
                total_retries=0,
                nodes_executed=["research"],
                critical_issues_found=0,
                major_issues_found=0,
                minor_issues_found=0,
                total_issues_found=0,
                llm_tokens_used=500,
                llm_tokens_input=300,
                llm_tokens_output=200,
                cost_estimate=0.008,
                cache_hits=3,
                cache_misses=1,
                db_queries_count=1,
                external_api_calls=1,
                verification_pass_rate=1.0,
                escalation_rate=0.0
            )
            
            log = WorkflowExecutionLog(
                workflow_id=f"workflow_{i}",
                metrics=metrics
            )
            
            tracker.start_workflow(log)
            
            if i >= 2:  # Should be overloaded after 2 workflows
                assert tracker.is_overloaded()
            else:
                assert not tracker.is_overloaded()


if __name__ == "__main__":
    pytest.main([__file__])