"""
Unit Tests cho WorkflowState Pydantic Model

Tests validation rules, computed properties, và utility functions
cho enhanced WorkflowState implementation.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from verification.models.state import (
    WorkflowStateValidator,
    WorkflowConfig,
    create_initial_workflow_state,
    validate_workflow_state
)
from verification.models.verification import VerificationResult, RubricCriteria
from verification.models.execution import ExecutionStep, ExecutionStatus


class TestWorkflowStateValidator:
    """Test WorkflowStateValidator Pydantic model"""
    
    def test_minimal_valid_state(self):
        """Test minimal valid WorkflowState"""
        state_data = {
            "objection_text": "iPhone quá đắt, tại sao tôi nên mua?",
            "draft_response": "iPhone mang lại giá trị vượt trội với hệ sinh thái Apple tích hợp hoàn hảo, camera chất lượng cao và hiệu năng mạnh mẽ.",
            "research_reasoning": "Analyzed price objection and provided comprehensive value proposition comparison",
            "start_time": datetime.now().isoformat()
        }
        
        state = WorkflowStateValidator(**state_data)
        
        assert state.objection_text == state_data["objection_text"]
        assert state.workflow_status == "initialized"
        assert state.retry_count == 0
        assert state.max_retries == 3
        assert len(state.execution_log) == 0
        assert len(state.error_log) == 0
        assert state.workflow_id.startswith("wf_")
        assert state.correlation_id.startswith("corr_")
    
    def test_objection_text_validation(self):
        """Test objection_text length validation"""
        # Too short
        with pytest.raises(ValueError, match="at least 10 characters"):
            WorkflowStateValidator(
                objection_text="short",
                draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
                research_reasoning="This is valid reasoning that meets minimum length requirements",
                start_time=datetime.now().isoformat()
            )
        
        # Too long
        long_text = "x" * 5001
        with pytest.raises(ValueError, match="at most 5000 characters"):
            WorkflowStateValidator(
                objection_text=long_text,
                draft_response="This is a valid response that meets the minimum length requirement for testing purposes.", 
                research_reasoning="This is valid reasoning that meets minimum length requirements",
                start_time=datetime.now().isoformat()
            )
    
    def test_retry_count_validation(self):
        """Test retry_count vs max_retries validation"""
        with pytest.raises(ValueError, match="cannot exceed max_retries"):
            WorkflowStateValidator(
                objection_text="Valid objection text here for testing purposes",
                draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
                research_reasoning="This is valid reasoning that meets minimum length requirements for testing",
                start_time=datetime.now().isoformat(),
                retry_count=5,
                max_retries=3
            )
    
    def test_timestamp_validation(self):
        """Test start_time and end_time validation"""
        now = datetime.now()
        start_time = now.isoformat()
        end_time = (now + timedelta(seconds=10)).isoformat()
        
        state = WorkflowStateValidator(
            objection_text="Valid objection text here for testing purposes",
            draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
            research_reasoning="This is valid reasoning that meets minimum length requirements for testing",
            start_time=start_time,
            end_time=end_time
        )
        
        assert state.start_time == start_time
        assert state.end_time == end_time
        assert state.execution_duration_seconds == pytest.approx(10.0, rel=1e-1)
    
    def test_end_time_before_start_time_validation(self):
        """Test end_time must be after start_time"""
        now = datetime.now()
        start_time = now.isoformat()
        end_time = (now - timedelta(seconds=10)).isoformat()  # Before start
        
        with pytest.raises(ValueError, match="end_time must be after start_time"):
            WorkflowStateValidator(
                objection_text="Valid objection text here for testing purposes",
                draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
                research_reasoning="This is valid reasoning that meets minimum length requirements for testing",
                start_time=start_time,
                end_time=end_time
            )
    
    def test_resource_usage_validation(self):
        """Test resource_usage structure validation"""
        state = WorkflowStateValidator(
            objection_text="Valid objection text here for testing purposes",
            draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
            research_reasoning="This is valid reasoning that meets minimum length requirements for testing",
            start_time=datetime.now().isoformat(),
            resource_usage={
                "cpu_time_seconds": 2.5,
                "memory_peak_mb": 128.0,
                "llm_tokens_total": 1500,
                "llm_cost_usd": 0.025,
                "db_queries_count": 5,
                "cache_hits": 3,
                "cache_misses": 2
            }
        )
        
        assert state.resource_usage["cpu_time_seconds"] == 2.5
        assert state.resource_usage["llm_tokens_total"] == 1500
        assert state.cache_hit_rate == 0.6  # 3/(3+2)
    
    def test_error_log_validation(self):
        """Test error_log structure validation"""
        valid_error = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "LLM_TIMEOUT",
            "message": "LLM request timed out after 30 seconds"
        }
        
        state = WorkflowStateValidator(
            objection_text="Valid objection text here for testing purposes",
            draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
            research_reasoning="This is valid reasoning that meets minimum length requirements for testing",
            start_time=datetime.now().isoformat(),
            error_log=[valid_error]
        )
        
        assert len(state.error_log) == 1
        assert state.error_log[0]["error_type"] == "LLM_TIMEOUT"
    
    def test_computed_properties(self):
        """Test computed properties"""
        state = WorkflowStateValidator(
            objection_text="Valid objection text here for testing purposes",
            draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
            research_reasoning="This is valid reasoning that meets minimum length requirements for testing",
            start_time=datetime.now().isoformat(),
            workflow_status="approved"
        )
        
        assert state.is_terminal_state == True
        
        # Test non-terminal state
        state.workflow_status = "verifying"
        assert state.is_terminal_state == False
    
    def test_add_execution_step(self):
        """Test adding execution steps"""
        state = WorkflowStateValidator(
            objection_text="Valid objection text here for testing purposes",
            draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
            research_reasoning="This is valid reasoning that meets minimum length requirements for testing",
            start_time=datetime.now().isoformat()
        )
        
        step = ExecutionStep(
            timestamp=datetime.now().isoformat(),
            node_name="verification",
            execution_time=2.5,
            status=ExecutionStatus.SUCCESS,
            input_summary="draft response",
            output_summary="verification passed"
        )
        
        state.add_execution_step(step)
        assert len(state.execution_log) == 1
        assert state.execution_log[0].node_name == "verification"
    
    def test_add_error(self):
        """Test adding structured errors"""
        state = WorkflowStateValidator(
            objection_text="Valid objection text here for testing purposes",
            draft_response="This is a valid response that meets the minimum length requirement for testing purposes.",
            research_reasoning="This is valid reasoning that meets minimum length requirements for testing",
            start_time=datetime.now().isoformat()
        )
        
        state.add_error("LLM_ERROR", "API timeout", {"timeout_seconds": 30})
        
        assert len(state.error_log) == 1
        assert state.error_log[0]["error_type"] == "LLM_ERROR"
        assert state.error_log[0]["message"] == "API timeout"
        assert state.error_log[0]["details"]["timeout_seconds"] == 30


class TestWorkflowConfig:
    """Test WorkflowConfig Pydantic model"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = WorkflowConfig()
        
        assert config.price_tolerance_percent == 1.0
        assert config.policy_citation_required == True
        assert config.relevance_min_coverage == 0.7
        assert config.max_retries == 3
        assert config.parallel_verification == True
        assert config.early_termination == True
    
    def test_verification_weights_validation(self):
        """Test verification weights sum to 1.0"""
        # Valid weights
        config = WorkflowConfig(
            verification_weights={
                "price_accuracy": 0.5,
                "policy_authenticity": 0.3,
                "topic_relevance": 0.2
            }
        )
        assert sum(config.verification_weights.values()) == 1.0
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            WorkflowConfig(
                verification_weights={
                    "price_accuracy": 0.6,
                    "policy_authenticity": 0.3,
                    "topic_relevance": 0.2  # Sum = 1.1
                }
            )
    
    def test_computed_properties(self):
        """Test computed properties"""
        # High performance config
        config = WorkflowConfig(
            parallel_verification=True,
            early_termination=True,
            enable_llm_caching=True,
            max_concurrent_workflows=10
        )
        assert config.is_high_performance_mode == True
        
        # Strict verification config
        config = WorkflowConfig(
            price_tolerance_percent=0.5,
            policy_citation_required=True,
            relevance_min_coverage=0.8,
            critical_issue_escalation=True
        )
        assert config.is_strict_verification_mode == True
    
    def test_production_validation(self):
        """Test production validation warnings"""
        # Unsafe config
        config = WorkflowConfig(
            input_sanitization=False,
            audit_logging=False,
            price_tolerance_percent=10.0
        )
        
        warnings = config.validate_for_production()
        assert len(warnings) > 0
        assert any("security risk" in w for w in warnings)
        assert any("compliance risk" in w for w in warnings)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_initial_workflow_state(self):
        """Test creating initial workflow state"""
        objection = "iPhone quá đắt, tại sao tôi nên mua?"
        config = WorkflowConfig(max_retries=5)
        
        state = create_initial_workflow_state(
            objection_text=objection,
            config=config,
            customer_context={"segment": "price_sensitive"}
        )
        
        assert state["objection_text"] == objection
        assert state["workflow_status"] == "initialized"
        assert state["max_retries"] == 5
        assert state["customer_context"]["segment"] == "price_sensitive"
        assert state["workflow_id"].startswith("wf_")
        assert state["correlation_id"].startswith("corr_")
    
    def test_validate_workflow_state(self):
        """Test workflow state validation function"""
        state_dict = {
            "objection_text": "Valid objection text here for testing purposes",
            "draft_response": "This is a valid response that meets the minimum length requirement for testing purposes.",
            "research_reasoning": "This is valid reasoning that meets minimum length requirements for testing",
            "start_time": datetime.now().isoformat()
        }
        
        # Should not raise exception
        validated_state = validate_workflow_state(state_dict)
        assert isinstance(validated_state, WorkflowStateValidator)
        assert validated_state.objection_text == state_dict["objection_text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])