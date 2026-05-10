"""
Execution Tracking Models cho Workflow Observability

Performance metrics và execution logging cho LangGraph StateGraph:
- ExecutionStep: Individual node execution tracking với correlation IDs
- WorkflowMetrics: Overall workflow performance analysis với comprehensive tracking
- WorkflowExecutionLog: Complete workflow execution logging
- Structured logging với correlation IDs cho distributed tracing
- Performance tracking cho optimization và debugging
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Literal, Union
from datetime import datetime
from enum import Enum
import uuid


class ExecutionStatus(str, Enum):
    """Execution status cho workflow steps"""
    SUCCESS = "success"
    FAILED = "failed" 
    RETRY = "retry"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ExecutionStep(BaseModel):
    """
    Individual workflow step execution tracking với comprehensive observability
    
    Structured logging cho each StateGraph node execution
    với performance metrics, correlation IDs, và error details.
    Supports distributed tracing và workflow debugging.
    """
    
    # Basic Execution Info
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp của execution"
    )
    node_name: str = Field(description="Tên StateGraph node")
    execution_time: float = Field(ge=0.0, description="Thời gian thực thi (seconds)")
    status: ExecutionStatus = Field(description="Execution status")
    
    # Input/Output Summaries (truncated for performance)
    input_summary: str = Field(
        max_length=200, 
        description="Tóm tắt input state"
    )
    output_summary: str = Field(
        max_length=200,
        description="Tóm tắt output state"
    )
    
    # Error Handling
    error_details: Optional[str] = Field(
        default=None,
        description="Chi tiết lỗi nếu execution failed"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Loại lỗi (timeout, api_error, validation_error, etc.)"
    )
    
    # Performance Metrics với Enhanced Tracking
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metrics cho node execution"
    )
    
    # Resource Usage Tracking
    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Memory usage during execution (MB)"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage during execution (%)"
    )
    
    # Correlation Tracking cho Distributed Tracing
    correlation_id: str = Field(
        default_factory=lambda: f"exec_{uuid.uuid4().hex[:8]}",
        description="Correlation ID cho distributed tracing"
    )
    parent_correlation_id: Optional[str] = Field(
        default=None,
        description="Parent workflow correlation ID"
    )
    
    # StateGraph Integration
    workflow_id: Optional[str] = Field(
        default=None,
        description="StateGraph workflow instance ID"
    )
    step_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Step index trong workflow execution"
    )
    
    # LLM Integration Metrics
    llm_tokens_input: Optional[int] = Field(
        default=None,
        ge=0,
        description="Input tokens sent to LLM"
    )
    llm_tokens_output: Optional[int] = Field(
        default=None,
        ge=0,
        description="Output tokens received from LLM"
    )
    llm_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated LLM cost (USD)"
    )
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        """Auto-generate timestamp if not provided"""
        if v is None:
            return datetime.now().isoformat()
        return v
    
    @validator('correlation_id', pre=True, always=True)
    def ensure_correlation_id(cls, v):
        """Ensure correlation ID is always present"""
        if not v:
            return f"exec_{uuid.uuid4().hex[:8]}"
        return v
    
    def is_successful(self) -> bool:
        """Check if execution was successful"""
        return self.status == ExecutionStatus.SUCCESS
    
    def is_failed(self) -> bool:
        """Check if execution failed"""
        return self.status in [ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT]
    
    def get_total_llm_tokens(self) -> int:
        """Get total LLM tokens used"""
        input_tokens = self.llm_tokens_input or 0
        output_tokens = self.llm_tokens_output or 0
        return input_tokens + output_tokens
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add custom metric to execution step"""
        self.metrics[key] = value
    
    def get_performance_summary(self) -> str:
        """Generate performance summary for this step"""
        return f"""
Step: {self.node_name} | Status: {self.status.value}
Time: {self.execution_time:.2f}s | Tokens: {self.get_total_llm_tokens()}
Memory: {self.memory_usage_mb or 'N/A'}MB | CPU: {self.cpu_usage_percent or 'N/A'}%
Correlation: {self.correlation_id}
        """.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:15.123Z",
                "node_name": "verification",
                "execution_time": 2.5,
                "status": "success",
                "input_summary": "draft: 'iPhone 15 Pro Max có giá...'",
                "output_summary": "verification_result: PASS (no issues)",
                "error_details": None,
                "error_type": None,
                "metrics": {
                    "db_queries": 3,
                    "cache_hits": 2,
                    "issues_found": 0
                },
                "memory_usage_mb": 45.2,
                "cpu_usage_percent": 12.5,
                "correlation_id": "exec_abc12345",
                "parent_correlation_id": "wf_xyz789",
                "workflow_id": "wf_20240115_103015",
                "step_index": 1,
                "llm_tokens_input": 850,
                "llm_tokens_output": 400,
                "llm_cost_usd": 0.0125
            }
        }


class WorkflowMetrics(BaseModel):
    """
    Comprehensive workflow performance metrics với advanced analytics
    
    Aggregate metrics cho entire workflow execution
    với efficiency analysis, cost tracking, và performance optimization insights.
    Supports real-time monitoring và historical analysis.
    """
    
    # Timing Metrics với Enhanced Granularity
    total_execution_time: float = Field(ge=0.0, description="Total workflow time (seconds)")
    average_step_time: float = Field(
        default=0.0,
        ge=0.0, 
        description="Average step execution time"
    )
    min_step_time: float = Field(
        default=0.0,
        ge=0.0, 
        description="Fastest step execution time"
    )
    max_step_time: float = Field(
        default=0.0,
        ge=0.0, 
        description="Slowest step execution time"
    )
    
    # Retry & Success Metrics với Detailed Breakdown
    total_retries: int = Field(ge=0, description="Total retry attempts")
    total_steps: int = Field(ge=0, description="Total steps executed")
    successful_steps: int = Field(ge=0, description="Successfully completed steps")
    failed_steps: int = Field(ge=0, description="Failed steps")
    timeout_steps: int = Field(ge=0, description="Steps that timed out")
    
    # Node Execution Tracking với Performance Analysis
    nodes_executed: List[str] = Field(description="List of executed node names")
    node_execution_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Execution count per node type"
    )
    node_average_times: Dict[str, float] = Field(
        default_factory=dict,
        description="Average execution time per node type"
    )
    
    # Success Rate Calculation với Granular Metrics
    success_rate: float = Field(
        default=0.0,
        ge=0.0, 
        le=1.0, 
        description="Overall success rate (0-1)"
    )
    retry_success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Success rate after retries (0-1)"
    )
    
    # Issue Detection Metrics với Severity Breakdown
    critical_issues_found: int = Field(ge=0, description="Number of critical issues")
    major_issues_found: int = Field(ge=0, description="Number of major issues")
    minor_issues_found: int = Field(ge=0, description="Number of minor issues")
    total_issues_found: int = Field(ge=0, description="Total issues found")
    
    # Resource Usage với Detailed Tracking
    llm_tokens_used: int = Field(ge=0, description="Total LLM tokens consumed")
    llm_tokens_input: int = Field(ge=0, description="Total input tokens")
    llm_tokens_output: int = Field(ge=0, description="Total output tokens")
    cost_estimate: float = Field(ge=0.0, description="Estimated cost (USD)")
    
    # Memory & CPU Metrics
    peak_memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Peak memory usage (MB)"
    )
    average_cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Average CPU usage (%)"
    )
    
    # Cache Performance với Hit Rate Analysis
    cache_hits: int = Field(ge=0, description="Cache hit count")
    cache_misses: int = Field(ge=0, description="Cache miss count")
    cache_hit_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cache hit rate (0-1)"
    )
    
    # Database & External Service Metrics
    db_queries_count: int = Field(ge=0, description="Total database queries")
    external_api_calls: int = Field(ge=0, description="External API calls made")
    network_latency_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Average network latency (ms)"
    )
    
    # Workflow Quality Metrics
    verification_pass_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Verification pass rate (0-1)"
    )
    escalation_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Human escalation rate (0-1)"
    )
    
    @validator('success_rate', always=True)
    def calculate_success_rate(cls, v, values):
        """Auto-calculate success rate from step counts"""
        total = values.get('total_steps', 0)
        successful = values.get('successful_steps', 0)
        
        if total == 0:
            return 0.0
        
        return successful / total
    
    @validator('average_step_time', always=True)
    def calculate_average_step_time(cls, v, values):
        """Auto-calculate average step time"""
        total_time = values.get('total_execution_time', 0.0)
        total_steps = values.get('total_steps', 0)
        
        if total_steps == 0:
            return 0.0
            
        return total_time / total_steps
    
    @validator('cache_hit_rate', always=True)
    def calculate_cache_hit_rate(cls, v, values):
        """Auto-calculate cache hit rate"""
        hits = values.get('cache_hits', 0)
        misses = values.get('cache_misses', 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return hits / total
    
    @property
    def efficiency_score(self) -> float:
        """
        Workflow efficiency: success_rate / (retries + 1)
        
        Higher score = better efficiency (fewer retries needed)
        """
        return self.success_rate / (self.total_retries + 1)
    
    @property
    def performance_grade(self) -> str:
        """
        Performance grade based on efficiency và success rate
        
        A: >90% success, <2 avg retries
        B: >80% success, <3 avg retries  
        C: >70% success, <4 avg retries
        D: >60% success, <5 avg retries
        F: <60% success or >5 avg retries
        """
        if self.success_rate >= 0.9 and self.total_retries < 2:
            return "A"
        elif self.success_rate >= 0.8 and self.total_retries < 3:
            return "B"
        elif self.success_rate >= 0.7 and self.total_retries < 4:
            return "C"
        elif self.success_rate >= 0.6 and self.total_retries < 5:
            return "D"
        else:
            return "F"
    
    @property
    def cost_per_success(self) -> float:
        """Cost per successful workflow execution"""
        if self.successful_steps == 0:
            return 0.0
        return self.cost_estimate / self.successful_steps
    
    def get_performance_summary(self) -> str:
        """Generate comprehensive performance summary"""
        return f"""
🚀 WORKFLOW PERFORMANCE SUMMARY
⏱️  Total Time: {self.total_execution_time:.2f}s (avg: {self.average_step_time:.2f}s/step)
✅ Success Rate: {self.success_rate:.1%} (Grade: {self.performance_grade})
🔄 Retries: {self.total_retries} | Efficiency: {self.efficiency_score:.2f}
💰 Cost: ${self.cost_estimate:.4f} (${self.cost_per_success:.4f}/success)
🎯 Cache Hit Rate: {self.cache_hit_rate:.1%}
🔍 Issues: {self.critical_issues_found}C/{self.major_issues_found}M/{self.minor_issues_found}m
📊 Verification Pass: {self.verification_pass_rate:.1%} | Escalation: {self.escalation_rate:.1%}
💾 Memory Peak: {self.peak_memory_usage_mb or 'N/A'}MB | CPU Avg: {self.average_cpu_usage_percent or 'N/A'}%
        """.strip()
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []
        
        if self.cache_hit_rate < 0.7:
            recommendations.append("🎯 Improve caching strategy - hit rate below 70%")
        
        if self.total_retries > self.successful_steps * 0.5:
            recommendations.append("🔄 Reduce retry rate - too many retries detected")
        
        if self.average_step_time > 5.0:
            recommendations.append("⚡ Optimize step execution - average time > 5s")
        
        if self.cost_per_success > 0.05:
            recommendations.append("💰 Optimize LLM usage - cost per success > $0.05")
        
        if self.escalation_rate > 0.2:
            recommendations.append("🚨 Review escalation triggers - rate > 20%")
        
        if not recommendations:
            recommendations.append("✨ Performance looks good! No major optimizations needed.")
        
        return recommendations
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_execution_time": 8.5,
                "average_step_time": 2.83,
                "min_step_time": 1.2,
                "max_step_time": 4.1,
                "total_retries": 1,
                "total_steps": 3,
                "successful_steps": 3,
                "failed_steps": 0,
                "timeout_steps": 0,
                "nodes_executed": ["research", "verification", "correction"],
                "node_execution_counts": {"research": 2, "verification": 1, "correction": 1},
                "node_average_times": {"research": 3.2, "verification": 2.5, "correction": 1.8},
                "success_rate": 1.0,
                "retry_success_rate": 1.0,
                "critical_issues_found": 0,
                "major_issues_found": 1,
                "minor_issues_found": 1,
                "total_issues_found": 2,
                "llm_tokens_used": 3500,
                "llm_tokens_input": 2100,
                "llm_tokens_output": 1400,
                "cost_estimate": 0.0175,
                "peak_memory_usage_mb": 128.5,
                "average_cpu_usage_percent": 15.2,
                "cache_hits": 5,
                "cache_misses": 2,
                "cache_hit_rate": 0.714,
                "db_queries_count": 8,
                "external_api_calls": 3,
                "network_latency_ms": 45.2,
                "verification_pass_rate": 0.67,
                "escalation_rate": 0.0
            }
        }


class WorkflowExecutionLog(BaseModel):
    """
    Complete workflow execution log với structured data và real-time tracking
    
    Comprehensive logging cho entire workflow execution
    với steps, metrics, correlation tracking, và StateGraph integration.
    Supports real-time monitoring, debugging, và performance analysis.
    """
    
    # Workflow Identification với Enhanced Tracking
    workflow_id: str = Field(description="Unique workflow identifier")
    correlation_id: str = Field(
        default_factory=lambda: f"wf_{uuid.uuid4().hex[:12]}",
        description="Correlation ID cho distributed tracing"
    )
    parent_workflow_id: Optional[str] = Field(
        default=None,
        description="Parent workflow ID for nested workflows"
    )
    
    # Execution Timeline với Precision Tracking
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Workflow start time"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Workflow end time"
    )
    last_update_time: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )
    
    # Execution Steps với Enhanced Tracking
    steps: List[ExecutionStep] = Field(
        default_factory=list,
        description="All execution steps"
    )
    current_step_index: int = Field(
        default=0,
        ge=0,
        description="Current step index"
    )
    
    # Performance Metrics với Real-time Updates
    metrics: WorkflowMetrics = Field(description="Aggregate performance metrics")
    
    # Final Status với Detailed Tracking
    final_status: Literal[
        "running",
        "completed", 
        "failed", 
        "escalated", 
        "timeout",
        "cancelled"
    ] = Field(
        default="running",
        description="Final workflow status"
    )
    
    # Configuration Used với Versioning
    config_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration snapshot at execution time"
    )
    config_version: Optional[str] = Field(
        default=None,
        description="Configuration version used"
    )
    
    # StateGraph Integration
    graph_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current StateGraph state snapshot"
    )
    node_history: List[str] = Field(
        default_factory=list,
        description="History of executed nodes"
    )
    
    # Error Tracking với Detailed Context
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All errors encountered during execution"
    )
    warnings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All warnings generated during execution"
    )
    
    # User Context
    user_id: Optional[str] = Field(
        default=None,
        description="User who initiated the workflow"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for user tracking"
    )
    
    def add_step(self, step: ExecutionStep) -> None:
        """Add execution step to log với automatic indexing"""
        step.step_index = len(self.steps)
        step.parent_correlation_id = self.correlation_id
        step.workflow_id = self.workflow_id
        self.steps.append(step)
        self.current_step_index = len(self.steps) - 1
        self.last_update_time = datetime.now()
        
        # Update node history
        if step.node_name not in self.node_history:
            self.node_history.append(step.node_name)
    
    def add_error(self, error_type: str, error_message: str, 
                  node_name: Optional[str] = None, **kwargs) -> None:
        """Add error to tracking với structured format"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "node_name": node_name,
            "step_index": self.current_step_index,
            "correlation_id": f"err_{uuid.uuid4().hex[:8]}",
            **kwargs
        }
        self.errors.append(error_entry)
        self.last_update_time = datetime.now()
    
    def add_warning(self, warning_type: str, warning_message: str,
                   node_name: Optional[str] = None, **kwargs) -> None:
        """Add warning to tracking với structured format"""
        warning_entry = {
            "timestamp": datetime.now().isoformat(),
            "warning_type": warning_type,
            "warning_message": warning_message,
            "node_name": node_name,
            "step_index": self.current_step_index,
            "correlation_id": f"warn_{uuid.uuid4().hex[:8]}",
            **kwargs
        }
        self.warnings.append(warning_entry)
        self.last_update_time = datetime.now()
    
    def update_status(self, status: str) -> None:
        """Update workflow status với timestamp"""
        self.final_status = status
        self.last_update_time = datetime.now()
        if status in ["completed", "failed", "escalated", "timeout", "cancelled"]:
            self.end_time = datetime.now()
    
    def get_failed_steps(self) -> List[ExecutionStep]:
        """Get all failed execution steps"""
        return [step for step in self.steps if step.is_failed()]
    
    def get_successful_steps(self) -> List[ExecutionStep]:
        """Get all successful execution steps"""
        return [step for step in self.steps if step.is_successful()]
    
    def get_total_execution_time(self) -> float:
        """Calculate total execution time"""
        if self.end_time is None:
            end_time = datetime.now()
        else:
            end_time = self.end_time
        return (end_time - self.start_time).total_seconds()
    
    def get_current_node(self) -> Optional[str]:
        """Get currently executing node"""
        if self.steps and self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index].node_name
        return None
    
    def is_running(self) -> bool:
        """Check if workflow is currently running"""
        return self.final_status == "running"
    
    def is_completed(self) -> bool:
        """Check if workflow completed successfully"""
        return self.final_status == "completed"
    
    def is_failed(self) -> bool:
        """Check if workflow failed"""
        return self.final_status in ["failed", "timeout", "cancelled"]
    
    def get_execution_summary(self) -> str:
        """Generate comprehensive execution summary"""
        duration = self.get_total_execution_time()
        return f"""
📋 WORKFLOW EXECUTION SUMMARY
🆔 ID: {self.workflow_id} | Correlation: {self.correlation_id}
⏱️  Duration: {duration:.2f}s | Status: {self.final_status.upper()}
📊 Steps: {len(self.steps)} | Errors: {len(self.errors)} | Warnings: {len(self.warnings)}
🔄 Nodes: {' → '.join(self.node_history)}
👤 User: {self.user_id or 'N/A'} | Session: {self.session_id or 'N/A'}
        """.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "wf_20240115_103015_abc123",
                "correlation_id": "wf_xyz789abc123",
                "parent_workflow_id": None,
                "start_time": "2024-01-15T10:30:15.000Z",
                "end_time": "2024-01-15T10:30:23.500Z",
                "last_update_time": "2024-01-15T10:30:23.500Z",
                "steps": [],
                "current_step_index": 2,
                "metrics": {},
                "final_status": "completed",
                "config_snapshot": {
                    "max_retries": 3,
                    "price_tolerance_percent": 1.0
                },
                "config_version": "v1.2.0",
                "graph_state": {"current_node": "verification"},
                "node_history": ["research", "verification"],
                "errors": [],
                "warnings": [],
                "user_id": "user_123",
                "session_id": "session_456"
            }
        }


class WorkflowTracker(BaseModel):
    """
    Real-time workflow tracking và monitoring
    
    Lightweight tracker cho active workflows với real-time updates.
    Supports concurrent workflow monitoring và performance alerts.
    """
    
    # Active Workflows Tracking
    active_workflows: Dict[str, WorkflowExecutionLog] = Field(
        default_factory=dict,
        description="Currently active workflows"
    )
    
    # Performance Monitoring
    total_workflows_processed: int = Field(
        default=0,
        ge=0,
        description="Total workflows processed"
    )
    successful_workflows: int = Field(
        default=0,
        ge=0,
        description="Successfully completed workflows"
    )
    failed_workflows: int = Field(
        default=0,
        ge=0,
        description="Failed workflows"
    )
    
    # Real-time Metrics
    current_load: int = Field(
        default=0,
        ge=0,
        description="Current number of active workflows"
    )
    peak_concurrent_workflows: int = Field(
        default=0,
        ge=0,
        description="Peak concurrent workflows"
    )
    
    # Performance Thresholds
    max_concurrent_workflows: int = Field(
        default=10,
        ge=1,
        description="Maximum allowed concurrent workflows"
    )
    performance_alert_threshold: float = Field(
        default=15.0,
        ge=0.0,
        description="Performance alert threshold (seconds)"
    )
    
    def start_workflow(self, workflow_log: WorkflowExecutionLog) -> None:
        """Start tracking a new workflow"""
        self.active_workflows[workflow_log.workflow_id] = workflow_log
        self.current_load = len(self.active_workflows)
        self.peak_concurrent_workflows = max(
            self.peak_concurrent_workflows, 
            self.current_load
        )
    
    def complete_workflow(self, workflow_id: str, status: str) -> Optional[WorkflowExecutionLog]:
        """Complete workflow tracking"""
        if workflow_id in self.active_workflows:
            workflow_log = self.active_workflows.pop(workflow_id)
            workflow_log.update_status(status)
            
            self.total_workflows_processed += 1
            if status == "completed":
                self.successful_workflows += 1
            else:
                self.failed_workflows += 1
            
            self.current_load = len(self.active_workflows)
            return workflow_log
        return None
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowExecutionLog]:
        """Get active workflow by ID"""
        return self.active_workflows.get(workflow_id)
    
    def is_overloaded(self) -> bool:
        """Check if system is overloaded"""
        return self.current_load > self.max_concurrent_workflows
    
    def get_slow_workflows(self) -> List[WorkflowExecutionLog]:
        """Get workflows exceeding performance threshold"""
        slow_workflows = []
        for workflow in self.active_workflows.values():
            if workflow.get_total_execution_time() > self.performance_alert_threshold:
                slow_workflows.append(workflow)
        return slow_workflows
    
    @property
    def success_rate(self) -> float:
        """Overall success rate"""
        if self.total_workflows_processed == 0:
            return 0.0
        return self.successful_workflows / self.total_workflows_processed
    
    def get_system_status(self) -> str:
        """Get system status summary"""
        return f"""
🖥️  SYSTEM STATUS
🔄 Active: {self.current_load}/{self.max_concurrent_workflows}
📊 Processed: {self.total_workflows_processed} (Success: {self.success_rate:.1%})
⚡ Peak Load: {self.peak_concurrent_workflows}
🐌 Slow Workflows: {len(self.get_slow_workflows())}
        """.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "active_workflows": {},
                "total_workflows_processed": 150,
                "successful_workflows": 135,
                "failed_workflows": 15,
                "current_load": 3,
                "peak_concurrent_workflows": 8,
                "max_concurrent_workflows": 10,
                "performance_alert_threshold": 15.0
            }
        }