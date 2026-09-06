"""
WorkflowState Pydantic Model cho LangGraph StateGraph

Enhanced shared state object chứa toàn bộ workflow context:
- Input objection và draft response với validation
- Verification results với binary decisions
- Self-correction feedback với structured issues  
- Execution tracking với performance metrics
- Complete observability và resource usage tracking
"""

from typing import TypedDict, List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from datetime import datetime
import uuid

from .verification import VerificationResult
from .execution import ExecutionStep, WorkflowMetrics


class WorkflowState(TypedDict):
    """
    LangGraph StateGraph shared state với enhanced type safety và validation
    
    Chứa complete workflow context từ input đến final output,
    hỗ trợ binary verification, structured issue tracking, và comprehensive observability.
    
    **Validates: Requirements 7.1** - Complete execution history tracking
    """
    
    # === INPUT CONTEXT ===
    objection_text: str  # Required: Customer objection input
    customer_context: Optional[Dict[str, Any]]  # Optional: Additional customer data
    
    # === RESEARCH AGENT OUTPUT ===  
    draft_response: str  # Required: Generated draft response
    tools_used: List[str]  # Tools executed by Research Agent
    research_reasoning: str  # Agent reasoning process
    research_sources: List[str]  # Data sources consulted
    
    # === VERIFICATION AGENT OUTPUT ===
    verification_result: Optional[VerificationResult]  # Binary verification result
    
    # === SELF-CORRECTION LOOP ===
    correction_feedback: Optional[str]  # Structured correction instructions
    retry_count: int  # Current retry attempt number
    max_retries: int  # Maximum allowed retries
    
    # === FINAL OUTPUT ===
    final_response: str  # Approved final response
    workflow_status: Literal[
        "initialized",    # Workflow created
        "researching",    # Research Agent executing
        "verifying",      # Verification Agent executing
        "correcting",     # Self-correction in progress
        "approved",       # Verification passed
        "escalated",      # Human escalation required
        "failed"          # Workflow failed
    ]
    
    # === EXECUTION TRACKING (Requirement 7.1) ===
    execution_log: List[ExecutionStep]  # Complete step-by-step execution history
    start_time: str  # Workflow start timestamp (ISO format)
    end_time: Optional[str]  # Workflow completion timestamp
    
    # === RESOURCE USAGE TRACKING (Requirement 7.1) ===
    resource_usage: Dict[str, Any]  # CPU, memory, token usage metrics
    
    # === ERROR HANDLING ===
    error_log: List[Dict[str, Any]]  # Structured error tracking
    
    # === CONFIGURATION ===
    config: Dict[str, Any]  # Workflow configuration snapshot
    
    # === OBSERVABILITY (Requirement 7.2) ===
    workflow_id: str  # Unique workflow identifier
    correlation_id: str  # Distributed tracing correlation ID


class WorkflowStateValidator(BaseModel):
    """
    Pydantic validator cho WorkflowState với comprehensive validation rules
    
    Provides runtime validation cho WorkflowState TypedDict để ensure data integrity
    và compliance với LangGraph StateGraph requirements.
    
    **Validates: Requirements 7.1, 7.2** - State management và observability
    """
    
    # === INPUT VALIDATION ===
    objection_text: str = Field(
        min_length=10,
        max_length=5000,
        description="Customer objection text (10-5000 characters)"
    )
    
    customer_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional customer context data"
    )
    
    # === RESEARCH AGENT OUTPUT VALIDATION ===
    draft_response: str = Field(
        min_length=0,  # Allow empty for initial state
        max_length=10000,
        description="Generated draft response (0-10000 characters, empty for initial state)"
    )
    
    tools_used: List[str] = Field(
        default_factory=list,
        description="List of tools executed by Research Agent"
    )
    
    research_reasoning: str = Field(
        min_length=0,  # Allow empty for initial state
        max_length=2000,
        description="Agent reasoning process (0-2000 characters, empty for initial state)"
    )
    
    research_sources: List[str] = Field(
        default_factory=list,
        description="Data sources consulted during research"
    )
    
    # === VERIFICATION RESULTS ===
    verification_result: Optional[VerificationResult] = Field(
        default=None,
        description="Binary verification result with issue breakdown"
    )
    
    # === SELF-CORRECTION VALIDATION ===
    correction_feedback: Optional[str] = Field(
        default=None,
        max_length=3000,
        description="Structured correction instructions (max 3000 chars)"
    )
    
    retry_count: int = Field(
        ge=0,
        le=10,
        default=0,
        description="Current retry attempt (0-10)"
    )
    
    max_retries: int = Field(
        ge=1,
        le=10,
        default=3,
        description="Maximum allowed retries (1-10)"
    )
    
    # === FINAL OUTPUT ===
    final_response: str = Field(
        default="",
        max_length=10000,
        description="Approved final response (max 10000 chars)"
    )
    
    workflow_status: Literal[
        "initialized",
        "researching", 
        "verifying",
        "correcting",
        "approved",
        "escalated",
        "failed"
    ] = Field(
        default="initialized",
        description="Current workflow execution status"
    )
    
    # === EXECUTION TRACKING ===
    execution_log: List[ExecutionStep] = Field(
        default_factory=list,
        description="Complete step-by-step execution history"
    )
    
    start_time: str = Field(
        description="Workflow start timestamp (ISO format)"
    )
    
    end_time: Optional[str] = Field(
        default=None,
        description="Workflow completion timestamp (ISO format)"
    )
    
    # === RESOURCE USAGE TRACKING ===
    resource_usage: Dict[str, Any] = Field(
        default_factory=lambda: {
            "cpu_time_seconds": 0.0,
            "memory_peak_mb": 0.0,
            "llm_tokens_total": 0,
            "llm_cost_usd": 0.0,
            "db_queries_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        },
        description="Resource usage metrics tracking"
    )
    
    # === ERROR HANDLING ===
    error_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured error tracking with timestamps"
    )
    
    # === CONFIGURATION ===
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow configuration snapshot"
    )
    
    # === OBSERVABILITY ===
    workflow_id: str = Field(
        default_factory=lambda: f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        description="Unique workflow identifier"
    )
    
    correlation_id: str = Field(
        default_factory=lambda: f"corr_{uuid.uuid4().hex[:12]}",
        description="Distributed tracing correlation ID"
    )
    
    # === VALIDATION RULES ===
    
    @field_validator('start_time', mode='before')
    @classmethod
    def validate_start_time(cls, v):
        """Ensure start_time is valid ISO format"""
        if v is None:
            return datetime.now().isoformat()
        
        # Validate ISO format
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("start_time must be valid ISO format timestamp")
    
    @model_validator(mode='after')
    def validate_cross_field_state(self):
        if self.retry_count > self.max_retries:
            raise ValueError(
                f"retry_count ({self.retry_count}) cannot exceed "
                f"max_retries ({self.max_retries})"
            )
        if self.end_time is not None:
            try:
                start_dt = datetime.fromisoformat(
                    self.start_time.replace('Z', '+00:00')
                )
                end_dt = datetime.fromisoformat(
                    self.end_time.replace('Z', '+00:00')
                )
            except ValueError as exc:
                raise ValueError(
                    "end_time must be valid ISO format timestamp"
                ) from exc
            if end_dt <= start_dt:
                raise ValueError("end_time must be after start_time")
        return self
    
    @field_validator('resource_usage')
    @classmethod
    def validate_resource_usage_structure(cls, v):
        """Ensure resource_usage has required fields"""
        required_fields = [
            "cpu_time_seconds",
            "memory_peak_mb", 
            "llm_tokens_total",
            "llm_cost_usd",
            "db_queries_count",
            "cache_hits",
            "cache_misses"
        ]
        
        for field in required_fields:
            if field not in v:
                v[field] = 0.0 if "seconds" in field or "mb" in field or "usd" in field else 0
        
        # Validate numeric types
        numeric_fields = ["cpu_time_seconds", "memory_peak_mb", "llm_cost_usd"]
        for field in numeric_fields:
            if not isinstance(v[field], (int, float)) or v[field] < 0:
                raise ValueError(f"{field} must be non-negative number")
        
        integer_fields = ["llm_tokens_total", "db_queries_count", "cache_hits", "cache_misses"]
        for field in integer_fields:
            if not isinstance(v[field], int) or v[field] < 0:
                raise ValueError(f"{field} must be non-negative integer")
        
        return v
    
    @field_validator('error_log')
    @classmethod
    def validate_error_log_structure(cls, v):
        """Ensure error log entries have required structure"""
        for i, error in enumerate(v):
            if not isinstance(error, dict):
                raise ValueError(f"Error log entry {i} must be dictionary")
            
            required_fields = ["timestamp", "error_type", "message"]
            for field in required_fields:
                if field not in error:
                    raise ValueError(f"Error log entry {i} missing required field: {field}")
        
        return v
    
    # === COMPUTED PROPERTIES ===
    
    @property
    def is_terminal_state(self) -> bool:
        """Check if workflow is in terminal state"""
        return self.workflow_status in ["approved", "escalated", "failed"]
    
    @property
    def execution_duration_seconds(self) -> Optional[float]:
        """Calculate total execution duration"""
        if not self.end_time:
            return None
        
        try:
            start_dt = datetime.fromisoformat(self.start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(self.end_time.replace('Z', '+00:00'))
            return (end_dt - start_dt).total_seconds()
        except ValueError:
            return None
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if verification found critical issues"""
        if not self.verification_result:
            return False
        return self.verification_result.criteria.critical_issues_count > 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        hits = self.resource_usage.get("cache_hits", 0)
        misses = self.resource_usage.get("cache_misses", 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def add_execution_step(self, step: ExecutionStep) -> None:
        """Add execution step to log with validation"""
        self.execution_log.append(step)
    
    def add_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add structured error to error log"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": message,
            "details": details or {}
        }
        self.error_log.append(error_entry)
    
    def update_resource_usage(self, **kwargs) -> None:
        """Update resource usage metrics"""
        for key, value in kwargs.items():
            if key in self.resource_usage:
                self.resource_usage[key] = value
    
    def get_workflow_summary(self) -> str:
        """Generate human-readable workflow summary"""
        duration = self.execution_duration_seconds
        duration_str = f"{duration:.2f}s" if duration else "ongoing"
        
        return f"""
🔄 WORKFLOW SUMMARY
ID: {self.workflow_id}
Status: {self.workflow_status.upper()}
Duration: {duration_str}
Retries: {self.retry_count}/{self.max_retries}
Steps: {len(self.execution_log)}
Errors: {len(self.error_log)}
Cache Hit Rate: {self.cache_hit_rate:.1%}
        """.strip()
    model_config = ConfigDict(validate_assignment=True, extra="forbid", use_enum_values=True, json_schema_extra={
        "example": {
            "objection_text": "iPhone quá đắt so với Samsung, tại sao tôi nên mua?",
            "customer_context": {"segment": "price_sensitive", "previous_purchases": []},
            "draft_response": "iPhone mang lại giá trị vượt trội với hệ sinh thái Apple...",
            "tools_used": ["product_search", "price_comparison", "policy_lookup"],
            "research_reasoning": "Analyzed price objection, compared features and value proposition",
            "research_sources": ["product_db", "pricing_policy", "competitor_analysis"],
            "verification_result": None,
            "correction_feedback": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_response": "",
            "workflow_status": "initialized",
            "execution_log": [],
            "start_time": "2024-01-15T10:30:00.000Z",
            "end_time": None,
            "resource_usage": {
                "cpu_time_seconds": 0.0,
                "memory_peak_mb": 0.0,
                "llm_tokens_total": 0,
                "llm_cost_usd": 0.0,
                "db_queries_count": 0,
                "cache_hits": 0,
                "cache_misses": 0
            },
            "error_log": [],
            "config": {"price_tolerance_percent": 1.0, "max_retries": 3},
            "workflow_id": "wf_20240115_103000_abc12345",
            "correlation_id": "corr_xyz789abc123"
        }
    })
class WorkflowConfig(BaseModel):
    """
    Enhanced workflow configuration với comprehensive Pydantic validation
    
    Configurable parameters cho binary verification thresholds, performance settings,
    và observability options với runtime validation.
    
    **Validates: Requirements 10.1, 10.2** - Configuration management
    """
    
    # === BINARY VERIFICATION THRESHOLDS ===
    price_tolerance_percent: float = Field(
        default=1.0, 
        ge=0.0, 
        le=100.0,
        description="Price accuracy tolerance (±%) - 0% = exact match required"
    )
    
    policy_citation_required: bool = Field(
        default=True,
        description="Require policy citations for authenticity verification"
    )
    
    relevance_min_coverage: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0, 
        description="Minimum topic coverage ratio (0-1) for relevance pass"
    )
    
    # === RETRY & ESCALATION LOGIC ===
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum correction attempts before escalation"
    )
    
    critical_issue_escalation: bool = Field(
        default=True,
        description="Escalate immediately on critical issues (bypass retries)"
    )
    
    critical_issue_threshold: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of critical issues to trigger immediate escalation"
    )
    
    # === PERFORMANCE SETTINGS ===
    async_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Async operation timeout (5-300 seconds)"
    )
    
    parallel_verification: bool = Field(
        default=True,
        description="Run price/policy/relevance checks in parallel"
    )
    
    early_termination: bool = Field(
        default=True,
        description="Stop verification on first critical issue detected"
    )
    
    max_concurrent_workflows: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent workflow executions"
    )
    
    # === LLM OPTIMIZATION ===
    llm_token_limit: int = Field(
        default=4000,
        ge=1000,
        le=32000,
        description="Maximum LLM tokens per verification request"
    )
    
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for consistent verification (0.0-2.0)"
    )
    
    enable_llm_caching: bool = Field(
        default=True,
        description="Enable LLM response caching for identical inputs"
    )
    
    # === CACHING CONFIGURATION ===
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds (1 minute - 24 hours)"
    )
    
    cache_max_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum cache entries (10-10000)"
    )
    
    enable_policy_caching: bool = Field(
        default=True,
        description="Enable policy document caching"
    )
    
    enable_price_caching: bool = Field(
        default=True,
        description="Enable product price caching"
    )
    
    # === OBSERVABILITY & LOGGING ===
    detailed_logging: bool = Field(
        default=True,
        description="Enable detailed execution logging"
    )
    
    performance_tracking: bool = Field(
        default=True,
        description="Track performance metrics and resource usage"
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level for workflow execution"
    )
    
    enable_distributed_tracing: bool = Field(
        default=True,
        description="Enable distributed tracing with correlation IDs"
    )
    
    metrics_export_interval: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Metrics export interval in seconds"
    )
    
    # === SECURITY & COMPLIANCE ===
    input_sanitization: bool = Field(
        default=True,
        description="Enable input sanitization for objection text"
    )
    
    pii_detection: bool = Field(
        default=True,
        description="Enable PII detection and masking in logs"
    )
    
    audit_logging: bool = Field(
        default=True,
        description="Enable audit logging for compliance"
    )
    
    # === BUSINESS RULES ===
    business_unit: str = Field(
        default="default",
        min_length=1,
        max_length=50,
        description="Business unit identifier for configuration scoping"
    )
    
    verification_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "price_accuracy": 0.4,
            "policy_authenticity": 0.3,
            "topic_relevance": 0.3
        },
        description="Verification criteria weights (must sum to 1.0)"
    )
    
    # === VALIDATION RULES ===
    
    @field_validator('verification_weights')
    @classmethod
    def validate_weights_sum(cls, v):
        """Ensure verification weights sum to 1.0"""
        total = sum(v.values())
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Verification weights must sum to 1.0, got {total}")
        
        # Ensure all weights are non-negative
        for criterion, weight in v.items():
            if weight < 0:
                raise ValueError(f"Weight for {criterion} must be non-negative, got {weight}")
        
        return v
    
    @field_validator('llm_temperature')
    @classmethod
    def validate_temperature_for_verification(cls, v):
        """Recommend low temperature for consistent verification"""
        if v > 0.5:
            # Warning: high temperature may cause inconsistent verification
            pass  # Allow but could add warning in logs
        return v
    
    @field_validator('cache_ttl_seconds')
    @classmethod
    def validate_cache_ttl(cls, v):
        """Ensure cache TTL is reasonable for verification use case"""
        # For verification, shorter TTL is better to ensure fresh data
        if v > 7200:  # 2 hours
            # Warning: long cache TTL may cause stale verification data
            pass  # Allow but could add warning
        return v
    
    # === COMPUTED PROPERTIES ===
    
    @property
    def is_high_performance_mode(self) -> bool:
        """Check if configured for high performance"""
        return (
            self.parallel_verification and
            self.early_termination and
            self.enable_llm_caching and
            self.max_concurrent_workflows >= 5
        )
    
    @property
    def is_strict_verification_mode(self) -> bool:
        """Check if configured for strict verification"""
        return (
            self.price_tolerance_percent <= 1.0 and
            self.policy_citation_required and
            self.relevance_min_coverage >= 0.8 and
            self.critical_issue_escalation
        )
    
    def get_config_summary(self) -> str:
        """Generate human-readable configuration summary"""
        return f"""
⚙️  WORKFLOW CONFIGURATION
🎯 Verification Mode: {'Strict' if self.is_strict_verification_mode else 'Standard'}
⚡ Performance Mode: {'High' if self.is_high_performance_mode else 'Standard'}
💰 Price Tolerance: ±{self.price_tolerance_percent}%
📋 Policy Citations: {'Required' if self.policy_citation_required else 'Optional'}
🎯 Relevance Coverage: {self.relevance_min_coverage:.0%} minimum
🔄 Max Retries: {self.max_retries}
⏱️  Timeout: {self.async_timeout_seconds}s
🚀 Parallel Verification: {'Enabled' if self.parallel_verification else 'Disabled'}
        """.strip()
    
    def validate_for_production(self) -> List[str]:
        """Validate configuration for production deployment"""
        warnings = []
        
        # Performance warnings
        if not self.parallel_verification:
            warnings.append("Parallel verification disabled - may impact performance")
        
        if self.async_timeout_seconds < 15:
            warnings.append("Low timeout may cause premature failures")
        
        if self.max_concurrent_workflows < 5:
            warnings.append("Low concurrency limit may create bottlenecks")
        
        # Security warnings
        if not self.input_sanitization:
            warnings.append("Input sanitization disabled - security risk")
        
        if not self.audit_logging:
            warnings.append("Audit logging disabled - compliance risk")
        
        # Verification warnings
        if self.price_tolerance_percent > 5.0:
            warnings.append("High price tolerance may allow significant errors")
        
        if not self.policy_citation_required:
            warnings.append("Policy citations not required - authenticity risk")
        
        return warnings
    model_config = ConfigDict(validate_assignment=True, extra="forbid", use_enum_values=True, json_schema_extra={
        "example": {
            "price_tolerance_percent": 1.0,
            "policy_citation_required": True,
            "relevance_min_coverage": 0.7,
            "max_retries": 3,
            "critical_issue_escalation": True,
            "critical_issue_threshold": 1,
            "async_timeout_seconds": 30,
            "parallel_verification": True,
            "early_termination": True,
            "max_concurrent_workflows": 10,
            "llm_token_limit": 4000,
            "llm_temperature": 0.1,
            "enable_llm_caching": True,
            "cache_ttl_seconds": 3600,
            "cache_max_size": 1000,
            "enable_policy_caching": True,
            "enable_price_caching": True,
            "detailed_logging": True,
            "performance_tracking": True,
            "log_level": "INFO",
            "enable_distributed_tracing": True,
            "metrics_export_interval": 60,
            "input_sanitization": True,
            "pii_detection": True,
            "audit_logging": True,
            "business_unit": "sales_vietnam",
            "verification_weights": {
                "price_accuracy": 0.4,
                "policy_authenticity": 0.3,
                "topic_relevance": 0.3
            }
        }
    })


# === UTILITY FUNCTIONS ===

def create_initial_workflow_state(
    objection_text: str,
    config: Optional[WorkflowConfig] = None,
    customer_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create initial WorkflowState với proper defaults và validation
    
    **Validates: Requirements 7.1** - Proper state initialization
    """
    if config is None:
        config = WorkflowConfig()
    
    # Generate unique identifiers
    workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    correlation_id = f"corr_{uuid.uuid4().hex[:12]}"
    
    initial_state = {
        # Input
        "objection_text": objection_text,
        "customer_context": customer_context,
        
        # Research Agent Output (empty initially - will be populated by research node)
        "draft_response": "",  # Empty initially, validation allows 0 length
        "tools_used": [],
        "research_reasoning": "",  # Empty initially, validation allows 0 length
        "research_sources": [],
        
        # Verification Agent Output
        "verification_result": None,
        
        # Self-Correction Loop
        "correction_feedback": None,
        "retry_count": 0,
        "max_retries": config.max_retries,
        
        # Final Output
        "final_response": "",
        "workflow_status": "initialized",
        
        # Execution Tracking
        "execution_log": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        
        # Resource Usage
        "resource_usage": {
            "cpu_time_seconds": 0.0,
            "memory_peak_mb": 0.0,
            "llm_tokens_total": 0,
            "llm_cost_usd": 0.0,
            "db_queries_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        },
        
        # Error Handling
        "error_log": [],
        
        # Configuration
        "config": config.model_dump(),  # Use model_dump instead of deprecated dict()
        
        # Observability
        "workflow_id": workflow_id,
        "correlation_id": correlation_id
    }
    
    # Validate initial state
    validator = WorkflowStateValidator(**initial_state)
    return validator.model_dump()  # Use model_dump instead of deprecated dict()


def validate_workflow_state(state: Dict[str, Any]) -> WorkflowStateValidator:
    """
    Validate WorkflowState dictionary against Pydantic schema
    
    **Validates: Requirements 7.4** - State validation và integrity
    """
    return WorkflowStateValidator(**state)
