"""
Utility Functions cho Verification Agent

Helper functions và utilities:
- Logging utilities với structured formatting
- Performance monitoring helpers
- Cache management utilities
- Text processing helpers
"""

from .logging import setup_verification_logger, get_logger
from .logging import (
    set_correlation_context,
    get_correlation_context,
    clear_correlation_context,
    set_async_correlation_context,
    reset_async_correlation_context,
    async_correlation_context,
    CorrelationIDGenerator,
    CorrelationContext,
)
from .performance import PerformanceMonitor, measure_execution_time
from .cache import VerificationCache, PolicyDocumentCache, get_policy_document_cache, configure_policy_document_cache
from .text_utils import extract_prices, extract_policies, calculate_similarity
from .early_termination import EarlyTerminationManager, TerminationResult, create_early_termination_manager
from .severity_processor import SeverityBasedProcessor, SeverityLevel, IssuePriority
from .adaptive_timeout import (
    AdaptiveTimeoutManager,
    TimeoutConfig,
    ComplexityScore,
    run_with_adaptive_timeout,
    get_adaptive_timeout_manager,
    reset_adaptive_timeout_manager,
)
from .price_extractor import PriceExtractor, ExtractedPrice, extract_prices_detailed
from .semantic_similarity import SemanticSimilarityAnalyzer, SimilarityResult, INTENT_KEYWORDS, EMPATHY_PHRASES
from .intent_classifier import (
    IntentClassifier,
    ClassificationResult,
    IntentScore,
    INTENT_TAXONOMY,
    INTENT_LABELS,
)
from .cache_warmer import (
    WarmingResult,
    AllCachesWarmingResult,
    warm_prompt_cache,
    warm_product_price_cache,
    warm_policy_cache,
    warm_all_caches,
)
from .cache_invalidation import (
    CacheInvalidationManager,
    ConfigChangeEvent,
    ConfigChangeType,
    CacheAdapter,
    get_cache_invalidation_manager,
    reset_cache_invalidation_manager,
    build_default_manager,
)
from .prompt_compressor import PromptCompressor, CompressionResult, count_tokens, get_prompt_compressor
from .batch_processor import (
    BatchProcessor,
    BatchResult,
    BatchMetrics,
    VerificationRequest,
    RequestPriority,
)
from .resource_monitor import (
    ResourceSnapshot,
    ResourceUsageReport,
    ResourceUsageMonitor,
    get_resource_monitor,
    reset_resource_monitor,
)
from .token_tracker import (
    DEFAULT_MODEL_PRICING,
    estimate_cost,
    TokenUsageRecord,
    TokenUsageSummary,
    CostAlert,
    TokenAlertThresholds,
    LLMTokenTracker,
    get_token_tracker,
    reset_token_tracker,
)
from .metrics_collector import MetricsCollector, CheckerDecision, WorkflowDecision, CheckerStats
from .distributed_tracing import (
    Span,
    Trace,
    DistributedTracer,
    get_tracer,
    reset_tracer,
)
from .error_rate_tracker import (
    ErrorRateTracker,
    WORKFLOW_COMPONENTS,
    get_error_rate_tracker,
    reset_error_rate_tracker,
)
from .circuit_breaker import (
    CircuitState,
    CircuitOpenError,
    CircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    reset_circuit_breaker_registry,
)
from .error_classifier import (
    ErrorCategory,
    CATEGORY_SEVERITY_MAP,
    CATEGORY_ACTION_MAP,
    ClassifiedError,
    ErrorClassifier,
    get_error_classifier,
)
from .critical_alert_manager import (
    AlertSeverity,
    CriticalAlert,
    CriticalAlertManager,
    get_critical_alert_manager,
    reset_critical_alert_manager,
)
from .async_timeout_handler import (
    OperationTimeoutError,
    AsyncTimeoutHandler,
    get_async_timeout_handler,
    reset_async_timeout_handler,
)
from .fallback_verification import (
    FallbackMode,
    FallbackConfig,
    FallbackVerificationManager,
    get_fallback_verification_manager,
    reset_fallback_verification_manager,
)
from .health_check import (
    HealthStatus,
    ServiceHealthDetail,
    ErrorRateHealthDetail,
    ResourceHealthDetail,
    CircuitBreakerHealthDetail,
    HealthReport,
    HealthChecker,
    get_health_checker,
    reset_health_checker,
    run_health_check,
)
from .input_sanitizer import (
    SanitizationViolationType,
    InputSanitizationError,
    SanitizationResult,
    InputSanitizer,
    get_input_sanitizer,
    reset_input_sanitizer,
    sanitize_objection_text,
    sanitize_draft_response,
    async_sanitize_objection_text,
    async_sanitize_draft_response,
    MAX_OBJECTION_TEXT_LENGTH,
    MAX_DRAFT_RESPONSE_LENGTH,
)
from .api_key_manager import (
    SUPPORTED_PROVIDERS,
    KeySelectionStrategy,
    ApiKeyError,
    MaskedApiKey,
    ApiKeyManager,
    get_api_key_manager,
    reset_api_key_manager,
)
from .backup_manager import BackupManager

__all__ = [
    "setup_verification_logger",
    "get_logger",
    # Correlation ID tracking (Task 6.1.1)
    "set_correlation_context",
    "get_correlation_context",
    "clear_correlation_context",
    "set_async_correlation_context",
    "reset_async_correlation_context",
    "async_correlation_context",
    "CorrelationIDGenerator",
    "CorrelationContext",
    "PerformanceMonitor", 
    "measure_execution_time",
    "VerificationCache",
    "PolicyDocumentCache",
    "get_policy_document_cache",
    "configure_policy_document_cache",
    "extract_prices",
    "extract_policies",
    "calculate_similarity",
    "EarlyTerminationManager",
    "TerminationResult",
    "create_early_termination_manager",
    # Severity-based processing priorities (Task 5.4.3)
    "SeverityBasedProcessor",
    "SeverityLevel",
    "IssuePriority",
    # Adaptive timeout mechanisms (Task 5.4.4)
    "AdaptiveTimeoutManager",
    "TimeoutConfig",
    "ComplexityScore",
    "run_with_adaptive_timeout",
    "get_adaptive_timeout_manager",
    "reset_adaptive_timeout_manager",
    "PriceExtractor",
    "ExtractedPrice",
    "extract_prices_detailed",
    "SemanticSimilarityAnalyzer",
    "SimilarityResult",
    "INTENT_KEYWORDS",
    "EMPATHY_PHRASES",
    "IntentClassifier",
    "ClassificationResult",
    "IntentScore",
    "INTENT_TAXONOMY",
    "INTENT_LABELS",
    "WarmingResult",
    "AllCachesWarmingResult",
    "warm_prompt_cache",
    "warm_product_price_cache",
    "warm_policy_cache",
    "warm_all_caches",
    "CacheInvalidationManager",
    "ConfigChangeEvent",
    "ConfigChangeType",
    "CacheAdapter",
    "get_cache_invalidation_manager",
    "reset_cache_invalidation_manager",
    "build_default_manager",
    # Prompt compression (Task 5.2.2)
    "PromptCompressor",
    "CompressionResult",
    "count_tokens",
    "get_prompt_compressor",
    # Batch processing (Task 5.2.4)
    "BatchProcessor",
    "BatchResult",
    "BatchMetrics",
    "VerificationRequest",
    "RequestPriority",
    # Resource usage metrics (Task 5.3.3)
    "ResourceSnapshot",
    "ResourceUsageReport",
    "ResourceUsageMonitor",
    "get_resource_monitor",
    "reset_resource_monitor",
    # LLM token tracking & cost alerts (Task 5.3.4)
    "DEFAULT_MODEL_PRICING",
    "estimate_cost",
    "TokenUsageRecord",
    "TokenUsageSummary",
    "CostAlert",
    "TokenAlertThresholds",
    "LLMTokenTracker",
    "get_token_tracker",
    "reset_token_tracker",
    # Execution metrics collection with binary decision tracking (Task 6.1.2)
    "MetricsCollector",
    "CheckerDecision",
    "WorkflowDecision",
    "CheckerStats",
    # Distributed tracing for async verification operations (Task 6.1.4)
    "Span",
    "Trace",
    "DistributedTracer",
    "get_tracer",
    "reset_tracer",
    # Error rate tracking per async workflow component (Task 6.2.1)
    "ErrorRateTracker",
    "WORKFLOW_COMPONENTS",
    "get_error_rate_tracker",
    "reset_error_rate_tracker",
    # Circuit breaker pattern for external services (Task 6.2.2)
    "CircuitState",
    "CircuitOpenError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "reset_circuit_breaker_registry",
    # Error classification with severity mapping (Task 6.2.3)
    "ErrorCategory",
    "CATEGORY_SEVERITY_MAP",
    "CATEGORY_ACTION_MAP",
    "ClassifiedError",
    "ErrorClassifier",
    "get_error_classifier",
    # Automated alerting for critical failures (Task 6.2.4)
    "AlertSeverity",
    "CriticalAlert",
    "CriticalAlertManager",
    "get_critical_alert_manager",
    "reset_critical_alert_manager",
    # Async timeout handling with configurable thresholds (Task 6.3.2)
    "OperationTimeoutError",
    "AsyncTimeoutHandler",
    "get_async_timeout_handler",
    "reset_async_timeout_handler",
    # Fallback verification modes for service outages (Task 6.3.3)
    "FallbackMode",
    "FallbackConfig",
    "FallbackVerificationManager",
    "get_fallback_verification_manager",
    "reset_fallback_verification_manager",
    # Health check endpoints for monitoring systems (Task 6.3.4)
    "HealthStatus",
    "ServiceHealthDetail",
    "ErrorRateHealthDetail",
    "ResourceHealthDetail",
    "CircuitBreakerHealthDetail",
    "HealthReport",
    "HealthChecker",
    "get_health_checker",
    "reset_health_checker",
    "run_health_check",
    # Input sanitization for objection text and draft responses (Task 7.2.1)
    "SanitizationViolationType",
    "InputSanitizationError",
    "SanitizationResult",
    "InputSanitizer",
    "get_input_sanitizer",
    "reset_input_sanitizer",
    "sanitize_objection_text",
    "sanitize_draft_response",
    "async_sanitize_objection_text",
    "async_sanitize_draft_response",
    "MAX_OBJECTION_TEXT_LENGTH",
    "MAX_DRAFT_RESPONSE_LENGTH",
    # Secure API key management for LLM services (Task 7.2.2)
    "SUPPORTED_PROVIDERS",
    "KeySelectionStrategy",
    "ApiKeyError",
    "MaskedApiKey",
    "ApiKeyManager",
    "get_api_key_manager",
    "reset_api_key_manager",
    # Backup procedures with Pydantic state serialization (Task 7.3.4)
    "BackupManager",
]