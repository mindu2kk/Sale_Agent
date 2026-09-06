"""
Verification Agent Configuration với Pydantic Validation

Binary verification thresholds và workflow settings:
- Configurable pass/fail criteria cho Price/Policy/Relevance
- Retry logic với issue severity-based limits
- Performance optimization settings
- Environment-specific overrides
- Unified config combining VerificationConfig + VerificationThresholdsConfig
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class VerificationConfig(BaseModel):
    """
    Main configuration cho Verification Agent với binary thresholds

    Comprehensive configuration với Pydantic validation,
    environment overrides, và runtime updates support.
    """

    # === BINARY VERIFICATION THRESHOLDS ===

    # Price Accuracy Settings
    price_tolerance_percent: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Price accuracy tolerance (±%)"
    )

    price_critical_threshold: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="Price deviation threshold for critical issues (%)"
    )

    # Policy Authenticity Settings
    policy_citation_required: bool = Field(
        default=True,
        description="Require policy citations for authenticity verification"
    )

    policy_forbidden_phrases: List[str] = Field(
        default_factory=lambda: [
            "tự bịa", "không có trong hệ thống", "theo ý kiến cá nhân",
            "fabricated", "made up", "personal opinion"
        ],
        description="Forbidden phrases indicating fabricated policies"
    )

    # Topic Relevance Settings
    relevance_min_coverage: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum topic coverage ratio for PASS (0-1)"
    )

    relevance_empathy_bonus: bool = Field(
        default=True,
        description="Give bonus points for empathy statements"
    )

    # === WORKFLOW SETTINGS ===

    # Retry Logic
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum correction attempts before escalation"
    )

    critical_issue_escalation: bool = Field(
        default=True,
        description="Escalate immediately when critical issues detected"
    )

    retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0.0,
        le=60.0,
        description="Backoff delay between retries (seconds)"
    )

    # === PERFORMANCE OPTIMIZATION ===

    # Async Settings
    async_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Async operation timeout"
    )

    parallel_verification: bool = Field(
        default=True,
        description="Run Price/Policy/Relevance checks in parallel"
    )

    early_termination: bool = Field(
        default=True,
        description="Stop verification on first critical issue"
    )

    # Caching Settings
    enable_caching: bool = Field(
        default=True,
        description="Enable LRU caching for verification data"
    )

    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL for verification results"
    )

    cache_max_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum cache entries"
    )

    # === LLM SETTINGS ===

    # Model Configuration
    llm_model_name: str = Field(
        default="gpt-4",
        description="LLM model for verification reasoning"
    )

    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for consistent results"
    )

    llm_max_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens per LLM call"
    )

    # Token Optimization
    enable_token_optimization: bool = Field(
        default=True,
        description="Enable prompt compression and token optimization"
    )

    max_cost_per_verification: float = Field(
        default=0.05,
        ge=0.001,
        le=1.0,
        description="Maximum cost per verification (USD)"
    )

    # === OBSERVABILITY SETTINGS ===

    # Logging Configuration
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )

    detailed_logging: bool = Field(
        default=True,
        description="Enable detailed execution logging"
    )

    performance_tracking: bool = Field(
        default=True,
        description="Track performance metrics"
    )

    # Monitoring
    enable_metrics_export: bool = Field(
        default=True,
        description="Export metrics for monitoring systems"
    )

    metrics_export_interval: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Metrics export interval (seconds)"
    )

    # === DATABASE SETTINGS ===

    # Internal DB Configuration
    db_connection_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Database connection timeout (seconds)"
    )

    db_query_timeout: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Database query timeout (seconds)"
    )

    db_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Database retry attempts on failure"
    )

    # === SECURITY SETTINGS ===

    # Input Validation
    max_objection_length: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Maximum objection text length (characters)"
    )

    max_draft_length: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum draft response length (characters)"
    )

    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Maximum verifications per minute"
    )

    # === VALIDATION METHODS ===

    @model_validator(mode='after')
    def validate_price_thresholds(self):
        """Ensure critical threshold > tolerance"""
        if self.price_critical_threshold <= self.price_tolerance_percent:
            raise ValueError("Critical threshold must be greater than tolerance")
        return self

    @field_validator('llm_model_name')
    @classmethod
    def validate_llm_model(cls, v):
        """Validate supported LLM models — accepts any model name for flexibility."""
        # Allow any model name so we can swap LLM providers (Gemini, OpenAI, Anthropic, etc.)
        return v

    # === HELPER METHODS ===

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration dict"""
        return {
            "enabled": self.enable_caching,
            "ttl_seconds": self.cache_ttl_seconds,
            "max_size": self.cache_max_size
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dict"""
        return {
            "model_name": self.llm_model_name,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "enable_optimization": self.enable_token_optimization
        }

    def get_db_config(self) -> Dict[str, Any]:
        """Get database configuration dict"""
        return {
            "connection_timeout": self.db_connection_timeout,
            "query_timeout": self.db_query_timeout,
            "retry_attempts": self.db_retry_attempts
        }

    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return os.getenv("ENVIRONMENT", "production").lower() in ["dev", "development"]
    model_config = ConfigDict(validate_assignment=True, extra="forbid", use_enum_values=True, json_schema_extra={
        "example": {
            "price_tolerance_percent": 1.0,
            "policy_citation_required": True,
            "relevance_min_coverage": 0.7,
            "max_retries": 3,
            "parallel_verification": True,
            "llm_model_name": "gpt-4",
            "log_level": "INFO"
        }
    })


def load_config(config_path: Optional[str] = None) -> VerificationConfig:
    """
    Load configuration từ file với environment overrides

    Args:
        config_path: Path to config file (optional)

    Returns:
        VerificationConfig instance với validated settings
    """

    # Default config path
    if config_path is None:
        config_path = os.getenv("VERIFICATION_CONFIG_PATH", "backend/verification/config/default.json")

    # Load base configuration
    config_data = {}
    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")

    # Apply environment overrides
    env_overrides = _get_env_overrides()
    config_data.update(env_overrides)

    # Create and validate configuration
    try:
        return VerificationConfig(**config_data)
    except Exception as e:
        print(f"Error: Invalid configuration: {e}")
        print("Using default configuration")
        return get_default_config()


def get_default_config() -> VerificationConfig:
    """Get default configuration instance"""
    return VerificationConfig()


def _get_env_overrides() -> Dict[str, Any]:
    """
    Get configuration overrides từ environment variables

    Environment variables format: VERIFICATION_{FIELD_NAME}
    Example: VERIFICATION_MAX_RETRIES=5
    """

    overrides = {}
    prefix = "VERIFICATION_"

    # Mapping environment variables to config fields
    env_mappings = {
        f"{prefix}PRICE_TOLERANCE_PERCENT": ("price_tolerance_percent", float),
        f"{prefix}MAX_RETRIES": ("max_retries", int),
        f"{prefix}PARALLEL_VERIFICATION": ("parallel_verification", bool),
        f"{prefix}EARLY_TERMINATION": ("early_termination", bool),
        f"{prefix}LLM_MODEL_NAME": ("llm_model_name", str),
        f"{prefix}LLM_TEMPERATURE": ("llm_temperature", float),
        f"{prefix}LOG_LEVEL": ("log_level", str),
        f"{prefix}ENABLE_CACHING": ("enable_caching", bool),
        f"{prefix}ASYNC_TIMEOUT_SECONDS": ("async_timeout_seconds", int),
    }

    for env_var, (field_name, field_type) in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                if field_type == bool:
                    # Handle boolean environment variables
                    overrides[field_name] = env_value.lower() in ["true", "1", "yes", "on"]
                else:
                    overrides[field_name] = field_type(env_value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid environment variable {env_var}={env_value}: {e}")

    return overrides


def save_config(config: VerificationConfig, config_path: str) -> None:
    """
    Save configuration to file

    Args:
        config: VerificationConfig instance to save
        config_path: Path to save configuration file
    """

    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise RuntimeError(f"Failed to save config to {config_path}: {e}")


# ---------------------------------------------------------------------------
# Unified Configuration (combines VerificationConfig + thresholds + workflow)
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts, override wins on conflicts."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, return empty dict if missing."""
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as e:
        logger.warning(f"Failed to load YAML {path}: {e}")
        return {}


def _extract_workflow_overrides(env_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract VerificationConfig-compatible fields from environment YAML."""
    overrides: Dict[str, Any] = {}

    # price_accuracy.tolerance_percent → price_tolerance_percent
    pa = env_data.get("price_accuracy", {})
    pc = pa.get("pass_criteria", {})
    if "tolerance_percent" in pc:
        overrides["price_tolerance_percent"] = pc["tolerance_percent"]
    if "critical_threshold_percent" in pa:
        overrides["price_critical_threshold"] = pa["critical_threshold_percent"]

    # topic_relevance.min_coverage_ratio → relevance_min_coverage
    tr = env_data.get("topic_relevance", {})
    pc2 = tr.get("pass_criteria", {})
    if "min_coverage_ratio" in pc2:
        overrides["relevance_min_coverage"] = pc2["min_coverage_ratio"]

    # retry_settings.max_retries → max_retries
    rs = env_data.get("retry_settings", {})
    if "max_retries" in rs:
        overrides["max_retries"] = rs["max_retries"]

    # logging.level → log_level
    lg = env_data.get("logging", {})
    if "level" in lg:
        overrides["log_level"] = lg["level"]

    # performance.verification_timeout_seconds → async_timeout_seconds
    perf = env_data.get("performance", {})
    if "verification_timeout_seconds" in perf:
        overrides["async_timeout_seconds"] = perf["verification_timeout_seconds"]
    caching = perf.get("caching", {})
    if "enabled" in caching:
        overrides["enable_caching"] = caching["enabled"]
    if "ttl_seconds" in caching:
        overrides["cache_ttl_seconds"] = caching["ttl_seconds"]

    return overrides


def load_config_from_yaml(
    config_dir: Optional[Path] = None,
    environment: Optional[str] = None,
) -> VerificationConfig:
    """
    Load VerificationConfig from YAML files with environment overrides.

    Resolution order (later wins):
    1. Default Pydantic values
    2. verification_config.yaml base values
    3. environments/{environment}.yaml overrides
    4. Environment variable overrides (VERIFICATION_*)

    Args:
        config_dir: Directory containing config files. Defaults to this file's directory.
        environment: Environment name (development/production/testing).
                     Auto-detected from VERIFICATION_ENV or ENVIRONMENT if None.
    """
    if config_dir is None:
        config_dir = Path(__file__).parent

    # Auto-detect environment
    if environment is None:
        environment = os.getenv("VERIFICATION_ENV") or os.getenv("ENVIRONMENT")

    # 1. Load base verification_config.yaml
    base_data = _load_yaml(config_dir / "verification_config.yaml")
    merged = _extract_workflow_overrides(base_data)

    # 2. Apply environment-specific YAML overrides
    if environment:
        env_path = config_dir / "environments" / f"{environment}.yaml"
        env_data = _load_yaml(env_path)
        if env_data:
            env_overrides = _extract_workflow_overrides(env_data)
            merged = _deep_merge(merged, env_overrides)
            logger.info(f"Applied environment config overrides from: {env_path.name}")

    # 3. Apply environment variable overrides
    env_var_overrides = _get_env_overrides()
    merged = _deep_merge(merged, env_var_overrides)

    try:
        config = VerificationConfig(**merged)
        logger.info(f"Loaded VerificationConfig (env={environment or 'default'})")
        return config
    except Exception as e:
        logger.warning(f"Config validation error: {e}. Using defaults.")
        return VerificationConfig()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_config: Optional[VerificationConfig] = None


def get_config() -> VerificationConfig:
    """
    Get global VerificationConfig singleton.

    Loads from YAML on first call. Environment is auto-detected from
    VERIFICATION_ENV or ENVIRONMENT env vars.
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config_from_yaml()
    return _global_config


def reload(environment: Optional[str] = None) -> VerificationConfig:
    """
    Reload global configuration at runtime without restart.

    Args:
        environment: Override environment name. If None, auto-detects from env vars.

    Returns:
        Freshly loaded VerificationConfig instance.
    """
    global _global_config
    _global_config = load_config_from_yaml(environment=environment)
    logger.info("Global VerificationConfig reloaded.")
    return _global_config


def reload_config(config_path: Optional[str] = None) -> VerificationConfig:
    """Reload global configuration from file (legacy API, kept for compatibility)."""
    global _global_config
    _global_config = load_config(config_path)
    return _global_config
