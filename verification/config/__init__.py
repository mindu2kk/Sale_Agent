"""
Configuration Management cho Verification Agent

Binary verification thresholds và workflow settings:
- VerificationConfig: Main configuration với Pydantic validation
- Environment-specific overrides với type safety
- Runtime configuration updates without restart
- PromptTemplateManager: Load and render binary verification prompt templates
"""

from .config import (
    VerificationConfig,
    load_config,
    load_config_from_yaml,
    get_default_config,
    LogLevel,
    get_config,
    reload,
    reload_config,
)
from .config_loader import ConfigLoader, ConfigurationError, get_config_loader
from .prompt_templates import PromptTemplateManager, PromptTemplateError, get_prompt_manager
from .binary_verification_config import (
    BinaryVerificationConfig,
    get_binary_verification_config,
    load_binary_verification_config,
    apply_environment_override_to_config,
)
from .environment_config_override import (
    EnvironmentConfigOverride,
    EnvironmentOverrideError,
    SUPPORTED_ENVIRONMENTS,
    apply_environment_override,
    get_environment_override,
    detect_environment,
)

__all__ = [
    # Pydantic config model
    "VerificationConfig",
    "LogLevel",
    # Loading functions
    "load_config",
    "load_config_from_yaml",
    "get_default_config",
    # Singleton access
    "get_config",
    # Runtime reload (Requirement 10.5)
    "reload",
    "reload_config",
    # Config loader
    "ConfigLoader",
    "ConfigurationError",
    "get_config_loader",
    # Prompt templates
    "PromptTemplateManager",
    "PromptTemplateError",
    "get_prompt_manager",
    # Binary verification config schema (Task 7.1.1)
    "BinaryVerificationConfig",
    "get_binary_verification_config",
    "load_binary_verification_config",
    "apply_environment_override_to_config",
    # Environment-specific config overrides (Task 7.1.3)
    "EnvironmentConfigOverride",
    "EnvironmentOverrideError",
    "SUPPORTED_ENVIRONMENTS",
    "apply_environment_override",
    "get_environment_override",
    "detect_environment",
]