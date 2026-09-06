"""
Configuration Loader for Verification Agent
Handles loading and merging of YAML configuration files with environment overrides.
Integrates with Pydantic models for validated configuration access.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ConfigPaths:
    """Configuration file paths"""
    verification_config: str = "backend/verification/config/verification_config.yaml"
    workflow_config: str = "backend/verification/config/workflow_config.yaml"
    prompts_config: str = "backend/verification/config/prompts.yaml"
    environment_dir: str = "backend/verification/config/environments"


class ConfigurationError(Exception):
    """Configuration loading and validation errors"""
    pass


class ConfigLoader:
    """
    Configuration loader with environment override support

    Loads YAML configuration files and applies environment-specific overrides.
    Supports configuration merging, validation, and runtime updates.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize configuration loader

        Args:
            base_path: Base directory for configuration files (optional)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.paths = ConfigPaths()
        self._config_cache: Dict[str, Dict[str, Any]] = {}

    def load_verification_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Load verification configuration with environment overrides

        Args:
            environment: Environment name (dev, prod, test, etc.)

        Returns:
            Merged configuration dictionary
        """
        try:
            # Load base verification config
            base_config = self._load_yaml_file(self.paths.verification_config)

            # Apply environment overrides if specified
            if environment:
                env_config = self._load_environment_config(environment)
                if env_config:
                    base_config = self._merge_configs(base_config, env_config)

            # Apply environment variable overrides
            base_config = self._apply_env_var_overrides(base_config)

            # Cache the configuration
            cache_key = f"verification_{environment or 'default'}"
            self._config_cache[cache_key] = base_config

            logger.info(f"Loaded verification config for environment: {environment or 'default'}")
            return base_config

        except Exception as e:
            raise ConfigurationError(f"Failed to load verification config: {e}")

    def load_workflow_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Load workflow configuration with environment overrides

        Args:
            environment: Environment name (dev, prod, test, etc.)

        Returns:
            Merged workflow configuration dictionary
        """
        try:
            # Load base workflow config
            base_config = self._load_yaml_file(self.paths.workflow_config)

            # Apply environment overrides if specified
            if environment:
                env_config = self._load_environment_config(environment)
                if env_config and 'workflow' in env_config:
                    # Merge workflow-specific settings
                    workflow_overrides = {'workflow': env_config['workflow']}
                    if 'nodes' in env_config:
                        workflow_overrides['nodes'] = env_config['nodes']
                    if 'performance' in env_config:
                        workflow_overrides['performance'] = env_config['performance']

                    base_config = self._merge_configs(base_config, workflow_overrides)

            # Cache the configuration
            cache_key = f"workflow_{environment or 'default'}"
            self._config_cache[cache_key] = base_config

            logger.info(f"Loaded workflow config for environment: {environment or 'default'}")
            return base_config

        except Exception as e:
            raise ConfigurationError(f"Failed to load workflow config: {e}")

    def load_prompts_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Load prompts configuration with environment overrides

        Args:
            environment: Environment name (dev, prod, test, etc.)

        Returns:
            Prompts configuration dictionary
        """
        try:
            # Load base prompts config
            base_config = self._load_yaml_file(self.paths.prompts_config)

            # Apply environment overrides if specified
            if environment:
                env_config = self._load_environment_config(environment)
                if env_config and 'environment_overrides' in base_config:
                    env_overrides = base_config['environment_overrides'].get(environment, {})
                    if env_overrides:
                        base_config = self._merge_configs(base_config, env_overrides)

            # Cache the configuration
            cache_key = f"prompts_{environment or 'default'}"
            self._config_cache[cache_key] = base_config

            logger.info(f"Loaded prompts config for environment: {environment or 'default'}")
            return base_config

        except Exception as e:
            raise ConfigurationError(f"Failed to load prompts config: {e}")

    def load_all_configs(self, environment: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files

        Args:
            environment: Environment name (dev, prod, test, etc.)

        Returns:
            Dictionary containing all configurations
        """
        return {
            'verification': self.load_verification_config(environment),
            'workflow': self.load_workflow_config(environment),
            'prompts': self.load_prompts_config(environment)
        }

    def get_environment_from_env_var(self) -> Optional[str]:
        """
        Get environment name from environment variables

        Returns:
            Environment name or None if not set
        """
        return os.getenv('VERIFICATION_ENVIRONMENT') or os.getenv('ENVIRONMENT')

    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load YAML file with error handling

        Args:
            file_path: Path to YAML file

        Returns:
            Parsed YAML content as dictionary
        """
        full_path = self.base_path / file_path

        if not full_path.exists():
            raise ConfigurationError(f"Configuration file not found: {full_path}")

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                if content is None:
                    return {}
                return content
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {full_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read {full_path}: {e}")

    def _load_environment_config(self, environment: str) -> Optional[Dict[str, Any]]:
        """
        Load environment-specific configuration

        Args:
            environment: Environment name

        Returns:
            Environment configuration or None if not found
        """
        env_file = f"{self.paths.environment_dir}/{environment}.yaml"
        env_path = self.base_path / env_file

        if not env_path.exists():
            logger.warning(f"Environment config not found: {env_path}")
            return None

        try:
            return self._load_yaml_file(env_file)
        except ConfigurationError:
            logger.warning(f"Failed to load environment config: {env_file}")
            return None

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge configuration dictionaries

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _apply_env_var_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration

        Args:
            config: Base configuration

        Returns:
            Configuration with environment variable overrides applied
        """
        result = deepcopy(config)

        # Define environment variable mappings
        env_mappings = {
            'VERIFICATION_PRICE_TOLERANCE': ('price_accuracy', 'tolerance_percent', float),
            'VERIFICATION_MAX_RETRIES': ('retry_settings', 'max_retries', int),
            'VERIFICATION_PARALLEL_CHECKS': ('performance', 'parallel_verification', bool),
            'VERIFICATION_LOG_LEVEL': ('logging', 'level', str),
            'VERIFICATION_CACHE_ENABLED': ('performance', 'caching', 'enabled', bool),
            'VERIFICATION_TIMEOUT': ('performance', 'verification_timeout_seconds', int),
        }

        for env_var, (section, key, *rest) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Handle nested configuration paths
                    if rest:
                        nested_key, value_type = rest[0], rest[1] if len(rest) > 1 else str
                        if section not in result:
                            result[section] = {}
                        if key not in result[section]:
                            result[section][key] = {}

                        # Convert value to appropriate type
                        converted_value = self._convert_env_value(env_value, value_type)
                        result[section][key][nested_key] = converted_value
                    else:
                        value_type = key if len(rest) == 0 else rest[0]
                        if section not in result:
                            result[section] = {}

                        # Convert value to appropriate type
                        converted_value = self._convert_env_value(env_value, value_type)
                        result[section][key] = converted_value

                    logger.info(f"Applied environment override: {env_var}={env_value}")

                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")

        return result

    def _convert_env_value(self, value: str, value_type: type) -> Union[str, int, float, bool]:
        """
        Convert environment variable string to appropriate type

        Args:
            value: String value from environment variable
            value_type: Target type for conversion

        Returns:
            Converted value
        """
        if value_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif value_type == int:
            return int(value)
        elif value_type == float:
            return float(value)
        else:
            return value

    def validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """
        Validate configuration structure and values

        Args:
            config: Configuration to validate
            config_type: Type of configuration (verification, workflow, prompts)

        Returns:
            True if valid, raises ConfigurationError if invalid
        """
        try:
            if config_type == 'verification':
                return self._validate_verification_config(config)
            elif config_type == 'workflow':
                return self._validate_workflow_config(config)
            elif config_type == 'prompts':
                return self._validate_prompts_config(config)
            else:
                raise ConfigurationError(f"Unknown config type: {config_type}")
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    def _validate_verification_config(self, config: Dict[str, Any]) -> bool:
        """Validate verification configuration structure"""
        required_sections = ['price_accuracy', 'policy_authenticity', 'topic_relevance']

        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")

        # Validate price accuracy settings
        price_config = config['price_accuracy']
        if 'tolerance_percent' not in price_config:
            raise ConfigurationError("Missing price_accuracy.tolerance_percent")

        tolerance = price_config['tolerance_percent']
        if not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ConfigurationError("price_accuracy.tolerance_percent must be non-negative number")

        return True

    def _validate_workflow_config(self, config: Dict[str, Any]) -> bool:
        """Validate workflow configuration structure"""
        required_sections = ['workflow', 'nodes', 'edges']

        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")

        # Validate node configuration
        nodes_config = config['nodes']
        required_nodes = ['research_node', 'verification_node', 'correction_node']

        for node in required_nodes:
            if node not in nodes_config:
                raise ConfigurationError(f"Missing required node: {node}")

        return True

    def _validate_prompts_config(self, config: Dict[str, Any]) -> bool:
        """Validate prompts configuration structure"""
        required_sections = ['verification_prompts', 'price_accuracy_prompts', 'policy_authenticity_prompts']

        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")

        return True

    def clear_cache(self) -> None:
        """Clear configuration cache"""
        self._config_cache.clear()
        logger.info("Configuration cache cleared")

    def reload(self, environment: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Reload all configurations at runtime without restart.

        Clears the cache and reloads all config files, applying environment
        overrides. Supports Requirement 10.5 (config changes without restart).
        Emits a ConfigChangeType.ALL event to invalidate all caches.

        Args:
            environment: Environment name. Auto-detected if None.

        Returns:
            Freshly loaded configuration dictionary.
        """
        self.clear_cache()
        if environment is None:
            environment = self.get_environment_from_env_var()
        configs = self.load_all_configs(environment)
        logger.info(f"All configurations reloaded (env={environment or 'default'})")
        self._emit_config_change("all")
        return configs

    def get_pydantic_config(self, environment: Optional[str] = None) -> "VerificationConfig":
        """
        Load and return a Pydantic-validated VerificationConfig.

        Combines YAML loading with Pydantic validation, providing a single
        entry point for fully-validated configuration access.

        Args:
            environment: Environment name. Auto-detected if None.

        Returns:
            Validated VerificationConfig instance.
        """
        from .config import load_config_from_yaml
        if environment is None:
            environment = self.get_environment_from_env_var()
        return load_config_from_yaml(
            config_dir=self.base_path / "verification" / "config"
            if (self.base_path / "verification" / "config").exists()
            else Path(__file__).parent,
            environment=environment,
        )

    def reload_config(self, config_type: str, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Reload specific configuration type and emit a cache invalidation event.

        Args:
            config_type: Type of configuration to reload
            environment: Environment name

        Returns:
            Reloaded configuration
        """
        # Clear relevant cache entries
        cache_key = f"{config_type}_{environment or 'default'}"
        if cache_key in self._config_cache:
            del self._config_cache[cache_key]

        # Reload configuration
        if config_type == 'verification':
            result = self.load_verification_config(environment)
        elif config_type == 'workflow':
            result = self.load_workflow_config(environment)
        elif config_type == 'prompts':
            result = self.load_prompts_config(environment)
            self._emit_config_change("prompts")
            return result
        else:
            raise ConfigurationError(f"Unknown config type: {config_type}")

        # Emit thresholds/verification change for verification and workflow configs
        self._emit_config_change("thresholds")
        return result

    def _emit_config_change(self, change_type_str: str) -> None:
        """
        Emit a cache invalidation event for the given change type.

        Deferred import avoids circular dependencies.
        """
        try:
            from backend.verification.utils.cache_invalidation import (
                ConfigChangeEvent,
                ConfigChangeType,
                get_cache_invalidation_manager,
            )
            change_type = ConfigChangeType(change_type_str)
            manager = get_cache_invalidation_manager()
            manager.on_config_changed(ConfigChangeEvent(change_type))
        except Exception as exc:
            logger.warning("Could not emit cache invalidation event: %s", exc)


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(base_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration loader instance (singleton pattern)

    Args:
        base_path: Base directory for configuration files

    Returns:
        ConfigLoader instance
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = ConfigLoader(base_path)

    return _config_loader


def load_config_for_environment(environment: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to load all configurations for an environment

    Args:
        environment: Environment name (auto-detected if None)

    Returns:
        Dictionary containing all configurations
    """
    loader = get_config_loader()

    # Auto-detect environment if not specified
    if environment is None:
        environment = loader.get_environment_from_env_var()

    return loader.load_all_configs(environment)


def validate_all_configs(configs: Dict[str, Dict[str, Any]]) -> bool:
    """
    Validate all loaded configurations

    Args:
        configs: Dictionary of configurations to validate

    Returns:
        True if all configurations are valid
    """
    loader = get_config_loader()

    for config_type, config_data in configs.items():
        loader.validate_config(config_data, config_type)

    return True
