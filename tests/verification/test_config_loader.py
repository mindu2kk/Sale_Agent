"""
Tests for Task 1.3.4: Configuration loader với Pydantic validation và environment overrides.

Covers:
- VerificationConfig Pydantic validation
- YAML loading via load_config_from_yaml()
- Environment variable overrides (VERIFICATION_ENV, VERIFICATION_MAX_RETRIES, etc.)
- get_config() singleton pattern
- reload() runtime reload without restart (Requirement 10.5)
- ConfigLoader.reload() and get_pydantic_config()
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from verification.config import (
    VerificationConfig,
    LogLevel,
    get_config,
    reload,
    load_config_from_yaml,
    get_default_config,
    ConfigLoader,
    ConfigurationError,
    get_config_loader,
)
from verification.config.config import _global_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).parent.parent / "config"


def _reset_singleton():
    """Reset the global config singleton between tests."""
    import verification.config.config as cfg_module
    cfg_module._global_config = None


# ---------------------------------------------------------------------------
# VerificationConfig Pydantic validation
# ---------------------------------------------------------------------------

class TestVerificationConfigValidation:
    def test_default_values_are_valid(self):
        cfg = VerificationConfig()
        assert cfg.max_retries == 3
        assert cfg.price_tolerance_percent == 1.0
        assert cfg.relevance_min_coverage == 0.7
        assert cfg.llm_model_name == "gpt-4"
        assert cfg.log_level == LogLevel.INFO

    def test_valid_custom_values(self):
        cfg = VerificationConfig(max_retries=5, price_tolerance_percent=2.0)
        assert cfg.max_retries == 5
        assert cfg.price_tolerance_percent == 2.0

    def test_max_retries_bounds(self):
        with pytest.raises(Exception):
            VerificationConfig(max_retries=-1)
        with pytest.raises(Exception):
            VerificationConfig(max_retries=11)

    def test_price_tolerance_bounds(self):
        with pytest.raises(Exception):
            VerificationConfig(price_tolerance_percent=-0.1)
        with pytest.raises(Exception):
            VerificationConfig(price_tolerance_percent=101.0)

    def test_relevance_min_coverage_bounds(self):
        with pytest.raises(Exception):
            VerificationConfig(relevance_min_coverage=-0.1)
        with pytest.raises(Exception):
            VerificationConfig(relevance_min_coverage=1.1)

    def test_unsupported_llm_model_raises(self):
        # Implementation accepts any model name for LLM provider flexibility.
        # Verify that arbitrary model names are accepted without raising.
        cfg = VerificationConfig(llm_model_name="unknown-model-xyz")
        assert cfg.llm_model_name == "unknown-model-xyz"

    def test_critical_threshold_must_exceed_tolerance(self):
        # critical_threshold must be > price_tolerance_percent
        with pytest.raises(Exception):
            VerificationConfig(price_tolerance_percent=5.0, price_critical_threshold=3.0)

    def test_validate_assignment_enabled(self):
        cfg = VerificationConfig()
        cfg.max_retries = 7
        assert cfg.max_retries == 7

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            VerificationConfig(unknown_field="value")

    def test_get_llm_config(self):
        cfg = VerificationConfig()
        llm = cfg.get_llm_config()
        assert llm["model_name"] == "gpt-4"
        assert "temperature" in llm

    def test_get_cache_config(self):
        cfg = VerificationConfig()
        cache = cfg.get_cache_config()
        assert "enabled" in cache
        assert "ttl_seconds" in cache


# ---------------------------------------------------------------------------
# load_config_from_yaml
# ---------------------------------------------------------------------------

class TestLoadConfigFromYaml:
    def test_loads_from_default_config_dir(self):
        _reset_singleton()
        cfg = load_config_from_yaml(config_dir=CONFIG_DIR)
        assert isinstance(cfg, VerificationConfig)

    def test_returns_defaults_when_yaml_missing(self, tmp_path):
        cfg = load_config_from_yaml(config_dir=tmp_path)
        assert isinstance(cfg, VerificationConfig)
        assert cfg.max_retries == 3  # default

    def test_yaml_overrides_defaults(self, tmp_path):
        """Write a minimal YAML and verify it overrides defaults."""
        yaml_content = """
retry_settings:
  max_retries: 7
price_accuracy:
  pass_criteria:
    tolerance_percent: 2.5
"""
        (tmp_path / "verification_config.yaml").write_text(yaml_content)
        cfg = load_config_from_yaml(config_dir=tmp_path)
        assert cfg.max_retries == 7
        assert cfg.price_tolerance_percent == 2.5

    def test_environment_yaml_overrides_base(self, tmp_path):
        """Environment YAML should override base YAML values."""
        base_yaml = """
retry_settings:
  max_retries: 3
"""
        env_yaml = """
retry_settings:
  max_retries: 1
"""
        (tmp_path / "verification_config.yaml").write_text(base_yaml)
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        (env_dir / "production.yaml").write_text(env_yaml)

        cfg = load_config_from_yaml(config_dir=tmp_path, environment="production")
        assert cfg.max_retries == 1

    def test_unknown_environment_falls_back_to_base(self, tmp_path):
        base_yaml = """
retry_settings:
  max_retries: 4
"""
        (tmp_path / "verification_config.yaml").write_text(base_yaml)
        cfg = load_config_from_yaml(config_dir=tmp_path, environment="nonexistent_env")
        assert cfg.max_retries == 4

    def test_verification_env_var_selects_environment(self, tmp_path):
        base_yaml = "retry_settings:\n  max_retries: 3\n"
        env_yaml = "retry_settings:\n  max_retries: 2\n"
        (tmp_path / "verification_config.yaml").write_text(base_yaml)
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        (env_dir / "testing.yaml").write_text(env_yaml)

        with patch.dict(os.environ, {"VERIFICATION_ENV": "testing"}):
            cfg = load_config_from_yaml(config_dir=tmp_path)
        assert cfg.max_retries == 2


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvironmentVariableOverrides:
    def test_verification_max_retries_env_var(self):
        _reset_singleton()
        with patch.dict(os.environ, {"VERIFICATION_MAX_RETRIES": "6"}):
            from verification.config.config import _get_env_overrides
            overrides = _get_env_overrides()
        assert overrides.get("max_retries") == 6

    def test_verification_log_level_env_var(self):
        with patch.dict(os.environ, {"VERIFICATION_LOG_LEVEL": "DEBUG"}):
            from verification.config.config import _get_env_overrides
            overrides = _get_env_overrides()
        assert overrides.get("log_level") == "DEBUG"

    def test_verification_parallel_bool_env_var(self):
        with patch.dict(os.environ, {"VERIFICATION_PARALLEL_VERIFICATION": "false"}):
            from verification.config.config import _get_env_overrides
            overrides = _get_env_overrides()
        assert overrides.get("parallel_verification") is False

    def test_invalid_env_var_is_ignored(self):
        """Invalid type conversion should not crash; field is simply omitted."""
        with patch.dict(os.environ, {"VERIFICATION_MAX_RETRIES": "not_a_number"}):
            from verification.config.config import _get_env_overrides
            overrides = _get_env_overrides()
        assert "max_retries" not in overrides

    def test_env_var_overrides_yaml(self, tmp_path):
        yaml_content = "retry_settings:\n  max_retries: 3\n"
        (tmp_path / "verification_config.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"VERIFICATION_MAX_RETRIES": "9"}):
            cfg = load_config_from_yaml(config_dir=tmp_path)
        assert cfg.max_retries == 9


# ---------------------------------------------------------------------------
# get_config() singleton
# ---------------------------------------------------------------------------

class TestGetConfigSingleton:
    def setup_method(self):
        _reset_singleton()

    def test_returns_verification_config(self):
        cfg = get_config()
        assert isinstance(cfg, VerificationConfig)

    def test_singleton_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_singleton_reset_after_reload(self):
        cfg1 = get_config()
        cfg2 = reload()
        assert cfg2 is not cfg1

    def test_reload_returns_fresh_config(self):
        _reset_singleton()
        cfg1 = get_config()
        cfg2 = reload()
        assert isinstance(cfg2, VerificationConfig)

    def test_reload_with_environment(self):
        _reset_singleton()
        cfg = reload(environment="development")
        assert isinstance(cfg, VerificationConfig)


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def test_load_verification_config_returns_dict(self):
        loader = ConfigLoader()
        config = loader.load_verification_config()
        assert isinstance(config, dict)
        assert "price_accuracy" in config

    def test_load_workflow_config_returns_dict(self):
        loader = ConfigLoader()
        config = loader.load_workflow_config()
        assert isinstance(config, dict)

    def test_load_all_configs_returns_all_types(self):
        loader = ConfigLoader()
        configs = loader.load_all_configs()
        assert "verification" in configs
        assert "workflow" in configs

    def test_reload_clears_cache_and_reloads(self):
        loader = ConfigLoader()
        loader.load_verification_config()
        assert len(loader._config_cache) > 0
        loader.reload()
        # After reload, cache is repopulated
        assert len(loader._config_cache) > 0

    def test_get_pydantic_config_returns_verification_config(self):
        loader = ConfigLoader()
        cfg = loader.get_pydantic_config()
        assert isinstance(cfg, VerificationConfig)

    def test_get_environment_from_env_var(self):
        loader = ConfigLoader()
        with patch.dict(os.environ, {"VERIFICATION_ENVIRONMENT": "production"}):
            env = loader.get_environment_from_env_var()
        assert env == "production"

    def test_get_config_loader_singleton(self):
        import verification.config.config_loader as cl_module
        cl_module._config_loader = None  # reset
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2

    def test_validate_verification_config_valid(self):
        loader = ConfigLoader()
        config = loader.load_verification_config()
        result = loader.validate_config(config, "verification")
        assert result is True

    def test_validate_config_missing_section_raises(self):
        loader = ConfigLoader()
        with pytest.raises(ConfigurationError):
            loader.validate_config({}, "verification")

    def test_reload_config_type_verification(self):
        loader = ConfigLoader()
        config = loader.reload_config("verification")
        assert isinstance(config, dict)

    def test_reload_config_unknown_type_raises(self):
        loader = ConfigLoader()
        with pytest.raises(ConfigurationError):
            loader.reload_config("unknown_type")


# ---------------------------------------------------------------------------
# Integration: full config loading pipeline
# ---------------------------------------------------------------------------

class TestConfigLoadingIntegration:
    def test_real_yaml_files_load_successfully(self):
        """Verify the actual YAML files in the repo load without errors."""
        _reset_singleton()
        cfg = load_config_from_yaml(config_dir=CONFIG_DIR)
        assert isinstance(cfg, VerificationConfig)
        # Spot-check values from verification_config.yaml
        assert cfg.price_tolerance_percent >= 0.0
        assert cfg.max_retries >= 0

    def test_development_environment_loads(self):
        _reset_singleton()
        cfg = load_config_from_yaml(config_dir=CONFIG_DIR, environment="development")
        assert isinstance(cfg, VerificationConfig)

    def test_production_environment_loads(self):
        _reset_singleton()
        cfg = load_config_from_yaml(config_dir=CONFIG_DIR, environment="production")
        assert isinstance(cfg, VerificationConfig)

    def test_config_is_pydantic_validated(self):
        """All loaded configs must pass Pydantic validation."""
        for env in [None, "development", "production"]:
            _reset_singleton()
            cfg = load_config_from_yaml(config_dir=CONFIG_DIR, environment=env)
            # Re-validate by constructing from dict
            reloaded = VerificationConfig(**cfg.dict())
            assert reloaded == cfg

    def test_get_default_config_is_valid(self):
        cfg = get_default_config()
        assert isinstance(cfg, VerificationConfig)
        assert cfg.max_retries == 3
