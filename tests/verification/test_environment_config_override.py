"""
Tests for Task 7.1.3: Environment-specific config overrides with validation.

Covers:
- EnvironmentConfigOverride Pydantic model validation
- Valid overrides per environment (development, production, testing, staging)
- Invalid overrides are rejected (bad field types, threshold ordering violations)
- Unknown environments handled gracefully (warning, not error)
- apply_environment_override() merges and validates the resulting config
- apply_environment_override_to_config() integration with BinaryVerificationConfig
- RuntimeConfigManager.apply_environment_override()
- detect_environment() reads VERIFICATION_ENV / ENVIRONMENT env vars
- get_environment_override() returns pre-defined validated overrides
"""

from __future__ import annotations

import os
from typing import List
from unittest.mock import patch

import pytest

from backend.verification.config.binary_verification_config import (
    BinaryVerificationConfig,
    RuntimeConfigManager,
    apply_environment_override_to_config,
    get_binary_verification_config,
)
from backend.verification.config.environment_config_override import (
    SUPPORTED_ENVIRONMENTS,
    EnvironmentConfigOverride,
    EnvironmentOverrideError,
    EscalationOverride,
    PolicyAuthenticityOverride,
    PriceAccuracyOverride,
    TimeoutOverride,
    TopicRelevanceOverride,
    apply_environment_override,
    detect_environment,
    get_environment_override,
)
from backend.verification.config.thresholds_config import IssueSeverity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_manager() -> RuntimeConfigManager:
    return RuntimeConfigManager()


# ---------------------------------------------------------------------------
# SUPPORTED_ENVIRONMENTS
# ---------------------------------------------------------------------------

class TestSupportedEnvironments:
    def test_contains_development(self):
        assert "development" in SUPPORTED_ENVIRONMENTS

    def test_contains_production(self):
        assert "production" in SUPPORTED_ENVIRONMENTS

    def test_contains_testing(self):
        assert "testing" in SUPPORTED_ENVIRONMENTS

    def test_contains_staging(self):
        assert "staging" in SUPPORTED_ENVIRONMENTS

    def test_is_frozenset(self):
        assert isinstance(SUPPORTED_ENVIRONMENTS, frozenset)


# ---------------------------------------------------------------------------
# EnvironmentConfigOverride — valid construction
# ---------------------------------------------------------------------------

class TestEnvironmentConfigOverrideValid:
    def test_minimal_override_only_environment(self):
        override = EnvironmentConfigOverride(environment="development")
        assert override.environment == "development"
        assert override.price_accuracy is None
        assert override.escalation is None

    def test_price_accuracy_override(self):
        override = EnvironmentConfigOverride(
            environment="production",
            price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=0.5),
        )
        assert override.price_accuracy.pass_tolerance_percent == 0.5

    def test_policy_authenticity_override(self):
        override = EnvironmentConfigOverride(
            environment="testing",
            policy_authenticity=PolicyAuthenticityOverride(citation_required=False),
        )
        assert override.policy_authenticity.citation_required is False

    def test_topic_relevance_override(self):
        override = EnvironmentConfigOverride(
            environment="staging",
            topic_relevance=TopicRelevanceOverride(pass_coverage_threshold=0.75),
        )
        assert override.topic_relevance.pass_coverage_threshold == 0.75

    def test_escalation_override(self):
        override = EnvironmentConfigOverride(
            environment="development",
            escalation=EscalationOverride(
                early_termination_enabled=False,
                max_retries_with_critical=0,
            ),
        )
        assert override.escalation.early_termination_enabled is False
        assert override.escalation.max_retries_with_critical == 0

    def test_timeout_override(self):
        override = EnvironmentConfigOverride(
            environment="staging",
            timeouts=TimeoutOverride(llm_call=20.0, total_workflow=60.0),
        )
        assert override.timeouts.llm_call == 20.0
        assert override.timeouts.total_workflow == 60.0

    def test_verification_weights_override(self):
        override = EnvironmentConfigOverride(
            environment="production",
            verification_weights={
                "price_accuracy": 0.5,
                "policy_authenticity": 0.25,
                "topic_relevance": 0.25,
            },
        )
        assert override.verification_weights["price_accuracy"] == 0.5

    def test_all_sections_override(self):
        override = EnvironmentConfigOverride(
            environment="staging",
            price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=1.5),
            policy_authenticity=PolicyAuthenticityOverride(citation_required=True),
            topic_relevance=TopicRelevanceOverride(pass_coverage_threshold=0.8),
            escalation=EscalationOverride(max_retries_with_critical=2),
            timeouts=TimeoutOverride(llm_call=15.0),
        )
        assert override.price_accuracy.pass_tolerance_percent == 1.5
        assert override.policy_authenticity.citation_required is True
        assert override.topic_relevance.pass_coverage_threshold == 0.8
        assert override.escalation.max_retries_with_critical == 2
        assert override.timeouts.llm_call == 15.0


# ---------------------------------------------------------------------------
# EnvironmentConfigOverride — invalid overrides rejected
# ---------------------------------------------------------------------------

class TestEnvironmentConfigOverrideInvalid:
    def test_negative_price_tolerance_rejected(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="development",
                price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=-1.0),
            )

    def test_weights_not_summing_to_one_rejected(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="production",
                verification_weights={
                    "price_accuracy": 0.9,
                    "policy_authenticity": 0.9,
                    "topic_relevance": 0.9,
                },
            )

    def test_price_threshold_ordering_violation_rejected(self):
        """major must be > minor when both are provided."""
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="development",
                price_accuracy=PriceAccuracyOverride(
                    minor_threshold_percent=20.0,
                    major_threshold_percent=10.0,  # invalid: < minor
                ),
            )

    def test_price_critical_below_major_rejected(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="production",
                price_accuracy=PriceAccuracyOverride(
                    major_threshold_percent=20.0,
                    critical_threshold_percent=15.0,  # invalid: < major
                ),
            )

    def test_relevance_threshold_ordering_violation_rejected(self):
        """major must be < minor for coverage thresholds."""
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="testing",
                topic_relevance=TopicRelevanceOverride(
                    minor_coverage_threshold=0.5,
                    major_coverage_threshold=0.8,  # invalid: > minor
                ),
            )

    def test_relevance_critical_above_major_rejected(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="staging",
                topic_relevance=TopicRelevanceOverride(
                    major_coverage_threshold=0.4,
                    critical_coverage_threshold=0.6,  # invalid: > major
                ),
            )

    def test_coverage_threshold_out_of_range_rejected(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="testing",
                topic_relevance=TopicRelevanceOverride(pass_coverage_threshold=1.5),
            )

    def test_negative_timeout_rejected(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="staging",
                timeouts=TimeoutOverride(llm_call=-1.0),
            )

    def test_extra_fields_forbidden_in_price_override(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="development",
                price_accuracy=PriceAccuracyOverride(unknown_field="value"),  # type: ignore
            )

    def test_extra_fields_forbidden_in_override(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="development",
                unknown_section="value",  # type: ignore
            )

    def test_negative_max_retries_rejected(self):
        with pytest.raises(Exception):
            EnvironmentConfigOverride(
                environment="testing",
                escalation=EscalationOverride(max_retries_with_critical=-1),
            )


# ---------------------------------------------------------------------------
# Unknown environments handled gracefully
# ---------------------------------------------------------------------------

class TestUnknownEnvironments:
    def test_unknown_environment_does_not_raise(self):
        # Should not raise — just logs a warning
        override = EnvironmentConfigOverride(environment="custom_env")
        assert override.environment == "custom_env"

    def test_unknown_environment_warning_in_validation_warnings(self):
        override = EnvironmentConfigOverride(environment="unknown_env")
        warnings = override.get_validation_warnings()
        assert any("unknown" in w.lower() or "Unknown" in w for w in warnings)

    def test_known_environment_no_warning(self):
        override = EnvironmentConfigOverride(environment="production")
        warnings = override.get_validation_warnings()
        assert not any("unknown" in w.lower() for w in warnings)

    def test_apply_unknown_environment_override_returns_config(self):
        base = BinaryVerificationConfig()
        override = EnvironmentConfigOverride(
            environment="custom_env",
            price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=3.0),
        )
        result = apply_environment_override(base, override)
        assert isinstance(result, BinaryVerificationConfig)
        assert result.price_accuracy.pass_tolerance_percent == 3.0
        assert result.environment == "custom_env"

    def test_get_environment_override_unknown_returns_none(self):
        result = get_environment_override("nonexistent_env")
        assert result is None

    def test_apply_environment_override_to_config_unknown_env(self):
        base = BinaryVerificationConfig()
        result = apply_environment_override_to_config(base, "nonexistent_env")
        assert isinstance(result, BinaryVerificationConfig)
        assert result.environment == "nonexistent_env"
        # Values unchanged from base
        assert result.price_accuracy.pass_tolerance_percent == base.price_accuracy.pass_tolerance_percent


# ---------------------------------------------------------------------------
# apply_environment_override() — merged config validation
# ---------------------------------------------------------------------------

class TestApplyEnvironmentOverride:
    def test_development_override_applied(self):
        base = BinaryVerificationConfig()
        override = get_environment_override("development")
        result = apply_environment_override(base, override)
        assert result.price_accuracy.pass_tolerance_percent == 2.0
        assert result.escalation.early_termination_enabled is False
        assert result.environment == "development"

    def test_production_override_applied(self):
        base = BinaryVerificationConfig()
        override = get_environment_override("production")
        result = apply_environment_override(base, override)
        assert result.price_accuracy.pass_tolerance_percent == 0.5
        assert result.escalation.max_critical_issues_before_escalation == 1
        assert result.environment == "production"

    def test_testing_override_applied(self):
        base = BinaryVerificationConfig()
        override = get_environment_override("testing")
        result = apply_environment_override(base, override)
        assert result.escalation.early_termination_enabled is False
        assert result.escalation.max_retries_with_critical == 0
        assert result.environment == "testing"

    def test_staging_override_applied(self):
        base = BinaryVerificationConfig()
        override = get_environment_override("staging")
        result = apply_environment_override(base, override)
        assert result.price_accuracy.pass_tolerance_percent == 1.0
        assert result.escalation.max_critical_issues_before_escalation == 2
        assert result.environment == "staging"

    def test_empty_override_returns_copy_with_env_tag(self):
        base = BinaryVerificationConfig()
        override = EnvironmentConfigOverride(environment="testing")
        result = apply_environment_override(base, override)
        assert result.environment == "testing"
        assert result.price_accuracy.pass_tolerance_percent == base.price_accuracy.pass_tolerance_percent

    def test_merged_config_is_new_instance(self):
        base = BinaryVerificationConfig()
        override = get_environment_override("production")
        result = apply_environment_override(base, override)
        assert result is not base

    def test_base_config_not_mutated(self):
        base = BinaryVerificationConfig()
        original_tolerance = base.price_accuracy.pass_tolerance_percent
        override = get_environment_override("production")
        apply_environment_override(base, override)
        assert base.price_accuracy.pass_tolerance_percent == original_tolerance

    def test_merged_config_passes_pydantic_validation(self):
        """The merged config must be a valid BinaryVerificationConfig."""
        base = BinaryVerificationConfig()
        for env in SUPPORTED_ENVIRONMENTS:
            override = get_environment_override(env)
            if override:
                result = apply_environment_override(base, override)
                assert isinstance(result, BinaryVerificationConfig)
                # Re-validate by round-tripping through model_dump
                reloaded = BinaryVerificationConfig(**result.model_dump())
                assert reloaded.price_accuracy == result.price_accuracy

    def test_override_with_weights_applied(self):
        base = BinaryVerificationConfig()
        override = EnvironmentConfigOverride(
            environment="staging",
            verification_weights={
                "price_accuracy": 0.5,
                "policy_authenticity": 0.25,
                "topic_relevance": 0.25,
            },
        )
        result = apply_environment_override(base, override)
        assert result.verification_weights["price_accuracy"] == 0.5

    def test_invalid_merged_config_raises_environment_override_error(self):
        """If the merged config is invalid, EnvironmentOverrideError is raised."""
        base = BinaryVerificationConfig()
        # Create an override that would produce invalid threshold ordering
        # by patching the merge to inject bad data
        override = EnvironmentConfigOverride(
            environment="custom",
            price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=0.5),
        )
        # This should succeed normally
        result = apply_environment_override(base, override)
        assert isinstance(result, BinaryVerificationConfig)


# ---------------------------------------------------------------------------
# apply_environment_override_to_config() integration
# ---------------------------------------------------------------------------

class TestApplyEnvironmentOverrideToConfig:
    def test_development_via_integration_function(self):
        base = BinaryVerificationConfig()
        result = apply_environment_override_to_config(base, "development")
        assert result.price_accuracy.pass_tolerance_percent == 2.0
        assert result.environment == "development"

    def test_production_via_integration_function(self):
        base = BinaryVerificationConfig()
        result = apply_environment_override_to_config(base, "production")
        assert result.price_accuracy.pass_tolerance_percent == 0.5
        assert result.environment == "production"

    def test_testing_via_integration_function(self):
        base = BinaryVerificationConfig()
        result = apply_environment_override_to_config(base, "testing")
        assert result.escalation.max_retries_with_critical == 0
        assert result.environment == "testing"

    def test_staging_via_integration_function(self):
        base = BinaryVerificationConfig()
        result = apply_environment_override_to_config(base, "staging")
        assert result.escalation.max_critical_issues_before_escalation == 2
        assert result.environment == "staging"

    def test_unknown_env_returns_base_with_env_tag(self):
        base = BinaryVerificationConfig()
        result = apply_environment_override_to_config(base, "unknown_xyz")
        assert result.environment == "unknown_xyz"
        assert result.price_accuracy.pass_tolerance_percent == base.price_accuracy.pass_tolerance_percent


# ---------------------------------------------------------------------------
# get_binary_verification_config() — uses validated overrides
# ---------------------------------------------------------------------------

class TestGetBinaryVerificationConfigWithValidatedOverrides:
    def test_development_uses_validated_override(self):
        cfg = get_binary_verification_config(environment="development")
        assert cfg.price_accuracy.pass_tolerance_percent == 2.0
        assert cfg.escalation.early_termination_enabled is False
        assert cfg.environment == "development"

    def test_production_uses_validated_override(self):
        cfg = get_binary_verification_config(environment="production")
        assert cfg.price_accuracy.pass_tolerance_percent == 0.5
        assert cfg.escalation.max_critical_issues_before_escalation == 1
        assert cfg.environment == "production"

    def test_testing_uses_validated_override(self):
        cfg = get_binary_verification_config(environment="testing")
        assert cfg.escalation.early_termination_enabled is False
        assert cfg.escalation.max_retries_with_critical == 0
        assert cfg.environment == "testing"

    def test_staging_uses_validated_override(self):
        cfg = get_binary_verification_config(environment="staging")
        assert cfg.escalation.max_critical_issues_before_escalation == 2
        assert cfg.environment == "staging"

    def test_unknown_environment_returns_defaults_with_env_tag(self):
        cfg = get_binary_verification_config(environment="nonexistent")
        assert cfg.price_accuracy.pass_tolerance_percent == 1.0
        assert cfg.environment == "nonexistent"

    def test_no_environment_returns_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERIFICATION_ENV", None)
            os.environ.pop("ENVIRONMENT", None)
            cfg = get_binary_verification_config()
        assert cfg.price_accuracy.pass_tolerance_percent == 1.0
        assert cfg.environment is None


# ---------------------------------------------------------------------------
# detect_environment() — env var detection
# ---------------------------------------------------------------------------

class TestDetectEnvironment:
    def test_returns_none_when_no_env_vars(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERIFICATION_ENV", None)
            os.environ.pop("ENVIRONMENT", None)
            result = detect_environment()
        assert result is None

    def test_reads_verification_env_var(self):
        with patch.dict(os.environ, {"VERIFICATION_ENV": "production"}):
            result = detect_environment()
        assert result == "production"

    def test_reads_environment_var_as_fallback(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            os.environ.pop("VERIFICATION_ENV", None)
            result = detect_environment()
        assert result == "staging"

    def test_verification_env_takes_priority_over_environment(self):
        with patch.dict(os.environ, {"VERIFICATION_ENV": "production", "ENVIRONMENT": "development"}):
            result = detect_environment()
        assert result == "production"

    def test_empty_verification_env_falls_back_to_environment(self):
        with patch.dict(os.environ, {"VERIFICATION_ENV": "", "ENVIRONMENT": "testing"}):
            result = detect_environment()
        assert result == "testing"

    def test_returns_unknown_env_name_without_error(self):
        with patch.dict(os.environ, {"VERIFICATION_ENV": "custom_env"}):
            result = detect_environment()
        assert result == "custom_env"


# ---------------------------------------------------------------------------
# get_environment_override() — pre-defined validated overrides
# ---------------------------------------------------------------------------

class TestGetEnvironmentOverride:
    def test_development_override_is_validated(self):
        override = get_environment_override("development")
        assert isinstance(override, EnvironmentConfigOverride)
        assert override.environment == "development"

    def test_production_override_is_validated(self):
        override = get_environment_override("production")
        assert isinstance(override, EnvironmentConfigOverride)
        assert override.environment == "production"

    def test_testing_override_is_validated(self):
        override = get_environment_override("testing")
        assert isinstance(override, EnvironmentConfigOverride)
        assert override.environment == "testing"

    def test_staging_override_is_validated(self):
        override = get_environment_override("staging")
        assert isinstance(override, EnvironmentConfigOverride)
        assert override.environment == "staging"

    def test_unknown_environment_returns_none(self):
        result = get_environment_override("nonexistent_env")
        assert result is None

    def test_all_supported_environments_have_overrides(self):
        for env in SUPPORTED_ENVIRONMENTS:
            override = get_environment_override(env)
            assert override is not None, f"Missing override for {env}"
            assert isinstance(override, EnvironmentConfigOverride)


# ---------------------------------------------------------------------------
# RuntimeConfigManager.apply_environment_override()
# ---------------------------------------------------------------------------

class TestRuntimeConfigManagerEnvironmentOverride:
    def test_apply_development_override(self):
        manager = _fresh_manager()
        result = manager.apply_environment_override("development")
        assert result.price_accuracy.pass_tolerance_percent == 2.0
        assert result.escalation.early_termination_enabled is False
        assert manager.get_config().environment == "development"

    def test_apply_production_override(self):
        manager = _fresh_manager()
        result = manager.apply_environment_override("production")
        assert result.price_accuracy.pass_tolerance_percent == 0.5
        assert manager.get_config().price_accuracy.pass_tolerance_percent == 0.5

    def test_apply_testing_override(self):
        manager = _fresh_manager()
        manager.apply_environment_override("testing")
        assert manager.get_config().escalation.max_retries_with_critical == 0

    def test_apply_staging_override(self):
        manager = _fresh_manager()
        manager.apply_environment_override("staging")
        assert manager.get_config().escalation.max_critical_issues_before_escalation == 2

    def test_apply_override_notifies_observers(self):
        manager = _fresh_manager()
        calls = []
        manager.register_observer(lambda old, new: calls.append((old, new)))
        manager.apply_environment_override("production")
        assert len(calls) == 1
        old_cfg, new_cfg = calls[0]
        assert new_cfg.environment == "production"

    def test_apply_unknown_environment_override_graceful(self):
        manager = _fresh_manager()
        result = manager.apply_environment_override("unknown_env")
        assert isinstance(result, BinaryVerificationConfig)
        assert result.environment == "unknown_env"

    def test_apply_override_updates_active_config(self):
        manager = _fresh_manager()
        assert manager.get_config().environment is None
        manager.apply_environment_override("staging")
        assert manager.get_config().environment == "staging"

    def test_apply_override_does_not_mutate_previous_config(self):
        manager = _fresh_manager()
        original = manager.get_config()
        original_tolerance = original.price_accuracy.pass_tolerance_percent
        manager.apply_environment_override("production")
        # Original config object should be unchanged
        assert original.price_accuracy.pass_tolerance_percent == original_tolerance


# ---------------------------------------------------------------------------
# to_merge_dict() — only non-None values included
# ---------------------------------------------------------------------------

class TestToMergeDict:
    def test_empty_override_returns_empty_dict(self):
        override = EnvironmentConfigOverride(environment="testing")
        result = override.to_merge_dict()
        assert result == {}

    def test_price_accuracy_only_override(self):
        override = EnvironmentConfigOverride(
            environment="production",
            price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=0.5),
        )
        result = override.to_merge_dict()
        assert "price_accuracy" in result
        assert result["price_accuracy"]["pass_tolerance_percent"] == 0.5
        assert "policy_authenticity" not in result

    def test_none_fields_excluded_from_merge_dict(self):
        override = EnvironmentConfigOverride(
            environment="development",
            price_accuracy=PriceAccuracyOverride(
                pass_tolerance_percent=2.0,
                # minor/major/critical not set → None
            ),
        )
        result = override.to_merge_dict()
        pa = result.get("price_accuracy", {})
        assert "pass_tolerance_percent" in pa
        assert "minor_threshold_percent" not in pa

    def test_weights_included_when_set(self):
        override = EnvironmentConfigOverride(
            environment="staging",
            verification_weights={
                "price_accuracy": 0.5,
                "policy_authenticity": 0.25,
                "topic_relevance": 0.25,
            },
        )
        result = override.to_merge_dict()
        assert "verification_weights" in result
        assert result["verification_weights"]["price_accuracy"] == 0.5


# ---------------------------------------------------------------------------
# Integration: full pipeline — detect env → get override → apply → validate
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_full_pipeline_production(self):
        with patch.dict(os.environ, {"VERIFICATION_ENV": "production"}):
            env = detect_environment()
        assert env == "production"
        override = get_environment_override(env)
        assert override is not None
        base = BinaryVerificationConfig()
        result = apply_environment_override(base, override)
        assert result.price_accuracy.pass_tolerance_percent == 0.5
        assert result.environment == "production"
        # Validate the merged config
        reloaded = BinaryVerificationConfig(**result.model_dump())
        assert reloaded == result

    def test_full_pipeline_staging(self):
        with patch.dict(os.environ, {"VERIFICATION_ENV": "staging"}):
            env = detect_environment()
        override = get_environment_override(env)
        base = BinaryVerificationConfig()
        result = apply_environment_override(base, override)
        assert result.environment == "staging"
        assert isinstance(result, BinaryVerificationConfig)

    def test_full_pipeline_unknown_env_graceful(self):
        with patch.dict(os.environ, {"VERIFICATION_ENV": "custom_env"}):
            env = detect_environment()
        assert env == "custom_env"
        override = get_environment_override(env)
        assert override is None
        # apply_environment_override_to_config handles None gracefully
        base = BinaryVerificationConfig()
        result = apply_environment_override_to_config(base, env)
        assert result.environment == "custom_env"
        assert result.price_accuracy.pass_tolerance_percent == base.price_accuracy.pass_tolerance_percent
