"""
Tests for Task 7.1.1: BinaryVerificationConfig Pydantic schema.

Covers:
- BinaryVerificationConfig default values and Pydantic validation
- Sub-model validators (price thresholds ordering, weights sum, etc.)
- is_binary_pass() convenience method
- get_max_retries() and should_terminate_early() delegation
- validate_configuration() consistency warnings
- Environment-specific overrides via get_binary_verification_config()
- YAML loading via load_binary_verification_config()
- Round-trip conversion to/from VerificationThresholdsConfig
- Environment variable auto-detection
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from verification.config.binary_verification_config import (
    BinaryVerificationConfig,
    EscalationThresholds,
    IssueSeverity,
    PolicyAuthenticityThresholds,
    PriceAccuracyThresholds,
    TimeoutConfig,
    TopicRelevanceThresholds,
    get_binary_verification_config,
    load_binary_verification_config,
)
from verification.config.thresholds_config import VerificationThresholdsConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).parent.parent / "config"
THRESHOLDS_YAML = CONFIG_DIR / "thresholds.yaml"


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestBinaryVerificationConfigDefaults:
    def test_creates_with_defaults(self):
        cfg = BinaryVerificationConfig()
        assert isinstance(cfg.price_accuracy, PriceAccuracyThresholds)
        assert isinstance(cfg.policy_authenticity, PolicyAuthenticityThresholds)
        assert isinstance(cfg.topic_relevance, TopicRelevanceThresholds)
        assert isinstance(cfg.escalation, EscalationThresholds)
        assert isinstance(cfg.timeouts, TimeoutConfig)

    def test_default_price_tolerance(self):
        cfg = BinaryVerificationConfig()
        assert cfg.price_accuracy.pass_tolerance_percent == 1.0

    def test_default_policy_fabricated_severity(self):
        cfg = BinaryVerificationConfig()
        assert cfg.policy_authenticity.fabricated_policy_severity == IssueSeverity.CRITICAL

    def test_default_relevance_pass_threshold(self):
        cfg = BinaryVerificationConfig()
        assert cfg.topic_relevance.pass_coverage_threshold == 0.7

    def test_default_max_retries_critical(self):
        cfg = BinaryVerificationConfig()
        assert cfg.escalation.max_retries_with_critical == 1

    def test_default_weights_sum_to_one(self):
        cfg = BinaryVerificationConfig()
        total = sum(cfg.verification_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_default_environment_is_none(self):
        cfg = BinaryVerificationConfig()
        assert cfg.environment is None


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------

class TestBinaryVerificationConfigValidation:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(Exception):
            BinaryVerificationConfig(
                verification_weights={
                    "price_accuracy": 0.5,
                    "policy_authenticity": 0.3,
                    "topic_relevance": 0.3,  # sum = 1.1
                }
            )

    def test_valid_custom_weights(self):
        cfg = BinaryVerificationConfig(
            verification_weights={
                "price_accuracy": 0.5,
                "policy_authenticity": 0.25,
                "topic_relevance": 0.25,
            }
        )
        assert cfg.verification_weights["price_accuracy"] == 0.5

    def test_price_threshold_ordering_enforced(self):
        """major must exceed minor; critical must exceed major."""
        with pytest.raises(Exception):
            BinaryVerificationConfig(
                price_accuracy=PriceAccuracyThresholds(
                    minor_threshold_percent=15.0,
                    major_threshold_percent=10.0,  # invalid: < minor
                    critical_threshold_percent=30.0,
                )
            )

    def test_relevance_threshold_ordering_enforced(self):
        with pytest.raises(Exception):
            BinaryVerificationConfig(
                topic_relevance=TopicRelevanceThresholds(
                    minor_coverage_threshold=0.5,
                    major_coverage_threshold=0.8,  # invalid: > minor
                    critical_coverage_threshold=0.3,
                )
            )

    def test_validate_assignment_enabled(self):
        cfg = BinaryVerificationConfig()
        cfg.environment = "production"
        assert cfg.environment == "production"

    def test_environment_field_accepts_string(self):
        cfg = BinaryVerificationConfig(environment="testing")
        assert cfg.environment == "testing"


# ---------------------------------------------------------------------------
# validate_configuration() warnings
# ---------------------------------------------------------------------------

class TestValidateConfiguration:
    def test_default_config_has_no_warnings(self):
        cfg = BinaryVerificationConfig()
        warnings = cfg.validate_configuration()
        assert warnings == []

    def test_warns_when_tolerance_exceeds_minor_threshold(self):
        cfg = BinaryVerificationConfig()
        # Manually set an inconsistent value after construction
        cfg.price_accuracy = PriceAccuracyThresholds(
            pass_tolerance_percent=6.0,   # >= minor_threshold_percent (5.0)
            minor_threshold_percent=5.0,
            major_threshold_percent=15.0,
            critical_threshold_percent=30.0,
        )
        warnings = cfg.validate_configuration()
        assert len(warnings) > 0
        assert any("tolerance" in w.lower() for w in warnings)

    def test_warns_when_pass_coverage_below_major_threshold(self):
        cfg = BinaryVerificationConfig()
        cfg.topic_relevance = TopicRelevanceThresholds(
            pass_coverage_threshold=0.4,   # < major_coverage_threshold (0.5)
            minor_coverage_threshold=0.8,
            major_coverage_threshold=0.5,
            critical_coverage_threshold=0.3,
        )
        warnings = cfg.validate_configuration()
        assert len(warnings) > 0
        assert any("coverage" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# is_binary_pass()
# ---------------------------------------------------------------------------

class TestIsBinaryPass:
    def setup_method(self):
        self.cfg = BinaryVerificationConfig()

    def test_all_pass_returns_true(self):
        assert self.cfg.is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.9,
            relevance_empathy=0.8,
        ) is True

    def test_price_fail_returns_false(self):
        assert self.cfg.is_binary_pass(
            price_deviation_percent=5.0,   # > 1% tolerance → FAIL
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.9,
        ) is False

    def test_fabricated_policy_returns_false(self):
        assert self.cfg.is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=True,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.9,
        ) is False

    def test_low_relevance_returns_false(self):
        assert self.cfg.is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.5,   # < 0.7 threshold → FAIL
        ) is False

    def test_missing_citation_returns_false(self):
        assert self.cfg.is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=False,  # citation required by default
            relevance_coverage=0.9,
        ) is False


# ---------------------------------------------------------------------------
# Delegation helpers
# ---------------------------------------------------------------------------

class TestDelegationHelpers:
    def setup_method(self):
        self.cfg = BinaryVerificationConfig()

    def test_get_max_retries_critical(self):
        assert self.cfg.get_max_retries(critical_count=1, major_count=0, minor_count=0) == 1

    def test_get_max_retries_major(self):
        assert self.cfg.get_max_retries(critical_count=0, major_count=2, minor_count=0) == 3

    def test_get_max_retries_minor_only(self):
        assert self.cfg.get_max_retries(critical_count=0, major_count=0, minor_count=3) == 5

    def test_should_terminate_early_below_threshold(self):
        assert self.cfg.should_terminate_early(critical_count=1) is False

    def test_should_terminate_early_at_threshold(self):
        # default multiple_critical_threshold = 3
        assert self.cfg.should_terminate_early(critical_count=3) is True


# ---------------------------------------------------------------------------
# Round-trip conversion
# ---------------------------------------------------------------------------

class TestRoundTripConversion:
    def test_to_thresholds_config(self):
        cfg = BinaryVerificationConfig()
        tc = cfg.to_thresholds_config()
        assert isinstance(tc, VerificationThresholdsConfig)
        assert tc.price_accuracy.pass_tolerance_percent == cfg.price_accuracy.pass_tolerance_percent

    def test_from_thresholds_config(self):
        tc = VerificationThresholdsConfig()
        cfg = BinaryVerificationConfig.from_thresholds_config(tc, environment="testing")
        assert cfg.environment == "testing"
        assert cfg.price_accuracy.pass_tolerance_percent == tc.price_accuracy.pass_tolerance_percent

    def test_round_trip_preserves_values(self):
        original = BinaryVerificationConfig()
        tc = original.to_thresholds_config()
        restored = BinaryVerificationConfig.from_thresholds_config(tc)
        assert restored.price_accuracy == original.price_accuracy
        assert restored.escalation == original.escalation
        assert restored.verification_weights == original.verification_weights


# ---------------------------------------------------------------------------
# get_binary_verification_config() — environment overrides
# ---------------------------------------------------------------------------

class TestGetBinaryVerificationConfig:
    def test_returns_default_when_no_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERIFICATION_ENV", None)
            os.environ.pop("ENVIRONMENT", None)
            cfg = get_binary_verification_config()
        assert isinstance(cfg, BinaryVerificationConfig)
        assert cfg.price_accuracy.pass_tolerance_percent == 1.0

    def test_development_overrides_applied(self):
        cfg = get_binary_verification_config(environment="development")
        assert cfg.price_accuracy.pass_tolerance_percent == 2.0
        assert cfg.escalation.early_termination_enabled is False
        assert cfg.environment == "development"

    def test_production_overrides_applied(self):
        cfg = get_binary_verification_config(environment="production")
        assert cfg.price_accuracy.pass_tolerance_percent == 0.5
        assert cfg.escalation.max_critical_issues_before_escalation == 1
        assert cfg.environment == "production"

    def test_testing_overrides_applied(self):
        cfg = get_binary_verification_config(environment="testing")
        assert cfg.escalation.early_termination_enabled is False
        assert cfg.escalation.max_retries_with_critical == 0

    def test_unknown_environment_returns_defaults(self):
        cfg = get_binary_verification_config(environment="nonexistent")
        assert cfg.price_accuracy.pass_tolerance_percent == 1.0
        assert cfg.environment == "nonexistent"

    def test_auto_detects_verification_env_var(self):
        with patch.dict(os.environ, {"VERIFICATION_ENV": "production"}):
            cfg = get_binary_verification_config()
        assert cfg.environment == "production"
        assert cfg.price_accuracy.pass_tolerance_percent == 0.5

    def test_auto_detects_environment_var(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "VERIFICATION_ENV": ""}):
            cfg = get_binary_verification_config()
        # VERIFICATION_ENV is empty string → falls through to ENVIRONMENT
        assert cfg.environment == "development"


# ---------------------------------------------------------------------------
# load_binary_verification_config() — YAML loading
# ---------------------------------------------------------------------------

class TestLoadBinaryVerificationConfig:
    def test_loads_from_default_thresholds_yaml(self):
        if not THRESHOLDS_YAML.exists():
            pytest.skip("thresholds.yaml not present")
        cfg = load_binary_verification_config()
        assert isinstance(cfg, BinaryVerificationConfig)

    def test_loads_from_explicit_path(self):
        if not THRESHOLDS_YAML.exists():
            pytest.skip("thresholds.yaml not present")
        cfg = load_binary_verification_config(config_path=str(THRESHOLDS_YAML))
        assert isinstance(cfg, BinaryVerificationConfig)

    def test_missing_yaml_falls_back_to_defaults(self, tmp_path):
        cfg = load_binary_verification_config(
            config_path=str(tmp_path / "nonexistent.yaml")
        )
        assert isinstance(cfg, BinaryVerificationConfig)
        assert cfg.price_accuracy.pass_tolerance_percent == 1.0

    def test_yaml_with_environment_override(self, tmp_path):
        yaml_content = """
price_accuracy:
  pass_criteria:
    tolerance_percent: 1.0
"""
        yaml_file = tmp_path / "thresholds.yaml"
        yaml_file.write_text(yaml_content)
        cfg = load_binary_verification_config(
            config_path=str(yaml_file),
            environment="production",
        )
        # production override sets tolerance to 0.5
        assert cfg.price_accuracy.pass_tolerance_percent == 0.5
        assert cfg.environment == "production"

    def test_env_var_overrides_applied(self, tmp_path):
        yaml_file = tmp_path / "thresholds.yaml"
        yaml_file.write_text("")
        with patch.dict(
            os.environ,
            {"VERIFICATION_THRESHOLDS_PRICE_TOLERANCE": "0.25"},
        ):
            cfg = load_binary_verification_config(config_path=str(yaml_file))
        assert cfg.price_accuracy.pass_tolerance_percent == 0.25


# ---------------------------------------------------------------------------
# Integration: severity classification end-to-end
# ---------------------------------------------------------------------------

class TestSeverityClassificationIntegration:
    def setup_method(self):
        self.cfg = BinaryVerificationConfig()

    def test_critical_price_deviation(self):
        sev = self.cfg.price_accuracy.classify_price_deviation(35.0)
        assert sev == IssueSeverity.CRITICAL

    def test_major_price_deviation(self):
        sev = self.cfg.price_accuracy.classify_price_deviation(20.0)
        assert sev == IssueSeverity.MAJOR

    def test_minor_price_deviation(self):
        sev = self.cfg.price_accuracy.classify_price_deviation(3.0)
        assert sev == IssueSeverity.MINOR

    def test_fabricated_policy_is_critical(self):
        sev = self.cfg.policy_authenticity.classify_policy_issue(
            is_fabricated=True,
            is_inaccurate=False,
            is_incomplete=False,
        )
        assert sev == IssueSeverity.CRITICAL

    def test_low_relevance_is_critical(self):
        sev = self.cfg.topic_relevance.classify_relevance_issue(coverage_ratio=0.1)
        assert sev == IssueSeverity.CRITICAL

    def test_escalation_on_fabricated_policy(self):
        should = self.cfg.escalation.should_escalate_immediately(
            critical_count=1,
            major_count=0,
            total_count=1,
            has_fabricated_policy=True,
        )
        assert should is True
