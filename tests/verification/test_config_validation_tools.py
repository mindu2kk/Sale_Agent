"""
Tests for Task 7.1.5: Configuration validation tools with Pydantic models.

Covers:
- ConfigValidationReport: structure, is_valid flag, format_report()
- validate_config_dict(): valid dicts, invalid dicts, cross-field issues
- validate_config(): existing BinaryVerificationConfig instances
- validate_yaml_file(): missing file, bad YAML, valid YAML
- ConfigValidator: environment-specific constraints, suggestions
"""
from __future__ import annotations

import pytest

from verification.config.binary_verification_config import BinaryVerificationConfig
from verification.config.config_validation_tools import (
    ConfigValidationReport,
    ConfigValidator,
    validate_config,
    validate_config_dict,
    validate_yaml_file,
)
from verification.config.thresholds_config import (
    EscalationThresholds,
    PriceAccuracyThresholds,
    TopicRelevanceThresholds,
)


# ---------------------------------------------------------------------------
# ConfigValidationReport
# ---------------------------------------------------------------------------

class TestConfigValidationReport:
    def test_valid_report_has_no_errors(self):
        report = ConfigValidationReport(is_valid=True)
        assert report.is_valid is True
        assert report.errors == []
        assert report.warnings == []
        assert report.suggestions == []

    def test_invalid_report_has_errors(self):
        report = ConfigValidationReport(is_valid=False, errors=["something broke"])
        assert report.is_valid is False
        assert len(report.errors) == 1

    def test_format_report_shows_valid(self):
        report = ConfigValidationReport(is_valid=True)
        text = report.format_report()
        assert "VALID" in text
        assert "INVALID" not in text

    def test_format_report_shows_invalid(self):
        report = ConfigValidationReport(is_valid=False, errors=["bad field"])
        text = report.format_report()
        assert "INVALID" in text
        assert "bad field" in text

    def test_format_report_shows_warnings(self):
        report = ConfigValidationReport(is_valid=True, warnings=["check this"])
        text = report.format_report()
        assert "check this" in text
        assert "[WARN]" in text

    def test_format_report_shows_suggestions(self):
        report = ConfigValidationReport(is_valid=True, suggestions=["try this"])
        text = report.format_report()
        assert "try this" in text
        assert "[INFO]" in text

    def test_format_report_no_issues_message(self):
        report = ConfigValidationReport(is_valid=True)
        text = report.format_report()
        assert "No issues found" in text


# ---------------------------------------------------------------------------
# validate_config_dict()
# ---------------------------------------------------------------------------

class TestValidateConfigDict:
    def test_empty_dict_is_valid(self):
        """Empty dict uses all defaults — should be valid."""
        report = validate_config_dict({})
        assert report.is_valid is True

    def test_valid_dict_with_custom_weights(self):
        report = validate_config_dict({
            "verification_weights": {
                "price_accuracy": 0.5,
                "policy_authenticity": 0.25,
                "topic_relevance": 0.25,
            }
        })
        assert report.is_valid is True

    def test_invalid_weights_sum_produces_error(self):
        report = validate_config_dict({
            "verification_weights": {
                "price_accuracy": 0.5,
                "policy_authenticity": 0.3,
                "topic_relevance": 0.3,  # sum = 1.1
            }
        })
        assert report.is_valid is False
        assert len(report.errors) > 0

    def test_invalid_price_threshold_ordering_produces_error(self):
        report = validate_config_dict({
            "price_accuracy": {
                "minor_threshold_percent": 15.0,
                "major_threshold_percent": 10.0,  # must be > minor
                "critical_threshold_percent": 30.0,
            }
        })
        assert report.is_valid is False
        assert len(report.errors) > 0

    def test_valid_environment_field(self):
        report = validate_config_dict({"environment": "production"})
        assert report.is_valid is True

    def test_cross_field_warning_tolerance_near_minor(self):
        """pass_tolerance close to minor_threshold should produce a warning."""
        report = validate_config_dict({
            "price_accuracy": {
                "pass_tolerance_percent": 4.5,
                "minor_threshold_percent": 5.0,
                "major_threshold_percent": 15.0,
                "critical_threshold_percent": 30.0,
            }
        })
        assert report.is_valid is True
        # Should warn that tolerance is close to minor threshold
        assert len(report.warnings) > 0

    def test_production_environment_tight_tolerance_no_warning(self):
        report = validate_config_dict({
            "environment": "production",
            "price_accuracy": {"pass_tolerance_percent": 0.5},
        })
        assert report.is_valid is True
        # No warning about tolerance being high
        tolerance_warnings = [w for w in report.warnings if "pass_tolerance_percent" in w and "high" in w]
        assert len(tolerance_warnings) == 0

    def test_production_environment_loose_tolerance_warning(self):
        report = validate_config_dict({
            "environment": "production",
            "price_accuracy": {
                "pass_tolerance_percent": 5.0,
                "minor_threshold_percent": 10.0,
                "major_threshold_percent": 20.0,
                "critical_threshold_percent": 40.0,
            },
        })
        assert report.is_valid is True
        prod_warnings = [w for w in report.warnings if "production" in w]
        assert len(prod_warnings) > 0


# ---------------------------------------------------------------------------
# validate_config()
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_default_config_is_valid(self):
        cfg = BinaryVerificationConfig()
        report = validate_config(cfg)
        assert report.is_valid is True

    def test_default_config_has_no_errors(self):
        cfg = BinaryVerificationConfig()
        report = validate_config(cfg)
        assert report.errors == []

    def test_config_with_environment_checks_constraints(self):
        cfg = BinaryVerificationConfig(environment="production")
        report = validate_config(cfg)
        # Default config has tolerance=1.0 which is fine for production
        assert report.is_valid is True

    def test_config_with_high_tolerance_in_production_warns(self):
        cfg = BinaryVerificationConfig(
            price_accuracy=PriceAccuracyThresholds(
                pass_tolerance_percent=5.0,
                minor_threshold_percent=10.0,
                major_threshold_percent=20.0,
                critical_threshold_percent=40.0,
            ),
            environment="production",
        )
        report = validate_config(cfg)
        assert report.is_valid is True
        prod_warnings = [w for w in report.warnings if "production" in w]
        assert len(prod_warnings) > 0

    def test_config_with_early_termination_disabled_in_production_warns(self):
        cfg = BinaryVerificationConfig(
            escalation=EscalationThresholds(early_termination_enabled=False),
            environment="production",
        )
        report = validate_config(cfg)
        assert report.is_valid is True
        et_warnings = [w for w in report.warnings if "early_termination" in w]
        assert len(et_warnings) > 0

    def test_config_with_high_tolerance_gets_suggestion(self):
        cfg = BinaryVerificationConfig(
            price_accuracy=PriceAccuracyThresholds(
                pass_tolerance_percent=3.0,
                minor_threshold_percent=10.0,
                major_threshold_percent=20.0,
                critical_threshold_percent=40.0,
            )
        )
        report = validate_config(cfg)
        assert report.is_valid is True
        assert len(report.suggestions) > 0

    def test_config_with_early_termination_disabled_gets_suggestion(self):
        cfg = BinaryVerificationConfig(
            escalation=EscalationThresholds(early_termination_enabled=False)
        )
        report = validate_config(cfg)
        assert report.is_valid is True
        et_suggestions = [s for s in report.suggestions if "early_termination" in s]
        assert len(et_suggestions) > 0

    def test_critical_retries_exceeding_major_warns(self):
        cfg = BinaryVerificationConfig(
            escalation=EscalationThresholds(
                max_retries_with_critical=5,
                max_retries_with_major=3,
                max_retries_with_minor=5,
            )
        )
        report = validate_config(cfg)
        assert report.is_valid is True
        retry_warnings = [w for w in report.warnings if "max_retries_with_critical" in w]
        assert len(retry_warnings) > 0

    def test_development_environment_early_termination_enabled_warns(self):
        cfg = BinaryVerificationConfig(
            escalation=EscalationThresholds(early_termination_enabled=True),
            environment="development",
        )
        report = validate_config(cfg)
        assert report.is_valid is True
        dev_warnings = [w for w in report.warnings if "development" in w]
        assert len(dev_warnings) > 0


# ---------------------------------------------------------------------------
# validate_yaml_file()
# ---------------------------------------------------------------------------

class TestValidateYamlFile:
    def test_missing_file_returns_invalid(self, tmp_path):
        report = validate_yaml_file(str(tmp_path / "nonexistent.yaml"))
        assert report.is_valid is False
        assert any("not found" in e for e in report.errors)

    def test_invalid_yaml_returns_error(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [unclosed bracket\n")
        report = validate_yaml_file(str(bad_yaml))
        assert report.is_valid is False
        assert len(report.errors) > 0

    def test_empty_yaml_is_valid(self, tmp_path):
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")
        report = validate_yaml_file(str(empty_yaml))
        assert report.is_valid is True

    def test_valid_yaml_with_price_accuracy(self, tmp_path):
        yaml_content = """
price_accuracy:
  pass_criteria:
    tolerance_percent: 1.0
  thresholds:
    minor_threshold_percent: 5.0
    major_threshold_percent: 15.0
    critical_threshold_percent: 30.0
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        report = validate_yaml_file(str(yaml_file))
        assert report.is_valid is True

    def test_non_mapping_yaml_returns_error(self, tmp_path):
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n")
        report = validate_yaml_file(str(list_yaml))
        assert report.is_valid is False
        assert any("mapping" in e for e in report.errors)

    def test_existing_thresholds_yaml_is_valid(self):
        """The project's own thresholds.yaml should pass validation."""
        from pathlib import Path
        thresholds_yaml = Path(__file__).parent.parent / "config" / "thresholds.yaml"
        if not thresholds_yaml.exists():
            pytest.skip("thresholds.yaml not present")
        report = validate_yaml_file(str(thresholds_yaml))
        assert report.is_valid is True


# ---------------------------------------------------------------------------
# ConfigValidator — direct usage
# ---------------------------------------------------------------------------

class TestConfigValidator:
    def setup_method(self):
        self.validator = ConfigValidator()

    def test_validate_dict_returns_report(self):
        report = self.validator.validate_dict({})
        assert isinstance(report, ConfigValidationReport)

    def test_validate_instance_returns_report(self):
        cfg = BinaryVerificationConfig()
        report = self.validator.validate_instance(cfg)
        assert isinstance(report, ConfigValidationReport)

    def test_validate_yaml_returns_report(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        report = self.validator.validate_yaml(str(yaml_file))
        assert isinstance(report, ConfigValidationReport)

    def test_staging_environment_constraints(self):
        cfg = BinaryVerificationConfig(
            price_accuracy=PriceAccuracyThresholds(
                pass_tolerance_percent=5.0,
                minor_threshold_percent=10.0,
                major_threshold_percent=20.0,
                critical_threshold_percent=40.0,
            ),
            environment="staging",
        )
        report = self.validator.validate_instance(cfg)
        assert report.is_valid is True
        staging_warnings = [w for w in report.warnings if "staging" in w]
        assert len(staging_warnings) > 0

    def test_testing_environment_early_termination_enabled_warns(self):
        cfg = BinaryVerificationConfig(
            escalation=EscalationThresholds(early_termination_enabled=True),
            environment="testing",
        )
        report = self.validator.validate_instance(cfg)
        assert report.is_valid is True
        test_warnings = [w for w in report.warnings if "testing" in w]
        assert len(test_warnings) > 0

    def test_missing_weight_keys_warns(self):
        cfg = BinaryVerificationConfig(
            verification_weights={"price_accuracy": 1.0}
        )
        report = self.validator.validate_instance(cfg)
        assert report.is_valid is True
        weight_warnings = [w for w in report.warnings if "missing" in w.lower()]
        assert len(weight_warnings) > 0

    def test_format_report_is_string(self):
        cfg = BinaryVerificationConfig()
        report = validate_config(cfg)
        text = report.format_report()
        assert isinstance(text, str)
        assert len(text) > 0
