"""
Configuration Validation Tools - Task 7.1.5

Provides dedicated validation tools for BinaryVerificationConfig:
- ConfigValidationReport: structured Pydantic model with errors, warnings, suggestions
- ConfigValidator: class wrapping all validation logic
- validate_config_dict(): validate a raw dict before constructing BinaryVerificationConfig
- validate_config(): validate an existing BinaryVerificationConfig instance
- validate_yaml_file(): validate a YAML config file
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigValidationReport",
    "ConfigValidator",
    "validate_config_dict",
    "validate_config",
    "validate_yaml_file",
]


# ---------------------------------------------------------------------------
# ConfigValidationReport
# ---------------------------------------------------------------------------

class ConfigValidationReport(BaseModel):
    """Structured report produced by config validation."""

    is_valid: bool = Field(..., description="True if no errors were found")
    errors: List[str] = Field(default_factory=list, description="Fatal validation errors")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")

    def format_report(self) -> str:
        """Return a human-readable, CLI-friendly validation report."""
        lines: List[str] = []
        status = "VALID" if self.is_valid else "INVALID"
        lines.append(f"Config Validation Report: {status}")
        lines.append("=" * 40)

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  [WARN]  {w}")

        if self.suggestions:
            lines.append(f"\nSuggestions ({len(self.suggestions)}):")
            for s in self.suggestions:
                lines.append(f"  [INFO]  {s}")

        if self.is_valid and not self.warnings and not self.suggestions:
            lines.append("\nNo issues found.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------

class ConfigValidator:
    """
    Wraps all validation logic for BinaryVerificationConfig.

    Validates:
    1. Pydantic schema (field types, ranges, weights sum)
    2. Cross-field consistency (threshold ordering, weights sum)
    3. Environment-specific constraints
    4. Suggestions for common misconfigurations
    """

    # Environments that have known stricter constraints
    _STRICT_ENVIRONMENTS = frozenset({"production", "staging"})
    _RELAXED_ENVIRONMENTS = frozenset({"development", "testing"})

    def validate_dict(self, data: Dict[str, Any]) -> ConfigValidationReport:
        """
        Validate a raw config dict before constructing BinaryVerificationConfig.

        Attempts to construct the model and collects all Pydantic errors,
        then runs additional cross-field and environment checks.
        """
        from backend.verification.config.binary_verification_config import BinaryVerificationConfig

        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        # 1. Pydantic schema validation
        config_instance: Optional[BinaryVerificationConfig] = None
        try:
            config_instance = BinaryVerificationConfig(**data)
        except Exception as exc:
            # Collect all Pydantic validation errors
            errors.extend(self._extract_pydantic_errors(exc))

        # 2. Cross-field and environment checks (only if Pydantic passed)
        if config_instance is not None:
            warnings.extend(self._check_cross_field_consistency(config_instance))
            env = data.get("environment") or config_instance.environment
            if env:
                env_warnings, env_errors = self._check_environment_constraints(config_instance, env)
                errors.extend(env_errors)
                warnings.extend(env_warnings)
            suggestions.extend(self._generate_suggestions(config_instance))

        return ConfigValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

    def validate_instance(
        self,
        config: "BinaryVerificationConfig",
    ) -> ConfigValidationReport:
        """
        Validate an existing BinaryVerificationConfig instance.

        Runs cross-field consistency checks, environment constraints,
        and generates improvement suggestions.
        """
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        warnings.extend(self._check_cross_field_consistency(config))

        if config.environment:
            env_warnings, env_errors = self._check_environment_constraints(config, config.environment)
            errors.extend(env_errors)
            warnings.extend(env_warnings)

        suggestions.extend(self._generate_suggestions(config))

        return ConfigValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

    def validate_yaml(self, yaml_path: str) -> ConfigValidationReport:
        """
        Validate a YAML config file.

        Loads the YAML, converts it to a config dict, then validates.
        Returns errors if the file is missing or unparseable.
        """
        path = Path(yaml_path)
        if not path.exists():
            return ConfigValidationReport(
                is_valid=False,
                errors=[f"Config file not found: {yaml_path}"],
            )

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            return ConfigValidationReport(
                is_valid=False,
                errors=[f"YAML parse error in {yaml_path}: {exc}"],
            )

        if not isinstance(raw, dict):
            return ConfigValidationReport(
                is_valid=False,
                errors=[f"YAML file must contain a mapping at the top level, got {type(raw).__name__}"],
            )

        # Convert YAML structure to flat Pydantic-compatible dict
        from backend.verification.config.thresholds_config import _convert_yaml_to_pydantic_structure
        try:
            config_dict = _convert_yaml_to_pydantic_structure(raw)
        except Exception as exc:
            return ConfigValidationReport(
                is_valid=False,
                errors=[f"Failed to parse YAML structure: {exc}"],
            )

        return self.validate_dict(config_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_pydantic_errors(self, exc: Exception) -> List[str]:
        """Extract human-readable error messages from a Pydantic ValidationError."""
        try:
            # Pydantic v2
            errors = exc.errors()  # type: ignore[attr-defined]
            messages = []
            for e in errors:
                loc = " -> ".join(str(x) for x in e.get("loc", []))
                msg = e.get("msg", str(e))
                messages.append(f"{loc}: {msg}" if loc else msg)
            return messages
        except AttributeError:
            return [str(exc)]

    def _check_cross_field_consistency(
        self,
        config: "BinaryVerificationConfig",
    ) -> List[str]:
        """
        Check cross-field consistency rules that Pydantic validators don't cover.

        Returns a list of warning strings.
        """
        warnings: List[str] = []

        # Delegate to the built-in validate_configuration() method
        warnings.extend(config.validate_configuration())

        # Additional cross-field checks
        pa = config.price_accuracy
        # pass_tolerance should be well below minor threshold
        if pa.pass_tolerance_percent > 0 and pa.minor_threshold_percent > 0:
            ratio = pa.pass_tolerance_percent / pa.minor_threshold_percent
            if ratio > 0.8:
                warnings.append(
                    f"price_accuracy.pass_tolerance_percent ({pa.pass_tolerance_percent}) "
                    f"is close to minor_threshold_percent ({pa.minor_threshold_percent}); "
                    "consider widening the gap"
                )

        # Escalation: max_retries_with_critical should be <= max_retries_with_major
        esc = config.escalation
        if esc.max_retries_with_critical > esc.max_retries_with_major:
            warnings.append(
                f"escalation.max_retries_with_critical ({esc.max_retries_with_critical}) "
                f"exceeds max_retries_with_major ({esc.max_retries_with_major}); "
                "critical issues should have fewer retries"
            )

        # Weights: all keys should be present
        expected_keys = {"price_accuracy", "policy_authenticity", "topic_relevance"}
        missing_keys = expected_keys - set(config.verification_weights.keys())
        if missing_keys:
            warnings.append(
                f"verification_weights is missing expected keys: {sorted(missing_keys)}"
            )

        return warnings

    def _check_environment_constraints(
        self,
        config: "BinaryVerificationConfig",
        environment: str,
    ) -> tuple[List[str], List[str]]:
        """
        Validate environment-specific constraints.

        Returns (warnings, errors).
        """
        warnings: List[str] = []
        errors: List[str] = []

        if environment in self._STRICT_ENVIRONMENTS:
            # Production/staging: tolerance must be tight
            if config.price_accuracy.pass_tolerance_percent > 1.0:
                warnings.append(
                    f"[{environment}] price_accuracy.pass_tolerance_percent "
                    f"({config.price_accuracy.pass_tolerance_percent}) is high for a strict environment; "
                    "recommended ≤ 1.0%"
                )
            # Early termination should be enabled
            if not config.escalation.early_termination_enabled:
                warnings.append(
                    f"[{environment}] escalation.early_termination_enabled is False; "
                    "recommended True for strict environments"
                )
            # Citation required
            if not config.policy_authenticity.citation_required:
                warnings.append(
                    f"[{environment}] policy_authenticity.citation_required is False; "
                    "recommended True for strict environments"
                )

        if environment in self._RELAXED_ENVIRONMENTS:
            # Development/testing: early termination disabled is fine, just note it
            if config.escalation.early_termination_enabled:
                warnings.append(
                    f"[{environment}] escalation.early_termination_enabled is True; "
                    "consider disabling for easier debugging in development/testing"
                )

        return warnings, errors

    def _generate_suggestions(
        self,
        config: "BinaryVerificationConfig",
    ) -> List[str]:
        """Generate improvement suggestions for the config."""
        suggestions: List[str] = []

        # Suggest tighter price tolerance for better accuracy
        if config.price_accuracy.pass_tolerance_percent > 2.0:
            suggestions.append(
                f"Consider tightening price_accuracy.pass_tolerance_percent "
                f"(currently {config.price_accuracy.pass_tolerance_percent}%) "
                "to improve price accuracy enforcement"
            )

        # Suggest enabling early termination if disabled
        if not config.escalation.early_termination_enabled:
            suggestions.append(
                "Consider enabling escalation.early_termination_enabled "
                "to stop processing on critical issues and save resources"
            )

        # Suggest reasonable retry limits
        if config.escalation.max_retries_with_minor > 10:
            suggestions.append(
                f"escalation.max_retries_with_minor ({config.escalation.max_retries_with_minor}) "
                "is very high; consider reducing to avoid excessive retries"
            )

        return suggestions


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_validator = ConfigValidator()


def validate_config_dict(data: Dict[str, Any]) -> ConfigValidationReport:
    """
    Validate a raw config dict before constructing BinaryVerificationConfig.

    Args:
        data: Raw dictionary that would be passed to BinaryVerificationConfig(**data).

    Returns:
        ConfigValidationReport with errors, warnings, and suggestions.
    """
    return _default_validator.validate_dict(data)


def validate_config(config: "BinaryVerificationConfig") -> ConfigValidationReport:
    """
    Validate an existing BinaryVerificationConfig instance.

    Args:
        config: An already-constructed BinaryVerificationConfig.

    Returns:
        ConfigValidationReport with warnings and suggestions (no Pydantic errors
        since the instance already passed construction).
    """
    return _default_validator.validate_instance(config)


def validate_yaml_file(yaml_path: str) -> ConfigValidationReport:
    """
    Validate a YAML config file.

    Args:
        yaml_path: Path to the YAML file to validate.

    Returns:
        ConfigValidationReport with errors if the file is missing, unparseable,
        or contains invalid config values.
    """
    return _default_validator.validate_yaml(yaml_path)
