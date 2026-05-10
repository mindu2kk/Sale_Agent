"""
Environment-Specific Config Overrides with Pydantic Validation - Task 7.1.3

Provides:
- EnvironmentConfigOverride: Pydantic model that validates override structure per environment
- SUPPORTED_ENVIRONMENTS: set of known environments (development, production, testing, staging)
- apply_environment_override(): merges an override into a BinaryVerificationConfig and
  validates the resulting merged config (not just the override dict)
- get_environment_override(): returns the validated EnvironmentConfigOverride for a given env
- detect_environment(): reads VERIFICATION_ENV / ENVIRONMENT env vars
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from verification.config.thresholds_config import (
    EscalationThresholds,
    IssueSeverity,
    PolicyAuthenticityThresholds,
    PriceAccuracyThresholds,
    TimeoutConfig,
    TopicRelevanceThresholds,
    _deep_merge,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EnvironmentConfigOverride",
    "SUPPORTED_ENVIRONMENTS",
    "apply_environment_override",
    "get_environment_override",
    "detect_environment",
    "EnvironmentOverrideError",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_ENVIRONMENTS = frozenset({"development", "production", "testing", "staging"})


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class EnvironmentOverrideError(ValueError):
    """Raised when an environment override fails validation."""


# ---------------------------------------------------------------------------
# Partial override sub-models (all fields optional)
# ---------------------------------------------------------------------------

class PriceAccuracyOverride(BaseModel):
    """Partial override for PriceAccuracyThresholds — all fields optional."""

    pass_tolerance_percent: Optional[float] = Field(None, ge=0.0)
    minor_threshold_percent: Optional[float] = Field(None, ge=0.0)
    major_threshold_percent: Optional[float] = Field(None, ge=0.0)
    critical_threshold_percent: Optional[float] = Field(None, ge=0.0)
    missing_price_severity: Optional[IssueSeverity] = None

    model_config = {"extra": "forbid"}


class PolicyAuthenticityOverride(BaseModel):
    """Partial override for PolicyAuthenticityThresholds — all fields optional."""

    fabricated_policy_severity: Optional[IssueSeverity] = None
    inaccurate_policy_severity: Optional[IssueSeverity] = None
    incomplete_policy_severity: Optional[IssueSeverity] = None
    missing_citation_severity: Optional[IssueSeverity] = None
    citation_required: Optional[bool] = None

    model_config = {"extra": "forbid"}


class TopicRelevanceOverride(BaseModel):
    """Partial override for TopicRelevanceThresholds — all fields optional."""

    pass_coverage_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    minor_coverage_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    major_coverage_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    critical_coverage_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    empathy_required: Optional[bool] = None
    min_empathy_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class EscalationOverride(BaseModel):
    """Partial override for EscalationThresholds — all fields optional."""

    max_critical_issues_before_escalation: Optional[int] = Field(None, ge=0)
    max_major_issues_before_escalation: Optional[int] = Field(None, ge=0)
    max_total_issues_before_escalation: Optional[int] = Field(None, ge=0)
    fabricated_policy_immediate_escalation: Optional[bool] = None
    critical_price_deviation_escalation: Optional[bool] = None
    early_termination_enabled: Optional[bool] = None
    stop_on_first_critical: Optional[bool] = None
    multiple_critical_threshold: Optional[int] = Field(None, ge=1)
    max_retries_with_critical: Optional[int] = Field(None, ge=0)
    max_retries_with_major: Optional[int] = Field(None, ge=0)
    max_retries_with_minor: Optional[int] = Field(None, ge=0)

    model_config = {"extra": "forbid"}


class TimeoutOverride(BaseModel):
    """Partial override for TimeoutConfig — all fields optional."""

    llm_call: Optional[float] = Field(None, gt=0.0)
    price_check: Optional[float] = Field(None, gt=0.0)
    policy_check: Optional[float] = Field(None, gt=0.0)
    relevance_check: Optional[float] = Field(None, gt=0.0)
    total_workflow: Optional[float] = Field(None, gt=0.0)
    escalate_on_critical_timeout: Optional[bool] = None

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Main EnvironmentConfigOverride model
# ---------------------------------------------------------------------------

class EnvironmentConfigOverride(BaseModel):
    """
    Validated environment-specific configuration override.

    Each environment (development, production, testing, staging) can define
    partial overrides for any sub-section of BinaryVerificationConfig.
    Unknown environments are accepted but logged as warnings.

    The model validates:
    1. Individual field types and ranges in each sub-section override.
    2. Cross-field consistency within the override itself (e.g. threshold ordering).
    3. After merging with a base config, the resulting config is also validated.
    """

    environment: str = Field(..., description="Target environment name")
    price_accuracy: Optional[PriceAccuracyOverride] = None
    policy_authenticity: Optional[PolicyAuthenticityOverride] = None
    topic_relevance: Optional[TopicRelevanceOverride] = None
    escalation: Optional[EscalationOverride] = None
    timeouts: Optional[TimeoutOverride] = None
    verification_weights: Optional[Dict[str, float]] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def warn_unknown_environment(self) -> "EnvironmentConfigOverride":
        if self.environment not in SUPPORTED_ENVIRONMENTS:
            logger.warning(
                "EnvironmentConfigOverride: unknown environment %r. "
                "Supported: %s. Override will still be applied.",
                self.environment,
                sorted(SUPPORTED_ENVIRONMENTS),
            )
        return self

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "EnvironmentConfigOverride":
        if self.verification_weights is not None:
            total = sum(self.verification_weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"verification_weights must sum to 1.0, got {total:.3f}"
                )
        return self

    @model_validator(mode="after")
    def validate_price_threshold_ordering(self) -> "EnvironmentConfigOverride":
        """Validate threshold ordering within the override (when all three are provided)."""
        pa = self.price_accuracy
        if pa is None:
            return self
        minor = pa.minor_threshold_percent
        major = pa.major_threshold_percent
        critical = pa.critical_threshold_percent
        if minor is not None and major is not None and major <= minor:
            raise ValueError(
                f"price_accuracy: major_threshold_percent ({major}) must be > "
                f"minor_threshold_percent ({minor})"
            )
        if major is not None and critical is not None and critical <= major:
            raise ValueError(
                f"price_accuracy: critical_threshold_percent ({critical}) must be > "
                f"major_threshold_percent ({major})"
            )
        return self

    @model_validator(mode="after")
    def validate_relevance_threshold_ordering(self) -> "EnvironmentConfigOverride":
        """Validate coverage threshold ordering within the override."""
        tr = self.topic_relevance
        if tr is None:
            return self
        minor = tr.minor_coverage_threshold
        major = tr.major_coverage_threshold
        critical = tr.critical_coverage_threshold
        if minor is not None and major is not None and major >= minor:
            raise ValueError(
                f"topic_relevance: major_coverage_threshold ({major}) must be < "
                f"minor_coverage_threshold ({minor})"
            )
        if major is not None and critical is not None and critical >= major:
            raise ValueError(
                f"topic_relevance: critical_coverage_threshold ({critical}) must be < "
                f"major_coverage_threshold ({major})"
            )
        return self

    def to_merge_dict(self) -> dict:
        """
        Return a plain dict containing only the non-None override values,
        suitable for deep-merging into a BinaryVerificationConfig.model_dump().
        """
        result: dict = {}
        if self.price_accuracy is not None:
            pa_dict = {
                k: v
                for k, v in self.price_accuracy.model_dump().items()
                if v is not None
            }
            if pa_dict:
                result["price_accuracy"] = pa_dict
        if self.policy_authenticity is not None:
            pol_dict = {
                k: v
                for k, v in self.policy_authenticity.model_dump().items()
                if v is not None
            }
            if pol_dict:
                result["policy_authenticity"] = pol_dict
        if self.topic_relevance is not None:
            tr_dict = {
                k: v
                for k, v in self.topic_relevance.model_dump().items()
                if v is not None
            }
            if tr_dict:
                result["topic_relevance"] = tr_dict
        if self.escalation is not None:
            esc_dict = {
                k: v
                for k, v in self.escalation.model_dump().items()
                if v is not None
            }
            if esc_dict:
                result["escalation"] = esc_dict
        if self.timeouts is not None:
            to_dict = {
                k: v
                for k, v in self.timeouts.model_dump().items()
                if v is not None
            }
            if to_dict:
                result["timeouts"] = to_dict
        if self.verification_weights is not None:
            result["verification_weights"] = self.verification_weights
        return result

    def get_validation_warnings(self) -> List[str]:
        """Return human-readable warnings about this override (not errors)."""
        warnings: List[str] = []
        if self.environment not in SUPPORTED_ENVIRONMENTS:
            warnings.append(
                f"Unknown environment: {self.environment!r}. "
                f"Supported environments: {sorted(SUPPORTED_ENVIRONMENTS)}"
            )
        pa = self.price_accuracy
        if pa is not None and pa.pass_tolerance_percent is not None:
            minor = pa.minor_threshold_percent
            if minor is not None and pa.pass_tolerance_percent >= minor:
                warnings.append(
                    f"price_accuracy.pass_tolerance_percent ({pa.pass_tolerance_percent}) "
                    f"should be < minor_threshold_percent ({minor})"
                )
        return warnings


# ---------------------------------------------------------------------------
# Pre-defined environment overrides (validated at import time)
# ---------------------------------------------------------------------------

_VALIDATED_ENVIRONMENT_OVERRIDES: Dict[str, EnvironmentConfigOverride] = {
    "development": EnvironmentConfigOverride(
        environment="development",
        price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=2.0, critical_threshold_percent=50.0),
        escalation=EscalationOverride(
            max_critical_issues_before_escalation=5,
            early_termination_enabled=False,
        ),
    ),
    "production": EnvironmentConfigOverride(
        environment="production",
        price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=0.5),
        escalation=EscalationOverride(
            max_critical_issues_before_escalation=1,
            fabricated_policy_immediate_escalation=True,
        ),
    ),
    "testing": EnvironmentConfigOverride(
        environment="testing",
        escalation=EscalationOverride(
            early_termination_enabled=False,
            max_retries_with_critical=0,
        ),
    ),
    "staging": EnvironmentConfigOverride(
        environment="staging",
        price_accuracy=PriceAccuracyOverride(pass_tolerance_percent=1.0),
        escalation=EscalationOverride(
            max_critical_issues_before_escalation=2,
            early_termination_enabled=True,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_environment_override(environment: str) -> Optional[EnvironmentConfigOverride]:
    """
    Return the validated EnvironmentConfigOverride for the given environment.

    Returns None for unknown environments (with a warning logged).
    """
    override = _VALIDATED_ENVIRONMENT_OVERRIDES.get(environment)
    if override is None and environment:
        logger.warning(
            "get_environment_override: no pre-defined override for environment %r. "
            "Returning None.",
            environment,
        )
    return override


def apply_environment_override(
    base_config: "BinaryVerificationConfig",
    override: EnvironmentConfigOverride,
) -> "BinaryVerificationConfig":
    """
    Apply a validated EnvironmentConfigOverride to a BinaryVerificationConfig.

    The merge is performed on the raw dict representation, then the result is
    re-validated by constructing a new BinaryVerificationConfig. This ensures
    the *merged* config (not just the override) passes Pydantic validation.

    Args:
        base_config: The base BinaryVerificationConfig to override.
        override: A validated EnvironmentConfigOverride instance.

    Returns:
        A new BinaryVerificationConfig with the override applied.

    Raises:
        EnvironmentOverrideError: If the merged config fails Pydantic validation.
    """
    # Lazy import to avoid circular dependency
    from verification.config.binary_verification_config import BinaryVerificationConfig

    merge_dict = override.to_merge_dict()
    if not merge_dict:
        # Nothing to override — return a copy with environment set
        data = base_config.model_dump()
        data["environment"] = override.environment
        return BinaryVerificationConfig(**data)

    data = base_config.model_dump()
    _deep_merge(data, merge_dict)
    data["environment"] = override.environment

    try:
        merged = BinaryVerificationConfig(**data)
    except Exception as exc:
        raise EnvironmentOverrideError(
            f"Merged config for environment {override.environment!r} failed validation: {exc}"
        ) from exc

    # Run consistency warnings (non-fatal)
    warnings = merged.validate_configuration()
    for w in warnings:
        logger.warning("apply_environment_override [%s]: %s", override.environment, w)

    return merged


def detect_environment() -> Optional[str]:
    """
    Detect the current environment from environment variables.

    Checks VERIFICATION_ENV first, then ENVIRONMENT.
    Returns None if neither is set.
    """
    env = os.environ.get("VERIFICATION_ENV") or os.environ.get("ENVIRONMENT") or None
    if env:
        logger.debug("detect_environment: detected environment %r", env)
    return env
