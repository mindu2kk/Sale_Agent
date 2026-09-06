"""
Verification Thresholds Configuration with Pydantic Models

Pydantic models for binary verification thresholds:
- PriceAccuracyThresholds: critical/major/minor price deviation classification
- PolicyAuthenticityThresholds: fabricated/inaccurate/incomplete policy classification
- TopicRelevanceThresholds: coverage ratio-based severity classification
- EscalationThresholds: early termination and escalation rules
- VerificationThresholdsConfig: top-level config with environment overrides

Supports Task 1.3.1: Design verification thresholds config for critical/major/minor issues
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Severity Enum
# ---------------------------------------------------------------------------

class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


# ---------------------------------------------------------------------------
# Price Accuracy Thresholds
# ---------------------------------------------------------------------------

class PriceAccuracyThresholds(BaseModel):
    """Thresholds for price deviation severity classification."""

    minor_threshold_percent: float = Field(5.0, ge=0.0)
    major_threshold_percent: float = Field(15.0, ge=0.0)
    critical_threshold_percent: float = Field(30.0, ge=0.0)
    pass_tolerance_percent: float = Field(1.0, ge=0.0)
    missing_price_severity: IssueSeverity = Field(IssueSeverity.MAJOR)

    @model_validator(mode="after")
    def validate_threshold_order(self):
        if self.major_threshold_percent <= self.minor_threshold_percent:
            raise ValueError(
                f"major_threshold_percent ({self.major_threshold_percent}) must be > "
                f"minor_threshold_percent ({self.minor_threshold_percent})"
            )
        if self.critical_threshold_percent <= self.major_threshold_percent:
            raise ValueError(
                f"critical_threshold_percent ({self.critical_threshold_percent}) must be > "
                f"major_threshold_percent ({self.major_threshold_percent})"
            )
        return self

    def classify_price_deviation(self, deviation_percent: float) -> IssueSeverity:
        if deviation_percent >= self.critical_threshold_percent:
            return IssueSeverity.CRITICAL
        if deviation_percent >= self.minor_threshold_percent:
            return IssueSeverity.MAJOR
        return IssueSeverity.MINOR

    def should_pass_price_check(self, deviation_percent: float) -> bool:
        return deviation_percent <= self.pass_tolerance_percent


# ---------------------------------------------------------------------------
# Policy Authenticity Thresholds
# ---------------------------------------------------------------------------

class PolicyAuthenticityThresholds(BaseModel):
    """Thresholds for policy issue severity classification."""

    fabricated_policy_severity: IssueSeverity = Field(IssueSeverity.CRITICAL)
    inaccurate_policy_severity: IssueSeverity = Field(IssueSeverity.MAJOR)
    incomplete_policy_severity: IssueSeverity = Field(IssueSeverity.MINOR)
    missing_citation_severity: IssueSeverity = Field(IssueSeverity.MAJOR)
    citation_required: bool = Field(True)

    policy_type_severity: Dict[str, IssueSeverity] = Field(
        default_factory=lambda: {
            "warranty": IssueSeverity.CRITICAL,
            "return": IssueSeverity.MAJOR,
            "exchange": IssueSeverity.MAJOR,
            "service": IssueSeverity.MINOR,
            "support": IssueSeverity.MINOR,
        }
    )

    def classify_policy_issue(
        self,
        is_fabricated: bool,
        is_inaccurate: bool,
        is_incomplete: bool,
        policy_type: str = "service",
        has_citation: bool = True,
    ) -> IssueSeverity:
        if is_fabricated:
            return IssueSeverity.CRITICAL
        if self.citation_required and not has_citation:
            type_severity = self.policy_type_severity.get(policy_type, IssueSeverity.MAJOR)
            if type_severity == IssueSeverity.CRITICAL:
                return IssueSeverity.CRITICAL
            return self.missing_citation_severity
        if is_inaccurate:
            return self.inaccurate_policy_severity
        if is_incomplete:
            return self.incomplete_policy_severity
        return IssueSeverity.MINOR

    def should_pass_policy_check(
        self,
        is_fabricated: bool,
        is_inaccurate: bool,
        has_required_citation: bool,
    ) -> bool:
        if is_fabricated or is_inaccurate:
            return False
        if self.citation_required and not has_required_citation:
            return False
        return True


# ---------------------------------------------------------------------------
# Topic Relevance Thresholds
# ---------------------------------------------------------------------------

class TopicRelevanceThresholds(BaseModel):
    """Thresholds for topic relevance severity classification."""

    minor_coverage_threshold: float = Field(0.8, ge=0.0, le=1.0)
    major_coverage_threshold: float = Field(0.5, ge=0.0, le=1.0)
    critical_coverage_threshold: float = Field(0.3, ge=0.0, le=1.0)
    pass_coverage_threshold: float = Field(0.7, ge=0.0, le=1.0)
    empathy_required: bool = Field(True)
    min_empathy_score: float = Field(0.5, ge=0.0, le=1.0)
    max_off_topic_ratio: float = Field(0.3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_coverage_order(self):
        if self.major_coverage_threshold >= self.minor_coverage_threshold:
            raise ValueError(
                f"major_coverage_threshold ({self.major_coverage_threshold}) must be < "
                f"minor_coverage_threshold ({self.minor_coverage_threshold})"
            )
        if self.critical_coverage_threshold >= self.major_coverage_threshold:
            raise ValueError(
                f"critical_coverage_threshold ({self.critical_coverage_threshold}) must be < "
                f"major_coverage_threshold ({self.major_coverage_threshold})"
            )
        return self

    def classify_relevance_issue(
        self,
        coverage_ratio: float,
        empathy_score: Optional[float] = None,
        off_topic_ratio: float = 0.0,
    ) -> IssueSeverity:
        if off_topic_ratio > self.max_off_topic_ratio:
            return IssueSeverity.CRITICAL
        if coverage_ratio < self.critical_coverage_threshold:
            return IssueSeverity.CRITICAL
        if coverage_ratio < self.major_coverage_threshold:
            return IssueSeverity.CRITICAL
        if coverage_ratio < self.minor_coverage_threshold:
            return IssueSeverity.MAJOR
        return IssueSeverity.MINOR

    def should_pass_relevance_check(
        self,
        coverage_ratio: float,
        empathy_score: Optional[float] = None,
    ) -> bool:
        if coverage_ratio < self.pass_coverage_threshold:
            return False
        if self.empathy_required and empathy_score is not None:
            if empathy_score < self.min_empathy_score:
                return False
        return True


# ---------------------------------------------------------------------------
# Escalation Thresholds
# ---------------------------------------------------------------------------

class EscalationThresholds(BaseModel):
    """Rules for early termination and human escalation."""

    max_critical_issues_before_escalation: int = Field(2, ge=0)
    max_major_issues_before_escalation: int = Field(5, ge=0)
    max_total_issues_before_escalation: int = Field(10, ge=0)

    fabricated_policy_immediate_escalation: bool = Field(True)
    critical_price_deviation_escalation: bool = Field(True)
    completely_irrelevant_escalation: bool = Field(True)

    early_termination_enabled: bool = Field(True)
    stop_on_first_critical: bool = Field(False)
    multiple_critical_threshold: int = Field(3, ge=1)

    max_retries_with_critical: int = Field(1, ge=0)
    max_retries_with_major: int = Field(3, ge=0)
    max_retries_with_minor: int = Field(5, ge=0)

    def should_escalate_immediately(
        self,
        critical_count: int,
        major_count: int,
        total_count: int,
        has_fabricated_policy: bool = False,
        has_critical_price_deviation: bool = False,
        is_completely_irrelevant: bool = False,
    ) -> bool:
        if self.fabricated_policy_immediate_escalation and has_fabricated_policy:
            return True
        if self.critical_price_deviation_escalation and has_critical_price_deviation:
            return True
        if self.completely_irrelevant_escalation and is_completely_irrelevant:
            return True
        if critical_count > self.max_critical_issues_before_escalation:
            return True
        if major_count > self.max_major_issues_before_escalation:
            return True
        if total_count > self.max_total_issues_before_escalation:
            return True
        return False

    def should_terminate_early(self, critical_count: int) -> bool:
        if not self.early_termination_enabled:
            return False
        if self.stop_on_first_critical and critical_count >= 1:
            return True
        if critical_count >= self.multiple_critical_threshold:
            return True
        return False

    def get_max_retries_for_severity(
        self,
        critical_count: int,
        major_count: int,
        minor_count: int,
    ) -> int:
        if critical_count > 0:
            return self.max_retries_with_critical
        if major_count > 0:
            return self.max_retries_with_major
        return self.max_retries_with_minor


# ---------------------------------------------------------------------------
# Timeout Configuration
# ---------------------------------------------------------------------------

class TimeoutConfig(BaseModel):
    """
    Configurable timeout thresholds for async verification operations.

    Supports Task 6.3.2: Async timeout handling with configurable thresholds.
    """

    llm_call: float = Field(10.0, gt=0.0, description="Timeout in seconds for a single LLM API call")
    price_check: float = Field(5.0, gt=0.0, description="Timeout in seconds for price accuracy check")
    policy_check: float = Field(5.0, gt=0.0, description="Timeout in seconds for policy authenticity check")
    relevance_check: float = Field(5.0, gt=0.0, description="Timeout in seconds for topic relevance check")
    total_workflow: float = Field(30.0, gt=0.0, description="Timeout in seconds for the entire verification workflow")
    escalate_on_critical_timeout: bool = Field(
        True,
        description="Whether to trigger escalation when a critical-path operation times out",
    )

    def get_timeout(self, operation: str) -> float:
        """
        Return the configured timeout for the given operation name.

        Falls back to ``llm_call`` timeout for unknown operations.

        Args:
            operation: One of "llm_call", "price_check", "policy_check",
                       "relevance_check", "total_workflow", or any custom key.

        Returns:
            Timeout in seconds.
        """
        return getattr(self, operation, self.llm_call)


# ---------------------------------------------------------------------------
# Top-level Configuration
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "price_accuracy": 0.4,
    "policy_authenticity": 0.3,
    "topic_relevance": 0.3,
}


class VerificationThresholdsConfig(BaseModel):
    """Top-level verification thresholds configuration."""

    price_accuracy: PriceAccuracyThresholds = Field(default_factory=PriceAccuracyThresholds)
    policy_authenticity: PolicyAuthenticityThresholds = Field(default_factory=PolicyAuthenticityThresholds)
    topic_relevance: TopicRelevanceThresholds = Field(default_factory=TopicRelevanceThresholds)
    escalation: EscalationThresholds = Field(default_factory=EscalationThresholds)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    verification_weights: Dict[str, float] = Field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))

    @field_validator("verification_weights")
    @classmethod
    def weights_must_sum_to_one(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"verification_weights must sum to 1.0, got {total:.3f}")
        return v

    def validate_configuration(self) -> List[str]:
        """Return a list of configuration warnings (empty list = valid)."""
        warnings: List[str] = []
        pa = self.price_accuracy
        if pa.pass_tolerance_percent >= pa.minor_threshold_percent:
            warnings.append(
                f"pass_tolerance_percent ({pa.pass_tolerance_percent}) should be "
                f"< minor_threshold_percent ({pa.minor_threshold_percent})"
            )
        tr = self.topic_relevance
        if tr.pass_coverage_threshold < tr.major_coverage_threshold:
            warnings.append(
                f"pass_coverage_threshold ({tr.pass_coverage_threshold}) should be "
                f">= major_coverage_threshold ({tr.major_coverage_threshold})"
            )
        return warnings

    def get_environment_config(self, environment: str) -> "VerificationThresholdsConfig":
        """Return a copy with environment-specific overrides applied."""
        overrides = _ENVIRONMENT_OVERRIDES.get(environment)
        if overrides is None:
            return self
        data = self.model_dump()
        _deep_merge(data, overrides)
        return VerificationThresholdsConfig(**data)


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------

_ENVIRONMENT_OVERRIDES: Dict[str, dict] = {
    "development": {
        "price_accuracy": {
            "pass_tolerance_percent": 2.0,
            "critical_threshold_percent": 50.0,
        },
        "escalation": {
            "max_critical_issues_before_escalation": 5,
            "early_termination_enabled": False,
        },
    },
    "production": {
        "price_accuracy": {
            "pass_tolerance_percent": 0.5,
        },
        "escalation": {
            "max_critical_issues_before_escalation": 1,
            "fabricated_policy_immediate_escalation": True,
        },
    },
    "testing": {
        "escalation": {
            "early_termination_enabled": False,
            "max_retries_with_critical": 0,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _convert_yaml_to_pydantic_structure(yaml_data: dict) -> dict:
    """Flatten nested YAML structure into flat Pydantic field names."""
    result: dict = {}

    pa_yaml = yaml_data.get("price_accuracy", {})
    pa_out: dict = {}
    for k, v in pa_yaml.get("thresholds", {}).items():
        pa_out[k] = v
    pc = pa_yaml.get("pass_criteria", {})
    if "tolerance_percent" in pc:
        pa_out["pass_tolerance_percent"] = pc["tolerance_percent"]
    if pa_out:
        result["price_accuracy"] = pa_out

    pol_yaml = yaml_data.get("policy_authenticity", {})
    pol_out: dict = {}
    sev_map = {
        "fabricated_policy": "fabricated_policy_severity",
        "inaccurate_policy": "inaccurate_policy_severity",
        "incomplete_policy": "incomplete_policy_severity",
        "missing_citation": "missing_citation_severity",
    }
    for yaml_key, pydantic_key in sev_map.items():
        if yaml_key in pol_yaml.get("severity_rules", {}):
            pol_out[pydantic_key] = pol_yaml["severity_rules"][yaml_key]
    cit = pol_yaml.get("citation_requirements", {})
    if "citation_required" in cit:
        pol_out["citation_required"] = cit["citation_required"]
    if pol_out:
        result["policy_authenticity"] = pol_out

    tr_yaml = yaml_data.get("topic_relevance", {})
    tr_out: dict = {}
    for k, v in tr_yaml.get("coverage_thresholds", {}).items():
        tr_out[k] = v
    pc2 = tr_yaml.get("pass_criteria", {})
    if "min_coverage_ratio" in pc2:
        tr_out["pass_coverage_threshold"] = pc2["min_coverage_ratio"]
    emp = tr_yaml.get("empathy_requirements", {})
    if "empathy_required" in emp:
        tr_out["empathy_required"] = emp["empathy_required"]
    if "min_empathy_score" in emp:
        tr_out["min_empathy_score"] = emp["min_empathy_score"]
    if tr_out:
        result["topic_relevance"] = tr_out

    esc_yaml = yaml_data.get("escalation", {})
    esc_out: dict = {}
    cnt = esc_yaml.get("count_thresholds", {})
    if "max_critical_before_escalation" in cnt:
        esc_out["max_critical_issues_before_escalation"] = cnt["max_critical_before_escalation"]
    if "max_major_before_escalation" in cnt:
        esc_out["max_major_issues_before_escalation"] = cnt["max_major_before_escalation"]
    if "max_total_before_escalation" in cnt:
        esc_out["max_total_issues_before_escalation"] = cnt["max_total_before_escalation"]
    imm = esc_yaml.get("immediate_triggers", {})
    if "fabricated_policy_detected" in imm:
        esc_out["fabricated_policy_immediate_escalation"] = imm["fabricated_policy_detected"]
    if "critical_price_deviation" in imm:
        esc_out["critical_price_deviation_escalation"] = imm["critical_price_deviation"]
    et = esc_yaml.get("early_termination", {})
    if "enabled" in et:
        esc_out["early_termination_enabled"] = et["enabled"]
    if "stop_on_first_critical" in et:
        esc_out["stop_on_first_critical"] = et["stop_on_first_critical"]
    if "multiple_critical_threshold" in et:
        esc_out["multiple_critical_threshold"] = et["multiple_critical_threshold"]
    rl = esc_yaml.get("retry_limits", {})
    if "max_retries_with_critical" in rl:
        esc_out["max_retries_with_critical"] = rl["max_retries_with_critical"]
    if "max_retries_with_major" in rl:
        esc_out["max_retries_with_major"] = rl["max_retries_with_major"]
    if "max_retries_with_minor" in rl:
        esc_out["max_retries_with_minor"] = rl["max_retries_with_minor"]
    if esc_out:
        result["escalation"] = esc_out

    # Timeout thresholds (flat mapping — keys match TimeoutConfig field names directly)
    to_yaml = yaml_data.get("timeouts", {})
    if to_yaml:
        result["timeouts"] = {k: v for k, v in to_yaml.items()}

    return result


def _get_thresholds_env_overrides() -> dict:
    """Read environment variables and return a partial config dict."""
    overrides: dict = {}
    price_tolerance = os.environ.get("VERIFICATION_THRESHOLDS_PRICE_TOLERANCE")
    if price_tolerance is not None:
        overrides.setdefault("price_accuracy", {})["pass_tolerance_percent"] = float(price_tolerance)
    max_critical = os.environ.get("VERIFICATION_THRESHOLDS_ESCALATION_MAX_CRITICAL")
    if max_critical is not None:
        overrides.setdefault("escalation", {})["max_critical_issues_before_escalation"] = int(max_critical)
    early_term = os.environ.get("VERIFICATION_THRESHOLDS_EARLY_TERMINATION")
    if early_term is not None:
        overrides.setdefault("escalation", {})["early_termination_enabled"] = early_term.lower() == "true"
    return overrides


def get_default_thresholds_config() -> VerificationThresholdsConfig:
    """Return a VerificationThresholdsConfig with default values."""
    return VerificationThresholdsConfig()


def enhanced_load_thresholds_config(config_path: str) -> VerificationThresholdsConfig:
    """Load VerificationThresholdsConfig from a YAML file, then apply env-var overrides."""
    data: dict = {}
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        data = _convert_yaml_to_pydantic_structure(raw)
    env_overrides = _get_thresholds_env_overrides()
    _deep_merge(data, env_overrides)
    return VerificationThresholdsConfig(**data)


def save_thresholds_config(
    config: VerificationThresholdsConfig,
    config_path: str,
) -> None:
    """Persist a validated thresholds configuration as human-readable YAML."""

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config.model_dump(mode="json"),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
