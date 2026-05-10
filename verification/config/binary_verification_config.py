"""
Binary Verification Configuration Schema - Task 7.1.1 / 7.1.2 / 7.1.3 / 7.1.4

Task 7.1.2 additions:
- RuntimeConfigManager: thread-safe singleton for hot-reload of severity thresholds
- ConfigObserver protocol: callback pattern so components react to config changes
- BinaryVerificationConfig.update_severity_thresholds(): runtime update without restart

Task 7.1.3 additions:
- apply_environment_override_to_config(): uses EnvironmentConfigOverride for validated overrides
- RuntimeConfigManager.apply_environment_override(): apply env override at runtime

Task 7.1.4 additions:
- ConfigVersion: Pydantic model capturing version number, timestamp, description, environment
- ConfigVersionManager: versioning + bounded history + rollback by version number or steps
- RuntimeConfigManager gains versioning via an embedded ConfigVersionManager
"""
from __future__ import annotations

import logging
import threading
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator
from verification.config.thresholds_config import (
    EscalationThresholds, IssueSeverity, PolicyAuthenticityThresholds,
    PriceAccuracyThresholds, TimeoutConfig, TopicRelevanceThresholds,
    VerificationThresholdsConfig, _deep_merge, _get_thresholds_env_overrides,
    enhanced_load_thresholds_config, get_default_thresholds_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IssueSeverity", "PriceAccuracyThresholds", "PolicyAuthenticityThresholds",
    "TopicRelevanceThresholds", "EscalationThresholds", "TimeoutConfig",
    "BinaryVerificationConfig", "get_binary_verification_config",
    "load_binary_verification_config",
    "ConfigObserver", "RuntimeConfigManager", "get_runtime_config_manager",
    "apply_environment_override_to_config",
    "ConfigVersion", "ConfigVersionManager",
]


@runtime_checkable
class ConfigObserver(Protocol):
    """Protocol for components that want to react to config changes."""

    def on_config_updated(
        self,
        old_config: "BinaryVerificationConfig",
        new_config: "BinaryVerificationConfig",
    ) -> None:
        ...


_ENVIRONMENT_OVERRIDES: Dict[str, dict] = {
    "development": {
        "price_accuracy": {"pass_tolerance_percent": 2.0, "critical_threshold_percent": 50.0},
        "escalation": {"max_critical_issues_before_escalation": 5, "early_termination_enabled": False},
    },
    "production": {
        "price_accuracy": {"pass_tolerance_percent": 0.5},
        "escalation": {"max_critical_issues_before_escalation": 1, "fabricated_policy_immediate_escalation": True},
    },
    "testing": {
        "escalation": {"early_termination_enabled": False, "max_retries_with_critical": 0},
    },
}


class BinaryVerificationConfig(BaseModel):
    price_accuracy: PriceAccuracyThresholds = Field(default_factory=PriceAccuracyThresholds)
    policy_authenticity: PolicyAuthenticityThresholds = Field(default_factory=PolicyAuthenticityThresholds)
    topic_relevance: TopicRelevanceThresholds = Field(default_factory=TopicRelevanceThresholds)
    escalation: EscalationThresholds = Field(default_factory=EscalationThresholds)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    verification_weights: Dict[str, float] = Field(
        default_factory=lambda: {"price_accuracy": 0.4, "policy_authenticity": 0.3, "topic_relevance": 0.3}
    )
    environment: Optional[str] = Field(default=None)

    @field_validator("verification_weights")
    @classmethod
    def weights_must_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"verification_weights must sum to 1.0, got {total:.3f}")
        return v

    def validate_configuration(self) -> List[str]:
        warnings: List[str] = []
        pa = self.price_accuracy
        if pa.pass_tolerance_percent >= pa.minor_threshold_percent:
            warnings.append(
                f"price_accuracy.pass_tolerance_percent ({pa.pass_tolerance_percent}) "
                f"should be < minor_threshold_percent ({pa.minor_threshold_percent})"
            )
        tr = self.topic_relevance
        if tr.pass_coverage_threshold < tr.major_coverage_threshold:
            warnings.append(
                f"topic_relevance.pass_coverage_threshold ({tr.pass_coverage_threshold}) "
                f"should be >= major_coverage_threshold ({tr.major_coverage_threshold})"
            )
        return warnings

    def is_binary_pass(
        self,
        price_deviation_percent: float,
        policy_fabricated: bool,
        policy_inaccurate: bool,
        policy_has_citation: bool,
        relevance_coverage: float,
        relevance_empathy: Optional[float] = None,
    ) -> bool:
        price_ok = self.price_accuracy.should_pass_price_check(price_deviation_percent)
        policy_ok = self.policy_authenticity.should_pass_policy_check(
            is_fabricated=policy_fabricated,
            is_inaccurate=policy_inaccurate,
            has_required_citation=policy_has_citation,
        )
        relevance_ok = self.topic_relevance.should_pass_relevance_check(
            coverage_ratio=relevance_coverage,
            empathy_score=relevance_empathy,
        )
        return price_ok and policy_ok and relevance_ok

    def get_max_retries(self, critical_count: int, major_count: int, minor_count: int) -> int:
        return self.escalation.get_max_retries_for_severity(
            critical_count=critical_count,
            major_count=major_count,
            minor_count=minor_count,
        )

    def should_terminate_early(self, critical_count: int) -> bool:
        return self.escalation.should_terminate_early(critical_count)

    def update_severity_thresholds(
        self,
        price_accuracy: Optional[Dict] = None,
        policy_authenticity: Optional[Dict] = None,
        topic_relevance: Optional[Dict] = None,
        escalation: Optional[Dict] = None,
        timeouts: Optional[Dict] = None,
        verification_weights: Optional[Dict[str, float]] = None,
    ) -> "BinaryVerificationConfig":
        """Return a new BinaryVerificationConfig with partial overrides applied."""
        data = self.model_dump()
        if price_accuracy:
            _deep_merge(data["price_accuracy"], price_accuracy)
        if policy_authenticity:
            _deep_merge(data["policy_authenticity"], policy_authenticity)
        if topic_relevance:
            _deep_merge(data["topic_relevance"], topic_relevance)
        if escalation:
            _deep_merge(data["escalation"], escalation)
        if timeouts:
            _deep_merge(data["timeouts"], timeouts)
        if verification_weights is not None:
            data["verification_weights"] = verification_weights
        return BinaryVerificationConfig(**data)

    def to_thresholds_config(self) -> VerificationThresholdsConfig:
        return VerificationThresholdsConfig(
            price_accuracy=self.price_accuracy,
            policy_authenticity=self.policy_authenticity,
            topic_relevance=self.topic_relevance,
            escalation=self.escalation,
            timeouts=self.timeouts,
            verification_weights=self.verification_weights,
        )

    @classmethod
    def from_thresholds_config(
        cls,
        tc: VerificationThresholdsConfig,
        environment: Optional[str] = None,
    ) -> "BinaryVerificationConfig":
        return cls(
            price_accuracy=tc.price_accuracy,
            policy_authenticity=tc.policy_authenticity,
            topic_relevance=tc.topic_relevance,
            escalation=tc.escalation,
            timeouts=tc.timeouts,
            verification_weights=tc.verification_weights,
            environment=environment,
        )

    model_config = {"validate_assignment": True}


# ---------------------------------------------------------------------------
# Task 7.1.4: Configuration Versioning
# ---------------------------------------------------------------------------

class ConfigVersion(BaseModel):
    """Immutable snapshot of a single configuration version."""

    version: int = Field(..., ge=1, description="Monotonically increasing version number")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: BinaryVerificationConfig
    description: Optional[str] = Field(default=None, description="Human-readable change description")
    environment: Optional[str] = Field(default=None)

    model_config = {"frozen": True}


class ConfigVersionManager:
    """
    Manages a bounded history of BinaryVerificationConfig versions with rollback support.

    - Each call to ``commit()`` creates a new ConfigVersion with a monotonically
      increasing version number and a UTC timestamp.
    - History is bounded to ``max_history`` entries (default 10); oldest entries
      are dropped when the limit is exceeded.
    - ``rollback(steps=1)`` reverts to the previous version (steps back in history).
    - ``rollback_to_version(version)`` reverts to a specific version number.
    - Thread-safe via an internal RLock.
    """

    DEFAULT_MAX_HISTORY = 10

    def __init__(self, max_history: int = DEFAULT_MAX_HISTORY) -> None:
        if max_history < 1:
            raise ValueError(f"max_history must be >= 1, got {max_history}")
        self._lock = threading.RLock()
        self._max_history = max_history
        self._history: List[ConfigVersion] = []
        self._current_version: int = 0

    @property
    def current_version(self) -> int:
        """The version number of the most recently committed config (0 if empty)."""
        with self._lock:
            return self._current_version

    def get_current(self) -> Optional[ConfigVersion]:
        """Return the most recent ConfigVersion, or None if no version has been committed."""
        with self._lock:
            return self._history[-1] if self._history else None

    def commit(
        self,
        config: BinaryVerificationConfig,
        description: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> ConfigVersion:
        """
        Commit a new config version.

        Args:
            config: The new BinaryVerificationConfig to record.
            description: Optional human-readable description of the change.
            environment: Optional environment tag (falls back to config.environment).

        Returns:
            The newly created ConfigVersion.
        """
        with self._lock:
            self._current_version += 1
            env = environment if environment is not None else config.environment
            version = ConfigVersion(
                version=self._current_version,
                config=config,
                description=description,
                environment=env,
            )
            self._history.append(version)
            # Trim oldest entries when history exceeds the bound
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            logger.debug(
                "ConfigVersionManager: committed version %d (%s)",
                self._current_version,
                description or "no description",
            )
            return version

    def get_history(self) -> List[ConfigVersion]:
        """Return a copy of the version history (oldest first)."""
        with self._lock:
            return list(self._history)

    def get_version(self, version: int) -> Optional[ConfigVersion]:
        """Return the ConfigVersion with the given version number, or None if not in history."""
        with self._lock:
            for v in self._history:
                if v.version == version:
                    return v
            return None

    def rollback(self, steps: int = 1) -> ConfigVersion:
        """
        Roll back by ``steps`` versions.

        Args:
            steps: Number of versions to go back (1 = previous version).

        Returns:
            The ConfigVersion that is now current after rollback.

        Raises:
            ValueError: If steps < 1, or if there are not enough versions in history.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        with self._lock:
            if len(self._history) < steps + 1:
                raise ValueError(
                    f"Cannot roll back {steps} step(s): only {len(self._history)} version(s) in history"
                )
            target = self._history[-(steps + 1)]
            # Trim history to the rollback point (inclusive)
            self._history = self._history[:len(self._history) - steps]
            self._current_version = target.version
            logger.info(
                "ConfigVersionManager: rolled back %d step(s) to version %d",
                steps,
                self._current_version,
            )
            return target

    def rollback_to_version(self, version: int) -> ConfigVersion:
        """
        Roll back to a specific version number.

        Args:
            version: The target version number to restore.

        Returns:
            The ConfigVersion that is now current after rollback.

        Raises:
            ValueError: If the version is not found in history.
        """
        with self._lock:
            idx = None
            for i, v in enumerate(self._history):
                if v.version == version:
                    idx = i
                    break
            if idx is None:
                available = [v.version for v in self._history]
                raise ValueError(
                    f"Version {version} not found in history. Available: {available}"
                )
            target = self._history[idx]
            # Trim history to the rollback point (inclusive)
            self._history = self._history[:idx + 1]
            self._current_version = target.version
            logger.info(
                "ConfigVersionManager: rolled back to version %d",
                self._current_version,
            )
            return target

    def clear_history(self) -> None:
        """Clear all version history and reset the version counter."""
        with self._lock:
            self._history.clear()
            self._current_version = 0


# ---------------------------------------------------------------------------
# RuntimeConfigManager — thread-safe hot-reload with observer pattern + versioning
# ---------------------------------------------------------------------------

class RuntimeConfigManager:
    """Thread-safe manager for hot-reload of BinaryVerificationConfig with observer pattern and versioning."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._config = BinaryVerificationConfig()
        self._observers: List[Callable] = []
        self._version_manager = ConfigVersionManager()
        # Commit the initial default config as version 1
        self._version_manager.commit(self._config, description="initial default config")

    def register_observer(self, observer) -> None:
        with self._lock:
            if isinstance(observer, ConfigObserver):
                callback = observer.on_config_updated
            elif callable(observer):
                callback = observer
            else:
                raise TypeError(
                    f"observer must be a ConfigObserver or callable, got {type(observer)}"
                )
            if callback not in self._observers:
                self._observers.append(callback)

    def unregister_observer(self, observer) -> None:
        with self._lock:
            callback = (
                observer.on_config_updated
                if isinstance(observer, ConfigObserver)
                else observer
            )
            self._observers = [cb for cb in self._observers if cb != callback]

    def get_config(self) -> BinaryVerificationConfig:
        with self._lock:
            return self._config

    def update_severity_thresholds(
        self,
        price_accuracy=None,
        policy_authenticity=None,
        topic_relevance=None,
        escalation=None,
        timeouts=None,
        verification_weights=None,
        description: Optional[str] = None,
    ) -> BinaryVerificationConfig:
        with self._lock:
            old_config = self._config
            new_config = old_config.update_severity_thresholds(
                price_accuracy=price_accuracy,
                policy_authenticity=policy_authenticity,
                topic_relevance=topic_relevance,
                escalation=escalation,
                timeouts=timeouts,
                verification_weights=verification_weights,
            )
            self._config = new_config
            self._version_manager.commit(
                new_config, description=description or "severity thresholds updated"
            )
            logger.info("RuntimeConfigManager: severity thresholds updated at runtime.")
        self._notify_observers(old_config, new_config)
        return new_config

    def reload_from_file(self, config_path=None, environment=None) -> BinaryVerificationConfig:
        new_config = load_binary_verification_config(config_path=config_path, environment=environment)
        with self._lock:
            old_config = self._config
            self._config = new_config
            self._version_manager.commit(new_config, description="reloaded from file")
            logger.info("RuntimeConfigManager: config reloaded from file.")
        self._notify_observers(old_config, new_config)
        return new_config

    def apply_environment_override(self, environment: str) -> BinaryVerificationConfig:
        """Apply a validated environment-specific override to the current config."""
        with self._lock:
            old_config = self._config
            new_config = apply_environment_override_to_config(old_config, environment)
            self._config = new_config
            self._version_manager.commit(
                new_config,
                description=f"environment override: {environment}",
                environment=environment,
            )
            logger.info("RuntimeConfigManager: environment override %r applied.", environment)
        self._notify_observers(old_config, new_config)
        return new_config

    def reset_to_defaults(self, environment=None) -> BinaryVerificationConfig:
        new_config = get_binary_verification_config(environment=environment)
        with self._lock:
            old_config = self._config
            self._config = new_config
            self._version_manager.commit(new_config, description="reset to defaults")
        self._notify_observers(old_config, new_config)
        return new_config

    # ------------------------------------------------------------------
    # Versioning API
    # ------------------------------------------------------------------

    def get_version_history(self) -> List[ConfigVersion]:
        """Return the bounded version history (oldest first)."""
        return self._version_manager.get_history()

    def get_current_version(self) -> Optional[ConfigVersion]:
        """Return the current ConfigVersion metadata."""
        return self._version_manager.get_current()

    def rollback(self, steps: int = 1) -> BinaryVerificationConfig:
        """
        Roll back the active config by ``steps`` versions.

        Args:
            steps: Number of versions to go back (1 = previous version).

        Returns:
            The BinaryVerificationConfig that is now active.

        Raises:
            ValueError: If there are not enough versions in history.
        """
        with self._lock:
            target_version = self._version_manager.rollback(steps=steps)
            old_config = self._config
            self._config = target_version.config
        self._notify_observers(old_config, target_version.config)
        return target_version.config

    def rollback_to_version(self, version: int) -> BinaryVerificationConfig:
        """
        Roll back the active config to a specific version number.

        Args:
            version: The target version number.

        Returns:
            The BinaryVerificationConfig that is now active.

        Raises:
            ValueError: If the version is not found in history.
        """
        with self._lock:
            target_version = self._version_manager.rollback_to_version(version)
            old_config = self._config
            self._config = target_version.config
        self._notify_observers(old_config, target_version.config)
        return target_version.config

    def _notify_observers(self, old_config, new_config) -> None:
        with self._lock:
            observers_snapshot = list(self._observers)
        for callback in observers_snapshot:
            try:
                callback(old_config, new_config)
            except Exception as exc:
                logger.warning("RuntimeConfigManager: observer %s raised %s", callback, exc)


_runtime_config_manager: Optional[RuntimeConfigManager] = None
_manager_lock = threading.Lock()


def get_runtime_config_manager() -> RuntimeConfigManager:
    global _runtime_config_manager
    if _runtime_config_manager is None:
        with _manager_lock:
            if _runtime_config_manager is None:
                _runtime_config_manager = RuntimeConfigManager()
    return _runtime_config_manager


def apply_environment_override_to_config(
    base_config: BinaryVerificationConfig,
    environment: str,
) -> BinaryVerificationConfig:
    """Apply a validated environment-specific override to a BinaryVerificationConfig."""
    from verification.config.environment_config_override import (
        apply_environment_override,
        get_environment_override,
    )

    validated_override = get_environment_override(environment)
    if validated_override is not None:
        return apply_environment_override(base_config, validated_override)

    if environment in _ENVIRONMENT_OVERRIDES:
        data = base_config.model_dump()
        _deep_merge(data, _ENVIRONMENT_OVERRIDES[environment])
        data["environment"] = environment
        return BinaryVerificationConfig(**data)

    data = base_config.model_dump()
    data["environment"] = environment
    return BinaryVerificationConfig(**data)


def get_binary_verification_config(environment: Optional[str] = None) -> BinaryVerificationConfig:
    if environment is None:
        environment = os.environ.get("VERIFICATION_ENV") or os.environ.get("ENVIRONMENT") or None
    base = BinaryVerificationConfig()
    if environment:
        return apply_environment_override_to_config(base, environment)
    return base


def load_binary_verification_config(
    config_path: Optional[str] = None,
    environment: Optional[str] = None,
) -> BinaryVerificationConfig:
    if config_path is None:
        config_path = str(Path(__file__).parent / "thresholds.yaml")
    if environment is None:
        environment = os.environ.get("VERIFICATION_ENV") or os.environ.get("ENVIRONMENT") or None
    tc = enhanced_load_thresholds_config(config_path)
    base = BinaryVerificationConfig.from_thresholds_config(tc, environment=None)
    if environment:
        return apply_environment_override_to_config(base, environment)
    return base
