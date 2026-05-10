"""
Tests for Task 7.1.2: Runtime configuration updates for issue severity levels.

Covers:
- BinaryVerificationConfig.update_severity_thresholds() — immutable partial updates
- RuntimeConfigManager thread-safe hot-reload
- Observer/callback pattern (register, unregister, notification)
- Thread-safety under concurrent updates
- Validation errors are raised before committing bad config
- reload_from_file() and reset_to_defaults()
"""

from __future__ import annotations

import threading
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from verification.config.binary_verification_config import (
    BinaryVerificationConfig,
    ConfigObserver,
    RuntimeConfigManager,
    get_binary_verification_config,
    get_runtime_config_manager,
)
from verification.config.thresholds_config import IssueSeverity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_manager() -> RuntimeConfigManager:
    """Return a new (non-singleton) RuntimeConfigManager for isolated tests."""
    return RuntimeConfigManager()


# ---------------------------------------------------------------------------
# BinaryVerificationConfig.update_severity_thresholds()
# ---------------------------------------------------------------------------

class TestUpdateSeverityThresholds:
    def test_returns_new_instance(self):
        cfg = BinaryVerificationConfig()
        updated = cfg.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.5}
        )
        assert updated is not cfg

    def test_original_is_not_mutated(self):
        cfg = BinaryVerificationConfig()
        original_tolerance = cfg.price_accuracy.pass_tolerance_percent
        cfg.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        assert cfg.price_accuracy.pass_tolerance_percent == original_tolerance

    def test_price_accuracy_partial_update(self):
        cfg = BinaryVerificationConfig()
        updated = cfg.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.25}
        )
        assert updated.price_accuracy.pass_tolerance_percent == 0.25
        # Other fields unchanged
        assert updated.price_accuracy.minor_threshold_percent == cfg.price_accuracy.minor_threshold_percent

    def test_policy_authenticity_partial_update(self):
        cfg = BinaryVerificationConfig()
        updated = cfg.update_severity_thresholds(
            policy_authenticity={"citation_required": False}
        )
        assert updated.policy_authenticity.citation_required is False

    def test_topic_relevance_partial_update(self):
        cfg = BinaryVerificationConfig()
        updated = cfg.update_severity_thresholds(
            topic_relevance={"pass_coverage_threshold": 0.8}
        )
        assert updated.topic_relevance.pass_coverage_threshold == 0.8

    def test_escalation_partial_update(self):
        cfg = BinaryVerificationConfig()
        updated = cfg.update_severity_thresholds(
            escalation={"max_retries_with_critical": 3}
        )
        assert updated.escalation.max_retries_with_critical == 3

    def test_timeouts_partial_update(self):
        cfg = BinaryVerificationConfig()
        updated = cfg.update_severity_thresholds(
            timeouts={"llm_call": 20.0}
        )
        assert updated.timeouts.llm_call == 20.0

    def test_verification_weights_replacement(self):
        cfg = BinaryVerificationConfig()
        new_weights = {"price_accuracy": 0.5, "policy_authenticity": 0.25, "topic_relevance": 0.25}
        updated = cfg.update_severity_thresholds(verification_weights=new_weights)
        assert updated.verification_weights["price_accuracy"] == 0.5

    def test_invalid_weights_raises_validation_error(self):
        cfg = BinaryVerificationConfig()
        with pytest.raises(Exception):
            cfg.update_severity_thresholds(
                verification_weights={"price_accuracy": 0.9, "policy_authenticity": 0.9, "topic_relevance": 0.9}
            )

    def test_multiple_sections_updated_at_once(self):
        cfg = BinaryVerificationConfig()
        updated = cfg.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.1},
            escalation={"early_termination_enabled": False},
        )
        assert updated.price_accuracy.pass_tolerance_percent == 0.1
        assert updated.escalation.early_termination_enabled is False

    def test_no_args_returns_equivalent_config(self):
        cfg = BinaryVerificationConfig()
        updated = cfg.update_severity_thresholds()
        assert updated.model_dump() == cfg.model_dump()


# ---------------------------------------------------------------------------
# RuntimeConfigManager — basic operations
# ---------------------------------------------------------------------------

class TestRuntimeConfigManagerBasic:
    def test_initial_config_is_default(self):
        manager = _fresh_manager()
        cfg = manager.get_config()
        assert isinstance(cfg, BinaryVerificationConfig)
        assert cfg.price_accuracy.pass_tolerance_percent == 1.0

    def test_update_severity_thresholds_changes_active_config(self):
        manager = _fresh_manager()
        manager.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.3}
        )
        assert manager.get_config().price_accuracy.pass_tolerance_percent == 0.3

    def test_update_returns_new_config(self):
        manager = _fresh_manager()
        returned = manager.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.3}
        )
        assert returned.price_accuracy.pass_tolerance_percent == 0.3
        assert returned is manager.get_config()

    def test_invalid_update_does_not_change_config(self):
        manager = _fresh_manager()
        original = manager.get_config()
        with pytest.raises(Exception):
            manager.update_severity_thresholds(
                verification_weights={"bad": 99.0}
            )
        # Config must remain unchanged after failed update
        assert manager.get_config() is original

    def test_reset_to_defaults(self):
        manager = _fresh_manager()
        manager.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.1}
        )
        manager.reset_to_defaults()
        assert manager.get_config().price_accuracy.pass_tolerance_percent == 1.0

    def test_reset_to_defaults_with_environment(self):
        manager = _fresh_manager()
        manager.reset_to_defaults(environment="production")
        assert manager.get_config().price_accuracy.pass_tolerance_percent == 0.5


# ---------------------------------------------------------------------------
# Observer / callback pattern
# ---------------------------------------------------------------------------

class TestObserverPattern:
    def test_callable_observer_is_called_on_update(self):
        manager = _fresh_manager()
        calls: List[tuple] = []

        def my_observer(old_cfg, new_cfg):
            calls.append((old_cfg, new_cfg))

        manager.register_observer(my_observer)
        manager.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})

        assert len(calls) == 1
        old, new = calls[0]
        assert old.price_accuracy.pass_tolerance_percent == 1.0
        assert new.price_accuracy.pass_tolerance_percent == 0.5

    def test_config_observer_protocol_is_called(self):
        manager = _fresh_manager()

        class MyObserver:
            def __init__(self):
                self.received: List[tuple] = []

            def on_config_updated(self, old_cfg, new_cfg):
                self.received.append((old_cfg, new_cfg))

        obs = MyObserver()
        assert isinstance(obs, ConfigObserver)
        manager.register_observer(obs)
        manager.update_severity_thresholds(escalation={"max_retries_with_critical": 2})

        assert len(obs.received) == 1

    def test_multiple_observers_all_notified(self):
        manager = _fresh_manager()
        counts = [0, 0]

        manager.register_observer(lambda o, n: counts.__setitem__(0, counts[0] + 1))
        manager.register_observer(lambda o, n: counts.__setitem__(1, counts[1] + 1))
        manager.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})

        assert counts == [1, 1]

    def test_unregister_observer_stops_notifications(self):
        manager = _fresh_manager()
        calls: List[int] = []

        def my_observer(old_cfg, new_cfg):
            calls.append(1)

        manager.register_observer(my_observer)
        manager.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        assert len(calls) == 1

        manager.unregister_observer(my_observer)
        manager.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.8})
        assert len(calls) == 1  # no new call

    def test_duplicate_registration_ignored(self):
        manager = _fresh_manager()
        calls: List[int] = []

        def my_observer(old_cfg, new_cfg):
            calls.append(1)

        manager.register_observer(my_observer)
        manager.register_observer(my_observer)  # duplicate
        manager.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})

        assert len(calls) == 1  # called only once

    def test_observer_exception_does_not_abort_update(self):
        manager = _fresh_manager()

        def bad_observer(old_cfg, new_cfg):
            raise RuntimeError("observer failure")

        manager.register_observer(bad_observer)
        # Should not raise
        manager.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        assert manager.get_config().price_accuracy.pass_tolerance_percent == 0.5

    def test_observer_receives_correct_old_and_new_configs(self):
        manager = _fresh_manager()
        received: List[tuple] = []

        manager.register_observer(lambda o, n: received.append((o, n)))
        manager.update_severity_thresholds(
            topic_relevance={"pass_coverage_threshold": 0.9}
        )

        old_cfg, new_cfg = received[0]
        assert old_cfg.topic_relevance.pass_coverage_threshold == 0.7
        assert new_cfg.topic_relevance.pass_coverage_threshold == 0.9

    def test_invalid_observer_type_raises_type_error(self):
        manager = _fresh_manager()
        with pytest.raises(TypeError):
            manager.register_observer("not_a_callable")  # type: ignore


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_updates_do_not_corrupt_config(self):
        manager = _fresh_manager()
        errors: List[Exception] = []

        def worker(tolerance: float):
            try:
                manager.update_severity_thresholds(
                    price_accuracy={"pass_tolerance_percent": tolerance}
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(0.1 * i,)) for i in range(1, 11)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Config must be a valid BinaryVerificationConfig after all updates
        cfg = manager.get_config()
        assert isinstance(cfg, BinaryVerificationConfig)
        assert cfg.price_accuracy.pass_tolerance_percent >= 0.0

    def test_concurrent_reads_are_consistent(self):
        manager = _fresh_manager()
        results: List[BinaryVerificationConfig] = []
        lock = threading.Lock()

        def reader():
            cfg = manager.get_config()
            with lock:
                results.append(cfg)

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        # All reads should return valid configs
        for cfg in results:
            assert isinstance(cfg, BinaryVerificationConfig)

    def test_concurrent_observer_registration_and_update(self):
        manager = _fresh_manager()
        call_count = [0]
        count_lock = threading.Lock()

        def observer(old_cfg, new_cfg):
            with count_lock:
                call_count[0] += 1

        def register_and_update():
            manager.register_observer(observer)
            manager.update_severity_thresholds(
                price_accuracy={"pass_tolerance_percent": 0.5}
            )

        threads = [threading.Thread(target=register_and_update) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No assertion on exact count — just ensure no crash
        assert isinstance(manager.get_config(), BinaryVerificationConfig)


# ---------------------------------------------------------------------------
# reload_from_file()
# ---------------------------------------------------------------------------

class TestReloadFromFile:
    def test_reload_from_missing_file_uses_defaults(self, tmp_path):
        manager = _fresh_manager()
        manager.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.1}
        )
        manager.reload_from_file(config_path=str(tmp_path / "nonexistent.yaml"))
        # Falls back to defaults
        assert manager.get_config().price_accuracy.pass_tolerance_percent == 1.0

    def test_reload_from_file_notifies_observers(self, tmp_path):
        manager = _fresh_manager()
        calls: List[int] = []
        manager.register_observer(lambda o, n: calls.append(1))

        manager.reload_from_file(config_path=str(tmp_path / "nonexistent.yaml"))
        assert len(calls) == 1

    def test_reload_from_file_with_environment(self, tmp_path):
        manager = _fresh_manager()
        manager.reload_from_file(
            config_path=str(tmp_path / "nonexistent.yaml"),
            environment="production",
        )
        assert manager.get_config().price_accuracy.pass_tolerance_percent == 0.5


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_runtime_config_manager_returns_same_instance(self):
        m1 = get_runtime_config_manager()
        m2 = get_runtime_config_manager()
        assert m1 is m2

    def test_singleton_is_runtime_config_manager(self):
        assert isinstance(get_runtime_config_manager(), RuntimeConfigManager)


# ---------------------------------------------------------------------------
# Integration: severity threshold hot-reload affects binary pass/fail
# ---------------------------------------------------------------------------

class TestHotReloadAffectsBinaryDecisions:
    def test_tightening_tolerance_causes_previously_passing_to_fail(self):
        manager = _fresh_manager()
        # Default tolerance = 1.0 → 0.8% deviation passes
        assert manager.get_config().is_binary_pass(
            price_deviation_percent=0.8,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.9,
        ) is True

        # Tighten tolerance to 0.5 → 0.8% now fails
        manager.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.5}
        )
        assert manager.get_config().is_binary_pass(
            price_deviation_percent=0.8,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.9,
        ) is False

    def test_relaxing_relevance_threshold_causes_previously_failing_to_pass(self):
        manager = _fresh_manager()
        # Default pass_coverage_threshold = 0.7 → 0.65 fails
        assert manager.get_config().is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.65,
        ) is False

        # Relax threshold to 0.6 → 0.65 now passes
        manager.update_severity_thresholds(
            topic_relevance={"pass_coverage_threshold": 0.6}
        )
        assert manager.get_config().is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.65,
        ) is True

    def test_escalation_threshold_update_affects_retry_limits(self):
        manager = _fresh_manager()
        # Default max_retries_with_critical = 1
        assert manager.get_config().get_max_retries(critical_count=1, major_count=0, minor_count=0) == 1

        manager.update_severity_thresholds(
            escalation={"max_retries_with_critical": 3}
        )
        assert manager.get_config().get_max_retries(critical_count=1, major_count=0, minor_count=0) == 3
