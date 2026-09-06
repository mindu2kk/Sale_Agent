"""
Tests for Task 7.1.4: Configuration versioning with rollback capabilities.

Covers:
- ConfigVersion: immutable Pydantic model with version number, timestamp, description, environment
- ConfigVersionManager: commit, get_history, get_version, rollback(steps), rollback_to_version
- Bounded history (max_history enforcement)
- Monotonically increasing version numbers
- RuntimeConfigManager versioning integration: get_version_history, get_current_version,
  rollback, rollback_to_version
- Thread-safety of ConfigVersionManager
- Observer notifications on rollback
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import List

import pytest

from backend.verification.config.binary_verification_config import (
    BinaryVerificationConfig,
    ConfigVersion,
    ConfigVersionManager,
    RuntimeConfigManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_manager(max_history: int = 10) -> ConfigVersionManager:
    return ConfigVersionManager(max_history=max_history)


def _fresh_runtime() -> RuntimeConfigManager:
    return RuntimeConfigManager()


def _cfg(tolerance: float = 1.0) -> BinaryVerificationConfig:
    return BinaryVerificationConfig().update_severity_thresholds(
        price_accuracy={"pass_tolerance_percent": tolerance}
    )


# ---------------------------------------------------------------------------
# ConfigVersion model
# ---------------------------------------------------------------------------

class TestConfigVersion:
    def test_version_number_stored(self):
        cfg = BinaryVerificationConfig()
        v = ConfigVersion(version=1, config=cfg)
        assert v.version == 1

    def test_timestamp_defaults_to_utc_now(self):
        before = datetime.now(timezone.utc)
        cfg = BinaryVerificationConfig()
        v = ConfigVersion(version=1, config=cfg)
        after = datetime.now(timezone.utc)
        assert before <= v.timestamp <= after

    def test_description_optional(self):
        cfg = BinaryVerificationConfig()
        v = ConfigVersion(version=1, config=cfg)
        assert v.description is None

    def test_description_stored(self):
        cfg = BinaryVerificationConfig()
        v = ConfigVersion(version=1, config=cfg, description="initial")
        assert v.description == "initial"

    def test_environment_optional(self):
        cfg = BinaryVerificationConfig()
        v = ConfigVersion(version=1, config=cfg)
        assert v.environment is None

    def test_environment_stored(self):
        cfg = BinaryVerificationConfig()
        v = ConfigVersion(version=1, config=cfg, environment="production")
        assert v.environment == "production"

    def test_version_is_immutable(self):
        cfg = BinaryVerificationConfig()
        v = ConfigVersion(version=1, config=cfg)
        with pytest.raises(Exception):
            v.version = 99  # type: ignore[misc]

    def test_version_must_be_positive(self):
        cfg = BinaryVerificationConfig()
        with pytest.raises(Exception):
            ConfigVersion(version=0, config=cfg)


# ---------------------------------------------------------------------------
# ConfigVersionManager — basic commit
# ---------------------------------------------------------------------------

class TestConfigVersionManagerCommit:
    def test_commit_returns_config_version(self):
        vm = _fresh_manager()
        cfg = BinaryVerificationConfig()
        v = vm.commit(cfg)
        assert isinstance(v, ConfigVersion)

    def test_first_commit_is_version_1(self):
        vm = _fresh_manager()
        v = vm.commit(BinaryVerificationConfig())
        assert v.version == 1

    def test_version_numbers_are_monotonically_increasing(self):
        vm = _fresh_manager()
        v1 = vm.commit(BinaryVerificationConfig())
        v2 = vm.commit(BinaryVerificationConfig())
        v3 = vm.commit(BinaryVerificationConfig())
        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3

    def test_current_version_property_tracks_latest(self):
        vm = _fresh_manager()
        assert vm.current_version == 0
        vm.commit(BinaryVerificationConfig())
        assert vm.current_version == 1
        vm.commit(BinaryVerificationConfig())
        assert vm.current_version == 2

    def test_description_stored_in_version(self):
        vm = _fresh_manager()
        v = vm.commit(BinaryVerificationConfig(), description="my change")
        assert v.description == "my change"

    def test_environment_from_argument_takes_precedence(self):
        vm = _fresh_manager()
        cfg = BinaryVerificationConfig(environment="development")
        v = vm.commit(cfg, environment="production")
        assert v.environment == "production"

    def test_environment_falls_back_to_config_environment(self):
        vm = _fresh_manager()
        cfg = BinaryVerificationConfig(environment="staging")
        v = vm.commit(cfg)
        assert v.environment == "staging"

    def test_config_stored_in_version(self):
        vm = _fresh_manager()
        cfg = _cfg(tolerance=0.5)
        v = vm.commit(cfg)
        assert v.config.price_accuracy.pass_tolerance_percent == 0.5

    def test_get_current_returns_latest(self):
        vm = _fresh_manager()
        assert vm.get_current() is None
        vm.commit(BinaryVerificationConfig(), description="v1")
        v2 = vm.commit(BinaryVerificationConfig(), description="v2")
        assert vm.get_current() is v2


# ---------------------------------------------------------------------------
# ConfigVersionManager — history
# ---------------------------------------------------------------------------

class TestConfigVersionManagerHistory:
    def test_history_empty_initially(self):
        vm = _fresh_manager()
        assert vm.get_history() == []

    def test_history_grows_with_commits(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig())
        vm.commit(BinaryVerificationConfig())
        assert len(vm.get_history()) == 2

    def test_history_is_oldest_first(self):
        vm = _fresh_manager()
        v1 = vm.commit(BinaryVerificationConfig(), description="first")
        v2 = vm.commit(BinaryVerificationConfig(), description="second")
        history = vm.get_history()
        assert history[0] is v1
        assert history[1] is v2

    def test_history_is_a_copy(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig())
        h1 = vm.get_history()
        vm.commit(BinaryVerificationConfig())
        h2 = vm.get_history()
        assert len(h1) == 1
        assert len(h2) == 2

    def test_bounded_history_drops_oldest(self):
        vm = _fresh_manager(max_history=3)
        for i in range(5):
            vm.commit(BinaryVerificationConfig(), description=f"v{i+1}")
        history = vm.get_history()
        assert len(history) == 3
        # Oldest 2 dropped; versions 3, 4, 5 remain
        assert history[0].version == 3
        assert history[-1].version == 5

    def test_max_history_1_keeps_only_latest(self):
        vm = _fresh_manager(max_history=1)
        vm.commit(BinaryVerificationConfig(), description="v1")
        vm.commit(BinaryVerificationConfig(), description="v2")
        history = vm.get_history()
        assert len(history) == 1
        assert history[0].description == "v2"

    def test_max_history_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            ConfigVersionManager(max_history=0)

    def test_get_version_returns_correct_entry(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig(), description="v1")
        v2 = vm.commit(BinaryVerificationConfig(), description="v2")
        vm.commit(BinaryVerificationConfig(), description="v3")
        found = vm.get_version(2)
        assert found is v2

    def test_get_version_returns_none_for_missing(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig())
        assert vm.get_version(99) is None

    def test_get_version_returns_none_for_evicted_version(self):
        vm = _fresh_manager(max_history=2)
        vm.commit(BinaryVerificationConfig(), description="v1")  # will be evicted
        vm.commit(BinaryVerificationConfig(), description="v2")
        vm.commit(BinaryVerificationConfig(), description="v3")
        # version 1 was evicted
        assert vm.get_version(1) is None

    def test_clear_history_resets_everything(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig())
        vm.commit(BinaryVerificationConfig())
        vm.clear_history()
        assert vm.get_history() == []
        assert vm.current_version == 0
        assert vm.get_current() is None


# ---------------------------------------------------------------------------
# ConfigVersionManager — rollback by steps
# ---------------------------------------------------------------------------

class TestConfigVersionManagerRollback:
    def test_rollback_1_step_returns_previous_version(self):
        vm = _fresh_manager()
        v1 = vm.commit(_cfg(0.5), description="v1")
        vm.commit(_cfg(1.0), description="v2")
        result = vm.rollback(steps=1)
        assert result is v1

    def test_rollback_updates_current_version(self):
        vm = _fresh_manager()
        vm.commit(_cfg(0.5), description="v1")
        vm.commit(_cfg(1.0), description="v2")
        vm.rollback(steps=1)
        assert vm.current_version == 1

    def test_rollback_trims_history(self):
        vm = _fresh_manager()
        vm.commit(_cfg(0.5))
        vm.commit(_cfg(1.0))
        vm.commit(_cfg(1.5))
        vm.rollback(steps=1)
        assert len(vm.get_history()) == 2

    def test_rollback_2_steps(self):
        vm = _fresh_manager()
        v1 = vm.commit(_cfg(0.5), description="v1")
        vm.commit(_cfg(1.0), description="v2")
        vm.commit(_cfg(1.5), description="v3")
        result = vm.rollback(steps=2)
        assert result is v1
        assert vm.current_version == 1

    def test_rollback_raises_if_not_enough_history(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig())
        with pytest.raises(ValueError, match="Cannot roll back"):
            vm.rollback(steps=1)

    def test_rollback_raises_for_zero_steps(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig())
        vm.commit(BinaryVerificationConfig())
        with pytest.raises(ValueError):
            vm.rollback(steps=0)

    def test_rollback_raises_for_negative_steps(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig())
        vm.commit(BinaryVerificationConfig())
        with pytest.raises(ValueError):
            vm.rollback(steps=-1)

    def test_rollback_default_is_1_step(self):
        vm = _fresh_manager()
        v1 = vm.commit(_cfg(0.5))
        vm.commit(_cfg(1.0))
        result = vm.rollback()
        assert result is v1


# ---------------------------------------------------------------------------
# ConfigVersionManager — rollback_to_version
# ---------------------------------------------------------------------------

class TestConfigVersionManagerRollbackToVersion:
    def test_rollback_to_version_restores_config(self):
        vm = _fresh_manager()
        v1 = vm.commit(_cfg(0.5), description="v1")
        vm.commit(_cfg(1.0), description="v2")
        vm.commit(_cfg(1.5), description="v3")
        result = vm.rollback_to_version(1)
        assert result is v1
        assert vm.current_version == 1

    def test_rollback_to_version_trims_history(self):
        vm = _fresh_manager()
        vm.commit(_cfg(0.5))
        vm.commit(_cfg(1.0))
        vm.commit(_cfg(1.5))
        vm.rollback_to_version(1)
        assert len(vm.get_history()) == 1

    def test_rollback_to_version_raises_for_missing_version(self):
        vm = _fresh_manager()
        vm.commit(BinaryVerificationConfig())
        with pytest.raises(ValueError, match="not found"):
            vm.rollback_to_version(99)

    def test_rollback_to_current_version_is_noop(self):
        vm = _fresh_manager()
        v1 = vm.commit(_cfg(0.5))
        result = vm.rollback_to_version(1)
        assert result is v1
        assert vm.current_version == 1

    def test_rollback_to_version_raises_for_evicted_version(self):
        vm = _fresh_manager(max_history=2)
        vm.commit(_cfg(0.5))  # version 1 — will be evicted
        vm.commit(_cfg(1.0))  # version 2
        vm.commit(_cfg(1.5))  # version 3 — evicts version 1
        with pytest.raises(ValueError):
            vm.rollback_to_version(1)


# ---------------------------------------------------------------------------
# RuntimeConfigManager — versioning integration
# ---------------------------------------------------------------------------

class TestRuntimeConfigManagerVersioning:
    def test_initial_version_is_1(self):
        mgr = _fresh_runtime()
        v = mgr.get_current_version()
        assert v is not None
        assert v.version == 1

    def test_initial_version_description(self):
        mgr = _fresh_runtime()
        v = mgr.get_current_version()
        assert v.description == "initial default config"

    def test_update_creates_new_version(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        v = mgr.get_current_version()
        assert v.version == 2

    def test_update_with_description(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 0.5},
            description="tighten price tolerance",
        )
        v = mgr.get_current_version()
        assert v.description == "tighten price tolerance"

    def test_history_grows_with_updates(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.8})
        history = mgr.get_version_history()
        assert len(history) == 3  # initial + 2 updates

    def test_rollback_1_step_restores_previous_config(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        mgr.rollback(steps=1)
        assert mgr.get_config().price_accuracy.pass_tolerance_percent == 1.0

    def test_rollback_updates_active_config(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.3})
        mgr.rollback(steps=1)
        assert mgr.get_config().price_accuracy.pass_tolerance_percent == 0.5

    def test_rollback_to_version_restores_config(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.3})
        mgr.rollback_to_version(1)
        assert mgr.get_config().price_accuracy.pass_tolerance_percent == 1.0

    def test_rollback_notifies_observers(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        calls: List[tuple] = []
        mgr.register_observer(lambda o, n: calls.append((o, n)))
        mgr.rollback(steps=1)
        assert len(calls) == 1

    def test_rollback_to_version_notifies_observers(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        calls: List[tuple] = []
        mgr.register_observer(lambda o, n: calls.append((o, n)))
        mgr.rollback_to_version(1)
        assert len(calls) == 1

    def test_rollback_raises_if_no_previous_version(self):
        mgr = _fresh_runtime()
        # Only version 1 exists (initial)
        with pytest.raises(ValueError):
            mgr.rollback(steps=1)

    def test_rollback_to_nonexistent_version_raises(self):
        mgr = _fresh_runtime()
        with pytest.raises(ValueError):
            mgr.rollback_to_version(99)

    def test_reset_to_defaults_creates_new_version(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        mgr.reset_to_defaults()
        history = mgr.get_version_history()
        assert len(history) == 3
        assert history[-1].description == "reset to defaults"

    def test_reload_from_file_creates_new_version(self, tmp_path):
        mgr = _fresh_runtime()
        mgr.reload_from_file(config_path=str(tmp_path / "nonexistent.yaml"))
        history = mgr.get_version_history()
        assert len(history) == 2
        assert history[-1].description == "reloaded from file"

    def test_apply_environment_override_creates_new_version(self):
        mgr = _fresh_runtime()
        mgr.apply_environment_override("production")
        v = mgr.get_current_version()
        assert v.version == 2
        assert "production" in v.description

    def test_version_history_is_oldest_first(self):
        mgr = _fresh_runtime()
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.5})
        mgr.update_severity_thresholds(price_accuracy={"pass_tolerance_percent": 0.3})
        history = mgr.get_version_history()
        versions = [v.version for v in history]
        assert versions == sorted(versions)

    def test_version_timestamps_are_utc(self):
        mgr = _fresh_runtime()
        v = mgr.get_current_version()
        assert v.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Thread-safety of ConfigVersionManager
# ---------------------------------------------------------------------------

class TestConfigVersionManagerThreadSafety:
    def test_concurrent_commits_produce_unique_versions(self):
        vm = _fresh_manager()
        results: List[ConfigVersion] = []
        lock = threading.Lock()

        def worker():
            v = vm.commit(BinaryVerificationConfig())
            with lock:
                results.append(v)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        version_numbers = [v.version for v in results]
        # All version numbers must be unique
        assert len(set(version_numbers)) == 20

    def test_concurrent_commits_and_reads_are_safe(self):
        vm = _fresh_manager()
        errors: List[Exception] = []

        def writer():
            try:
                for _ in range(5):
                    vm.commit(BinaryVerificationConfig())
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(5):
                    vm.get_history()
                    vm.get_current()
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=writer) for _ in range(5)]
            + [threading.Thread(target=reader) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_rollbacks_do_not_corrupt_state(self):
        vm = _fresh_manager()
        # Commit enough versions so rollbacks have something to work with
        for _ in range(10):
            vm.commit(BinaryVerificationConfig())

        errors: List[Exception] = []

        def rollback_worker():
            try:
                vm.rollback(steps=1)
            except ValueError:
                pass  # expected when history is too short
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=rollback_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # History must still be a valid list of ConfigVersion objects
        for v in vm.get_history():
            assert isinstance(v, ConfigVersion)
