"""
Tests for AuditLogger - Task 7.2.4

Covers:
- AuditEventType enum values
- AuditOutcome enum values
- AuditLogEntry Pydantic model structure and validation
- AuditLogger.log() — creates and stores entries
- AuditLogger.get_entries() — returns all entries in order
- AuditLogger.filter_entries() — filters by event_type, actor, outcome,
  time range, correlation_id
- Ring buffer eviction (max_entries)
- File output (JSONL)
- Thread safety (basic)
- Singleton factory get_audit_logger() / reset_audit_logger()
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.verification.utils.audit_logger import (
    AuditEventType,
    AuditLogEntry,
    AuditLogger,
    AuditOutcome,
    get_audit_logger,
    reset_audit_logger,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_audit_logger()
    yield
    reset_audit_logger()


@pytest.fixture
def audit():
    return AuditLogger()


def _log_entry(
    audit: AuditLogger,
    event_type: AuditEventType = AuditEventType.ACCESS_GRANTED,
    actor: str = "admin",
    action: str = "execute_workflow",
    resource: str = "verification_workflow",
    outcome: AuditOutcome = AuditOutcome.SUCCESS,
    details: str = "",
    correlation_id: str | None = None,
) -> AuditLogEntry:
    return audit.log(
        event_type=event_type,
        actor=actor,
        action=action,
        resource=resource,
        outcome=outcome,
        details=details,
        correlation_id=correlation_id,
    )


# ---------------------------------------------------------------------------
# AuditEventType enum
# ---------------------------------------------------------------------------


class TestAuditEventType:
    def test_access_denied_exists(self):
        assert AuditEventType.ACCESS_DENIED

    def test_access_granted_exists(self):
        assert AuditEventType.ACCESS_GRANTED

    def test_config_modified_exists(self):
        assert AuditEventType.CONFIG_MODIFIED

    def test_key_rotated_exists(self):
        assert AuditEventType.KEY_ROTATED

    def test_workflow_executed_exists(self):
        assert AuditEventType.WORKFLOW_EXECUTED

    def test_escalation_triggered_exists(self):
        assert AuditEventType.ESCALATION_TRIGGERED

    def test_input_rejected_exists(self):
        assert AuditEventType.INPUT_REJECTED

    def test_seven_event_types(self):
        assert len(AuditEventType) == 7

    def test_string_values(self):
        assert AuditEventType.ACCESS_DENIED.value == "ACCESS_DENIED"
        assert AuditEventType.INPUT_REJECTED.value == "INPUT_REJECTED"


# ---------------------------------------------------------------------------
# AuditOutcome enum
# ---------------------------------------------------------------------------


class TestAuditOutcome:
    def test_success_exists(self):
        assert AuditOutcome.SUCCESS

    def test_failure_exists(self):
        assert AuditOutcome.FAILURE

    def test_string_values(self):
        assert AuditOutcome.SUCCESS.value == "success"
        assert AuditOutcome.FAILURE.value == "failure"


# ---------------------------------------------------------------------------
# AuditLogEntry model
# ---------------------------------------------------------------------------


class TestAuditLogEntry:
    def test_required_fields_present(self):
        entry = AuditLogEntry(
            event_type=AuditEventType.ACCESS_DENIED,
            timestamp="2024-01-15T10:30:00+00:00",
            actor="sales_rep",
            action="modify_config",
            resource="verification_config",
            outcome=AuditOutcome.FAILURE,
        )
        assert entry.event_type == AuditEventType.ACCESS_DENIED
        assert entry.actor == "sales_rep"
        assert entry.action == "modify_config"
        assert entry.resource == "verification_config"
        assert entry.outcome == AuditOutcome.FAILURE

    def test_details_defaults_to_empty_string(self):
        entry = AuditLogEntry(
            event_type=AuditEventType.ACCESS_GRANTED,
            timestamp="2024-01-15T10:30:00+00:00",
            actor="admin",
            action="view",
            resource="config",
            outcome=AuditOutcome.SUCCESS,
        )
        assert entry.details == ""

    def test_correlation_id_defaults_to_none(self):
        entry = AuditLogEntry(
            event_type=AuditEventType.ACCESS_GRANTED,
            timestamp="2024-01-15T10:30:00+00:00",
            actor="admin",
            action="view",
            resource="config",
            outcome=AuditOutcome.SUCCESS,
        )
        assert entry.correlation_id is None

    def test_correlation_id_can_be_set(self):
        entry = AuditLogEntry(
            event_type=AuditEventType.WORKFLOW_EXECUTED,
            timestamp="2024-01-15T10:30:00+00:00",
            actor="system",
            action="run",
            resource="workflow",
            outcome=AuditOutcome.SUCCESS,
            correlation_id="corr-abc123",
        )
        assert entry.correlation_id == "corr-abc123"

    def test_is_pydantic_model(self):
        from pydantic import BaseModel
        entry = AuditLogEntry(
            event_type=AuditEventType.KEY_ROTATED,
            timestamp="2024-01-15T10:30:00+00:00",
            actor="admin",
            action="rotate_key",
            resource="api_key",
            outcome=AuditOutcome.SUCCESS,
        )
        assert isinstance(entry, BaseModel)

    def test_serializes_to_json(self):
        entry = AuditLogEntry(
            event_type=AuditEventType.INPUT_REJECTED,
            timestamp="2024-01-15T10:30:00+00:00",
            actor="system",
            action="sanitize",
            resource="objection_text",
            outcome=AuditOutcome.FAILURE,
            details="Prompt injection detected",
        )
        data = json.loads(entry.model_dump_json())
        assert data["event_type"] == "INPUT_REJECTED"
        assert data["outcome"] == "failure"
        assert data["details"] == "Prompt injection detected"


# ---------------------------------------------------------------------------
# AuditLogger.log()
# ---------------------------------------------------------------------------


class TestAuditLoggerLog:
    def test_log_returns_audit_log_entry(self, audit):
        entry = _log_entry(audit)
        assert isinstance(entry, AuditLogEntry)

    def test_log_stores_event_type(self, audit):
        entry = _log_entry(audit, event_type=AuditEventType.ACCESS_DENIED)
        assert entry.event_type == AuditEventType.ACCESS_DENIED

    def test_log_stores_actor(self, audit):
        entry = _log_entry(audit, actor="sales_rep")
        assert entry.actor == "sales_rep"

    def test_log_stores_action(self, audit):
        entry = _log_entry(audit, action="modify_config")
        assert entry.action == "modify_config"

    def test_log_stores_resource(self, audit):
        entry = _log_entry(audit, resource="verification_config")
        assert entry.resource == "verification_config"

    def test_log_stores_outcome_success(self, audit):
        entry = _log_entry(audit, outcome=AuditOutcome.SUCCESS)
        assert entry.outcome == AuditOutcome.SUCCESS

    def test_log_stores_outcome_failure(self, audit):
        entry = _log_entry(audit, outcome=AuditOutcome.FAILURE)
        assert entry.outcome == AuditOutcome.FAILURE

    def test_log_stores_details(self, audit):
        entry = _log_entry(audit, details="some detail")
        assert entry.details == "some detail"

    def test_log_stores_correlation_id(self, audit):
        entry = _log_entry(audit, correlation_id="corr-xyz")
        assert entry.correlation_id == "corr-xyz"

    def test_log_sets_timestamp(self, audit):
        entry = _log_entry(audit)
        # Timestamp should be a non-empty ISO string
        assert isinstance(entry.timestamp, str)
        assert len(entry.timestamp) > 0

    def test_log_increments_buffer(self, audit):
        assert len(audit.get_entries()) == 0
        _log_entry(audit)
        assert len(audit.get_entries()) == 1
        _log_entry(audit)
        assert len(audit.get_entries()) == 2

    def test_log_all_event_types(self, audit):
        for event_type in AuditEventType:
            _log_entry(audit, event_type=event_type)
        entries = audit.get_entries()
        logged_types = {e.event_type for e in entries}
        assert logged_types == set(AuditEventType)


# ---------------------------------------------------------------------------
# AuditLogger.get_entries()
# ---------------------------------------------------------------------------


class TestGetEntries:
    def test_empty_initially(self, audit):
        assert audit.get_entries() == []

    def test_returns_list(self, audit):
        assert isinstance(audit.get_entries(), list)

    def test_entries_in_insertion_order(self, audit):
        _log_entry(audit, actor="first")
        _log_entry(audit, actor="second")
        _log_entry(audit, actor="third")
        entries = audit.get_entries()
        assert [e.actor for e in entries] == ["first", "second", "third"]

    def test_returns_copy_not_reference(self, audit):
        _log_entry(audit)
        entries1 = audit.get_entries()
        entries1.clear()
        entries2 = audit.get_entries()
        assert len(entries2) == 1


# ---------------------------------------------------------------------------
# Ring buffer eviction
# ---------------------------------------------------------------------------


class TestRingBuffer:
    def test_max_entries_respected(self):
        audit = AuditLogger(max_entries=3)
        for i in range(5):
            _log_entry(audit, actor=f"actor_{i}")
        entries = audit.get_entries()
        assert len(entries) == 3

    def test_oldest_entries_evicted(self):
        audit = AuditLogger(max_entries=3)
        for i in range(5):
            _log_entry(audit, actor=f"actor_{i}")
        entries = audit.get_entries()
        actors = [e.actor for e in entries]
        # Only the last 3 should remain
        assert actors == ["actor_2", "actor_3", "actor_4"]

    def test_single_entry_buffer(self):
        audit = AuditLogger(max_entries=1)
        _log_entry(audit, actor="first")
        _log_entry(audit, actor="second")
        entries = audit.get_entries()
        assert len(entries) == 1
        assert entries[0].actor == "second"


# ---------------------------------------------------------------------------
# AuditLogger.filter_entries()
# ---------------------------------------------------------------------------


class TestFilterEntries:
    def _populate(self, audit: AuditLogger) -> None:
        """Add a variety of entries for filtering tests."""
        _log_entry(
            audit,
            event_type=AuditEventType.ACCESS_DENIED,
            actor="sales_rep",
            action="modify_config",
            resource="config",
            outcome=AuditOutcome.FAILURE,
            correlation_id="corr-1",
        )
        _log_entry(
            audit,
            event_type=AuditEventType.ACCESS_GRANTED,
            actor="admin",
            action="execute_workflow",
            resource="workflow",
            outcome=AuditOutcome.SUCCESS,
            correlation_id="corr-2",
        )
        _log_entry(
            audit,
            event_type=AuditEventType.INPUT_REJECTED,
            actor="system",
            action="sanitize",
            resource="objection_text",
            outcome=AuditOutcome.FAILURE,
            correlation_id="corr-1",
        )
        _log_entry(
            audit,
            event_type=AuditEventType.CONFIG_MODIFIED,
            actor="admin",
            action="update_threshold",
            resource="config",
            outcome=AuditOutcome.SUCCESS,
            correlation_id="corr-3",
        )

    def test_no_filter_returns_all(self, audit):
        self._populate(audit)
        assert len(audit.filter_entries()) == 4

    def test_filter_by_event_type(self, audit):
        self._populate(audit)
        results = audit.filter_entries(event_type=AuditEventType.ACCESS_DENIED)
        assert len(results) == 1
        assert results[0].event_type == AuditEventType.ACCESS_DENIED

    def test_filter_by_actor(self, audit):
        self._populate(audit)
        results = audit.filter_entries(actor="admin")
        assert len(results) == 2
        assert all(e.actor == "admin" for e in results)

    def test_filter_by_outcome_failure(self, audit):
        self._populate(audit)
        results = audit.filter_entries(outcome=AuditOutcome.FAILURE)
        assert len(results) == 2
        assert all(e.outcome == AuditOutcome.FAILURE for e in results)

    def test_filter_by_outcome_success(self, audit):
        self._populate(audit)
        results = audit.filter_entries(outcome=AuditOutcome.SUCCESS)
        assert len(results) == 2
        assert all(e.outcome == AuditOutcome.SUCCESS for e in results)

    def test_filter_by_correlation_id(self, audit):
        self._populate(audit)
        results = audit.filter_entries(correlation_id="corr-1")
        assert len(results) == 2
        assert all(e.correlation_id == "corr-1" for e in results)

    def test_filter_combined_event_type_and_actor(self, audit):
        self._populate(audit)
        results = audit.filter_entries(
            event_type=AuditEventType.CONFIG_MODIFIED, actor="admin"
        )
        assert len(results) == 1
        assert results[0].event_type == AuditEventType.CONFIG_MODIFIED

    def test_filter_no_match_returns_empty(self, audit):
        self._populate(audit)
        results = audit.filter_entries(actor="nonexistent_actor")
        assert results == []

    def test_filter_by_time_range(self, audit):
        """Entries with timestamps within range are returned."""
        # Log two entries with a small sleep to ensure different timestamps
        entry1 = _log_entry(audit, actor="early")
        time.sleep(0.01)
        mid_time = entry1.timestamp  # use first entry's timestamp as start
        entry2 = _log_entry(audit, actor="late")

        results = audit.filter_entries(start_time=mid_time)
        actors = [e.actor for e in results]
        assert "early" in actors
        assert "late" in actors

    def test_filter_start_time_excludes_older(self, audit):
        """Entries before start_time are excluded."""
        entry1 = _log_entry(audit, actor="old")
        time.sleep(0.02)
        after_first = datetime.now(tz=timezone.utc).isoformat()
        _log_entry(audit, actor="new")

        results = audit.filter_entries(start_time=after_first)
        actors = [e.actor for e in results]
        assert "old" not in actors
        assert "new" in actors

    def test_filter_end_time_excludes_newer(self, audit):
        """Entries after end_time are excluded."""
        _log_entry(audit, actor="old")
        time.sleep(0.02)
        cutoff = datetime.now(tz=timezone.utc).isoformat()
        time.sleep(0.01)
        _log_entry(audit, actor="new")

        results = audit.filter_entries(end_time=cutoff)
        actors = [e.actor for e in results]
        assert "old" in actors
        assert "new" not in actors


# ---------------------------------------------------------------------------
# AuditLogger.clear()
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_empties_buffer(self, audit):
        _log_entry(audit)
        _log_entry(audit)
        audit.clear()
        assert audit.get_entries() == []

    def test_can_log_after_clear(self, audit):
        _log_entry(audit)
        audit.clear()
        _log_entry(audit, actor="after_clear")
        entries = audit.get_entries()
        assert len(entries) == 1
        assert entries[0].actor == "after_clear"


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


class TestFileOutput:
    def test_entries_written_to_file(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_file=str(log_file))
        _log_entry(audit, actor="file_test", event_type=AuditEventType.KEY_ROTATED)

        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["actor"] == "file_test"
        assert data["event_type"] == "KEY_ROTATED"

    def test_multiple_entries_appended(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_file=str(log_file))
        _log_entry(audit, actor="first")
        _log_entry(audit, actor="second")

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_no_file_created_without_log_file(self, tmp_path):
        audit = AuditLogger()
        _log_entry(audit)
        # No file should be created in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_parent_directories_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "audit.jsonl"
        audit = AuditLogger(log_file=str(nested))
        _log_entry(audit)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_logging(self, audit):
        """Multiple threads logging simultaneously should not lose entries."""
        n_threads = 10
        entries_per_thread = 20
        audit_large = AuditLogger(max_entries=n_threads * entries_per_thread)

        def log_many():
            for _ in range(entries_per_thread):
                _log_entry(audit_large)

        threads = [threading.Thread(target=log_many) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(audit_large.get_entries()) == n_threads * entries_per_thread


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_audit_logger_returns_same_instance(self):
        a1 = get_audit_logger()
        a2 = get_audit_logger()
        assert a1 is a2

    def test_reset_creates_new_instance(self):
        a1 = get_audit_logger()
        reset_audit_logger()
        a2 = get_audit_logger()
        assert a1 is not a2

    def test_singleton_is_audit_logger(self):
        assert isinstance(get_audit_logger(), AuditLogger)

    def test_singleton_stores_entries(self):
        audit = get_audit_logger()
        _log_entry(audit)
        assert len(get_audit_logger().get_entries()) == 1
