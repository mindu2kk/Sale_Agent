"""
Audit Logging for Security Events - Task 7.2.4

Provides structured audit logging for security-relevant events in the
verification workflow:
- AuditEventType enum — security event categories
- AuditLogEntry Pydantic model — structured log record
- AuditLogger — in-memory ring buffer + optional file output
- get_audit_logger() — module-level singleton

Structured fields per entry:
    event_type, timestamp, actor (role), action, resource,
    outcome (success/failure), details, correlation_id

Requirements:
- 7.2.4: Audit logging for security events with structured issue tracking
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Deque, List, Optional

from pydantic import ConfigDict, BaseModel, Field

logger = logging.getLogger("backend.verification.audit_logger")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AuditEventType(str, Enum):
    """Security event types for audit logging."""

    ACCESS_DENIED = "ACCESS_DENIED"
    ACCESS_GRANTED = "ACCESS_GRANTED"
    CONFIG_MODIFIED = "CONFIG_MODIFIED"
    KEY_ROTATED = "KEY_ROTATED"
    WORKFLOW_EXECUTED = "WORKFLOW_EXECUTED"
    ESCALATION_TRIGGERED = "ESCALATION_TRIGGERED"
    INPUT_REJECTED = "INPUT_REJECTED"


class AuditOutcome(str, Enum):
    """Outcome of an audited action."""

    SUCCESS = "success"
    FAILURE = "failure"


# ---------------------------------------------------------------------------
# AuditLogEntry model
# ---------------------------------------------------------------------------


class AuditLogEntry(BaseModel):
    """
    Structured audit log record for a single security event.

    Attributes:
        event_type: Category of the security event.
        timestamp: ISO-8601 UTC timestamp of the event.
        actor: Role or identity that performed the action.
        action: Human-readable description of the action taken.
        resource: The resource that was accessed or modified.
        outcome: Whether the action succeeded or failed.
        details: Additional context or structured data about the event.
        correlation_id: Optional ID linking related events across the system.
    """

    event_type: AuditEventType = Field(description="Security event category")
    timestamp: str = Field(description="ISO-8601 UTC timestamp")
    actor: str = Field(description="Role or identity performing the action")
    action: str = Field(description="Description of the action taken")
    resource: str = Field(description="Resource accessed or modified")
    outcome: AuditOutcome = Field(description="success or failure")
    details: str = Field(default="", description="Additional context about the event")
    correlation_id: Optional[str] = Field(
        default=None, description="Optional ID linking related events"
    )
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "event_type": "ACCESS_DENIED",
            "timestamp": "2024-01-15T10:30:00Z",
            "actor": "sales_rep",
            "action": "execute_workflow",
            "resource": "verification_workflow",
            "outcome": "failure",
            "details": "Role 'sales_rep' lacks permission 'modify_config'",
            "correlation_id": "corr-abc123",
        }
    })


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------


class AuditLogger:
    """
    Thread-safe audit logger with in-memory ring buffer and optional file output.

    Stores up to *max_entries* entries in a deque (ring buffer). When the buffer
    is full, the oldest entry is automatically discarded.

    Optionally writes each entry as a JSON line to a file for persistence.

    Usage::

        audit = AuditLogger(max_entries=1000, log_file="logs/audit.jsonl")

        audit.log(
            event_type=AuditEventType.ACCESS_DENIED,
            actor="sales_rep",
            action="modify_config",
            resource="verification_config",
            outcome=AuditOutcome.FAILURE,
            details="Insufficient permissions",
            correlation_id="corr-xyz",
        )

        entries = audit.get_entries()
        denied = audit.filter_entries(event_type=AuditEventType.ACCESS_DENIED)
    """

    def __init__(
        self,
        max_entries: int = 10_000,
        log_file: Optional[str] = None,
    ) -> None:
        """
        Initialise the AuditLogger.

        Args:
            max_entries: Maximum number of entries to keep in the ring buffer.
            log_file: Optional path to a JSONL file for persistent storage.
                      The file is created (and parent directories) if needed.
        """
        self._buffer: Deque[AuditLogEntry] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._log_file: Optional[Path] = Path(log_file) if log_file else None

        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        outcome: AuditOutcome,
        details: str = "",
        correlation_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Record a security event.

        Args:
            event_type: Category of the security event.
            actor: Role or identity performing the action.
            action: Description of the action taken.
            resource: Resource accessed or modified.
            outcome: Whether the action succeeded or failed.
            details: Additional context (default empty string).
            correlation_id: Optional ID linking related events.

        Returns:
            The created AuditLogEntry.
        """
        entry = AuditLogEntry(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details,
            correlation_id=correlation_id,
        )

        with self._lock:
            self._buffer.append(entry)
            self._write_to_file(entry)

        logger.debug(
            "Audit event: type=%s actor=%s action=%s resource=%s outcome=%s",
            event_type.value,
            actor,
            action,
            resource,
            outcome.value,
        )

        return entry

    def get_entries(self) -> List[AuditLogEntry]:
        """
        Return all entries currently in the ring buffer (oldest first).

        Returns:
            List of AuditLogEntry objects.
        """
        with self._lock:
            return list(self._buffer)

    def filter_entries(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        outcome: Optional[AuditOutcome] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """
        Filter audit log entries by one or more criteria.

        All provided criteria are combined with AND logic.

        Args:
            event_type: Filter by event type.
            actor: Filter by actor (exact match).
            outcome: Filter by outcome (success/failure).
            start_time: ISO-8601 string; include entries at or after this time.
            end_time: ISO-8601 string; include entries at or before this time.
            correlation_id: Filter by correlation ID (exact match).

        Returns:
            Filtered list of AuditLogEntry objects (oldest first).
        """
        with self._lock:
            entries = list(self._buffer)

        results: List[AuditLogEntry] = []
        for entry in entries:
            if event_type is not None and entry.event_type != event_type:
                continue
            if actor is not None and entry.actor != actor:
                continue
            if outcome is not None and entry.outcome != outcome:
                continue
            if correlation_id is not None and entry.correlation_id != correlation_id:
                continue
            if start_time is not None and entry.timestamp < start_time:
                continue
            if end_time is not None and entry.timestamp > end_time:
                continue
            results.append(entry)

        return results

    def clear(self) -> None:
        """Clear all entries from the in-memory buffer (does not affect file)."""
        with self._lock:
            self._buffer.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_to_file(self, entry: AuditLogEntry) -> None:
        """Append entry as a JSON line to the log file (if configured)."""
        if self._log_file is None:
            return
        try:
            with self._log_file.open("a", encoding="utf-8") as fh:
                fh.write(entry.model_dump_json() + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write audit entry to file: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_audit_logger: Optional[AuditLogger] = None
_singleton_lock = threading.Lock()


def get_audit_logger(
    max_entries: int = 10_000,
    log_file: Optional[str] = None,
) -> AuditLogger:
    """
    Return the module-level singleton AuditLogger.

    Creates the instance on first call.  Subsequent calls ignore the
    *max_entries* and *log_file* arguments and return the existing instance.

    Args:
        max_entries: Ring buffer size (used only on first call).
        log_file: Optional JSONL file path (used only on first call).

    Returns:
        The singleton AuditLogger instance.
    """
    global _default_audit_logger
    with _singleton_lock:
        if _default_audit_logger is None:
            _default_audit_logger = AuditLogger(
                max_entries=max_entries, log_file=log_file
            )
    return _default_audit_logger


def reset_audit_logger() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_audit_logger
    with _singleton_lock:
        _default_audit_logger = None


__all__ = [
    "AuditEventType",
    "AuditOutcome",
    "AuditLogEntry",
    "AuditLogger",
    "get_audit_logger",
    "reset_audit_logger",
]
