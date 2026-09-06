"""
Disaster Recovery Manager - Task 6.3.5

Provides disaster recovery procedures that leverage state persistence to allow
workflows to recover from failures. Builds on top of WorkflowPersistenceManager
to add:
- Snapshot saving at key checkpoints with metadata
- Loading and resuming workflows from the last valid saved state
- Detection of corrupted/invalid states with safe fallback
- Listing recoverable workflows
- Cleanup of old snapshots with configurable retention

Requirements:
- 7.4: State persistence to support resume after failures
- 8.3: Error Scenario 3 - StateGraph execution error → rollback to last valid state
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..models.state import WorkflowStateValidator, create_initial_workflow_state
from ..workflow.persistence import WorkflowPersistenceManager

logger = logging.getLogger("backend.verification.disaster_recovery")

# Default directory for disaster recovery snapshots (separate from normal checkpoints)
DEFAULT_SNAPSHOT_DIR = "logs/workflow/snapshots"

# Default retention period in days
DEFAULT_RETENTION_DAYS = 7


class SnapshotMetadata:
    """Lightweight metadata for a saved snapshot."""

    __slots__ = ("workflow_id", "checkpoint_name", "saved_at", "status", "path")

    def __init__(
        self,
        workflow_id: str,
        checkpoint_name: str,
        saved_at: str,
        status: str,
        path: str,
    ) -> None:
        self.workflow_id = workflow_id
        self.checkpoint_name = checkpoint_name
        self.saved_at = saved_at
        self.status = status
        self.path = path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "checkpoint_name": self.checkpoint_name,
            "saved_at": self.saved_at,
            "status": self.status,
            "path": self.path,
        }


class DisasterRecoveryManager:
    """
    Manages disaster recovery procedures for the verification workflow.

    Saves workflow state snapshots to disk at key checkpoints so that
    interrupted or corrupted workflows can be resumed from the last valid state.

    File layout::

        {snapshot_dir}/{workflow_id}__{checkpoint_name}.json

    Each snapshot file contains:
        - saved_at: ISO timestamp
        - workflow_id: unique workflow identifier
        - checkpoint_name: logical name of the checkpoint (e.g. "post_research")
        - status: workflow_status at save time
        - state: full serialized WorkflowState (via WorkflowStateValidator)

    **Validates: Requirements 7.4** - state persistence for resume after failures
    **Validates: Requirements 8.3** - rollback to last valid state on execution error
    """

    def __init__(
        self,
        snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        """
        Initialize the disaster recovery manager.

        Args:
            snapshot_dir: Directory to store snapshot files.
                          Created automatically if it does not exist.
            retention_days: Number of days to retain snapshots before cleanup.
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.retention_days = retention_days
        self._ensure_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        state: Dict[str, Any],
        checkpoint_name: str = "checkpoint",
    ) -> bool:
        """
        Save a workflow state snapshot to disk at a named checkpoint.

        Uses WorkflowStateValidator (Pydantic) for serialization so that all
        nested models are correctly serialized.

        Args:
            state: WorkflowState dict to persist.
            checkpoint_name: Logical name for this checkpoint
                             (e.g. "post_research", "pre_verification").

        Returns:
            True on success, False on validation or I/O error (warning logged).
        """
        workflow_id = state.get("workflow_id")
        if not workflow_id:
            logger.warning("Cannot save snapshot: missing workflow_id")
            return False

        try:
            # Validate & serialize via Pydantic
            validator = WorkflowStateValidator(**state)
            state_json = validator.model_dump_json()

            snapshot = {
                "saved_at": datetime.now().isoformat(),
                "workflow_id": workflow_id,
                "checkpoint_name": checkpoint_name,
                "status": state.get("workflow_status", "unknown"),
                "state": json.loads(state_json),
            }

            snapshot_path = self._snapshot_path(workflow_id, checkpoint_name)
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)

            logger.info(
                "Snapshot saved: workflow_id=%s checkpoint=%s path=%s",
                workflow_id,
                checkpoint_name,
                snapshot_path,
            )
            return True

        except ValidationError as exc:
            logger.warning(
                "Snapshot validation failed for %s at %s: %s",
                workflow_id,
                checkpoint_name,
                exc,
            )
            return False
        except Exception as exc:
            logger.warning(
                "Failed to save snapshot for %s at %s: %s",
                workflow_id,
                checkpoint_name,
                exc,
            )
            return False

    def load_snapshot(
        self,
        workflow_id: str,
        checkpoint_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load and deserialize a workflow state snapshot from disk.

        If checkpoint_name is None, loads the most recent snapshot for the
        given workflow_id (by saved_at timestamp).

        Args:
            workflow_id: Unique workflow identifier.
            checkpoint_name: Specific checkpoint to load, or None for latest.

        Returns:
            Deserialized WorkflowState dict, or None if not found / corrupted.
        """
        if checkpoint_name is not None:
            return self._load_single_snapshot(workflow_id, checkpoint_name)

        # Find the most recent snapshot for this workflow
        candidates = self._find_snapshots_for_workflow(workflow_id)
        if not candidates:
            logger.debug("No snapshots found for workflow_id=%s", workflow_id)
            return None

        # Sort by saved_at descending, try each until one loads successfully
        candidates.sort(key=lambda p: self._read_saved_at(p), reverse=True)
        for path in candidates:
            result = self._load_snapshot_file(path)
            if result is not None:
                return result

        logger.warning(
            "All snapshots for workflow_id=%s are corrupted or invalid", workflow_id
        )
        return None

    def load_snapshot_with_fallback(
        self,
        workflow_id: str,
        fallback_objection_text: str = "Unknown objection",
    ) -> Dict[str, Any]:
        """
        Load the last valid snapshot, falling back to a safe initial state if
        all snapshots are corrupted or missing.

        Args:
            workflow_id: Unique workflow identifier.
            fallback_objection_text: Objection text to use when creating the
                                     safe initial state fallback.

        Returns:
            WorkflowState dict — either the recovered state or a fresh initial state.
        """
        recovered = self.load_snapshot(workflow_id)
        if recovered is not None:
            logger.info(
                "Disaster recovery: loaded snapshot for workflow_id=%s", workflow_id
            )
            return recovered

        logger.warning(
            "Disaster recovery: no valid snapshot for workflow_id=%s — "
            "falling back to safe initial state",
            workflow_id,
        )
        safe_state = create_initial_workflow_state(fallback_objection_text)
        # Preserve the original workflow_id so callers can track it
        safe_state["workflow_id"] = workflow_id
        return safe_state

    def list_recoverable_workflows(self) -> List[Dict[str, Any]]:
        """
        Enumerate all saved snapshots with their metadata.

        Returns:
            List of dicts with keys: workflow_id, checkpoint_name, status,
            saved_at, path.  Sorted by saved_at descending (most recent first).
        """
        results: List[Dict[str, Any]] = []

        try:
            for path in self.snapshot_dir.glob("*.json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    results.append(
                        {
                            "workflow_id": data.get("workflow_id", path.stem),
                            "checkpoint_name": data.get("checkpoint_name", "unknown"),
                            "status": data.get("status", "unknown"),
                            "saved_at": data.get("saved_at", ""),
                            "path": str(path),
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping unreadable snapshot %s: %s", path, exc
                    )
        except Exception as exc:
            logger.warning("Failed to list snapshots: %s", exc)

        results.sort(key=lambda r: r.get("saved_at", ""), reverse=True)
        return results

    def cleanup_old_snapshots(
        self, retention_days: Optional[int] = None
    ) -> int:
        """
        Delete snapshots older than the retention period.

        Args:
            retention_days: Override the instance-level retention_days.
                            Defaults to self.retention_days.

        Returns:
            Number of snapshot files deleted.
        """
        days = retention_days if retention_days is not None else self.retention_days
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0

        try:
            for path in self.snapshot_dir.glob("*.json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    saved_at_str = data.get("saved_at", "")
                    if not saved_at_str:
                        continue

                    saved_at = datetime.fromisoformat(saved_at_str)
                    if saved_at < cutoff:
                        path.unlink()
                        deleted += 1
                        logger.debug("Deleted old snapshot: %s", path)

                except Exception as exc:
                    logger.warning(
                        "Error processing snapshot %s during cleanup: %s", path, exc
                    )
        except Exception as exc:
            logger.warning("Failed to cleanup snapshots: %s", exc)

        if deleted:
            logger.info(
                "Cleanup complete: deleted %d snapshots older than %d days",
                deleted,
                days,
            )
        return deleted

    def delete_snapshot(
        self, workflow_id: str, checkpoint_name: Optional[str] = None
    ) -> int:
        """
        Delete snapshot(s) for a workflow.

        Args:
            workflow_id: Workflow identifier.
            checkpoint_name: Specific checkpoint to delete, or None to delete all
                             snapshots for this workflow.

        Returns:
            Number of files deleted.
        """
        deleted = 0
        if checkpoint_name is not None:
            path = self._snapshot_path(workflow_id, checkpoint_name)
            if path.exists():
                path.unlink()
                deleted = 1
        else:
            for path in self._find_snapshots_for_workflow(workflow_id):
                try:
                    path.unlink()
                    deleted += 1
                except Exception as exc:
                    logger.warning("Failed to delete snapshot %s: %s", path, exc)
        return deleted

    def snapshot_exists(
        self, workflow_id: str, checkpoint_name: Optional[str] = None
    ) -> bool:
        """Check whether a snapshot exists for the given workflow."""
        if checkpoint_name is not None:
            return self._snapshot_path(workflow_id, checkpoint_name).exists()
        return bool(self._find_snapshots_for_workflow(workflow_id))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_path(self, workflow_id: str, checkpoint_name: str) -> Path:
        """Return the full path for a snapshot file."""
        safe_name = checkpoint_name.replace("/", "_").replace("\\", "_")
        return self.snapshot_dir / f"{workflow_id}__{safe_name}.json"

    def _find_snapshots_for_workflow(self, workflow_id: str) -> List[Path]:
        """Return all snapshot paths that belong to a given workflow_id."""
        try:
            return list(self.snapshot_dir.glob(f"{workflow_id}__*.json"))
        except Exception:
            return []

    def _load_single_snapshot(
        self, workflow_id: str, checkpoint_name: str
    ) -> Optional[Dict[str, Any]]:
        """Load a specific named snapshot."""
        path = self._snapshot_path(workflow_id, checkpoint_name)
        if not path.exists():
            logger.debug(
                "Snapshot not found: workflow_id=%s checkpoint=%s",
                workflow_id,
                checkpoint_name,
            )
            return None
        return self._load_snapshot_file(path)

    def _load_snapshot_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Load and validate a snapshot file.

        Returns the deserialized WorkflowState dict, or None if the file is
        corrupted or fails Pydantic validation.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)

            raw_state = snapshot.get("state", {})

            # Validate via Pydantic — raises ValidationError on bad data
            validator = WorkflowStateValidator(**raw_state)
            return validator.model_dump()

        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "Corrupted or invalid snapshot %s: %s", path, exc
            )
            return None
        except Exception as exc:
            logger.warning("Failed to load snapshot %s: %s", path, exc)
            return None

    def _read_saved_at(self, path: Path) -> str:
        """Read the saved_at field from a snapshot file for sorting."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("saved_at", "")
        except Exception:
            return ""

    def _ensure_dir(self) -> None:
        """Create snapshot directory if it does not exist."""
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(
                "Could not create snapshot directory %s: %s",
                self.snapshot_dir,
                exc,
            )


__all__ = ["DisasterRecoveryManager"]
