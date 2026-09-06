"""
Backup Manager với Pydantic State Serialization

Provides timestamped, incremental backup procedures for WorkflowState:
- Serialize WorkflowState (TypedDict + nested Pydantic models) to JSON
- Save backups with timestamped filenames: {workflow_id}_{timestamp}.json
- Checkpoint-based incremental backups at each workflow node transition
- Restore state from backup with full Pydantic validation
- Configurable retention: auto-cleanup of old backups
- Async save/restore using aiofiles (fallback to asyncio.to_thread)

Supports Requirement 7: Workflow persistence to support resume after failures
Supports Requirement 8: Rollback to last valid state on StateGraph execution error
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiofiles
    _AIOFILES_AVAILABLE = True
except ImportError:
    _AIOFILES_AVAILABLE = False

from ..models.state import WorkflowStateValidator

logger = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = "backups/workflow_states"
DEFAULT_MAX_BACKUPS = 10


class BackupManager:
    """
    Manages timestamped backups of WorkflowState with Pydantic serialization.

    File layout:
        {backup_dir}/{workflow_id}_{YYYYMMDD_HHMMSS_ffffff}.json

    Each backup file contains:
        - backup_id: unique backup identifier
        - workflow_id: workflow identifier
        - node_name: workflow node that triggered the backup (optional)
        - created_at: ISO timestamp
        - state: full serialized WorkflowState (via WorkflowStateValidator)

    Retention:
        When max_backups is set, the oldest backups for a given workflow_id
        are automatically removed after each save, keeping only the most
        recent N backups.
    """

    def __init__(
        self,
        backup_dir: str = DEFAULT_BACKUP_DIR,
        max_backups: int = DEFAULT_MAX_BACKUPS,
    ) -> None:
        """
        Initialize BackupManager.

        Args:
            backup_dir: Directory to store backup files.
                        Created automatically if it does not exist.
            max_backups: Maximum number of backups to retain per workflow_id.
                         Older backups are deleted when this limit is exceeded.
                         Set to 0 to disable retention cleanup.
        """
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self._ensure_dir()

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def save(
        self,
        state: Dict[str, Any],
        node_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Serialize and save WorkflowState to a timestamped backup file.

        Uses WorkflowStateValidator (Pydantic v2) so that all nested models
        (ExecutionStep, VerificationResult, …) are correctly serialized.

        Args:
            state: WorkflowState dict to back up.
            node_name: Optional name of the workflow node triggering the backup.

        Returns:
            Path to the created backup file, or None on failure.
        """
        workflow_id = state.get("workflow_id")
        if not workflow_id:
            logger.warning("Cannot save backup: missing workflow_id")
            return None

        try:
            validator = WorkflowStateValidator(**state)
            state_data = json.loads(validator.model_dump_json())

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{workflow_id}_{timestamp}.json"
            backup_path = self.backup_dir / filename

            backup_doc = {
                "backup_id": f"bkp_{timestamp}",
                "workflow_id": workflow_id,
                "node_name": node_name,
                "created_at": datetime.now().isoformat(),
                "state": state_data,
            }

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_doc, f, ensure_ascii=False, indent=2, default=str)

            logger.debug("Backup saved: %s", backup_path)

            # Enforce retention policy
            if self.max_backups > 0:
                self._enforce_retention(workflow_id)

            return str(backup_path)

        except Exception as exc:
            logger.warning("Failed to save backup for %s: %s", workflow_id, exc)
            return None

    def restore(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """
        Restore WorkflowState from a backup file with full Pydantic validation.

        Args:
            backup_path: Path to the backup JSON file.

        Returns:
            Deserialized WorkflowState dict, or None if not found / invalid.
        """
        path = Path(backup_path)
        if not path.exists():
            logger.debug("Backup file not found: %s", backup_path)
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                backup_doc = json.load(f)

            raw_state = backup_doc.get("state", {})
            validator = WorkflowStateValidator(**raw_state)
            return validator.model_dump()

        except Exception as exc:
            logger.warning("Failed to restore backup from %s: %s", backup_path, exc)
            return None

    def restore_latest(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore the most recent backup for a given workflow_id.

        Args:
            workflow_id: Workflow identifier to look up.

        Returns:
            Deserialized WorkflowState dict, or None if no backup exists.
        """
        backups = self.list_backups(workflow_id)
        if not backups:
            return None
        # list_backups returns most-recent-first
        return self.restore(backups[0]["path"])

    def list_backups(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List backup files, optionally filtered by workflow_id.

        Args:
            workflow_id: If provided, only return backups for this workflow.

        Returns:
            List of dicts with keys: backup_id, workflow_id, node_name,
            created_at, path. Sorted by created_at descending (newest first).
        """
        results: List[Dict[str, Any]] = []

        try:
            pattern = f"{workflow_id}_*.json" if workflow_id else "*.json"
            for path in self.backup_dir.glob(pattern):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                    results.append(
                        {
                            "backup_id": doc.get("backup_id", path.stem),
                            "workflow_id": doc.get("workflow_id", ""),
                            "node_name": doc.get("node_name"),
                            "created_at": doc.get("created_at", ""),
                            "path": str(path),
                        }
                    )
                except Exception as exc:
                    logger.warning("Skipping unreadable backup %s: %s", path, exc)

        except Exception as exc:
            logger.warning("Failed to list backups: %s", exc)

        results.sort(key=lambda b: b.get("created_at", ""), reverse=True)
        return results

    def delete_backup(self, backup_path: str) -> bool:
        """
        Delete a specific backup file.

        Args:
            backup_path: Path to the backup file to remove.

        Returns:
            True if deleted, False if not found or error.
        """
        path = Path(backup_path)
        if not path.exists():
            return False
        try:
            path.unlink()
            logger.debug("Backup deleted: %s", backup_path)
            return True
        except Exception as exc:
            logger.warning("Failed to delete backup %s: %s", backup_path, exc)
            return False

    def delete_all_backups(self, workflow_id: str) -> int:
        """
        Delete all backups for a given workflow_id.

        Args:
            workflow_id: Workflow identifier whose backups to remove.

        Returns:
            Number of backup files deleted.
        """
        backups = self.list_backups(workflow_id)
        deleted = 0
        for backup in backups:
            if self.delete_backup(backup["path"]):
                deleted += 1
        return deleted

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def async_save(
        self,
        state: Dict[str, Any],
        node_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Async version of save(). Uses aiofiles when available, otherwise
        falls back to asyncio.to_thread for non-blocking I/O.

        Args:
            state: WorkflowState dict to back up.
            node_name: Optional workflow node name triggering the backup.

        Returns:
            Path to the created backup file, or None on failure.
        """
        workflow_id = state.get("workflow_id")
        if not workflow_id:
            logger.warning("Cannot save backup: missing workflow_id")
            return None

        try:
            validator = WorkflowStateValidator(**state)
            state_data = json.loads(validator.model_dump_json())

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{workflow_id}_{timestamp}.json"
            backup_path = self.backup_dir / filename

            backup_doc = {
                "backup_id": f"bkp_{timestamp}",
                "workflow_id": workflow_id,
                "node_name": node_name,
                "created_at": datetime.now().isoformat(),
                "state": state_data,
            }

            content = json.dumps(backup_doc, ensure_ascii=False, indent=2, default=str)

            if _AIOFILES_AVAILABLE:
                async with aiofiles.open(backup_path, "w", encoding="utf-8") as f:
                    await f.write(content)
            else:
                await asyncio.to_thread(
                    backup_path.write_text, content, "utf-8"
                )

            logger.debug("Async backup saved: %s", backup_path)

            if self.max_backups > 0:
                await asyncio.to_thread(self._enforce_retention, workflow_id)

            return str(backup_path)

        except Exception as exc:
            logger.warning("Failed to async save backup for %s: %s", workflow_id, exc)
            return None

    async def async_restore(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """
        Async version of restore(). Uses aiofiles when available, otherwise
        falls back to asyncio.to_thread.

        Args:
            backup_path: Path to the backup JSON file.

        Returns:
            Deserialized WorkflowState dict, or None if not found / invalid.
        """
        path = Path(backup_path)
        if not path.exists():
            logger.debug("Backup file not found: %s", backup_path)
            return None

        try:
            if _AIOFILES_AVAILABLE:
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    content = await f.read()
                backup_doc = json.loads(content)
            else:
                backup_doc = await asyncio.to_thread(
                    lambda: json.loads(path.read_text("utf-8"))
                )

            raw_state = backup_doc.get("state", {})
            validator = WorkflowStateValidator(**raw_state)
            return validator.model_dump()

        except Exception as exc:
            logger.warning("Failed to async restore backup from %s: %s", backup_path, exc)
            return None

    async def async_restore_latest(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Async version of restore_latest().

        Args:
            workflow_id: Workflow identifier to look up.

        Returns:
            Deserialized WorkflowState dict, or None if no backup exists.
        """
        backups = self.list_backups(workflow_id)
        if not backups:
            return None
        return await self.async_restore(backups[0]["path"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        """Create backup directory if it does not exist."""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(
                "Could not create backup directory %s: %s", self.backup_dir, exc
            )

    def _enforce_retention(self, workflow_id: str) -> None:
        """
        Delete oldest backups for workflow_id when count exceeds max_backups.

        Keeps the most recent max_backups files and removes the rest.
        """
        if self.max_backups <= 0:
            return

        backups = self.list_backups(workflow_id)  # newest first
        excess = backups[self.max_backups:]        # oldest entries beyond limit
        for backup in excess:
            try:
                Path(backup["path"]).unlink()
                logger.debug("Retention cleanup: deleted %s", backup["path"])
            except Exception as exc:
                logger.warning(
                    "Retention cleanup failed for %s: %s", backup["path"], exc
                )
