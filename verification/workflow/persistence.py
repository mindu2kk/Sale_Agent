"""
Workflow Persistence Manager

Implements workflow state persistence với Pydantic serialization:
- Save/load WorkflowState to/from disk as JSON
- Resume capabilities after failures
- List resumable checkpoints
- Auto-checkpoint after each node execution

Supports Requirement 7.4: Workflow persistence để support resume after failures
Supports Requirement 8.3: Rollback to last valid state on StateGraph execution error
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.state import WorkflowStateValidator, validate_workflow_state

logger = logging.getLogger(__name__)

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = "logs/workflow"


class WorkflowPersistenceManager:
    """
    Manages workflow state persistence với Pydantic serialization.

    Saves WorkflowState checkpoints to disk as JSON files so that
    interrupted workflows can be resumed from the last saved state.

    File layout:
        {checkpoint_dir}/{workflow_id}.json

    Each checkpoint file contains:
        - saved_at: ISO timestamp of when the checkpoint was written
        - workflow_id: unique workflow identifier
        - status: current workflow_status value
        - state: full serialized WorkflowState (via WorkflowStateValidator)
    """

    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> None:
        """
        Initialize persistence manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
                            Created automatically if it does not exist.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self._ensure_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, state: Dict[str, Any]) -> bool:
        """
        Save workflow state to disk as JSON.

        Uses WorkflowStateValidator (Pydantic v2) for serialization so that
        all nested models (ExecutionStep, VerificationResult, …) are correctly
        serialized.

        Args:
            state: WorkflowState dict to persist.

        Returns:
            True on success, False on I/O error (warning is logged).
        """
        workflow_id = state.get("workflow_id")
        if not workflow_id:
            logger.warning("Cannot save workflow state: missing workflow_id")
            return False

        try:
            # Validate & serialize via Pydantic v2
            validator = WorkflowStateValidator(**state)
            state_json = validator.model_dump_json()

            checkpoint = {
                "saved_at": datetime.now().isoformat(),
                "workflow_id": workflow_id,
                "status": state.get("workflow_status", "unknown"),
                "state": json.loads(state_json),
            }

            checkpoint_path = self._checkpoint_path(workflow_id)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2, default=str)

            logger.debug("Checkpoint saved: %s", checkpoint_path)
            return True

        except Exception as exc:
            logger.warning("Failed to save checkpoint for %s: %s", workflow_id, exc)
            return False

    def load(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Load and deserialize a workflow state checkpoint from disk.

        Args:
            workflow_id: Unique workflow identifier used as filename key.

        Returns:
            Deserialized WorkflowState dict, or None if not found / invalid.
        """
        checkpoint_path = self._checkpoint_path(workflow_id)

        if not checkpoint_path.exists():
            logger.debug("No checkpoint found for workflow_id=%s", workflow_id)
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            raw_state = checkpoint.get("state", {})

            # Validate via Pydantic v2 — raises ValidationError on bad data
            validator = WorkflowStateValidator(**raw_state)
            return validator.model_dump()

        except Exception as exc:
            logger.warning("Failed to load checkpoint for %s: %s", workflow_id, exc)
            return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all saved checkpoints with their metadata.

        Returns:
            List of dicts with keys: workflow_id, status, saved_at, path.
            Sorted by saved_at descending (most recent first).
        """
        checkpoints: List[Dict[str, Any]] = []

        try:
            for path in self.checkpoint_dir.glob("*.json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    checkpoints.append(
                        {
                            "workflow_id": data.get("workflow_id", path.stem),
                            "status": data.get("status", "unknown"),
                            "saved_at": data.get("saved_at", ""),
                            "path": str(path),
                        }
                    )
                except Exception as exc:
                    logger.warning("Skipping unreadable checkpoint %s: %s", path, exc)

        except Exception as exc:
            logger.warning("Failed to list checkpoints: %s", exc)

        # Most recent first
        checkpoints.sort(key=lambda c: c.get("saved_at", ""), reverse=True)
        return checkpoints

    def delete(self, workflow_id: str) -> bool:
        """
        Delete a checkpoint file after successful workflow completion.

        Args:
            workflow_id: Workflow identifier whose checkpoint to remove.

        Returns:
            True if deleted, False if not found or error.
        """
        checkpoint_path = self._checkpoint_path(workflow_id)

        if not checkpoint_path.exists():
            return False

        try:
            checkpoint_path.unlink()
            logger.debug("Checkpoint deleted: %s", checkpoint_path)
            return True
        except Exception as exc:
            logger.warning("Failed to delete checkpoint for %s: %s", workflow_id, exc)
            return False

    def exists(self, workflow_id: str) -> bool:
        """Check whether a checkpoint exists for the given workflow_id."""
        return self._checkpoint_path(workflow_id).exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, workflow_id: str) -> Path:
        """Return the full path for a workflow checkpoint file."""
        return self.checkpoint_dir / f"{workflow_id}.json"

    def _ensure_dir(self) -> None:
        """Create checkpoint directory if it does not exist."""
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Could not create checkpoint directory %s: %s", self.checkpoint_dir, exc)
