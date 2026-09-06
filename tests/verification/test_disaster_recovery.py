"""
Unit tests for DisasterRecoveryManager - Task 6.3.5

Tests cover:
- Saving and loading state snapshots
- Resuming a workflow from a saved checkpoint
- Handling corrupted state files gracefully
- Cleanup of old snapshots
- list_recoverable_workflows()
- Fallback to safe initial state when all snapshots are invalid
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from backend.verification.utils.disaster_recovery import DisasterRecoveryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_state(workflow_id: str = "wf_dr_001") -> Dict[str, Any]:
    """Return a minimal but valid WorkflowState dict."""
    return {
        "objection_text": "iPhone quá đắt so với Samsung, tại sao tôi nên mua?",
        "customer_context": {},
        "draft_response": "",
        "tools_used": [],
        "research_reasoning": "",
        "research_sources": [],
        "verification_result": None,
        "correction_feedback": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_response": "",
        "workflow_status": "initialized",
        "execution_log": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "resource_usage": {
            "cpu_time_seconds": 0.0,
            "memory_peak_mb": 0.0,
            "llm_tokens_total": 0,
            "llm_cost_usd": 0.0,
            "db_queries_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        },
        "error_log": [],
        "config": {},
        "workflow_id": workflow_id,
        "correlation_id": "corr_abc123def456",
    }


@pytest.fixture
def manager(tmp_path):
    """DisasterRecoveryManager backed by a temporary directory."""
    return DisasterRecoveryManager(snapshot_dir=str(tmp_path), retention_days=7)


# ---------------------------------------------------------------------------
# save_snapshot()
# ---------------------------------------------------------------------------

class TestSaveSnapshot:
    def test_save_creates_json_file(self, manager, tmp_path):
        state = _minimal_state("wf_save_001")
        result = manager.save_snapshot(state, "post_research")

        assert result is True
        assert (tmp_path / "wf_save_001__post_research.json").exists()

    def test_save_file_contains_required_metadata(self, manager, tmp_path):
        state = _minimal_state("wf_meta_001")
        manager.save_snapshot(state, "pre_verification")

        with open(tmp_path / "wf_meta_001__pre_verification.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["workflow_id"] == "wf_meta_001"
        assert data["checkpoint_name"] == "pre_verification"
        assert "saved_at" in data
        assert "status" in data
        assert "state" in data

    def test_save_status_reflects_workflow_status(self, manager, tmp_path):
        state = _minimal_state("wf_status_001")
        state["workflow_status"] = "verifying"
        manager.save_snapshot(state, "checkpoint")

        with open(tmp_path / "wf_status_001__checkpoint.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "verifying"

    def test_save_returns_false_when_workflow_id_missing(self, manager):
        state = _minimal_state()
        del state["workflow_id"]
        result = manager.save_snapshot(state, "checkpoint")
        assert result is False

    def test_save_returns_false_on_io_error(self, tmp_path):
        mgr = DisasterRecoveryManager(snapshot_dir=str(tmp_path))
        state = _minimal_state("wf_io_err")

        with patch("builtins.open", side_effect=OSError("disk full")):
            result = mgr.save_snapshot(state, "checkpoint")

        assert result is False

    def test_save_default_checkpoint_name(self, manager, tmp_path):
        state = _minimal_state("wf_default_cp")
        result = manager.save_snapshot(state)

        assert result is True
        assert (tmp_path / "wf_default_cp__checkpoint.json").exists()

    def test_save_multiple_checkpoints_same_workflow(self, manager, tmp_path):
        state = _minimal_state("wf_multi_cp")
        manager.save_snapshot(state, "post_research")
        state["workflow_status"] = "verifying"
        manager.save_snapshot(state, "post_verification")

        assert (tmp_path / "wf_multi_cp__post_research.json").exists()
        assert (tmp_path / "wf_multi_cp__post_verification.json").exists()


# ---------------------------------------------------------------------------
# load_snapshot()
# ---------------------------------------------------------------------------

class TestLoadSnapshot:
    def test_load_returns_none_when_not_found(self, manager):
        result = manager.load_snapshot("wf_nonexistent")
        assert result is None

    def test_load_returns_dict_after_save(self, manager):
        state = _minimal_state("wf_load_001")
        manager.save_snapshot(state, "checkpoint")

        loaded = manager.load_snapshot("wf_load_001", "checkpoint")
        assert loaded is not None
        assert isinstance(loaded, dict)

    def test_load_preserves_workflow_id(self, manager):
        state = _minimal_state("wf_load_002")
        manager.save_snapshot(state, "checkpoint")

        loaded = manager.load_snapshot("wf_load_002", "checkpoint")
        assert loaded["workflow_id"] == "wf_load_002"

    def test_load_preserves_objection_text(self, manager):
        state = _minimal_state("wf_load_003")
        state["objection_text"] = "Sản phẩm có bảo hành không?"
        manager.save_snapshot(state, "checkpoint")

        loaded = manager.load_snapshot("wf_load_003", "checkpoint")
        assert loaded["objection_text"] == "Sản phẩm có bảo hành không?"

    def test_load_preserves_workflow_status(self, manager):
        state = _minimal_state("wf_load_004")
        state["workflow_status"] = "correcting"
        manager.save_snapshot(state, "checkpoint")

        loaded = manager.load_snapshot("wf_load_004", "checkpoint")
        assert loaded["workflow_status"] == "correcting"

    def test_load_returns_none_on_corrupt_json(self, manager, tmp_path):
        corrupt_path = tmp_path / "wf_corrupt__checkpoint.json"
        corrupt_path.write_text("not valid json", encoding="utf-8")

        result = manager.load_snapshot("wf_corrupt", "checkpoint")
        assert result is None

    def test_load_returns_none_on_invalid_state(self, manager, tmp_path):
        """A JSON file with invalid WorkflowState data should return None."""
        bad_path = tmp_path / "wf_bad__checkpoint.json"
        bad_data = {
            "saved_at": datetime.now().isoformat(),
            "workflow_id": "wf_bad",
            "checkpoint_name": "checkpoint",
            "status": "initialized",
            "state": {"invalid_field": "garbage"},  # fails Pydantic validation
        }
        bad_path.write_text(json.dumps(bad_data), encoding="utf-8")

        result = manager.load_snapshot("wf_bad", "checkpoint")
        assert result is None

    def test_load_latest_when_no_checkpoint_name(self, manager):
        """load_snapshot without checkpoint_name returns the most recent snapshot."""
        state = _minimal_state("wf_latest")
        manager.save_snapshot(state, "first")
        state["workflow_status"] = "verifying"
        manager.save_snapshot(state, "second")

        loaded = manager.load_snapshot("wf_latest")
        assert loaded is not None
        # Should be the most recent (second) snapshot
        assert loaded["workflow_status"] == "verifying"

    def test_roundtrip_save_load(self, manager):
        state = _minimal_state("wf_roundtrip")
        state["workflow_status"] = "researching"
        state["retry_count"] = 1
        manager.save_snapshot(state, "checkpoint")

        loaded = manager.load_snapshot("wf_roundtrip", "checkpoint")
        assert loaded["workflow_status"] == "researching"
        assert loaded["retry_count"] == 1
        assert loaded["objection_text"] == state["objection_text"]


# ---------------------------------------------------------------------------
# load_snapshot_with_fallback()
# ---------------------------------------------------------------------------

class TestLoadSnapshotWithFallback:
    def test_returns_saved_state_when_snapshot_exists(self, manager):
        state = _minimal_state("wf_fallback_001")
        state["workflow_status"] = "verifying"
        manager.save_snapshot(state, "checkpoint")

        result = manager.load_snapshot_with_fallback("wf_fallback_001")
        assert result["workflow_status"] == "verifying"
        assert result["workflow_id"] == "wf_fallback_001"

    def test_returns_initial_state_when_no_snapshot(self, manager):
        result = manager.load_snapshot_with_fallback(
            "wf_no_snapshot", fallback_objection_text="Test objection text here"
        )
        assert result is not None
        assert isinstance(result, dict)
        # Fallback state should have the original workflow_id preserved
        assert result["workflow_id"] == "wf_no_snapshot"
        # And the fallback objection text
        assert result["objection_text"] == "Test objection text here"

    def test_fallback_state_is_valid_workflow_state(self, manager):
        """The fallback initial state must pass WorkflowStateValidator."""
        from backend.verification.models.state import WorkflowStateValidator

        result = manager.load_snapshot_with_fallback(
            "wf_fallback_valid", fallback_objection_text="Valid objection text here"
        )
        # Should not raise
        validator = WorkflowStateValidator(**result)
        assert validator.workflow_status == "initialized"

    def test_returns_fallback_when_all_snapshots_corrupted(self, manager, tmp_path):
        """When all snapshots are corrupted, fallback to safe initial state."""
        corrupt_path = tmp_path / "wf_all_corrupt__checkpoint.json"
        corrupt_path.write_text("not valid json", encoding="utf-8")

        result = manager.load_snapshot_with_fallback(
            "wf_all_corrupt", fallback_objection_text="Fallback objection text here"
        )
        assert result is not None
        assert result["workflow_id"] == "wf_all_corrupt"
        assert result["workflow_status"] == "initialized"


# ---------------------------------------------------------------------------
# list_recoverable_workflows()
# ---------------------------------------------------------------------------

class TestListRecoverableWorkflows:
    def test_empty_when_no_snapshots(self, manager):
        assert manager.list_recoverable_workflows() == []

    def test_returns_one_entry_after_save(self, manager):
        manager.save_snapshot(_minimal_state("wf_list_001"), "checkpoint")
        entries = manager.list_recoverable_workflows()
        assert len(entries) == 1

    def test_returns_multiple_entries(self, manager):
        for i in range(3):
            manager.save_snapshot(_minimal_state(f"wf_list_{i:03d}"), "checkpoint")
        entries = manager.list_recoverable_workflows()
        assert len(entries) == 3

    def test_entry_has_required_keys(self, manager):
        manager.save_snapshot(_minimal_state("wf_list_keys"), "post_research")
        entry = manager.list_recoverable_workflows()[0]
        assert "workflow_id" in entry
        assert "checkpoint_name" in entry
        assert "status" in entry
        assert "saved_at" in entry
        assert "path" in entry

    def test_entry_checkpoint_name_matches(self, manager):
        manager.save_snapshot(_minimal_state("wf_list_cp"), "post_research")
        entry = manager.list_recoverable_workflows()[0]
        assert entry["checkpoint_name"] == "post_research"

    def test_skips_unreadable_files(self, manager, tmp_path):
        (tmp_path / "bad_file.json").write_text("garbage", encoding="utf-8")
        manager.save_snapshot(_minimal_state("wf_list_good"), "checkpoint")

        entries = manager.list_recoverable_workflows()
        assert len(entries) == 1
        assert entries[0]["workflow_id"] == "wf_list_good"

    def test_sorted_most_recent_first(self, manager):
        """Entries should be sorted by saved_at descending."""
        manager.save_snapshot(_minimal_state("wf_sort_a"), "checkpoint")
        manager.save_snapshot(_minimal_state("wf_sort_b"), "checkpoint")

        entries = manager.list_recoverable_workflows()
        assert len(entries) == 2
        # Most recent first
        assert entries[0]["saved_at"] >= entries[1]["saved_at"]


# ---------------------------------------------------------------------------
# cleanup_old_snapshots()
# ---------------------------------------------------------------------------

class TestCleanupOldSnapshots:
    def _write_snapshot_with_age(
        self, tmp_path: Path, workflow_id: str, days_old: int
    ) -> None:
        """Write a snapshot file with a saved_at timestamp `days_old` days ago."""
        saved_at = (datetime.now() - timedelta(days=days_old)).isoformat()
        state = _minimal_state(workflow_id)
        from backend.verification.models.state import WorkflowStateValidator
        validator = WorkflowStateValidator(**state)
        snapshot = {
            "saved_at": saved_at,
            "workflow_id": workflow_id,
            "checkpoint_name": "checkpoint",
            "status": "initialized",
            "state": json.loads(validator.model_dump_json()),
        }
        path = tmp_path / f"{workflow_id}__checkpoint.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")

    def test_deletes_snapshots_older_than_retention(self, tmp_path):
        mgr = DisasterRecoveryManager(snapshot_dir=str(tmp_path), retention_days=7)
        self._write_snapshot_with_age(tmp_path, "wf_old", days_old=10)
        self._write_snapshot_with_age(tmp_path, "wf_new", days_old=2)

        deleted = mgr.cleanup_old_snapshots()

        assert deleted == 1
        assert not (tmp_path / "wf_old__checkpoint.json").exists()
        assert (tmp_path / "wf_new__checkpoint.json").exists()

    def test_keeps_snapshots_within_retention(self, tmp_path):
        mgr = DisasterRecoveryManager(snapshot_dir=str(tmp_path), retention_days=7)
        self._write_snapshot_with_age(tmp_path, "wf_recent", days_old=3)

        deleted = mgr.cleanup_old_snapshots()

        assert deleted == 0
        assert (tmp_path / "wf_recent__checkpoint.json").exists()

    def test_returns_zero_when_nothing_to_delete(self, manager):
        deleted = manager.cleanup_old_snapshots()
        assert deleted == 0

    def test_override_retention_days(self, tmp_path):
        mgr = DisasterRecoveryManager(snapshot_dir=str(tmp_path), retention_days=30)
        self._write_snapshot_with_age(tmp_path, "wf_mid", days_old=10)

        # With default 30-day retention, nothing should be deleted
        deleted = mgr.cleanup_old_snapshots()
        assert deleted == 0

        # Override to 5 days — now it should be deleted
        deleted = mgr.cleanup_old_snapshots(retention_days=5)
        assert deleted == 1

    def test_deletes_all_old_snapshots(self, tmp_path):
        mgr = DisasterRecoveryManager(snapshot_dir=str(tmp_path), retention_days=7)
        for i in range(3):
            self._write_snapshot_with_age(tmp_path, f"wf_old_{i}", days_old=15)

        deleted = mgr.cleanup_old_snapshots()
        assert deleted == 3


# ---------------------------------------------------------------------------
# snapshot_exists() and delete_snapshot()
# ---------------------------------------------------------------------------

class TestSnapshotExistsAndDelete:
    def test_exists_false_before_save(self, manager):
        assert manager.snapshot_exists("wf_exists_001") is False

    def test_exists_true_after_save(self, manager):
        manager.save_snapshot(_minimal_state("wf_exists_002"), "checkpoint")
        assert manager.snapshot_exists("wf_exists_002") is True

    def test_exists_with_checkpoint_name(self, manager):
        manager.save_snapshot(_minimal_state("wf_exists_003"), "post_research")
        assert manager.snapshot_exists("wf_exists_003", "post_research") is True
        assert manager.snapshot_exists("wf_exists_003", "other_checkpoint") is False

    def test_delete_specific_checkpoint(self, manager, tmp_path):
        state = _minimal_state("wf_del_001")
        manager.save_snapshot(state, "post_research")
        manager.save_snapshot(state, "post_verification")

        deleted = manager.delete_snapshot("wf_del_001", "post_research")
        assert deleted == 1
        assert not (tmp_path / "wf_del_001__post_research.json").exists()
        assert (tmp_path / "wf_del_001__post_verification.json").exists()

    def test_delete_all_checkpoints_for_workflow(self, manager, tmp_path):
        state = _minimal_state("wf_del_002")
        manager.save_snapshot(state, "post_research")
        manager.save_snapshot(state, "post_verification")

        deleted = manager.delete_snapshot("wf_del_002")
        assert deleted == 2
        assert not manager.snapshot_exists("wf_del_002")

    def test_delete_returns_zero_when_not_found(self, manager):
        deleted = manager.delete_snapshot("wf_nonexistent")
        assert deleted == 0


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

class TestDirectoryCreation:
    def test_creates_nested_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        mgr = DisasterRecoveryManager(snapshot_dir=str(nested))
        assert nested.exists()

    def test_works_with_existing_directory(self, tmp_path):
        mgr = DisasterRecoveryManager(snapshot_dir=str(tmp_path))
        state = _minimal_state("wf_dir_001")
        assert mgr.save_snapshot(state, "checkpoint") is True
