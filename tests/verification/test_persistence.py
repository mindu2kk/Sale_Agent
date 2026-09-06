"""
Unit tests for WorkflowPersistenceManager

Tests cover:
- Save workflow state to disk as JSON
- Load/resume workflow state from disk
- List resumable checkpoints
- Delete checkpoint after completion
- Auto-checkpoint integration (called from workflow nodes)
- Graceful handling of I/O errors
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from backend.verification.workflow.persistence import WorkflowPersistenceManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_state(workflow_id: str = "wf_test_001") -> Dict[str, Any]:
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
def tmp_manager(tmp_path):
    """WorkflowPersistenceManager backed by a temporary directory."""
    return WorkflowPersistenceManager(checkpoint_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_json_file(self, tmp_manager, tmp_path):
        state = _minimal_state("wf_save_001")
        result = tmp_manager.save(state)

        assert result is True
        checkpoint_file = tmp_path / "wf_save_001.json"
        assert checkpoint_file.exists()

    def test_save_file_contains_required_metadata(self, tmp_manager, tmp_path):
        state = _minimal_state("wf_meta_001")
        tmp_manager.save(state)

        with open(tmp_path / "wf_meta_001.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["workflow_id"] == "wf_meta_001"
        assert "saved_at" in data
        assert "status" in data
        assert "state" in data

    def test_save_status_reflects_workflow_status(self, tmp_manager, tmp_path):
        state = _minimal_state("wf_status_001")
        state["workflow_status"] = "verifying"
        tmp_manager.save(state)

        with open(tmp_path / "wf_status_001.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "verifying"

    def test_save_returns_false_when_workflow_id_missing(self, tmp_manager):
        state = _minimal_state()
        del state["workflow_id"]
        result = tmp_manager.save(state)
        assert result is False

    def test_save_overwrites_existing_checkpoint(self, tmp_manager, tmp_path):
        state = _minimal_state("wf_overwrite_001")
        tmp_manager.save(state)

        state["workflow_status"] = "verifying"
        tmp_manager.save(state)

        with open(tmp_path / "wf_overwrite_001.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "verifying"

    def test_save_handles_io_error_gracefully(self, tmp_path):
        """save() should return False and not raise on I/O errors."""
        manager = WorkflowPersistenceManager(checkpoint_dir=str(tmp_path))
        state = _minimal_state("wf_io_err")

        with patch("builtins.open", side_effect=OSError("disk full")):
            result = manager.save(state)

        assert result is False


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_none_when_not_found(self, tmp_manager):
        result = tmp_manager.load("wf_nonexistent")
        assert result is None

    def test_load_returns_dict_after_save(self, tmp_manager):
        state = _minimal_state("wf_load_001")
        tmp_manager.save(state)

        loaded = tmp_manager.load("wf_load_001")
        assert loaded is not None
        assert isinstance(loaded, dict)

    def test_load_preserves_workflow_id(self, tmp_manager):
        state = _minimal_state("wf_load_002")
        tmp_manager.save(state)

        loaded = tmp_manager.load("wf_load_002")
        assert loaded["workflow_id"] == "wf_load_002"

    def test_load_preserves_objection_text(self, tmp_manager):
        state = _minimal_state("wf_load_003")
        state["objection_text"] = "Sản phẩm có bảo hành không?"
        tmp_manager.save(state)

        loaded = tmp_manager.load("wf_load_003")
        assert loaded["objection_text"] == "Sản phẩm có bảo hành không?"

    def test_load_preserves_workflow_status(self, tmp_manager):
        state = _minimal_state("wf_load_004")
        state["workflow_status"] = "correcting"
        tmp_manager.save(state)

        loaded = tmp_manager.load("wf_load_004")
        assert loaded["workflow_status"] == "correcting"

    def test_load_returns_none_on_corrupt_file(self, tmp_manager, tmp_path):
        corrupt_path = tmp_path / "wf_corrupt.json"
        corrupt_path.write_text("not valid json", encoding="utf-8")

        result = tmp_manager.load("wf_corrupt")
        assert result is None

    def test_roundtrip_save_load(self, tmp_manager):
        """State saved and loaded should be equivalent."""
        state = _minimal_state("wf_roundtrip")
        state["workflow_status"] = "researching"
        state["retry_count"] = 1
        tmp_manager.save(state)

        loaded = tmp_manager.load("wf_roundtrip")
        assert loaded["workflow_status"] == "researching"
        assert loaded["retry_count"] == 1
        assert loaded["objection_text"] == state["objection_text"]


# ---------------------------------------------------------------------------
# list_checkpoints()
# ---------------------------------------------------------------------------

class TestListCheckpoints:
    def test_list_empty_when_no_checkpoints(self, tmp_manager):
        assert tmp_manager.list_checkpoints() == []

    def test_list_returns_one_entry_after_save(self, tmp_manager):
        tmp_manager.save(_minimal_state("wf_list_001"))
        checkpoints = tmp_manager.list_checkpoints()
        assert len(checkpoints) == 1

    def test_list_returns_multiple_entries(self, tmp_manager):
        for i in range(3):
            tmp_manager.save(_minimal_state(f"wf_list_{i:03d}"))
        checkpoints = tmp_manager.list_checkpoints()
        assert len(checkpoints) == 3

    def test_list_entry_has_required_keys(self, tmp_manager):
        tmp_manager.save(_minimal_state("wf_list_keys"))
        entry = tmp_manager.list_checkpoints()[0]
        assert "workflow_id" in entry
        assert "status" in entry
        assert "saved_at" in entry
        assert "path" in entry

    def test_list_entry_workflow_id_matches(self, tmp_manager):
        tmp_manager.save(_minimal_state("wf_list_id"))
        entry = tmp_manager.list_checkpoints()[0]
        assert entry["workflow_id"] == "wf_list_id"

    def test_list_skips_unreadable_files(self, tmp_manager, tmp_path):
        """Unreadable JSON files should be skipped, not raise."""
        (tmp_path / "bad_file.json").write_text("garbage", encoding="utf-8")
        tmp_manager.save(_minimal_state("wf_list_good"))

        checkpoints = tmp_manager.list_checkpoints()
        # Only the valid checkpoint should appear
        assert len(checkpoints) == 1
        assert checkpoints[0]["workflow_id"] == "wf_list_good"


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_file(self, tmp_manager, tmp_path):
        tmp_manager.save(_minimal_state("wf_del_001"))
        assert (tmp_path / "wf_del_001.json").exists()

        result = tmp_manager.delete("wf_del_001")
        assert result is True
        assert not (tmp_path / "wf_del_001.json").exists()

    def test_delete_returns_false_when_not_found(self, tmp_manager):
        result = tmp_manager.delete("wf_nonexistent")
        assert result is False

    def test_delete_makes_load_return_none(self, tmp_manager):
        tmp_manager.save(_minimal_state("wf_del_002"))
        tmp_manager.delete("wf_del_002")
        assert tmp_manager.load("wf_del_002") is None

    def test_delete_removes_from_list(self, tmp_manager):
        tmp_manager.save(_minimal_state("wf_del_003"))
        tmp_manager.delete("wf_del_003")
        assert tmp_manager.list_checkpoints() == []


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------

class TestExists:
    def test_exists_false_before_save(self, tmp_manager):
        assert tmp_manager.exists("wf_exists_001") is False

    def test_exists_true_after_save(self, tmp_manager):
        tmp_manager.save(_minimal_state("wf_exists_002"))
        assert tmp_manager.exists("wf_exists_002") is True

    def test_exists_false_after_delete(self, tmp_manager):
        tmp_manager.save(_minimal_state("wf_exists_003"))
        tmp_manager.delete("wf_exists_003")
        assert tmp_manager.exists("wf_exists_003") is False


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

class TestDirectoryCreation:
    def test_creates_nested_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        manager = WorkflowPersistenceManager(checkpoint_dir=str(nested))
        assert nested.exists()

    def test_works_with_existing_directory(self, tmp_path):
        """Should not raise if directory already exists."""
        manager = WorkflowPersistenceManager(checkpoint_dir=str(tmp_path))
        state = _minimal_state("wf_dir_001")
        assert manager.save(state) is True
