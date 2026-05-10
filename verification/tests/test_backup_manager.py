"""
Unit tests for BackupManager

Tests cover:
- Serialization round-trip (serialize → deserialize → same state)
- Backup file creation and naming convention
- Restore from backup (sync and async)
- Retention / cleanup logic
- Error handling (corrupt backup, missing file)
- Async save/restore methods
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from verification.utils.backup_manager import BackupManager, DEFAULT_BACKUP_DIR, DEFAULT_MAX_BACKUPS


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
def manager(tmp_path):
    """BackupManager backed by a temporary directory."""
    return BackupManager(backup_dir=str(tmp_path), max_backups=5)


@pytest.fixture
def manager_no_retention(tmp_path):
    """BackupManager with retention disabled (max_backups=0)."""
    return BackupManager(backup_dir=str(tmp_path), max_backups=0)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_backup_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        BackupManager(backup_dir=str(nested))
        assert nested.exists()

    def test_works_with_existing_directory(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path))
        assert mgr.save(_minimal_state("wf_init_001")) is not None

    def test_default_max_backups(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path))
        assert mgr.max_backups == DEFAULT_MAX_BACKUPS


# ---------------------------------------------------------------------------
# save() — file creation and naming
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_returns_path_string(self, manager):
        result = manager.save(_minimal_state("wf_save_001"))
        assert result is not None
        assert isinstance(result, str)

    def test_save_creates_json_file(self, manager, tmp_path):
        result = manager.save(_minimal_state("wf_save_002"))
        assert Path(result).exists()

    def test_save_filename_contains_workflow_id(self, manager):
        result = manager.save(_minimal_state("wf_naming_001"))
        assert "wf_naming_001" in Path(result).name

    def test_save_filename_contains_timestamp(self, manager):
        result = manager.save(_minimal_state("wf_naming_002"))
        # Filename format: {workflow_id}_{YYYYMMDD_HHMMSS_ffffff}.json
        stem = Path(result).stem
        parts = stem.split("_")
        # At minimum: wf, naming, 002, YYYYMMDD, HHMMSS, ffffff
        assert len(parts) >= 4

    def test_save_file_has_json_extension(self, manager):
        result = manager.save(_minimal_state("wf_ext_001"))
        assert result.endswith(".json")

    def test_save_file_contains_required_keys(self, manager):
        result = manager.save(_minimal_state("wf_keys_001"))
        with open(result, encoding="utf-8") as f:
            doc = json.load(f)
        assert "backup_id" in doc
        assert "workflow_id" in doc
        assert "created_at" in doc
        assert "state" in doc

    def test_save_stores_correct_workflow_id(self, manager):
        result = manager.save(_minimal_state("wf_id_check"))
        with open(result, encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["workflow_id"] == "wf_id_check"

    def test_save_stores_node_name(self, manager):
        result = manager.save(_minimal_state("wf_node_001"), node_name="verification")
        with open(result, encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["node_name"] == "verification"

    def test_save_returns_none_when_workflow_id_missing(self, manager):
        state = _minimal_state()
        del state["workflow_id"]
        result = manager.save(state)
        assert result is None

    def test_save_multiple_creates_multiple_files(self, manager, tmp_path):
        for i in range(3):
            manager.save(_minimal_state("wf_multi_001"))
        files = list(tmp_path.glob("wf_multi_001_*.json"))
        assert len(files) == 3

    def test_save_handles_io_error_gracefully(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path))
        state = _minimal_state("wf_io_err")
        with patch("builtins.open", side_effect=OSError("disk full")):
            result = mgr.save(state)
        assert result is None


# ---------------------------------------------------------------------------
# restore() — deserialize from backup
# ---------------------------------------------------------------------------

class TestRestore:
    def test_restore_returns_dict(self, manager):
        path = manager.save(_minimal_state("wf_restore_001"))
        result = manager.restore(path)
        assert isinstance(result, dict)

    def test_restore_preserves_workflow_id(self, manager):
        path = manager.save(_minimal_state("wf_restore_002"))
        result = manager.restore(path)
        assert result["workflow_id"] == "wf_restore_002"

    def test_restore_preserves_objection_text(self, manager):
        state = _minimal_state("wf_restore_003")
        state["objection_text"] = "Sản phẩm có bảo hành không?"
        path = manager.save(state)
        result = manager.restore(path)
        assert result["objection_text"] == "Sản phẩm có bảo hành không?"

    def test_restore_preserves_workflow_status(self, manager):
        state = _minimal_state("wf_restore_004")
        state["workflow_status"] = "verifying"
        path = manager.save(state)
        result = manager.restore(path)
        assert result["workflow_status"] == "verifying"

    def test_restore_returns_none_for_missing_file(self, manager):
        result = manager.restore("/nonexistent/path/backup.json")
        assert result is None

    def test_restore_returns_none_for_corrupt_file(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path))
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("not valid json", encoding="utf-8")
        result = mgr.restore(str(corrupt))
        assert result is None

    def test_roundtrip_serialization(self, manager):
        """Serialize → deserialize → same state values."""
        state = _minimal_state("wf_roundtrip_001")
        state["workflow_status"] = "correcting"
        state["retry_count"] = 2
        state["draft_response"] = "Test draft response for roundtrip"

        path = manager.save(state)
        restored = manager.restore(path)

        assert restored["workflow_id"] == state["workflow_id"]
        assert restored["workflow_status"] == state["workflow_status"]
        assert restored["retry_count"] == state["retry_count"]
        assert restored["draft_response"] == state["draft_response"]
        assert restored["objection_text"] == state["objection_text"]


# ---------------------------------------------------------------------------
# restore_latest()
# ---------------------------------------------------------------------------

class TestRestoreLatest:
    def test_restore_latest_returns_none_when_no_backups(self, manager):
        result = manager.restore_latest("wf_no_backups")
        assert result is None

    def test_restore_latest_returns_most_recent(self, manager):
        state = _minimal_state("wf_latest_001")
        manager.save(state)

        state["workflow_status"] = "verifying"
        manager.save(state)

        state["workflow_status"] = "correcting"
        manager.save(state)

        result = manager.restore_latest("wf_latest_001")
        assert result is not None
        assert result["workflow_status"] == "correcting"


# ---------------------------------------------------------------------------
# list_backups()
# ---------------------------------------------------------------------------

class TestListBackups:
    def test_list_empty_when_no_backups(self, manager):
        assert manager.list_backups("wf_empty") == []

    def test_list_returns_entries_after_save(self, manager):
        manager.save(_minimal_state("wf_list_001"))
        backups = manager.list_backups("wf_list_001")
        assert len(backups) == 1

    def test_list_entry_has_required_keys(self, manager):
        manager.save(_minimal_state("wf_list_keys"))
        entry = manager.list_backups("wf_list_keys")[0]
        assert "backup_id" in entry
        assert "workflow_id" in entry
        assert "node_name" in entry
        assert "created_at" in entry
        assert "path" in entry

    def test_list_sorted_newest_first(self, manager):
        state = _minimal_state("wf_sort_001")
        for i in range(3):
            state["workflow_status"] = ["initialized", "verifying", "correcting"][i]
            manager.save(state)

        backups = manager.list_backups("wf_sort_001")
        assert len(backups) == 3
        # Verify descending order by created_at
        dates = [b["created_at"] for b in backups]
        assert dates == sorted(dates, reverse=True)

    def test_list_without_filter_returns_all(self, manager):
        manager.save(_minimal_state("wf_all_001"))
        manager.save(_minimal_state("wf_all_002"))
        all_backups = manager.list_backups()
        assert len(all_backups) >= 2

    def test_list_with_filter_returns_only_matching(self, manager):
        manager.save(_minimal_state("wf_filter_001"))
        manager.save(_minimal_state("wf_filter_002"))
        backups = manager.list_backups("wf_filter_001")
        assert all(b["workflow_id"] == "wf_filter_001" for b in backups)

    def test_list_skips_unreadable_files(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path), max_backups=0)
        (tmp_path / "bad_file.json").write_text("garbage", encoding="utf-8")
        mgr.save(_minimal_state("wf_skip_001"))
        backups = mgr.list_backups("wf_skip_001")
        assert len(backups) == 1


# ---------------------------------------------------------------------------
# delete_backup() and delete_all_backups()
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_backup_removes_file(self, manager, tmp_path):
        path = manager.save(_minimal_state("wf_del_001"))
        assert Path(path).exists()
        result = manager.delete_backup(path)
        assert result is True
        assert not Path(path).exists()

    def test_delete_backup_returns_false_for_missing(self, manager):
        result = manager.delete_backup("/nonexistent/backup.json")
        assert result is False

    def test_delete_all_backups_removes_all(self, manager):
        for _ in range(3):
            manager.save(_minimal_state("wf_del_all_001"))
        count = manager.delete_all_backups("wf_del_all_001")
        assert count == 3
        assert manager.list_backups("wf_del_all_001") == []

    def test_delete_all_backups_returns_zero_when_none(self, manager):
        count = manager.delete_all_backups("wf_nonexistent")
        assert count == 0


# ---------------------------------------------------------------------------
# Retention / cleanup logic
# ---------------------------------------------------------------------------

class TestRetention:
    def test_retention_keeps_max_backups(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path), max_backups=3)
        state = _minimal_state("wf_retention_001")
        for _ in range(5):
            mgr.save(state)
        backups = mgr.list_backups("wf_retention_001")
        assert len(backups) == 3

    def test_retention_keeps_newest_backups(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path), max_backups=2)
        state = _minimal_state("wf_retention_002")
        for i in range(4):
            state["retry_count"] = i
            mgr.save(state)

        backups = mgr.list_backups("wf_retention_002")
        assert len(backups) == 2
        # The two remaining should be the most recent (highest retry_count)
        for backup in backups:
            restored = mgr.restore(backup["path"])
            assert restored["retry_count"] in (2, 3)

    def test_no_retention_when_max_backups_zero(self, manager_no_retention):
        state = _minimal_state("wf_no_ret_001")
        for _ in range(8):
            manager_no_retention.save(state)
        backups = manager_no_retention.list_backups("wf_no_ret_001")
        assert len(backups) == 8

    def test_retention_does_not_affect_other_workflows(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path), max_backups=2)
        for _ in range(4):
            mgr.save(_minimal_state("wf_ret_a"))
        for _ in range(3):
            mgr.save(_minimal_state("wf_ret_b"))

        assert len(mgr.list_backups("wf_ret_a")) == 2
        assert len(mgr.list_backups("wf_ret_b")) == 2


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------

class TestAsyncSave:
    def test_async_save_creates_file(self, manager, tmp_path):
        state = _minimal_state("wf_async_001")
        result = asyncio.get_event_loop().run_until_complete(
            manager.async_save(state, node_name="research")
        )
        assert result is not None
        assert Path(result).exists()

    def test_async_save_returns_none_when_workflow_id_missing(self, manager):
        state = _minimal_state()
        del state["workflow_id"]
        result = asyncio.get_event_loop().run_until_complete(
            manager.async_save(state)
        )
        assert result is None

    def test_async_save_stores_node_name(self, manager):
        state = _minimal_state("wf_async_node")
        path = asyncio.get_event_loop().run_until_complete(
            manager.async_save(state, node_name="verification")
        )
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["node_name"] == "verification"


class TestAsyncRestore:
    def test_async_restore_returns_dict(self, manager):
        path = manager.save(_minimal_state("wf_async_restore_001"))
        result = asyncio.get_event_loop().run_until_complete(
            manager.async_restore(path)
        )
        assert isinstance(result, dict)

    def test_async_restore_preserves_workflow_id(self, manager):
        path = manager.save(_minimal_state("wf_async_restore_002"))
        result = asyncio.get_event_loop().run_until_complete(
            manager.async_restore(path)
        )
        assert result["workflow_id"] == "wf_async_restore_002"

    def test_async_restore_returns_none_for_missing_file(self, manager):
        result = asyncio.get_event_loop().run_until_complete(
            manager.async_restore("/nonexistent/backup.json")
        )
        assert result is None

    def test_async_restore_returns_none_for_corrupt_file(self, tmp_path):
        mgr = BackupManager(backup_dir=str(tmp_path))
        corrupt = tmp_path / "corrupt_async.json"
        corrupt.write_text("not valid json", encoding="utf-8")
        result = asyncio.get_event_loop().run_until_complete(
            mgr.async_restore(str(corrupt))
        )
        assert result is None

    def test_async_restore_latest_returns_none_when_no_backups(self, manager):
        result = asyncio.get_event_loop().run_until_complete(
            manager.async_restore_latest("wf_no_async_backups")
        )
        assert result is None

    def test_async_roundtrip(self, manager):
        """Async save → async restore → same state."""
        state = _minimal_state("wf_async_roundtrip")
        state["workflow_status"] = "verifying"
        state["retry_count"] = 1

        loop = asyncio.get_event_loop()
        path = loop.run_until_complete(manager.async_save(state))
        restored = loop.run_until_complete(manager.async_restore(path))

        assert restored["workflow_id"] == state["workflow_id"]
        assert restored["workflow_status"] == state["workflow_status"]
        assert restored["retry_count"] == state["retry_count"]
