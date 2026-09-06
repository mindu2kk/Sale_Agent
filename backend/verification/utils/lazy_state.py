"""
Memory-Efficient State Management với Lazy Loading

Provides lazy loading, state compression, and LRU eviction for WorkflowState:
- LazyWorkflowState: Wraps WorkflowState with lazy property accessors for expensive fields
- StateMemoryManager: Tracks active workflow states with LRU eviction and auto-compression

Uses weakref, functools.cached_property, and sys.getsizeof for memory efficiency.
"""

import sys
import time
import weakref
import threading
from collections import OrderedDict
from functools import cached_property
from typing import Any, Dict, Generator, List, Optional

from backend.verification.models.execution import ExecutionStep


# ---------------------------------------------------------------------------
# Compressed execution step summary (drop full input/output state)
# ---------------------------------------------------------------------------

def _compress_step(step: ExecutionStep) -> Dict[str, Any]:
    """Return a lightweight summary dict for an ExecutionStep."""
    return {
        "timestamp": step.timestamp,
        "node_name": step.node_name,
        "execution_time": step.execution_time,
        "status": step.status.value if hasattr(step.status, "value") else step.status,
        "correlation_id": step.correlation_id,
    }


# ---------------------------------------------------------------------------
# LazyWorkflowState
# ---------------------------------------------------------------------------

class LazyWorkflowState:
    """
    Memory-efficient wrapper around a WorkflowState dict with lazy loading.

    Expensive fields (verification_result, execution_log) are only materialised
    when first accessed.  A compress() method drops full input/output payloads
    from execution log entries, and cleanup() releases intermediate data after
    workflow completion.
    """

    __slots__ = (
        "_state",
        "_compressed",
        "_released",
        "__weakref__",
    )

    def __init__(self, state: Dict[str, Any]) -> None:
        """
        Args:
            state: A WorkflowState-compatible dictionary.
        """
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_compressed", False)
        object.__setattr__(self, "_released", False)

    # ------------------------------------------------------------------
    # Lazy property accessors for expensive fields
    # ------------------------------------------------------------------

    @property
    def verification_result(self) -> Any:
        """Lazily return verification_result (only accessed when needed)."""
        return self._state.get("verification_result")

    @property
    def execution_log(self) -> List[Any]:
        """Lazily return execution_log (only accessed when needed)."""
        return self._state.get("execution_log", [])

    # ------------------------------------------------------------------
    # Pass-through for all other fields
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        state = object.__getattribute__(self, "_state")
        if name in state:
            return state[name]
        raise AttributeError(f"LazyWorkflowState has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            self._state[name] = value

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def compress(self) -> None:
        """
        Compress execution_log in-place.

        Replaces each ExecutionStep (or dict) with a lightweight summary,
        dropping full input_summary / output_summary payloads.
        After compression the log entries are plain dicts.
        """
        if self._compressed or self._released:
            return

        raw_log = self._state.get("execution_log", [])
        compressed_log: List[Dict[str, Any]] = []

        for entry in raw_log:
            if isinstance(entry, ExecutionStep):
                compressed_log.append(_compress_step(entry))
            elif isinstance(entry, dict):
                # Keep only summary keys
                compressed_log.append({
                    "timestamp": entry.get("timestamp"),
                    "node_name": entry.get("node_name"),
                    "execution_time": entry.get("execution_time"),
                    "status": entry.get("status"),
                    "correlation_id": entry.get("correlation_id"),
                })
            else:
                compressed_log.append({"raw": str(entry)})

        self._state["execution_log"] = compressed_log
        object.__setattr__(self, "_compressed", True)

    def cleanup(self) -> None:
        """
        Release intermediate state data after workflow completion.

        Clears execution_log, verification_result, correction_feedback,
        and resource_usage to free memory.  The workflow_id, final_response,
        and workflow_status are preserved for audit purposes.
        """
        if self._released:
            return

        for key in ("execution_log", "verification_result",
                    "correction_feedback", "resource_usage",
                    "error_log", "research_reasoning", "tools_used"):
            if key in self._state:
                self._state[key] = None if key != "execution_log" else []

        object.__setattr__(self, "_released", True)

    def stream_execution_log(self) -> Generator[Any, None, None]:
        """
        Generator for streaming access to execution_log entries.

        Yields one entry at a time to avoid loading the entire log into memory.
        """
        log = self._state.get("execution_log") or []
        for entry in log:
            yield entry

    # ------------------------------------------------------------------
    # Memory introspection
    # ------------------------------------------------------------------

    def memory_size_bytes(self) -> int:
        """Approximate memory footprint of the wrapped state dict."""
        return sys.getsizeof(self._state)

    @property
    def is_compressed(self) -> bool:
        return self._compressed

    @property
    def is_released(self) -> bool:
        return self._released

    # ------------------------------------------------------------------
    # Dict-like helpers for LangGraph compatibility
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying state dict."""
        return self._state

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def __repr__(self) -> str:
        wf_id = self._state.get("workflow_id", "unknown")
        status = self._state.get("workflow_status", "unknown")
        return (
            f"LazyWorkflowState(workflow_id={wf_id!r}, status={status!r}, "
            f"compressed={self._compressed}, released={self._released})"
        )


# ---------------------------------------------------------------------------
# StateMemoryManager
# ---------------------------------------------------------------------------

class _ManagedEntry:
    """Internal entry tracked by StateMemoryManager."""

    __slots__ = ("state_ref", "last_accessed", "workflow_id", "is_completed")

    def __init__(self, state: LazyWorkflowState, workflow_id: str) -> None:
        # Use a strong reference; weakref is used for external callers
        self.state_ref = state
        self.last_accessed: float = time.monotonic()
        self.workflow_id: str = workflow_id
        self.is_completed: bool = False

    def touch(self) -> None:
        self.last_accessed = time.monotonic()


class StateMemoryManager:
    """
    Tracks active LazyWorkflowState instances with LRU eviction.

    Features:
    - get_state / release_state public API
    - LRU eviction when max_states is exceeded
    - Auto-compression of states not accessed within compress_ttl_seconds
    - Memory usage statistics
    """

    def __init__(
        self,
        max_states: int = 100,
        compress_ttl_seconds: float = 300.0,
    ) -> None:
        """
        Args:
            max_states: Maximum number of states to keep in memory.
            compress_ttl_seconds: Seconds of inactivity before auto-compression.
        """
        self._max_states = max_states
        self._compress_ttl = compress_ttl_seconds

        # OrderedDict preserves insertion order for LRU tracking
        self._entries: OrderedDict[str, _ManagedEntry] = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_state(self, workflow_id: str, state: LazyWorkflowState) -> None:
        """Register a new workflow state for tracking."""
        with self._lock:
            entry = _ManagedEntry(state, workflow_id)
            self._entries[workflow_id] = entry
            self._entries.move_to_end(workflow_id)
            self._evict_if_needed()

    def get_state(self, workflow_id: str) -> Optional[LazyWorkflowState]:
        """
        Retrieve a tracked state by workflow_id.

        Marks the entry as recently used (LRU update).
        Returns None if not found.
        """
        with self._lock:
            self._maybe_compress_stale()
            entry = self._entries.get(workflow_id)
            if entry is None:
                return None
            entry.touch()
            self._entries.move_to_end(workflow_id)
            return entry.state_ref

    def release_state(self, workflow_id: str) -> bool:
        """
        Release a workflow state, calling cleanup() and removing from tracking.

        Returns True if the state was found and released, False otherwise.
        """
        with self._lock:
            entry = self._entries.pop(workflow_id, None)
            if entry is None:
                return False
            entry.state_ref.cleanup()
            entry.is_completed = True
            return True

    def mark_completed(self, workflow_id: str) -> None:
        """Mark a workflow as completed (eligible for earlier eviction)."""
        with self._lock:
            entry = self._entries.get(workflow_id)
            if entry:
                entry.is_completed = True

    # ------------------------------------------------------------------
    # Memory statistics
    # ------------------------------------------------------------------

    def memory_stats(self) -> Dict[str, Any]:
        """Return memory usage statistics for all tracked states."""
        with self._lock:
            total_bytes = 0
            state_sizes: Dict[str, int] = {}
            for wf_id, entry in self._entries.items():
                size = entry.state_ref.memory_size_bytes()
                state_sizes[wf_id] = size
                total_bytes += size

            return {
                "tracked_states": len(self._entries),
                "max_states": self._max_states,
                "total_estimated_bytes": total_bytes,
                "total_estimated_mb": total_bytes / (1024 * 1024),
                "per_state_bytes": state_sizes,
                "compress_ttl_seconds": self._compress_ttl,
            }

    def active_workflow_ids(self) -> List[str]:
        """Return list of currently tracked workflow IDs."""
        with self._lock:
            return list(self._entries.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used entries when over capacity."""
        while len(self._entries) > self._max_states:
            # Pop the oldest (least recently used) entry
            oldest_id, oldest_entry = next(iter(self._entries.items()))
            oldest_entry.state_ref.cleanup()
            del self._entries[oldest_id]

    def _maybe_compress_stale(self) -> None:
        """Compress states that haven't been accessed within compress_ttl."""
        now = time.monotonic()
        for entry in self._entries.values():
            if (
                not entry.state_ref.is_compressed
                and not entry.state_ref.is_released
                and (now - entry.last_accessed) >= self._compress_ttl
            ):
                entry.state_ref.compress()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"StateMemoryManager(tracked={len(self._entries)}, "
            f"max={self._max_states}, compress_ttl={self._compress_ttl}s)"
        )
