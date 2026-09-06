"""
Distributed Tracing cho Async Verification Operations

Self-contained distributed tracing system (no external dependencies):
- Trace spans for each async verification operation
- Context propagation across async boundaries via contextvars
- Nested spans: workflow → node → checker
- Integration with existing correlation ID system
- Structured JSON export compatible with execution log format

Requirements: 7.1 (execution history), 7.3 (structured JSON log), 9.3 (parallel checks)
"""

import asyncio
import contextvars
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ContextVar propagates automatically into child asyncio tasks
_active_span_var: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "_active_span_var", default=None
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """A single trace span representing one unit of async work."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: str  # ISO format
    end_time: Optional[str] = None
    status: str = "in_progress"  # in_progress | ok | error
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Child spans recorded inline for nested tracing
    children: List["Span"] = field(default_factory=list)

    # ---- helpers ----

    def finish(self, status: str = "ok", **metadata_updates: Any) -> None:
        """Mark span as finished."""
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.status = status
        self.metadata.update(metadata_updates)

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds, or None if not finished."""
        if self.end_time is None:
            return None
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        return (end - start).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict (execution log format)."""
        d = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class Trace:
    """A complete trace (collection of spans for one workflow execution)."""

    trace_id: str
    correlation_id: str
    root_span: Optional[Span] = None
    # Flat index for fast lookup by span_id
    _spans: Dict[str, Span] = field(default_factory=dict)

    def add_span(self, span: Span) -> None:
        self._spans[span.span_id] = span

    def get_span(self, span_id: str) -> Optional[Span]:
        return self._spans.get(span_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "root_span": self.root_span.to_dict() if self.root_span else None,
            "total_spans": len(self._spans),
        }


# ---------------------------------------------------------------------------
# DistributedTracer
# ---------------------------------------------------------------------------

class DistributedTracer:
    """
    Self-contained distributed tracer for async verification operations.

    Usage
    -----
    tracer = DistributedTracer()

    # Start a root workflow span
    async with tracer.start_workflow_span("wf_123", "corr_abc") as root:
        # Start a node span (child of root)
        async with tracer.start_node_span("verification") as node_span:
            # Start checker spans (children of node)
            async with tracer.start_checker_span("price_check") as _:
                ...

    # Export trace as JSON
    trace_data = tracer.export_trace("wf_123")
    """

    def __init__(self) -> None:
        # workflow_id → Trace
        self._traces: Dict[str, Trace] = {}

    # ------------------------------------------------------------------
    # Span factory helpers
    # ------------------------------------------------------------------

    def _new_span(
        self,
        operation_name: str,
        trace_id: str,
        parent_span: Optional[Span] = None,
        **metadata: Any,
    ) -> Span:
        span = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span.span_id if parent_span else None,
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc).isoformat(),
            metadata=dict(metadata),
        )
        if parent_span is not None:
            parent_span.children.append(span)
        return span

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def start_workflow_span(
        self,
        workflow_id: str,
        correlation_id: str,
        **metadata: Any,
    ):
        """
        Start a root workflow span and register a new Trace.

        Sets the active span in the async context so child spans can
        automatically discover their parent.
        """
        trace_id = uuid.uuid4().hex
        trace = Trace(trace_id=trace_id, correlation_id=correlation_id)
        self._traces[workflow_id] = trace

        span = self._new_span(
            "workflow",
            trace_id=trace_id,
            parent_span=None,
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            **metadata,
        )
        trace.root_span = span
        trace.add_span(span)

        token = _active_span_var.set(span)
        try:
            yield span
            span.finish(status="ok")
        except Exception as exc:
            span.finish(status="error", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            _active_span_var.reset(token)

    @asynccontextmanager
    async def start_node_span(self, node_name: str, **metadata: Any):
        """
        Start a node-level span (child of the current active span).

        Propagates automatically across async boundaries because
        contextvars are inherited by child tasks.
        """
        parent = _active_span_var.get()
        trace_id = parent.trace_id if parent else uuid.uuid4().hex

        span = self._new_span(
            f"node:{node_name}",
            trace_id=trace_id,
            parent_span=parent,
            node_name=node_name,
            **metadata,
        )

        # Register span in the trace if we can find it
        for trace in self._traces.values():
            if trace.trace_id == trace_id:
                trace.add_span(span)
                break

        token = _active_span_var.set(span)
        try:
            yield span
            span.finish(status="ok")
        except Exception as exc:
            span.finish(status="error", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            _active_span_var.reset(token)

    @asynccontextmanager
    async def start_checker_span(self, checker_name: str, **metadata: Any):
        """
        Start a checker-level span (child of the current node span).

        Designed for price_check, policy_check, relevance_check.
        """
        parent = _active_span_var.get()
        trace_id = parent.trace_id if parent else uuid.uuid4().hex

        span = self._new_span(
            f"checker:{checker_name}",
            trace_id=trace_id,
            parent_span=parent,
            checker_name=checker_name,
            **metadata,
        )

        for trace in self._traces.values():
            if trace.trace_id == trace_id:
                trace.add_span(span)
                break

        token = _active_span_var.set(span)
        try:
            yield span
            span.finish(status="ok")
        except Exception as exc:
            span.finish(status="error", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            _active_span_var.reset(token)

    # ------------------------------------------------------------------
    # Sync context manager (for non-async nodes)
    # ------------------------------------------------------------------

    @contextmanager
    def start_span_sync(self, operation_name: str, **metadata: Any):
        """
        Sync context manager for non-async code paths.

        Reads the current active span from the contextvar (works in
        threads that share the same context copy).
        """
        parent = _active_span_var.get()
        trace_id = parent.trace_id if parent else uuid.uuid4().hex

        span = self._new_span(
            operation_name,
            trace_id=trace_id,
            parent_span=parent,
            **metadata,
        )

        for trace in self._traces.values():
            if trace.trace_id == trace_id:
                trace.add_span(span)
                break

        token = _active_span_var.set(span)
        try:
            yield span
            span.finish(status="ok")
        except Exception as exc:
            span.finish(status="error", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            _active_span_var.reset(token)

    # ------------------------------------------------------------------
    # Context propagation helpers
    # ------------------------------------------------------------------

    def get_current_span(self) -> Optional[Span]:
        """Return the currently active span in this async context."""
        return _active_span_var.get()

    def inject_context(self) -> Dict[str, Optional[str]]:
        """
        Extract trace context for propagation across service boundaries.

        Returns a dict with trace_id, span_id, parent_span_id that can
        be serialised into headers or metadata.
        """
        span = _active_span_var.get()
        if span is None:
            return {"trace_id": None, "span_id": None, "parent_span_id": None}
        return {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
        }

    def extract_context(self, carrier: Dict[str, Optional[str]]) -> Optional[Span]:
        """
        Reconstruct a parent span reference from a propagated carrier dict.

        Returns a lightweight Span stub (no children list) that can be
        used as parent_span when creating new spans in a remote context.
        """
        trace_id = carrier.get("trace_id")
        span_id = carrier.get("span_id")
        if not trace_id or not span_id:
            return None
        # Look up in known traces
        for trace in self._traces.values():
            found = trace.get_span(span_id)
            if found:
                return found
        return None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_trace(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Export trace data as structured JSON (execution log compatible).

        Returns None if no trace exists for the given workflow_id.
        """
        trace = self._traces.get(workflow_id)
        if trace is None:
            return None
        return trace.to_dict()

    def export_all_traces(self) -> List[Dict[str, Any]]:
        """Export all traces as a list of dicts."""
        return [t.to_dict() for t in self._traces.values()]

    def clear_trace(self, workflow_id: str) -> None:
        """Remove a completed trace to free memory."""
        self._traces.pop(workflow_id, None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_tracer: Optional[DistributedTracer] = None


def get_tracer() -> DistributedTracer:
    """Return the global DistributedTracer singleton."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DistributedTracer()
    return _global_tracer


def reset_tracer() -> None:
    """Reset the global tracer (useful in tests)."""
    global _global_tracer
    _global_tracer = None
