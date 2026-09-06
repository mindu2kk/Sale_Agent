"""
Tests for distributed_tracing module (Task 6.1.4)

Validates: Requirements 7.1 (execution history), 7.3 (structured JSON log), 9.3 (parallel checks)
"""

import asyncio
import pytest

from backend.verification.utils.distributed_tracing import (
    DistributedTracer,
    Span,
    get_tracer,
    reset_tracer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_tracer():
    """Reset global tracer before each test."""
    reset_tracer()
    yield
    reset_tracer()


@pytest.fixture
def tracer():
    return DistributedTracer()


# ---------------------------------------------------------------------------
# Span creation & lifecycle
# ---------------------------------------------------------------------------

class TestSpan:
    def test_span_initial_state(self):
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            operation_name="test_op",
            start_time="2024-01-01T00:00:00+00:00",
        )
        assert span.status == "in_progress"
        assert span.end_time is None
        assert span.duration_ms is None

    def test_span_finish_ok(self):
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            operation_name="test_op",
            start_time="2024-01-01T00:00:00+00:00",
        )
        span.finish(status="ok", result="done")
        assert span.status == "ok"
        assert span.end_time is not None
        assert span.metadata["result"] == "done"

    def test_span_duration_ms(self):
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            operation_name="test_op",
            start_time="2024-01-01T00:00:00+00:00",
        )
        span.finish()
        # duration should be non-negative
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_span_to_dict(self):
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            operation_name="test_op",
            start_time="2024-01-01T00:00:00+00:00",
        )
        span.finish()
        d = span.to_dict()
        assert d["trace_id"] == "t1"
        assert d["span_id"] == "s1"
        assert d["operation_name"] == "test_op"
        assert "duration_ms" in d


# ---------------------------------------------------------------------------
# Workflow span (async)
# ---------------------------------------------------------------------------

class TestWorkflowSpan:
    @pytest.mark.asyncio
    async def test_workflow_span_creates_trace(self, tracer):
        async with tracer.start_workflow_span("wf_001", "corr_abc") as span:
            assert span.operation_name == "workflow"
            assert span.trace_id is not None
            assert span.parent_span_id is None

        trace = tracer.export_trace("wf_001")
        assert trace is not None
        assert trace["correlation_id"] == "corr_abc"

    @pytest.mark.asyncio
    async def test_workflow_span_finishes_ok(self, tracer):
        async with tracer.start_workflow_span("wf_002", "corr_xyz") as span:
            pass
        assert span.status == "ok"
        assert span.end_time is not None

    @pytest.mark.asyncio
    async def test_workflow_span_records_error(self, tracer):
        with pytest.raises(ValueError):
            async with tracer.start_workflow_span("wf_003", "corr_err") as span:
                raise ValueError("boom")
        assert span.status == "error"
        assert "boom" in span.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_workflow_span_metadata(self, tracer):
        async with tracer.start_workflow_span("wf_004", "corr_meta", env="test") as span:
            pass
        assert span.metadata["env"] == "test"
        assert span.metadata["workflow_id"] == "wf_004"


# ---------------------------------------------------------------------------
# Node span (nested under workflow)
# ---------------------------------------------------------------------------

class TestNodeSpan:
    @pytest.mark.asyncio
    async def test_node_span_is_child_of_workflow(self, tracer):
        async with tracer.start_workflow_span("wf_010", "corr_n") as wf_span:
            async with tracer.start_node_span("verification") as node_span:
                assert node_span.parent_span_id == wf_span.span_id
                assert node_span.trace_id == wf_span.trace_id

    @pytest.mark.asyncio
    async def test_node_span_added_to_children(self, tracer):
        async with tracer.start_workflow_span("wf_011", "corr_c") as wf_span:
            async with tracer.start_node_span("research"):
                pass
        assert len(wf_span.children) == 1
        assert wf_span.children[0].operation_name == "node:research"

    @pytest.mark.asyncio
    async def test_multiple_node_spans(self, tracer):
        async with tracer.start_workflow_span("wf_012", "corr_m") as wf_span:
            async with tracer.start_node_span("research"):
                pass
            async with tracer.start_node_span("verification"):
                pass
        assert len(wf_span.children) == 2


# ---------------------------------------------------------------------------
# Checker span (nested under node)
# ---------------------------------------------------------------------------

class TestCheckerSpan:
    @pytest.mark.asyncio
    async def test_checker_span_is_child_of_node(self, tracer):
        async with tracer.start_workflow_span("wf_020", "corr_ch"):
            async with tracer.start_node_span("verification") as node_span:
                async with tracer.start_checker_span("price_check") as checker_span:
                    assert checker_span.parent_span_id == node_span.span_id

    @pytest.mark.asyncio
    async def test_parallel_checker_spans(self, tracer):
        """Validates: Requirements 9.3 - parallel checks each get their own span."""
        async with tracer.start_workflow_span("wf_021", "corr_par"):
            async with tracer.start_node_span("verification") as node_span:
                async def run_price():
                    async with tracer.start_checker_span("price_check"):
                        await asyncio.sleep(0)

                async def run_policy():
                    async with tracer.start_checker_span("policy_check"):
                        await asyncio.sleep(0)

                async def run_relevance():
                    async with tracer.start_checker_span("relevance_check"):
                        await asyncio.sleep(0)

                await asyncio.gather(run_price(), run_policy(), run_relevance())

        # All 3 checker spans should be children of the node span
        checker_names = {c.operation_name for c in node_span.children}
        assert "checker:price_check" in checker_names
        assert "checker:policy_check" in checker_names
        assert "checker:relevance_check" in checker_names


# ---------------------------------------------------------------------------
# Sync span
# ---------------------------------------------------------------------------

class TestSyncSpan:
    def test_sync_span_standalone(self, tracer):
        with tracer.start_span_sync("sync_op", key="val") as span:
            assert span.operation_name == "sync_op"
            assert span.metadata["key"] == "val"
        assert span.status == "ok"

    def test_sync_span_error(self, tracer):
        with pytest.raises(RuntimeError):
            with tracer.start_span_sync("sync_err") as span:
                raise RuntimeError("sync fail")
        assert span.status == "error"


# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

class TestContextPropagation:
    @pytest.mark.asyncio
    async def test_inject_extract_context(self, tracer):
        async with tracer.start_workflow_span("wf_030", "corr_prop") as wf_span:
            carrier = tracer.inject_context()
            assert carrier["trace_id"] == wf_span.trace_id
            assert carrier["span_id"] == wf_span.span_id
            assert carrier["parent_span_id"] is None

    @pytest.mark.asyncio
    async def test_get_current_span(self, tracer):
        async with tracer.start_workflow_span("wf_031", "corr_cur") as wf_span:
            current = tracer.get_current_span()
            assert current is wf_span

    @pytest.mark.asyncio
    async def test_context_propagates_to_child_task(self, tracer):
        """contextvars propagate into child asyncio tasks."""
        captured = {}

        async def child_task():
            captured["span"] = tracer.get_current_span()

        async with tracer.start_workflow_span("wf_032", "corr_task") as wf_span:
            await asyncio.create_task(child_task())

        # Child task should have inherited the workflow span
        assert captured["span"] is wf_span


# ---------------------------------------------------------------------------
# Export / JSON compatibility (Req 7.3)
# ---------------------------------------------------------------------------

class TestExport:
    @pytest.mark.asyncio
    async def test_export_trace_structure(self, tracer):
        """Validates: Requirements 7.3 - structured JSON execution log."""
        async with tracer.start_workflow_span("wf_040", "corr_exp") as wf_span:
            async with tracer.start_node_span("verification"):
                async with tracer.start_checker_span("price_check"):
                    pass

        data = tracer.export_trace("wf_040")
        assert data is not None
        assert "trace_id" in data
        assert "correlation_id" in data
        assert data["correlation_id"] == "corr_exp"
        assert "root_span" in data
        assert data["root_span"]["operation_name"] == "workflow"

    @pytest.mark.asyncio
    async def test_export_all_traces(self, tracer):
        async with tracer.start_workflow_span("wf_041", "c1"):
            pass
        async with tracer.start_workflow_span("wf_042", "c2"):
            pass
        all_traces = tracer.export_all_traces()
        assert len(all_traces) == 2

    @pytest.mark.asyncio
    async def test_export_none_for_unknown_workflow(self, tracer):
        result = tracer.export_trace("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_trace(self, tracer):
        async with tracer.start_workflow_span("wf_043", "c3"):
            pass
        tracer.clear_trace("wf_043")
        assert tracer.export_trace("wf_043") is None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

class TestGlobalTracer:
    def test_get_tracer_returns_singleton(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_reset_tracer_creates_new_instance(self):
        t1 = get_tracer()
        reset_tracer()
        t2 = get_tracer()
        assert t1 is not t2
