"""
Tests for BatchProcessor - Multiple Verification Requests

Covers:
- Basic batch processing (success, failure, mixed)
- Priority ordering (CRITICAL → HIGH → NORMAL)
- Concurrency limiting via semaphore
- Batch size chunking
- Metrics computation (success_rate, avg_latency, wall_time)
- Edge cases: empty batch, single request, all failures

Supports Requirement 9: Performance & Scalability (Req 9.2, 9.3)
"""

import asyncio
import random

import pytest

from backend.verification.utils.batch_processor import (
    BatchMetrics,
    BatchProcessor,
    BatchResult,
    RequestPriority,
    VerificationRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_request(request_id="req-1", priority=RequestPriority.NORMAL):
    return VerificationRequest(
        objection_text="Sản phẩm có bảo hành không?",
        draft_response="Sản phẩm được bảo hành 12 tháng.",
        request_id=request_id,
        priority=priority,
    )


async def _ok_verify(req):
    return {"approved": True, "request_id": req.request_id}


async def _fail_verify(req):
    raise ValueError(f"Simulated failure for {req.request_id}")


def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# VerificationRequest model
# ---------------------------------------------------------------------------

class TestVerificationRequest:
    def test_default_priority_is_normal(self):
        req = VerificationRequest(objection_text="test", draft_response="draft")
        assert req.priority == RequestPriority.NORMAL

    def test_priority_value_ordering(self):
        critical = make_request(priority=RequestPriority.CRITICAL)
        high = make_request(priority=RequestPriority.HIGH)
        normal = make_request(priority=RequestPriority.NORMAL)
        assert critical.priority_value() < high.priority_value() < normal.priority_value()

    def test_metadata_defaults_to_empty_dict(self):
        req = VerificationRequest(objection_text="x", draft_response="y")
        assert req.metadata == {}

    def test_request_id_optional(self):
        req = VerificationRequest(objection_text="x", draft_response="y")
        assert req.request_id is None


# ---------------------------------------------------------------------------
# BatchMetrics
# ---------------------------------------------------------------------------

class TestBatchMetrics:
    def test_success_rate_zero_when_no_processed(self):
        m = BatchMetrics()
        assert m.success_rate == 0.0

    def test_success_rate_calculation(self):
        m = BatchMetrics(total_processed=4, success_count=3, failure_count=1)
        assert m.success_rate == pytest.approx(0.75)

    def test_to_dict_contains_all_keys(self):
        m = BatchMetrics(
            total_requests=2,
            total_processed=2,
            success_count=2,
            failure_count=0,
            avg_latency_seconds=0.1,
            total_latency_seconds=0.2,
            batch_wall_time_seconds=0.15,
        )
        d = m.to_dict()
        expected = {
            "total_requests", "total_processed", "success_count",
            "failure_count", "avg_latency_seconds", "total_latency_seconds",
            "batch_wall_time_seconds", "success_rate",
        }
        assert expected == set(d.keys())


# ---------------------------------------------------------------------------
# BatchProcessor construction
# ---------------------------------------------------------------------------

class TestBatchProcessorConstruction:
    def test_invalid_max_concurrency_raises(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            BatchProcessor(verify_fn=_ok_verify, max_concurrency=0)

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            BatchProcessor(verify_fn=_ok_verify, batch_size=0)

    def test_none_batch_size_allowed(self):
        p = BatchProcessor(verify_fn=_ok_verify, batch_size=None)
        assert p._batch_size is None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

class TestBatchProcessorCore:
    def test_empty_batch_returns_empty_results(self):
        p = BatchProcessor(verify_fn=_ok_verify)
        results, metrics = run(p.process([]))
        assert results == []
        assert metrics.total_requests == 0
        assert metrics.total_processed == 0

    def test_single_request_success(self):
        p = BatchProcessor(verify_fn=_ok_verify)
        results, metrics = run(p.process([make_request("r1")]))

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].request_id == "r1"
        assert results[0].error is None
        assert results[0].latency_seconds >= 0.0
        assert metrics.success_count == 1
        assert metrics.failure_count == 0

    def test_single_request_failure(self):
        p = BatchProcessor(verify_fn=_fail_verify)
        results, metrics = run(p.process([make_request("r1")]))

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].result is None
        assert isinstance(results[0].error, ValueError)
        assert metrics.failure_count == 1
        assert metrics.success_count == 0

    def test_mixed_success_and_failure(self):
        async def mixed_verify(req):
            if req.request_id == "fail":
                raise RuntimeError("fail")
            return {"ok": True}

        requests = [make_request("ok1"), make_request("fail"), make_request("ok2")]
        p = BatchProcessor(verify_fn=mixed_verify)
        _, metrics = run(p.process(requests))

        assert metrics.total_processed == 3
        assert metrics.success_count == 2
        assert metrics.failure_count == 1
        assert metrics.success_rate == pytest.approx(2 / 3)

    def test_all_requests_processed(self):
        requests = [make_request(f"r{i}") for i in range(20)]
        p = BatchProcessor(verify_fn=_ok_verify, max_concurrency=5)
        results, metrics = run(p.process(requests))

        assert len(results) == 20
        assert metrics.total_processed == 20
        assert metrics.total_requests == 20


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_critical_requests_come_first_in_results(self):
        requests = [
            make_request("normal-1", RequestPriority.NORMAL),
            make_request("critical-1", RequestPriority.CRITICAL),
            make_request("high-1", RequestPriority.HIGH),
            make_request("normal-2", RequestPriority.NORMAL),
            make_request("critical-2", RequestPriority.CRITICAL),
        ]
        p = BatchProcessor(verify_fn=_ok_verify)
        results, _ = run(p.process(requests))

        priorities = [r.priority for r in results]
        critical_idx = [i for i, pr in enumerate(priorities) if pr == RequestPriority.CRITICAL]
        high_idx = [i for i, pr in enumerate(priorities) if pr == RequestPriority.HIGH]
        normal_idx = [i for i, pr in enumerate(priorities) if pr == RequestPriority.NORMAL]

        assert max(critical_idx) < min(high_idx)
        assert max(high_idx) < min(normal_idx)

    def test_same_priority_preserves_relative_order(self):
        requests = [make_request(f"n{i}", RequestPriority.NORMAL) for i in range(5)]
        p = BatchProcessor(verify_fn=_ok_verify)
        results, _ = run(p.process(requests))

        ids = [r.request_id for r in results]
        assert ids == [f"n{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Concurrency limiting
# ---------------------------------------------------------------------------

class TestConcurrencyLimiting:
    def test_max_concurrency_respected(self):
        max_concurrency = 3
        state = {"active": 0, "peak": 0}

        async def counting_verify(req):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return {"ok": True}

        requests = [make_request(f"r{i}") for i in range(10)]
        p = BatchProcessor(verify_fn=counting_verify, max_concurrency=max_concurrency)
        run(p.process(requests))

        assert state["peak"] <= max_concurrency

    def test_high_concurrency_processes_all(self):
        requests = [make_request(f"r{i}") for i in range(10)]
        p = BatchProcessor(verify_fn=_ok_verify, max_concurrency=10)
        _, metrics = run(p.process(requests))

        assert metrics.success_count == 10


# ---------------------------------------------------------------------------
# Batch size chunking
# ---------------------------------------------------------------------------

class TestBatchSizeChunking:
    def test_batch_size_chunks_all_processed(self):
        requests = [make_request(f"r{i}") for i in range(10)]
        p = BatchProcessor(verify_fn=_ok_verify, batch_size=3)
        results, metrics = run(p.process(requests))

        assert len(results) == 10
        assert metrics.total_processed == 10

    def test_batch_size_larger_than_input(self):
        requests = [make_request(f"r{i}") for i in range(3)]
        p = BatchProcessor(verify_fn=_ok_verify, batch_size=100)
        results, metrics = run(p.process(requests))

        assert len(results) == 3
        assert metrics.success_count == 3

    def test_batch_size_one_processes_all(self):
        requests = [make_request(f"r{i}") for i in range(5)]
        p = BatchProcessor(verify_fn=_ok_verify, batch_size=1)
        _, metrics = run(p.process(requests))

        assert metrics.total_processed == 5


# ---------------------------------------------------------------------------
# Metrics accuracy
# ---------------------------------------------------------------------------

class TestMetricsAccuracy:
    def test_total_latency_equals_sum_of_individual(self):
        requests = [make_request(f"r{i}") for i in range(5)]
        p = BatchProcessor(verify_fn=_ok_verify)
        results, metrics = run(p.process(requests))

        expected = sum(r.latency_seconds for r in results)
        assert metrics.total_latency_seconds == pytest.approx(expected, rel=1e-5)

    def test_batch_wall_time_is_positive(self):
        requests = [make_request(f"r{i}") for i in range(3)]
        p = BatchProcessor(verify_fn=_ok_verify)
        _, metrics = run(p.process(requests))

        assert metrics.batch_wall_time_seconds > 0.0

    def test_total_requests_matches_input_length(self):
        requests = [make_request(f"r{i}") for i in range(7)]
        p = BatchProcessor(verify_fn=_ok_verify)
        _, metrics = run(p.process(requests))

        assert metrics.total_requests == 7

    def test_success_rate_all_pass(self):
        requests = [make_request(f"r{i}") for i in range(5)]
        p = BatchProcessor(verify_fn=_ok_verify)
        _, metrics = run(p.process(requests))

        assert metrics.success_rate == pytest.approx(1.0)

    def test_success_rate_all_fail(self):
        requests = [make_request(f"r{i}") for i in range(5)]
        p = BatchProcessor(verify_fn=_fail_verify)
        _, metrics = run(p.process(requests))

        assert metrics.success_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BatchResult fields
# ---------------------------------------------------------------------------

class TestBatchResultFields:
    def test_result_carries_priority(self):
        req = make_request("r1", RequestPriority.CRITICAL)
        p = BatchProcessor(verify_fn=_ok_verify)
        results, _ = run(p.process([req]))

        assert results[0].priority == RequestPriority.CRITICAL

    def test_result_carries_request_id(self):
        p = BatchProcessor(verify_fn=_ok_verify)
        results, _ = run(p.process([make_request("my-unique-id")]))

        assert results[0].request_id == "my-unique-id"

    def test_failed_result_has_error_instance(self):
        p = BatchProcessor(verify_fn=_fail_verify)
        results, _ = run(p.process([make_request("r1")]))

        assert results[0].error is not None
        assert isinstance(results[0].error, Exception)

    def test_successful_result_has_no_error(self):
        p = BatchProcessor(verify_fn=_ok_verify)
        results, _ = run(p.process([make_request("r1")]))

        assert results[0].error is None
        assert results[0].result is not None


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_verify_fn_receives_correct_request_fields(self):
        received = []

        async def capturing_verify(req):
            received.append(req)
            return {"approved": True}

        req = VerificationRequest(
            objection_text="Giá iPhone có đắt không?",
            draft_response="iPhone có giá 29,990,000 VND.",
            request_id="integration-1",
            priority=RequestPriority.HIGH,
            metadata={"workflow_id": "wf_001"},
        )
        p = BatchProcessor(verify_fn=capturing_verify)
        run(p.process([req]))

        assert len(received) == 1
        assert received[0].objection_text == "Giá iPhone có đắt không?"
        assert received[0].metadata == {"workflow_id": "wf_001"}

    def test_large_batch_mixed_priorities(self):
        random.seed(42)
        priorities = [RequestPriority.CRITICAL, RequestPriority.HIGH, RequestPriority.NORMAL]
        requests = [
            make_request(f"r{i}", random.choice(priorities))
            for i in range(50)
        ]
        p = BatchProcessor(verify_fn=_ok_verify, max_concurrency=10, batch_size=15)
        _, metrics = run(p.process(requests))

        assert metrics.total_processed == 50
        assert metrics.success_count == 50
        assert metrics.failure_count == 0
