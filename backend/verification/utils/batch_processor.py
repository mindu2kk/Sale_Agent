"""
Batch Processing for Multiple Verification Requests

Provides concurrent batch processing of verification requests:
- BatchProcessor: Accepts a list of VerificationRequest objects and processes
  them concurrently using asyncio with configurable batch size and concurrency.
- Priority queuing: critical requests are processed before normal ones.
- Batch metrics: total processed, success/fail counts, average latency.
- Integrates with VerificationAgent and WorkflowState patterns.

Supports Requirement 9: Performance & Scalability
- ≥10 parallel workflows (Req 9.2)
- Parallel checks to minimise latency (Req 9.3)
- Batch processing to optimise LLM calls (design Performance Considerations)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------

class RequestPriority(str, Enum):
    """Priority levels for batch verification requests."""
    CRITICAL = "critical"   # Processed first
    HIGH = "high"
    NORMAL = "normal"       # Default


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VerificationRequest:
    """
    A single verification request to be processed in a batch.

    Attributes
    ----------
    objection_text:
        The customer objection text.
    draft_response:
        The draft response to verify.
    request_id:
        Optional caller-supplied identifier for correlation.
    priority:
        Processing priority — CRITICAL requests are dequeued first.
    metadata:
        Arbitrary caller-supplied key/value pairs (e.g. workflow_id).
    """

    objection_text: str
    draft_response: str
    request_id: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Numeric priority for sorting (lower = higher priority)
    _PRIORITY_ORDER: Dict[RequestPriority, int] = field(
        default_factory=lambda: {
            RequestPriority.CRITICAL: 0,
            RequestPriority.HIGH: 1,
            RequestPriority.NORMAL: 2,
        },
        init=False,
        repr=False,
        compare=False,
    )

    def priority_value(self) -> int:
        """Return numeric sort key (lower = processed first)."""
        order = {
            RequestPriority.CRITICAL: 0,
            RequestPriority.HIGH: 1,
            RequestPriority.NORMAL: 2,
        }
        return order.get(self.priority, 2)


@dataclass
class BatchResult:
    """
    Result for a single request within a batch.

    Attributes
    ----------
    request_id:
        Mirrors the request_id from the originating VerificationRequest.
    result:
        The VerificationResult returned by the agent, or None on failure.
    success:
        True if verification completed without exception.
    error:
        Exception instance if success is False, else None.
    latency_seconds:
        Wall-clock time taken to process this request.
    priority:
        Priority of the originating request.
    """

    request_id: Optional[str]
    result: Any  # VerificationResult | None
    success: bool
    error: Optional[Exception]
    latency_seconds: float
    priority: RequestPriority = RequestPriority.NORMAL


@dataclass
class BatchMetrics:
    """
    Aggregate metrics for a completed batch.

    Attributes
    ----------
    total_requests:
        Number of requests submitted.
    total_processed:
        Number of requests that completed (success or failure).
    success_count:
        Number of requests that completed successfully.
    failure_count:
        Number of requests that raised an exception.
    avg_latency_seconds:
        Mean per-request latency across all processed requests.
    total_latency_seconds:
        Sum of all per-request latencies.
    batch_wall_time_seconds:
        Total elapsed wall-clock time for the entire batch.
    """

    total_requests: int = 0
    total_processed: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0
    batch_wall_time_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of processed requests that succeeded (0.0–1.0)."""
        if self.total_processed == 0:
            return 0.0
        return self.success_count / self.total_processed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_processed": self.total_processed,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_latency_seconds": self.avg_latency_seconds,
            "total_latency_seconds": self.total_latency_seconds,
            "batch_wall_time_seconds": self.batch_wall_time_seconds,
            "success_rate": self.success_rate,
        }


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

# Type alias for the async callable that processes a single request.
# Signature: async (request: VerificationRequest) -> Any
VerifyFn = Callable[["VerificationRequest"], Coroutine[Any, Any, Any]]


class BatchProcessor:
    """
    Concurrent batch processor for multiple verification requests.

    Processes a list of :class:`VerificationRequest` objects concurrently,
    respecting a configurable concurrency limit (semaphore) and optional
    batch size chunking.  Requests are sorted by priority before processing
    so that CRITICAL requests are always handled first.

    Parameters
    ----------
    verify_fn:
        Async callable ``async (request) -> VerificationResult``.
        Typically ``VerificationAgent.verify_draft`` wrapped to accept a
        :class:`VerificationRequest` instead of a raw ``WorkflowState``.
    max_concurrency:
        Maximum number of requests processed simultaneously.
        Mirrors the ``asyncio.Semaphore(10)`` pattern used in
        :class:`VerificationAgent`.  Defaults to 10.
    batch_size:
        Maximum requests per processing chunk.  When the input list is
        larger than *batch_size*, it is split into sequential chunks so
        that memory usage stays bounded.  ``None`` (default) means all
        requests are processed in a single chunk.

    Example
    -------
    ::

        async def my_verify(req: VerificationRequest):
            state = {"objection_text": req.objection_text,
                     "draft_response": req.draft_response}
            return await agent.verify_draft(state)

        processor = BatchProcessor(verify_fn=my_verify, max_concurrency=5)
        results, metrics = await processor.process(requests)
    """

    def __init__(
        self,
        verify_fn: VerifyFn,
        max_concurrency: int = 10,
        batch_size: Optional[int] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be ≥ 1")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be ≥ 1 or None")

        self._verify_fn = verify_fn
        self._max_concurrency = max_concurrency
        self._batch_size = batch_size
        # Semaphore is created lazily inside the running event loop
        self._semaphore: Optional[asyncio.Semaphore] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(
        self,
        requests: List[VerificationRequest],
    ) -> Tuple[List[BatchResult], BatchMetrics]:
        """
        Process a list of verification requests concurrently.

        Requests are sorted by priority (CRITICAL → HIGH → NORMAL) before
        processing.  Results are returned in the same order as the *sorted*
        input (i.e. critical results come first).

        Parameters
        ----------
        requests:
            List of :class:`VerificationRequest` objects to process.

        Returns
        -------
        results:
            One :class:`BatchResult` per request, in priority order.
        metrics:
            Aggregate :class:`BatchMetrics` for the entire batch.
        """
        if not requests:
            return [], BatchMetrics()

        # Initialise semaphore inside the running event loop
        self._semaphore = asyncio.Semaphore(self._max_concurrency)

        # Sort by priority (stable sort preserves original order within same priority)
        sorted_requests = sorted(requests, key=lambda r: r.priority_value())

        batch_start = time.monotonic()

        if self._batch_size is None:
            # Process all in one chunk
            results = await self._process_chunk(sorted_requests)
        else:
            # Process in sequential chunks to bound memory usage
            results = []
            for chunk_start in range(0, len(sorted_requests), self._batch_size):
                chunk = sorted_requests[chunk_start: chunk_start + self._batch_size]
                chunk_results = await self._process_chunk(chunk)
                results.extend(chunk_results)

        batch_wall_time = time.monotonic() - batch_start
        metrics = self._compute_metrics(results, batch_wall_time, len(requests))
        return results, metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_chunk(
        self, requests: List[VerificationRequest]
    ) -> List[BatchResult]:
        """Process a single chunk of requests concurrently."""
        tasks = [self._process_single(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def _process_single(
        self, request: VerificationRequest
    ) -> BatchResult:
        """Process one request under the semaphore, capturing errors."""
        assert self._semaphore is not None  # set in process()

        start = time.monotonic()
        async with self._semaphore:
            try:
                result = await self._verify_fn(request)
                latency = time.monotonic() - start
                return BatchResult(
                    request_id=request.request_id,
                    result=result,
                    success=True,
                    error=None,
                    latency_seconds=latency,
                    priority=request.priority,
                )
            except Exception as exc:  # noqa: BLE001
                latency = time.monotonic() - start
                return BatchResult(
                    request_id=request.request_id,
                    result=None,
                    success=False,
                    error=exc,
                    latency_seconds=latency,
                    priority=request.priority,
                )

    @staticmethod
    def _compute_metrics(
        results: List[BatchResult],
        batch_wall_time: float,
        total_requests: int,
    ) -> BatchMetrics:
        """Compute aggregate metrics from a list of BatchResult objects."""
        success_count = sum(1 for r in results if r.success)
        failure_count = sum(1 for r in results if not r.success)
        total_latency = sum(r.latency_seconds for r in results)
        avg_latency = total_latency / len(results) if results else 0.0

        return BatchMetrics(
            total_requests=total_requests,
            total_processed=len(results),
            success_count=success_count,
            failure_count=failure_count,
            avg_latency_seconds=avg_latency,
            total_latency_seconds=total_latency,
            batch_wall_time_seconds=batch_wall_time,
        )
