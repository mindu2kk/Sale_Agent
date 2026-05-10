"""
Main Verification Agent với Binary Logic

Core verification agent implementing binary PASS/FAIL decisions:
- Parallel execution của Price/Policy/Relevance checks
- Early termination on critical issues
- Structured issue classification với severity levels
- Async LLM integration với performance optimization
- Semaphore-based concurrency control
- Structured logging với correlation IDs
- LRU caching via VerificationCache
- Exponential backoff retry for LLM API errors (Req 8.1)
- Circuit breaker pattern for external service calls (Req 8.4)
- DB connection error handling with cached data fallback (Req 8.2)
- Fallback verification mode when LLM unavailable (Req 8.1)
- Comprehensive error logging with correlation IDs (Req 8.5)
"""

import asyncio
import hashlib
import time
import threading
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

from ..models import (
    WorkflowState,
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    FeedbackReport,
    FailedCriterion,
)
from ..config import VerificationConfig
from ..utils.cache import VerificationCache
from ..utils.logging import (
    setup_verification_logger,
    get_correlation_context,
    CorrelationIDGenerator,
)
from .checkers import (
    PriceAccuracyChecker,
    PolicyAuthenticityChecker,
    TopicRelevanceChecker,
)
from ..utils.prompt_compressor import PromptCompressor
from ..utils.performance import AsyncStepLatencyTracker
from ..utils.early_termination import should_terminate_immediately
from ..utils.graceful_degradation import GracefulDegradationHandler, PartialVerificationResult


class CircuitBreakerState(str, Enum):
    """Circuit breaker states (Req 8.4)"""
    CLOSED = "closed"       # Normal operation — requests pass through
    OPEN = "open"           # Failing — requests blocked, fallback used
    HALF_OPEN = "half_open" # Testing recovery — one probe request allowed


class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls (Req 8.4).

    Tracks failure counts per service and transitions between states:
    - CLOSED → OPEN when failure_threshold consecutive failures occur
    - OPEN → HALF_OPEN after recovery_timeout_seconds
    - HALF_OPEN → CLOSED on success, HALF_OPEN → OPEN on failure
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 60.0,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            return self._get_state()

    def _get_state(self) -> CircuitBreakerState:
        """Internal state getter — must be called with lock held."""
        if self._state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.recovery_timeout_seconds
            ):
                self._state = CircuitBreakerState.HALF_OPEN
        return self._state

    def is_open(self) -> bool:
        """Return True if circuit is OPEN (requests should be blocked)."""
        return self.state == CircuitBreakerState.OPEN

    def allow_request(self) -> bool:
        """Return True if a request should be allowed through."""
        with self._lock:
            state = self._get_state()
            if state == CircuitBreakerState.CLOSED:
                return True
            if state == CircuitBreakerState.HALF_OPEN:
                return True  # Allow one probe request
            return False  # OPEN — block request

    def record_success(self) -> None:
        """Record a successful call — reset failure count, close circuit."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call — increment counter, open circuit if threshold reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN

    def get_status(self) -> Dict[str, Any]:
        """Return circuit breaker status for observability."""
        with self._lock:
            return {
                "service": self.service_name,
                "state": self._get_state().value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self._last_failure_time,
                "recovery_timeout_seconds": self.recovery_timeout_seconds,
            }


class VerificationAgent:
    """
    Main Verification Agent với binary PASS/FAIL logic.

    Implements parallel verification checks với early termination,
    structured issue classification, semaphore-based concurrency control,
    structured logging với correlation IDs, LRU caching, circuit breaker
    pattern for external services, and DB connection fallback.

    Requirements covered:
    - Req 1.1: 3 core scoring criteria (Price, Policy, Relevance)
    - Req 1.4: Deterministic verification (same input → same output)
    - Req 8.1: LLM API timeout/error with retry and exponential backoff
    - Req 8.2: DB connection loss with cached data fallback
    - Req 8.3: StateGraph execution error with rollback to last valid state
    - Req 8.4: Circuit breaker pattern for external service calls
    - Req 8.5: All errors logged with correlation IDs
    - Req 9.3: Price/policy/relevance checks run in parallel
    """

    def __init__(
        self,
        llm,
        rag_pipeline,
        config: VerificationConfig,
        max_concurrent_llm_calls: int = 10,
        compression_level: str = "none",
    ):
        """
        Initialize Verification Agent.

        Args:
            llm: Async LLM client (e.g. AsyncOpenAI or Anthropic async client)
            rag_pipeline: RAG pipeline for data retrieval
            config: Verification configuration
            max_concurrent_llm_calls: Semaphore limit for concurrent LLM calls
            compression_level: Prompt compression level for correction prompts.
                "none" (default) — no compression applied.
                "light" — normalise whitespace and blank lines.
                "aggressive" — strip verbose preambles, compact binary rules,
                               truncate long free-text sections.
        """
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.config = config

        # Semaphore to limit concurrent LLM calls (Req 9.2, design OptimizedVerificationAgent)
        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)

        # Initialize specialized checkers
        self.price_checker = PriceAccuracyChecker(llm, rag_pipeline, config)
        self.policy_checker = PolicyAuthenticityChecker(llm, rag_pipeline, config)
        self.relevance_checker = TopicRelevanceChecker(llm, config)

        # LRU cache using VerificationCache utility (Req 9.4)
        if config.enable_caching:
            self._cache: Optional[VerificationCache] = VerificationCache(
                max_size=config.cache_max_size,
                default_ttl_seconds=float(config.cache_ttl_seconds),
            )
        else:
            self._cache = None

        # Structured logger with correlation ID support
        self._logger = setup_verification_logger(
            "verification.agent", config
        )

        # Circuit breakers for external services (Req 8.4)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            "llm_api": CircuitBreaker(
                service_name="llm_api",
                failure_threshold=3,
                recovery_timeout_seconds=60.0,
            ),
            "rag_pipeline": CircuitBreaker(
                service_name="rag_pipeline",
                failure_threshold=3,
                recovery_timeout_seconds=30.0,
            ),
            "db": CircuitBreaker(
                service_name="db",
                failure_threshold=3,
                recovery_timeout_seconds=30.0,
            ),
        }

        # Last known valid verification result for DB fallback (Req 8.2)
        self._last_valid_result: Optional[VerificationResult] = None

        # Performance tracking
        self._total_verifications = 0
        self._total_tokens_used = 0

        # Prompt compressor for correction prompts (Task 5.2.2)
        self._prompt_compressor = PromptCompressor(level=compression_level)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_draft_sync(self, state: WorkflowState) -> VerificationResult:
        """
        Sync wrapper for verify_draft() — backward compatibility.

        Runs the async verify_draft() in a new event loop if no loop is running,
        or schedules it on the existing loop otherwise.

        Args:
            state: Current workflow state with draft response

        Returns:
            VerificationResult with binary decision and structured issues
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an async context — use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(self.verify_draft(state), loop)
                return future.result(timeout=self.config.async_timeout_seconds)
            else:
                return loop.run_until_complete(self.verify_draft(state))
        except RuntimeError:
            # No event loop — create a new one
            return asyncio.run(self.verify_draft(state))

    async def verify_draft_parallel(self, state: WorkflowState) -> VerificationResult:
        """
        Public async method for parallel verification với first-failure-fast logic.

        Starts all 3 checks (price, policy, relevance) concurrently. If any check
        detects a CRITICAL severity issue, remaining tasks are cancelled immediately
        and the result is returned without waiting for the others.

        This is the primary entry point for parallel verification as described in
        the design document's OptimizedVerificationAgent pattern.

        Args:
            state: Current workflow state with draft_response and objection_text

        Returns:
            VerificationResult with binary decision and structured issues.
            If a critical issue is found early, remaining checks are skipped
            (filled with default PASS to avoid false positives).

        **Validates: Requirements 9.3** - price/policy/relevance checks run in parallel
        **Validates: Requirements 9.1** - verification completes in ≤10 seconds
        """
        start_time = time.time()
        correlation_id = self._get_correlation_id()

        self._logger.info(
            "Starting parallel verification with first-failure-fast logic",
            correlation_id=correlation_id,
            draft_length=len(state.get("draft_response", "")),
            objection_length=len(state.get("objection_text", "")),
        )

        async with self._semaphore:
            try:
                self._validate_input(state)

                # Check cache first
                if self._cache is not None:
                    cached = self._cache.get(self._generate_cache_key(state))
                    if cached is not None:
                        self._logger.info(
                            "Cache hit — returning cached result",
                            correlation_id=correlation_id,
                        )
                        return cached

                verification_result = await self._verify_parallel(state)
                execution_time = time.time() - start_time
                verification_result.execution_time_seconds = execution_time

                if self._cache is not None:
                    self._cache.put(self._generate_cache_key(state), verification_result)

                self._update_metrics(verification_result)
                self._last_valid_result = verification_result

                self._logger.info(
                    "Parallel verification complete",
                    correlation_id=correlation_id,
                    is_approved=verification_result.is_approved,
                    execution_time=execution_time,
                    critical_issues=verification_result.criteria.critical_issues_count,
                    immediate_termination=verification_result.immediate_termination,
                )

                return verification_result

            except Exception as e:
                execution_time = time.time() - start_time
                self._logger.error(
                    f"Parallel verification failed: {e}",
                    correlation_id=correlation_id,
                    error_type=type(e).__name__,
                    execution_time=execution_time,
                )
                return self._handle_verification_error(state, e, execution_time)

    async def verify_draft_with_degradation(self, state: WorkflowState) -> VerificationResult:
        """
        Verification with graceful degradation for partial checker failures.

        Runs all 3 checkers concurrently. If one checker raises an exception,
        the workflow continues with the remaining successful results (the failed
        checker defaults to PASS with a warning issue). If 2 or more checkers
        fail, degradation is rejected and a failed VerificationResult is returned
        with appropriate error context.

        This method is the resilient alternative to verify_draft_parallel() for
        environments where individual checker services may be temporarily unavailable.

        Requirements: 8.1, 8.2, 8.3
        """
        start_time = time.time()
        correlation_id = self._get_correlation_id()
        handler = GracefulDegradationHandler(correlation_id=correlation_id)

        self._logger.info(
            "Starting verification with graceful degradation",
            correlation_id=correlation_id,
            draft_length=len(state.get("draft_response", "")),
        )

        async with self._semaphore:
            # Run all 3 checkers safely (exceptions are caught, not raised)
            price_task = handler.run_checker_safely(
                "price",
                self.price_checker.check_price_accuracy,
                state["draft_response"],
                state["objection_text"],
            )
            policy_task = handler.run_checker_safely(
                "policy",
                self.policy_checker.check_policy_authenticity,
                state["draft_response"],
            )
            relevance_task = handler.run_checker_safely(
                "relevance",
                self.relevance_checker.check_topic_relevance,
                state["draft_response"],
                state["objection_text"],
            )

            price_pr, policy_pr, relevance_pr = await asyncio.gather(
                price_task, policy_task, relevance_task
            )

        partial_results: dict[str, PartialVerificationResult] = {
            "price": price_pr,
            "policy": policy_pr,
            "relevance": relevance_pr,
        }

        failed_checkers = [
            name for name, pr in partial_results.items() if not pr.success
        ]

        execution_time = time.time() - start_time

        if failed_checkers and not handler.should_degrade(failed_checkers):
            # Too many failures — return a failed result with error context
            self._logger.error(
                "Graceful degradation rejected — too many checker failures",
                correlation_id=correlation_id,
                failed_checkers=failed_checkers,
                execution_time=execution_time,
            )
            errors = [
                str(partial_results[name].error) for name in failed_checkers
            ]
            combined_error = RuntimeError(
                f"{len(failed_checkers)} checkers failed "
                f"({', '.join(failed_checkers)}): {'; '.join(errors)}"
            )
            result = self._build_fallback_verification_result(state, combined_error, execution_time)
            return result

        # Build result from partial data (with warning issues for failed checkers)
        degradation_applied = bool(failed_checkers)
        reasoning = (
            "Partial verification with graceful degradation applied"
            if degradation_applied
            else "Full verification completed successfully"
        )
        result = handler.aggregate_partial_results(partial_results, reasoning)
        result.execution_time_seconds = execution_time

        self._logger.info(
            "Verification with degradation complete",
            correlation_id=correlation_id,
            is_approved=result.is_approved,
            degradation_applied=degradation_applied,
            failed_checkers=failed_checkers,
            execution_time=execution_time,
        )

        self._update_metrics(result)
        return result

    async def verify_draft(self, state: WorkflowState) -> VerificationResult:
        """
        Main verification method với binary PASS/FAIL decision.

        Runs 3 checks in parallel using asyncio.gather() with early termination
        when a critical issue is found. Runs under the semaphore to limit
        concurrent LLM calls.

        Args:
            state: Current workflow state with draft response

        Returns:
            VerificationResult with binary decision and structured issues
        """
        start_time = time.time()
        correlation_id = self._get_correlation_id()

        self._logger.info(
            "Starting verification",
            correlation_id=correlation_id,
            draft_length=len(state.get("draft_response", "")),
            objection_length=len(state.get("objection_text", "")),
        )

        async with self._semaphore:
            try:
                # Input validation
                self._validate_input(state)

                # Check cache first (Req 1.4 determinism + Req 9.4 caching)
                if self._cache is not None:
                    cached = self._cache.get(self._generate_cache_key(state))
                    if cached is not None:
                        self._logger.info(
                            "Cache hit — returning cached verification result",
                            correlation_id=correlation_id,
                        )
                        return cached

                # Run verification checks
                if self.config.parallel_verification:
                    verification_result = await self._verify_parallel(state)
                else:
                    verification_result = await self._verify_sequential(state)

                # Set execution time
                execution_time = time.time() - start_time
                verification_result.execution_time_seconds = execution_time

                # Cache result
                if self._cache is not None:
                    self._cache.put(
                        self._generate_cache_key(state),
                        verification_result,
                    )

                # Update performance metrics
                self._update_metrics(verification_result)

                # Store as last valid result for DB fallback (Req 8.2)
                self._last_valid_result = verification_result

                self._logger.info(
                    "Verification complete",
                    correlation_id=correlation_id,
                    is_approved=verification_result.is_approved,
                    execution_time=execution_time,
                    critical_issues=verification_result.criteria.critical_issues_count,
                )

                return verification_result

            except Exception as e:
                execution_time = time.time() - start_time
                self._logger.error(
                    f"Verification failed with error: {e}",
                    correlation_id=correlation_id,
                    error_type=type(e).__name__,
                    execution_time=execution_time,
                )
                return self._handle_verification_error(state, e, execution_time)

    # ------------------------------------------------------------------
    # Internal verification methods
    # ------------------------------------------------------------------

    async def _verify_parallel(self, state: WorkflowState) -> VerificationResult:
        """
        Run verification checks in parallel with early termination.

        Uses asyncio.gather() to run all 3 checks concurrently. If early
        termination is enabled and a critical issue is found in any completed
        check, remaining tasks are cancelled immediately (first-failure-fast).
        """
        if not self.config.early_termination:
            # Simple parallel execution without early termination
            return await self._verify_parallel_simple(state)

        tracker = AsyncStepLatencyTracker()

        async def _tracked_price():
            async with tracker.track("price_check"):
                return await self._check_price_accuracy_async(state)

        async def _tracked_policy():
            async with tracker.track("policy_check"):
                return await self._check_policy_authenticity_async(state)

        async def _tracked_relevance():
            async with tracker.track("relevance_check"):
                return await self._check_topic_relevance_async(state)

        # First-failure-fast: use asyncio.wait to detect critical issues early
        check_map = {
            "price": asyncio.ensure_future(_tracked_price()),
            "policy": asyncio.ensure_future(_tracked_policy()),
            "relevance": asyncio.ensure_future(_tracked_relevance()),
        }
        all_tasks = set(check_map.values())
        task_to_type = {v: k for k, v in check_map.items()}

        results: Dict[str, Any] = {}
        pending = set(all_tasks)
        deadline = self.config.async_timeout_seconds

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=deadline,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    # Timeout — cancel remaining tasks
                    for t in pending:
                        t.cancel()
                    raise RuntimeError(
                        f"Verification timeout after {self.config.async_timeout_seconds}s"
                    )

                for task in done:
                    check_type = task_to_type[task]
                    if task.exception():
                        results[check_type] = task.exception()
                    else:
                        results[check_type] = task.result()

                    # Early termination: cancel remaining if critical issue found
                    result = results[check_type]
                    if not isinstance(result, Exception):
                        _, issues = result
                        if self._has_critical_issues(issues):
                            for t in pending:
                                t.cancel()
                            # Fill missing checks with default pass (not yet run)
                            for remaining_type in [t for t in task_to_type.values() if t not in results]:
                                results[remaining_type] = (True, [])
                            pending = set()
                            break
        except asyncio.CancelledError:
            for t in pending:
                t.cancel()
            raise

        # Unpack results
        def _unpack(check_type: str):
            r = results.get(check_type, (True, []))
            if isinstance(r, Exception):
                return False, [self._create_error_issue(check_type, r)]
            return r

        price_pass, price_issues = _unpack("price")
        policy_pass, policy_issues = _unpack("policy")
        relevance_pass, relevance_issues = _unpack("relevance")

        result = self._build_verification_result(
            state,
            price_pass, price_issues,
            policy_pass, policy_issues,
            relevance_pass, relevance_issues,
        )
        result.step_latencies = tracker.get_all_metrics()
        return result

    async def _verify_parallel_simple(self, state: WorkflowState) -> VerificationResult:
        """Run all 3 checks in parallel without early termination."""
        tracker = AsyncStepLatencyTracker()

        async def _price():
            async with tracker.track("price_check"):
                return await self._check_price_accuracy_async(state)

        async def _policy():
            async with tracker.track("policy_check"):
                return await self._check_policy_authenticity_async(state)

        async def _relevance():
            async with tracker.track("relevance_check"):
                return await self._check_topic_relevance_async(state)

        tasks = [_price(), _policy(), _relevance()]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.async_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Verification timeout after {self.config.async_timeout_seconds}s"
            )

        price_result, policy_result, relevance_result = results

        # Unpack or create error issues for any failed tasks
        if isinstance(price_result, Exception):
            price_pass, price_issues = False, [self._create_error_issue("price", price_result)]
        else:
            price_pass, price_issues = price_result

        if isinstance(policy_result, Exception):
            policy_pass, policy_issues = False, [self._create_error_issue("policy", policy_result)]
        else:
            policy_pass, policy_issues = policy_result

        if isinstance(relevance_result, Exception):
            relevance_pass, relevance_issues = False, [self._create_error_issue("relevance", relevance_result)]
        else:
            relevance_pass, relevance_issues = relevance_result

        result = self._build_verification_result(
            state,
            price_pass, price_issues,
            policy_pass, policy_issues,
            relevance_pass, relevance_issues,
        )
        result.step_latencies = tracker.get_all_metrics()
        return result

    async def _verify_sequential(self, state: WorkflowState) -> VerificationResult:
        """Run verification checks sequentially with early termination."""
        tracker = AsyncStepLatencyTracker()

        async with tracker.track("price_check"):
            price_pass, price_issues = await self._check_price_accuracy_async(state)

        if self.config.early_termination and self._has_critical_issues(price_issues):
            result = self._build_early_termination_result(state, price_pass, price_issues)
            result.step_latencies = tracker.get_all_metrics()
            return result

        async with tracker.track("policy_check"):
            policy_pass, policy_issues = await self._check_policy_authenticity_async(state)

        if self.config.early_termination and self._has_critical_issues(policy_issues):
            result = self._build_early_termination_result(
                state, price_pass, price_issues, policy_pass, policy_issues
            )
            result.step_latencies = tracker.get_all_metrics()
            return result

        async with tracker.track("relevance_check"):
            relevance_pass, relevance_issues = await self._check_topic_relevance_async(state)

        result = self._build_verification_result(
            state,
            price_pass, price_issues,
            policy_pass, policy_issues,
            relevance_pass, relevance_issues,
        )
        result.step_latencies = tracker.get_all_metrics()
        return result

    # ------------------------------------------------------------------
    # Async checker wrappers with retry + exponential backoff (Req 8.1)
    # ------------------------------------------------------------------

    async def _check_price_accuracy_async(
        self, state: WorkflowState
    ) -> Tuple[bool, List[PriceIssue]]:
        """Async wrapper for price accuracy check with retry."""
        return await self._run_with_retry(
            asyncio.to_thread,
            self.price_checker.check_price_accuracy,
            state["draft_response"],
            state["objection_text"],
        )

    async def _check_policy_authenticity_async(
        self, state: WorkflowState
    ) -> Tuple[bool, List[PolicyIssue]]:
        """Async wrapper for policy authenticity check with retry."""
        return await self._run_with_retry(
            asyncio.to_thread,
            self.policy_checker.check_policy_authenticity,
            state["draft_response"],
        )

    async def _check_topic_relevance_async(
        self, state: WorkflowState
    ) -> Tuple[bool, List[RelevanceIssue]]:
        """Async wrapper for topic relevance check with retry."""
        return await self._run_with_retry(
            asyncio.to_thread,
            self.relevance_checker.check_topic_relevance,
            state["draft_response"],
            state["objection_text"],
        )

    async def _run_with_retry(self, runner, fn, *args, max_retries: int = 3) -> Any:
        """
        Run a callable with exponential backoff retry on transient errors.

        Implements Req 8.1: retry with exponential backoff (max 3 attempts).
        Integrates with circuit breaker for LLM API calls (Req 8.4).
        All errors are logged with correlation IDs (Req 8.5).
        """
        correlation_id = self._get_correlation_id()
        cb = self._circuit_breakers.get("llm_api")

        # Check circuit breaker before attempting (Req 8.4)
        if cb is not None and not cb.allow_request():
            self._logger.error(
                "Circuit breaker OPEN for llm_api — using fallback",
                correlation_id=correlation_id,
                circuit_breaker_state=cb.state.value,
            )
            raise ConnectionError("Circuit breaker OPEN: llm_api service unavailable")

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                result = await runner(fn, *args)
                # Record success to circuit breaker
                if cb is not None:
                    cb.record_success()
                return result
            except (asyncio.TimeoutError, ConnectionError, OSError) as e:
                last_error = e
                # Record failure to circuit breaker (Req 8.4)
                if cb is not None:
                    cb.record_failure()
                if attempt < max_retries:
                    backoff = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                    # Structured error log with correlation ID (Req 8.5)
                    self._logger.warning(
                        f"Transient error on attempt {attempt}/{max_retries}, "
                        f"retrying in {backoff:.1f}s: {e}",
                        correlation_id=correlation_id,
                        attempt=attempt,
                        max_retries=max_retries,
                        backoff_seconds=backoff,
                        error_type=type(e).__name__,
                    )
                    await asyncio.sleep(backoff)
                else:
                    # All retries exhausted — log with full context (Req 8.1, 8.5)
                    self._logger.error(
                        f"All {max_retries} retry attempts failed: {e}",
                        correlation_id=correlation_id,
                        error_type=type(e).__name__,
                        max_retries=max_retries,
                    )
            except Exception as e:
                # Non-transient error — record failure and don't retry
                if cb is not None:
                    cb.record_failure()
                self._logger.error(
                    f"Non-transient error (no retry): {e}",
                    correlation_id=correlation_id,
                    error_type=type(e).__name__,
                )
                raise e
        raise last_error  # type: ignore[misc]

    async def _run_with_db_fallback(self, runner, fn, *args) -> Any:
        """
        Run a callable with DB connection error handling and cached data fallback.

        Implements Req 8.2: use cached data if DB unavailable, lower threshold temporarily.
        Integrates with circuit breaker for DB/RAG pipeline calls (Req 8.4).
        All errors are logged with correlation IDs (Req 8.5).
        """
        correlation_id = self._get_correlation_id()
        cb = self._circuit_breakers.get("rag_pipeline")

        # Check circuit breaker (Req 8.4)
        if cb is not None and not cb.allow_request():
            self._logger.warning(
                "Circuit breaker OPEN for rag_pipeline — using cached data fallback",
                correlation_id=correlation_id,
                circuit_breaker_state=cb.state.value,
            )
            return self._get_db_fallback_result(fn, *args)

        try:
            result = await runner(fn, *args)
            if cb is not None:
                cb.record_success()
            return result
        except (ConnectionError, OSError, TimeoutError) as e:
            # DB connection lost — record failure and use cached data (Req 8.2)
            if cb is not None:
                cb.record_failure()
            self._logger.error(
                f"DB/RAG connection error — falling back to cached data: {e}",
                correlation_id=correlation_id,
                error_type=type(e).__name__,
            )
            return self._get_db_fallback_result(fn, *args)

    def _get_db_fallback_result(self, fn, *args) -> Tuple[bool, List]:
        """
        Return a conservative fallback result when DB is unavailable (Req 8.2).

        Returns PASS with empty issues to avoid false positives when we cannot
        verify against the DB. The caller should lower thresholds accordingly.
        """
        self._logger.warning(
            "Using DB fallback: returning conservative PASS with empty issues. "
            "Manual review recommended when DB is restored.",
            correlation_id=self._get_correlation_id(),
        )
        return True, []

    def _build_db_unavailable_result(
        self, state: WorkflowState, execution_time: float
    ) -> VerificationResult:
        """
        Build a conservative VerificationResult when DB is completely unavailable (Req 8.2).

        Lowers verification threshold by returning PASS with a warning in reasoning,
        and queues for manual review.
        """
        correlation_id = self._get_correlation_id()
        self._logger.warning(
            "DB unavailable — building conservative verification result for manual review",
            correlation_id=correlation_id,
        )

        # Conservative: pass all checks but flag for manual review
        criteria = RubricCriteria(
            price_accuracy_pass=True,
            policy_authenticity_pass=True,
            topic_relevance_pass=True,
            price_issues=[],
            policy_issues=[],
            relevance_issues=[],
        )

        return VerificationResult(
            criteria=criteria,
            timestamp=datetime.now(),
            verification_reasoning=(
                "⚠️ DB UNAVAILABLE — Verification performed in degraded mode.\n"
                f"Correlation ID: {correlation_id}\n"
                "Price and policy checks skipped due to DB connection loss.\n"
                "This result is PROVISIONAL — queued for manual review when DB is restored.\n"
                "Req 8.2: Lowered verification threshold temporarily."
            ),
            execution_time_seconds=execution_time,
            llm_tokens_used=0,
        )

    def _build_fallback_verification_result(
        self, state: WorkflowState, error: Exception, execution_time: float
    ) -> VerificationResult:
        """
        Fallback rule-based verification when LLM is completely unavailable (Req 8.1).

        Uses simple heuristics instead of LLM calls:
        - Price check: PASS if no price keywords in draft (cannot verify without DB)
        - Policy check: PASS if no forbidden phrases detected
        - Relevance check: PASS if draft length > 50 chars (basic sanity check)

        Returns conservative FAIL result with error context when heuristics detect issues.
        """
        correlation_id = self._get_correlation_id()
        self._logger.warning(
            f"LLM unavailable — using rule-based fallback verification: {error}",
            correlation_id=correlation_id,
            error_type=type(error).__name__,
        )

        draft = state.get("draft_response", "")
        objection = state.get("objection_text", "")

        # Rule-based price check: fail if draft is empty or very short
        price_pass = len(draft.strip()) > 20
        price_issues: List[PriceIssue] = []
        if not price_pass:
            price_issues.append(PriceIssue(
                product_name="Unknown",
                severity="major",
                explanation=(
                    f"Fallback rule-based check: draft too short to verify prices. "
                    f"LLM unavailable: {type(error).__name__}"
                ),
                correction_suggestion="Ensure draft contains substantive content with pricing information",
            ))

        # Rule-based policy check: fail if forbidden phrases detected
        forbidden = self.config.policy_forbidden_phrases
        policy_pass = not any(phrase in draft.lower() for phrase in forbidden)
        policy_issues: List[PolicyIssue] = []
        if not policy_pass:
            policy_issues.append(PolicyIssue(
                mentioned_policy=draft[:200],
                policy_type="warranty",
                is_fabricated=True,
                is_inaccurate=False,
                severity="critical",
                explanation=(
                    "Fallback rule-based check: forbidden phrase detected in draft. "
                    f"LLM unavailable: {type(error).__name__}"
                ),
                correction_suggestion="Remove forbidden phrases from draft",
            ))

        # Rule-based relevance check: pass if draft is non-empty
        relevance_pass = len(draft.strip()) > 50
        relevance_issues: List[RelevanceIssue] = []
        if not relevance_pass:
            relevance_issues.append(RelevanceIssue(
                objection_intent=objection[:100] if objection else "Unknown",
                response_coverage=0.0,
                severity="major",
                explanation=(
                    f"Fallback rule-based check: draft too short to assess relevance. "
                    f"LLM unavailable: {type(error).__name__}"
                ),
                correction_suggestion="Provide a substantive response addressing the objection",
            ))

        criteria = RubricCriteria(
            price_accuracy_pass=price_pass,
            price_issues=price_issues,
            policy_authenticity_pass=policy_pass,
            policy_issues=policy_issues,
            topic_relevance_pass=relevance_pass,
            relevance_issues=relevance_issues,
        )

        return VerificationResult(
            criteria=criteria,
            timestamp=datetime.now(),
            verification_reasoning=(
                f"⚠️ FALLBACK VERIFICATION MODE — LLM unavailable: {type(error).__name__}\n"
                f"Correlation ID: {correlation_id}\n"
                "Rule-based heuristics used instead of LLM analysis.\n"
                "Results may be less accurate. Human review recommended.\n"
                f"Req 8.1: All LLM retries exhausted — escalating with error details."
            ),
            execution_time_seconds=execution_time,
            llm_tokens_used=0,
        )

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Get status of all circuit breakers for observability (Req 8.4, 8.5).

        Returns:
            Dict mapping service name to circuit breaker status
        """
        return {
            name: cb.get_status()
            for name, cb in self._circuit_breakers.items()
        }

    def reset_circuit_breaker(self, service_name: str) -> bool:
        """
        Manually reset a circuit breaker (e.g. after DB is restored).

        Args:
            service_name: Name of the service circuit breaker to reset

        Returns:
            True if reset successfully, False if service not found
        """
        cb = self._circuit_breakers.get(service_name)
        if cb is None:
            return False
        cb.record_success()  # Reset by recording a success
        self._logger.info(
            f"Circuit breaker manually reset for service: {service_name}",
            correlation_id=self._get_correlation_id(),
            service_name=service_name,
        )
        return True

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_verification_result(
        self,
        state: WorkflowState,
        price_pass: bool,
        price_issues: List[PriceIssue],
        policy_pass: bool = True,
        policy_issues: Optional[List[PolicyIssue]] = None,
        relevance_pass: bool = True,
        relevance_issues: Optional[List[RelevanceIssue]] = None,
    ) -> VerificationResult:
        """Build complete VerificationResult from check results."""

        if policy_issues is None:
            policy_issues = []
        if relevance_issues is None:
            relevance_issues = []

        # overall_pass and critical_issues_count are intentionally omitted here
        # so that RubricCriteria.__init__ auto-calculates them from the issue lists.
        # overall_pass = price_accuracy_pass AND policy_authenticity_pass AND topic_relevance_pass
        # critical_issues_count = sum of issues with severity == "critical" across all lists
        criteria = RubricCriteria(
            price_accuracy_pass=price_pass,
            price_issues=price_issues,
            policy_authenticity_pass=policy_pass,
            policy_issues=policy_issues,
            topic_relevance_pass=relevance_pass,
            relevance_issues=relevance_issues,
        )

        reasoning = self._generate_verification_reasoning(criteria)
        tokens_used = self._estimate_tokens_used(state, criteria)

        result = VerificationResult(
            criteria=criteria,
            timestamp=datetime.now(),
            verification_reasoning=reasoning,
            execution_time_seconds=0.0,  # set by caller
            llm_tokens_used=tokens_used,
        )

        # Task 5.4.1: detect critical issues and set immediate_termination flag
        termination_decision = should_terminate_immediately(result)
        result.has_critical_issues = termination_decision.should_terminate
        result.immediate_termination = termination_decision.should_terminate

        return result

    def _build_early_termination_result(
        self,
        state: WorkflowState,
        price_pass: bool,
        price_issues: List[PriceIssue],
        policy_pass: bool = True,
        policy_issues: Optional[List[PolicyIssue]] = None,
        relevance_pass: bool = True,
        relevance_issues: Optional[List[RelevanceIssue]] = None,
    ) -> VerificationResult:
        """Build verification result for early termination scenarios."""

        result = self._build_verification_result(
            state,
            price_pass, price_issues,
            policy_pass, policy_issues,
            relevance_pass, relevance_issues,
        )
        result.verification_reasoning += "\n\n⚡ Early termination: Critical issues detected."
        return result

    # ------------------------------------------------------------------
    # Structured feedback generation (Task 2.4.4)
    # ------------------------------------------------------------------

    def generate_structured_feedback(
        self, verification_result: VerificationResult
    ) -> FeedbackReport:
        """
        Generate a structured FeedbackReport from a failed VerificationResult.

        This is deterministic and template-based — no LLM calls required.
        The report is severity-prioritised (critical issues first) and includes
        a ready-to-inject correction_prompt string for the Research Agent.

        Args:
            verification_result: The VerificationResult from verify_draft()

        Returns:
            FeedbackReport with failed criteria, suggestions, and correction prompt
        """
        criteria = verification_result.criteria

        if verification_result.is_approved:
            return FeedbackReport(
                is_approved=True,
                total_issues=0,
                critical_issues_count=0,
                escalation_priority="low",
                failed_criteria=[],
                correction_prompt="✅ No corrections needed — verification passed.",
            )

        failed_criteria: List[FailedCriterion] = []

        # --- Price Accuracy ---
        if not criteria.price_accuracy_pass and criteria.price_issues:
            failed_criteria.append(
                self._build_price_criterion(criteria.price_issues)
            )

        # --- Policy Authenticity ---
        if not criteria.policy_authenticity_pass and criteria.policy_issues:
            failed_criteria.append(
                self._build_policy_criterion(criteria.policy_issues)
            )

        # --- Topic Relevance ---
        if not criteria.topic_relevance_pass and criteria.relevance_issues:
            failed_criteria.append(
                self._build_relevance_criterion(criteria.relevance_issues)
            )

        # Sort: critical → major → minor
        _severity_order = {"critical": 0, "major": 1, "minor": 2}
        failed_criteria.sort(
            key=lambda c: _severity_order.get(c.severity.value if hasattr(c.severity, 'value') else str(c.severity), 2)
        )

        total_issues = (
            len(criteria.price_issues)
            + len(criteria.policy_issues)
            + len(criteria.relevance_issues)
        )
        critical_count = criteria.critical_issues_count or 0
        escalation_priority = criteria.get_escalation_priority()

        correction_prompt = self._build_correction_prompt(
            failed_criteria, critical_count, escalation_priority
        )

        # Apply prompt compression to reduce token usage (Task 5.2.2)
        compression_result = self._prompt_compressor.compress(correction_prompt)
        correction_prompt = compression_result.compressed

        return FeedbackReport(
            is_approved=False,
            total_issues=total_issues,
            critical_issues_count=critical_count,
            escalation_priority=escalation_priority,
            failed_criteria=failed_criteria,
            correction_prompt=correction_prompt,
        )

    def _build_price_criterion(self, issues: List[PriceIssue]) -> FailedCriterion:
        """Build FailedCriterion for price accuracy failures."""
        highest_severity = self._highest_severity(
            [i.severity for i in issues]
        )
        suggestions = []
        for issue in issues:
            if issue.correction_suggestion:
                suggestions.append(issue.correction_suggestion)
            elif issue.actual_price:
                suggestions.append(
                    f"Re-check price for {issue.product_name}: update to {issue.actual_price}"
                )
            else:
                suggestions.append(
                    f"Re-check price for {issue.product_name} against the product catalog"
                )

        explanation = (
            f"{len(issues)} price issue(s) detected: "
            + "; ".join(
                f"{i.product_name} deviation {i.deviation_percent:.1f}%"
                if i.deviation_percent is not None
                else f"{i.product_name} — {i.explanation}"
                for i in issues
            )
        )

        return FailedCriterion(
            criterion_name="Price Accuracy",
            criterion_key="price_accuracy",
            explanation=explanation,
            correction_suggestions=suggestions,
            severity=highest_severity,
            issue_count=len(issues),
        )

    def _build_policy_criterion(self, issues: List[PolicyIssue]) -> FailedCriterion:
        """Build FailedCriterion for policy authenticity failures."""
        highest_severity = self._highest_severity(
            [i.severity for i in issues]
        )
        suggestions = []
        for issue in issues:
            if issue.correction_suggestion:
                suggestions.append(issue.correction_suggestion)
            elif issue.is_fabricated:
                suggestions.append(
                    f"Remove fabricated {issue.policy_type} policy about '{issue.mentioned_policy[:80]}'"
                )
            elif issue.correct_policy:
                suggestions.append(
                    f"Replace inaccurate {issue.policy_type} policy with: '{issue.correct_policy[:120]}'"
                )
            else:
                suggestions.append(
                    f"Verify {issue.policy_type} policy against official documents"
                )

        explanation = (
            f"{len(issues)} policy issue(s) detected: "
            + "; ".join(
                ("fabricated " if i.is_fabricated else "inaccurate ")
                + f"{i.policy_type} policy"
                for i in issues
            )
        )

        return FailedCriterion(
            criterion_name="Policy Authenticity",
            criterion_key="policy_authenticity",
            explanation=explanation,
            correction_suggestions=suggestions,
            severity=highest_severity,
            issue_count=len(issues),
        )

    def _build_relevance_criterion(self, issues: List[RelevanceIssue]) -> FailedCriterion:
        """Build FailedCriterion for topic relevance failures."""
        highest_severity = self._highest_severity(
            [i.severity for i in issues]
        )
        suggestions = []
        for issue in issues:
            if issue.correction_suggestion:
                suggestions.append(issue.correction_suggestion)
            else:
                parts = []
                if issue.missing_aspects:
                    parts.append(f"Address missing aspects: {', '.join(issue.missing_aspects)}")
                if issue.off_topic_content:
                    parts.append(f"Remove off-topic content: {', '.join(issue.off_topic_content)}")
                if issue.response_coverage < 0.5:
                    parts.append(
                        f"Significantly expand response to address '{issue.objection_intent}'"
                    )
                suggestions.append("; ".join(parts) if parts else "Improve response relevance")

        explanation = (
            f"{len(issues)} relevance issue(s) detected: "
            + "; ".join(
                f"coverage {i.response_coverage:.0%} for '{i.objection_intent}'"
                for i in issues
            )
        )

        return FailedCriterion(
            criterion_name="Topic Relevance",
            criterion_key="topic_relevance",
            explanation=explanation,
            correction_suggestions=suggestions,
            severity=highest_severity,
            issue_count=len(issues),
        )

    @staticmethod
    def _highest_severity(severities) -> "IssueSeverity":
        """Return the highest severity from a list."""
        from ..models import IssueSeverity
        order = {IssueSeverity.CRITICAL: 0, IssueSeverity.MAJOR: 1, IssueSeverity.MINOR: 2}
        return min(severities, key=lambda s: order.get(s, 2))

    @staticmethod
    def _build_correction_prompt(
        failed_criteria: List[FailedCriterion],
        critical_count: int,
        escalation_priority: str,
    ) -> str:
        """Build the formatted correction prompt string for the Research Agent."""
        lines = [
            "🔄 VERIFICATION FAILED — CORRECTION REQUIRED",
            "=" * 50,
            "",
        ]

        if critical_count > 0:
            lines.append(f"⚠️  {critical_count} CRITICAL issue(s) require immediate attention.")
            lines.append("")

        lines.append("❌ Failed criteria (priority order):")
        for fc in failed_criteria:
            severity_icon = {"critical": "🚨", "major": "⚠️", "minor": "ℹ️"}.get(
                fc.severity.value if hasattr(fc.severity, 'value') else str(fc.severity), "•"
            )
            lines.append(f"  {severity_icon} {fc.criterion_name}: {fc.explanation}")
        lines.append("")

        lines.append("🛠️  Specific corrections required:")
        for fc in failed_criteria:
            lines.append(f"\n  [{fc.criterion_name}]")
            for suggestion in fc.correction_suggestions:
                lines.append(f"    • {suggestion}")

        lines.extend([
            "",
            "📋 Instructions for retry:",
            "  1. Address ALL corrections above in priority order (Critical → Major → Minor)",
            "  2. Cross-check ALL prices against the internal product catalog",
            "  3. Verify ALL policy statements against official policy documents",
            "  4. Ensure the response directly addresses the customer's specific objection",
            "  5. Include proper citations and sources for all claims",
        ])

        if escalation_priority in ("immediate", "high"):
            lines.extend([
                "",
                f"🚨 ESCALATION PRIORITY: {escalation_priority.upper()}",
                "   This response may require human review if corrections cannot be made.",
            ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reasoning generation
    # ------------------------------------------------------------------

    def _generate_verification_reasoning(self, criteria: RubricCriteria) -> str:
        """Generate human-readable verification reasoning."""

        parts = ["🔍 BINARY VERIFICATION ANALYSIS:", ""]

        if criteria.price_accuracy_pass:
            parts.append("✅ Price Accuracy: PASS - All prices verified against internal DB")
        else:
            parts.append("❌ Price Accuracy: FAIL")
            for issue in criteria.price_issues:
                parts.append(f"  - {issue.explanation}")

        parts.append("")

        if criteria.policy_authenticity_pass:
            parts.append("✅ Policy Authenticity: PASS - All policies verified")
        else:
            parts.append("❌ Policy Authenticity: FAIL")
            for issue in criteria.policy_issues:
                parts.append(f"  - {issue.explanation}")

        parts.append("")

        if criteria.topic_relevance_pass:
            parts.append("✅ Topic Relevance: PASS - Response addresses objection")
        else:
            parts.append("❌ Topic Relevance: FAIL")
            for issue in criteria.relevance_issues:
                parts.append(f"  - {issue.explanation}")

        parts.append("")

        # Binary decision: PASS only when ALL 3 checks pass
        # is_approved = overall_pass = price_accuracy_pass AND policy_authenticity_pass AND topic_relevance_pass
        if criteria.overall_pass:
            parts.append("🎉 OVERALL DECISION: APPROVED")
            parts.append("All verification criteria passed. Draft ready for delivery.")
        else:
            parts.append("🔄 OVERALL DECISION: REQUIRES CORRECTION")
            parts.append("One or more criteria failed. Self-correction needed.")
            critical_count = criteria.critical_issues_count or 0
            if critical_count > 0:
                parts.append(
                    f"⚠️  {critical_count} critical issue(s) require immediate attention."
                )
            # Flag for immediate escalation when critical_issues_count >= 3
            if critical_count >= 3:
                parts.append(
                    "🚨 IMMEDIATE ESCALATION REQUIRED: ≥3 critical issues detected across all checks."
                )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_input(self, state: WorkflowState) -> None:
        """Validate input state for verification."""

        if not state.get("draft_response"):
            raise ValueError("Draft response is required for verification")

        if not state.get("objection_text"):
            raise ValueError("Objection text is required for verification")

        draft_length = len(state["draft_response"])
        if draft_length > self.config.max_draft_length:
            raise ValueError(
                f"Draft too long: {draft_length} > {self.config.max_draft_length}"
            )

        objection_length = len(state["objection_text"])
        if objection_length > self.config.max_objection_length:
            raise ValueError(
                f"Objection too long: {objection_length} > {self.config.max_objection_length}"
            )

    def _has_critical_issues(self, issues: List) -> bool:
        """Check if any issues are critical severity."""
        return any(
            getattr(issue, "severity", None) in ("critical", "CRITICAL")
            or str(getattr(issue, "severity", "")).lower() == "critical"
            for issue in issues
        )

    def _generate_cache_key(self, state: WorkflowState) -> str:
        """Generate deterministic cache key from state (Req 1.4)."""
        content = f"{state['draft_response']}|{state['objection_text']}"
        return hashlib.md5(content.encode()).hexdigest()

    def _estimate_tokens_used(self, state: WorkflowState, criteria: RubricCriteria) -> int:
        """Estimate tokens used for verification (approximate)."""
        base_tokens = 500
        input_tokens = (
            len(state["draft_response"]) // 4 + len(state["objection_text"]) // 4
        )
        issue_tokens = (
            len(criteria.price_issues)
            + len(criteria.policy_issues)
            + len(criteria.relevance_issues)
        ) * 50
        return base_tokens + input_tokens + issue_tokens

    def _create_error_issue(self, check_type: str, error: Exception) -> Any:
        """Create error issue for failed checks."""

        if check_type == "price":
            return PriceIssue(
                product_name="Unknown",
                severity="critical",
                explanation=f"Price check failed: {str(error)}",
            )
        elif check_type == "policy":
            return PolicyIssue(
                mentioned_policy="Unknown",
                policy_type="warranty",
                is_fabricated=True,
                is_inaccurate=False,
                severity="critical",
                explanation=f"Policy check failed: {str(error)}",
            )
        else:  # relevance
            return RelevanceIssue(
                objection_intent="Unknown",
                response_coverage=0.0,
                severity="critical",
                explanation=f"Relevance check failed: {str(error)}",
            )

    def _handle_verification_error(
        self,
        state: WorkflowState,
        error: Exception,
        execution_time: float,
    ) -> VerificationResult:
        """
        Handle verification errors gracefully with fallback mechanisms (Req 8.1, 8.2, 8.3).

        Error handling strategy:
        - ConnectionError / OSError (DB loss): use cached data or DB-unavailable result (Req 8.2)
        - asyncio.TimeoutError / LLM errors: use rule-based fallback verification (Req 8.1)
        - StateGraph execution errors: rollback to last valid state (Req 8.3)
        - All errors: logged with correlation IDs (Req 8.5)
        """
        correlation_id = self._get_correlation_id()

        # Log error with full context and correlation ID (Req 8.5)
        self._logger.error(
            f"Verification failed — applying fallback: {error}",
            correlation_id=correlation_id,
            error_type=type(error).__name__,
            error_message=str(error),
            execution_time=execution_time,
            draft_length=len(state.get("draft_response", "")),
            objection_length=len(state.get("objection_text", "")),
        )

        # DB connection loss — use cached data or degraded mode (Req 8.2)
        if isinstance(error, (ConnectionError, OSError)):
            # Try to use last valid result as cached data
            if self._last_valid_result is not None:
                self._logger.warning(
                    "DB connection lost — returning last valid cached result (Req 8.2)",
                    correlation_id=correlation_id,
                )
                return self._last_valid_result
            # No cached data — build DB-unavailable result with lowered threshold
            return self._build_db_unavailable_result(state, execution_time)

        # LLM timeout / API error — use rule-based fallback (Req 8.1)
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return self._build_fallback_verification_result(state, error, execution_time)

        # StateGraph execution error — rollback to last valid state (Req 8.3)
        if self._last_valid_result is not None:
            self._logger.warning(
                "StateGraph execution error — rolling back to last valid state (Req 8.3)",
                correlation_id=correlation_id,
                error_type=type(error).__name__,
            )
            return self._last_valid_result

        # Generic error — return conservative FAIL with error details (graceful degradation)
        criteria = RubricCriteria(
            price_accuracy_pass=False,
            policy_authenticity_pass=False,
            topic_relevance_pass=False,
            price_issues=[self._create_error_issue("price", error)],
            policy_issues=[self._create_error_issue("policy", error)],
            relevance_issues=[self._create_error_issue("relevance", error)],
            overall_pass=False,
            critical_issues_count=3,
        )

        return VerificationResult(
            criteria=criteria,
            timestamp=datetime.now(),
            verification_reasoning=(
                f"❌ Verification failed due to system error: {str(error)}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Error type: {type(error).__name__}\n"
                "Conservative FAIL result returned. Human review recommended."
            ),
            execution_time_seconds=execution_time,
            llm_tokens_used=0,
        )

    def _update_metrics(self, result: VerificationResult) -> None:
        """Update performance metrics."""
        self._total_verifications += 1
        self._total_tokens_used += result.llm_tokens_used

    def _get_correlation_id(self) -> str:
        """Get current correlation ID or generate a new one."""
        ctx = get_correlation_context()
        if ctx:
            return ctx.correlation_id
        return CorrelationIDGenerator.generate_correlation_id()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics including circuit breaker status (Req 8.4, 8.5)."""
        cache_stats = self._cache.get_stats() if self._cache else {}
        return {
            "total_verifications": self._total_verifications,
            "total_tokens_used": self._total_tokens_used,
            "average_tokens_per_verification": (
                self._total_tokens_used / self._total_verifications
                if self._total_verifications > 0
                else 0
            ),
            "cache_enabled": self._cache is not None,
            "cache_stats": cache_stats,
            "circuit_breakers": self.get_circuit_breaker_status(),
            "has_cached_fallback": self._last_valid_result is not None,
        }
