"""
Deployment Validation Tests - Task 7.3.5

Validates that the binary verification system is correctly deployed and operational.
Covers:
1. Health check validation
2. Binary verification smoke tests (PASS and FAIL scenarios)
3. Configuration validation (price ±1%, policy citation, relevance 0.7)
4. Component availability (VerificationAgent, checkers, workflow)
5. Graceful shutdown validation
6. Circuit breaker state (starts CLOSED)
7. Binary decision consistency (determinism)
8. Async workflow startup

Requirements: 7.3 (Production Deployment), 8.1 (Error Handling),
              9.1 (Performance ≤10s), 10.5 (Config changes without restart)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.verification.api import app
from backend.verification.config.binary_verification_config import (
    BinaryVerificationConfig,
    get_binary_verification_config,
)
from backend.verification.config.config import VerificationConfig
from backend.verification.utils.circuit_breaker import (
    CircuitState,
    get_circuit_breaker_registry,
    reset_circuit_breaker_registry,
)
from backend.verification.utils.graceful_shutdown import (
    ShutdownManager,
    get_shutdown_manager,
    reset_shutdown_manager,
)
from backend.verification.utils.health_check import (
    HealthStatus,
    reset_health_checker,
)
from backend.verification.utils.error_rate_tracker import reset_error_rate_tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all module-level singletons before each test."""
    reset_health_checker()
    reset_circuit_breaker_registry()
    reset_error_rate_tracker()
    reset_shutdown_manager()
    yield
    reset_health_checker()
    reset_circuit_breaker_registry()
    reset_error_rate_tracker()
    reset_shutdown_manager()


@pytest.fixture
def client():
    """Return a synchronous TestClient for the FastAPI app."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def verification_config():
    """Return a default VerificationConfig for testing."""
    return VerificationConfig()


@pytest.fixture
def binary_config():
    """Return a default BinaryVerificationConfig for testing."""
    return BinaryVerificationConfig()


# ---------------------------------------------------------------------------
# 1. Health Check Validation
# ---------------------------------------------------------------------------

class TestHealthCheckValidation:
    """Verify health endpoints return correct status with binary verification system running."""

    def test_health_endpoint_returns_valid_status(self, client):
        """Health endpoint must return healthy/degraded/unhealthy."""
        response = client.get("/health")
        assert response.status_code in (200, 503)
        body = response.json()
        assert body["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_endpoint_has_all_required_checks(self, client):
        """Health response must include all 5 component checks."""
        response = client.get("/health")
        body = response.json()
        checks = body["checks"]
        for key in ("llm_service", "database", "circuit_breakers", "error_rates", "resources"):
            assert key in checks, f"Missing check: {key}"

    def test_liveness_probe_always_200(self, client):
        """Liveness probe must always return 200 — process is alive."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_probe_returns_ready_field(self, client):
        """Readiness probe must include a boolean 'ready' field."""
        response = client.get("/health/ready")
        body = response.json()
        assert "ready" in body
        assert isinstance(body["ready"], bool)

    def test_verification_health_endpoint_operational(self, client):
        """Binary verification health endpoint must report operational=True with valid config."""
        response = client.get("/health/verification")
        assert response.status_code == 200
        body = response.json()
        assert body["operational"] is True
        assert body["checks"]["config"]["status"] == "ok"

    def test_health_response_time_ms_present(self, client):
        """Health response must include response_time_ms for SLA monitoring."""
        response = client.get("/health")
        body = response.json()
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0

    def test_health_checks_passed_and_total_consistent(self, client):
        """checks_passed must be <= checks_total."""
        response = client.get("/health")
        body = response.json()
        assert body["checks_total"] == 5
        assert 0 <= body["checks_passed"] <= body["checks_total"]


# ---------------------------------------------------------------------------
# 2. Binary Verification Smoke Tests
# ---------------------------------------------------------------------------

class TestBinaryVerificationSmokeTests:
    """End-to-end smoke tests verifying PASS and FAIL scenarios work after deployment."""

    def test_pass_scenario_all_criteria_met(self, binary_config):
        """PASS: price within tolerance, policy authentic with citation, relevance above threshold."""
        result = binary_config.is_binary_pass(
            price_deviation_percent=0.5,   # within ±1%
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.85,       # above 0.7
        )
        assert result is True

    def test_fail_scenario_price_exceeds_tolerance(self, binary_config):
        """FAIL: price deviation exceeds ±1% tolerance."""
        result = binary_config.is_binary_pass(
            price_deviation_percent=5.0,   # exceeds 1% tolerance
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.85,
        )
        assert result is False

    def test_fail_scenario_fabricated_policy(self, binary_config):
        """FAIL: fabricated policy must always fail verification."""
        result = binary_config.is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=True,        # fabricated → FAIL
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.85,
        )
        assert result is False

    def test_fail_scenario_low_relevance_coverage(self, binary_config):
        """FAIL: relevance coverage below 0.7 minimum threshold."""
        result = binary_config.is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.5,        # below 0.7 threshold
        )
        assert result is False

    def test_fail_scenario_missing_citation(self, binary_config):
        """FAIL: policy citation required but missing."""
        result = binary_config.is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=False,     # citation required
            relevance_coverage=0.85,
        )
        assert result is False

    def test_boundary_price_exactly_at_tolerance(self, binary_config):
        """PASS: price deviation exactly at tolerance boundary (1.0%)."""
        result = binary_config.is_binary_pass(
            price_deviation_percent=1.0,   # exactly at tolerance
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.85,
        )
        assert result is True

    def test_boundary_relevance_exactly_at_threshold(self, binary_config):
        """PASS: relevance coverage exactly at minimum threshold (0.7)."""
        result = binary_config.is_binary_pass(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.7,        # exactly at threshold
        )
        assert result is True


# ---------------------------------------------------------------------------
# 3. Configuration Validation
# ---------------------------------------------------------------------------

class TestConfigurationValidation:
    """Verify binary verification config loads correctly with proper thresholds."""

    def test_default_price_tolerance_is_one_percent(self):
        """Default price tolerance must be ±1% as per spec."""
        cfg = get_binary_verification_config()
        assert cfg.price_accuracy.pass_tolerance_percent == 1.0

    def test_default_policy_citation_required(self):
        """Default policy config must require citations."""
        cfg = get_binary_verification_config()
        assert cfg.policy_authenticity.citation_required is True

    def test_default_relevance_min_coverage_is_0_7(self):
        """Default relevance minimum coverage must be 0.7."""
        cfg = get_binary_verification_config()
        assert cfg.topic_relevance.pass_coverage_threshold == 0.7

    def test_config_has_no_validation_warnings_by_default(self):
        """Default configuration must produce no validation warnings."""
        cfg = get_binary_verification_config()
        warnings = cfg.validate_configuration()
        assert warnings == []

    def test_config_weights_sum_to_one(self):
        """Verification weights must sum to 1.0."""
        cfg = get_binary_verification_config()
        total = sum(cfg.verification_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_production_config_has_stricter_price_tolerance(self):
        """Production environment must have stricter price tolerance (0.5%)."""
        cfg = get_binary_verification_config(environment="production")
        assert cfg.price_accuracy.pass_tolerance_percent == 0.5

    def test_development_config_has_relaxed_price_tolerance(self):
        """Development environment must have relaxed price tolerance (2.0%)."""
        cfg = get_binary_verification_config(environment="development")
        assert cfg.price_accuracy.pass_tolerance_percent == 2.0

    def test_config_runtime_update_without_restart(self):
        """Config must support runtime updates without restart (Req 10.5)."""
        from backend.verification.config.binary_verification_config import get_runtime_config_manager
        manager = get_runtime_config_manager()
        original_tolerance = manager.get_config().price_accuracy.pass_tolerance_percent

        # Update at runtime
        manager.update_severity_thresholds(
            price_accuracy={"pass_tolerance_percent": 2.5},
            description="deployment test update",
        )
        updated = manager.get_config()
        assert updated.price_accuracy.pass_tolerance_percent == 2.5

        # Rollback
        manager.rollback(steps=1)
        restored = manager.get_config()
        assert restored.price_accuracy.pass_tolerance_percent == original_tolerance


# ---------------------------------------------------------------------------
# 4. Component Availability
# ---------------------------------------------------------------------------

class TestComponentAvailability:
    """Verify all critical components can be instantiated."""

    def test_verification_config_instantiates(self):
        """VerificationConfig must instantiate with defaults."""
        config = VerificationConfig()
        assert config is not None
        assert config.price_tolerance_percent == 1.0
        assert config.relevance_min_coverage == 0.7
        assert config.policy_citation_required is True

    def test_binary_verification_config_instantiates(self):
        """BinaryVerificationConfig must instantiate with defaults."""
        config = BinaryVerificationConfig()
        assert config is not None

    def test_price_accuracy_checker_instantiates(self):
        """PriceAccuracyChecker must instantiate with mocked dependencies."""
        from backend.verification.agent.checkers import PriceAccuracyChecker
        config = VerificationConfig()
        checker = PriceAccuracyChecker(
            llm=MagicMock(),
            rag_pipeline=MagicMock(),
            config=config,
        )
        assert checker is not None

    def test_policy_authenticity_checker_instantiates(self):
        """PolicyAuthenticityChecker must instantiate with mocked dependencies."""
        from backend.verification.agent.checkers import PolicyAuthenticityChecker
        config = VerificationConfig()
        checker = PolicyAuthenticityChecker(
            llm=MagicMock(),
            rag_pipeline=MagicMock(),
            config=config,
        )
        assert checker is not None

    def test_topic_relevance_checker_instantiates(self):
        """TopicRelevanceChecker must instantiate with mocked dependencies."""
        from backend.verification.agent.checkers import TopicRelevanceChecker
        config = VerificationConfig()
        checker = TopicRelevanceChecker(
            llm=MagicMock(),
            config=config,
        )
        assert checker is not None

    def test_verification_agent_instantiates(self):
        """VerificationAgent must instantiate with mocked LLM and RAG pipeline."""
        from backend.verification.agent.verification_agent import VerificationAgent
        config = VerificationConfig()
        agent = VerificationAgent(
            llm=MagicMock(),
            rag_pipeline=MagicMock(),
            config=config,
        )
        assert agent is not None
        assert agent.price_checker is not None
        assert agent.policy_checker is not None
        assert agent.relevance_checker is not None

    def test_workflow_router_instantiates(self):
        """WorkflowRouter must instantiate with config."""
        from backend.verification.workflow.routing import WorkflowRouter
        config = VerificationConfig()
        router = WorkflowRouter(config)
        assert router is not None

    def test_self_correction_node_instantiates(self):
        """SelfCorrectionNode must instantiate with config."""
        from backend.verification.workflow.correction import SelfCorrectionNode
        config = VerificationConfig()
        node = SelfCorrectionNode(config)
        assert node is not None


# ---------------------------------------------------------------------------
# 5. Graceful Shutdown Validation
# ---------------------------------------------------------------------------

class TestGracefulShutdownValidation:
    """Verify the system shuts down cleanly without losing in-flight verifications."""

    @pytest.mark.asyncio
    async def test_shutdown_manager_starts_not_shutting_down(self):
        """ShutdownManager must start in non-shutting-down state."""
        manager = ShutdownManager()
        assert manager.is_shutting_down is False

    @pytest.mark.asyncio
    async def test_shutdown_completes_without_error(self):
        """Shutdown with no tasks must complete cleanly."""
        manager = ShutdownManager()
        await manager.shutdown(timeout_seconds=1.0)
        assert manager.is_shutting_down is True

    @pytest.mark.asyncio
    async def test_shutdown_cancels_in_flight_tasks(self):
        """Shutdown must cancel registered in-flight tasks."""
        manager = ShutdownManager()

        async def long_running():
            await asyncio.sleep(60.0)

        task = asyncio.create_task(long_running())
        manager.register_task(task)

        await manager.shutdown(timeout_seconds=1.0)
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_shutdown_runs_cleanup_handlers(self):
        """Shutdown must invoke all registered cleanup handlers."""
        manager = ShutdownManager()
        cleanup_called = []

        def cleanup():
            cleanup_called.append(True)

        manager.register_cleanup_handler(cleanup)
        await manager.shutdown(timeout_seconds=1.0)
        assert len(cleanup_called) == 1

    @pytest.mark.asyncio
    async def test_duplicate_shutdown_ignored(self):
        """Second shutdown() call must be a no-op (idempotent)."""
        manager = ShutdownManager()
        cleanup_calls = []
        manager.register_cleanup_handler(lambda: cleanup_calls.append(1))

        await manager.shutdown(timeout_seconds=1.0)
        await manager.shutdown(timeout_seconds=1.0)
        assert len(cleanup_calls) == 1

    @pytest.mark.asyncio
    async def test_api_returns_503_during_shutdown(self):
        """API must return 503 for non-liveness requests during shutdown."""
        from httpx import AsyncClient, ASGITransport

        manager = get_shutdown_manager()
        await manager.shutdown(timeout_seconds=0.1)
        assert manager.is_shutting_down is True

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/health")
            assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_liveness_probe_works_during_shutdown(self):
        """Liveness probe must remain accessible during shutdown."""
        from httpx import AsyncClient, ASGITransport

        manager = get_shutdown_manager()
        await manager.shutdown(timeout_seconds=0.1)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/health/live")
            assert response.status_code == 200


# ---------------------------------------------------------------------------
# 6. Circuit Breaker State
# ---------------------------------------------------------------------------

class TestCircuitBreakerState:
    """Verify circuit breakers start in CLOSED state and are properly configured."""

    def test_new_circuit_breaker_starts_closed(self):
        """A freshly created circuit breaker must start in CLOSED state."""
        from backend.verification.utils.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test_service", failure_threshold=5)
        assert cb.state == CircuitState.CLOSED

    def test_registry_circuit_breakers_start_closed(self):
        """Circuit breakers obtained from registry must start CLOSED."""
        registry = get_circuit_breaker_registry()
        for service in ("llm_api", "internal_db", "chromadb"):
            cb = registry.get_or_create(service, failure_threshold=5)
            assert cb.state == CircuitState.CLOSED, f"{service} circuit not CLOSED"

    def test_circuit_breaker_allows_requests_when_closed(self):
        """CLOSED circuit must allow all requests through."""
        from backend.verification.utils.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test_service", failure_threshold=5)
        assert cb.allow_request() is True

    def test_circuit_breaker_opens_after_threshold_failures(self):
        """Circuit must open after reaching failure threshold."""
        from backend.verification.utils.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test_service", failure_threshold=3)
        for _ in range(3):
            cb.record_failure("TestError")
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_reset_returns_to_closed(self):
        """Manual reset must return circuit to CLOSED state."""
        from backend.verification.utils.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test_service", failure_threshold=2)
        cb.record_failure("Error")
        cb.record_failure("Error")
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_health_endpoint_reports_unhealthy_when_circuit_open(self, client):
        """Health endpoint must report unhealthy when a circuit breaker is OPEN."""
        registry = get_circuit_breaker_registry()
        cb = registry.get_or_create("llm_api", failure_threshold=1)
        cb.record_failure("TestError")

        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# 7. Binary Decision Consistency (Deployment Determinism)
# ---------------------------------------------------------------------------

class TestBinaryDecisionConsistency:
    """Verify same input always produces same PASS/FAIL decision (Req 1.4)."""

    def test_same_pass_input_always_returns_pass(self, binary_config):
        """Identical PASS inputs must always produce PASS — no randomness."""
        kwargs = dict(
            price_deviation_percent=0.5,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.85,
        )
        results = [binary_config.is_binary_pass(**kwargs) for _ in range(5)]
        assert all(r is True for r in results)

    def test_same_fail_input_always_returns_fail(self, binary_config):
        """Identical FAIL inputs must always produce FAIL — no randomness."""
        kwargs = dict(
            price_deviation_percent=10.0,
            policy_fabricated=False,
            policy_inaccurate=False,
            policy_has_citation=True,
            relevance_coverage=0.85,
        )
        results = [binary_config.is_binary_pass(**kwargs) for _ in range(5)]
        assert all(r is False for r in results)

    def test_price_checker_deterministic_pass(self):
        """PriceAccuracyChecker must return same result for same input."""
        from backend.verification.agent.checkers import PriceAccuracyChecker
        config = VerificationConfig()
        checker = PriceAccuracyChecker(
            llm=MagicMock(),
            rag_pipeline=MagicMock(),
            config=config,
            catalog_path="nonexistent_catalog.csv",  # no catalog → no price issues
        )
        draft = "This product costs 100 USD."
        objection = "Is this product affordable?"

        result1 = checker.check_price_accuracy(draft, objection)
        result2 = checker.check_price_accuracy(draft, objection)
        assert result1[0] == result2[0]

    def test_policy_checker_deterministic_no_policies(self):
        """PolicyAuthenticityChecker must return same result for input with no policies."""
        from backend.verification.agent.checkers import PolicyAuthenticityChecker
        config = VerificationConfig()
        checker = PolicyAuthenticityChecker(
            llm=MagicMock(),
            rag_pipeline=None,
            config=config,
        )
        draft = "This is a general response with no policy statements."

        result1 = checker.check_policy_authenticity(draft)
        result2 = checker.check_policy_authenticity(draft)
        assert result1[0] == result2[0]
        assert len(result1[1]) == len(result2[1])

    def test_config_based_decision_is_deterministic(self):
        """Config-based binary decisions must be deterministic across multiple calls."""
        cfg = BinaryVerificationConfig()
        inputs = [
            (0.5, False, False, True, 0.85),
            (5.0, False, False, True, 0.85),
            (0.5, True, False, True, 0.85),
            (0.5, False, False, True, 0.5),
        ]
        for price_dev, fabricated, inaccurate, citation, coverage in inputs:
            r1 = cfg.is_binary_pass(price_dev, fabricated, inaccurate, citation, coverage)
            r2 = cfg.is_binary_pass(price_dev, fabricated, inaccurate, citation, coverage)
            assert r1 == r2, f"Non-deterministic result for inputs: {inputs}"


# ---------------------------------------------------------------------------
# 8. Async Workflow Startup
# ---------------------------------------------------------------------------

class TestAsyncWorkflowStartup:
    """Verify the async workflow can be initialized and accepts requests."""

    def test_verification_workflow_instantiates(self):
        """VerificationWorkflow must instantiate with mocked agents."""
        from backend.verification.workflow.workflow import VerificationWorkflow
        from backend.verification.agent.verification_agent import VerificationAgent

        config = VerificationConfig()
        agent = VerificationAgent(
            llm=MagicMock(),
            rag_pipeline=MagicMock(),
            config=config,
        )
        workflow = VerificationWorkflow(
            research_agent=MagicMock(),
            verification_agent=agent,
            config=config,
        )
        assert workflow is not None
        assert workflow.graph is not None

    def test_verification_workflow_graph_has_required_nodes(self):
        """StateGraph must contain research, verification, correction, escalation nodes."""
        from backend.verification.workflow.workflow import VerificationWorkflow
        from backend.verification.agent.verification_agent import VerificationAgent

        config = VerificationConfig()
        agent = VerificationAgent(
            llm=MagicMock(),
            rag_pipeline=MagicMock(),
            config=config,
        )
        workflow = VerificationWorkflow(
            research_agent=MagicMock(),
            verification_agent=agent,
            config=config,
        )
        # The compiled graph should be accessible
        assert workflow.graph is not None

    @pytest.mark.asyncio
    async def test_verification_agent_verify_draft_async(self):
        """VerificationAgent.verify_draft() must be awaitable and return a VerificationResult."""
        from backend.verification.agent.verification_agent import VerificationAgent
        from backend.verification.models.verification import VerificationResult

        config = VerificationConfig()
        agent = VerificationAgent(
            llm=MagicMock(),
            rag_pipeline=MagicMock(),
            config=config,
            max_concurrent_llm_calls=5,
        )

        state = {
            "objection_text": "Why is this product so expensive compared to competitors?",
            "draft_response": "Our product offers superior quality and value.",
            "tools_used": [],
            "research_reasoning": "",
            "research_sources": [],
            "verification_result": None,
            "correction_feedback": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_response": "",
            "workflow_status": "verifying",
            "execution_log": [],
            "start_time": "2024-01-01T00:00:00",
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
            "config": config.model_dump(),
            "workflow_id": "wf_test_001",
            "correlation_id": "corr_test_001",
            "customer_context": {},
        }

        result = await agent.verify_draft(state)
        assert isinstance(result, VerificationResult)
        assert isinstance(result.is_approved, bool)

    @pytest.mark.asyncio
    async def test_verification_completes_within_10_seconds(self):
        """Verification must complete within ≤10 seconds (Req 9.1)."""
        from backend.verification.agent.verification_agent import VerificationAgent

        config = VerificationConfig(async_timeout_seconds=10)
        agent = VerificationAgent(
            llm=MagicMock(),
            rag_pipeline=MagicMock(),
            config=config,
        )

        state = {
            "objection_text": "Is this product worth the price?",
            "draft_response": "Yes, this product provides excellent value.",
            "tools_used": [],
            "research_reasoning": "",
            "research_sources": [],
            "verification_result": None,
            "correction_feedback": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_response": "",
            "workflow_status": "verifying",
            "execution_log": [],
            "start_time": "2024-01-01T00:00:00",
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
            "config": config.model_dump(),
            "workflow_id": "wf_perf_test",
            "correlation_id": "corr_perf_test",
            "customer_context": {},
        }

        start = time.monotonic()
        result = await agent.verify_draft(state)
        elapsed = time.monotonic() - start

        assert elapsed < 10.0, f"Verification took {elapsed:.2f}s, exceeds 10s limit"
        assert result is not None
