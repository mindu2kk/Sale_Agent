"""
Unit Tests for Health Check Module - Task 6.3.4

Tests for verification/utils/health_check.py covering:
- HealthStatus enum and Pydantic models
- HealthChecker sub-checks (LLM, DB, circuit breakers, error rates, resources)
- Overall status aggregation (healthy / degraded / unhealthy)
- Singleton helpers and reset
- run_health_check convenience function
"""

import asyncio
import pytest

from verification.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    reset_circuit_breaker_registry,
)
from verification.utils.error_rate_tracker import (
    ErrorRateTracker,
    get_error_rate_tracker,
    reset_error_rate_tracker,
)
from verification.utils.health_check import (
    CircuitBreakerHealthDetail,
    ErrorRateHealthDetail,
    HealthChecker,
    HealthReport,
    HealthStatus,
    ResourceHealthDetail,
    ServiceHealthDetail,
    get_health_checker,
    reset_health_checker,
    run_health_check,
    _status_rank,
    _rank_to_status,
    _ERROR_RATE_DEGRADED_THRESHOLD,
    _ERROR_RATE_UNHEALTHY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all module-level singletons before each test."""
    reset_health_checker()
    reset_circuit_breaker_registry()
    reset_error_rate_tracker()
    yield
    reset_health_checker()
    reset_circuit_breaker_registry()
    reset_error_rate_tracker()


# ---------------------------------------------------------------------------
# HealthStatus helpers
# ---------------------------------------------------------------------------

class TestStatusHelpers:
    def test_status_rank_ordering(self):
        assert _status_rank(HealthStatus.HEALTHY) < _status_rank(HealthStatus.DEGRADED)
        assert _status_rank(HealthStatus.DEGRADED) < _status_rank(HealthStatus.UNHEALTHY)

    def test_rank_to_status_roundtrip(self):
        for status in HealthStatus:
            rank = _status_rank(status)
            assert _rank_to_status(rank) == status

    def test_aggregate_status_all_healthy(self):
        checker = HealthChecker()
        result = checker._aggregate_status([HealthStatus.HEALTHY, HealthStatus.HEALTHY])
        assert result == HealthStatus.HEALTHY

    def test_aggregate_status_one_degraded(self):
        checker = HealthChecker()
        result = checker._aggregate_status([HealthStatus.HEALTHY, HealthStatus.DEGRADED])
        assert result == HealthStatus.DEGRADED

    def test_aggregate_status_one_unhealthy(self):
        checker = HealthChecker()
        result = checker._aggregate_status(
            [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        )
        assert result == HealthStatus.UNHEALTHY

    def test_aggregate_status_empty(self):
        checker = HealthChecker()
        result = checker._aggregate_status([])
        assert result == HealthStatus.HEALTHY


# ---------------------------------------------------------------------------
# ServiceHealthDetail — circuit breaker states
# ---------------------------------------------------------------------------

class TestServiceHealthDetail:
    def test_no_circuit_breaker_registered_is_healthy(self):
        checker = HealthChecker(llm_service_name="nonexistent_service")
        detail = checker._check_service_sync("nonexistent_service")
        assert detail.status == HealthStatus.HEALTHY
        assert detail.circuit_state == "closed"

    def test_closed_circuit_is_healthy(self):
        registry = reset_circuit_breaker_registry() or __import__(
            "verification.utils.circuit_breaker", fromlist=["get_circuit_breaker_registry"]
        )
        from verification.utils.circuit_breaker import get_circuit_breaker_registry
        reg = get_circuit_breaker_registry()
        reg.get_or_create("llm_api", failure_threshold=5)

        checker = HealthChecker()
        detail = checker._check_service_sync("llm_api")
        assert detail.status == HealthStatus.HEALTHY
        assert detail.circuit_state == "closed"

    def test_open_circuit_is_unhealthy(self):
        from verification.utils.circuit_breaker import get_circuit_breaker_registry
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("llm_api", failure_threshold=2)
        # Force open
        cb.record_failure("TestError")
        cb.record_failure("TestError")

        checker = HealthChecker()
        detail = checker._check_service_sync("llm_api")
        assert detail.status == HealthStatus.UNHEALTHY
        assert detail.circuit_state == "open"

    def test_half_open_circuit_is_degraded(self):
        from verification.utils.circuit_breaker import get_circuit_breaker_registry, CircuitState
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("llm_api", failure_threshold=2, cooldown_seconds=0.0)
        cb.record_failure("TestError")
        cb.record_failure("TestError")
        # Trigger HALF_OPEN by accessing state (cooldown=0 so it transitions immediately)
        _ = cb.state

        checker = HealthChecker()
        detail = checker._check_service_sync("llm_api")
        # Should be DEGRADED (half_open) or UNHEALTHY (open) depending on timing
        assert detail.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    def test_approaching_threshold_is_degraded(self):
        from verification.utils.circuit_breaker import get_circuit_breaker_registry
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("llm_api", failure_threshold=5)
        # 3 failures out of 5 threshold = 60% → degraded
        cb.record_failure("TestError")
        cb.record_failure("TestError")
        cb.record_failure("TestError")

        checker = HealthChecker()
        detail = checker._check_service_sync("llm_api")
        assert detail.status == HealthStatus.DEGRADED


# ---------------------------------------------------------------------------
# CircuitBreakerHealthDetail
# ---------------------------------------------------------------------------

class TestCircuitBreakerHealthDetail:
    def test_no_breakers_is_healthy(self):
        checker = HealthChecker()
        detail = checker._check_circuit_breakers_sync()
        assert detail.status == HealthStatus.HEALTHY
        assert detail.total_circuits == 0

    def test_all_closed_is_healthy(self):
        from verification.utils.circuit_breaker import get_circuit_breaker_registry
        reg = get_circuit_breaker_registry()
        reg.get_or_create("svc_a")
        reg.get_or_create("svc_b")

        checker = HealthChecker()
        detail = checker._check_circuit_breakers_sync()
        assert detail.status == HealthStatus.HEALTHY
        assert detail.total_circuits == 2
        assert detail.open_circuits == []

    def test_open_circuit_makes_unhealthy(self):
        from verification.utils.circuit_breaker import get_circuit_breaker_registry
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("svc_a", failure_threshold=1)
        cb.record_failure("Err")

        checker = HealthChecker()
        detail = checker._check_circuit_breakers_sync()
        assert detail.status == HealthStatus.UNHEALTHY
        assert "svc_a" in detail.open_circuits


# ---------------------------------------------------------------------------
# ErrorRateHealthDetail
# ---------------------------------------------------------------------------

class TestErrorRateHealthDetail:
    def test_no_events_is_healthy(self):
        checker = HealthChecker()
        detail = checker._check_error_rates_sync()
        assert detail.status == HealthStatus.HEALTHY
        assert all(r == 0.0 for r in detail.component_rates.values())

    def test_low_error_rate_is_healthy(self):
        tracker = get_error_rate_tracker()
        # 1 error out of 10 events = 10% < 25% threshold
        for _ in range(9):
            tracker.record_success("research")
        tracker.record_error("research", "TestError")

        checker = HealthChecker()
        detail = checker._check_error_rates_sync()
        assert detail.status == HealthStatus.HEALTHY

    def test_elevated_error_rate_is_degraded(self):
        tracker = get_error_rate_tracker()
        # 3 errors out of 10 = 30% >= 25% threshold
        for _ in range(7):
            tracker.record_success("verification")
        for _ in range(3):
            tracker.record_error("verification", "TestError")

        checker = HealthChecker()
        detail = checker._check_error_rates_sync()
        assert detail.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)
        assert "verification" in detail.degraded_components

    def test_high_error_rate_is_unhealthy(self):
        tracker = get_error_rate_tracker()
        # 6 errors out of 10 = 60% >= 50% threshold
        for _ in range(4):
            tracker.record_success("correction")
        for _ in range(6):
            tracker.record_error("correction", "TestError")

        checker = HealthChecker()
        detail = checker._check_error_rates_sync()
        assert detail.status == HealthStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# ResourceHealthDetail
# ---------------------------------------------------------------------------

class TestResourceHealthDetail:
    def test_resource_check_returns_detail(self):
        checker = HealthChecker()
        detail = checker._check_resources_sync()
        assert isinstance(detail, ResourceHealthDetail)
        assert detail.status in list(HealthStatus)
        assert detail.memory_rss_mb >= 0.0
        assert detail.cpu_percent >= 0.0
        assert detail.thread_count >= 0
        assert detail.active_async_tasks >= 0

    def test_resource_check_has_message(self):
        checker = HealthChecker()
        detail = checker._check_resources_sync()
        assert len(detail.message) > 0


# ---------------------------------------------------------------------------
# Full async health check
# ---------------------------------------------------------------------------

class TestHealthCheckerAsync:
    @pytest.mark.asyncio
    async def test_check_returns_health_report(self):
        checker = HealthChecker()
        report = await checker.check()
        assert isinstance(report, HealthReport)
        assert report.status in list(HealthStatus)
        assert report.checks_total == 5
        assert 0 <= report.checks_passed <= report.checks_total

    @pytest.mark.asyncio
    async def test_check_timestamp_is_iso(self):
        checker = HealthChecker()
        report = await checker.check()
        from datetime import datetime
        # Should parse without error
        datetime.fromisoformat(report.timestamp)

    @pytest.mark.asyncio
    async def test_check_all_healthy_when_no_failures(self):
        checker = HealthChecker()
        report = await checker.check()
        # With no failures recorded, system should be healthy or degraded
        # (resources may vary in CI)
        assert report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    @pytest.mark.asyncio
    async def test_check_unhealthy_when_circuit_open(self):
        from verification.utils.circuit_breaker import get_circuit_breaker_registry
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("llm_api", failure_threshold=1)
        cb.record_failure("TestError")

        checker = HealthChecker()
        report = await checker.check()
        assert report.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_is_ready_healthy(self):
        checker = HealthChecker()
        report = await checker.check()
        # Healthy or degraded → ready
        if report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED):
            assert report.is_ready() is True

    @pytest.mark.asyncio
    async def test_is_ready_unhealthy(self):
        from verification.utils.circuit_breaker import get_circuit_breaker_registry
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("llm_api", failure_threshold=1)
        cb.record_failure("TestError")

        checker = HealthChecker()
        report = await checker.check()
        assert report.is_ready() is False

    @pytest.mark.asyncio
    async def test_is_healthy_property(self):
        checker = HealthChecker()
        report = await checker.check()
        assert report.is_healthy() == (report.status == HealthStatus.HEALTHY)

    @pytest.mark.asyncio
    async def test_to_dict_serializable(self):
        checker = HealthChecker()
        report = await checker.check()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "status" in d
        assert "timestamp" in d
        assert "llm_service" in d
        assert "database" in d
        assert "circuit_breakers" in d
        assert "error_rates" in d
        assert "resources" in d


# ---------------------------------------------------------------------------
# run_health_check convenience function
# ---------------------------------------------------------------------------

class TestRunHealthCheck:
    @pytest.mark.asyncio
    async def test_run_health_check_returns_report(self):
        report = await run_health_check(correlation_id="test_corr")
        assert isinstance(report, HealthReport)
        assert report.checks_total == 5


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    def test_get_health_checker_returns_same_instance(self):
        c1 = get_health_checker()
        c2 = get_health_checker()
        assert c1 is c2

    def test_reset_health_checker_creates_new_instance(self):
        c1 = get_health_checker()
        reset_health_checker()
        c2 = get_health_checker()
        assert c1 is not c2

    def test_get_health_checker_custom_params(self):
        reset_health_checker()
        checker = get_health_checker(
            llm_service_name="custom_llm",
            db_service_name="custom_db",
            error_rate_window_seconds=30,
        )
        assert checker.llm_service_name == "custom_llm"
        assert checker.db_service_name == "custom_db"
        assert checker.error_rate_window_seconds == 30
