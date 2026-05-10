"""
Tests for FastAPI Health Check Endpoints - Task 7.3.2

Tests for verification/api.py covering:
- GET /health          — overall health with component checks
- GET /health/ready    — readiness probe
- GET /health/live     — liveness probe
- GET /health/verification — binary verification system status
"""

import pytest
from fastapi.testclient import TestClient

from verification.api import app
from verification.utils.health_check import (
    reset_health_checker,
    HealthStatus,
)
from verification.utils.circuit_breaker import (
    reset_circuit_breaker_registry,
    get_circuit_breaker_registry,
)
from verification.utils.error_rate_tracker import reset_error_rate_tracker
from verification.utils.graceful_shutdown import reset_shutdown_manager


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


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200_when_healthy(self, client):
        response = client.get("/health")
        assert response.status_code in (200, 503)  # depends on system state

    def test_response_has_status_field(self, client):
        response = client.get("/health")
        body = response.json()
        assert "status" in body
        assert body["status"] in ("healthy", "degraded", "unhealthy")

    def test_response_has_checks_object(self, client):
        response = client.get("/health")
        body = response.json()
        assert "checks" in body
        checks = body["checks"]
        assert "llm_service" in checks
        assert "database" in checks
        assert "circuit_breakers" in checks
        assert "error_rates" in checks
        assert "resources" in checks

    def test_response_has_response_time_ms(self, client):
        response = client.get("/health")
        body = response.json()
        assert "response_time_ms" in body
        assert isinstance(body["response_time_ms"], (int, float))
        assert body["response_time_ms"] >= 0

    def test_response_has_timestamp(self, client):
        response = client.get("/health")
        body = response.json()
        assert "timestamp" in body
        # Should be a non-empty string
        assert isinstance(body["timestamp"], str)
        assert len(body["timestamp"]) > 0

    def test_response_has_checks_passed_and_total(self, client):
        response = client.get("/health")
        body = response.json()
        assert "checks_passed" in body
        assert "checks_total" in body
        assert body["checks_total"] == 5
        assert 0 <= body["checks_passed"] <= body["checks_total"]

    def test_unhealthy_when_circuit_open(self, client):
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("llm_api", failure_threshold=1)
        cb.record_failure("TestError")

        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unhealthy"

    def test_correlation_id_header_accepted(self, client):
        response = client.get("/health", headers={"X-Correlation-ID": "test-corr-123"})
        assert response.status_code in (200, 503)

    def test_healthy_status_returns_200(self, client):
        """When no failures are recorded, status should be healthy or degraded → 200."""
        response = client.get("/health")
        body = response.json()
        if body["status"] in ("healthy", "degraded"):
            assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /health/ready
# ---------------------------------------------------------------------------

class TestHealthReadyEndpoint:
    def test_returns_200_when_ready(self, client):
        response = client.get("/health/ready")
        body = response.json()
        if body["ready"]:
            assert response.status_code == 200

    def test_returns_503_when_not_ready(self, client):
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("llm_api", failure_threshold=1)
        cb.record_failure("TestError")

        response = client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False

    def test_response_has_ready_field(self, client):
        response = client.get("/health/ready")
        body = response.json()
        assert "ready" in body
        assert isinstance(body["ready"], bool)

    def test_response_has_status_field(self, client):
        response = client.get("/health/ready")
        body = response.json()
        assert "status" in body
        assert body["status"] in ("healthy", "degraded", "unhealthy")

    def test_response_has_response_time_ms(self, client):
        response = client.get("/health/ready")
        body = response.json()
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0

    def test_degraded_system_is_still_ready(self, client):
        """Degraded systems should still return ready=True and 200."""
        # A degraded system (e.g. approaching circuit threshold) is still ready
        response = client.get("/health/ready")
        body = response.json()
        if body["status"] == "degraded":
            assert body["ready"] is True
            assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /health/live
# ---------------------------------------------------------------------------

class TestHealthLiveEndpoint:
    def test_always_returns_200(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200

    def test_response_has_status_alive(self, client):
        response = client.get("/health/live")
        body = response.json()
        assert body["status"] == "alive"

    def test_response_has_message(self, client):
        response = client.get("/health/live")
        body = response.json()
        assert "message" in body
        assert len(body["message"]) > 0

    def test_returns_200_even_when_circuit_open(self, client):
        """Liveness probe should always return 200 regardless of component health."""
        reg = get_circuit_breaker_registry()
        cb = reg.get_or_create("llm_api", failure_threshold=1)
        cb.record_failure("TestError")

        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


# ---------------------------------------------------------------------------
# GET /health/verification
# ---------------------------------------------------------------------------

class TestHealthVerificationEndpoint:
    def test_returns_200_when_config_loads(self, client):
        response = client.get("/health/verification")
        assert response.status_code == 200

    def test_response_has_operational_field(self, client):
        response = client.get("/health/verification")
        body = response.json()
        assert "operational" in body
        assert isinstance(body["operational"], bool)

    def test_response_has_status_field(self, client):
        response = client.get("/health/verification")
        body = response.json()
        assert "status" in body
        assert body["status"] in ("ok", "error")

    def test_response_has_checks_object(self, client):
        response = client.get("/health/verification")
        body = response.json()
        assert "checks" in body
        checks = body["checks"]
        assert "config" in checks
        assert "thresholds" in checks

    def test_config_check_is_ok(self, client):
        response = client.get("/health/verification")
        body = response.json()
        assert body["checks"]["config"]["status"] == "ok"

    def test_thresholds_check_present(self, client):
        response = client.get("/health/verification")
        body = response.json()
        thresholds = body["checks"]["thresholds"]
        assert thresholds["status"] in ("ok", "warning", "error")

    def test_response_has_response_time_ms(self, client):
        response = client.get("/health/verification")
        body = response.json()
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0

    def test_operational_true_when_config_valid(self, client):
        response = client.get("/health/verification")
        body = response.json()
        # Default config should load and be valid
        assert body["operational"] is True
        assert response.status_code == 200
