"""
FastAPI Application for Verification Agent - Task 7.3.2 / 7.3.3

Provides health check endpoints with binary verification status monitoring:
- GET /health          — overall health (healthy/degraded/unhealthy) with component checks
- GET /health/ready    — readiness probe (is the system ready to serve requests?)
- GET /health/live     — liveness probe (is the process alive?)
- GET /health/verification — binary verification system status (config loaded, thresholds valid)

Graceful shutdown (Task 7.3.3):
- SIGTERM/SIGINT triggers ShutdownManager to cancel async tasks and run cleanup handlers
- FastAPI lifespan context manager handles startup/shutdown lifecycle

Requirements:
- 7.2: Real-time workflow status observability
- 8.1: Error handling with logging and correlation IDs
- 8.3: StateGraph execution error recovery
- 8.4: Circuit breaker pattern for external service calls
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from verification.utils.health_check import (
    HealthReport,
    HealthStatus,
    run_health_check,
)
from verification.config.binary_verification_config import (
    get_binary_verification_config,
    BinaryVerificationConfig,
)
from verification.utils.graceful_shutdown import (
    get_shutdown_manager,
    lifespan_with_shutdown,
)

logger = logging.getLogger("verification.api")

# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: register signal handlers on startup, graceful shutdown on exit."""
    async with lifespan_with_shutdown(app, shutdown_manager=get_shutdown_manager()):
        yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Verification Agent API",
    description="Health check and monitoring endpoints for the Verification Agent system",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Shutdown middleware — return 503 for non-health requests during shutdown
# ---------------------------------------------------------------------------

@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    """Return 503 Service Unavailable when the server is shutting down."""
    manager = get_shutdown_manager()
    if manager.is_shutting_down and not request.url.path.startswith("/health/live"):
        return JSONResponse(
            content={"detail": "Service is shutting down"},
            status_code=503,
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _correlation_id(request: Request) -> str:
    """Extract or generate a correlation ID for the request."""
    return request.headers.get("X-Correlation-ID", str(uuid.uuid4()))


def _health_status_code(status: HealthStatus) -> int:
    """Map HealthStatus to HTTP status code."""
    if status == HealthStatus.HEALTHY:
        return 200
    if status == HealthStatus.DEGRADED:
        return 200  # degraded is still serving, return 200 so load balancers keep it
    return 503  # unhealthy → service unavailable


def _report_to_response(report: HealthReport, elapsed_ms: float) -> Dict[str, Any]:
    """Serialize a HealthReport to the API response dict."""
    data = report.to_dict()
    data["response_time_ms"] = round(elapsed_ms, 2)
    return data


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", summary="Overall health check")
async def health(request: Request) -> JSONResponse:
    """
    Returns overall system health with individual component checks.

    Response body::

        {
          "status": "healthy" | "degraded" | "unhealthy",
          "timestamp": "<ISO 8601>",
          "checks": {
            "llm_service": {...},
            "database": {...},
            "circuit_breakers": {...},
            "error_rates": {...},
            "resources": {...}
          },
          "checks_passed": 5,
          "checks_total": 5,
          "message": "...",
          "response_time_ms": 12.3
        }
    """
    corr_id = _correlation_id(request)
    t0 = time.monotonic()

    report = await run_health_check(correlation_id=corr_id)
    elapsed_ms = (time.monotonic() - t0) * 1000

    body = _report_to_response(report, elapsed_ms)
    # Reshape to include a "checks" sub-object for clarity
    body["checks"] = {
        "llm_service": body.pop("llm_service", {}),
        "database": body.pop("database", {}),
        "circuit_breakers": body.pop("circuit_breakers", {}),
        "error_rates": body.pop("error_rates", {}),
        "resources": body.pop("resources", {}),
    }

    status_code = _health_status_code(HealthStatus(report.status))
    return JSONResponse(content=body, status_code=status_code)


# ---------------------------------------------------------------------------
# GET /health/ready
# ---------------------------------------------------------------------------

@app.get("/health/ready", summary="Readiness probe")
async def health_ready(request: Request) -> JSONResponse:
    """
    Readiness probe — returns 200 if the system is ready to serve requests,
    503 if it is unhealthy (degraded systems are still considered ready).
    """
    corr_id = _correlation_id(request)
    t0 = time.monotonic()

    report = await run_health_check(correlation_id=corr_id)
    elapsed_ms = (time.monotonic() - t0) * 1000

    ready = report.is_ready()
    body = {
        "status": report.status if isinstance(report.status, str) else report.status.value,
        "ready": ready,
        "message": report.message,
        "response_time_ms": round(elapsed_ms, 2),
    }
    return JSONResponse(content=body, status_code=200 if ready else 503)


# ---------------------------------------------------------------------------
# GET /health/live
# ---------------------------------------------------------------------------

@app.get("/health/live", summary="Liveness probe")
async def health_live() -> JSONResponse:
    """
    Liveness probe — simple alive check.
    Always returns 200 as long as the process is running.
    """
    return JSONResponse(
        content={"status": "alive", "message": "Verification Agent process is running"},
        status_code=200,
    )


# ---------------------------------------------------------------------------
# GET /health/verification
# ---------------------------------------------------------------------------

@app.get("/health/verification", summary="Binary verification system status")
async def health_verification(request: Request) -> JSONResponse:
    """
    Binary verification system status — checks that:
    - Configuration is loaded successfully
    - Thresholds are valid (no configuration warnings)

    Returns 200 if the verification system is operational, 503 otherwise.
    """
    corr_id = _correlation_id(request)
    t0 = time.monotonic()

    checks: Dict[str, Any] = {}
    overall_ok = True

    # 1. Config loaded
    try:
        config: BinaryVerificationConfig = get_binary_verification_config()
        checks["config"] = {
            "status": "ok",
            "environment": config.environment,
            "message": "Binary verification config loaded successfully",
        }
    except Exception as exc:
        logger.warning(
            "Verification config load failed (correlation_id=%s): %s", corr_id, exc
        )
        checks["config"] = {
            "status": "error",
            "message": f"Config load failed: {type(exc).__name__}: {exc}",
        }
        overall_ok = False
        config = None  # type: ignore[assignment]

    # 2. Thresholds valid
    if config is not None:
        try:
            warnings = config.validate_configuration()
            if warnings:
                checks["thresholds"] = {
                    "status": "warning",
                    "warnings": warnings,
                    "message": f"{len(warnings)} threshold warning(s) detected",
                }
                # Warnings don't make the system non-operational, but we surface them
            else:
                checks["thresholds"] = {
                    "status": "ok",
                    "message": "All thresholds are valid",
                }
        except Exception as exc:
            logger.warning(
                "Threshold validation failed (correlation_id=%s): %s", corr_id, exc
            )
            checks["thresholds"] = {
                "status": "error",
                "message": f"Threshold validation failed: {type(exc).__name__}: {exc}",
            }
            overall_ok = False
    else:
        checks["thresholds"] = {
            "status": "skipped",
            "message": "Skipped — config not loaded",
        }

    elapsed_ms = (time.monotonic() - t0) * 1000

    body = {
        "status": "ok" if overall_ok else "error",
        "operational": overall_ok,
        "checks": checks,
        "response_time_ms": round(elapsed_ms, 2),
    }
    return JSONResponse(content=body, status_code=200 if overall_ok else 503)
