"""
Health Check Endpoints for Monitoring Systems - Task 6.3.4

Provides async health check functionality for the Verification Agent system,
checking:
- Overall system health status
- LLM service connectivity (via circuit breaker state)
- Database/ChromaDB connectivity (via circuit breaker state)
- Circuit breaker states (integrates with circuit_breaker.py)
- Error rate status (integrates with error_rate_tracker.py)
- Resource usage (integrates with resource_monitor.py)

Returns structured Pydantic models with status: "healthy", "degraded", "unhealthy"

Requirements:
- 8.1: Error handling with logging and correlation IDs
- 8.4: Circuit breaker pattern for external service calls
- 7.2: Real-time workflow status observability
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .circuit_breaker import CircuitState, get_circuit_breaker_registry
from .error_rate_tracker import WORKFLOW_COMPONENTS, get_error_rate_tracker
from .resource_monitor import ResourceUsageMonitor

logger = logging.getLogger("verification.health_check")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Error rate threshold above which a component is considered degraded/unhealthy
_ERROR_RATE_DEGRADED_THRESHOLD = 0.25   # 25% error rate → degraded
_ERROR_RATE_UNHEALTHY_THRESHOLD = 0.50  # 50% error rate → unhealthy

# Memory threshold (MB) above which resource usage is considered degraded
_MEMORY_DEGRADED_MB = 512.0
_MEMORY_UNHEALTHY_MB = 1024.0

# CPU threshold (%) above which resource usage is considered degraded
_CPU_DEGRADED_PERCENT = 70.0
_CPU_UNHEALTHY_PERCENT = 90.0


# ---------------------------------------------------------------------------
# Status Enum
# ---------------------------------------------------------------------------

class HealthStatus(str, Enum):
    """Health status levels for components and overall system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ---------------------------------------------------------------------------
# Component Health Models
# ---------------------------------------------------------------------------

class ServiceHealthDetail(BaseModel):
    """Health detail for a single external service (LLM, DB, ChromaDB)."""

    service_name: str = Field(description="Name of the external service")
    status: HealthStatus = Field(description="Health status of this service")
    circuit_state: str = Field(description="Current circuit breaker state (closed/open/half_open)")
    failure_count: int = Field(ge=0, description="Current consecutive failure count")
    message: str = Field(description="Human-readable status message")

    class Config:
        use_enum_values = True


class ErrorRateHealthDetail(BaseModel):
    """Health detail for error rates across workflow components."""

    status: HealthStatus = Field(description="Overall error rate health status")
    component_rates: Dict[str, float] = Field(
        description="Error rate per workflow component (0.0–1.0)"
    )
    degraded_components: List[str] = Field(
        default_factory=list,
        description="Components with elevated error rates"
    )
    message: str = Field(description="Human-readable status message")

    class Config:
        use_enum_values = True


class ResourceHealthDetail(BaseModel):
    """Health detail for system resource usage."""

    status: HealthStatus = Field(description="Resource usage health status")
    memory_rss_mb: float = Field(ge=0.0, description="Current RSS memory usage in MB")
    cpu_percent: float = Field(ge=0.0, description="Current CPU usage percentage")
    active_async_tasks: int = Field(ge=0, description="Number of active asyncio tasks")
    thread_count: int = Field(ge=0, description="Number of active OS threads")
    message: str = Field(description="Human-readable status message")

    class Config:
        use_enum_values = True


class CircuitBreakerHealthDetail(BaseModel):
    """Health detail for all registered circuit breakers."""

    status: HealthStatus = Field(description="Overall circuit breaker health status")
    open_circuits: List[str] = Field(
        default_factory=list,
        description="Names of services with OPEN circuit breakers"
    )
    half_open_circuits: List[str] = Field(
        default_factory=list,
        description="Names of services with HALF_OPEN circuit breakers"
    )
    total_circuits: int = Field(ge=0, description="Total number of registered circuit breakers")
    message: str = Field(description="Human-readable status message")

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Top-level Health Report
# ---------------------------------------------------------------------------

class HealthReport(BaseModel):
    """
    Complete health report for the Verification Agent system.

    Aggregates health status from all subsystems into a single structured
    response suitable for monitoring systems (e.g. Kubernetes liveness/readiness
    probes, Prometheus, Datadog).

    **Validates: Requirements 8.1** - error handling with observability
    **Validates: Requirements 7.2** - real-time workflow status
    """

    status: HealthStatus = Field(description="Overall system health status")
    timestamp: str = Field(description="ISO 8601 timestamp of the health check")
    version: str = Field(default="1.0", description="Health check schema version")

    # Sub-system details
    llm_service: ServiceHealthDetail = Field(description="LLM API service health")
    database: ServiceHealthDetail = Field(description="Internal DB / ChromaDB health")
    circuit_breakers: CircuitBreakerHealthDetail = Field(description="Circuit breaker states")
    error_rates: ErrorRateHealthDetail = Field(description="Workflow component error rates")
    resources: ResourceHealthDetail = Field(description="System resource usage")

    # Summary
    checks_passed: int = Field(ge=0, description="Number of health checks that passed")
    checks_total: int = Field(ge=0, description="Total number of health checks performed")
    message: str = Field(description="Human-readable overall status message")

    class Config:
        use_enum_values = True

    def is_healthy(self) -> bool:
        """Return True if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    def is_ready(self) -> bool:
        """
        Return True if system is ready to serve requests.
        Degraded systems are still considered ready (they operate with reduced capacity).
        """
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation for JSON serialization."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# HealthChecker
# ---------------------------------------------------------------------------

class HealthChecker:
    """
    Async health checker for the Verification Agent system.

    Checks all subsystems and aggregates results into a HealthReport.

    Usage::

        checker = HealthChecker()
        report = await checker.check()

        if report.is_ready():
            # system can serve requests
            ...

    **Validates: Requirements 8.1** - error handling with logging
    **Validates: Requirements 8.4** - circuit breaker integration
    **Validates: Requirements 7.2** - real-time observability
    """

    def __init__(
        self,
        llm_service_name: str = "llm_api",
        db_service_name: str = "internal_db",
        chromadb_service_name: str = "chromadb",
        error_rate_window_seconds: int = 60,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            llm_service_name: Circuit breaker key for the LLM service.
            db_service_name: Circuit breaker key for the internal DB.
            chromadb_service_name: Circuit breaker key for ChromaDB.
            error_rate_window_seconds: Sliding window for error rate calculation.
            correlation_id: Optional correlation ID for log messages.
        """
        self.llm_service_name = llm_service_name
        self.db_service_name = db_service_name
        self.chromadb_service_name = chromadb_service_name
        self.error_rate_window_seconds = error_rate_window_seconds
        self.correlation_id = correlation_id or "health_check"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def check(self) -> HealthReport:
        """
        Run all health checks and return a HealthReport.

        All sub-checks run concurrently for minimal latency.
        Any unexpected exception in a sub-check is caught and reported
        as unhealthy rather than propagating.

        Returns:
            HealthReport with aggregated status from all subsystems.
        """
        logger.debug(
            "Running health check (correlation_id=%s)", self.correlation_id
        )

        # Run all checks concurrently
        (
            llm_detail,
            db_detail,
            cb_detail,
            error_rate_detail,
            resource_detail,
        ) = await asyncio.gather(
            self._check_llm_service(),
            self._check_database(),
            self._check_circuit_breakers(),
            self._check_error_rates(),
            self._check_resources(),
            return_exceptions=False,
        )

        # Aggregate overall status (worst of all sub-statuses)
        sub_statuses = [
            llm_detail.status,
            db_detail.status,
            cb_detail.status,
            error_rate_detail.status,
            resource_detail.status,
        ]
        overall_status = self._aggregate_status(sub_statuses)

        # Count passed checks
        checks_total = len(sub_statuses)
        checks_passed = sum(
            1 for s in sub_statuses if s == HealthStatus.HEALTHY
        )

        message = self._build_overall_message(overall_status, checks_passed, checks_total)

        report = HealthReport(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            llm_service=llm_detail,
            database=db_detail,
            circuit_breakers=cb_detail,
            error_rates=error_rate_detail,
            resources=resource_detail,
            checks_passed=checks_passed,
            checks_total=checks_total,
            message=message,
        )

        logger.info(
            "Health check complete: status=%s, passed=%d/%d (correlation_id=%s)",
            overall_status,
            checks_passed,
            checks_total,
            self.correlation_id,
        )

        return report

    # ------------------------------------------------------------------
    # Sub-checks
    # ------------------------------------------------------------------

    async def _check_llm_service(self) -> ServiceHealthDetail:
        """Check LLM service health via circuit breaker state."""
        return await asyncio.to_thread(self._check_service_sync, self.llm_service_name)

    async def _check_database(self) -> ServiceHealthDetail:
        """Check database/ChromaDB health via circuit breaker state."""
        # Check both internal_db and chromadb; report worst status
        db_detail = await asyncio.to_thread(self._check_service_sync, self.db_service_name)
        chroma_detail = await asyncio.to_thread(
            self._check_service_sync, self.chromadb_service_name
        )

        # Return the worse of the two
        if _status_rank(chroma_detail.status) > _status_rank(db_detail.status):
            return ServiceHealthDetail(
                service_name=f"{self.db_service_name}/{self.chromadb_service_name}",
                status=chroma_detail.status,
                circuit_state=chroma_detail.circuit_state,
                failure_count=chroma_detail.failure_count,
                message=f"ChromaDB: {chroma_detail.message}",
            )
        return db_detail

    async def _check_circuit_breakers(self) -> CircuitBreakerHealthDetail:
        """Check all registered circuit breakers."""
        return await asyncio.to_thread(self._check_circuit_breakers_sync)

    async def _check_error_rates(self) -> ErrorRateHealthDetail:
        """Check error rates for all workflow components."""
        return await asyncio.to_thread(self._check_error_rates_sync)

    async def _check_resources(self) -> ResourceHealthDetail:
        """Check current system resource usage."""
        return await asyncio.to_thread(self._check_resources_sync)

    # ------------------------------------------------------------------
    # Sync implementations (run in thread pool via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _check_service_sync(self, service_name: str) -> ServiceHealthDetail:
        """Check a single service via its circuit breaker state."""
        try:
            registry = get_circuit_breaker_registry()
            cb = registry.get(service_name)

            if cb is None:
                # No circuit breaker registered → assume healthy (not yet used)
                return ServiceHealthDetail(
                    service_name=service_name,
                    status=HealthStatus.HEALTHY,
                    circuit_state=CircuitState.CLOSED.value,
                    failure_count=0,
                    message=f"{service_name}: no circuit breaker registered (assumed healthy)",
                )

            status_dict = cb.get_status()
            state = CircuitState(status_dict["state"])
            failure_count = status_dict["failure_count"]
            threshold = status_dict["failure_threshold"]

            if state == CircuitState.OPEN:
                health = HealthStatus.UNHEALTHY
                msg = (
                    f"{service_name}: circuit OPEN — "
                    f"{failure_count}/{threshold} failures"
                )
            elif state == CircuitState.HALF_OPEN:
                health = HealthStatus.DEGRADED
                msg = (
                    f"{service_name}: circuit HALF_OPEN — "
                    f"testing recovery after {failure_count} failures"
                )
            else:
                # CLOSED — check if approaching threshold
                if threshold > 0 and failure_count >= threshold * 0.6:
                    health = HealthStatus.DEGRADED
                    msg = (
                        f"{service_name}: circuit CLOSED but "
                        f"{failure_count}/{threshold} failures (approaching threshold)"
                    )
                else:
                    health = HealthStatus.HEALTHY
                    msg = f"{service_name}: circuit CLOSED, operating normally"

            return ServiceHealthDetail(
                service_name=service_name,
                status=health,
                circuit_state=state.value,
                failure_count=failure_count,
                message=msg,
            )

        except Exception as exc:
            logger.warning(
                "Error checking service '%s': %s (correlation_id=%s)",
                service_name, exc, self.correlation_id,
            )
            return ServiceHealthDetail(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                circuit_state="unknown",
                failure_count=0,
                message=f"{service_name}: health check error — {type(exc).__name__}: {exc}",
            )

    def _check_circuit_breakers_sync(self) -> CircuitBreakerHealthDetail:
        """Inspect all registered circuit breakers."""
        try:
            registry = get_circuit_breaker_registry()
            all_statuses = registry.get_all_statuses()

            open_circuits = [
                name for name, s in all_statuses.items()
                if s["state"] == CircuitState.OPEN.value
            ]
            half_open_circuits = [
                name for name, s in all_statuses.items()
                if s["state"] == CircuitState.HALF_OPEN.value
            ]
            total = len(all_statuses)

            if open_circuits:
                status = HealthStatus.UNHEALTHY
                msg = f"{len(open_circuits)} circuit(s) OPEN: {', '.join(open_circuits)}"
            elif half_open_circuits:
                status = HealthStatus.DEGRADED
                msg = f"{len(half_open_circuits)} circuit(s) HALF_OPEN: {', '.join(half_open_circuits)}"
            else:
                status = HealthStatus.HEALTHY
                msg = f"All {total} circuit(s) CLOSED"

            return CircuitBreakerHealthDetail(
                status=status,
                open_circuits=open_circuits,
                half_open_circuits=half_open_circuits,
                total_circuits=total,
                message=msg,
            )

        except Exception as exc:
            logger.warning(
                "Error checking circuit breakers: %s (correlation_id=%s)",
                exc, self.correlation_id,
            )
            return CircuitBreakerHealthDetail(
                status=HealthStatus.UNHEALTHY,
                open_circuits=[],
                half_open_circuits=[],
                total_circuits=0,
                message=f"Circuit breaker check error — {type(exc).__name__}: {exc}",
            )

    def _check_error_rates_sync(self) -> ErrorRateHealthDetail:
        """Check error rates for all workflow components."""
        try:
            tracker = get_error_rate_tracker()
            component_rates: Dict[str, float] = {}
            degraded_components: List[str] = []

            for component in WORKFLOW_COMPONENTS:
                rate = tracker.get_error_rate(
                    component, window_seconds=self.error_rate_window_seconds
                )
                component_rates[component] = round(rate, 4)
                if rate >= _ERROR_RATE_DEGRADED_THRESHOLD:
                    degraded_components.append(component)

            # Determine overall status from worst component
            max_rate = max(component_rates.values()) if component_rates else 0.0

            if max_rate >= _ERROR_RATE_UNHEALTHY_THRESHOLD:
                status = HealthStatus.UNHEALTHY
                msg = (
                    f"High error rates detected: "
                    f"{', '.join(degraded_components)} "
                    f"(max={max_rate:.0%})"
                )
            elif max_rate >= _ERROR_RATE_DEGRADED_THRESHOLD:
                status = HealthStatus.DEGRADED
                msg = (
                    f"Elevated error rates: "
                    f"{', '.join(degraded_components)} "
                    f"(max={max_rate:.0%})"
                )
            else:
                status = HealthStatus.HEALTHY
                msg = f"Error rates normal (max={max_rate:.0%})"

            return ErrorRateHealthDetail(
                status=status,
                component_rates=component_rates,
                degraded_components=degraded_components,
                message=msg,
            )

        except Exception as exc:
            logger.warning(
                "Error checking error rates: %s (correlation_id=%s)",
                exc, self.correlation_id,
            )
            return ErrorRateHealthDetail(
                status=HealthStatus.UNHEALTHY,
                component_rates={},
                degraded_components=[],
                message=f"Error rate check failed — {type(exc).__name__}: {exc}",
            )

    def _check_resources_sync(self) -> ResourceHealthDetail:
        """Check current system resource usage."""
        try:
            snapshot = ResourceUsageMonitor.snapshot()

            memory_mb = snapshot.memory_rss_mb
            cpu_pct = snapshot.cpu_percent

            if memory_mb >= _MEMORY_UNHEALTHY_MB or cpu_pct >= _CPU_UNHEALTHY_PERCENT:
                status = HealthStatus.UNHEALTHY
                msg = (
                    f"Resource usage critical: "
                    f"memory={memory_mb:.0f}MB, cpu={cpu_pct:.0f}%"
                )
            elif memory_mb >= _MEMORY_DEGRADED_MB or cpu_pct >= _CPU_DEGRADED_PERCENT:
                status = HealthStatus.DEGRADED
                msg = (
                    f"Resource usage elevated: "
                    f"memory={memory_mb:.0f}MB, cpu={cpu_pct:.0f}%"
                )
            else:
                status = HealthStatus.HEALTHY
                msg = (
                    f"Resource usage normal: "
                    f"memory={memory_mb:.0f}MB, cpu={cpu_pct:.0f}%"
                )

            return ResourceHealthDetail(
                status=status,
                memory_rss_mb=round(memory_mb, 2),
                cpu_percent=round(cpu_pct, 2),
                active_async_tasks=snapshot.active_async_tasks,
                thread_count=snapshot.thread_count,
                message=msg,
            )

        except Exception as exc:
            logger.warning(
                "Error checking resources: %s (correlation_id=%s)",
                exc, self.correlation_id,
            )
            return ResourceHealthDetail(
                status=HealthStatus.UNHEALTHY,
                memory_rss_mb=0.0,
                cpu_percent=0.0,
                active_async_tasks=0,
                thread_count=0,
                message=f"Resource check failed — {type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_status(statuses: List[HealthStatus]) -> HealthStatus:
        """Return the worst status from a list of statuses."""
        if not statuses:
            return HealthStatus.HEALTHY
        worst_rank = max(_status_rank(s) for s in statuses)
        return _rank_to_status(worst_rank)

    @staticmethod
    def _build_overall_message(
        status: HealthStatus, passed: int, total: int
    ) -> str:
        if status == HealthStatus.HEALTHY:
            return f"All systems operational ({passed}/{total} checks passed)"
        elif status == HealthStatus.DEGRADED:
            return f"System degraded — some checks failed ({passed}/{total} checks passed)"
        else:
            return f"System unhealthy — critical checks failed ({passed}/{total} checks passed)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_rank(status: HealthStatus) -> int:
    """Map status to numeric rank for comparison (higher = worse)."""
    return {
        HealthStatus.HEALTHY: 0,
        HealthStatus.DEGRADED: 1,
        HealthStatus.UNHEALTHY: 2,
    }.get(status, 0)


def _rank_to_status(rank: int) -> HealthStatus:
    return {0: HealthStatus.HEALTHY, 1: HealthStatus.DEGRADED, 2: HealthStatus.UNHEALTHY}.get(
        rank, HealthStatus.UNHEALTHY
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_checker: Optional[HealthChecker] = None
_checker_lock = threading.Lock()


def get_health_checker(
    llm_service_name: str = "llm_api",
    db_service_name: str = "internal_db",
    chromadb_service_name: str = "chromadb",
    error_rate_window_seconds: int = 60,
    correlation_id: Optional[str] = None,
) -> HealthChecker:
    """Return the module-level singleton HealthChecker."""
    global _default_checker
    with _checker_lock:
        if _default_checker is None:
            _default_checker = HealthChecker(
                llm_service_name=llm_service_name,
                db_service_name=db_service_name,
                chromadb_service_name=chromadb_service_name,
                error_rate_window_seconds=error_rate_window_seconds,
                correlation_id=correlation_id,
            )
    return _default_checker


def reset_health_checker() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_checker
    with _checker_lock:
        _default_checker = None


async def run_health_check(correlation_id: Optional[str] = None) -> HealthReport:
    """
    Convenience function: run a full health check and return the report.

    Args:
        correlation_id: Optional correlation ID for log messages.

    Returns:
        HealthReport with aggregated status from all subsystems.
    """
    checker = HealthChecker(correlation_id=correlation_id or "health_check")
    return await checker.check()


__all__ = [
    "HealthStatus",
    "ServiceHealthDetail",
    "ErrorRateHealthDetail",
    "ResourceHealthDetail",
    "CircuitBreakerHealthDetail",
    "HealthReport",
    "HealthChecker",
    "get_health_checker",
    "reset_health_checker",
    "run_health_check",
]
