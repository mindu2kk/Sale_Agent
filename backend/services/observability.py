"""Lightweight runtime metrics and shadow-routing diagnostics."""

from __future__ import annotations

import os
import threading
from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class RequestTimer:
    started_at: float

    @classmethod
    def start(cls) -> "RequestTimer":
        return cls(perf_counter())

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self.started_at) * 1000, 2)


class AgentMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._intents: Counter[str] = Counter()
        self._modes: Counter[str] = Counter()
        self._clarifications = 0
        self._verification_failures = 0
        self._shadow_mismatches = 0
        self._latencies_ms: deque[float] = deque(maxlen=1000)

    def record(
        self,
        *,
        intent: str,
        mode: str,
        latency_ms: float,
        clarified: bool,
        verification_approved: bool,
        shadow_mismatch: bool,
    ) -> None:
        with self._lock:
            self._intents[intent] += 1
            self._modes[mode] += 1
            self._latencies_ms.append(latency_ms)
            self._clarifications += int(clarified)
            self._verification_failures += int(not verification_approved)
            self._shadow_mismatches += int(shadow_mismatch)

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            p95_index = max(0, int(len(latencies) * 0.95) - 1)
            return {
                "requests": sum(self._intents.values()),
                "intents": dict(self._intents),
                "modes": dict(self._modes),
                "clarifications": self._clarifications,
                "verification_failures": self._verification_failures,
                "shadow_mismatches": self._shadow_mismatches,
                "latency_p95_ms": (
                    latencies[p95_index] if latencies else 0.0
                ),
            }


def shadow_mode_enabled() -> bool:
    return os.getenv("AGENT_SHADOW_MODE", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def legacy_route(message: str, *, has_active_product: bool) -> str:
    """Approximate the former keyword router without executing its retrieval."""
    from backend.services.advisor import CatalogAdvisor
    from backend.services.catalog import get_catalog

    advisor = CatalogAdvisor(get_catalog())
    if has_active_product and advisor.is_alternative_request(message):
        return "cheaper_alternatives"
    if advisor.is_price_objection(message):
        return "price_objection"
    if has_active_product and advisor.is_detail_request(message):
        return "product_detail"
    return "catalog_search"


agent_metrics = AgentMetrics()
