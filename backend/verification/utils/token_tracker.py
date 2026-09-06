"""
LLM Token Usage Tracking with Cost Optimization Alerts - Task 5.3.4

Tracks per-call and aggregate LLM token usage, estimates costs, and fires
alerts when configurable thresholds are exceeded.

Integrates with:
- ExecutionStep.llm_tokens_input / llm_tokens_output / llm_cost_usd
- WorkflowMetrics.llm_tokens_used / llm_tokens_input / llm_tokens_output / cost_estimate
- VerificationAgent._total_tokens_used
"""

from __future__ import annotations

import time
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model pricing table (USD per 1 000 tokens)
# ---------------------------------------------------------------------------

#: Default pricing table keyed by model name (or prefix).
#: Values are (input_cost_per_1k, output_cost_per_1k) in USD.
DEFAULT_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o":               (0.005,  0.015),
    "gpt-4o-mini":          (0.00015, 0.0006),
    "gpt-4-turbo":          (0.01,   0.03),
    "gpt-4":                (0.03,   0.06),
    "gpt-3.5-turbo":        (0.0005, 0.0015),
    # Anthropic
    "claude-3-5-sonnet":    (0.003,  0.015),
    "claude-3-opus":        (0.015,  0.075),
    "claude-3-haiku":       (0.00025, 0.00125),
    # Fallback
    "default":              (0.002,  0.002),
}


def _lookup_pricing(model: str) -> Tuple[float, float]:
    """Return (input_cost_per_1k, output_cost_per_1k) for *model*."""
    # Exact match first
    if model in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[model]
    # Prefix match (e.g. "gpt-4o-2024-05-13" → "gpt-4o")
    for key, pricing in DEFAULT_MODEL_PRICING.items():
        if key != "default" and model.startswith(key):
            return pricing
    return DEFAULT_MODEL_PRICING["default"]


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "default",
) -> float:
    """Return estimated USD cost for *input_tokens* + *output_tokens*."""
    in_rate, out_rate = _lookup_pricing(model)
    return (input_tokens * in_rate + output_tokens * out_rate) / 1000.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenUsageRecord:
    """Token usage for a single LLM call."""

    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    node_name: str = ""
    correlation_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "node_name": self.node_name,
            "correlation_id": self.correlation_id,
        }


@dataclass
class TokenUsageSummary:
    """Aggregate token usage across multiple calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    records: List[TokenUsageRecord] = field(default_factory=list, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_tokens_per_call(self) -> float:
        return self.total_tokens / self.call_count if self.call_count else 0.0

    @property
    def avg_cost_per_call(self) -> float:
        return self.total_cost_usd / self.call_count if self.call_count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "call_count": self.call_count,
            "avg_tokens_per_call": round(self.avg_tokens_per_call, 1),
            "avg_cost_per_call": round(self.avg_cost_per_call, 6),
        }


@dataclass
class CostAlert:
    """Fired when a cost/token threshold is exceeded."""

    alert_type: str          # "total_cost", "per_call_cost", "total_tokens", "per_call_tokens"
    threshold: float
    actual_value: float
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "threshold": self.threshold,
            "actual_value": round(self.actual_value, 6),
            "message": self.message,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Alert thresholds config
# ---------------------------------------------------------------------------

@dataclass
class TokenAlertThresholds:
    """Configurable thresholds that trigger CostAlerts."""

    # Cumulative limits (per tracker lifetime / reset cycle)
    max_total_cost_usd: float = 1.0          # alert when total cost exceeds $1
    max_total_tokens: int = 500_000          # alert when total tokens exceed 500k

    # Per-call limits
    max_per_call_cost_usd: float = 0.10      # alert when a single call costs > $0.10
    max_per_call_tokens: int = 10_000        # alert when a single call uses > 10k tokens

    # Callback invoked with each CostAlert (optional)
    alert_callback: Optional[Callable[[CostAlert], None]] = None


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class LLMTokenTracker:
    """
    Tracks LLM token usage and estimated costs across verification workflow calls.

    Features:
    - Per-call recording with model-aware cost estimation
    - Aggregate summary (total tokens, total cost, call count)
    - Configurable alert thresholds with optional callback
    - Context-manager API for wrapping LLM calls
    - Integration helpers for ExecutionStep and WorkflowMetrics

    Usage::

        tracker = LLMTokenTracker()

        # Manual recording
        tracker.record(model="gpt-4o", input_tokens=500, output_tokens=200,
                       node_name="verification")

        # Context manager (records after the block)
        with tracker.track_call("gpt-4o", node_name="price_check") as ctx:
            response = llm.call(prompt)
            ctx.input_tokens = count_tokens(prompt)
            ctx.output_tokens = count_tokens(response)

        summary = tracker.summary()
        print(summary.to_dict())
        print(tracker.alerts)
    """

    def __init__(
        self,
        thresholds: Optional[TokenAlertThresholds] = None,
        model_pricing: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self._thresholds = thresholds or TokenAlertThresholds()
        self._pricing = model_pricing or DEFAULT_MODEL_PRICING
        self._records: List[TokenUsageRecord] = []
        self._alerts: List[CostAlert] = []

    # ------------------------------------------------------------------
    # Core recording API
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        node_name: str = "",
        correlation_id: str = "",
    ) -> TokenUsageRecord:
        """
        Record a completed LLM call and check alert thresholds.

        Args:
            model: Model name used for cost lookup (e.g. "gpt-4o").
            input_tokens: Number of prompt/input tokens consumed.
            output_tokens: Number of completion/output tokens generated.
            node_name: Workflow node that made the call (for attribution).
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            The created TokenUsageRecord.
        """
        in_rate, out_rate = _lookup_pricing(model)
        cost = (input_tokens * in_rate + output_tokens * out_rate) / 1000.0

        record = TokenUsageRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            node_name=node_name,
            correlation_id=correlation_id,
        )
        self._records.append(record)
        self._check_alerts(record)
        return record

    # ------------------------------------------------------------------
    # Context-manager API
    # ------------------------------------------------------------------

    @contextmanager
    def track_call(
        self,
        model: str,
        node_name: str = "",
        correlation_id: str = "",
    ):
        """
        Sync context manager for wrapping an LLM call.

        The caller sets ``ctx.input_tokens`` and ``ctx.output_tokens`` inside
        the block; the record is committed on exit.

        Example::

            with tracker.track_call("gpt-4o", node_name="verification") as ctx:
                response = llm.complete(prompt)
                ctx.input_tokens = len(prompt.split())
                ctx.output_tokens = len(response.split())
        """
        class _Ctx:
            input_tokens: int = 0
            output_tokens: int = 0
            record: Optional[TokenUsageRecord] = None

        ctx = _Ctx()
        try:
            yield ctx
        finally:
            ctx.record = self.record(
                model=model,
                input_tokens=ctx.input_tokens,
                output_tokens=ctx.output_tokens,
                node_name=node_name,
                correlation_id=correlation_id,
            )

    @asynccontextmanager
    async def async_track_call(
        self,
        model: str,
        node_name: str = "",
        correlation_id: str = "",
    ):
        """
        Async context manager for wrapping an async LLM call.

        Example::

            async with tracker.async_track_call("gpt-4o", node_name="policy_check") as ctx:
                response = await llm.acomplete(prompt)
                ctx.input_tokens = usage.prompt_tokens
                ctx.output_tokens = usage.completion_tokens
        """
        class _Ctx:
            input_tokens: int = 0
            output_tokens: int = 0
            record: Optional[TokenUsageRecord] = None

        ctx = _Ctx()
        try:
            yield ctx
        finally:
            ctx.record = self.record(
                model=model,
                input_tokens=ctx.input_tokens,
                output_tokens=ctx.output_tokens,
                node_name=node_name,
                correlation_id=correlation_id,
            )

    # ------------------------------------------------------------------
    # Summary & reporting
    # ------------------------------------------------------------------

    def summary(self) -> TokenUsageSummary:
        """Return aggregate token usage summary across all recorded calls."""
        total_in = sum(r.input_tokens for r in self._records)
        total_out = sum(r.output_tokens for r in self._records)
        total_cost = sum(r.cost_usd for r in self._records)
        return TokenUsageSummary(
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_cost_usd=total_cost,
            call_count=len(self._records),
            records=list(self._records),
        )

    def get_metrics_for_execution_step(self) -> Dict[str, Any]:
        """
        Return a dict suitable for merging into ``ExecutionStep.metrics``.

        Includes per-call totals for the most recent call (if any) and
        cumulative totals for the tracker lifetime.
        """
        s = self.summary()
        result: Dict[str, Any] = {
            "llm_total_input_tokens": s.total_input_tokens,
            "llm_total_output_tokens": s.total_output_tokens,
            "llm_total_tokens": s.total_tokens,
            "llm_total_cost_usd": round(s.total_cost_usd, 6),
            "llm_call_count": s.call_count,
        }
        if self._records:
            last = self._records[-1]
            result["llm_last_call_input_tokens"] = last.input_tokens
            result["llm_last_call_output_tokens"] = last.output_tokens
            result["llm_last_call_cost_usd"] = round(last.cost_usd, 6)
            result["llm_last_call_model"] = last.model
        return result

    def get_workflow_metrics_fields(self) -> Dict[str, Any]:
        """
        Return a dict with field names matching ``WorkflowMetrics``.

        Suitable for passing as kwargs when constructing WorkflowMetrics.
        """
        s = self.summary()
        return {
            "llm_tokens_used": s.total_tokens,
            "llm_tokens_input": s.total_input_tokens,
            "llm_tokens_output": s.total_output_tokens,
            "cost_estimate": round(s.total_cost_usd, 6),
        }

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    @property
    def alerts(self) -> List[CostAlert]:
        """All alerts fired since last reset."""
        return list(self._alerts)

    def has_alerts(self) -> bool:
        """Return True if any alerts have been fired."""
        return bool(self._alerts)

    def _check_alerts(self, record: TokenUsageRecord) -> None:
        """Check per-call and cumulative thresholds after a new record."""
        t = self._thresholds

        # Per-call cost alert
        if record.cost_usd > t.max_per_call_cost_usd:
            self._fire_alert(CostAlert(
                alert_type="per_call_cost",
                threshold=t.max_per_call_cost_usd,
                actual_value=record.cost_usd,
                message=(
                    f"Single LLM call cost ${record.cost_usd:.4f} exceeds threshold "
                    f"${t.max_per_call_cost_usd:.4f} "
                    f"(model={record.model}, node={record.node_name})"
                ),
            ))

        # Per-call token alert
        if record.total_tokens > t.max_per_call_tokens:
            self._fire_alert(CostAlert(
                alert_type="per_call_tokens",
                threshold=float(t.max_per_call_tokens),
                actual_value=float(record.total_tokens),
                message=(
                    f"Single LLM call used {record.total_tokens} tokens, exceeds threshold "
                    f"{t.max_per_call_tokens} "
                    f"(model={record.model}, node={record.node_name})"
                ),
            ))

        # Cumulative cost alert
        total_cost = sum(r.cost_usd for r in self._records)
        if total_cost > t.max_total_cost_usd:
            self._fire_alert(CostAlert(
                alert_type="total_cost",
                threshold=t.max_total_cost_usd,
                actual_value=total_cost,
                message=(
                    f"Cumulative LLM cost ${total_cost:.4f} exceeds threshold "
                    f"${t.max_total_cost_usd:.4f} after {len(self._records)} calls"
                ),
            ))

        # Cumulative token alert
        total_tokens = sum(r.total_tokens for r in self._records)
        if total_tokens > t.max_total_tokens:
            self._fire_alert(CostAlert(
                alert_type="total_tokens",
                threshold=float(t.max_total_tokens),
                actual_value=float(total_tokens),
                message=(
                    f"Cumulative token usage {total_tokens} exceeds threshold "
                    f"{t.max_total_tokens} after {len(self._records)} calls"
                ),
            ))

    def _fire_alert(self, alert: CostAlert) -> None:
        self._alerts.append(alert)
        if self._thresholds.alert_callback is not None:
            try:
                self._thresholds.alert_callback(alert)
            except Exception:
                pass  # Never let callback errors break the tracker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all records and alerts."""
        self._records.clear()
        self._alerts.clear()

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_global_tracker: Optional[LLMTokenTracker] = None


def get_token_tracker(
    thresholds: Optional[TokenAlertThresholds] = None,
) -> LLMTokenTracker:
    """Return (or create) the module-level LLMTokenTracker singleton."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LLMTokenTracker(thresholds=thresholds)
    return _global_tracker


def reset_token_tracker() -> None:
    """Reset the module-level singleton (useful in tests)."""
    global _global_tracker
    _global_tracker = None


__all__ = [
    "DEFAULT_MODEL_PRICING",
    "estimate_cost",
    "TokenUsageRecord",
    "TokenUsageSummary",
    "CostAlert",
    "TokenAlertThresholds",
    "LLMTokenTracker",
    "get_token_tracker",
    "reset_token_tracker",
]
