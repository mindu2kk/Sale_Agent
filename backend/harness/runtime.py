"""Trace-aware runtime harness for the catalog-grounded advisor.

The runtime externalizes the agent loop into explicit perception, planning,
execution, verification, recovery, and commit events. It intentionally stores
compact decision state rather than raw prompts or verbose tool output.
"""

from __future__ import annotations

import threading
import uuid
import json
import os
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal

from backend.services.catalog import CatalogProduct
from backend.services.conversation import ConversationPlan, DecisionContext


class BudgetExceededError(RuntimeError):
    def __init__(self, issues: list[str]) -> None:
        super().__init__("Execution budget exceeded: " + "; ".join(issues))
        self.issues = issues

HarnessPhase = Literal[
    "perception",
    "planning",
    "guard",
    "retrieval",
    "execution",
    "verification",
    "recovery",
    "commit",
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ExecutionBudget:
    max_events: int = 20
    max_candidates: int = 40
    max_elapsed_ms: float = 6_000.0


@dataclass(frozen=True)
class EvidenceRef:
    source: str
    product_code: str | None = None
    source_url: str = ""
    fetched_at: str = ""
    valid_until: str = ""
    trust: Literal["high", "medium", "low"] = "high"
    freshness: Literal["fresh", "stale", "unknown"] = "unknown"

    @classmethod
    def from_product(cls, product: CatalogProduct) -> "EvidenceRef":
        freshness: Literal["fresh", "stale", "unknown"] = "unknown"
        today = datetime.now(UTC).date()
        if product.price_valid_until:
            try:
                valid_until = datetime.fromisoformat(
                    product.price_valid_until.replace("Z", "+00:00")
                ).date()
                freshness = "fresh" if valid_until >= today else "stale"
            except ValueError:
                freshness = "unknown"
        elif product.fetched_at:
            try:
                fetched = datetime.fromisoformat(
                    product.fetched_at.replace("Z", "+00:00")
                )
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=UTC)
                age_days = (datetime.now(UTC) - fetched).days
                freshness = "fresh" if age_days <= 30 else "stale"
            except ValueError:
                freshness = "unknown"
        return cls(
            source="product_catalog",
            product_code=product.code,
            source_url=product.source_url,
            fetched_at=product.fetched_at,
            valid_until=product.price_valid_until,
            trust="high" if product.source_url else "medium",
            freshness=freshness,
        )


@dataclass
class BeliefState:
    category: str | None = None
    budget_target: int | None = None
    active_product_code: str | None = None
    candidate_codes: list[str] = field(default_factory=list)
    compared_brands: list[str] = field(default_factory=list)
    preferences: dict[str, float] = field(default_factory=dict)
    last_intent: str | None = None
    confidence: float = 0.0
    updated_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_context(cls, state: DecisionContext) -> "BeliefState":
        return cls(
            category=state.category,
            budget_target=state.budget_target,
            active_product_code=state.active_product_code,
            candidate_codes=list(state.candidate_codes),
            compared_brands=list(state.compared_brands),
            preferences=dict(state.preferences),
            last_intent=state.last_intent,
            confidence=1.0 if state.updated_at and not state.is_expired() else 0.6,
            updated_at=state.updated_at or _now_iso(),
        )


@dataclass(frozen=True)
class HarnessEvent:
    phase: HarnessPhase
    status: Literal["ok", "warning", "failed", "recovered"]
    timestamp: str
    elapsed_ms: float
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessRun:
    run_id: str
    query: str
    started_at: str
    _started_perf: float
    budget: ExecutionBudget
    belief: BeliefState
    events: list[HarnessEvent] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    skill: dict[str, Any] = field(default_factory=dict)
    governance: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    terminal_status: str | None = None

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._started_perf) * 1000, 2)

    def record(
        self,
        phase: HarnessPhase,
        summary: str,
        *,
        status: Literal["ok", "warning", "failed", "recovered"] = "ok",
        data: dict[str, Any] | None = None,
    ) -> None:
        if len(self.events) >= self.budget.max_events:
            return
        self.events.append(
            HarnessEvent(
                phase=phase,
                status=status,
                timestamp=_now_iso(),
                elapsed_ms=self.elapsed_ms(),
                summary=summary,
                data=data or {},
            )
        )

    def budget_issues(self, candidate_count: int = 0) -> list[str]:
        issues: list[str] = []
        if len(self.events) >= self.budget.max_events:
            issues.append("event budget exhausted")
        if candidate_count > self.budget.max_candidates:
            issues.append("candidate budget exceeded")
        if self.elapsed_ms() > self.budget.max_elapsed_ms:
            issues.append("latency budget exceeded")
        return issues

    def enforce_budget(self, candidate_count: int = 0) -> None:
        issues = self.budget_issues(candidate_count)
        if issues:
            raise BudgetExceededError(issues)

    def public_trace(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "terminal_status": self.terminal_status,
            "elapsed_ms": self.elapsed_ms(),
            "belief": {
                "category": self.belief.category,
                "budget_target": self.belief.budget_target,
                "active_product_code": self.belief.active_product_code,
                "candidate_codes": self.belief.candidate_codes,
                "compared_brands": self.belief.compared_brands,
                "preferences": self.belief.preferences,
                "last_intent": self.belief.last_intent,
                "confidence": self.belief.confidence,
            },
            "context": self.context,
            "skill": self.skill,
            "governance": self.governance,
            "environment": self.environment,
            "events": [
                {
                    "phase": event.phase,
                    "status": event.status,
                    "elapsed_ms": event.elapsed_ms,
                    "summary": event.summary,
                    "data": event.data,
                }
                for event in self.events
            ],
            "evidence": [
                {
                    "source": item.source,
                    "product_code": item.product_code,
                    "source_url": item.source_url,
                    "fetched_at": item.fetched_at,
                    "valid_until": item.valid_until,
                    "trust": item.trust,
                    "freshness": item.freshness,
                }
                for item in self.evidence
            ],
        }


class HarnessRuntime:
    def __init__(self, max_runs: int = 500) -> None:
        self._lock = threading.Lock()
        self._runs: deque[dict[str, Any]] = deque(maxlen=max_runs)
        self._phase_failures: Counter[str] = Counter()
        self._recoveries = 0
        self._budget_violations = 0
        self._audit_path = os.getenv("HARNESS_AUDIT_PATH", "").strip()

    def start(
        self,
        query: str,
        state: DecisionContext,
        budget: ExecutionBudget | None = None,
    ) -> HarnessRun:
        run = HarnessRun(
            run_id=f"hr_{uuid.uuid4().hex[:12]}",
            query=query,
            started_at=_now_iso(),
            _started_perf=perf_counter(),
            budget=budget or ExecutionBudget(),
            belief=BeliefState.from_context(state),
        )
        run.record(
            "perception",
            "Constructed compact belief state from request and conversation state.",
            data={
                "state_expired": state.is_expired(),
                "candidate_count": len(state.candidate_codes),
            },
        )
        return run

    @staticmethod
    def record_plan(run: HarnessRun, plan: ConversationPlan) -> list[str]:
        issues: list[str] = []
        if plan.product_codes and not plan.category:
            issues.append("named products have no resolved category")
        run.belief.category = plan.category
        run.belief.compared_brands = list(plan.brands)
        run.belief.preferences = dict(plan.preferences)
        run.belief.last_intent = plan.dialogue_act
        run.belief.confidence = plan.confidence
        run.record(
            "planning",
            "Conversation planner produced a structured executable plan.",
            status="warning" if issues else "ok",
            data={
                "intent": plan.dialogue_act,
                "confidence": plan.confidence,
                "category": plan.category,
                "product_codes": list(plan.product_codes),
                "brands": list(plan.brands),
                "issues": issues,
            },
        )
        run.record(
            "guard",
            "Validated plan constraints before retrieval.",
            status="failed" if issues else "ok",
            data={"issues": issues},
        )
        return issues

    @staticmethod
    def record_retrieval(
        run: HarnessRun,
        products: list[CatalogProduct],
    ) -> list[str]:
        codes = [product.code for product in products]
        issues = run.budget_issues(len(products))
        if run.belief.category:
            wrong_category = [
                product.code
                for product in products
                if product.category != run.belief.category
            ]
            if wrong_category:
                issues.append(
                    "retrieval returned products outside belief category: "
                    + ", ".join(wrong_category)
                )
        run.belief.candidate_codes = codes
        run.evidence = [EvidenceRef.from_product(product) for product in products]
        run.record(
            "retrieval",
            "Retrieved catalog candidates with provenance and freshness metadata.",
            status="warning" if issues else "ok",
            data={
                "candidate_codes": codes,
                "candidate_count": len(codes),
                "issues": issues,
            },
        )
        return issues

    @staticmethod
    def verify_answer(
        run: HarnessRun,
        *,
        answer_codes: list[str],
        verification_approved: bool,
        sources: list[dict[str, Any]],
    ) -> list[str]:
        issues: list[str] = []
        candidates = set(run.belief.candidate_codes)
        if candidates and not set(answer_codes).issubset(candidates):
            issues.append("answer escaped the retrieved candidate set")
        source_codes = {
            str(source.get("product_code"))
            for source in sources
            if source.get("product_code")
        }
        if answer_codes and not set(answer_codes).issubset(source_codes):
            issues.append("answer products are not fully covered by evidence")
        if not verification_approved:
            issues.append("answer verifier did not approve the result")
        stale_codes = [
            item.product_code
            for item in run.evidence
            if item.freshness == "stale" and item.product_code in answer_codes
        ]
        if stale_codes:
            issues.append(
                "answer uses stale catalog evidence: " + ", ".join(stale_codes)
            )
        run.record(
            "verification",
            "Checked candidate containment, evidence coverage, freshness, and verifier status.",
            status="failed" if issues else "ok",
            data={
                "answer_codes": answer_codes,
                "source_codes": sorted(source_codes),
                "issues": issues,
            },
        )
        return issues

    def finish(
        self,
        run: HarnessRun,
        *,
        status: str,
        next_state: DecisionContext,
        recovered: bool = False,
    ) -> None:
        if recovered:
            run.record(
                "recovery",
                "The runtime used clarification or deterministic degradation.",
                status="recovered",
            )
        run.belief = BeliefState.from_context(next_state)
        run.record(
            "commit",
            "Committed the verified decision state for the next turn.",
            data={
                "topic_id": next_state.topic_id,
                "candidate_codes": next_state.candidate_codes,
            },
        )
        run.terminal_status = status
        trace = run.public_trace()
        with self._lock:
            self._runs.append(trace)
            for event in run.events:
                if event.status == "failed":
                    self._phase_failures[event.phase] += 1
            self._recoveries += int(recovered)
            self._budget_violations += int(bool(run.budget_issues()))
            if self._audit_path:
                try:
                    directory = os.path.dirname(self._audit_path)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                    with open(self._audit_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                except OSError:
                    self._phase_failures["audit"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            terminal = Counter(
                str(run.get("terminal_status")) for run in self._runs
            )
            elapsed = sorted(
                float(run.get("elapsed_ms", 0.0)) for run in self._runs
            )
            p95_index = max(0, int(len(elapsed) * 0.95) - 1)
            return {
                "runs": len(self._runs),
                "terminal_statuses": dict(terminal),
                "phase_failures": dict(self._phase_failures),
                "recoveries": self._recoveries,
                "budget_violations": self._budget_violations,
                "latency_p95_ms": elapsed[p95_index] if elapsed else 0.0,
            }


harness_runtime = HarnessRuntime()
