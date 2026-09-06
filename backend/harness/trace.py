import uuid
from time import perf_counter
from datetime import datetime, UTC
from typing import Optional, Dict, Any

from backend.harness.types import (
    HarnessRun, PhaseEvent, ExecutionBudget, PhaseName, EventStatus,
    TerminalEvent, BudgetUsed
)

class BudgetExhaustedError(RuntimeError):
    pass

class TraceCollector:
    def __init__(
        self,
        request_id: str,
        user_message_hash: str,
        catalog_revision: str,
        budget: ExecutionBudget
    ):
        self._start_perf = perf_counter()
        self.run = HarnessRun(
            runId=f"run_{uuid.uuid4().hex[:12]}",
            requestId=request_id,
            startedAt=self._now_iso(),
            userMessageHash=user_message_hash,
            catalogRevision=catalog_revision,
            phases=[],
            budget=budget
        )
        self._candidate_count = 0
        self._retries = 0

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def elapsed_ms(self) -> int:
        return int((perf_counter() - self._start_perf) * 1000)

    def set_candidate_count(self, count: int):
        self._candidate_count = count

    def increment_retries(self):
        self._retries += 1

    def check_budget(self) -> None:
        if len(self.run.phases) >= self.run.budget.max_phase_events:
            self.record_phase("recover", "budget_exhausted", "failed", "Max phase events exceeded")
            raise BudgetExhaustedError("Event budget exhausted")
        if self._candidate_count > self.run.budget.max_candidates:
            self.record_phase("recover", "budget_exhausted", "failed", "Max candidates exceeded")
            raise BudgetExhaustedError("Candidate budget exceeded")
        if self.elapsed_ms() > self.run.budget.max_elapsed_ms:
            self.record_phase("recover", "budget_exhausted", "failed", "Max elapsed time exceeded")
            raise BudgetExhaustedError("Latency budget exceeded")
        if self._retries > self.run.budget.max_retries:
            self.record_phase("recover", "budget_exhausted", "failed", "Max retries exceeded")
            raise BudgetExhaustedError("Retry budget exhausted")

    def record_phase(
        self,
        phase: PhaseName,
        event_type: str,
        status: EventStatus,
        reason: Optional[str] = None
    ) -> None:
        event = PhaseEvent(
            eventId=f"evt_{uuid.uuid4().hex[:8]}",
            runId=self.run.run_id,
            phase=phase,
            eventType=event_type,
            timestamp=self._now_iso(),
            status=status,
            reason=reason,
            budgetUsed=BudgetUsed(
                elapsedMs=self.elapsed_ms(),
                phaseEvents=len(self.run.phases) + 1,
                candidateCount=self._candidate_count,
                retries=self._retries
            )
        )
        self.run.phases.append(event)

    def finish_run(self, terminal_status: EventStatus, reason: str) -> None:
        if self.run.terminal_event is not None:
            # Idempotent: Do not overwrite if already finished
            return

        self.run.ended_at = self._now_iso()
        self.run.terminal_event = TerminalEvent(
            eventId=f"term_{uuid.uuid4().hex[:8]}",
            status=terminal_status,
            reason=reason
        )

    def ensure_terminal_event(self) -> None:
        if self.run.terminal_event is None:
            raise RuntimeError("HarnessRun finished without a terminal event")

    def get_public_trace(self) -> Dict[str, Any]:
        return self.run.model_dump(by_alias=True, exclude_none=True)
