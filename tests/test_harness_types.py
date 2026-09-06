from typing import Dict, Any
import pytest
from pydantic import ValidationError

from backend.harness.types import (
    PhaseEvent, ExecutionBudget, HarnessRun, BeliefState, EvidenceRef,
    CandidateSet, ConversationPlan, DecisionPacket, VerificationResult,
    RecoveryAction, SkillDefinition, AskClarificationAction, SafeDegradeAction
)

def test_execution_budget_valid():
    budget = ExecutionBudget(
        maxPhaseEvents=10,
        maxCandidates=5,
        maxElapsedMs=6000,
        maxRetries=2
    )
    assert budget.max_phase_events == 10
    assert budget.max_elapsed_ms == 6000

def test_execution_budget_missing_field():
    with pytest.raises(ValidationError):
        ExecutionBudget(
            maxPhaseEvents=10,
            maxCandidates=5,
            # Missing maxElapsedMs
            maxRetries=2
        )

def test_phase_event_valid():
    event = PhaseEvent(
        eventId="evt1",
        runId="run1",
        phase="perceive",
        eventType="user_input",
        timestamp="2026-06-20T00:00:00Z",
        status="succeeded"
    )
    assert event.phase == "perceive"
    assert event.status == "succeeded"

def test_phase_event_invalid_enum():
    with pytest.raises(ValidationError) as exc:
        PhaseEvent(
            eventId="evt1",
            runId="run1",
            phase="invalid_phase", # type: ignore
            eventType="user_input",
            timestamp="2026-06-20T00:00:00Z",
            status="succeeded"
        )
    assert "Input should be" in str(exc.value)

def test_belief_state_valid():
    state = BeliefState(
        version=1,
        category="laptop",
        confidence=0.85,
        freshness="fresh",
        catalogRevision="rev_1"
    )
    assert state.version == 1
    assert state.freshness == "fresh"

def test_belief_state_invalid_category():
    with pytest.raises(ValidationError):
        BeliefState(
            version=1,
            category="tablet", # type: ignore (not in CategoryType)
            confidence=0.85,
            freshness="fresh",
            catalogRevision="rev_1"
        )

def test_evidence_ref_valid():
    evidence = EvidenceRef(
        evidenceId="ev1",
        source="catalog",
        field="price",
        value=15000000,
        fetchedAt="2026-06-20T00:00:00Z",
        catalogRevision="rev_1",
        trust="high",
        freshness="fresh"
    )
    assert evidence.source == "catalog"
    assert evidence.value == 15000000

def test_conversation_plan_valid():
    plan = ConversationPlan(
        intent="compare_products",
        skillName="product-comparison",
        objective="Compare two laptops",
        shouldAskClarification=False
    )
    assert plan.intent == "compare_products"

def test_verification_result_with_recovery():
    action = SafeDegradeAction(type="safe_degrade", message="Degrading safely")
    result = VerificationResult(
        passed=False,
        failures=[
            {"code": "stale_data", "severity": "warning", "message": "Price is old"}
        ],
        recoveryAction=action
    )
    assert result.passed is False
    assert result.recovery_action.type == "safe_degrade"
    assert isinstance(result.recovery_action, SafeDegradeAction)

def test_verification_result_invalid_recovery_type():
    with pytest.raises(ValidationError):
        VerificationResult(
            passed=False,
            recoveryAction={"type": "unknown_action_type"} # type: ignore
        )

def test_harness_run_valid():
    budget = ExecutionBudget(
        maxPhaseEvents=10, maxCandidates=5, maxElapsedMs=6000, maxRetries=2
    )
    run = HarnessRun(
        runId="run1",
        requestId="req1",
        startedAt="2026-06-20T00:00:00Z",
        userMessageHash="abc123hash",
        catalogRevision="rev_1",
        phases=[],
        budget=budget
    )
    assert run.run_id == "run1"
    assert run.budget.max_candidates == 5

def test_skill_definition_valid():
    skill = SkillDefinition(
        name="compare",
        version="1.0",
        lifecycle="active",
        owner="team-a",
        inputContract="schema",
        outputContract="schema"
    )
    assert skill.lifecycle == "active"
