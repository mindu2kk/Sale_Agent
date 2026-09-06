from __future__ import annotations

import pytest
from datetime import UTC, datetime, timedelta

from backend.services.catalog import CatalogProduct
from backend.services.conversation import DecisionContext, ConversationPlan
from backend.harness.runtime import BeliefState, EvidenceRef, HarnessRuntime, HarnessRun, ExecutionBudget, BudgetExceededError
from backend.harness.governance import PostflightPolicy, GovernanceViolation
from backend.harness.skills import skill_registry

def test_expired_conversation_drops_confidence():
    old_time = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    state = DecisionContext(category="Laptop", updated_at=old_time)
    assert state.is_expired() is True
    belief = BeliefState.from_context(state)
    assert belief.confidence == 0.6  # BeliefState sets 0.6 if expired or no updated_at

def test_verification_failure_triggers_postflight_violation():
    policy = PostflightPolicy()
    plan = type('FakePlan', (), {'dialogue_act': 'product_detail'})()
    skill = skill_registry.resolve(plan)
    violations = policy.evaluate(
        skill=skill,
        candidates=[],
        answer_codes=[],
        verification_approved=False,
        sources=[],
    )
    assert any(v.code == "verification_rejected" for v in violations)

def test_stale_evidence_triggers_verification_issue():
    # If a product has a price_valid_until that is in the past, EvidenceRef freshness is stale
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    product = CatalogProduct(
        code="TEST001",
        category="Laptop",
        brand="Test",
        price="10.000.000 VNĐ",
        context="",
        specs=(),
        title="Test",
        source_url="http://test",
        fetched_at="",
        price_valid_until=yesterday
    )
    evidence = EvidenceRef.from_product(product)
    assert evidence.freshness == "stale"

    run = HarnessRun(
        run_id="test",
        query="test",
        started_at=datetime.now(UTC).isoformat(),
        _started_perf=0.0,
        budget=ExecutionBudget(),
        belief=BeliefState(),
        evidence=[evidence]
    )

    runtime = HarnessRuntime()
    issues = runtime.verify_answer(
        run,
        answer_codes=["TEST001"],
        verification_approved=True,
        sources=[{"product_code": "TEST001"}]
    )
    assert any("stale catalog evidence" in issue for issue in issues)

def test_budget_exceeded_error_enforcement():
    run = HarnessRun(
        run_id="test",
        query="test",
        started_at=datetime.now(UTC).isoformat(),
        _started_perf=0.0,
        budget=ExecutionBudget(max_events=0),
        belief=BeliefState(),
    )
    run.record("perception", "test")
    with pytest.raises(BudgetExceededError) as exc_info:
        run.enforce_budget()
    assert "event budget exhausted" in str(exc_info.value)
