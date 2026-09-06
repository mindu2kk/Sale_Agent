from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from backend.services.catalog import CatalogProduct
from backend.services.conversation import ConversationPlanner, DecisionContext
from backend.harness.runtime import (
    BeliefState,
    EvidenceRef,
    ExecutionBudget,
    HarnessRuntime,
)
from backend.harness.advisor import AdvisorHarness
from backend.harness.context import ContextLifecycleManager
from backend.harness.governance import RecoveryPolicy
from backend.harness.skills import SkillRegistry
from backend.api.main import app
from backend.services.catalog import get_catalog


def product(
    code: str,
    *,
    category: str = "Laptop",
    valid_until: str = "",
) -> CatalogProduct:
    return CatalogProduct(
        code=code,
        category=category,
        brand="Test",
        price="10.000.000 VNĐ",
        context="",
        specs=("RAM 16 GB",),
        title=f"Test {code}",
        source_url=f"https://example.com/{code}",
        fetched_at=datetime.now(UTC).isoformat(),
        price_valid_until=valid_until,
    )


def test_belief_state_keeps_decision_state_not_raw_history() -> None:
    state = DecisionContext(
        category="Laptop",
        budget_target=20_000_000,
        candidate_codes=["00927423"],
        compared_brands=["Dell", "Asus"],
        preferences={"performance": 1.0},
    )
    belief = BeliefState.from_context(state)
    assert belief.category == "Laptop"
    assert belief.budget_target == 20_000_000
    assert belief.candidate_codes == ["00927423"]
    assert not hasattr(belief, "history")


def test_runtime_detects_category_drift_before_answer() -> None:
    runtime = HarnessRuntime()
    state = DecisionContext(category="Laptop")
    run = runtime.start("Tư vấn laptop", state)
    plan = ConversationPlanner(get_catalog()).plan("Tư vấn laptop", state)
    runtime.record_plan(run, plan)
    issues = runtime.record_retrieval(
        run,
        [product("00000001", category="Mobile Phone")],
    )
    assert any("outside belief category" in issue for issue in issues)


def test_evidence_ref_marks_expired_price_as_stale() -> None:
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    evidence = EvidenceRef.from_product(
        product("00000002", valid_until=yesterday)
    )
    assert evidence.freshness == "stale"
    assert evidence.trust == "high"


def test_budget_reports_candidate_and_latency_constraints() -> None:
    runtime = HarnessRuntime()
    run = runtime.start(
        "test",
        DecisionContext(),
        ExecutionBudget(max_events=2, max_candidates=1, max_elapsed_ms=0),
    )
    issues = run.budget_issues(candidate_count=2)
    assert "candidate budget exceeded" in issues
    assert "latency budget exceeded" in issues


def test_api_exposes_complete_harness_trajectory_in_development(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXPOSE_DECISION_TRACE", "true")
    payload = TestClient(app).post(
        "/api/chat",
        json={
            "message": "So sánh Oppo A6C 4GB, Tecno Spark 50 4GB",
            "history": [],
            "conversation_state": None,
        },
    ).json()
    trace = payload["decision_trace"]["harness"]
    phases = [event["phase"] for event in trace["events"]]
    assert trace["terminal_status"] == "approved"
    assert phases[0] == "perception"
    assert "planning" in phases
    assert "retrieval" in phases
    assert "execution" in phases
    assert "verification" in phases
    assert phases[-1] == "commit"
    assert trace["belief"]["category"] == "Mobile Phone"
    assert set(trace["belief"]["candidate_codes"]) == {
        "00928862",
        "00928700",
    }
    assert {
        item["product_code"] for item in trace["evidence"]
    } == {"00928862", "00928700"}


def test_metrics_include_harness_run_profile(monkeypatch) -> None:
    monkeypatch.setenv("EXPOSE_DECISION_TRACE", "true")
    client = TestClient(app)
    client.post(
        "/api/chat",
        json={
            "message": "Tư vấn điện thoại tầm 5 triệu",
            "history": [],
            "conversation_state": None,
        },
    )
    metrics = client.get("/metrics").json()
    assert metrics["harness"]["runs"] >= 1
    assert "latency_p95_ms" in metrics["harness"]
    assert "phase_failures" in metrics["harness"]


def test_context_lifecycle_discards_old_prose_but_keeps_decisions() -> None:
    catalog = get_catalog()

    class Turn:
        def __init__(self, role: str, text: str, product_codes=None) -> None:
            self.role = role
            self.text = text
            self.product_codes = product_codes or []

    history = [
        Turn("assistant", "Mô tả dài " * 100),
        Turn("user", "Ưu tiên độ bền và hiệu năng", ["00927423"]),
        Turn("assistant", "Kết quả có thể dựng lại từ catalog"),
    ]
    context = ContextLifecycleManager(
        max_turns=2, max_text_chars=1_000
    ).prepare(history, DecisionContext(), catalog)
    assert context.compacted is True
    assert "00927423" in context.product_codes
    assert context.user_decisions == ("Ưu tiên độ bền và hiệu năng",)


def test_skill_registry_resolves_versioned_comparison_contract() -> None:
    catalog = get_catalog()
    plan = ConversationPlanner(catalog).plan(
        "So sánh Oppo A6C 4GB, Tecno Spark 50 4GB",
        DecisionContext(),
    )
    skill = SkillRegistry().resolve(plan)
    assert skill.name == "product-comparison"
    assert skill.version == "1.1.0"
    assert skill.minimum_products == 2


def test_skill_registry_resolves_catalog_ranking_contract() -> None:
    catalog = get_catalog()
    plan = ConversationPlanner(catalog).plan(
        "Tôi cần laptop xịn nhất shop",
        DecisionContext(),
    )
    skill = SkillRegistry().resolve(plan)
    assert plan.dialogue_act == "catalog_ranking"
    assert plan.goal == "best_overall"
    assert skill.name == "catalog-ranking"
    assert skill.maximum_candidates == 16


def test_preflight_converts_cross_category_plan_to_clarification() -> None:
    catalog = get_catalog()
    laptop = next(item for item in catalog.products if item.category == "Laptop")
    phone = next(
        item for item in catalog.products if item.category == "Mobile Phone"
    )
    state = DecisionContext(
        category="Laptop",
        compared_codes=[laptop.code, phone.code],
        candidate_codes=[laptop.code, phone.code],
    )
    harness = AdvisorHarness(runtime=HarnessRuntime())
    session = harness.begin(
        query="So sánh hai máy này",
        history=[],
        state=state,
        catalog=catalog,
    )
    assert session.plan.dialogue_act == "clarify"
    assert session.recovery_action != "continue"
    assert session.run.governance["preflight"]


def test_api_trace_exposes_skill_context_and_governance(monkeypatch) -> None:
    monkeypatch.setenv("EXPOSE_DECISION_TRACE", "true")
    payload = TestClient(app).post(
        "/api/chat",
        json={
            "message": "So sánh Oppo A6C 4GB, Tecno Spark 50 4GB",
            "history": [],
            "conversation_state": None,
        },
    ).json()
    trace = payload["decision_trace"]["harness"]
    assert trace["skill"]["name"] == "product-comparison"
    assert trace["context"]["input_turns"] == 0
    assert trace["governance"]["preflight"] == []
    assert payload["conversation_state"]["state_version"] >= 2
    assert payload["conversation_state"]["catalog_revision"]


def test_catalog_revision_change_is_visible_and_forces_refetch_policy() -> None:
    catalog = get_catalog()
    state = DecisionContext(
        category="Laptop",
        catalog_revision="obsolete-revision",
    )
    session = AdvisorHarness(runtime=HarnessRuntime()).begin(
        query="Tư vấn laptop tầm 20 triệu",
        history=[],
        state=state,
        catalog=catalog,
    )
    codes = {
        item["code"] for item in session.run.governance["preflight"]
    }
    assert "catalog_revision_changed" in codes
    assert session.recovery_action == "continue"
    assert session.run.environment["catalog_revision"] != "obsolete-revision"
