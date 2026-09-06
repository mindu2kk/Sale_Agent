import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.services.conversation import ConversationPlan, DecisionContext
from backend.harness.types import ExecutionBudget, BudgetUsed
from backend.services.catalog import PriceIntent
from backend.harness.preflight import run_preflight

client = TestClient(app)

@pytest.fixture
def base_budget():
    return ExecutionBudget(maxPhaseEvents=20, maxCandidates=10, maxElapsedMs=5000, maxRetries=2)

@pytest.fixture
def base_context():
    return DecisionContext(category="laptop")

@pytest.fixture
def valid_codes():
    return {"LAP0001", "LAP0002", "PHO0001"}

@pytest.fixture
def code_to_cat():
    return {"LAP0001": "laptop", "LAP0002": "laptop", "PHO0001": "phone"}

def test_unknown_category_ranking(base_budget, base_context, valid_codes, code_to_cat):
    plan = ConversationPlan(dialogue_act="catalog_ranking", confidence=0.9, category=None, price_intent=None)
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat)
    assert not res.passed
    assert res.decision == "ask_clarification"
    assert res.trace_event == "preflight_category_mismatch"

def test_valid_ranking(base_budget, base_context, valid_codes, code_to_cat):
    plan = ConversationPlan(dialogue_act="catalog_ranking", confidence=0.9, category="laptop", price_intent=None, goal="best_overall")
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat)
    assert res.passed

def test_cross_category_comparison(base_budget, base_context, valid_codes, code_to_cat):
    plan = ConversationPlan(dialogue_act="brand_comparison", confidence=0.9, category="laptop", price_intent=None, product_codes=["LAP0001", "PHO0001"])
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat)
    assert not res.passed
    assert res.decision == "rejected"
    assert res.trace_event == "preflight_category_mismatch"

def test_unknown_product_code(base_budget, base_context, valid_codes, code_to_cat):
    plan = ConversationPlan(dialogue_act="product_detail", confidence=0.9, category="laptop", price_intent=None, product_codes=["UNKNOWN99"])
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat)
    assert not res.passed
    assert res.trace_event == "preflight_unknown_product"

def test_budget_min_greater_than_max(base_budget, base_context, valid_codes, code_to_cat):
    pi = PriceIntent(mode="range", minimum=2000, maximum=1000)
    plan = ConversationPlan(dialogue_act="catalog_search", confidence=0.9, category="laptop", price_intent=pi)
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat)
    assert not res.passed
    assert res.trace_event == "preflight_invalid_budget"
    assert res.decision == "ask_clarification"

def test_negative_budget(base_budget, base_context, valid_codes, code_to_cat):
    pi = PriceIntent(mode="max", maximum=-500)
    plan = ConversationPlan(dialogue_act="catalog_search", confidence=0.9, category="laptop", price_intent=pi)
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat)
    assert not res.passed
    assert res.decision == "safe_degrade"

def test_unsupported_ranking_objective(base_budget, base_context, valid_codes, code_to_cat):
    plan = ConversationPlan(dialogue_act="catalog_ranking", confidence=0.9, category="laptop", price_intent=None, goal="most_beautiful")
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat)
    assert not res.passed
    assert res.trace_event == "preflight_invalid_skill"

def test_budget_exhausted(base_budget, base_context, valid_codes, code_to_cat):
    plan = ConversationPlan(dialogue_act="catalog_search", confidence=0.9, category="laptop", price_intent=None)
    used = BudgetUsed(elapsedMs=6000, phaseEvents=5, candidateCount=0, retries=0)
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat, used)
    assert not res.passed
    assert res.decision == "safe_degrade"
    assert res.trace_event == "preflight_budget_blocked"

def test_unsupported_skill(base_budget, base_context, valid_codes, code_to_cat):
    plan = ConversationPlan(dialogue_act="hack_system", confidence=0.9, category="laptop", price_intent=None) # type: ignore
    res = run_preflight(plan, base_context, base_budget, valid_codes, code_to_cat)
    assert not res.passed
    assert res.trace_event == "preflight_invalid_skill"
    assert res.decision == "rejected"

# Integration tests with mocking
@patch("backend.harness.preflight.run_preflight")
@patch("backend.api.main.get_ai_service")
def test_ai_not_called_on_preflight_fail(mock_get_ai, mock_preflight):
    from unittest.mock import AsyncMock, MagicMock
    from backend.harness.types import PreflightResult

    mock_preflight.return_value = PreflightResult(passed=False, decision="rejected", trace_event="preflight_blocked", reason="forced_fail")

    mock_ai = MagicMock()
    mock_ai.answer = AsyncMock()
    mock_get_ai.return_value = mock_ai

    response = client.post("/api/chat", json={
        "message": "Tìm laptop LAP9999999", # Unknown product
        "history": []
    })

    assert response.status_code == 200
    mock_ai.answer.assert_not_called()

    data = response.json()
    status_val = data.get("workflow_status", data.get("workflowStatus", ""))
    assert status_val in ["blocked", "clarify", "succeeded"]

@patch("backend.api.main.get_ai_service")
def test_ai_called_on_preflight_pass(mock_get_ai):
    from unittest.mock import AsyncMock, MagicMock
    mock_ai = MagicMock()
    mock_ai.answer = AsyncMock()
    from backend.services.ai_service import AIAnswer
    mock_ai.answer.return_value = AIAnswer(mode="approved", status="succeeded", answer_type="direct", text="OK", tools_used=[], sources=[])
    mock_get_ai.return_value = mock_ai

    response = client.post("/api/chat", json={
        "message": "Tìm laptop chơi game",
        "history": []
    })

    mock_ai.answer.assert_called()
