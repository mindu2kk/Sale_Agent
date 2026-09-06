import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from backend.harness.types import BeliefState, ConversationPlan, EvidenceRef
from backend.services.ai_service import AIAnswer
from backend.services.catalog import CatalogProduct
from backend.harness.postflight import run_postflight_verification

@pytest.fixture
def base_context():
    return BeliefState(
        version=1,
        category="laptop",
        candidateProductIds=["LAP0001", "LAP0002"],
        confidence=0.9,
        freshness="fresh",
        catalogRevision="v1"
    )

@pytest.fixture
def base_plan():
    return ConversationPlan(
        intent="product_detail",
        skillName="detail",
        objective="detail",
        confidence=0.9,
        category="laptop",
        price_intent=None,
        shouldAskClarification=False
    )

@pytest.fixture
def candidates():
    return [
        CatalogProduct(code="LAP0001", category="laptop", price="1000", context="GPU SSD bền bỉ pin tốt", brand="A", specs={}),
        CatalogProduct(code="LAP0002", category="laptop", price="2000", context="RAM CPU màn hình bảo hành", brand="B", specs={})
    ]

@pytest.fixture
def evidence():
    from backend.harness.runtime import EvidenceRef as RuntimeEvidenceRef
    return [
        RuntimeEvidenceRef(source="catalog", product_code="LAP0001", freshness="fresh"),
        RuntimeEvidenceRef(source="catalog", product_code="LAP0002", freshness="fresh")
    ]

def test_valid_answer_passes(candidates, evidence, base_context, base_plan):
    answer = AIAnswer(
        text="Laptop A giá 1000, cực kỳ bền bỉ",
        status="succeeded",
        tools_used=[],
        sources=[],
        mode="direct",
        product_codes=["LAP0001"]
    )
    res = run_postflight_verification(answer, candidates, evidence, base_context, base_plan, "v1")
    assert res.passed

def test_hallucinated_sku_fails(candidates, evidence, base_context, base_plan):
    answer = AIAnswer(
        text="Tôi tìm thấy Laptop XYZ",
        status="succeeded",
        tools_used=[],
        sources=[],
        mode="direct",
        product_codes=["LAP9999"]
    )
    res = run_postflight_verification(answer, candidates, evidence, base_context, base_plan, "v1")
    assert not res.passed
    assert any(f.code == "candidate_not_contained" for f in res.failures)

def test_unsupported_durability_claim(base_context, base_plan):
    candidates = [CatalogProduct(code="LAP0001", category="laptop", price="1000", context="Chỉ có thông tin giá", brand="A", specs={})]
    from backend.harness.runtime import EvidenceRef as RuntimeEvidenceRef
    ev = [RuntimeEvidenceRef(source="catalog", product_code="LAP0001", freshness="fresh")]

    answer = AIAnswer(
        text="Sản phẩm này rất bền bỉ chống sốc tốt.",
        status="succeeded",
        tools_used=[],
        sources=[],
        mode="direct",
        product_codes=["LAP0001"]
    )
    res = run_postflight_verification(answer, candidates, ev, base_context, base_plan, "v1")
    assert not res.passed
    assert any(f.code == "unsupported_claim" for f in res.failures)

def test_unsupported_price_claim(base_context, base_plan):
    candidates = [CatalogProduct(code="LAP0001", category="laptop", price=None, context="GPU SSD bền bỉ", brand="A", specs={})]
    from backend.harness.runtime import EvidenceRef as RuntimeEvidenceRef
    ev = [RuntimeEvidenceRef(source="catalog", product_code="LAP0001", freshness="fresh")]

    answer = AIAnswer(
        text="Đây là sản phẩm rẻ nhất.",
        status="succeeded",
        tools_used=[],
        sources=[],
        mode="direct",
        product_codes=["LAP0001"]
    )
    res = run_postflight_verification(answer, candidates, ev, base_context, base_plan, "v1")
    assert not res.passed
    assert any(f.code == "unsupported_claim" for f in res.failures)

def test_stale_price_evidence(candidates, base_context, base_plan):
    from backend.harness.runtime import EvidenceRef as RuntimeEvidenceRef
    ev = [RuntimeEvidenceRef(source="catalog", product_code="LAP0001", freshness="stale")]

    answer = AIAnswer(
        text="Laptop A giá 1000",
        status="succeeded",
        tools_used=[],
        sources=[],
        mode="direct",
        product_codes=["LAP0001"]
    )
    res = run_postflight_verification(answer, candidates, ev, base_context, base_plan, "v1")
    assert not res.passed
    assert any(f.code == "stale_evidence" for f in res.failures)

def test_phone_category_mismatch(candidates, evidence, base_context):
    plan = ConversationPlan(intent="product_detail", skillName="detail", objective="detail", confidence=0.9, expectedCategory="phone", price_intent=None, shouldAskClarification=False)
    answer = AIAnswer(
        text="Điện thoại này không có card đồ họa rời GPU.",
        status="succeeded",
        tools_used=[],
        sources=[],
        mode="direct",
        product_codes=["LAP0001"]
    )
    res = run_postflight_verification(answer, candidates, evidence, base_context, plan, "v1")
    assert not res.passed
    assert any(f.code == "category_policy_mismatch" for f in res.failures)

def test_weak_recommendation_no_criterion(evidence, base_context):
    test_candidates = [
        CatalogProduct(code="LAP0001", category="Laptop", price=None, context="GPU SSD bền bỉ pin tốt", brand="A", specs={}),
        CatalogProduct(code="LAP0002", category="Laptop", price=None, context="RAM CPU màn hình bảo hành", brand="B", specs={})
    ]
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    answer = AIAnswer(
        text="Laptop A chắc chắn vượt trội Laptop B.",
        status="succeeded",
        tools_used=[],
        sources=[],
        mode="direct",
        confidence=0.9,
        product_codes=["LAP0001", "LAP0002"]
    )
    # No price advantage, no tradeoff, no criterion
    res = run_postflight_verification(answer, test_candidates, [], base_context, plan, "v1")
    print("\nFAILURES:", res.failures)
    assert not res.passed
    assert res.recovery_action is not None
    assert "cân nhắc" in getattr(res.recovery_action, "message", "") or "Laptop" in getattr(res.recovery_action, "message", "")
    assert any("decision_gate" in f.code for f in res.failures)

def test_strong_recommendation_price_advantage(candidates, base_context):
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="compare", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    answer = AIAnswer(
        text="Laptop A có lợi thế về giá (price) rẻ hơn.",
        status="succeeded",
        tools_used=[],
        sources=[],
        mode="direct",
        confidence=0.9,
        product_codes=["LAP0001", "LAP0002"]
    )
    from backend.harness.runtime import EvidenceRef as RuntimeEvidenceRef
    ev = [RuntimeEvidenceRef(source="catalog", product_code="LAP0001", freshness="fresh")]
    res = run_postflight_verification(answer, candidates, ev, base_context, plan, "v1")
    assert res.passed

from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

@patch("backend.api.main.get_ai_service")
def test_integration_verification_fail_recovery(mock_get_ai, monkeypatch):
    monkeypatch.setenv("HARNESS_DEV_MODE", "1")
    mock_ai = MagicMock()
    mock_ai.answer = AsyncMock()

    # Return an answer that hallucinate products
    mock_ai.answer.return_value = AIAnswer(
        mode="direct", status="succeeded", answer_type="direct", text="OK", tools_used=[], sources=[], product_codes=["LAP9999"]
    )
    mock_get_ai.return_value = mock_ai

    response = client.post("/api/chat", json={
        "message": "Tìm laptop DELL",
        "history": []
    })

    assert response.status_code == 200
    data = response.json()

    # Normal answer is blocked
    assert data.get("workflow_status", data.get("workflowStatus")) == "blocked"
    assert data.get("ai_mode", data.get("aiMode")) == "safe_degraded"

    # The integration test correctly returns blocked status and safe_degraded mode
    pass
