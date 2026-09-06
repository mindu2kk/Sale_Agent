import pytest
from backend.harness.types import (
    BeliefState, ConversationPlan, EvidenceRef
)
from backend.services.ai_service import AIAnswer
from backend.services.catalog import CatalogProduct
from backend.harness.epistemic import evaluate_decision_gate

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
def candidates():
    return [
        CatalogProduct(code="LAP0001", category="laptop", price="20.000.000", context="GPU RTX 3050 SSD 512GB", brand="A", specs={}),
        CatalogProduct(code="LAP0002", category="laptop", price="25.000.000", context="RAM 16GB CPU i7", brand="B", specs={})
    ]

@pytest.fixture
def evidence():
    from backend.harness.runtime import EvidenceRef as RuntimeEvidenceRef
    return [
        RuntimeEvidenceRef(source="catalog", product_code="LAP0001", freshness="fresh"),
        RuntimeEvidenceRef(source="catalog", product_code="LAP0002", freshness="fresh")
    ]

def test_non_recommendation_passes(candidates, evidence, base_context):
    plan = ConversationPlan(intent="product_detail", skillName="detail", objective="detail", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    answer = AIAnswer(text="Đây là chi tiết.", status="succeeded", tools_used=[], sources=[], mode="direct", product_codes=["LAP0001"])
    res = evaluate_decision_gate(answer, plan, candidates, evidence, base_context)
    assert res.allowed is True
    assert res.abstained is False

def test_winner_with_criterion_evidence_diff_reason_passes(candidates, evidence, base_context):
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="gaming", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    # Winner LAP0001 has GPU -> gaming evidence OK.
    answer = AIAnswer(text="Laptop A tốt hơn vì có GPU mạnh.", status="succeeded", tools_used=[], sources=[], mode="direct", confidence=0.85, product_codes=["LAP0001"])
    res = evaluate_decision_gate(answer, plan, candidates, evidence, base_context)
    assert res.allowed is True

def test_winner_fresh_price_advantage_passes(candidates, evidence, base_context):
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    # LAP0001 is 20m, LAP0002 is 25m. Price advantage > 3%.
    answer = AIAnswer(text="Laptop A đáng mua hơn vì rẻ hơn hẳn.", status="succeeded", tools_used=[], sources=[], mode="direct", confidence=0.9, product_codes=["LAP0001"])
    res = evaluate_decision_gate(answer, plan, candidates, evidence, base_context)
    assert res.allowed is True
    assert res.has_verifiable_advantage is True

def test_winner_missing_criterion_abstains(candidates, evidence, base_context):
    # No price advantage (same price)
    candidates_same_price = [
        CatalogProduct(code="LAP0001", category="laptop", price="20.000.000", context="", brand="A", specs={}),
        CatalogProduct(code="LAP0002", category="laptop", price="20.000.000", context="", brand="B", specs={})
    ]
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    answer = AIAnswer(text="Tôi chọn A.", status="succeeded", tools_used=[], sources=[], mode="direct", confidence=0.9, product_codes=["LAP0001"])
    res = evaluate_decision_gate(answer, plan, candidates_same_price, evidence, base_context)
    assert res.abstained is True
    assert res.trace_event == "decision_gate_missing_criterion"

def test_winner_low_confidence_abstains(candidates, evidence, base_context):
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="gaming", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    answer = AIAnswer(text="Chắc là A.", status="succeeded", tools_used=[], sources=[], mode="direct", confidence=0.7, product_codes=["LAP0001"])
    res = evaluate_decision_gate(answer, plan, candidates, evidence, base_context)
    assert res.abstained is True
    assert res.trace_event == "decision_gate_low_confidence"

def test_winner_insufficient_margin_abstains(base_context, evidence):
    candidates_close_price = [
        CatalogProduct(code="LAP0001", category="laptop", price="20.000.000", context="", brand="A", specs={}),
        CatalogProduct(code="LAP0002", category="laptop", price="19.800.000", context="", brand="B", specs={}) # Only 1% difference
    ]
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    # Recommending LAP0002 but price difference is too small
    answer = AIAnswer(text="B rẻ hơn A.", status="succeeded", tools_used=[], sources=[], mode="direct", confidence=0.9, product_codes=["LAP0002"])
    res = evaluate_decision_gate(answer, plan, candidates_close_price, evidence, base_context)
    assert res.abstained is True
    assert res.trace_event == "decision_gate_insufficient_margin"

def test_winner_insufficient_evidence_abstains(evidence, base_context):
    candidates_same_price = [
        CatalogProduct(code="LAP0001", category="laptop", price="20.000.000", context="", brand="A", specs={}),
        CatalogProduct(code="LAP0002", category="laptop", price="20.000.000", context="", brand="B", specs={})
    ]
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="battery", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    # Neither candidate has battery evidence
    answer = AIAnswer(text="Laptop A pin rất tốt.", status="succeeded", tools_used=[], sources=[], mode="direct", confidence=0.9, product_codes=["LAP0001"])
    res = evaluate_decision_gate(answer, plan, candidates_same_price, evidence, base_context)
    assert res.abstained is True
    assert res.trace_event == "decision_gate_insufficient_evidence"

def test_winner_no_diff_reason_abstains(base_context):
    # One candidate, no comparison context, but claims it's the best or cheaper
    candidates_one = [CatalogProduct(code="LAP0001", category="laptop", price="20.000.000", context="GPU", brand="A", specs={})]
    from backend.harness.runtime import EvidenceRef as RuntimeEvidenceRef
    ev = [RuntimeEvidenceRef(source="catalog", product_code="LAP0001", freshness="fresh")]
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="gaming", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    answer = AIAnswer(text="Laptop này tốt nhất.", status="succeeded", tools_used=[], sources=[], mode="direct", confidence=0.9, product_codes=["LAP0001"])
    res = evaluate_decision_gate(answer, plan, candidates_one, ev, base_context)
    assert res.abstained is True
    assert res.trace_event == "decision_gate_no_differentiating_reason"

def test_trade_off_answer_no_winner_passes(candidates, evidence, base_context):
    plan = ConversationPlan(intent="compare_products", skillName="compare", objective="gaming", confidence=0.9, expectedCategory="laptop", price_intent=None, shouldAskClarification=False)
    # Passing both product codes implies trade-offs, no single winner.
    answer = AIAnswer(text="A có GPU, B có RAM.", status="succeeded", tools_used=[], sources=[], mode="direct", confidence=0.9, product_codes=["LAP0001", "LAP0002"])
    res = evaluate_decision_gate(answer, plan, candidates, evidence, base_context)
    # Even if no strict differentiating reason, trade off passes because we only strictly check winner validity.
    # Actually my logic treats multiple output codes as winner_codes.
    # If both have GPU, evidence_ok = True.
    # For trade-offs to pass correctly without being blocked, allowed = True.
    assert res.allowed is True
