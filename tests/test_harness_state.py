import pytest
from pydantic import ValidationError
from backend.services.conversation import DecisionContext

def test_decision_context_valid_init():
    ctx = DecisionContext(
        category="laptop",
        unresolved_questions=["What is the budget?"],
        confirmed_constraints=["needs gaming GPU"],
        freshness="fresh"
    )
    assert ctx.state_version == 1
    assert ctx.category == "laptop"
    assert ctx.unresolved_questions == ["What is the budget?"]
    assert ctx.confirmed_constraints == ["needs gaming GPU"]
    assert ctx.freshness == "fresh"

def test_freshness_invalid_value():
    with pytest.raises(ValidationError):
        DecisionContext(freshness="very_fresh")

def test_check_freshness_revision_change():
    ctx = DecisionContext(catalog_revision="rev1", candidate_codes=["A1234567"])
    valid_codes = {"A1234567", "B1234567"}
    report = ctx.check_freshness("rev2", valid_codes)
    assert report["was_stale"] is True
    assert report["reason"] == "Catalog revision changed"
    assert ctx.freshness == "stale"
    assert ctx.catalog_revision == "rev2"
    assert "A1234567" in ctx.candidate_codes

def test_check_freshness_removes_invalid_codes():
    ctx = DecisionContext(
        catalog_revision="rev1",
        candidate_codes=["A1234567", "INVALID1"],
        compared_codes=["B1234567", "INVALID2"]
    )
    valid_codes = {"A1234567", "B1234567"}
    report = ctx.check_freshness("rev1", valid_codes)
    assert report["was_stale"] is True
    assert "INVALID1" in report["removed_candidate_ids"]
    assert "INVALID2" in report["removed_compared_ids"]
    assert ctx.candidate_codes == ["A1234567"]
    assert ctx.compared_codes == ["B1234567"]
    assert ctx.freshness == "stale"

def test_compact_trims_lists_and_keeps_reasons():
    ctx = DecisionContext(
        rejected_codes={f"R{i:07d}": "Too expensive" for i in range(15)},
        unresolved_questions=["Keep this?"],
        confirmed_constraints=["Keep this too"]
    )
    # Bypass init validation to simulate state growing during operations
    ctx.candidate_codes = [f"C{i:07d}" for i in range(15)]
    ctx.compared_codes = [f"D{i:07d}" for i in range(10)]

    report = ctx.compact()
    assert report["changed"] is True
    assert report["trimmed_candidates"] == 3
    assert report["trimmed_compared"] == 2
    assert report["trimmed_rejected"] == 5
    assert len(ctx.candidate_codes) == 12
    assert len(ctx.compared_codes) == 8
    assert len(ctx.rejected_codes) == 10

    # Check that reasons and unresolved questions are kept
    assert list(ctx.rejected_codes.values())[0] == "Too expensive"
    assert ctx.unresolved_questions == ["Keep this?"]
    assert ctx.confirmed_constraints == ["Keep this too"]

def test_no_persist_raw_catalog_output():
    with pytest.raises(ValidationError) as exc:
        DecisionContext(category="laptop", raw_catalog_output="Very long description here")
    assert "Extra inputs are not permitted" in str(exc.value)

def test_category_drift_rules():
    # unknown -> phone
    ctx = DecisionContext()
    assert ctx.detect_category_drift("phone") is False

    # laptop -> phone
    ctx = DecisionContext(category="laptop")
    assert ctx.detect_category_drift("phone") is True
