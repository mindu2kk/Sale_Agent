from __future__ import annotations

from backend.agent.product_resolver import resolve_product_reference
from backend.agent.state import AgentState, CandidateRef


def _candidate(
    code: str,
    name: str,
    brand: str,
    *,
    category: str = "Laptop",
) -> CandidateRef:
    return CandidateRef(
        code=code,
        name=name,
        brand=brand,
        category=category,
        price_value=19_490_000,
        summary_specs=("CPU Core 5", "RAM 16GB"),
    )


MSI = _candidate("00921510", "MSI Gaming Thin 15 B13UC-3247VN i5 13420H", "MSI")
ACER = _candidate("00912993", "Acer Swift Lite 14 AI SFL14-51M-78XZ U7 155U", "Acer")
HP = _candidate("00921548", "HP 14-ep1179TU Core 5 120U (C89ZSPA)", "HP")
HP_OTHER = _candidate("00926893", "HP 14-em0024AU Ryzen 5-7520U (D0BG8PA)", "HP")


def _state() -> AgentState:
    return AgentState(
        active_category="Laptop",
        last_shown_candidates=[MSI, ACER, HP],
    )


def test_selects_previous_candidate_by_model_name_and_sets_focus_ready_code() -> None:
    result = resolve_product_reference("HP 14-ep1179TU Core 5 120U đi", _state())

    assert result.resolved is True
    assert result.code == "00921548"
    assert result.source in {"exact_name", "previous_candidate_name"}
    assert result.confidence >= 0.9


def test_focused_product_field_question_uses_focused_product() -> None:
    state = _state()
    state.focused_product_code = HP.code
    state.focused_product_name = HP.name

    result = resolve_product_reference("máy đó nặng bao kg", state)

    assert result.resolved is True
    assert result.code == HP.code
    assert result.source == "focused_product"


def test_ordinal_selection_resolves_second_candidate() -> None:
    result = resolve_product_reference("con thứ 2 đi", _state())

    assert result.resolved is True
    assert result.code == ACER.code
    assert result.source == "ordinal_selection"


def test_correction_resolves_to_previous_hp_and_can_replace_wrong_focus() -> None:
    state = _state()
    state.focused_product_code = MSI.code
    state.focused_product_name = MSI.name

    result = resolve_product_reference("máy HP cơ mà", state)

    assert result.resolved is True
    assert result.code == HP.code
    assert result.source == "correction"
    assert result.should_clear_focus is False


def test_correction_without_replacement_requests_focus_clear() -> None:
    state = _state()
    state.focused_product_code = MSI.code
    state.focused_product_name = MSI.name

    result = resolve_product_reference("không phải con đó", state)

    assert result.resolved is False
    assert result.source == "correction"
    assert result.should_clear_focus is True


def test_ambiguous_brand_reference_lists_only_previous_candidates() -> None:
    state = AgentState(
        active_category="Laptop",
        last_shown_candidates=[HP, HP_OTHER, MSI],
    )

    result = resolve_product_reference("máy HP cơ mà", state)

    assert result.resolved is False
    assert result.source == "ambiguous"
    assert [candidate.code for candidate in result.ambiguous_candidates] == [
        HP.code,
        HP_OTHER.code,
    ]


def test_agent_state_does_not_store_raw_catalog_context_in_candidate_refs() -> None:
    state = _state()

    assert state.last_shown_candidates[0].summary_specs == ("CPU Core 5", "RAM 16GB")
    assert not hasattr(state.last_shown_candidates[0], "context")
