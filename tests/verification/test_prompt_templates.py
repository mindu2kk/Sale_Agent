"""
Unit tests for PromptTemplateManager (task 1.3.3).

Tests cover:
- Loading templates from the real prompts.yaml
- get_template() returns raw template strings
- render() substitutes variables correctly
- Missing variable raises PromptTemplateError
- Unknown template name raises PromptTemplateError
- Caching: second call does not re-read disk
- reload() clears cache
- list_templates() returns expected names
- All four binary checker templates are present and renderable
"""

import pytest
from pathlib import Path

from verification.config.prompt_templates import (
    PromptTemplateManager,
    PromptTemplateError,
)

PROMPTS_PATH = Path(__file__).parent.parent.parent / "verification" / "config" / "prompts.yaml"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def manager() -> PromptTemplateManager:
    return PromptTemplateManager(PROMPTS_PATH)


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

def test_prompts_yaml_exists():
    assert PROMPTS_PATH.exists(), "prompts.yaml must exist"


def test_get_template_returns_string(manager):
    tmpl = manager.get_template("price_accuracy_check")
    assert isinstance(tmpl, str)
    assert len(tmpl) > 50


def test_get_template_unknown_raises(manager):
    with pytest.raises(PromptTemplateError, match="not found"):
        manager.get_template("nonexistent_template_xyz")


# ---------------------------------------------------------------------------
# All four required binary checker templates exist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "price_accuracy_check",
    "policy_authenticity_check",
    "topic_relevance_check",
    "correction_feedback",
])
def test_required_template_exists(manager, name):
    tmpl = manager.get_template(name)
    assert tmpl.strip(), f"Template '{name}' must not be empty"


# ---------------------------------------------------------------------------
# Template content checks — binary PASS/FAIL keywords
# ---------------------------------------------------------------------------

def test_price_accuracy_check_contains_pass_fail(manager):
    tmpl = manager.get_template("price_accuracy_check")
    assert "PASS" in tmpl
    assert "FAIL" in tmpl
    assert "price_accuracy_pass" in tmpl


def test_policy_authenticity_check_contains_pass_fail(manager):
    tmpl = manager.get_template("policy_authenticity_check")
    assert "PASS" in tmpl
    assert "FAIL" in tmpl
    assert "policy_authenticity_pass" in tmpl


def test_topic_relevance_check_contains_pass_fail(manager):
    tmpl = manager.get_template("topic_relevance_check")
    assert "PASS" in tmpl
    assert "FAIL" in tmpl
    assert "topic_relevance_pass" in tmpl


def test_correction_feedback_contains_json_output(manager):
    tmpl = manager.get_template("correction_feedback")
    assert "correction_priority" in tmpl
    assert "critical_fixes_required" in tmpl


# ---------------------------------------------------------------------------
# JSON output structure matches Pydantic issue models
# ---------------------------------------------------------------------------

def test_price_accuracy_check_json_fields(manager):
    tmpl = manager.get_template("price_accuracy_check")
    for field in ("product_name", "mentioned_price", "actual_price",
                  "deviation_percent", "severity", "explanation"):
        assert field in tmpl, f"price_accuracy_check template missing field '{field}'"


def test_policy_authenticity_check_json_fields(manager):
    tmpl = manager.get_template("policy_authenticity_check")
    for field in ("mentioned_policy", "policy_type", "is_fabricated",
                  "is_inaccurate", "severity", "explanation"):
        assert field in tmpl, f"policy_authenticity_check template missing field '{field}'"


def test_topic_relevance_check_json_fields(manager):
    tmpl = manager.get_template("topic_relevance_check")
    for field in ("objection_intent", "response_coverage", "missing_aspects",
                  "empathy_score", "severity", "explanation"):
        assert field in tmpl, f"topic_relevance_check template missing field '{field}'"


# ---------------------------------------------------------------------------
# render() — variable substitution
# ---------------------------------------------------------------------------

def test_render_price_accuracy_check(manager):
    rendered = manager.render(
        "price_accuracy_check",
        objection_text="iPhone 15 giá bao nhiêu?",
        draft_response="iPhone 15 giá 25,000,000 VND",
        db_data="iPhone 15: 24,990,000 VND",
        price_tolerance="1",
        critical_threshold="30",
    )
    assert "iPhone 15 giá bao nhiêu?" in rendered
    assert "24,990,000 VND" in rendered
    assert "{objection_text}" not in rendered
    assert "{db_data}" not in rendered


def test_render_policy_authenticity_check(manager):
    rendered = manager.render(
        "policy_authenticity_check",
        draft_response="Bảo hành 2 năm toàn bộ sản phẩm",
        policy_documents="Apple: 1 năm bảo hành",
        forbidden_phrases="tự bịa, không có trong hệ thống",
    )
    assert "Bảo hành 2 năm" in rendered
    assert "Apple: 1 năm" in rendered
    assert "{policy_documents}" not in rendered


def test_render_topic_relevance_check(manager):
    rendered = manager.render(
        "topic_relevance_check",
        objection_text="Sản phẩm này có tốt không?",
        draft_response="Sản phẩm rất tốt, giá hợp lý",
        relevance_threshold="0.7",
        empathy_phrases="tôi hiểu, tôi đồng ý",
    )
    assert "Sản phẩm này có tốt không?" in rendered
    assert "0.7" in rendered
    assert "{objection_text}" not in rendered


def test_render_correction_feedback(manager):
    rendered = manager.render(
        "correction_feedback",
        objection_text="Giá iPhone 15?",
        failed_draft="iPhone 15 giá 30 triệu",
        verification_issues="Price deviation 20%",
    )
    assert "Giá iPhone 15?" in rendered
    assert "Price deviation 20%" in rendered
    assert "{verification_issues}" not in rendered


# ---------------------------------------------------------------------------
# render() — error cases
# ---------------------------------------------------------------------------

def test_render_missing_variable_raises(manager):
    with pytest.raises(PromptTemplateError, match="requires variables"):
        manager.render(
            "price_accuracy_check",
            # missing most required variables
            objection_text="test",
        )


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

def test_cache_populated_after_first_call(manager):
    assert "price_accuracy_check" not in manager._cache
    manager.get_template("price_accuracy_check")
    assert "price_accuracy_check" in manager._cache


def test_second_call_uses_cache(manager, monkeypatch):
    manager.get_template("price_accuracy_check")  # populate cache
    # Patch _resolve_template to detect if it's called again
    calls = []
    original = manager._resolve_template
    monkeypatch.setattr(manager, "_resolve_template", lambda n: calls.append(n) or original(n))
    manager.get_template("price_accuracy_check")
    assert calls == [], "Cache should prevent _resolve_template from being called again"


def test_reload_clears_cache(manager):
    manager.get_template("price_accuracy_check")
    assert "price_accuracy_check" in manager._cache
    manager.reload()
    assert manager._cache == {}
    assert manager._raw is None


# ---------------------------------------------------------------------------
# list_templates()
# ---------------------------------------------------------------------------

def test_list_templates_includes_required(manager):
    names = manager.list_templates()
    for required in ("price_accuracy_check", "policy_authenticity_check",
                     "topic_relevance_check", "correction_feedback"):
        assert required in names, f"list_templates() missing '{required}'"


def test_list_templates_returns_sorted(manager):
    names = manager.list_templates()
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------

def test_missing_file_raises(tmp_path):
    mgr = PromptTemplateManager(tmp_path / "nonexistent.yaml")
    with pytest.raises(PromptTemplateError, match="not found"):
        mgr.get_template("anything")
