"""
Unit tests for verification/utils/prompt_compressor.py

Covers:
- Compression reduces token count
- Compressed prompts still contain essential binary decision instructions
- Different compression levels produce expected results
- Token counting works correctly
"""

import pytest
from verification.utils.prompt_compressor import (
    PromptCompressor,
    CompressionResult,
    count_tokens,
    get_prompt_compressor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRICE_PROMPT = """You are a Price Accuracy Checker. Your task is to verify whether prices mentioned in the
sales draft response are accurate compared to the internal product database.

**CUSTOMER OBJECTION:**
iPhone 15 Pro Max giá bao nhiêu?

**DRAFT RESPONSE TO VERIFY:**
iPhone 15 Pro Max có giá 29,990,000 VND.

**INTERNAL DATABASE PRICES:**
iPhone 15 Pro Max: 29,990,000 VND

**VERIFICATION RULES:**
- PASS: All prices deviate ≤ 1% from database values
- FAIL: Any price deviates > 1% OR a required price is missing
- Critical threshold: >30% deviation = CRITICAL severity
- Tolerance for rounding: ±1%

**TASK:**
1. Extract every price mention from the draft response.
2. Cross-reference each price with the database records.
3. Calculate deviation percentage for each price found.
4. Determine overall PASS or FAIL.

**OUTPUT — respond with valid JSON only:**
```json
{
  "price_accuracy_pass": true,
  "price_issues": [],
  "overall_reasoning": "All prices are accurate."
}
```
"""

POLICY_PROMPT = """You are a Policy Authenticity Checker. Your task is to verify that all policy statements
in the sales draft response are accurate and sourced from official policy documents.

**DRAFT RESPONSE TO VERIFY:**
Sản phẩm được bảo hành 12 tháng.

**OFFICIAL POLICY DOCUMENTS:**
Apple Warranty: 12 months standard warranty.

**VERIFICATION RULES:**
- PASS: All policy statements are verified against official documents with proper citations
- FAIL: Any policy is fabricated, inaccurate, or lacks required citation
- Fabricated policy (not found in any document) = CRITICAL severity
- Inaccurate policy (found but details wrong) = MAJOR severity
- Incomplete policy (correct but missing details) = MINOR severity

**FORBIDDEN INDICATORS (suggest fabrication):**
tự bịa, không có trong hệ thống

**TASK:**
1. Identify all policy statements in the draft.
2. Verify each statement against the official policy documents.
3. Flag any fabricated, inaccurate, or uncited policies.
4. Determine overall PASS or FAIL.

**OUTPUT — respond with valid JSON only:**
```json
{
  "policy_authenticity_pass": true,
  "policy_issues": [],
  "overall_reasoning": "All policies verified."
}
```
"""

MINIMAL_PROMPT = "Answer yes or no: is the price correct?"


# ---------------------------------------------------------------------------
# Token counting tests
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") >= 0

    def test_single_word(self):
        # At least 1 token for any non-empty text
        assert count_tokens("hello") >= 1

    def test_longer_text_has_more_tokens(self):
        short = "yes"
        long = "yes " * 100
        assert count_tokens(long) > count_tokens(short)

    def test_returns_integer(self):
        result = count_tokens("some text here")
        assert isinstance(result, int)

    def test_approximation_reasonable(self):
        # 100 words should give roughly 100-200 tokens
        text = "word " * 100
        tokens = count_tokens(text)
        assert 80 <= tokens <= 250


# ---------------------------------------------------------------------------
# PromptCompressor construction
# ---------------------------------------------------------------------------

class TestPromptCompressorInit:
    def test_default_level_is_light(self):
        c = PromptCompressor()
        assert c.level == "light"

    def test_valid_levels(self):
        for level in ("none", "light", "aggressive"):
            c = PromptCompressor(level=level)
            assert c.level == level

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level must be one of"):
            PromptCompressor(level="extreme")

    def test_custom_token_limits(self):
        c = PromptCompressor(max_input_tokens=100, max_context_tokens=200)
        assert c.max_input_tokens == 100
        assert c.max_context_tokens == 200


# ---------------------------------------------------------------------------
# Compression level: none
# ---------------------------------------------------------------------------

class TestCompressionNone:
    def test_returns_unchanged(self):
        c = PromptCompressor(level="none")
        result = c.compress(PRICE_PROMPT)
        assert result.compressed == PRICE_PROMPT

    def test_compression_ratio_is_zero(self):
        c = PromptCompressor(level="none")
        result = c.compress(PRICE_PROMPT)
        assert result.compression_ratio == 0.0

    def test_tokens_saved_is_zero(self):
        c = PromptCompressor(level="none")
        result = c.compress(PRICE_PROMPT)
        assert result.tokens_saved == 0


# ---------------------------------------------------------------------------
# Compression level: light
# ---------------------------------------------------------------------------

class TestCompressionLight:
    def test_reduces_token_count(self):
        # Add lots of redundant whitespace
        padded = PRICE_PROMPT.replace("\n", "\n\n\n") + "   \n   \n   "
        c = PromptCompressor(level="light")
        result = c.compress(padded)
        assert result.compressed_tokens <= result.original_tokens

    def test_preserves_binary_decision_keywords(self):
        c = PromptCompressor(level="light")
        result = c.compress(PRICE_PROMPT)
        assert "PASS" in result.compressed
        assert "FAIL" in result.compressed

    def test_preserves_json_output_structure(self):
        c = PromptCompressor(level="light")
        result = c.compress(PRICE_PROMPT)
        assert "price_accuracy_pass" in result.compressed

    def test_no_trailing_spaces_on_lines(self):
        padded = "line one   \nline two   \nline three   \n"
        c = PromptCompressor(level="light")
        result = c.compress(padded)
        for line in result.compressed.splitlines():
            assert not line.endswith(" "), f"Trailing space found: {line!r}"

    def test_collapses_multiple_blank_lines(self):
        text = "section one\n\n\n\n\nsection two"
        c = PromptCompressor(level="light")
        result = c.compress(text)
        assert "\n\n\n" not in result.compressed


# ---------------------------------------------------------------------------
# Compression level: aggressive
# ---------------------------------------------------------------------------

class TestCompressionAggressive:
    def test_reduces_token_count_vs_original(self):
        c = PromptCompressor(level="aggressive")
        result = c.compress(PRICE_PROMPT)
        assert result.compressed_tokens < result.original_tokens

    def test_more_aggressive_than_light(self):
        c_light = PromptCompressor(level="light")
        c_agg = PromptCompressor(level="aggressive")
        light_result = c_light.compress(PRICE_PROMPT)
        agg_result = c_agg.compress(PRICE_PROMPT)
        assert agg_result.compressed_tokens <= light_result.compressed_tokens

    def test_preserves_pass_fail_decision_keywords(self):
        c = PromptCompressor(level="aggressive")
        result = c.compress(PRICE_PROMPT)
        # Essential binary decision keywords must survive
        assert "PASS" in result.compressed or "pass" in result.compressed.lower()
        assert "FAIL" in result.compressed or "fail" in result.compressed.lower()

    def test_preserves_json_output_keys(self):
        c = PromptCompressor(level="aggressive")
        result = c.compress(PRICE_PROMPT)
        assert "price_accuracy_pass" in result.compressed

    def test_policy_prompt_preserves_essential_content(self):
        c = PromptCompressor(level="aggressive")
        result = c.compress(POLICY_PROMPT)
        assert "policy_authenticity_pass" in result.compressed

    def test_minimal_prompt_not_broken(self):
        c = PromptCompressor(level="aggressive")
        result = c.compress(MINIMAL_PROMPT)
        assert len(result.compressed) > 0
        assert "yes or no" in result.compressed


# ---------------------------------------------------------------------------
# Level override in compress()
# ---------------------------------------------------------------------------

class TestCompressionLevelOverride:
    def test_override_none_on_aggressive_instance(self):
        c = PromptCompressor(level="aggressive")
        result = c.compress(PRICE_PROMPT, level="none")
        assert result.compressed == PRICE_PROMPT
        assert result.level == "none"

    def test_override_aggressive_on_none_instance(self):
        c = PromptCompressor(level="none")
        result_none = c.compress(PRICE_PROMPT, level="none")
        result_agg = c.compress(PRICE_PROMPT, level="aggressive")
        assert result_agg.compressed_tokens <= result_none.compressed_tokens

    def test_invalid_override_raises(self):
        c = PromptCompressor(level="light")
        with pytest.raises(ValueError):
            c.compress(PRICE_PROMPT, level="ultra")


# ---------------------------------------------------------------------------
# truncate_field
# ---------------------------------------------------------------------------

class TestTruncateField:
    def test_short_text_unchanged(self):
        c = PromptCompressor(max_input_tokens=500)
        text = "short text"
        assert c.truncate_field(text) == text

    def test_long_text_is_truncated(self):
        c = PromptCompressor(max_input_tokens=10)
        long_text = "word " * 200
        result = c.truncate_field(long_text)
        assert "[truncated]" in result
        assert count_tokens(result) <= 30  # some slack for the marker

    def test_truncated_text_shorter_than_original(self):
        c = PromptCompressor(max_input_tokens=20)
        long_text = "word " * 200
        result = c.truncate_field(long_text)
        assert len(result) < len(long_text)

    def test_custom_max_tokens_override(self):
        c = PromptCompressor(max_input_tokens=500)
        long_text = "word " * 200
        # With a very small override it should truncate
        result = c.truncate_field(long_text, max_tokens=5)
        assert "[truncated]" in result


# ---------------------------------------------------------------------------
# CompressionResult
# ---------------------------------------------------------------------------

class TestCompressionResult:
    def test_tokens_saved_non_negative(self):
        c = PromptCompressor(level="light")
        result = c.compress(PRICE_PROMPT)
        assert result.tokens_saved >= 0

    def test_compression_ratio_between_0_and_1(self):
        c = PromptCompressor(level="aggressive")
        result = c.compress(PRICE_PROMPT)
        assert 0.0 <= result.compression_ratio <= 1.0

    def test_original_preserved(self):
        c = PromptCompressor(level="aggressive")
        result = c.compress(PRICE_PROMPT)
        assert result.original == PRICE_PROMPT

    def test_level_recorded(self):
        c = PromptCompressor(level="light")
        result = c.compress(PRICE_PROMPT)
        assert result.level == "light"


# ---------------------------------------------------------------------------
# get_token_stats
# ---------------------------------------------------------------------------

class TestGetTokenStats:
    def test_returns_all_levels(self):
        c = PromptCompressor()
        stats = c.get_token_stats(PRICE_PROMPT)
        assert set(stats.keys()) == {"none", "light", "aggressive"}

    def test_none_has_most_tokens(self):
        c = PromptCompressor()
        stats = c.get_token_stats(PRICE_PROMPT)
        assert stats["none"] >= stats["light"]
        assert stats["light"] >= stats["aggressive"]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestGetPromptCompressor:
    def test_returns_compressor_instance(self):
        c = get_prompt_compressor()
        assert isinstance(c, PromptCompressor)

    def test_singleton_reuse(self):
        c1 = get_prompt_compressor()
        c2 = get_prompt_compressor()
        assert c1 is c2


# ---------------------------------------------------------------------------
# Integration: CachedPromptTemplates with compression
# ---------------------------------------------------------------------------

class TestCachedPromptTemplatesCompression:
    def test_compression_level_none_by_default(self):
        from verification.config.prompt_templates import CachedPromptTemplates
        cpt = CachedPromptTemplates()
        assert cpt.compressor.level == "none"

    def test_compression_level_light(self):
        from verification.config.prompt_templates import CachedPromptTemplates
        cpt = CachedPromptTemplates(compression_level="light")
        assert cpt.compressor.level == "light"

    def test_render_returns_string(self):
        from verification.config.prompt_templates import CachedPromptTemplates
        cpt = CachedPromptTemplates(compression_level="light")
        rendered = cpt.render(
            "price_accuracy_check",
            objection_text="test objection",
            draft_response="test draft",
            db_data="test db",
            price_tolerance="1",
            critical_threshold="30",
        )
        assert isinstance(rendered, str)
        assert len(rendered) > 0

    def test_last_compression_result_available(self):
        from verification.config.prompt_templates import CachedPromptTemplates
        cpt = CachedPromptTemplates(compression_level="light")
        cpt.render(
            "price_accuracy_check",
            objection_text="test",
            draft_response="test draft",
            db_data="db data",
            price_tolerance="1",
            critical_threshold="30",
        )
        result = cpt.get_last_compression_result()
        assert result is not None
        assert isinstance(result, CompressionResult)

    def test_compressor_property_accessible(self):
        from verification.config.prompt_templates import CachedPromptTemplates
        cpt = CachedPromptTemplates(compression_level="aggressive")
        assert isinstance(cpt.compressor, PromptCompressor)
        assert cpt.compressor.level == "aggressive"
