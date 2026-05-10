"""
Prompt Compression Utilities for Binary Verification Decisions

Provides compression strategies to reduce token usage in LLM prompts
for price accuracy, policy authenticity, and topic relevance checks.

Compression levels:
- "none":       No compression applied
- "light":      Remove redundant whitespace and normalize formatting
- "aggressive": Strip verbose preambles, truncate long inputs, compact mode
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _try_import_tiktoken():
    try:
        import tiktoken  # type: ignore
        return tiktoken
    except ImportError:
        return None


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for *text*.

    Uses tiktoken when available; falls back to a word-based approximation
    (1 token ≈ 0.75 words, i.e. words * 4/3) otherwise.
    """
    tiktoken = _try_import_tiktoken()
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: word-based approximation
    words = len(text.split())
    return max(1, int(words * 4 / 3))


# ---------------------------------------------------------------------------
# Compression result
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Holds the compressed prompt and token statistics."""

    original: str
    compressed: str
    level: str
    original_tokens: int = field(init=False)
    compressed_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.original_tokens = count_tokens(self.original)
        self.compressed_tokens = count_tokens(self.compressed)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return self.tokens_saved / self.original_tokens


# ---------------------------------------------------------------------------
# PromptCompressor
# ---------------------------------------------------------------------------

class PromptCompressor:
    """
    Compresses verification prompts to reduce LLM token usage.

    Supports three compression levels:
    - ``"none"``       – returns the prompt unchanged
    - ``"light"``      – normalises whitespace, removes blank lines
    - ``"aggressive"`` – strips verbose preambles, truncates long variable
                         sections, and applies compact-mode rewrites for
                         binary yes/no decisions

    Parameters
    ----------
    level:
        Default compression level (``"none"``, ``"light"``, or ``"aggressive"``).
    max_input_tokens:
        Maximum tokens allowed for long free-text inputs (``draft_response``,
        ``objection_text``, etc.) before truncation in aggressive mode.
    max_context_tokens:
        Maximum tokens for supporting context fields (``db_data``,
        ``policy_documents``) in aggressive mode.
    """

    LEVELS = ("none", "light", "aggressive")

    # Patterns that mark the start of verbose preamble sections
    _PREAMBLE_PATTERNS = [
        r"^You are a [^\n]+\.\s*Your task is to [^\n]+\.\n",
        r"^You are a [^\n]+\.\n",
    ]

    # Verbose instruction blocks that can be stripped in aggressive mode
    _VERBOSE_BLOCKS = [
        # "**TASK:**\n1. ...\n2. ...\n3. ..." style numbered lists
        r"\*\*TASK:\*\*\n(?:\d+\.[^\n]+\n)+",
        # Repeated "respond with valid JSON only" boilerplate
        r"\*\*OUTPUT — respond with valid JSON only:\*\*\n",
        r"Provide structured JSON response:\n",
    ]

    def __init__(
        self,
        level: str = "light",
        max_input_tokens: int = 300,
        max_context_tokens: int = 500,
    ) -> None:
        if level not in self.LEVELS:
            raise ValueError(f"level must be one of {self.LEVELS}, got {level!r}")
        self.level = level
        self.max_input_tokens = max_input_tokens
        self.max_context_tokens = max_context_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, prompt: str, level: Optional[str] = None) -> CompressionResult:
        """
        Compress *prompt* and return a :class:`CompressionResult`.

        Parameters
        ----------
        prompt:
            The full prompt string to compress.
        level:
            Override the instance-level compression level for this call.
        """
        effective_level = level if level is not None else self.level
        if effective_level not in self.LEVELS:
            raise ValueError(f"level must be one of {self.LEVELS}, got {effective_level!r}")

        if effective_level == "none":
            compressed = prompt
        elif effective_level == "light":
            compressed = self._light_compress(prompt)
        else:  # aggressive
            compressed = self._aggressive_compress(prompt)

        return CompressionResult(original=prompt, compressed=compressed, level=effective_level)

    def truncate_field(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate *text* so it fits within *max_tokens* (word-based approximation).

        A ``[truncated]`` marker is appended when truncation occurs.
        """
        limit = max_tokens if max_tokens is not None else self.max_input_tokens
        if count_tokens(text) <= limit:
            return text
        # Approximate: 1 token ≈ 0.75 words
        max_words = max(1, int(limit * 0.75))
        words = text.split()
        return " ".join(words[:max_words]) + " [truncated]"

    def get_token_stats(self, prompt: str) -> Dict[str, int]:
        """Return token count statistics for *prompt* at all compression levels."""
        return {
            level: count_tokens(self.compress(prompt, level=level).compressed)
            for level in self.LEVELS
        }

    # ------------------------------------------------------------------
    # Compression implementations
    # ------------------------------------------------------------------

    def _light_compress(self, prompt: str) -> str:
        """Remove redundant whitespace and normalise blank lines."""
        # Collapse runs of 3+ blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", prompt)
        # Strip trailing spaces on each line
        text = re.sub(r"[ \t]+\n", "\n", text)
        # Collapse multiple spaces (but not leading indentation)
        text = re.sub(r"(?<=\S) {2,}", " ", text)
        return text.strip()

    def _aggressive_compress(self, prompt: str) -> str:
        """Apply all compression techniques for maximum token reduction."""
        text = self._light_compress(prompt)

        # 1. Strip verbose role preambles
        for pattern in self._PREAMBLE_PATTERNS:
            text = re.sub(pattern, "", text, count=1, flags=re.MULTILINE)

        # 2. Remove verbose instruction boilerplate
        for pattern in self._VERBOSE_BLOCKS:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)

        # 3. Compact binary decision instructions
        text = self._compact_binary_instructions(text)

        # 4. Truncate long free-text sections
        text = self._truncate_long_sections(text)

        # 5. Final whitespace cleanup
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compact_binary_instructions(self, text: str) -> str:
        """
        Replace multi-line PASS/FAIL rule blocks with compact single-line versions.

        E.g.::

            **VERIFICATION RULES:**
            - PASS: All prices deviate ≤ {price_tolerance}% from database values
            - FAIL: Any price deviates > {price_tolerance}% OR a required price is missing
            - Critical threshold: >{critical_threshold}% deviation = CRITICAL severity
            - Tolerance for rounding: ±{price_tolerance}%

        becomes::

            Rules: PASS if deviation≤{price_tolerance}%; FAIL otherwise. Critical>{critical_threshold}%.
        """
        # Compact verbose VERIFICATION RULES blocks
        def _compact_rules(m: re.Match) -> str:
            block = m.group(0)
            # Extract PASS/FAIL lines
            pass_line = re.search(r"- PASS: (.+)", block)
            fail_line = re.search(r"- FAIL: (.+)", block)
            critical_line = re.search(r"- Critical[^:]*: (.+)", block)
            parts = []
            if pass_line:
                parts.append(f"PASS: {pass_line.group(1).strip()}")
            if fail_line:
                parts.append(f"FAIL: {fail_line.group(1).strip()}")
            if critical_line:
                parts.append(f"Critical: {critical_line.group(1).strip()}")
            if parts:
                return "Rules: " + "; ".join(parts) + "\n"
            return block

        text = re.sub(
            r"\*\*VERIFICATION RULES:\*\*\n(?:- [^\n]+\n)+",
            _compact_rules,
            text,
        )

        # Compact "**CRITICAL RULES:**" blocks similarly
        text = re.sub(
            r"\*\*CRITICAL RULES:\*\*\n(?:- [^\n]+\n)+",
            "Note: Be strict, provide specific feedback, flag critical issues.\n",
            text,
        )

        return text

    def _truncate_long_sections(self, text: str) -> str:
        """
        Truncate the content of known long-text placeholder sections.

        Looks for patterns like ``**DRAFT RESPONSE TO VERIFY:**\n{content}``
        where content is already substituted (i.e. not a ``{variable}``).
        """
        # Sections that may contain long free-text content
        section_patterns = [
            (r"(\*\*DRAFT RESPONSE TO VERIFY:\*\*\n)((?:(?!\*\*).)+)", self.max_input_tokens),
            (r"(\*\*CUSTOMER OBJECTION:\*\*\n)((?:(?!\*\*).)+)", self.max_input_tokens),
            (r"(\*\*FAILED DRAFT RESPONSE:\*\*\n)((?:(?!\*\*).)+)", self.max_input_tokens),
            (r"(\*\*INTERNAL DATABASE PRICES:\*\*\n)((?:(?!\*\*).)+)", self.max_context_tokens),
            (r"(\*\*OFFICIAL POLICY DOCUMENTS:\*\*\n)((?:(?!\*\*).)+)", self.max_context_tokens),
        ]

        for pattern, max_tok in section_patterns:
            def _truncate_match(m: re.Match, _max=max_tok) -> str:
                header = m.group(1)
                content = m.group(2)
                # Only truncate if content doesn't look like an unrendered template variable
                if re.fullmatch(r"\s*\{[^}]+\}\s*", content.strip()):
                    return m.group(0)
                return header + self.truncate_field(content.strip(), _max) + "\n"

            text = re.sub(pattern, _truncate_match, text, flags=re.DOTALL)

        return text


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_compressor: Optional[PromptCompressor] = None


def get_prompt_compressor(
    level: str = "light",
    max_input_tokens: int = 300,
    max_context_tokens: int = 500,
) -> PromptCompressor:
    """Return the module-level :class:`PromptCompressor` singleton."""
    global _default_compressor
    if _default_compressor is None:
        _default_compressor = PromptCompressor(
            level=level,
            max_input_tokens=max_input_tokens,
            max_context_tokens=max_context_tokens,
        )
    return _default_compressor
