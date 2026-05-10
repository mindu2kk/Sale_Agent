"""
Input Sanitization for Objection Text and Draft Responses - Task 7.2.1

Provides async-capable input sanitization to protect the verification workflow
from malicious or malformed inputs:
- Strip/escape HTML injection and dangerous content
- Validate input length limits
- Detect and reject prompt injection patterns
- Normalize excessive whitespace

Components:
- InputSanitizationError: structured exception with violation details
- SanitizationResult: Pydantic model with sanitized text, modification flag, warnings
- InputSanitizer: class with sync and async sanitize methods
- sanitize_objection_text() / sanitize_draft_response(): convenience functions

Requirements:
- 7.2.1: Input sanitization for objection text with async validation
- 8.1: Error handling with structured context
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("verification.input_sanitizer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default length limits (characters)
MAX_OBJECTION_TEXT_LENGTH = 5000
MAX_DRAFT_RESPONSE_LENGTH = 10000

# Prompt injection patterns (case-insensitive)
_PROMPT_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(?:previous|all|prior|\w+\s+prior|\w+\s+previous)\s+instructions?",
    r"disregard\s+(?:previous|all|prior|\w+\s+prior|\w+\s+previous)\s+instructions?",
    r"forget\s+(?:previous|all|prior|\w+\s+prior|\w+\s+previous)\s+instructions?",
    r"override\s+(?:previous|all|prior|\w+\s+prior|\w+\s+previous)\s+instructions?",
    r"\bsystem\s*:",
    r"\buser\s*:",
    r"\bassistant\s*:",
    r"\[system\]",
    r"\[user\]",
    r"\[assistant\]",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"act\s+as\s+(?:a|an)\s+\w+",
    r"pretend\s+(?:you\s+are|to\s+be)\s+",
    r"new\s+instructions?\s*:",
    r"jailbreak",
    r"DAN\b",  # "Do Anything Now" jailbreak
    r"prompt\s+injection",
]

_COMPILED_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _PROMPT_INJECTION_PATTERNS
]

# HTML/script tags to detect
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>", re.IGNORECASE)

# Excessive whitespace: 3+ consecutive newlines or 4+ consecutive spaces
_EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")
_EXCESSIVE_SPACES_PATTERN = re.compile(r" {4,}")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SanitizationViolationType(str, Enum):
    """Types of sanitization violations."""
    PROMPT_INJECTION = "prompt_injection"
    HTML_INJECTION = "html_injection"
    LENGTH_EXCEEDED = "length_exceeded"
    EXCESSIVE_WHITESPACE = "excessive_whitespace"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class InputSanitizationError(Exception):
    """
    Raised when input fails sanitization and cannot be safely processed.

    Attributes:
        violation_type: The type of violation detected.
        field_name: The name of the input field that failed.
        details: Human-readable description of the violation.
        original_length: Length of the original input (for length violations).
        max_length: Maximum allowed length (for length violations).
    """

    def __init__(
        self,
        violation_type: SanitizationViolationType,
        field_name: str,
        details: str,
        original_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.violation_type = violation_type
        self.field_name = field_name
        self.details = details
        self.original_length = original_length
        self.max_length = max_length
        super().__init__(
            f"InputSanitizationError [{violation_type.value}] on field '{field_name}': {details}"
        )


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class SanitizationResult(BaseModel):
    """
    Result of a sanitization operation.

    Attributes:
        sanitized_text: The cleaned, safe text ready for processing.
        was_modified: True if the original text was altered during sanitization.
        warnings: List of non-fatal issues found and corrected (e.g. whitespace).
    """

    sanitized_text: str = Field(description="Cleaned text after sanitization")
    was_modified: bool = Field(
        default=False,
        description="Whether the original text was modified during sanitization",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal sanitization warnings (e.g. whitespace normalized)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sanitized_text": "iPhone quá đắt so với Samsung",
                "was_modified": False,
                "warnings": [],
            }
        }


# ---------------------------------------------------------------------------
# InputSanitizer
# ---------------------------------------------------------------------------

class InputSanitizer:
    """
    Sanitizes objection text and draft responses for safe workflow processing.

    Performs:
    1. Length validation — raises InputSanitizationError if exceeded
    2. Prompt injection detection — raises InputSanitizationError if found
    3. HTML tag escaping — modifies text and records warning
    4. Excessive whitespace normalization — modifies text and records warning

    Usage::

        sanitizer = InputSanitizer()

        # Sync
        result = sanitizer.sanitize(text, field_name="objection_text", max_length=5000)

        # Async
        result = await sanitizer.async_sanitize(text, field_name="objection_text", max_length=5000)
    """

    def __init__(
        self,
        max_objection_length: int = MAX_OBJECTION_TEXT_LENGTH,
        max_draft_length: int = MAX_DRAFT_RESPONSE_LENGTH,
    ) -> None:
        self.max_objection_length = max_objection_length
        self.max_draft_length = max_draft_length

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def sanitize(
        self,
        text: str,
        field_name: str = "input",
        max_length: Optional[int] = None,
    ) -> SanitizationResult:
        """
        Sanitize a text input synchronously.

        Args:
            text: The raw input text to sanitize.
            field_name: Name of the field (used in error messages).
            max_length: Maximum allowed character count. If None, no limit applied.

        Returns:
            SanitizationResult with sanitized text and modification metadata.

        Raises:
            InputSanitizationError: If a hard violation is detected (length exceeded
                or prompt injection found).
        """
        warnings: List[str] = []
        was_modified = False
        current = text

        # 1. Length check (hard failure)
        if max_length is not None and len(current) > max_length:
            raise InputSanitizationError(
                violation_type=SanitizationViolationType.LENGTH_EXCEEDED,
                field_name=field_name,
                details=(
                    f"Input length {len(current)} exceeds maximum {max_length} characters"
                ),
                original_length=len(current),
                max_length=max_length,
            )

        # 2. Prompt injection detection (hard failure)
        matched_pattern = self._detect_prompt_injection(current)
        if matched_pattern is not None:
            raise InputSanitizationError(
                violation_type=SanitizationViolationType.PROMPT_INJECTION,
                field_name=field_name,
                details=(
                    f"Prompt injection pattern detected: '{matched_pattern}'"
                ),
            )

        # 3. HTML tag escaping (soft — modify and warn)
        if _HTML_TAG_PATTERN.search(current):
            current = html.escape(current, quote=True)
            was_modified = True
            warnings.append(
                "HTML tags detected and escaped to prevent injection"
            )

        # 4. Excessive whitespace normalization (soft — modify and warn)
        normalized, ws_modified = self._normalize_whitespace(current)
        if ws_modified:
            current = normalized
            was_modified = True
            warnings.append("Excessive whitespace normalized")

        logger.debug(
            "Sanitized field '%s': was_modified=%s, warnings=%d",
            field_name,
            was_modified,
            len(warnings),
        )

        return SanitizationResult(
            sanitized_text=current,
            was_modified=was_modified,
            warnings=warnings,
        )

    def sanitize_objection(self, text: str) -> SanitizationResult:
        """Sanitize objection text with the configured max length."""
        return self.sanitize(
            text,
            field_name="objection_text",
            max_length=self.max_objection_length,
        )

    def sanitize_draft(self, text: str) -> SanitizationResult:
        """Sanitize draft response text with the configured max length."""
        return self.sanitize(
            text,
            field_name="draft_response",
            max_length=self.max_draft_length,
        )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def async_sanitize(
        self,
        text: str,
        field_name: str = "input",
        max_length: Optional[int] = None,
    ) -> SanitizationResult:
        """
        Sanitize a text input asynchronously.

        Runs the synchronous sanitize() in the default executor so it does not
        block the event loop for large inputs.

        Args:
            text: The raw input text to sanitize.
            field_name: Name of the field (used in error messages).
            max_length: Maximum allowed character count.

        Returns:
            SanitizationResult with sanitized text and modification metadata.

        Raises:
            InputSanitizationError: If a hard violation is detected.
        """
        loop = asyncio.get_event_loop()
        # Run CPU-bound sanitization in thread pool to avoid blocking event loop
        result = await loop.run_in_executor(
            None,
            lambda: self.sanitize(text, field_name=field_name, max_length=max_length),
        )
        return result

    async def async_sanitize_objection(self, text: str) -> SanitizationResult:
        """Async sanitize objection text with the configured max length."""
        return await self.async_sanitize(
            text,
            field_name="objection_text",
            max_length=self.max_objection_length,
        )

    async def async_sanitize_draft(self, text: str) -> SanitizationResult:
        """Async sanitize draft response text with the configured max length."""
        return await self.async_sanitize(
            text,
            field_name="draft_response",
            max_length=self.max_draft_length,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_prompt_injection(self, text: str) -> Optional[str]:
        """
        Scan text for prompt injection patterns.

        Returns the matched pattern string if found, else None.
        """
        for pattern in _COMPILED_INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def _normalize_whitespace(self, text: str) -> tuple[str, bool]:
        """
        Normalize excessive whitespace.

        - Collapse 3+ consecutive newlines to 2
        - Collapse 4+ consecutive spaces to 1

        Returns (normalized_text, was_modified).
        """
        modified = False

        normalized = _EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", text)
        if normalized != text:
            modified = True

        result = _EXCESSIVE_SPACES_PATTERN.sub(" ", normalized)
        if result != normalized:
            modified = True

        return result, modified


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_sanitizer: Optional[InputSanitizer] = None


def get_input_sanitizer() -> InputSanitizer:
    """Return the module-level singleton InputSanitizer."""
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = InputSanitizer()
    return _default_sanitizer


def reset_input_sanitizer() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_sanitizer
    _default_sanitizer = None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def sanitize_objection_text(text: str) -> SanitizationResult:
    """
    Sanitize customer objection text using default limits.

    Raises:
        InputSanitizationError: On length exceeded or prompt injection.
    """
    return get_input_sanitizer().sanitize_objection(text)


def sanitize_draft_response(text: str) -> SanitizationResult:
    """
    Sanitize draft response text using default limits.

    Raises:
        InputSanitizationError: On length exceeded or prompt injection.
    """
    return get_input_sanitizer().sanitize_draft(text)


async def async_sanitize_objection_text(text: str) -> SanitizationResult:
    """Async version of sanitize_objection_text."""
    return await get_input_sanitizer().async_sanitize_objection(text)


async def async_sanitize_draft_response(text: str) -> SanitizationResult:
    """Async version of sanitize_draft_response."""
    return await get_input_sanitizer().async_sanitize_draft(text)


__all__ = [
    "SanitizationViolationType",
    "InputSanitizationError",
    "SanitizationResult",
    "InputSanitizer",
    "get_input_sanitizer",
    "reset_input_sanitizer",
    "sanitize_objection_text",
    "sanitize_draft_response",
    "async_sanitize_objection_text",
    "async_sanitize_draft_response",
    "MAX_OBJECTION_TEXT_LENGTH",
    "MAX_DRAFT_RESPONSE_LENGTH",
]
