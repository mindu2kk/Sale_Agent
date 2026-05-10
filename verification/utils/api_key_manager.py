"""
Secure API Key Management for LLM Services - Task 7.2.2

Provides secure API key management with rotation support for OpenAI/Anthropic:
- API key storage from environment variables only (never hardcoded)
- Key rotation support without service restart
- Key masking — never log or expose full API keys (format: sk-...****)
- Key format validation before use
- Multiple key support with round-robin and fallback strategies
- Sync and async get_key() methods

Components:
- ApiKeyError: structured exception for key management failures
- MaskedApiKey: Pydantic model for safe display (masked, provider, is_valid)
- ApiKeyManager: class with sync/async get_key(), rotate(), add_key()
- get_api_key_manager(): singleton factory

Requirements:
- 7.2.2: Secure API key management for LLM services with rotation
- 8.5: All errors logged with correlation IDs
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from enum import Enum
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger("verification.api_key_manager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported LLM providers
SUPPORTED_PROVIDERS = ("openai", "anthropic")

# Environment variable names per provider (supports multiple keys via _2, _3 suffixes)
_PROVIDER_ENV_VARS: Dict[str, List[str]] = {
    "openai": ["OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"],
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2", "ANTHROPIC_API_KEY_3"],
}

# Key format validation patterns
# OpenAI keys start with "sk-" but NOT "sk-ant-" (which is Anthropic)
_KEY_PATTERNS: Dict[str, re.Pattern] = {
    "openai": re.compile(r"^sk-(?!ant-)[A-Za-z0-9\-_]{20,}$"),
    "anthropic": re.compile(r"^sk-ant-[A-Za-z0-9\-_]{20,}$"),
}

# Mask: show prefix + last 4 chars, hide the middle
_MASK_SUFFIX_LEN = 4
_MASK_PLACEHOLDER = "****"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class KeySelectionStrategy(str, Enum):
    """Strategy for selecting among multiple keys."""
    ROUND_ROBIN = "round_robin"   # Cycle through keys in order
    FALLBACK = "fallback"         # Use first valid key; fall back on error


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ApiKeyError(Exception):
    """
    Raised when API key management fails.

    Attributes:
        provider: The LLM provider name (e.g. "openai", "anthropic").
        reason: Human-readable description of the failure.
    """

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"ApiKeyError [{provider}]: {reason}")


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class MaskedApiKey(BaseModel):
    """
    Safe representation of an API key for logging and display.

    The full key is never stored here — only the masked form.

    Attributes:
        masked: Masked key string, e.g. ``sk-abc...****``.
        provider: LLM provider name (``"openai"`` or ``"anthropic"``).
        is_valid: Whether the key passed format validation.
    """

    masked: str = Field(description="Masked key for safe display, e.g. sk-abc...****")
    provider: str = Field(description="LLM provider name")
    is_valid: bool = Field(description="Whether the key passed format validation")

    class Config:
        json_schema_extra = {
            "example": {
                "masked": "sk-abc...****",
                "provider": "openai",
                "is_valid": True,
            }
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    """
    Mask an API key for safe logging/display.

    Shows the prefix up to the first 10 characters, then ``...****``.
    If the key is too short to mask meaningfully, returns ``****``.

    Examples::

        _mask_key("sk-abcdefghijklmnopqrstuvwxyz")  # "sk-abcdefg...****"
        _mask_key("sk-ant-api03-xyz123")             # "sk-ant-api...****"
        _mask_key("short")                           # "****"
    """
    if len(key) <= _MASK_SUFFIX_LEN + 4:
        return _MASK_PLACEHOLDER
    prefix_len = min(10, len(key) - _MASK_SUFFIX_LEN - 3)
    return f"{key[:prefix_len]}...{_MASK_PLACEHOLDER}"


def _validate_key_format(key: str, provider: str) -> bool:
    """
    Validate API key format for the given provider.

    Returns True if the key matches the expected pattern, False otherwise.
    Unknown providers always return False.
    """
    pattern = _KEY_PATTERNS.get(provider)
    if pattern is None:
        return False
    return bool(pattern.match(key))


def _load_keys_from_env(provider: str) -> List[str]:
    """
    Load all non-empty API keys for a provider from environment variables.

    Reads the primary env var and up to 2 additional numbered variants
    (e.g. OPENAI_API_KEY, OPENAI_API_KEY_2, OPENAI_API_KEY_3).

    Returns a list of raw key strings (may be empty if none are set).
    """
    env_vars = _PROVIDER_ENV_VARS.get(provider, [])
    keys: List[str] = []
    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if value:
            keys.append(value)
    return keys


# ---------------------------------------------------------------------------
# ApiKeyManager
# ---------------------------------------------------------------------------

class ApiKeyManager:
    """
    Secure API key manager for LLM services.

    Features:
    - Reads keys from environment variables only (never hardcoded)
    - Supports multiple keys per provider with round-robin or fallback selection
    - Key rotation without service restart via ``rotate()`` / ``add_key()``
    - Keys are never logged — only masked representations are exposed
    - Format validation before returning a key

    Usage::

        manager = ApiKeyManager()

        # Sync
        key = manager.get_key("openai")

        # Async
        key = await manager.async_get_key("openai")

        # Rotate to a new key (e.g. after a 401 response)
        manager.rotate("openai")

        # Add a new key at runtime (e.g. from a secrets manager)
        manager.add_key("openai", new_key)

        # Inspect without exposing the real key
        info = manager.get_masked_key("openai")
        print(info.masked)   # "sk-abcdefg...****"
    """

    def __init__(
        self,
        strategy: KeySelectionStrategy = KeySelectionStrategy.ROUND_ROBIN,
        auto_load_env: bool = True,
    ) -> None:
        """
        Initialise the ApiKeyManager.

        Args:
            strategy: Key selection strategy when multiple keys are available.
            auto_load_env: If True, load keys from environment variables on init.
        """
        self._strategy = strategy
        self._lock = threading.Lock()

        # Internal storage: provider → list of raw keys
        self._keys: Dict[str, List[str]] = {p: [] for p in SUPPORTED_PROVIDERS}
        # Round-robin index per provider
        self._rr_index: Dict[str, int] = {p: 0 for p in SUPPORTED_PROVIDERS}

        if auto_load_env:
            self._load_all_from_env()

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def get_key(self, provider: str) -> str:
        """
        Return a valid API key for the given provider.

        Selects a key according to the configured strategy. Validates the key
        format before returning it.

        Args:
            provider: LLM provider name (``"openai"`` or ``"anthropic"``).

        Returns:
            Raw API key string.

        Raises:
            ApiKeyError: If no keys are configured, all keys are invalid, or
                the provider is not supported.
        """
        self._require_supported_provider(provider)
        with self._lock:
            keys = self._keys[provider]
            if not keys:
                raise ApiKeyError(
                    provider=provider,
                    reason=(
                        f"No API key configured for provider '{provider}'. "
                        f"Set the {_PROVIDER_ENV_VARS.get(provider, ['?'])[0]} "
                        f"environment variable."
                    ),
                )

            key = self._select_key(provider, keys)
            if not _validate_key_format(key, provider):
                masked = _mask_key(key)
                logger.warning(
                    "API key for provider '%s' failed format validation: %s",
                    provider,
                    masked,
                )
                raise ApiKeyError(
                    provider=provider,
                    reason=(
                        f"API key {masked} failed format validation for provider '{provider}'."
                    ),
                )

            logger.debug(
                "Returning API key for provider '%s': %s",
                provider,
                _mask_key(key),
            )
            return key

    def rotate(self, provider: str) -> None:
        """
        Advance to the next key for the given provider (round-robin rotation).

        This is a no-op if only one key is configured. Useful after receiving
        a 401/403 response to try the next available key.

        Args:
            provider: LLM provider name.

        Raises:
            ApiKeyError: If the provider is not supported or has no keys.
        """
        self._require_supported_provider(provider)
        with self._lock:
            keys = self._keys[provider]
            if not keys:
                raise ApiKeyError(
                    provider=provider,
                    reason=f"Cannot rotate — no keys configured for '{provider}'.",
                )
            self._rr_index[provider] = (self._rr_index[provider] + 1) % len(keys)
            logger.info(
                "Rotated API key for provider '%s' to index %d/%d",
                provider,
                self._rr_index[provider],
                len(keys),
            )

    def add_key(self, provider: str, key: str) -> None:
        """
        Add a new API key for the given provider at runtime.

        The key is validated before being stored. This allows integrating with
        external secrets managers without restarting the service.

        Args:
            provider: LLM provider name.
            key: Raw API key string.

        Raises:
            ApiKeyError: If the provider is not supported or the key is invalid.
        """
        self._require_supported_provider(provider)
        if not _validate_key_format(key, provider):
            raise ApiKeyError(
                provider=provider,
                reason=(
                    f"Cannot add key {_mask_key(key)} — "
                    f"failed format validation for provider '{provider}'."
                ),
            )
        with self._lock:
            if key not in self._keys[provider]:
                self._keys[provider].append(key)
                logger.info(
                    "Added new API key for provider '%s': %s (total: %d)",
                    provider,
                    _mask_key(key),
                    len(self._keys[provider]),
                )

    def remove_key(self, provider: str, key: str) -> bool:
        """
        Remove a specific API key for the given provider.

        Args:
            provider: LLM provider name.
            key: Raw API key string to remove.

        Returns:
            True if the key was found and removed, False otherwise.
        """
        self._require_supported_provider(provider)
        with self._lock:
            keys = self._keys[provider]
            if key in keys:
                keys.remove(key)
                # Reset round-robin index to avoid out-of-bounds
                if keys:
                    self._rr_index[provider] = self._rr_index[provider] % len(keys)
                else:
                    self._rr_index[provider] = 0
                logger.info(
                    "Removed API key for provider '%s': %s (remaining: %d)",
                    provider,
                    _mask_key(key),
                    len(keys),
                )
                return True
            return False

    def reload_from_env(self, provider: Optional[str] = None) -> None:
        """
        Reload API keys from environment variables.

        Useful for picking up rotated keys injected into the environment
        without restarting the process.

        Args:
            provider: If given, reload only this provider. Otherwise reload all.
        """
        providers = [provider] if provider else list(SUPPORTED_PROVIDERS)
        for p in providers:
            self._require_supported_provider(p)
            new_keys = _load_keys_from_env(p)
            with self._lock:
                self._keys[p] = new_keys
                self._rr_index[p] = 0
            logger.info(
                "Reloaded %d API key(s) for provider '%s' from environment",
                len(new_keys),
                p,
            )

    def get_masked_key(self, provider: str) -> MaskedApiKey:
        """
        Return a masked representation of the current key for the given provider.

        Safe to log or display — the full key is never exposed.

        Args:
            provider: LLM provider name.

        Returns:
            MaskedApiKey with masked display, provider, and validity flag.

        Raises:
            ApiKeyError: If the provider is not supported or has no keys.
        """
        self._require_supported_provider(provider)
        with self._lock:
            keys = self._keys[provider]
            if not keys:
                raise ApiKeyError(
                    provider=provider,
                    reason=f"No API key configured for provider '{provider}'.",
                )
            key = self._select_key(provider, keys)
            is_valid = _validate_key_format(key, provider)
            return MaskedApiKey(
                masked=_mask_key(key),
                provider=provider,
                is_valid=is_valid,
            )

    def key_count(self, provider: str) -> int:
        """Return the number of keys configured for the given provider."""
        self._require_supported_provider(provider)
        with self._lock:
            return len(self._keys[provider])

    def has_valid_key(self, provider: str) -> bool:
        """Return True if at least one valid key is configured for the provider."""
        self._require_supported_provider(provider)
        with self._lock:
            return any(
                _validate_key_format(k, provider) for k in self._keys[provider]
            )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def async_get_key(self, provider: str) -> str:
        """
        Async version of get_key().

        Runs the synchronous get_key() in the default executor so it does not
        block the event loop.

        Args:
            provider: LLM provider name.

        Returns:
            Raw API key string.

        Raises:
            ApiKeyError: If no valid key is available.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_key(provider))

    async def async_rotate(self, provider: str) -> None:
        """Async version of rotate()."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.rotate(provider))

    async def async_add_key(self, provider: str, key: str) -> None:
        """Async version of add_key()."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.add_key(provider, key))

    async def async_reload_from_env(self, provider: Optional[str] = None) -> None:
        """Async version of reload_from_env()."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.reload_from_env(provider))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all_from_env(self) -> None:
        """Load keys for all supported providers from environment variables."""
        for provider in SUPPORTED_PROVIDERS:
            keys = _load_keys_from_env(provider)
            self._keys[provider] = keys
            if keys:
                logger.debug(
                    "Loaded %d API key(s) for provider '%s' from environment",
                    len(keys),
                    provider,
                )

    def _select_key(self, provider: str, keys: List[str]) -> str:
        """
        Select a key from the list according to the configured strategy.

        Must be called with ``self._lock`` held.
        """
        if self._strategy == KeySelectionStrategy.ROUND_ROBIN:
            idx = self._rr_index[provider] % len(keys)
            return keys[idx]
        else:
            # FALLBACK: always return the first key
            return keys[0]

    @staticmethod
    def _require_supported_provider(provider: str) -> None:
        """Raise ApiKeyError if the provider is not in SUPPORTED_PROVIDERS."""
        if provider not in SUPPORTED_PROVIDERS:
            raise ApiKeyError(
                provider=provider,
                reason=(
                    f"Unsupported provider '{provider}'. "
                    f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
                ),
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_manager: Optional[ApiKeyManager] = None
_manager_lock = threading.Lock()


def get_api_key_manager(
    strategy: KeySelectionStrategy = KeySelectionStrategy.ROUND_ROBIN,
) -> ApiKeyManager:
    """
    Return the module-level singleton ApiKeyManager.

    Creates the instance on first call. Subsequent calls return the same object.

    Args:
        strategy: Key selection strategy (only used on first call).

    Returns:
        Singleton ApiKeyManager instance.
    """
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = ApiKeyManager(strategy=strategy)
    return _default_manager


def reset_api_key_manager() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_manager
    with _manager_lock:
        _default_manager = None


__all__ = [
    "SUPPORTED_PROVIDERS",
    "KeySelectionStrategy",
    "ApiKeyError",
    "MaskedApiKey",
    "ApiKeyManager",
    "get_api_key_manager",
    "reset_api_key_manager",
]
