"""
Tests for ApiKeyManager - Task 7.2.2

Covers:
- ApiKeyError exception structure
- MaskedApiKey Pydantic model
- Key loading from environment variables
- Key format validation (OpenAI and Anthropic)
- Key masking — full key never exposed
- Key selection strategies (round-robin, fallback)
- Key rotation via rotate()
- Adding keys at runtime via add_key()
- Removing keys via remove_key()
- Reloading from environment via reload_from_env()
- Sync get_key() and async async_get_key()
- get_masked_key() safe display
- Singleton factory get_api_key_manager()
- Unsupported provider handling
"""

import asyncio
import os
import pytest

from backend.verification.utils.api_key_manager import (
    ApiKeyError,
    ApiKeyManager,
    KeySelectionStrategy,
    MaskedApiKey,
    SUPPORTED_PROVIDERS,
    get_api_key_manager,
    reset_api_key_manager,
    _mask_key,
    _validate_key_format,
)


# ---------------------------------------------------------------------------
# Sample keys (fake — safe for tests, never real credentials)
# ---------------------------------------------------------------------------

FAKE_OPENAI_KEY = "sk-testFakeOpenAIKey1234567890abcdefghij"
FAKE_OPENAI_KEY_2 = "sk-testFakeOpenAIKey2ndKey0987654321xyz"
FAKE_ANTHROPIC_KEY = "sk-ant-testFakeAnthropicKey1234567890abcdef"
FAKE_ANTHROPIC_KEY_2 = "sk-ant-testFakeAnthropicKey2ndKey0987654321"

INVALID_KEY = "not-a-valid-key"
SHORT_KEY = "sk-short"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    reset_api_key_manager()
    yield
    reset_api_key_manager()


@pytest.fixture
def manager_no_env():
    """ApiKeyManager with auto_load_env=False (clean slate)."""
    return ApiKeyManager(auto_load_env=False)


@pytest.fixture
def manager_with_openai(monkeypatch):
    """ApiKeyManager with one OpenAI key loaded from env."""
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.delenv("OPENAI_API_KEY_2", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_3", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return ApiKeyManager()


@pytest.fixture
def manager_with_two_openai(monkeypatch):
    """ApiKeyManager with two OpenAI keys loaded from env."""
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setenv("OPENAI_API_KEY_2", FAKE_OPENAI_KEY_2)
    monkeypatch.delenv("OPENAI_API_KEY_3", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return ApiKeyManager()


# ---------------------------------------------------------------------------
# _mask_key helper
# ---------------------------------------------------------------------------

class TestMaskKey:

    def test_long_key_shows_prefix_and_placeholder(self):
        key = "sk-abcdefghijklmnopqrstuvwxyz"
        masked = _mask_key(key)
        assert "****" in masked
        assert key not in masked

    def test_short_key_returns_placeholder(self):
        masked = _mask_key("sk-x")
        assert masked == "****"

    def test_masked_starts_with_key_prefix(self):
        key = "sk-abcdefghijklmnopqrstuvwxyz"
        masked = _mask_key(key)
        assert masked.startswith("sk-")

    def test_masked_contains_ellipsis(self):
        key = "sk-abcdefghijklmnopqrstuvwxyz"
        masked = _mask_key(key)
        assert "..." in masked

    def test_full_key_not_in_masked(self):
        key = FAKE_OPENAI_KEY
        masked = _mask_key(key)
        assert key not in masked


# ---------------------------------------------------------------------------
# _validate_key_format helper
# ---------------------------------------------------------------------------

class TestValidateKeyFormat:

    def test_valid_openai_key(self):
        assert _validate_key_format(FAKE_OPENAI_KEY, "openai") is True

    def test_valid_anthropic_key(self):
        assert _validate_key_format(FAKE_ANTHROPIC_KEY, "anthropic") is True

    def test_invalid_key_fails_openai(self):
        assert _validate_key_format(INVALID_KEY, "openai") is False

    def test_invalid_key_fails_anthropic(self):
        assert _validate_key_format(INVALID_KEY, "anthropic") is False

    def test_openai_key_fails_anthropic_pattern(self):
        # OpenAI key doesn't match anthropic pattern
        assert _validate_key_format(FAKE_OPENAI_KEY, "anthropic") is False

    def test_anthropic_key_fails_openai_pattern(self):
        # Anthropic key doesn't match openai pattern
        assert _validate_key_format(FAKE_ANTHROPIC_KEY, "openai") is False

    def test_unknown_provider_returns_false(self):
        assert _validate_key_format(FAKE_OPENAI_KEY, "unknown_provider") is False

    def test_empty_string_fails(self):
        assert _validate_key_format("", "openai") is False

    def test_short_key_fails(self):
        assert _validate_key_format(SHORT_KEY, "openai") is False


# ---------------------------------------------------------------------------
# ApiKeyError
# ---------------------------------------------------------------------------

class TestApiKeyError:

    def test_is_exception(self):
        err = ApiKeyError(provider="openai", reason="test reason")
        assert isinstance(err, Exception)

    def test_attributes(self):
        err = ApiKeyError(provider="anthropic", reason="no key found")
        assert err.provider == "anthropic"
        assert err.reason == "no key found"

    def test_str_contains_provider(self):
        err = ApiKeyError(provider="openai", reason="missing")
        assert "openai" in str(err)

    def test_str_contains_reason(self):
        err = ApiKeyError(provider="openai", reason="missing key")
        assert "missing key" in str(err)


# ---------------------------------------------------------------------------
# MaskedApiKey
# ---------------------------------------------------------------------------

class TestMaskedApiKey:

    def test_is_pydantic_model(self):
        m = MaskedApiKey(masked="sk-abc...****", provider="openai", is_valid=True)
        assert isinstance(m, MaskedApiKey)

    def test_fields(self):
        m = MaskedApiKey(masked="sk-abc...****", provider="anthropic", is_valid=False)
        assert m.masked == "sk-abc...****"
        assert m.provider == "anthropic"
        assert m.is_valid is False

    def test_json_serializable(self):
        m = MaskedApiKey(masked="sk-abc...****", provider="openai", is_valid=True)
        data = m.model_dump()
        assert "masked" in data
        assert "provider" in data
        assert "is_valid" in data


# ---------------------------------------------------------------------------
# ApiKeyManager — no keys
# ---------------------------------------------------------------------------

class TestApiKeyManagerNoKeys:

    def test_get_key_raises_when_no_key(self, manager_no_env):
        with pytest.raises(ApiKeyError) as exc_info:
            manager_no_env.get_key("openai")
        assert exc_info.value.provider == "openai"

    def test_get_key_raises_for_anthropic_when_no_key(self, manager_no_env):
        with pytest.raises(ApiKeyError) as exc_info:
            manager_no_env.get_key("anthropic")
        assert exc_info.value.provider == "anthropic"

    def test_get_masked_key_raises_when_no_key(self, manager_no_env):
        with pytest.raises(ApiKeyError):
            manager_no_env.get_masked_key("openai")

    def test_rotate_raises_when_no_key(self, manager_no_env):
        with pytest.raises(ApiKeyError):
            manager_no_env.rotate("openai")

    def test_key_count_zero(self, manager_no_env):
        assert manager_no_env.key_count("openai") == 0

    def test_has_valid_key_false(self, manager_no_env):
        assert manager_no_env.has_valid_key("openai") is False


# ---------------------------------------------------------------------------
# ApiKeyManager — loading from environment
# ---------------------------------------------------------------------------

class TestApiKeyManagerEnvLoading:

    def test_loads_openai_key_from_env(self, manager_with_openai):
        key = manager_with_openai.get_key("openai")
        assert key == FAKE_OPENAI_KEY

    def test_key_count_one_after_env_load(self, manager_with_openai):
        assert manager_with_openai.key_count("openai") == 1

    def test_has_valid_key_true_after_env_load(self, manager_with_openai):
        assert manager_with_openai.has_valid_key("openai") is True

    def test_loads_two_openai_keys(self, manager_with_two_openai):
        assert manager_with_two_openai.key_count("openai") == 2

    def test_no_anthropic_key_when_not_set(self, manager_with_openai):
        assert manager_with_openai.key_count("anthropic") == 0

    def test_env_not_loaded_when_auto_load_false(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
        manager = ApiKeyManager(auto_load_env=False)
        assert manager.key_count("openai") == 0

    def test_reload_from_env_picks_up_new_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        manager = ApiKeyManager(auto_load_env=False)
        assert manager.key_count("openai") == 0

        monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
        manager.reload_from_env("openai")
        assert manager.key_count("openai") == 1

    def test_reload_all_providers(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
        manager = ApiKeyManager(auto_load_env=False)
        manager.reload_from_env()
        assert manager.key_count("openai") == 1
        assert manager.key_count("anthropic") == 1


# ---------------------------------------------------------------------------
# ApiKeyManager — key masking (security)
# ---------------------------------------------------------------------------

class TestApiKeyMasking:

    def test_get_masked_key_does_not_expose_full_key(self, manager_with_openai):
        info = manager_with_openai.get_masked_key("openai")
        assert FAKE_OPENAI_KEY not in info.masked

    def test_get_masked_key_contains_placeholder(self, manager_with_openai):
        info = manager_with_openai.get_masked_key("openai")
        assert "****" in info.masked

    def test_get_masked_key_provider_correct(self, manager_with_openai):
        info = manager_with_openai.get_masked_key("openai")
        assert info.provider == "openai"

    def test_get_masked_key_is_valid_true_for_valid_key(self, manager_with_openai):
        info = manager_with_openai.get_masked_key("openai")
        assert info.is_valid is True

    def test_get_masked_key_is_valid_false_for_invalid_key(self, manager_no_env):
        manager_no_env._keys["openai"] = [INVALID_KEY]
        info = manager_no_env.get_masked_key("openai")
        assert info.is_valid is False

    def test_get_masked_key_returns_masked_api_key_instance(self, manager_with_openai):
        info = manager_with_openai.get_masked_key("openai")
        assert isinstance(info, MaskedApiKey)


# ---------------------------------------------------------------------------
# ApiKeyManager — format validation
# ---------------------------------------------------------------------------

class TestApiKeyManagerValidation:

    def test_get_key_raises_for_invalid_format(self, manager_no_env):
        manager_no_env._keys["openai"] = [INVALID_KEY]
        with pytest.raises(ApiKeyError) as exc_info:
            manager_no_env.get_key("openai")
        assert exc_info.value.provider == "openai"

    def test_add_key_raises_for_invalid_format(self, manager_no_env):
        with pytest.raises(ApiKeyError):
            manager_no_env.add_key("openai", INVALID_KEY)

    def test_add_key_succeeds_for_valid_format(self, manager_no_env):
        manager_no_env.add_key("openai", FAKE_OPENAI_KEY)
        assert manager_no_env.key_count("openai") == 1

    def test_add_key_does_not_duplicate(self, manager_no_env):
        manager_no_env.add_key("openai", FAKE_OPENAI_KEY)
        manager_no_env.add_key("openai", FAKE_OPENAI_KEY)
        assert manager_no_env.key_count("openai") == 1


# ---------------------------------------------------------------------------
# ApiKeyManager — rotation
# ---------------------------------------------------------------------------

class TestApiKeyManagerRotation:

    def test_rotate_advances_to_next_key(self, manager_with_two_openai):
        first_key = manager_with_two_openai.get_key("openai")
        manager_with_two_openai.rotate("openai")
        second_key = manager_with_two_openai.get_key("openai")
        assert first_key != second_key

    def test_rotate_wraps_around(self, manager_with_two_openai):
        manager_with_two_openai.rotate("openai")  # index 1
        manager_with_two_openai.rotate("openai")  # index 0 (wrap)
        key = manager_with_two_openai.get_key("openai")
        assert key == FAKE_OPENAI_KEY  # back to first

    def test_rotate_single_key_stays_same(self, manager_with_openai):
        key_before = manager_with_openai.get_key("openai")
        manager_with_openai.rotate("openai")
        key_after = manager_with_openai.get_key("openai")
        assert key_before == key_after

    def test_rotate_raises_for_unsupported_provider(self, manager_no_env):
        with pytest.raises(ApiKeyError):
            manager_no_env.rotate("unsupported_provider")


# ---------------------------------------------------------------------------
# ApiKeyManager — add / remove keys
# ---------------------------------------------------------------------------

class TestApiKeyManagerAddRemove:

    def test_add_key_increases_count(self, manager_no_env):
        manager_no_env.add_key("openai", FAKE_OPENAI_KEY)
        assert manager_no_env.key_count("openai") == 1

    def test_remove_key_decreases_count(self, manager_no_env):
        manager_no_env.add_key("openai", FAKE_OPENAI_KEY)
        removed = manager_no_env.remove_key("openai", FAKE_OPENAI_KEY)
        assert removed is True
        assert manager_no_env.key_count("openai") == 0

    def test_remove_nonexistent_key_returns_false(self, manager_no_env):
        result = manager_no_env.remove_key("openai", FAKE_OPENAI_KEY)
        assert result is False

    def test_add_anthropic_key(self, manager_no_env):
        manager_no_env.add_key("anthropic", FAKE_ANTHROPIC_KEY)
        key = manager_no_env.get_key("anthropic")
        assert key == FAKE_ANTHROPIC_KEY


# ---------------------------------------------------------------------------
# ApiKeyManager — round-robin strategy
# ---------------------------------------------------------------------------

class TestRoundRobinStrategy:

    def test_round_robin_cycles_through_keys(self, manager_no_env):
        manager_no_env.add_key("openai", FAKE_OPENAI_KEY)
        manager_no_env.add_key("openai", FAKE_OPENAI_KEY_2)

        key1 = manager_no_env.get_key("openai")
        manager_no_env.rotate("openai")
        key2 = manager_no_env.get_key("openai")
        manager_no_env.rotate("openai")
        key3 = manager_no_env.get_key("openai")

        assert key1 == FAKE_OPENAI_KEY
        assert key2 == FAKE_OPENAI_KEY_2
        assert key3 == FAKE_OPENAI_KEY  # wrapped around


# ---------------------------------------------------------------------------
# ApiKeyManager — fallback strategy
# ---------------------------------------------------------------------------

class TestFallbackStrategy:

    def test_fallback_always_returns_first_key(self, manager_no_env):
        manager_fallback = ApiKeyManager(
            strategy=KeySelectionStrategy.FALLBACK,
            auto_load_env=False,
        )
        manager_fallback.add_key("openai", FAKE_OPENAI_KEY)
        manager_fallback.add_key("openai", FAKE_OPENAI_KEY_2)

        # Always returns first key regardless of rotate
        key1 = manager_fallback.get_key("openai")
        manager_fallback.rotate("openai")
        key2 = manager_fallback.get_key("openai")

        assert key1 == FAKE_OPENAI_KEY
        # Fallback ignores rr_index — always returns index 0
        assert key2 == FAKE_OPENAI_KEY


# ---------------------------------------------------------------------------
# ApiKeyManager — unsupported provider
# ---------------------------------------------------------------------------

class TestUnsupportedProvider:

    def test_get_key_raises_for_unsupported_provider(self, manager_no_env):
        with pytest.raises(ApiKeyError) as exc_info:
            manager_no_env.get_key("cohere")
        assert "cohere" in exc_info.value.reason

    def test_add_key_raises_for_unsupported_provider(self, manager_no_env):
        with pytest.raises(ApiKeyError):
            manager_no_env.add_key("cohere", FAKE_OPENAI_KEY)

    def test_key_count_raises_for_unsupported_provider(self, manager_no_env):
        with pytest.raises(ApiKeyError):
            manager_no_env.key_count("cohere")

    def test_has_valid_key_raises_for_unsupported_provider(self, manager_no_env):
        with pytest.raises(ApiKeyError):
            manager_no_env.has_valid_key("cohere")


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------

class TestAsyncApi:

    def test_async_get_key_returns_key(self, manager_with_openai):
        key = asyncio.get_event_loop().run_until_complete(
            manager_with_openai.async_get_key("openai")
        )
        assert key == FAKE_OPENAI_KEY

    def test_async_get_key_raises_when_no_key(self, manager_no_env):
        with pytest.raises(ApiKeyError):
            asyncio.get_event_loop().run_until_complete(
                manager_no_env.async_get_key("openai")
            )

    def test_async_rotate(self, manager_with_two_openai):
        asyncio.get_event_loop().run_until_complete(
            manager_with_two_openai.async_rotate("openai")
        )
        key = manager_with_two_openai.get_key("openai")
        assert key == FAKE_OPENAI_KEY_2

    def test_async_add_key(self, manager_no_env):
        asyncio.get_event_loop().run_until_complete(
            manager_no_env.async_add_key("openai", FAKE_OPENAI_KEY)
        )
        assert manager_no_env.key_count("openai") == 1

    def test_async_reload_from_env(self, monkeypatch, manager_no_env):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
        asyncio.get_event_loop().run_until_complete(
            manager_no_env.async_reload_from_env("openai")
        )
        assert manager_no_env.key_count("openai") == 1


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

class TestSingleton:

    def test_get_api_key_manager_returns_same_instance(self):
        m1 = get_api_key_manager()
        m2 = get_api_key_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_api_key_manager()
        reset_api_key_manager()
        m2 = get_api_key_manager()
        assert m1 is not m2

    def test_singleton_is_api_key_manager(self):
        assert isinstance(get_api_key_manager(), ApiKeyManager)


# ---------------------------------------------------------------------------
# Security: full key never exposed in masked output
# ---------------------------------------------------------------------------

class TestKeyNeverExposed:

    def test_masked_key_does_not_contain_full_openai_key(self, manager_with_openai):
        info = manager_with_openai.get_masked_key("openai")
        assert FAKE_OPENAI_KEY not in info.masked
        assert FAKE_OPENAI_KEY not in info.provider

    def test_masked_key_str_representation_safe(self, manager_with_openai):
        info = manager_with_openai.get_masked_key("openai")
        serialized = info.model_dump_json()
        assert FAKE_OPENAI_KEY not in serialized

    def test_api_key_error_does_not_expose_key(self, manager_no_env):
        manager_no_env._keys["openai"] = [INVALID_KEY]
        try:
            manager_no_env.get_key("openai")
        except ApiKeyError as e:
            assert INVALID_KEY not in str(e)


# ---------------------------------------------------------------------------
# SUPPORTED_PROVIDERS constant
# ---------------------------------------------------------------------------

class TestSupportedProviders:

    def test_openai_in_supported_providers(self):
        assert "openai" in SUPPORTED_PROVIDERS

    def test_anthropic_in_supported_providers(self):
        assert "anthropic" in SUPPORTED_PROVIDERS
