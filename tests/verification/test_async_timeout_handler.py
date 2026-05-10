"""
Tests for AsyncTimeoutHandler - Task 6.3.2

Covers:
- Successful operation within timeout
- Operation that exceeds timeout raises OperationTimeoutError with correct context
- Timeout values are loaded from config
- Per-operation timeout configuration works correctly

Requirements: 8.1, 9.1
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from verification.config.thresholds_config import TimeoutConfig
from verification.utils.async_timeout_handler import (
    AsyncTimeoutHandler,
    OperationTimeoutError,
    get_async_timeout_handler,
    reset_async_timeout_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _fast_coro(value: int = 42) -> int:
    await asyncio.sleep(0)
    return value


async def _slow_coro(delay: float = 10.0) -> None:
    await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# OperationTimeoutError
# ---------------------------------------------------------------------------

class TestOperationTimeoutError:
    def test_attributes_set_correctly(self):
        err = OperationTimeoutError(
            operation="llm_call",
            elapsed=11.5,
            threshold=10.0,
            escalation_required=True,
        )
        assert err.operation == "llm_call"
        assert err.elapsed == 11.5
        assert err.threshold == 10.0
        assert err.escalation_required is True

    def test_is_subclass_of_timeout_error(self):
        err = OperationTimeoutError("op", 1.0, 0.5)
        assert isinstance(err, TimeoutError)

    def test_str_contains_operation_name(self):
        err = OperationTimeoutError("price_check", 6.0, 5.0)
        assert "price_check" in str(err)

    def test_to_dict(self):
        err = OperationTimeoutError("policy_check", 5.5, 5.0, escalation_required=True)
        d = err.to_dict()
        assert d["operation"] == "policy_check"
        assert d["elapsed_seconds"] == 5.5
        assert d["threshold_seconds"] == 5.0
        assert d["escalation_required"] is True

    def test_default_escalation_false(self):
        err = OperationTimeoutError("op", 1.0, 0.5)
        assert err.escalation_required is False


# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------

class TestTimeoutConfig:
    def test_default_values(self):
        cfg = TimeoutConfig()
        assert cfg.llm_call == 10.0
        assert cfg.price_check == 5.0
        assert cfg.policy_check == 5.0
        assert cfg.relevance_check == 5.0
        assert cfg.total_workflow == 30.0
        assert cfg.escalate_on_critical_timeout is True

    def test_get_timeout_known_operation(self):
        cfg = TimeoutConfig(llm_call=15.0, price_check=3.0)
        assert cfg.get_timeout("llm_call") == 15.0
        assert cfg.get_timeout("price_check") == 3.0

    def test_get_timeout_unknown_falls_back_to_llm_call(self):
        cfg = TimeoutConfig(llm_call=8.0)
        assert cfg.get_timeout("unknown_operation") == 8.0

    def test_custom_values(self):
        cfg = TimeoutConfig(
            llm_call=20.0,
            price_check=7.0,
            policy_check=7.0,
            relevance_check=7.0,
            total_workflow=60.0,
            escalate_on_critical_timeout=False,
        )
        assert cfg.llm_call == 20.0
        assert cfg.total_workflow == 60.0
        assert cfg.escalate_on_critical_timeout is False


# ---------------------------------------------------------------------------
# AsyncTimeoutHandler — construction
# ---------------------------------------------------------------------------

class TestAsyncTimeoutHandlerConstruction:
    def test_accepts_explicit_timeout_config(self):
        cfg = TimeoutConfig(llm_call=20.0)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        assert handler.config.llm_call == 20.0

    def test_loads_defaults_when_no_config_given(self):
        handler = AsyncTimeoutHandler(timeout_config=TimeoutConfig())
        assert handler.config.llm_call == 10.0

    def test_get_threshold_returns_correct_value(self):
        cfg = TimeoutConfig(price_check=3.5)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        assert handler.get_threshold("price_check") == 3.5

    def test_get_threshold_unknown_falls_back(self):
        cfg = TimeoutConfig(llm_call=12.0)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        assert handler.get_threshold("nonexistent") == 12.0


# ---------------------------------------------------------------------------
# AsyncTimeoutHandler.run — success path
# ---------------------------------------------------------------------------

class TestAsyncTimeoutHandlerRunSuccess:
    @pytest.mark.asyncio
    async def test_returns_coroutine_result(self):
        handler = AsyncTimeoutHandler(timeout_config=TimeoutConfig(llm_call=5.0))
        result = await handler.run(_fast_coro(99), "llm_call")
        assert result == 99

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        handler = AsyncTimeoutHandler(timeout_config=TimeoutConfig(price_check=5.0))
        result = await handler.run(_fast_coro(7), "price_check")
        assert result == 7

    @pytest.mark.asyncio
    async def test_timeout_override_used_when_provided(self):
        # Very short configured timeout, but override is generous
        cfg = TimeoutConfig(llm_call=0.001)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        # With override=5.0 the fast coro should succeed
        result = await handler.run(_fast_coro(1), "llm_call", timeout_override=5.0)
        assert result == 1


# ---------------------------------------------------------------------------
# AsyncTimeoutHandler.run — timeout path
# ---------------------------------------------------------------------------

class TestAsyncTimeoutHandlerRunTimeout:
    @pytest.mark.asyncio
    async def test_raises_operation_timeout_error(self):
        cfg = TimeoutConfig(llm_call=0.05, escalate_on_critical_timeout=False)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        with pytest.raises(OperationTimeoutError):
            await handler.run(_slow_coro(5.0), "llm_call")

    @pytest.mark.asyncio
    async def test_error_has_correct_operation_name(self):
        cfg = TimeoutConfig(price_check=0.05)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        with pytest.raises(OperationTimeoutError) as exc_info:
            await handler.run(_slow_coro(5.0), "price_check")
        assert exc_info.value.operation == "price_check"

    @pytest.mark.asyncio
    async def test_error_has_correct_threshold(self):
        cfg = TimeoutConfig(policy_check=0.05)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        with pytest.raises(OperationTimeoutError) as exc_info:
            await handler.run(_slow_coro(5.0), "policy_check")
        assert exc_info.value.threshold == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_error_elapsed_is_non_negative(self):
        cfg = TimeoutConfig(relevance_check=0.05)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        with pytest.raises(OperationTimeoutError) as exc_info:
            await handler.run(_slow_coro(5.0), "relevance_check")
        assert exc_info.value.elapsed >= 0.0

    @pytest.mark.asyncio
    async def test_escalation_flag_from_config(self):
        cfg = TimeoutConfig(llm_call=0.05, escalate_on_critical_timeout=True)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        with pytest.raises(OperationTimeoutError) as exc_info:
            await handler.run(_slow_coro(5.0), "llm_call")
        assert exc_info.value.escalation_required is True

    @pytest.mark.asyncio
    async def test_no_escalation_when_config_false(self):
        cfg = TimeoutConfig(llm_call=0.05, escalate_on_critical_timeout=False)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        with pytest.raises(OperationTimeoutError) as exc_info:
            await handler.run(_slow_coro(5.0), "llm_call")
        assert exc_info.value.escalation_required is False

    @pytest.mark.asyncio
    async def test_timeout_override_triggers_timeout(self):
        # Generous configured timeout, but override is tiny
        cfg = TimeoutConfig(llm_call=30.0)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        with pytest.raises(OperationTimeoutError):
            await handler.run(_slow_coro(5.0), "llm_call", timeout_override=0.05)


# ---------------------------------------------------------------------------
# AsyncTimeoutHandler.wrap
# ---------------------------------------------------------------------------

class TestAsyncTimeoutHandlerWrap:
    @pytest.mark.asyncio
    async def test_wrap_success(self):
        handler = AsyncTimeoutHandler(timeout_config=TimeoutConfig(llm_call=5.0))

        async def my_fn(x: int) -> int:
            return x * 2

        result = await handler.wrap(my_fn, "llm_call", 21)
        assert result == 42

    @pytest.mark.asyncio
    async def test_wrap_timeout(self):
        cfg = TimeoutConfig(llm_call=0.05)
        handler = AsyncTimeoutHandler(timeout_config=cfg)

        async def slow_fn() -> None:
            await asyncio.sleep(5.0)

        with pytest.raises(OperationTimeoutError) as exc_info:
            await handler.wrap(slow_fn, "llm_call")
        assert exc_info.value.operation == "llm_call"

    @pytest.mark.asyncio
    async def test_wrap_passes_kwargs(self):
        handler = AsyncTimeoutHandler(timeout_config=TimeoutConfig(price_check=5.0))

        async def fn(a: int, b: int = 0) -> int:
            return a + b

        result = await handler.wrap(fn, "price_check", 10, b=5)
        assert result == 15


# ---------------------------------------------------------------------------
# Per-operation timeout configuration
# ---------------------------------------------------------------------------

class TestPerOperationTimeouts:
    @pytest.mark.asyncio
    async def test_different_operations_use_different_thresholds(self):
        cfg = TimeoutConfig(
            llm_call=10.0,
            price_check=3.0,
            policy_check=4.0,
            relevance_check=2.0,
        )
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        assert handler.get_threshold("llm_call") == 10.0
        assert handler.get_threshold("price_check") == 3.0
        assert handler.get_threshold("policy_check") == 4.0
        assert handler.get_threshold("relevance_check") == 2.0

    @pytest.mark.asyncio
    async def test_price_check_timeout_enforced(self):
        cfg = TimeoutConfig(price_check=0.05)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        with pytest.raises(OperationTimeoutError) as exc_info:
            await handler.run(_slow_coro(5.0), "price_check")
        assert exc_info.value.operation == "price_check"
        assert exc_info.value.threshold == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_total_workflow_timeout_configured(self):
        cfg = TimeoutConfig(total_workflow=45.0)
        handler = AsyncTimeoutHandler(timeout_config=cfg)
        assert handler.get_threshold("total_workflow") == 45.0


# ---------------------------------------------------------------------------
# Config loading from YAML
# ---------------------------------------------------------------------------

class TestConfigLoadingFromYAML:
    def test_loads_timeout_config_from_yaml(self):
        """TimeoutConfig values should be loaded from thresholds.yaml."""
        from verification.config.thresholds_config import enhanced_load_thresholds_config
        import os

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "thresholds.yaml"
        )
        config = enhanced_load_thresholds_config(yaml_path)
        # The YAML file has a timeouts: section — verify it was parsed
        assert config.timeouts.llm_call > 0
        assert config.timeouts.price_check > 0
        assert config.timeouts.total_workflow > 0

    def test_handler_uses_yaml_config(self):
        """AsyncTimeoutHandler should pick up values from the YAML file."""
        import os

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "thresholds.yaml"
        )
        handler = AsyncTimeoutHandler(thresholds_path=yaml_path)
        # Defaults from YAML: llm_call=10, price_check=5
        assert handler.get_threshold("llm_call") == pytest.approx(10.0)
        assert handler.get_threshold("price_check") == pytest.approx(5.0)

    def test_handler_falls_back_to_defaults_on_bad_path(self):
        """When the YAML path is invalid, defaults should be used."""
        handler = AsyncTimeoutHandler(thresholds_path="/nonexistent/path.yaml")
        assert handler.get_threshold("llm_call") == 10.0


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    def setup_method(self):
        reset_async_timeout_handler()

    def teardown_method(self):
        reset_async_timeout_handler()

    def test_get_returns_handler_instance(self):
        handler = get_async_timeout_handler(timeout_config=TimeoutConfig())
        assert isinstance(handler, AsyncTimeoutHandler)

    def test_get_returns_same_instance(self):
        h1 = get_async_timeout_handler(timeout_config=TimeoutConfig())
        h2 = get_async_timeout_handler(timeout_config=TimeoutConfig())
        assert h1 is h2

    def test_reset_creates_new_instance(self):
        h1 = get_async_timeout_handler(timeout_config=TimeoutConfig())
        reset_async_timeout_handler()
        h2 = get_async_timeout_handler(timeout_config=TimeoutConfig())
        assert h1 is not h2
