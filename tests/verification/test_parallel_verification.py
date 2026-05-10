"""
Tests for parallel verification with first-failure-fast logic — Task 5.4.2

Covers:
- All 3 checks start concurrently (parallel execution)
- Critical failure in any check causes early termination (first-failure-fast)
- verify_draft_parallel() public method works correctly
- Backward compatibility: sync verify_draft() still works
- Integration with early_termination utility

**Validates: Requirements 9.3** - price/policy/relevance checks run in parallel
**Validates: Requirements 9.1** - verification completes in ≤10 seconds
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import List, Tuple

from verification.agent.verification_agent import VerificationAgent
from verification.config.config import VerificationConfig
from verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> VerificationConfig:
    defaults = dict(
        price_tolerance_percent=1.0,
        price_critical_threshold=30.0,
        max_retries=3,
        parallel_verification=True,
        early_termination=True,
        async_timeout_seconds=10,
        enable_caching=False,
        llm_model_name="gpt-4",
    )
    defaults.update(overrides)
    return VerificationConfig(**defaults)


def _make_state(**overrides) -> dict:
    base = {
        "objection_text": "iPhone qua dat, tai sao toi nen mua?",
        "draft_response": "iPhone mang lai gia tri vuot troi voi he sinh thai Apple.",
        "tools_used": [],
        "research_reasoning": "",
        "research_sources": [],
        "verification_result": None,
        "correction_feedback": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_response": "",
        "workflow_status": "verifying",
        "execution_log": [],
        "start_time": "2024-01-15T10:30:00",
        "end_time": None,
        "resource_usage": {},
        "error_log": [],
        "config": {},
        "workflow_id": "wf_test_001",
        "correlation_id": "corr_test_001",
    }
    base.update(overrides)
    return base


def _make_price_issue(severity: IssueSeverity) -> PriceIssue:
    return PriceIssue(
        product_name="iPhone 15",
        severity=severity,
        explanation=f"Price issue with severity {severity.value}",
    )


def _make_policy_issue(severity: IssueSeverity) -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Bao hanh 2 nam",
        policy_type="warranty",
        is_fabricated=(severity == IssueSeverity.CRITICAL),
        is_inaccurate=False,
        severity=severity,
        explanation=f"Policy issue with severity {severity.value}",
    )


def _make_relevance_issue(severity: IssueSeverity) -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="price comparison",
        response_coverage=0.1 if severity == IssueSeverity.CRITICAL else 0.6,
        severity=severity,
        explanation=f"Relevance issue with severity {severity.value}",
    )


def _make_agent(config: VerificationConfig = None) -> VerificationAgent:
    """Create a VerificationAgent with mocked LLM and RAG pipeline."""
    if config is None:
        config = _make_config()
    llm = MagicMock()
    rag = MagicMock()
    return VerificationAgent(llm=llm, rag_pipeline=rag, config=config)


# ---------------------------------------------------------------------------
# Tests: verify_draft_parallel() public method exists and works
# ---------------------------------------------------------------------------

class TestVerifyDraftParallelExists:
    """Verify the public verify_draft_parallel() method is present and callable."""

    def test_method_exists(self):
        agent = _make_agent()
        assert hasattr(agent, "verify_draft_parallel")
        assert callable(agent.verify_draft_parallel)

    def test_method_is_coroutine(self):
        import inspect
        agent = _make_agent()
        assert inspect.iscoroutinefunction(agent.verify_draft_parallel)

    @pytest.mark.asyncio
    async def test_returns_verification_result(self):
        agent = _make_agent()
        state = _make_state()

        # Mock the internal parallel check to return all-pass
        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (True, [])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft_parallel(state)

        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_approved_when_all_checks_pass(self):
        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (True, [])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft_parallel(state)

        assert result.is_approved is True
        assert result.criteria.critical_issues_count == 0

    @pytest.mark.asyncio
    async def test_failed_when_any_check_fails(self):
        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (False, [_make_price_issue(IssueSeverity.MAJOR)])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft_parallel(state)

        assert result.is_approved is False


# ---------------------------------------------------------------------------
# Tests: Parallel execution (all 3 checks start concurrently)
# ---------------------------------------------------------------------------

class TestParallelExecution:
    """Verify that all 3 checks run concurrently, not sequentially."""

    @pytest.mark.asyncio
    async def test_all_three_checks_are_called(self):
        """All 3 async checker methods must be invoked."""
        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (True, [])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            await agent.verify_draft_parallel(state)

        mock_price.assert_called_once()
        mock_policy.assert_called_once()
        mock_relevance.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_execution_is_faster_than_sequential(self):
        """
        Parallel execution of 3 checks with 0.1s delay each should complete
        in ~0.1s (parallel), not ~0.3s (sequential).
        """
        agent = _make_agent()
        state = _make_state()

        async def slow_check(*args, **kwargs):
            await asyncio.sleep(0.05)
            return (True, [])

        with patch.object(agent, "_check_price_accuracy_async", side_effect=slow_check), \
             patch.object(agent, "_check_policy_authenticity_async", side_effect=slow_check), \
             patch.object(agent, "_check_topic_relevance_async", side_effect=slow_check):

            start = time.monotonic()
            await agent.verify_draft_parallel(state)
            elapsed = time.monotonic() - start

        # Sequential would take ~0.15s; parallel should be ~0.05s
        # Allow generous margin for CI environments
        assert elapsed < 0.14, (
            f"Parallel execution took {elapsed:.3f}s — expected < 0.14s (checks should run concurrently)"
        )

    @pytest.mark.asyncio
    async def test_checks_receive_correct_state(self):
        """Each checker receives the correct state argument."""
        agent = _make_agent()
        state = _make_state(
            draft_response="Custom draft",
            objection_text="Custom objection",
        )

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (True, [])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            await agent.verify_draft_parallel(state)

        # Each check should be called with the state
        mock_price.assert_called_once_with(state)
        mock_policy.assert_called_once_with(state)
        mock_relevance.assert_called_once_with(state)


# ---------------------------------------------------------------------------
# Tests: First-failure-fast logic (critical issue causes early termination)
# ---------------------------------------------------------------------------

class TestFirstFailureFast:
    """Verify that a CRITICAL issue in any check cancels remaining tasks."""

    @pytest.mark.asyncio
    async def test_critical_price_issue_triggers_early_termination(self):
        """CRITICAL price issue → immediate_termination=True in result."""
        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (False, [_make_price_issue(IssueSeverity.CRITICAL)])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft_parallel(state)

        assert result.is_approved is False
        assert result.has_critical_issues is True
        assert result.immediate_termination is True

    @pytest.mark.asyncio
    async def test_critical_policy_issue_triggers_early_termination(self):
        """CRITICAL policy issue → immediate_termination=True in result."""
        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (True, [])
            mock_policy.return_value = (False, [_make_policy_issue(IssueSeverity.CRITICAL)])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft_parallel(state)

        assert result.is_approved is False
        assert result.has_critical_issues is True
        assert result.immediate_termination is True

    @pytest.mark.asyncio
    async def test_critical_relevance_issue_triggers_early_termination(self):
        """CRITICAL relevance issue → immediate_termination=True in result."""
        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (True, [])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (False, [_make_relevance_issue(IssueSeverity.CRITICAL)])

            result = await agent.verify_draft_parallel(state)

        assert result.is_approved is False
        assert result.has_critical_issues is True
        assert result.immediate_termination is True

    @pytest.mark.asyncio
    async def test_major_issues_do_not_trigger_early_termination_flag(self):
        """MAJOR issues fail verification but do NOT set immediate_termination."""
        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (False, [_make_price_issue(IssueSeverity.MAJOR)])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft_parallel(state)

        assert result.is_approved is False
        assert result.has_critical_issues is False
        assert result.immediate_termination is False

    @pytest.mark.asyncio
    async def test_first_failure_fast_skips_remaining_checks_on_critical(self):
        """
        When a critical issue is found, remaining pending tasks are cancelled.
        The result should still be returned (not hang).
        """
        agent = _make_agent()
        state = _make_state()

        # Price check returns critical immediately; policy/relevance are slow
        async def fast_critical(*args, **kwargs):
            return (False, [_make_price_issue(IssueSeverity.CRITICAL)])

        async def slow_check(*args, **kwargs):
            await asyncio.sleep(5.0)  # Would timeout if not cancelled
            return (True, [])

        with patch.object(agent, "_check_price_accuracy_async", side_effect=fast_critical), \
             patch.object(agent, "_check_policy_authenticity_async", side_effect=slow_check), \
             patch.object(agent, "_check_topic_relevance_async", side_effect=slow_check):

            start = time.monotonic()
            result = await agent.verify_draft_parallel(state)
            elapsed = time.monotonic() - start

        # Should complete quickly (not wait 5s for slow checks)
        assert elapsed < 2.0, f"Expected early termination but took {elapsed:.2f}s"
        assert result.has_critical_issues is True
        assert result.immediate_termination is True

    @pytest.mark.asyncio
    async def test_no_early_termination_when_disabled(self):
        """When early_termination=False, all checks run to completion."""
        config = _make_config(early_termination=False)
        agent = _make_agent(config)
        state = _make_state()

        call_order = []

        async def price_check(*args, **kwargs):
            call_order.append("price")
            return (False, [_make_price_issue(IssueSeverity.CRITICAL)])

        async def policy_check(*args, **kwargs):
            call_order.append("policy")
            return (True, [])

        async def relevance_check(*args, **kwargs):
            call_order.append("relevance")
            return (True, [])

        with patch.object(agent, "_check_price_accuracy_async", side_effect=price_check), \
             patch.object(agent, "_check_policy_authenticity_async", side_effect=policy_check), \
             patch.object(agent, "_check_topic_relevance_async", side_effect=relevance_check):

            result = await agent.verify_draft_parallel(state)

        # All 3 checks should have run
        assert "price" in call_order
        assert "policy" in call_order
        assert "relevance" in call_order
        assert len(call_order) == 3


# ---------------------------------------------------------------------------
# Tests: Integration with early_termination utility
# ---------------------------------------------------------------------------

class TestEarlyTerminationIntegration:
    """Verify integration with CriticalIssueDetector from early_termination.py."""

    @pytest.mark.asyncio
    async def test_result_has_critical_issues_flag_set_by_detector(self):
        """
        The has_critical_issues and immediate_termination flags on VerificationResult
        are set by should_terminate_immediately() from early_termination.py.
        """
        from verification.utils.early_termination import should_terminate_immediately

        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (False, [_make_price_issue(IssueSeverity.CRITICAL)])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft_parallel(state)

        # Verify the result is consistent with should_terminate_immediately()
        decision = should_terminate_immediately(result)
        assert decision.should_terminate is True
        assert result.has_critical_issues == decision.should_terminate
        assert result.immediate_termination == decision.should_terminate

    @pytest.mark.asyncio
    async def test_no_critical_issues_flags_are_false(self):
        """When no critical issues, both flags remain False."""
        from verification.utils.early_termination import should_terminate_immediately

        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (True, [])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft_parallel(state)

        decision = should_terminate_immediately(result)
        assert decision.should_terminate is False
        assert result.has_critical_issues is False
        assert result.immediate_termination is False


# ---------------------------------------------------------------------------
# Tests: Backward compatibility — sync verify_draft() still works
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure the existing sync verify_draft() method still works."""

    def test_verify_draft_sync_method_exists(self):
        agent = _make_agent()
        assert hasattr(agent, "verify_draft_sync")
        assert callable(agent.verify_draft_sync)

    def test_verify_draft_async_method_exists(self):
        import inspect
        agent = _make_agent()
        assert hasattr(agent, "verify_draft")
        assert inspect.iscoroutinefunction(agent.verify_draft)

    @pytest.mark.asyncio
    async def test_verify_draft_still_works(self):
        """The original verify_draft() async method still returns VerificationResult."""
        agent = _make_agent()
        state = _make_state()

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = (True, [])
            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            result = await agent.verify_draft(state)

        assert isinstance(result, VerificationResult)
        assert result.is_approved is True

    @pytest.mark.asyncio
    async def test_verify_draft_parallel_and_verify_draft_produce_same_result(self):
        """Both methods should produce equivalent results for the same input."""
        agent = _make_agent()
        state = _make_state()

        price_result = (True, [])
        policy_result = (True, [])
        relevance_result = (True, [])

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = price_result
            mock_policy.return_value = policy_result
            mock_relevance.return_value = relevance_result

            result_parallel = await agent.verify_draft_parallel(state)

        with patch.object(agent, "_check_price_accuracy_async", new_callable=AsyncMock) as mock_price, \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_price.return_value = price_result
            mock_policy.return_value = policy_result
            mock_relevance.return_value = relevance_result

            result_original = await agent.verify_draft(state)

        assert result_parallel.is_approved == result_original.is_approved
        assert result_parallel.criteria.price_accuracy_pass == result_original.criteria.price_accuracy_pass
        assert result_parallel.criteria.policy_authenticity_pass == result_original.criteria.policy_authenticity_pass
        assert result_parallel.criteria.topic_relevance_pass == result_original.criteria.topic_relevance_pass


# ---------------------------------------------------------------------------
# Tests: Error handling in parallel verification
# ---------------------------------------------------------------------------

class TestParallelVerificationErrorHandling:
    """Verify error handling in verify_draft_parallel()."""

    @pytest.mark.asyncio
    async def test_exception_in_one_check_handled_gracefully(self):
        """If one check raises an exception, the result is still returned."""
        agent = _make_agent()
        state = _make_state()

        async def failing_check(*args, **kwargs):
            raise ConnectionError("DB unavailable")

        with patch.object(agent, "_check_price_accuracy_async", side_effect=failing_check), \
             patch.object(agent, "_check_policy_authenticity_async", new_callable=AsyncMock) as mock_policy, \
             patch.object(agent, "_check_topic_relevance_async", new_callable=AsyncMock) as mock_relevance:

            mock_policy.return_value = (True, [])
            mock_relevance.return_value = (True, [])

            # Should not raise — should return a VerificationResult
            result = await agent.verify_draft_parallel(state)

        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_invalid_state_raises_or_returns_error_result(self):
        """Empty draft_response should be handled gracefully."""
        agent = _make_agent()
        state = _make_state(draft_response="")

        # Should either raise ValueError or return a failed VerificationResult
        try:
            result = await agent.verify_draft_parallel(state)
            # If it returns, it should be a VerificationResult
            assert isinstance(result, VerificationResult)
        except ValueError:
            pass  # Also acceptable — input validation
