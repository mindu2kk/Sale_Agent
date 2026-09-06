"""
Tests for GracefulDegradationHandler - Task 6.3.1

Tests:
- Single checker failure → verification continues with remaining results
- Two checker failures → degradation rejected, escalation result returned
- aggregate_partial_results produces correct VerificationResult with warning issues
- Structured logging during degradation events

Requirements: 8.1, 8.2, 8.3
"""

from __future__ import annotations

import asyncio
import logging
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.verification.models.verification import (
    IssueSeverity,
    PolicyIssue,
    PriceIssue,
    RelevanceIssue,
    RubricCriteria,
    VerificationResult,
)
from backend.verification.utils.graceful_degradation import (
    GracefulDegradationHandler,
    MAX_DEGRADABLE_FAILURES,
    PartialVerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_issue(severity: IssueSeverity = IssueSeverity.MAJOR) -> PriceIssue:
    return PriceIssue(
        product_name="iPhone 15",
        mentioned_price="35,000,000 VND",
        actual_price="29,990,000 VND",
        deviation_percent=16.7,
        severity=severity,
        explanation="Price deviation detected",
    )


def _make_policy_issue(severity: IssueSeverity = IssueSeverity.MINOR) -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Bảo hành 2 năm",
        policy_type="warranty",
        is_fabricated=False,
        is_inaccurate=True,
        severity=severity,
        explanation="Policy inaccurate",
    )


def _make_relevance_issue(severity: IssueSeverity = IssueSeverity.MINOR) -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="price comparison",
        response_coverage=0.5,
        missing_aspects=["feature comparison"],
        severity=severity,
        explanation="Coverage insufficient",
    )


# ---------------------------------------------------------------------------
# PartialVerificationResult
# ---------------------------------------------------------------------------

class TestPartialVerificationResult:
    def test_success_result_returns_checker_values(self):
        pr = PartialVerificationResult(
            checker_name="price",
            success=True,
            result=(False, [_make_price_issue()]),
        )
        assert pr.pass_flag is False
        assert len(pr.issues) == 1

    def test_failed_result_defaults_to_pass_with_empty_issues(self):
        pr = PartialVerificationResult(
            checker_name="price",
            success=False,
            error=RuntimeError("timeout"),
        )
        assert pr.pass_flag is True
        assert pr.issues == []


# ---------------------------------------------------------------------------
# run_checker_safely
# ---------------------------------------------------------------------------

class TestRunCheckerSafely:
    @pytest.mark.asyncio
    async def test_successful_checker_returns_success_result(self):
        handler = GracefulDegradationHandler(correlation_id="test-001")

        def checker_fn(draft, objection):
            return (True, [])

        pr = await handler.run_checker_safely("price", checker_fn, "draft", "objection")

        assert pr.success is True
        assert pr.result == (True, [])
        assert pr.error is None

    @pytest.mark.asyncio
    async def test_failing_checker_returns_failure_result(self):
        handler = GracefulDegradationHandler(correlation_id="test-002")

        def checker_fn(draft):
            raise ConnectionError("DB unavailable")

        pr = await handler.run_checker_safely("policy", checker_fn, "draft")

        assert pr.success is False
        assert isinstance(pr.error, ConnectionError)
        assert pr.result is None

    @pytest.mark.asyncio
    async def test_failing_checker_logs_warning(self, caplog):
        handler = GracefulDegradationHandler(correlation_id="test-003")

        def checker_fn():
            raise TimeoutError("LLM timeout")

        with caplog.at_level(logging.WARNING, logger="backend.verification.utils.graceful_degradation"):
            await handler.run_checker_safely("relevance", checker_fn)

        assert any("graceful degradation" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# should_degrade
# ---------------------------------------------------------------------------

class TestShouldDegrade:
    def test_zero_failures_is_acceptable(self):
        handler = GracefulDegradationHandler()
        assert handler.should_degrade([]) is True

    def test_one_failure_is_acceptable(self):
        handler = GracefulDegradationHandler()
        assert handler.should_degrade(["price"]) is True

    def test_two_failures_is_not_acceptable(self):
        handler = GracefulDegradationHandler()
        assert handler.should_degrade(["price", "policy"]) is False

    def test_three_failures_is_not_acceptable(self):
        handler = GracefulDegradationHandler()
        assert handler.should_degrade(["price", "policy", "relevance"]) is False

    def test_max_degradable_failures_constant_is_one(self):
        assert MAX_DEGRADABLE_FAILURES == 1


# ---------------------------------------------------------------------------
# aggregate_partial_results
# ---------------------------------------------------------------------------

class TestAggregatePartialResults:
    def test_all_successful_checkers_produce_correct_result(self):
        handler = GracefulDegradationHandler()
        partial_results = {
            "price": PartialVerificationResult(
                checker_name="price", success=True, result=(True, [])
            ),
            "policy": PartialVerificationResult(
                checker_name="policy", success=True, result=(True, [])
            ),
            "relevance": PartialVerificationResult(
                checker_name="relevance", success=True, result=(True, [])
            ),
        }

        result = handler.aggregate_partial_results(partial_results)

        assert isinstance(result, VerificationResult)
        assert result.criteria.price_accuracy_pass is True
        assert result.criteria.policy_authenticity_pass is True
        assert result.criteria.topic_relevance_pass is True
        assert result.criteria.overall_pass is True

    def test_price_checker_failure_injects_warning_issue(self):
        handler = GracefulDegradationHandler()
        partial_results = {
            "price": PartialVerificationResult(
                checker_name="price",
                success=False,
                error=RuntimeError("price service down"),
            ),
            "policy": PartialVerificationResult(
                checker_name="policy", success=True, result=(True, [])
            ),
            "relevance": PartialVerificationResult(
                checker_name="relevance", success=True, result=(True, [])
            ),
        }

        result = handler.aggregate_partial_results(partial_results)

        # Price defaults to PASS (not a hard failure)
        assert result.criteria.price_accuracy_pass is True
        # But a warning issue is injected
        assert len(result.criteria.price_issues) == 1
        warning = result.criteria.price_issues[0]
        assert warning.severity == IssueSeverity.MINOR
        assert "Checker Unavailable" in warning.product_name

    def test_policy_checker_failure_injects_warning_issue(self):
        handler = GracefulDegradationHandler()
        partial_results = {
            "price": PartialVerificationResult(
                checker_name="price", success=True, result=(True, [])
            ),
            "policy": PartialVerificationResult(
                checker_name="policy",
                success=False,
                error=TimeoutError("policy service timeout"),
            ),
            "relevance": PartialVerificationResult(
                checker_name="relevance", success=True, result=(True, [])
            ),
        }

        result = handler.aggregate_partial_results(partial_results)

        assert result.criteria.policy_authenticity_pass is True
        assert len(result.criteria.policy_issues) == 1
        warning = result.criteria.policy_issues[0]
        assert warning.severity == IssueSeverity.MINOR

    def test_relevance_checker_failure_injects_warning_issue(self):
        handler = GracefulDegradationHandler()
        partial_results = {
            "price": PartialVerificationResult(
                checker_name="price", success=True, result=(True, [])
            ),
            "policy": PartialVerificationResult(
                checker_name="policy", success=True, result=(True, [])
            ),
            "relevance": PartialVerificationResult(
                checker_name="relevance",
                success=False,
                error=ConnectionError("relevance service down"),
            ),
        }

        result = handler.aggregate_partial_results(partial_results)

        assert result.criteria.topic_relevance_pass is True
        assert len(result.criteria.relevance_issues) == 1
        warning = result.criteria.relevance_issues[0]
        assert warning.severity == IssueSeverity.MINOR

    def test_failed_checker_does_not_override_real_failures(self):
        """A successful checker that returns FAIL should still fail."""
        handler = GracefulDegradationHandler()
        partial_results = {
            "price": PartialVerificationResult(
                checker_name="price",
                success=True,
                result=(False, [_make_price_issue()]),
            ),
            "policy": PartialVerificationResult(
                checker_name="policy", success=True, result=(True, [])
            ),
            "relevance": PartialVerificationResult(
                checker_name="relevance", success=True, result=(True, [])
            ),
        }

        result = handler.aggregate_partial_results(partial_results)

        assert result.criteria.price_accuracy_pass is False
        assert result.criteria.overall_pass is False

    def test_custom_reasoning_is_preserved(self):
        handler = GracefulDegradationHandler()
        partial_results = {
            "price": PartialVerificationResult(
                checker_name="price", success=True, result=(True, [])
            ),
            "policy": PartialVerificationResult(
                checker_name="policy", success=True, result=(True, [])
            ),
            "relevance": PartialVerificationResult(
                checker_name="relevance", success=True, result=(True, [])
            ),
        }

        result = handler.aggregate_partial_results(partial_results, "Custom reasoning text")

        assert result.verification_reasoning == "Custom reasoning text"


# ---------------------------------------------------------------------------
# Integration: verify_draft_with_degradation on VerificationAgent
# ---------------------------------------------------------------------------

class TestVerifyDraftWithDegradation:
    """Integration tests using VerificationAgent.verify_draft_with_degradation()."""

    def _make_agent(self):
        """Build a minimal VerificationAgent with mocked checkers."""
        from backend.verification.agent.verification_agent import VerificationAgent
        from backend.verification.config import VerificationConfig

        config = VerificationConfig()
        agent = VerificationAgent.__new__(VerificationAgent)
        agent.config = config
        agent._semaphore = asyncio.Semaphore(10)
        agent._cache = None
        agent._circuit_breakers = {}
        agent._last_valid_result = None
        agent._total_verifications = 0
        agent._total_tokens_used = 0

        # Set up a simple logger that accepts keyword arguments
        mock_logger = MagicMock()
        agent._logger = mock_logger

        # Mock checkers
        agent.price_checker = MagicMock()
        agent.policy_checker = MagicMock()
        agent.relevance_checker = MagicMock()

        return agent

    def _make_state(self) -> dict:
        return {
            "objection_text": "Giá iPhone 15 bao nhiêu?",
            "draft_response": "iPhone 15 giá 29,990,000 VND với bảo hành 1 năm.",
            "retry_count": 0,
            "max_retries": 3,
            "workflow_status": "verifying",
            "execution_log": [],
        }

    @pytest.mark.asyncio
    async def test_all_checkers_pass_returns_approved(self):
        agent = self._make_agent()
        agent.price_checker.check_price_accuracy.return_value = (True, [])
        agent.policy_checker.check_policy_authenticity.return_value = (True, [])
        agent.relevance_checker.check_topic_relevance.return_value = (True, [])

        result = await agent.verify_draft_with_degradation(self._make_state())

        assert result.is_approved is True
        assert result.criteria.overall_pass is True

    @pytest.mark.asyncio
    async def test_price_checker_exception_continues_with_warning(self):
        agent = self._make_agent()
        agent.price_checker.check_price_accuracy.side_effect = RuntimeError("price DB down")
        agent.policy_checker.check_policy_authenticity.return_value = (True, [])
        agent.relevance_checker.check_topic_relevance.return_value = (True, [])

        result = await agent.verify_draft_with_degradation(self._make_state())

        # Verification continues — price defaults to PASS with warning
        assert result.criteria.price_accuracy_pass is True
        assert len(result.criteria.price_issues) == 1
        assert result.criteria.price_issues[0].severity == IssueSeverity.MINOR
        # Overall still passes (policy + relevance both pass)
        assert result.is_approved is True

    @pytest.mark.asyncio
    async def test_two_checker_failures_returns_fallback_result(self):
        agent = self._make_agent()
        agent.price_checker.check_price_accuracy.side_effect = RuntimeError("price DB down")
        agent.policy_checker.check_policy_authenticity.side_effect = TimeoutError("policy timeout")
        agent.relevance_checker.check_topic_relevance.return_value = (True, [])

        # Need _build_fallback_verification_result — patch it
        fallback_result = VerificationResult(
            criteria=RubricCriteria(
                price_accuracy_pass=False,
                policy_authenticity_pass=False,
                topic_relevance_pass=True,
            ),
            verification_reasoning="Fallback: 2 checkers failed",
            execution_time_seconds=0.1,
            llm_tokens_used=0,
        )
        agent._build_fallback_verification_result = MagicMock(return_value=fallback_result)

        result = await agent.verify_draft_with_degradation(self._make_state())

        # Fallback was invoked
        agent._build_fallback_verification_result.assert_called_once()
        assert result is fallback_result

    @pytest.mark.asyncio
    async def test_single_checker_failure_does_not_call_fallback(self):
        agent = self._make_agent()
        agent.price_checker.check_price_accuracy.side_effect = RuntimeError("price DB down")
        agent.policy_checker.check_policy_authenticity.return_value = (True, [])
        agent.relevance_checker.check_topic_relevance.return_value = (True, [])

        agent._build_fallback_verification_result = MagicMock()

        await agent.verify_draft_with_degradation(self._make_state())

        agent._build_fallback_verification_result.assert_not_called()
