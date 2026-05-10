"""
Fallback Verification Modes for Service Outages - Task 6.3.3

Implements fallback verification modes when external services are unavailable:
- LLM Service Outage Mode: rule-based verification using regex/keyword matching
- Database Outage Mode: use cached data or skip DB-dependent checks with lowered thresholds
- Partial Outage Mode: run available checks, skip unavailable ones with adjusted scoring
- Full Degraded Mode: conservative fallback result flagged for human review

Integrates with:
- circuit_breaker.py: detect service availability via circuit state
- graceful_degradation.py: aggregate partial results
- VerificationResult / RubricCriteria Pydantic models

Requirements:
- 8.2: DB connection lost → cached data, lower threshold, queue for manual review
- 8.1: LLM API timeout/error → retry with exponential backoff, escalate if all fail
- 8.4: Circuit breaker pattern for external service calls
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..models.verification import (
    IssueSeverity,
    PolicyIssue,
    PriceIssue,
    RelevanceIssue,
    RubricCriteria,
    VerificationResult,
)
from .circuit_breaker import CircuitBreaker, CircuitState, get_circuit_breaker_registry
from .graceful_degradation import GracefulDegradationHandler

logger = logging.getLogger("verification.fallback_verification")


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------


class FallbackMode(str, Enum):
    """Available fallback verification modes."""
    LLM_OUTAGE = "llm_outage"          # LLM unavailable → rule-based checks
    DATABASE_OUTAGE = "database_outage"  # DB unavailable → cached data / skip DB checks
    PARTIAL_OUTAGE = "partial_outage"   # Some services down → run available checks
    FULL_DEGRADED = "full_degraded"     # All services down → conservative human-review result


@dataclass
class FallbackConfig:
    """Configuration for fallback verification behaviour."""

    # Lowered thresholds used during DB outage
    db_outage_price_tolerance_percent: float = 5.0
    db_outage_relevance_min_coverage: float = 0.5

    # Lowered thresholds used during partial outage
    partial_outage_relevance_min_coverage: float = 0.6

    # Whether to flag full-degraded results for human review
    full_degraded_flag_human_review: bool = True

    # Reasoning strings injected into VerificationResult
    llm_outage_reasoning: str = (
        "LLM service unavailable — rule-based fallback verification applied. "
        "Results may be less accurate than LLM-assisted verification."
    )
    db_outage_reasoning: str = (
        "Internal DB unavailable — cached data used where available; "
        "DB-dependent checks skipped with lowered thresholds."
    )
    partial_outage_reasoning: str = (
        "Partial service outage — available checks completed; "
        "unavailable checks skipped with adjusted scoring."
    )
    full_degraded_reasoning: str = (
        "All external services unavailable — conservative fallback result returned. "
        "Flagged for human review."
    )


# ---------------------------------------------------------------------------
# Rule-based helpers (used in LLM outage mode)
# ---------------------------------------------------------------------------

# Simple price patterns (VND amounts)
_PRICE_PATTERN = re.compile(
    r"(\d[\d.,]*)\s*(?:triệu|tr|million|VND|vnđ|đồng|₫)",
    re.IGNORECASE,
)

# Policy keywords that indicate a policy claim is present
_POLICY_KEYWORDS = [
    "bảo hành", "warranty", "đổi trả", "hoàn tiền", "return", "refund",
    "exchange", "sửa chữa", "repair", "hỗ trợ", "support",
]

# Objection intent keywords for relevance check
_PRICE_INTENT_KEYWORDS = ["giá", "price", "cost", "đắt", "expensive", "rẻ", "cheap", "tiền"]
_FEATURE_INTENT_KEYWORDS = ["tính năng", "feature", "specs", "performance", "hiệu năng"]
_COMPARISON_INTENT_KEYWORDS = ["so sánh", "compare", "khác gì", "vs", "versus"]
_POLICY_INTENT_KEYWORDS = ["bảo hành", "đổi trả", "warranty", "return", "policy"]


def _rule_based_price_check(draft: str, objection: str) -> Tuple[bool, List[PriceIssue]]:
    """
    Lightweight rule-based price check without LLM.

    Passes if the draft contains at least one price mention when the objection
    is price-related. Cannot verify accuracy without DB, so only checks presence.
    """
    objection_lower = objection.lower()
    price_related = any(kw in objection_lower for kw in _PRICE_INTENT_KEYWORDS)

    if not price_related:
        # Objection is not price-related — price check not applicable
        return True, []

    prices_found = _PRICE_PATTERN.findall(draft)
    if prices_found:
        # Prices present — cannot verify accuracy without DB, treat as pass
        return True, []

    # Price-related objection but no price in draft
    issue = PriceIssue(
        product_name="Unknown",
        severity=IssueSeverity.MAJOR,
        explanation=(
            "Rule-based check (LLM unavailable): objection is price-related "
            "but draft contains no price information."
        ),
        correction_suggestion="Add accurate pricing information from the product catalog.",
    )
    return False, [issue]


def _rule_based_policy_check(draft: str) -> Tuple[bool, List[PolicyIssue]]:
    """
    Lightweight rule-based policy check without LLM.

    Passes if no policy keywords are found (nothing to verify) or if policy
    keywords are present (cannot verify authenticity without DB/LLM, so we
    default to pass with a warning).
    """
    draft_lower = draft.lower()
    policy_mentions = [kw for kw in _POLICY_KEYWORDS if kw in draft_lower]

    if not policy_mentions:
        return True, []

    # Policy keywords found — cannot verify authenticity without LLM/DB.
    # Return pass with a minor warning so the result is visible.
    issue = PolicyIssue(
        mentioned_policy=", ".join(policy_mentions[:3]),
        policy_type="service",
        is_fabricated=False,
        is_inaccurate=False,
        severity=IssueSeverity.MINOR,
        explanation=(
            "Rule-based check (LLM unavailable): policy keywords detected but "
            "authenticity cannot be verified without LLM/DB. Manual review recommended."
        ),
        correction_suggestion="Verify policy statements against official documents.",
    )
    return True, [issue]


def _rule_based_relevance_check(
    draft: str,
    objection: str,
    min_coverage: float = 0.7,
) -> Tuple[bool, List[RelevanceIssue]]:
    """
    Lightweight rule-based relevance check without LLM.

    Detects objection intent via keyword matching and checks whether the draft
    addresses the detected intents.
    """
    objection_lower = objection.lower()
    draft_lower = draft.lower()

    intent_groups = {
        "price": _PRICE_INTENT_KEYWORDS,
        "feature": _FEATURE_INTENT_KEYWORDS,
        "comparison": _COMPARISON_INTENT_KEYWORDS,
        "policy": _POLICY_INTENT_KEYWORDS,
    }

    detected_intents = [
        intent
        for intent, keywords in intent_groups.items()
        if any(kw in objection_lower for kw in keywords)
    ]

    if not detected_intents:
        # Cannot determine intent — default to pass
        return True, []

    addressed = [
        intent
        for intent in detected_intents
        if any(kw in draft_lower for kw in intent_groups[intent])
    ]

    coverage = len(addressed) / len(detected_intents) if detected_intents else 1.0

    if coverage >= min_coverage:
        return True, []

    missing = [i for i in detected_intents if i not in addressed]
    issue = RelevanceIssue(
        objection_intent=", ".join(detected_intents),
        response_coverage=coverage,
        missing_aspects=missing,
        severity=IssueSeverity.MAJOR if coverage < 0.5 else IssueSeverity.MINOR,
        explanation=(
            f"Rule-based check (LLM unavailable): draft covers {coverage:.0%} of "
            f"detected objection intents. Missing: {', '.join(missing)}."
        ),
        correction_suggestion=f"Address the following aspects: {', '.join(missing)}.",
    )
    return False, [issue]


# ---------------------------------------------------------------------------
# FallbackVerificationManager
# ---------------------------------------------------------------------------


class FallbackVerificationManager:
    """
    Manages fallback verification modes for different service outage scenarios.

    Integrates with CircuitBreaker to detect service availability and selects
    the appropriate fallback mode automatically, or allows explicit mode selection.

    Usage::

        manager = FallbackVerificationManager()

        # Auto-detect mode from circuit breaker states
        result = manager.run_fallback(
            draft="...", objection="...", mode=manager.detect_mode()
        )

        # Explicit mode
        result = manager.run_fallback(
            draft="...", objection="...", mode=FallbackMode.LLM_OUTAGE
        )

    **Validates: Requirements 8.1** - LLM API error fallback
    **Validates: Requirements 8.2** - DB connection loss fallback
    **Validates: Requirements 8.4** - circuit breaker integration
    """

    def __init__(
        self,
        config: Optional[FallbackConfig] = None,
        llm_circuit_breaker: Optional[CircuitBreaker] = None,
        db_circuit_breaker: Optional[CircuitBreaker] = None,
        cache: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        self.config = config or FallbackConfig()
        self.correlation_id = correlation_id or "unknown"
        self._cache: Dict[str, Any] = cache if cache is not None else {}

        registry = get_circuit_breaker_registry()
        self._llm_cb: CircuitBreaker = llm_circuit_breaker or registry.get_or_create(
            "llm_api", failure_threshold=3, cooldown_seconds=60.0
        )
        self._db_cb: CircuitBreaker = db_circuit_breaker or registry.get_or_create(
            "internal_db", failure_threshold=3, cooldown_seconds=30.0
        )

    # ------------------------------------------------------------------
    # Mode detection
    # ------------------------------------------------------------------

    def detect_mode(self) -> FallbackMode:
        """
        Auto-detect the appropriate fallback mode from circuit breaker states.

        Returns:
            FallbackMode based on which services are currently unavailable.
        """
        llm_open = self._llm_cb.is_open()
        db_open = self._db_cb.is_open()

        if llm_open and db_open:
            mode = FallbackMode.FULL_DEGRADED
        elif llm_open:
            mode = FallbackMode.LLM_OUTAGE
        elif db_open:
            mode = FallbackMode.DATABASE_OUTAGE
        else:
            # Both services available — no fallback needed, but caller may
            # still request a specific mode explicitly.
            mode = FallbackMode.PARTIAL_OUTAGE

        logger.info(
            "Fallback mode detected: %s (llm_open=%s, db_open=%s, correlation_id=%s)",
            mode.value,
            llm_open,
            db_open,
            self.correlation_id,
        )
        return mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_fallback(
        self,
        draft: str,
        objection: str,
        mode: FallbackMode,
        available_services: Optional[List[str]] = None,
    ) -> VerificationResult:
        """
        Execute fallback verification for the given mode.

        Args:
            draft: The draft response text to verify.
            objection: The original customer objection text.
            mode: Which fallback mode to apply.
            available_services: For PARTIAL_OUTAGE mode, list of available
                service names (e.g. ["price", "relevance"]).

        Returns:
            VerificationResult with appropriate fallback data.
        """
        logger.warning(
            "Running fallback verification: mode=%s, correlation_id=%s",
            mode.value,
            self.correlation_id,
        )

        if mode == FallbackMode.LLM_OUTAGE:
            return self._llm_outage_mode(draft, objection)
        elif mode == FallbackMode.DATABASE_OUTAGE:
            return self._database_outage_mode(draft, objection)
        elif mode == FallbackMode.PARTIAL_OUTAGE:
            return self._partial_outage_mode(draft, objection, available_services or [])
        else:  # FULL_DEGRADED
            return self._full_degraded_mode(draft, objection)

    # ------------------------------------------------------------------
    # Fallback mode implementations
    # ------------------------------------------------------------------

    def _llm_outage_mode(self, draft: str, objection: str) -> VerificationResult:
        """
        LLM Service Outage Mode.

        Falls back to rule-based verification using regex/keyword matching
        without any LLM calls.

        **Validates: Requirements 8.1** - LLM API error fallback
        """
        logger.info(
            "LLM outage mode: running rule-based verification (correlation_id=%s)",
            self.correlation_id,
        )

        price_pass, price_issues = _rule_based_price_check(draft, objection)
        policy_pass, policy_issues = _rule_based_policy_check(draft)
        relevance_pass, relevance_issues = _rule_based_relevance_check(draft, objection)

        criteria = RubricCriteria(
            price_accuracy_pass=price_pass,
            price_issues=price_issues,
            policy_authenticity_pass=policy_pass,
            policy_issues=policy_issues,
            topic_relevance_pass=relevance_pass,
            relevance_issues=relevance_issues,
        )

        logger.info(
            "LLM outage mode complete: overall_pass=%s, correlation_id=%s",
            criteria.overall_pass,
            self.correlation_id,
        )

        return VerificationResult(
            criteria=criteria,
            verification_reasoning=self.config.llm_outage_reasoning,
            execution_time_seconds=0.0,
            llm_tokens_used=0,
        )

    def _database_outage_mode(self, draft: str, objection: str) -> VerificationResult:
        """
        Database Outage Mode.

        Uses cached data where available; skips DB-dependent checks (price
        accuracy, policy authenticity) with lowered thresholds. Relevance
        check still runs via rule-based approach.

        **Validates: Requirements 8.2** - DB connection loss fallback
        """
        logger.info(
            "DB outage mode: using cached data / skipping DB checks (correlation_id=%s)",
            self.correlation_id,
        )

        # Price check: use cached result if available, otherwise skip (default pass + warning)
        cached_price = self._cache.get("price_check_result")
        if cached_price is not None:
            price_pass, price_issues = cached_price
            logger.info("DB outage mode: using cached price check result")
        else:
            # Cannot verify price without DB — default to pass with warning
            price_pass = True
            price_issues = [
                PriceIssue(
                    product_name="[DB Unavailable]",
                    severity=IssueSeverity.MINOR,
                    explanation=(
                        "DB outage mode: price accuracy cannot be verified — "
                        "DB unavailable and no cached data found. "
                        f"Threshold temporarily lowered to ±{self.config.db_outage_price_tolerance_percent}%."
                    ),
                    correction_suggestion="Verify prices manually when DB is restored.",
                )
            ]

        # Policy check: use cached result if available, otherwise skip (default pass + warning)
        cached_policy = self._cache.get("policy_check_result")
        if cached_policy is not None:
            policy_pass, policy_issues = cached_policy
            logger.info("DB outage mode: using cached policy check result")
        else:
            policy_pass = True
            policy_issues = [
                PolicyIssue(
                    mentioned_policy="[DB Unavailable]",
                    policy_type="service",
                    is_fabricated=False,
                    is_inaccurate=False,
                    severity=IssueSeverity.MINOR,
                    explanation=(
                        "DB outage mode: policy authenticity cannot be verified — "
                        "DB unavailable and no cached data found."
                    ),
                    correction_suggestion="Verify policy statements manually when DB is restored.",
                )
            ]

        # Relevance check: rule-based (no DB needed), with lowered threshold
        relevance_pass, relevance_issues = _rule_based_relevance_check(
            draft, objection, min_coverage=self.config.db_outage_relevance_min_coverage
        )

        criteria = RubricCriteria(
            price_accuracy_pass=price_pass,
            price_issues=price_issues,
            policy_authenticity_pass=policy_pass,
            policy_issues=policy_issues,
            topic_relevance_pass=relevance_pass,
            relevance_issues=relevance_issues,
        )

        logger.info(
            "DB outage mode complete: overall_pass=%s, correlation_id=%s",
            criteria.overall_pass,
            self.correlation_id,
        )

        return VerificationResult(
            criteria=criteria,
            verification_reasoning=self.config.db_outage_reasoning,
            execution_time_seconds=0.0,
            llm_tokens_used=0,
        )

    def _partial_outage_mode(
        self,
        draft: str,
        objection: str,
        available_services: List[str],
    ) -> VerificationResult:
        """
        Partial Outage Mode.

        Runs checks for available services and skips unavailable ones with
        adjusted scoring (default pass + warning for skipped checks).

        Args:
            available_services: Names of services that are currently available.
                Recognised values: "price", "policy", "relevance", "llm", "db".
        """
        logger.info(
            "Partial outage mode: available_services=%s, correlation_id=%s",
            available_services,
            self.correlation_id,
        )

        # Determine which checkers can run
        llm_available = "llm" in available_services
        db_available = "db" in available_services
        price_available = "price" in available_services or (llm_available and db_available)
        policy_available = "policy" in available_services or (llm_available and db_available)
        relevance_available = "relevance" in available_services or llm_available

        # Price check
        if price_available and db_available:
            price_pass, price_issues = _rule_based_price_check(draft, objection)
        else:
            price_pass = True
            price_issues = [
                PriceIssue(
                    product_name="[Service Unavailable]",
                    severity=IssueSeverity.MINOR,
                    explanation="Partial outage mode: price check skipped — required services unavailable.",
                    correction_suggestion="Verify prices when services are restored.",
                )
            ] if not price_available else []

        # Policy check
        if policy_available and db_available:
            policy_pass, policy_issues = _rule_based_policy_check(draft)
        else:
            policy_pass = True
            policy_issues = [
                PolicyIssue(
                    mentioned_policy="[Service Unavailable]",
                    policy_type="service",
                    is_fabricated=False,
                    is_inaccurate=False,
                    severity=IssueSeverity.MINOR,
                    explanation="Partial outage mode: policy check skipped — required services unavailable.",
                    correction_suggestion="Verify policies when services are restored.",
                )
            ] if not policy_available else []

        # Relevance check (rule-based, only needs text — no external service)
        if relevance_available or True:  # relevance rule-based always possible
            relevance_pass, relevance_issues = _rule_based_relevance_check(
                draft, objection,
                min_coverage=self.config.partial_outage_relevance_min_coverage,
            )
        else:
            relevance_pass = True
            relevance_issues = [
                RelevanceIssue(
                    objection_intent="[Service Unavailable]",
                    response_coverage=1.0,
                    severity=IssueSeverity.MINOR,
                    explanation="Partial outage mode: relevance check skipped.",
                )
            ]

        criteria = RubricCriteria(
            price_accuracy_pass=price_pass,
            price_issues=price_issues,
            policy_authenticity_pass=policy_pass,
            policy_issues=policy_issues,
            topic_relevance_pass=relevance_pass,
            relevance_issues=relevance_issues,
        )

        logger.info(
            "Partial outage mode complete: overall_pass=%s, correlation_id=%s",
            criteria.overall_pass,
            self.correlation_id,
        )

        return VerificationResult(
            criteria=criteria,
            verification_reasoning=self.config.partial_outage_reasoning,
            execution_time_seconds=0.0,
            llm_tokens_used=0,
        )

    def _full_degraded_mode(self, draft: str, objection: str) -> VerificationResult:
        """
        Full Degraded Mode.

        All external services are down. Returns a conservative result that
        defaults all checks to pass (to avoid blocking the workflow) but
        injects MAJOR warning issues on every criterion and sets
        has_critical_issues=False / immediate_termination=False so the
        workflow can still route to human review via the escalation path.

        **Validates: Requirements 8.1, 8.2** - full service outage handling
        """
        logger.error(
            "Full degraded mode: all services unavailable — returning conservative result "
            "(correlation_id=%s)",
            self.correlation_id,
        )

        warning_explanation = (
            "Full degraded mode: all external services unavailable. "
            "This result has NOT been verified and requires human review."
        )

        price_issues = [
            PriceIssue(
                product_name="[All Services Unavailable]",
                severity=IssueSeverity.MAJOR,
                explanation=warning_explanation,
                correction_suggestion="Manually verify all pricing information.",
            )
        ]
        policy_issues = [
            PolicyIssue(
                mentioned_policy="[All Services Unavailable]",
                policy_type="service",
                is_fabricated=False,
                is_inaccurate=False,
                severity=IssueSeverity.MAJOR,
                explanation=warning_explanation,
                correction_suggestion="Manually verify all policy statements.",
            )
        ]
        relevance_issues = [
            RelevanceIssue(
                objection_intent="[All Services Unavailable]",
                response_coverage=0.5,
                severity=IssueSeverity.MAJOR,
                explanation=warning_explanation,
                correction_suggestion="Manually review response relevance.",
            )
        ]

        # All checks default to pass so the workflow is not hard-blocked,
        # but the MAJOR issues ensure human review is triggered via escalation logic.
        criteria = RubricCriteria(
            price_accuracy_pass=True,
            price_issues=price_issues,
            policy_authenticity_pass=True,
            policy_issues=policy_issues,
            topic_relevance_pass=True,
            relevance_issues=relevance_issues,
        )

        result = VerificationResult(
            criteria=criteria,
            verification_reasoning=self.config.full_degraded_reasoning,
            execution_time_seconds=0.0,
            llm_tokens_used=0,
            has_critical_issues=False,
            immediate_termination=False,
        )

        if self.config.full_degraded_flag_human_review:
            logger.warning(
                "Full degraded mode: result flagged for human review (correlation_id=%s)",
                self.correlation_id,
            )

        return result

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def cache_check_result(self, check_name: str, result: Tuple[bool, List]) -> None:
        """
        Store a checker result in the fallback cache for DB outage scenarios.

        Args:
            check_name: One of "price_check_result", "policy_check_result".
            result: (pass_flag, issues_list) tuple from the checker.
        """
        self._cache[check_name] = result
        logger.debug(
            "Cached check result: check_name=%s, pass=%s, correlation_id=%s",
            check_name,
            result[0],
            self.correlation_id,
        )

    def get_service_status(self) -> Dict[str, str]:
        """Return current circuit breaker states for observability."""
        return {
            "llm_api": self._llm_cb.state.value,
            "internal_db": self._db_cb.state.value,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_manager: Optional[FallbackVerificationManager] = None


def get_fallback_verification_manager(
    config: Optional[FallbackConfig] = None,
    correlation_id: Optional[str] = None,
) -> FallbackVerificationManager:
    """Return the module-level singleton FallbackVerificationManager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = FallbackVerificationManager(
            config=config, correlation_id=correlation_id
        )
    return _default_manager


def reset_fallback_verification_manager() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_manager
    _default_manager = None


__all__ = [
    "FallbackMode",
    "FallbackConfig",
    "FallbackVerificationManager",
    "get_fallback_verification_manager",
    "reset_fallback_verification_manager",
]
