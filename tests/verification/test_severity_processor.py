"""
Unit tests for SeverityBasedProcessor

Covers:
- Priority ordering (critical before major before minor)
- Early termination on critical issue detection
- Mixed issue type handling
- Empty issue list handling
- Processing order recommendations
"""

import pytest
from verification.models.verification import (
    IssueSeverity,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
)
from verification.utils.severity_processor import (
    SeverityBasedProcessor,
    SeverityLevel,
    IssuePriority,
    DEFAULT_CHECKER_ORDER,
)
from verification.utils.early_termination import EarlyTerminationManager
from verification.config.thresholds_config import (
    EscalationThresholds,
    VerificationThresholdsConfig,
    get_default_thresholds_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price_issue(severity: IssueSeverity) -> PriceIssue:
    return PriceIssue(
        product_name="Test Product",
        severity=severity,
        explanation=f"Price issue with severity {severity.value}",
    )


def make_policy_issue(severity: IssueSeverity) -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Test policy",
        policy_type="warranty",
        is_fabricated=severity == IssueSeverity.CRITICAL,
        is_inaccurate=severity == IssueSeverity.MAJOR,
        severity=severity,
        explanation=f"Policy issue with severity {severity.value}",
    )


def make_relevance_issue(severity: IssueSeverity) -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="Test intent",
        response_coverage=0.3 if severity == IssueSeverity.CRITICAL else 0.6,
        severity=severity,
        explanation=f"Relevance issue with severity {severity.value}",
    )


def make_processor_with_stop_on_first() -> SeverityBasedProcessor:
    """Processor configured to stop on first critical issue."""
    config = get_default_thresholds_config()
    config.escalation.stop_on_first_critical = True
    config.escalation.early_termination_enabled = True
    manager = EarlyTerminationManager(config)
    return SeverityBasedProcessor(early_termination_manager=manager)


# ---------------------------------------------------------------------------
# SeverityLevel tests
# ---------------------------------------------------------------------------

class TestSeverityLevel:
    def test_critical_has_lowest_numeric_value(self):
        assert SeverityLevel.CRITICAL < SeverityLevel.MAJOR
        assert SeverityLevel.MAJOR < SeverityLevel.MINOR

    def test_from_issue_severity_critical(self):
        assert SeverityLevel.from_issue_severity(IssueSeverity.CRITICAL) == SeverityLevel.CRITICAL

    def test_from_issue_severity_major(self):
        assert SeverityLevel.from_issue_severity(IssueSeverity.MAJOR) == SeverityLevel.MAJOR

    def test_from_issue_severity_minor(self):
        assert SeverityLevel.from_issue_severity(IssueSeverity.MINOR) == SeverityLevel.MINOR


# ---------------------------------------------------------------------------
# IssuePriority tests
# ---------------------------------------------------------------------------

class TestIssuePriority:
    def test_from_price_issue_critical(self):
        issue = make_price_issue(IssueSeverity.CRITICAL)
        ip = IssuePriority.from_issue(issue)
        assert ip.priority == SeverityLevel.CRITICAL
        assert ip.issue_type == "PriceIssue"
        assert ip.issue is issue

    def test_from_policy_issue_major(self):
        issue = make_policy_issue(IssueSeverity.MAJOR)
        ip = IssuePriority.from_issue(issue)
        assert ip.priority == SeverityLevel.MAJOR

    def test_from_relevance_issue_minor(self):
        issue = make_relevance_issue(IssueSeverity.MINOR)
        ip = IssuePriority.from_issue(issue)
        assert ip.priority == SeverityLevel.MINOR

    def test_ordering_critical_before_major(self):
        critical = IssuePriority.from_issue(make_price_issue(IssueSeverity.CRITICAL))
        major = IssuePriority.from_issue(make_price_issue(IssueSeverity.MAJOR))
        assert critical < major

    def test_ordering_major_before_minor(self):
        major = IssuePriority.from_issue(make_policy_issue(IssueSeverity.MAJOR))
        minor = IssuePriority.from_issue(make_policy_issue(IssueSeverity.MINOR))
        assert major < minor


# ---------------------------------------------------------------------------
# SeverityBasedProcessor — priority ordering
# ---------------------------------------------------------------------------

class TestProcessByPriority:
    def test_empty_list_yields_nothing(self):
        processor = SeverityBasedProcessor()
        result = list(processor.process_by_priority([]))
        assert result == []

    def test_single_minor_issue(self):
        processor = SeverityBasedProcessor()
        issue = make_price_issue(IssueSeverity.MINOR)
        result = list(processor.process_by_priority([issue]))
        assert len(result) == 1
        assert result[0].issue is issue

    def test_critical_before_major_before_minor(self):
        processor = SeverityBasedProcessor()
        minor = make_price_issue(IssueSeverity.MINOR)
        major = make_policy_issue(IssueSeverity.MAJOR)
        critical = make_relevance_issue(IssueSeverity.CRITICAL)

        # Pass in reverse order to verify sorting
        result = list(processor.process_by_priority([minor, major, critical]))
        priorities = [r.priority for r in result]
        assert priorities == sorted(priorities)
        assert priorities[0] == SeverityLevel.CRITICAL

    def test_mixed_types_sorted_by_severity(self):
        processor = SeverityBasedProcessor()
        issues = [
            make_price_issue(IssueSeverity.MINOR),
            make_policy_issue(IssueSeverity.CRITICAL),
            make_relevance_issue(IssueSeverity.MAJOR),
            make_price_issue(IssueSeverity.MAJOR),
        ]
        result = list(processor.process_by_priority(issues))
        priorities = [r.priority for r in result]
        assert priorities == sorted(priorities)

    def test_all_same_severity_yields_all(self):
        processor = SeverityBasedProcessor()
        issues = [
            make_price_issue(IssueSeverity.MAJOR),
            make_policy_issue(IssueSeverity.MAJOR),
            make_relevance_issue(IssueSeverity.MAJOR),
        ]
        result = list(processor.process_by_priority(issues))
        assert len(result) == 3


# ---------------------------------------------------------------------------
# SeverityBasedProcessor — early termination
# ---------------------------------------------------------------------------

class TestEarlyTermination:
    def test_stops_after_critical_when_stop_on_first_enabled(self):
        processor = make_processor_with_stop_on_first()
        issues = [
            make_price_issue(IssueSeverity.CRITICAL),
            make_policy_issue(IssueSeverity.MAJOR),
            make_relevance_issue(IssueSeverity.MINOR),
        ]
        result = list(processor.process_by_priority(issues))
        # Should stop after the critical issue
        assert len(result) == 1
        assert result[0].priority == SeverityLevel.CRITICAL

    def test_no_early_termination_without_critical(self):
        processor = make_processor_with_stop_on_first()
        issues = [
            make_price_issue(IssueSeverity.MAJOR),
            make_policy_issue(IssueSeverity.MINOR),
        ]
        result = list(processor.process_by_priority(issues))
        assert len(result) == 2

    def test_has_encountered_critical_after_processing(self):
        processor = make_processor_with_stop_on_first()
        issues = [make_price_issue(IssueSeverity.CRITICAL)]
        list(processor.process_by_priority(issues))
        assert processor.has_encountered_critical()

    def test_has_not_encountered_critical_with_only_major(self):
        processor = SeverityBasedProcessor()
        issues = [make_price_issue(IssueSeverity.MAJOR)]
        list(processor.process_by_priority(issues))
        assert not processor.has_encountered_critical()

    def test_encountered_severities_tracked(self):
        processor = SeverityBasedProcessor()
        issues = [
            make_price_issue(IssueSeverity.MINOR),
            make_policy_issue(IssueSeverity.MAJOR),
        ]
        list(processor.process_by_priority(issues))
        assert SeverityLevel.MINOR in processor.encountered_severities
        assert SeverityLevel.MAJOR in processor.encountered_severities
        assert SeverityLevel.CRITICAL not in processor.encountered_severities


# ---------------------------------------------------------------------------
# SeverityBasedProcessor — get_sorted_issues (no side effects)
# ---------------------------------------------------------------------------

class TestGetSortedIssues:
    def test_returns_sorted_list(self):
        processor = SeverityBasedProcessor()
        issues = [
            make_price_issue(IssueSeverity.MINOR),
            make_policy_issue(IssueSeverity.CRITICAL),
            make_relevance_issue(IssueSeverity.MAJOR),
        ]
        result = processor.get_sorted_issues(issues)
        priorities = [r.priority for r in result]
        assert priorities == sorted(priorities)

    def test_does_not_trigger_early_termination(self):
        """get_sorted_issues should not affect encountered_severities."""
        processor = make_processor_with_stop_on_first()
        issues = [make_price_issue(IssueSeverity.CRITICAL)]
        processor.get_sorted_issues(issues)
        # No side effects — encountered_severities should be empty
        assert len(processor.encountered_severities) == 0


# ---------------------------------------------------------------------------
# SeverityBasedProcessor — processing order recommendations
# ---------------------------------------------------------------------------

class TestGetProcessingOrder:
    def test_default_order_without_history(self):
        processor = SeverityBasedProcessor()
        order = processor.get_processing_order()
        assert order == list(DEFAULT_CHECKER_ORDER)

    def test_checker_with_most_critical_issues_comes_first(self):
        processor = SeverityBasedProcessor()
        # Record many critical issues for "policy"
        for _ in range(5):
            processor.record_checker_results(
                "policy", [make_policy_issue(IssueSeverity.CRITICAL)]
            )
        # Record fewer for "price"
        processor.record_checker_results(
            "price", [make_price_issue(IssueSeverity.MAJOR)]
        )
        order = processor.get_processing_order()
        assert order[0] == "policy"

    def test_record_checker_results_updates_counts(self):
        processor = SeverityBasedProcessor()
        processor.record_checker_results(
            "price",
            [
                make_price_issue(IssueSeverity.CRITICAL),
                make_price_issue(IssueSeverity.MAJOR),
                make_price_issue(IssueSeverity.MINOR),
            ],
        )
        counts = processor._checker_severity_counts["price"]
        assert counts["critical"] == 1
        assert counts["major"] == 1
        assert counts["minor"] == 1

    def test_reset_history_clears_counts(self):
        processor = SeverityBasedProcessor()
        processor.record_checker_results(
            "price", [make_price_issue(IssueSeverity.CRITICAL)]
        )
        processor.reset_history()
        assert processor._checker_severity_counts["price"]["critical"] == 0
        assert len(processor.encountered_severities) == 0

    def test_all_checkers_present_in_order(self):
        processor = SeverityBasedProcessor()
        order = processor.get_processing_order()
        assert set(order) == {"price", "policy", "relevance"}
        assert len(order) == 3
