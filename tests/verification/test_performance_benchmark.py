"""
Unit Tests: PerformanceBenchmark

Tests for the performance benchmarking module that validates optimization work
from Tasks 5.4.1–5.4.4.

Supports Task 5.4.5: Add performance benchmarking cho optimization validation
"""

import asyncio
import pytest

from backend.verification.utils.performance_benchmark import (
    BenchmarkReport,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    _improvement_percent,
)


# ---------------------------------------------------------------------------
# BenchmarkReport model tests
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_improvement_percent_computed_correctly(self):
        report = BenchmarkReport(
            benchmark_name="test",
            baseline_latency_ms=100.0,
            optimized_latency_ms=60.0,
            passed=True,
        )
        assert abs(report.improvement_percent - 40.0) < 0.01

    def test_improvement_percent_zero_when_no_change(self):
        report = BenchmarkReport(
            benchmark_name="test",
            baseline_latency_ms=100.0,
            optimized_latency_ms=100.0,
            passed=False,
        )
        assert report.improvement_percent == 0.0

    def test_improvement_percent_negative_when_optimized_is_slower(self):
        report = BenchmarkReport(
            benchmark_name="test",
            baseline_latency_ms=50.0,
            optimized_latency_ms=80.0,
            passed=False,
        )
        assert report.improvement_percent < 0.0

    def test_improvement_percent_zero_when_baseline_is_zero(self):
        report = BenchmarkReport(
            benchmark_name="test",
            baseline_latency_ms=0.0,
            optimized_latency_ms=0.0,
            passed=False,
        )
        assert report.improvement_percent == 0.0

    def test_passed_field_stored_correctly(self):
        report = BenchmarkReport(
            benchmark_name="my_bench",
            baseline_latency_ms=200.0,
            optimized_latency_ms=100.0,
            passed=True,
        )
        assert report.passed is True

    def test_details_default_empty_dict(self):
        report = BenchmarkReport(
            benchmark_name="x",
            baseline_latency_ms=10.0,
            optimized_latency_ms=5.0,
            passed=True,
        )
        assert report.details == {}

    def test_details_stored_correctly(self):
        report = BenchmarkReport(
            benchmark_name="x",
            baseline_latency_ms=10.0,
            optimized_latency_ms=5.0,
            passed=True,
            details={"checks_skipped": 2},
        )
        assert report.details["checks_skipped"] == 2

    def test_benchmark_name_stored(self):
        report = BenchmarkReport(
            benchmark_name="early_termination",
            baseline_latency_ms=100.0,
            optimized_latency_ms=30.0,
            passed=True,
        )
        assert report.benchmark_name == "early_termination"

    def test_full_improvement_100_percent(self):
        report = BenchmarkReport(
            benchmark_name="perfect",
            baseline_latency_ms=100.0,
            optimized_latency_ms=0.0,
            passed=True,
        )
        assert abs(report.improvement_percent - 100.0) < 0.01


# ---------------------------------------------------------------------------
# BenchmarkSuiteResult tests
# ---------------------------------------------------------------------------

class TestBenchmarkSuiteResult:
    def _make_report(self, name: str, passed: bool) -> BenchmarkReport:
        return BenchmarkReport(
            benchmark_name=name,
            baseline_latency_ms=100.0,
            optimized_latency_ms=50.0 if passed else 110.0,
            passed=passed,
        )

    def test_from_reports_counts_passed(self):
        reports = [
            self._make_report("a", True),
            self._make_report("b", True),
            self._make_report("c", False),
        ]
        result = BenchmarkSuiteResult.from_reports(reports)
        assert result.passed_count == 2
        assert result.failed_count == 1

    def test_from_reports_all_passed_true(self):
        reports = [self._make_report(f"bench_{i}", True) for i in range(3)]
        result = BenchmarkSuiteResult.from_reports(reports)
        assert result.all_passed is True

    def test_from_reports_all_passed_false_when_any_fails(self):
        reports = [
            self._make_report("a", True),
            self._make_report("b", False),
        ]
        result = BenchmarkSuiteResult.from_reports(reports)
        assert result.all_passed is False

    def test_from_reports_total_count(self):
        reports = [self._make_report(f"b{i}", True) for i in range(4)]
        result = BenchmarkSuiteResult.from_reports(reports)
        assert result.total_benchmarks == 4

    def test_from_reports_empty_list(self):
        result = BenchmarkSuiteResult.from_reports([])
        assert result.total_benchmarks == 0
        assert result.all_passed is True  # vacuously true


# ---------------------------------------------------------------------------
# BenchmarkSuite individual benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarkSuiteEarlyTermination:
    @pytest.mark.asyncio
    async def test_returns_benchmark_report(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_early_termination()
        assert isinstance(report, BenchmarkReport)

    @pytest.mark.asyncio
    async def test_benchmark_name_correct(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_early_termination()
        assert report.benchmark_name == "early_termination"

    @pytest.mark.asyncio
    async def test_shows_positive_improvement(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_early_termination()
        assert report.improvement_percent > 0

    @pytest.mark.asyncio
    async def test_optimized_faster_than_baseline(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_early_termination()
        assert report.optimized_latency_ms < report.baseline_latency_ms

    @pytest.mark.asyncio
    async def test_passed_with_default_threshold(self):
        suite = BenchmarkSuite(improvement_threshold_percent=10.0)
        report = await suite.benchmark_early_termination()
        assert report.passed is True

    @pytest.mark.asyncio
    async def test_details_contains_expected_keys(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_early_termination()
        assert "description" in report.details
        assert "checks_skipped" in report.details


class TestBenchmarkSuiteParallel:
    @pytest.mark.asyncio
    async def test_returns_benchmark_report(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_parallel_vs_sequential()
        assert isinstance(report, BenchmarkReport)

    @pytest.mark.asyncio
    async def test_benchmark_name_correct(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_parallel_vs_sequential()
        assert report.benchmark_name == "parallel_vs_sequential"

    @pytest.mark.asyncio
    async def test_shows_positive_improvement(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_parallel_vs_sequential()
        assert report.improvement_percent > 0

    @pytest.mark.asyncio
    async def test_optimized_faster_than_baseline(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_parallel_vs_sequential()
        assert report.optimized_latency_ms < report.baseline_latency_ms

    @pytest.mark.asyncio
    async def test_passed_with_default_threshold(self):
        suite = BenchmarkSuite(improvement_threshold_percent=10.0)
        report = await suite.benchmark_parallel_vs_sequential()
        assert report.passed is True


class TestBenchmarkSuiteSeverityPriorities:
    @pytest.mark.asyncio
    async def test_returns_benchmark_report(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_severity_based_priorities()
        assert isinstance(report, BenchmarkReport)

    @pytest.mark.asyncio
    async def test_benchmark_name_correct(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_severity_based_priorities()
        assert report.benchmark_name == "severity_based_priorities"

    @pytest.mark.asyncio
    async def test_shows_positive_improvement(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_severity_based_priorities()
        assert report.improvement_percent > 0

    @pytest.mark.asyncio
    async def test_passed_with_default_threshold(self):
        suite = BenchmarkSuite(improvement_threshold_percent=10.0)
        report = await suite.benchmark_severity_based_priorities()
        assert report.passed is True


class TestBenchmarkSuiteAdaptiveTimeout:
    @pytest.mark.asyncio
    async def test_returns_benchmark_report(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_adaptive_timeout()
        assert isinstance(report, BenchmarkReport)

    @pytest.mark.asyncio
    async def test_benchmark_name_correct(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_adaptive_timeout()
        assert report.benchmark_name == "adaptive_timeout"

    @pytest.mark.asyncio
    async def test_passed_with_default_threshold(self):
        suite = BenchmarkSuite(improvement_threshold_percent=10.0)
        report = await suite.benchmark_adaptive_timeout()
        assert report.passed is True

    @pytest.mark.asyncio
    async def test_details_contain_waste_metrics(self):
        suite = BenchmarkSuite()
        report = await suite.benchmark_adaptive_timeout()
        assert "baseline_waste_ms" in report.details
        assert "optimized_waste_ms" in report.details
        assert report.details["optimized_waste_ms"] < report.details["baseline_waste_ms"]


# ---------------------------------------------------------------------------
# BenchmarkSuite.run_all tests
# ---------------------------------------------------------------------------

class TestBenchmarkSuiteRunAll:
    @pytest.mark.asyncio
    async def test_run_all_returns_suite_result(self):
        suite = BenchmarkSuite()
        result = await suite.run_all()
        assert isinstance(result, BenchmarkSuiteResult)

    @pytest.mark.asyncio
    async def test_run_all_returns_four_reports(self):
        suite = BenchmarkSuite()
        result = await suite.run_all()
        assert result.total_benchmarks == 4

    @pytest.mark.asyncio
    async def test_run_all_report_names(self):
        suite = BenchmarkSuite()
        result = await suite.run_all()
        names = {r.benchmark_name for r in result.reports}
        assert "early_termination" in names
        assert "parallel_vs_sequential" in names
        assert "severity_based_priorities" in names
        assert "adaptive_timeout" in names

    @pytest.mark.asyncio
    async def test_run_all_all_passed(self):
        suite = BenchmarkSuite(improvement_threshold_percent=10.0)
        result = await suite.run_all()
        assert result.all_passed is True

    @pytest.mark.asyncio
    async def test_run_all_high_threshold_fails(self):
        """With an impossibly high threshold, benchmarks should fail."""
        suite = BenchmarkSuite(improvement_threshold_percent=99.9)
        result = await suite.run_all()
        # At least some benchmarks should fail with a 99.9% threshold
        assert result.failed_count > 0


# ---------------------------------------------------------------------------
# _improvement_percent utility tests
# ---------------------------------------------------------------------------

class TestImprovementPercent:
    def test_50_percent_improvement(self):
        assert abs(_improvement_percent(100.0, 50.0) - 50.0) < 0.01

    def test_zero_improvement(self):
        assert _improvement_percent(100.0, 100.0) == 0.0

    def test_negative_improvement(self):
        assert _improvement_percent(50.0, 100.0) < 0.0

    def test_zero_baseline_returns_zero(self):
        assert _improvement_percent(0.0, 50.0) == 0.0

    def test_full_improvement(self):
        assert abs(_improvement_percent(100.0, 0.0) - 100.0) < 0.01
