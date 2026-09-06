"""
Task 4.1.4 - Pydantic State Serialization Compatibility Tests

Verifies that all Pydantic models in the verification module support:
- Round-trip serialization: model → dict → model
- JSON serialization: model → JSON string → model
- Nested model serialization
- Enum field serialization
- Optional field serialization (None values)
- datetime field serialization (ISO format)
- List field serialization

Validates: Requirements 7.4 (State Immutability - Pydantic serialization),
           Requirements 2.1, 2.2 (StateGraph serializable for persistence)
"""

import json
import pytest
from datetime import datetime

from backend.verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
    FailedCriterion,
    FeedbackReport,
)
from backend.verification.models.execution import (
    ExecutionStep,
    WorkflowMetrics,
    ExecutionStatus,
)
from backend.verification.models.state import (
    WorkflowStateValidator,
    WorkflowConfig,
    create_initial_workflow_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_issue(severity: IssueSeverity = IssueSeverity.MINOR) -> PriceIssue:
    return PriceIssue(
        product_name="iPhone 15 Pro",
        product_sku="IP15P-256",
        mentioned_price="35,000,000 VND",
        actual_price="34,990,000 VND",
        deviation_percent=0.03,
        severity=severity,
        explanation="Giá sai lệch nhỏ",
        correction_suggestion="Cập nhật giá chính xác",
    )


def _make_policy_issue(severity: IssueSeverity = IssueSeverity.MAJOR) -> PolicyIssue:
    return PolicyIssue(
        mentioned_policy="Bảo hành 2 năm miễn phí",
        policy_type="warranty",
        is_fabricated=False,
        is_inaccurate=True,
        severity=severity,
        explanation="Bảo hành chỉ 1 năm",
        source_document="warranty_policy.pdf",
    )


def _make_relevance_issue(severity: IssueSeverity = IssueSeverity.MAJOR) -> RelevanceIssue:
    return RelevanceIssue(
        objection_intent="So sánh giá iPhone vs Samsung",
        detected_intents=["price_comparison", "feature_comparison"],
        response_coverage=0.5,
        missing_aspects=["Camera comparison", "Performance"],
        off_topic_content=["Apple history"],
        empathy_score=0.4,
        severity=severity,
        explanation="Response chỉ cover 50% objection",
    )


def _make_rubric_criteria(all_pass: bool = True) -> RubricCriteria:
    if all_pass:
        return RubricCriteria(
            price_accuracy_pass=True,
            policy_authenticity_pass=True,
            topic_relevance_pass=True,
        )
    return RubricCriteria(
        price_accuracy_pass=False,
        policy_authenticity_pass=False,
        topic_relevance_pass=True,
        price_issues=[_make_price_issue(IssueSeverity.CRITICAL)],
        policy_issues=[_make_policy_issue(IssueSeverity.MAJOR)],
    )


def _make_verification_result(approved: bool = True) -> VerificationResult:
    return VerificationResult(
        criteria=_make_rubric_criteria(all_pass=approved),
        verification_reasoning="Verification completed successfully for testing",
        execution_time_seconds=2.5,
        llm_tokens_used=1200,
    )


def _make_execution_step(status: ExecutionStatus = ExecutionStatus.SUCCESS) -> ExecutionStep:
    return ExecutionStep(
        node_name="verification",
        execution_time=1.5,
        status=status,
        input_summary="draft response input",
        output_summary="verification result output",
        metrics={"issues_found": 0, "cache_hits": 2},
        memory_usage_mb=45.0,
        cpu_usage_percent=12.5,
    )


def _make_workflow_metrics() -> WorkflowMetrics:
    return WorkflowMetrics(
        total_execution_time=8.5,
        total_retries=1,
        total_steps=3,
        successful_steps=3,
        failed_steps=0,
        timeout_steps=0,
        nodes_executed=["research", "verification", "correction"],
        node_execution_counts={"research": 2, "verification": 1},
        node_average_times={"research": 3.2, "verification": 2.5},
        critical_issues_found=0,
        major_issues_found=1,
        minor_issues_found=1,
        total_issues_found=2,
        llm_tokens_used=3500,
        llm_tokens_input=2100,
        llm_tokens_output=1400,
        cost_estimate=0.0175,
        cache_hits=5,
        cache_misses=2,
        db_queries_count=8,
        external_api_calls=3,
        verification_pass_rate=0.67,
        escalation_rate=0.0,
    )


# ---------------------------------------------------------------------------
# PriceIssue serialization
# ---------------------------------------------------------------------------

class TestPriceIssueSerialization:
    """Round-trip and JSON serialization for PriceIssue"""

    def test_round_trip_dict(self):
        issue = _make_price_issue()
        d = issue.model_dump()
        issue2 = PriceIssue.model_validate(d)
        assert issue2.product_name == issue.product_name
        assert issue2.severity == issue.severity
        assert issue2.deviation_percent == issue.deviation_percent

    def test_round_trip_json(self):
        issue = _make_price_issue(IssueSeverity.CRITICAL)
        j = issue.model_dump_json()
        issue2 = PriceIssue.model_validate_json(j)
        assert issue2.severity == IssueSeverity.CRITICAL
        assert issue2.product_sku == issue.product_sku

    def test_enum_serializes_to_string(self):
        issue = _make_price_issue(IssueSeverity.MAJOR)
        d = issue.model_dump(mode="json")
        assert d["severity"] == "major"

    def test_optional_fields_none(self):
        issue = PriceIssue(
            product_name="Test Product",
            severity=IssueSeverity.MINOR,
            explanation="Test explanation",
        )
        d = issue.model_dump()
        assert d["product_sku"] is None
        assert d["mentioned_price"] is None
        assert d["actual_price"] is None
        assert d["deviation_percent"] is None
        # Round-trip preserves None
        issue2 = PriceIssue.model_validate(d)
        assert issue2.product_sku is None

    def test_json_compatible_dict(self):
        issue = _make_price_issue()
        d = issue.model_dump(mode="json")
        # Should be JSON-serializable without errors
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# PolicyIssue serialization
# ---------------------------------------------------------------------------

class TestPolicyIssueSerialization:
    """Round-trip and JSON serialization for PolicyIssue"""

    def test_round_trip_dict(self):
        issue = _make_policy_issue()
        d = issue.model_dump()
        issue2 = PolicyIssue.model_validate(d)
        assert issue2.mentioned_policy == issue.mentioned_policy
        assert issue2.policy_type == issue.policy_type
        assert issue2.severity == issue.severity

    def test_round_trip_json(self):
        issue = _make_policy_issue(IssueSeverity.CRITICAL)
        j = issue.model_dump_json()
        issue2 = PolicyIssue.model_validate_json(j)
        assert issue2.severity == IssueSeverity.CRITICAL
        assert issue2.is_fabricated == issue.is_fabricated

    def test_enum_serializes_to_string(self):
        issue = _make_policy_issue(IssueSeverity.CRITICAL)
        d = issue.model_dump(mode="json")
        assert d["severity"] == "critical"

    def test_optional_fields_none(self):
        issue = PolicyIssue(
            mentioned_policy="Test policy",
            policy_type="warranty",
            is_fabricated=True,
            is_inaccurate=False,
            severity=IssueSeverity.CRITICAL,
            explanation="Fabricated policy",
        )
        d = issue.model_dump()
        assert d["correct_policy"] is None
        assert d["source_document"] is None
        issue2 = PolicyIssue.model_validate(d)
        assert issue2.correct_policy is None


# ---------------------------------------------------------------------------
# RelevanceIssue serialization
# ---------------------------------------------------------------------------

class TestRelevanceIssueSerialization:
    """Round-trip and JSON serialization for RelevanceIssue"""

    def test_round_trip_dict(self):
        issue = _make_relevance_issue()
        d = issue.model_dump()
        issue2 = RelevanceIssue.model_validate(d)
        assert issue2.objection_intent == issue.objection_intent
        assert issue2.response_coverage == issue.response_coverage
        assert issue2.missing_aspects == issue.missing_aspects

    def test_round_trip_json(self):
        issue = _make_relevance_issue(IssueSeverity.MINOR)
        j = issue.model_dump_json()
        issue2 = RelevanceIssue.model_validate_json(j)
        assert issue2.severity == IssueSeverity.MINOR
        assert issue2.detected_intents == issue.detected_intents

    def test_list_fields_serialization(self):
        issue = _make_relevance_issue()
        d = issue.model_dump()
        assert isinstance(d["missing_aspects"], list)
        assert isinstance(d["off_topic_content"], list)
        assert isinstance(d["detected_intents"], list)

    def test_optional_empathy_score_none(self):
        issue = RelevanceIssue(
            objection_intent="Test intent",
            response_coverage=0.6,
            severity=IssueSeverity.MINOR,
            explanation="Test explanation",
        )
        d = issue.model_dump()
        assert d["empathy_score"] is None
        issue2 = RelevanceIssue.model_validate(d)
        assert issue2.empathy_score is None


# ---------------------------------------------------------------------------
# RubricCriteria serialization
# ---------------------------------------------------------------------------

class TestRubricCriteriaSerialization:
    """Round-trip and JSON serialization for RubricCriteria"""

    def test_round_trip_dict_all_pass(self):
        criteria = _make_rubric_criteria(all_pass=True)
        d = criteria.model_dump()
        criteria2 = RubricCriteria.model_validate(d)
        assert criteria2.overall_pass is True
        assert criteria2.critical_issues_count == 0

    def test_round_trip_dict_with_issues(self):
        criteria = _make_rubric_criteria(all_pass=False)
        d = criteria.model_dump()
        criteria2 = RubricCriteria.model_validate(d)
        assert criteria2.overall_pass is False
        assert len(criteria2.price_issues) == 1
        assert len(criteria2.policy_issues) == 1
        assert criteria2.price_issues[0].severity == IssueSeverity.CRITICAL

    def test_round_trip_json(self):
        criteria = _make_rubric_criteria(all_pass=False)
        j = criteria.model_dump_json()
        criteria2 = RubricCriteria.model_validate_json(j)
        assert criteria2.overall_pass is False
        assert criteria2.critical_issues_count == criteria.critical_issues_count

    def test_nested_issue_list_serialization(self):
        criteria = RubricCriteria(
            price_accuracy_pass=False,
            policy_authenticity_pass=False,
            topic_relevance_pass=False,
            price_issues=[
                _make_price_issue(IssueSeverity.CRITICAL),
                _make_price_issue(IssueSeverity.MINOR),
            ],
            policy_issues=[_make_policy_issue(IssueSeverity.MAJOR)],
            relevance_issues=[_make_relevance_issue(IssueSeverity.MINOR)],
        )
        d = criteria.model_dump()
        criteria2 = RubricCriteria.model_validate(d)
        assert len(criteria2.price_issues) == 2
        assert len(criteria2.policy_issues) == 1
        assert len(criteria2.relevance_issues) == 1

    def test_empty_issue_lists_serialization(self):
        criteria = _make_rubric_criteria(all_pass=True)
        d = criteria.model_dump()
        assert d["price_issues"] == []
        assert d["policy_issues"] == []
        assert d["relevance_issues"] == []
        criteria2 = RubricCriteria.model_validate(d)
        assert criteria2.price_issues == []


# ---------------------------------------------------------------------------
# VerificationResult serialization
# ---------------------------------------------------------------------------

class TestVerificationResultSerialization:
    """Round-trip and JSON serialization for VerificationResult"""

    def test_round_trip_dict_approved(self):
        result = _make_verification_result(approved=True)
        d = result.model_dump()
        result2 = VerificationResult.model_validate(d)
        assert result2.is_approved is True
        assert result2.verification_reasoning == result.verification_reasoning

    def test_round_trip_dict_failed(self):
        result = _make_verification_result(approved=False)
        d = result.model_dump()
        result2 = VerificationResult.model_validate(d)
        assert result2.is_approved is False
        assert result2.requires_correction is True

    def test_round_trip_json(self):
        result = _make_verification_result(approved=False)
        j = result.model_dump_json()
        result2 = VerificationResult.model_validate_json(j)
        assert result2.is_approved is False
        assert result2.criteria.critical_issues_count == result.criteria.critical_issues_count

    def test_datetime_field_serialization(self):
        result = _make_verification_result()
        # model_dump returns datetime object
        d = result.model_dump()
        assert isinstance(d["timestamp"], datetime)
        # model_dump(mode='json') returns ISO string
        d_json = result.model_dump(mode="json")
        assert isinstance(d_json["timestamp"], str)
        # Round-trip from JSON-mode dict
        result2 = VerificationResult.model_validate(d_json)
        assert isinstance(result2.timestamp, datetime)

    def test_json_string_round_trip_preserves_timestamp(self):
        result = _make_verification_result()
        j = result.model_dump_json()
        result2 = VerificationResult.model_validate_json(j)
        # Timestamps should be equal (within microsecond precision)
        assert result2.timestamp == result.timestamp

    def test_nested_criteria_preserved_in_round_trip(self):
        criteria = RubricCriteria(
            price_accuracy_pass=False,
            policy_authenticity_pass=True,
            topic_relevance_pass=True,
            price_issues=[_make_price_issue(IssueSeverity.CRITICAL)],
        )
        result = VerificationResult(
            criteria=criteria,
            verification_reasoning="Nested model test with sufficient detail",
            execution_time_seconds=1.0,
            llm_tokens_used=500,
        )
        j = result.model_dump_json()
        result2 = VerificationResult.model_validate_json(j)
        assert result2.criteria.price_issues[0].severity == IssueSeverity.CRITICAL
        assert result2.criteria.price_issues[0].product_name == "iPhone 15 Pro"

    def test_json_compatible_dict_is_json_serializable(self):
        result = _make_verification_result(approved=False)
        d = result.model_dump(mode="json")
        # Should not raise
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["criteria"]["price_accuracy_pass"] is False


# ---------------------------------------------------------------------------
# ExecutionStep serialization
# ---------------------------------------------------------------------------

class TestExecutionStepSerialization:
    """Round-trip and JSON serialization for ExecutionStep"""

    def test_round_trip_dict(self):
        step = _make_execution_step()
        d = step.model_dump()
        step2 = ExecutionStep.model_validate(d)
        assert step2.node_name == step.node_name
        assert step2.status == ExecutionStatus.SUCCESS
        assert step2.execution_time == step.execution_time

    def test_round_trip_json(self):
        step = _make_execution_step(ExecutionStatus.FAILED)
        j = step.model_dump_json()
        step2 = ExecutionStep.model_validate_json(j)
        assert step2.status == ExecutionStatus.FAILED

    def test_enum_status_serializes_to_string(self):
        step = _make_execution_step(ExecutionStatus.RETRY)
        d = step.model_dump(mode="json")
        assert d["status"] == "retry"

    def test_optional_error_fields_none(self):
        step = _make_execution_step()
        d = step.model_dump()
        assert d["error_details"] is None
        assert d["error_type"] is None
        step2 = ExecutionStep.model_validate(d)
        assert step2.error_details is None

    def test_metrics_dict_preserved(self):
        step = _make_execution_step()
        d = step.model_dump()
        assert d["metrics"]["issues_found"] == 0
        assert d["metrics"]["cache_hits"] == 2
        step2 = ExecutionStep.model_validate(d)
        assert step2.metrics["issues_found"] == 0

    def test_optional_resource_fields(self):
        step = _make_execution_step()
        d = step.model_dump()
        assert d["memory_usage_mb"] == 45.0
        assert d["cpu_usage_percent"] == 12.5
        step2 = ExecutionStep.model_validate(d)
        assert step2.memory_usage_mb == 45.0


# ---------------------------------------------------------------------------
# WorkflowMetrics serialization
# ---------------------------------------------------------------------------

class TestWorkflowMetricsSerialization:
    """Round-trip and JSON serialization for WorkflowMetrics"""

    def test_round_trip_dict(self):
        metrics = _make_workflow_metrics()
        d = metrics.model_dump()
        metrics2 = WorkflowMetrics.model_validate(d)
        assert metrics2.total_execution_time == metrics.total_execution_time
        assert metrics2.success_rate == metrics.success_rate

    def test_round_trip_json(self):
        metrics = _make_workflow_metrics()
        j = metrics.model_dump_json()
        metrics2 = WorkflowMetrics.model_validate_json(j)
        assert metrics2.total_retries == metrics.total_retries
        assert metrics2.nodes_executed == metrics.nodes_executed

    def test_list_fields_preserved(self):
        metrics = _make_workflow_metrics()
        d = metrics.model_dump()
        assert d["nodes_executed"] == ["research", "verification", "correction"]
        metrics2 = WorkflowMetrics.model_validate(d)
        assert metrics2.nodes_executed == metrics.nodes_executed

    def test_dict_fields_preserved(self):
        metrics = _make_workflow_metrics()
        d = metrics.model_dump()
        assert d["node_execution_counts"]["research"] == 2
        metrics2 = WorkflowMetrics.model_validate(d)
        assert metrics2.node_execution_counts["research"] == 2

    def test_optional_resource_fields_none(self):
        metrics = _make_workflow_metrics()
        d = metrics.model_dump()
        # peak_memory_usage_mb and average_cpu_usage_percent are None by default
        assert d["peak_memory_usage_mb"] is None
        assert d["network_latency_ms"] is None
        metrics2 = WorkflowMetrics.model_validate(d)
        assert metrics2.peak_memory_usage_mb is None


# ---------------------------------------------------------------------------
# WorkflowConfig serialization
# ---------------------------------------------------------------------------

class TestWorkflowConfigSerialization:
    """Round-trip and JSON serialization for WorkflowConfig"""

    def test_round_trip_dict(self):
        config = WorkflowConfig()
        d = config.model_dump()
        config2 = WorkflowConfig.model_validate(d)
        assert config2.max_retries == config.max_retries
        assert config2.price_tolerance_percent == config.price_tolerance_percent

    def test_round_trip_json(self):
        config = WorkflowConfig(max_retries=5, price_tolerance_percent=2.0)
        j = config.model_dump_json()
        config2 = WorkflowConfig.model_validate_json(j)
        assert config2.max_retries == 5
        assert config2.price_tolerance_percent == 2.0

    def test_verification_weights_preserved(self):
        config = WorkflowConfig()
        d = config.model_dump()
        assert d["verification_weights"]["price_accuracy"] == pytest.approx(0.4)
        config2 = WorkflowConfig.model_validate(d)
        assert config2.verification_weights["price_accuracy"] == pytest.approx(0.4)

    def test_literal_field_serialization(self):
        config = WorkflowConfig(log_level="DEBUG")
        d = config.model_dump(mode="json")
        assert d["log_level"] == "DEBUG"
        config2 = WorkflowConfig.model_validate(d)
        assert config2.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# WorkflowStateValidator serialization
# ---------------------------------------------------------------------------

class TestWorkflowStateValidatorSerialization:
    """Round-trip and JSON serialization for WorkflowStateValidator"""

    def _make_state(self, **overrides) -> WorkflowStateValidator:
        base = create_initial_workflow_state(
            "iPhone quá đắt, tại sao tôi nên mua? Đây là câu hỏi dài hơn 10 ký tự."
        )
        base.update(overrides)
        return WorkflowStateValidator(**base)

    def test_round_trip_dict_initial_state(self):
        state = self._make_state()
        d = state.model_dump()
        state2 = WorkflowStateValidator.model_validate(d)
        assert state2.objection_text == state.objection_text
        assert state2.workflow_status == "initialized"
        assert state2.retry_count == 0

    def test_round_trip_json_initial_state(self):
        state = self._make_state()
        j = state.model_dump_json()
        state2 = WorkflowStateValidator.model_validate_json(j)
        assert state2.workflow_id == state.workflow_id
        assert state2.correlation_id == state.correlation_id

    def test_optional_verification_result_none(self):
        state = self._make_state()
        d = state.model_dump()
        assert d["verification_result"] is None
        state2 = WorkflowStateValidator.model_validate(d)
        assert state2.verification_result is None

    def test_nested_verification_result_round_trip(self):
        vr = _make_verification_result(approved=False)
        state = self._make_state(verification_result=vr)
        d = state.model_dump()
        state2 = WorkflowStateValidator.model_validate(d)
        assert state2.verification_result is not None
        assert state2.verification_result.is_approved is False
        assert state2.verification_result.criteria.critical_issues_count == 1

    def test_nested_verification_result_json_round_trip(self):
        vr = _make_verification_result(approved=False)
        state = self._make_state(verification_result=vr)
        j = state.model_dump_json()
        state2 = WorkflowStateValidator.model_validate_json(j)
        assert state2.verification_result is not None
        assert state2.verification_result.criteria.price_issues[0].severity == IssueSeverity.CRITICAL

    def test_execution_log_list_round_trip(self):
        step = _make_execution_step()
        state = self._make_state(execution_log=[step])
        d = state.model_dump()
        state2 = WorkflowStateValidator.model_validate(d)
        assert len(state2.execution_log) == 1
        assert state2.execution_log[0].node_name == "verification"
        assert state2.execution_log[0].status == ExecutionStatus.SUCCESS

    def test_execution_log_json_round_trip(self):
        step = _make_execution_step(ExecutionStatus.FAILED)
        state = self._make_state(execution_log=[step])
        j = state.model_dump_json()
        state2 = WorkflowStateValidator.model_validate_json(j)
        assert state2.execution_log[0].status == ExecutionStatus.FAILED

    def test_resource_usage_dict_preserved(self):
        state = self._make_state()
        d = state.model_dump()
        assert d["resource_usage"]["llm_tokens_total"] == 0
        assert d["resource_usage"]["cache_hits"] == 0
        state2 = WorkflowStateValidator.model_validate(d)
        assert state2.resource_usage["llm_tokens_total"] == 0

    def test_optional_end_time_none(self):
        state = self._make_state()
        d = state.model_dump()
        assert d["end_time"] is None
        state2 = WorkflowStateValidator.model_validate(d)
        assert state2.end_time is None

    def test_workflow_status_literal_preserved(self):
        state = self._make_state()
        d = state.model_dump(mode="json")
        assert d["workflow_status"] == "initialized"
        state2 = WorkflowStateValidator.model_validate(d)
        assert state2.workflow_status == "initialized"


# ---------------------------------------------------------------------------
# Full nested round-trip: WorkflowState → VerificationResult → RubricCriteria
# ---------------------------------------------------------------------------

class TestFullNestedSerialization:
    """End-to-end nested serialization tests"""

    def test_complete_workflow_state_round_trip(self):
        """WorkflowState with VerificationResult containing all issue types"""
        criteria = RubricCriteria(
            price_accuracy_pass=False,
            policy_authenticity_pass=False,
            topic_relevance_pass=False,
            price_issues=[
                _make_price_issue(IssueSeverity.CRITICAL),
                _make_price_issue(IssueSeverity.MINOR),
            ],
            policy_issues=[_make_policy_issue(IssueSeverity.MAJOR)],
            relevance_issues=[_make_relevance_issue(IssueSeverity.MINOR)],
        )
        vr = VerificationResult(
            criteria=criteria,
            verification_reasoning="All three criteria failed in this test scenario",
            execution_time_seconds=3.0,
            llm_tokens_used=2000,
        )
        steps = [
            _make_execution_step(ExecutionStatus.SUCCESS),
            _make_execution_step(ExecutionStatus.FAILED),
        ]

        base = create_initial_workflow_state(
            "iPhone quá đắt, tại sao tôi nên mua? Đây là câu hỏi dài hơn 10 ký tự."
        )
        base["verification_result"] = vr
        base["execution_log"] = steps
        base["retry_count"] = 1

        state = WorkflowStateValidator(**base)

        # Dict round-trip
        d = state.model_dump()
        state2 = WorkflowStateValidator.model_validate(d)
        assert state2.verification_result.criteria.critical_issues_count == 1
        assert len(state2.verification_result.criteria.price_issues) == 2
        assert len(state2.execution_log) == 2
        assert state2.retry_count == 1

        # JSON round-trip
        j = state.model_dump_json()
        state3 = WorkflowStateValidator.model_validate_json(j)
        assert state3.verification_result.criteria.policy_issues[0].severity == IssueSeverity.MAJOR
        assert state3.execution_log[1].status == ExecutionStatus.FAILED

    def test_json_compatible_dict_fully_serializable(self):
        """Ensure model_dump(mode='json') produces a fully JSON-serializable dict"""
        vr = _make_verification_result(approved=False)
        base = create_initial_workflow_state(
            "iPhone quá đắt, tại sao tôi nên mua? Đây là câu hỏi dài hơn 10 ký tự."
        )
        base["verification_result"] = vr
        base["execution_log"] = [_make_execution_step()]
        state = WorkflowStateValidator(**base)

        d = state.model_dump(mode="json")
        # Should not raise
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["workflow_status"] == "initialized"
        assert parsed["verification_result"]["criteria"]["price_accuracy_pass"] is False

    def test_severity_enum_values_in_nested_json(self):
        """Enum values serialize to their string values in JSON mode"""
        criteria = RubricCriteria(
            price_accuracy_pass=False,
            policy_authenticity_pass=False,
            topic_relevance_pass=True,
            price_issues=[_make_price_issue(IssueSeverity.CRITICAL)],
            policy_issues=[_make_policy_issue(IssueSeverity.MINOR)],
        )
        d = criteria.model_dump(mode="json")
        assert d["price_issues"][0]["severity"] == "critical"
        assert d["policy_issues"][0]["severity"] == "minor"

    def test_all_severity_levels_round_trip(self):
        """All three severity levels serialize and deserialize correctly"""
        for severity in [IssueSeverity.CRITICAL, IssueSeverity.MAJOR, IssueSeverity.MINOR]:
            issue = _make_price_issue(severity)
            j = issue.model_dump_json()
            issue2 = PriceIssue.model_validate_json(j)
            assert issue2.severity == severity


# ---------------------------------------------------------------------------
# create_initial_workflow_state utility
# ---------------------------------------------------------------------------

class TestCreateInitialWorkflowState:
    """Verify create_initial_workflow_state produces serializable output"""

    def test_returns_serializable_dict(self):
        state_dict = create_initial_workflow_state(
            "iPhone quá đắt, tại sao tôi nên mua? Đây là câu hỏi dài hơn 10 ký tự."
        )
        # Should be a plain dict (model_dump output)
        assert isinstance(state_dict, dict)
        # Should be JSON-serializable
        json_str = json.dumps(state_dict, default=str)
        assert isinstance(json_str, str)

    def test_state_dict_validates_back_to_model(self):
        state_dict = create_initial_workflow_state(
            "iPhone quá đắt, tại sao tôi nên mua? Đây là câu hỏi dài hơn 10 ký tự."
        )
        state = WorkflowStateValidator(**state_dict)
        assert state.workflow_status == "initialized"
        assert state.retry_count == 0
        assert state.verification_result is None

    def test_config_snapshot_is_serializable(self):
        config = WorkflowConfig(max_retries=5)
        state_dict = create_initial_workflow_state(
            "iPhone quá đắt, tại sao tôi nên mua? Đây là câu hỏi dài hơn 10 ký tự.",
            config=config,
        )
        # config key should be a plain dict
        assert isinstance(state_dict["config"], dict)
        assert state_dict["config"]["max_retries"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
