"""
Pydantic Data Models cho Verification Agent

Binary verification system với structured issue tracking:
- WorkflowState: Shared state object cho LangGraph StateGraph
- VerificationResult: Binary PASS/FAIL với detailed issue breakdown
- Issue Models: Structured classification cho Price/Policy/Relevance issues
- Execution Tracking: Performance metrics và observability data
"""

from .state import WorkflowState
from .verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity,
    FeedbackReport,
    FailedCriterion,
)
from .execution import ExecutionStep, WorkflowMetrics, ExecutionStatus

__all__ = [
    "WorkflowState",
    "VerificationResult",
    "RubricCriteria",
    "PriceIssue",
    "PolicyIssue",
    "RelevanceIssue",
    "IssueSeverity",
    "FeedbackReport",
    "FailedCriterion",
    "ExecutionStep",
    "WorkflowMetrics",
    "ExecutionStatus"
]