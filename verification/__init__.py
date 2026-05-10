"""
Verification Agent với LangGraph StateGraph

Hệ thống kiểm duyệt tự động cho Sales Research Agent drafts sử dụng:
- Binary verification logic (PASS/FAIL) thay vì scoring 0-10
- LangGraph StateGraph cho workflow orchestration
- Pydantic models cho structured data validation
- Async optimization với early termination
- Self-correction loop với structured issue feedback

Core Components:
- VerificationAgent: Binary verification với 3 criteria (Price, Policy, Relevance)
- StateGraphWorkflow: LangGraph orchestration với conditional routing
- SelfCorrectionNode: Structured issue-based feedback generation
- Pydantic Models: Type-safe data validation và serialization
"""

from .agent import VerificationAgent
from .workflow import VerificationWorkflow
from .models import (
    WorkflowState,
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    ExecutionStep,
    WorkflowMetrics
)
from .config import VerificationConfig

__version__ = "1.0.0"
__all__ = [
    "VerificationAgent",
    "VerificationWorkflow", 
    "WorkflowState",
    "VerificationResult",
    "RubricCriteria",
    "PriceIssue",
    "PolicyIssue", 
    "RelevanceIssue",
    "ExecutionStep",
    "WorkflowMetrics",
    "VerificationConfig"
]