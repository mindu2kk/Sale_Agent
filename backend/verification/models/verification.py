"""
Binary Verification Models với Pydantic

Structured models cho binary PASS/FAIL verification:
- VerificationResult: Overall binary decision với issue breakdown
- RubricCriteria: Binary checks cho Price/Policy/Relevance
- Issue Models: Detailed classification cho each failure type
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import List, Optional, Literal, Dict
from datetime import datetime
from enum import Enum


class IssueSeverity(str, Enum):
    """Issue severity levels cho structured classification"""
    CRITICAL = "critical"  # Immediate escalation required
    MAJOR = "major"       # Significant impact, needs correction
    MINOR = "minor"       # Low impact, optional correction


class PriceIssue(BaseModel):
    """
    Structured price accuracy issue với detailed context
    
    Supports Requirements 4: Price accuracy verification with detailed issue tracking
    """
    
    product_name: str = Field(description="Tên sản phẩm có vấn đề về giá")
    product_sku: Optional[str] = Field(default=None, description="SKU sản phẩm để cross-reference")
    mentioned_price: Optional[str] = Field(default=None, description="Giá được đề cập trong draft")
    actual_price: Optional[str] = Field(default=None, description="Giá chính xác từ DB")
    deviation_percent: Optional[float] = Field(default=None, description="Phần trăm sai lệch")
    currency: str = Field(default="VND", description="Đơn vị tiền tệ")
    severity: IssueSeverity = Field(description="Mức độ nghiêm trọng")
    explanation: str = Field(description="Chi tiết vấn đề và cách khắc phục")
    correction_suggestion: Optional[str] = Field(default=None, description="Gợi ý cụ thể để sửa lỗi")
    
    @field_validator('deviation_percent')
    @classmethod
    def validate_deviation(cls, v):
        if v is not None and v < 0:
            raise ValueError("Deviation percent must be non-negative")
        return v
    
    @property
    def is_critical_deviation(self) -> bool:
        """Check if price deviation is critical (>30%)"""
        return self.deviation_percent is not None and self.deviation_percent > 30.0
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "product_name": "iPhone 15 Pro Max",
            "mentioned_price": "35,000,000 VND",
            "actual_price": "34,990,000 VND",
            "deviation_percent": 0.03,
            "severity": "minor",
            "explanation": "Giá sai lệch nhỏ 0.03%, cần cập nhật chính xác"
        }
    })


class PolicyIssue(BaseModel):
    """
    Structured policy authenticity issue với verification context
    
    Supports Requirements 5: Policy authenticity verification with fabrication detection
    """
    
    mentioned_policy: str = Field(description="Chính sách được đề cập")
    policy_type: Literal["warranty", "return", "exchange", "service", "support"] = Field(
        description="Loại chính sách"
    )
    is_fabricated: bool = Field(description="Có phải chính sách bịa đặt không")
    is_inaccurate: bool = Field(description="Có phải chính sách không chính xác không")
    is_incomplete: bool = Field(default=False, description="Có phải chính sách thiếu thông tin không")
    correct_policy: Optional[str] = Field(default=None, description="Chính sách chính xác")
    severity: IssueSeverity = Field(description="Mức độ nghiêm trọng")
    explanation: str = Field(description="Chi tiết vấn đề và cách khắc phục")
    source_document: Optional[str] = Field(default=None, description="Tài liệu chính sách gốc")
    policy_section: Optional[str] = Field(default=None, description="Phần cụ thể trong tài liệu chính sách")
    correction_suggestion: Optional[str] = Field(default=None, description="Gợi ý cụ thể để sửa lỗi")
    
    @property
    def requires_immediate_escalation(self) -> bool:
        """Check if fabricated policy requires immediate escalation"""
        return self.is_fabricated and self.severity == IssueSeverity.CRITICAL
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "mentioned_policy": "Bảo hành 2 năm cho tất cả sản phẩm",
            "policy_type": "warranty",
            "is_fabricated": False,
            "is_inaccurate": True,
            "correct_policy": "Bảo hành 1 năm cho điện thoại, 2 năm cho laptop",
            "severity": "major",
            "explanation": "Chính sách bảo hành không chính xác, cần phân biệt theo loại sản phẩm",
            "source_document": "warranty_policy_2024.pdf"
        }
    })


class RelevanceIssue(BaseModel):
    """
    Structured topic relevance issue với semantic analysis
    
    Supports Requirements 6: Topic relevance assessment with coverage analysis
    """
    
    objection_intent: str = Field(description="Ý định chính của objection")
    detected_intents: List[str] = Field(
        default_factory=list,
        description="Các ý định được phát hiện trong objection"
    )
    response_coverage: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Tỷ lệ coverage của response (0-1)"
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Các khía cạnh chưa được giải quyết"
    )
    off_topic_content: List[str] = Field(
        default_factory=list,
        description="Nội dung lệch chủ đề"
    )
    empathy_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Điểm empathy trong response (0-1)"
    )
    severity: IssueSeverity = Field(description="Mức độ nghiêm trọng")
    explanation: str = Field(description="Chi tiết vấn đề và cách khắc phục")
    correction_suggestion: Optional[str] = Field(default=None, description="Gợi ý cụ thể để cải thiện relevance")
    
    @property
    def is_severely_off_topic(self) -> bool:
        """Check if response is severely off-topic (<30% coverage)"""
        return self.response_coverage < 0.3
    
    @property
    def has_empathy_issues(self) -> bool:
        """Check if response lacks empathy"""
        return self.empathy_score is not None and self.empathy_score < 0.5
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "objection_intent": "So sánh giá iPhone vs Samsung",
            "response_coverage": 0.6,
            "missing_aspects": ["Tính năng camera", "Hiệu năng gaming"],
            "off_topic_content": ["Lịch sử Apple", "Thông tin không liên quan"],
            "severity": "major",
            "explanation": "Response chỉ cover 60% objection, thiếu so sánh chi tiết"
        }
    })


class RubricCriteria(BaseModel):
    """
    Binary verification criteria với structured issue tracking
    
    Thay thế scoring 0-10 bằng binary PASS/FAIL với detailed issues.
    """
    
    # Binary Pass/Fail Results
    price_accuracy_pass: bool = Field(description="Giá có chính xác không?")
    policy_authenticity_pass: bool = Field(description="Chính sách có xác thực không?")
    topic_relevance_pass: bool = Field(description="Response có đúng trọng tâm không?")
    
    # Detailed Issue Breakdown
    price_issues: List[PriceIssue] = Field(default_factory=list)
    policy_issues: List[PolicyIssue] = Field(default_factory=list)
    relevance_issues: List[RelevanceIssue] = Field(default_factory=list)
    
    # Overall Assessment - computed fields
    overall_pass: Optional[bool] = Field(default=None, description="Tổng thể có pass verification không?")
    critical_issues_count: Optional[int] = Field(default=None, ge=0, description="Số lượng critical issues")
    
    @model_validator(mode="after")
    def compute_derived_fields(self):
        if self.overall_pass is None:
            self.overall_pass = (
                self.price_accuracy_pass and
                self.policy_authenticity_pass and
                self.topic_relevance_pass
            )
        
        if self.critical_issues_count is None:
            count = 0
            count += sum(1 for issue in self.price_issues if issue.severity == IssueSeverity.CRITICAL)
            count += sum(1 for issue in self.policy_issues if issue.severity == IssueSeverity.CRITICAL)
            count += sum(1 for issue in self.relevance_issues if issue.severity == IssueSeverity.CRITICAL)
            self.critical_issues_count = count
        return self
    
    def get_escalation_priority(self) -> Literal["immediate", "high", "medium", "low"]:
        """
        Determine escalation priority based on issue severity and types
        
        Supports Requirements 8: Error handling with severity-based escalation
        """
        if self.critical_issues_count >= 3:
            return "immediate"
        elif self.critical_issues_count >= 1:
            # Check for fabricated policies - immediate escalation
            fabricated_policies = any(
                issue.is_fabricated for issue in self.policy_issues 
                if issue.severity == IssueSeverity.CRITICAL
            )
            if fabricated_policies:
                return "immediate"
            return "high"
        elif self.get_major_issues_count() >= 2:
            return "medium"
        else:
            return "low"
    
    def get_major_issues_count(self) -> int:
        """Count major severity issues across all categories"""
        count = 0
        count += sum(1 for issue in self.price_issues if issue.severity == IssueSeverity.MAJOR)
        count += sum(1 for issue in self.policy_issues if issue.severity == IssueSeverity.MAJOR)
        count += sum(1 for issue in self.relevance_issues if issue.severity == IssueSeverity.MAJOR)
        return count
    
    def get_detailed_issue_summary(self) -> dict:
        """Get comprehensive issue breakdown for detailed tracking"""
        return {
            "total_issues": len(self.price_issues) + len(self.policy_issues) + len(self.relevance_issues),
            "critical_count": self.critical_issues_count,
            "major_count": self.get_major_issues_count(),
            "minor_count": self.get_minor_issues_count(),
            "fabricated_policies": sum(1 for issue in self.policy_issues if issue.is_fabricated),
            "critical_price_deviations": sum(1 for issue in self.price_issues if issue.is_critical_deviation),
            "severely_off_topic": sum(1 for issue in self.relevance_issues if issue.is_severely_off_topic),
            "escalation_priority": self.get_escalation_priority()
        }
    
    def get_minor_issues_count(self) -> int:
        """Count minor severity issues across all categories"""
        count = 0
        count += sum(1 for issue in self.price_issues if issue.severity == IssueSeverity.MINOR)
        count += sum(1 for issue in self.policy_issues if issue.severity == IssueSeverity.MINOR)
        count += sum(1 for issue in self.relevance_issues if issue.severity == IssueSeverity.MINOR)
        return count
    
    def get_failure_summary(self) -> str:
        if self.overall_pass:
            return "✅ Verification PASSED - No issues detected"
        
        issues = []
        if not self.price_accuracy_pass:
            issues.append("❌ Price Accuracy")
        if not self.policy_authenticity_pass:
            issues.append("❌ Policy Authenticity")
        if not self.topic_relevance_pass:
            issues.append("❌ Topic Relevance")
        
        return f"🔄 Verification FAILED: {', '.join(issues)}"
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "price_accuracy_pass": False,
            "policy_authenticity_pass": True,
            "topic_relevance_pass": True,
            "price_issues": [
                {
                    "product_name": "iPhone 15",
                    "mentioned_price": "30,000,000 VND",
                    "actual_price": "29,990,000 VND",
                    "deviation_percent": 0.03,
                    "severity": "minor",
                    "explanation": "Giá sai lệch nhỏ, cần cập nhật chính xác"
                }
            ],
            "policy_issues": [],
            "relevance_issues": [],
            "overall_pass": False,
            "critical_issues_count": 0
        }
    })


class FailedCriterion(BaseModel):
    """
    A single failed verification criterion with human-readable explanation
    and actionable correction suggestions.
    """
    criterion_name: str = Field(description="Name of the failed criterion (e.g. 'Price Accuracy')")
    criterion_key: Literal["price_accuracy", "policy_authenticity", "topic_relevance"] = Field(
        description="Machine-readable criterion key"
    )
    explanation: str = Field(description="Human-readable explanation of why this criterion failed")
    correction_suggestions: List[str] = Field(
        default_factory=list,
        description="Specific actionable suggestions to fix this criterion"
    )
    severity: IssueSeverity = Field(description="Highest severity issue in this criterion")
    issue_count: int = Field(ge=0, description="Total number of issues in this criterion")
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "criterion_name": "Price Accuracy",
            "criterion_key": "price_accuracy",
            "explanation": "1 price issue detected: iPhone 15 price deviation 15.2%",
            "correction_suggestions": [
                "Update iPhone 15 price from '35,000,000 VND' to '29,990,000 VND' (SKU: IP15-128)"
            ],
            "severity": "major",
            "issue_count": 1
        }
    })


class FeedbackReport(BaseModel):
    """
    Structured feedback report for failed verifications.

    Produced by VerificationAgent.generate_structured_feedback() and consumed
    by SelfCorrectionNode to build the correction prompt for the Research Agent.

    The report is deterministic and template-based — no LLM calls required.
    """

    # Overall summary
    is_approved: bool = Field(description="Whether verification passed (False = corrections needed)")
    total_issues: int = Field(ge=0, description="Total number of issues across all criteria")
    critical_issues_count: int = Field(ge=0, description="Number of critical severity issues")
    escalation_priority: str = Field(description="Escalation priority: immediate, high, medium, low")

    # Severity-prioritized list of failed criteria (critical first)
    failed_criteria: List[FailedCriterion] = Field(
        default_factory=list,
        description="Failed criteria sorted by severity (critical → major → minor)"
    )

    # Ready-to-inject correction prompt for the Research Agent
    correction_prompt: str = Field(
        default="",
        description="Formatted correction prompt string ready to inject into Research Agent context"
    )
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "is_approved": False,
            "total_issues": 2,
            "critical_issues_count": 1,
            "escalation_priority": "high",
            "failed_criteria": [],
            "correction_prompt": "🔄 CORRECTION REQUIRED: ..."
        }
    })


class VerificationResult(BaseModel):
    """
    Complete binary verification result với structured feedback
    
    Kết quả verification với binary decision và detailed issue breakdown
    cho self-correction feedback generation.
    """
    
    criteria: RubricCriteria = Field(description="Binary verification criteria results")
    timestamp: datetime = Field(default_factory=datetime.now)
    verification_reasoning: str = Field(
        min_length=10, 
        description="Chi tiết lý do verification decision"
    )
    
    # Performance Metrics
    execution_time_seconds: float = Field(ge=0.0, description="Thời gian thực thi")
    llm_tokens_used: int = Field(ge=0, description="Số token LLM đã sử dụng")

    # Per-step latency tracking (Task 5.3.1)
    # Keys: step names (e.g. "price_check", "policy_check", "relevance_check")
    # Values: latency metrics dict from AsyncStepLatencyTracker.get_all_metrics()
    step_latencies: Optional[Dict] = Field(
        default=None,
        description="Per-step latency metrics for each async verification step"
    )

    # Critical issue detection flags (Task 5.4.1)
    has_critical_issues: bool = Field(
        default=False,
        description="True if any critical severity issue was detected"
    )
    immediate_termination: bool = Field(
        default=False,
        description="True if workflow should terminate immediately (bypass correction loop)"
    )
    
    # Binary Decision Properties
    @property
    def is_approved(self) -> bool:
        """Binary approval decision"""
        return self.criteria.overall_pass
    
    @property
    def requires_correction(self) -> bool:
        """Có cần correction không"""
        return not self.criteria.overall_pass
    
    @property
    def requires_escalation(self) -> bool:
        """
        Có cần escalate to human không (critical issues or fabricated policies)
        
        Enhanced escalation logic supporting Requirements 8: Error handling with severity-based escalation
        """
        return (
            self.criteria.critical_issues_count > 0 or
            self.criteria.get_escalation_priority() in ["immediate", "high"]
        )
    
    @property
    def escalation_priority(self) -> str:
        """Get escalation priority level"""
        return self.criteria.get_escalation_priority()
    
    @property
    def issue_summary(self) -> dict:
        """Get detailed issue summary for tracking and analytics"""
        return self.criteria.get_detailed_issue_summary()
    
    def get_correction_feedback(self) -> str:
        """
        Generate structured correction feedback cho Self-Correction Node
        
        Enhanced feedback with specific correction suggestions for each issue type
        """
        if self.is_approved:
            return "✅ No corrections needed - verification passed"
        
        feedback_parts = [
            "🔄 VERIFICATION FAILED - Corrections needed:",
            "",
            self.criteria.get_failure_summary(),
            f"📊 Issue Summary: {self.criteria.critical_issues_count} critical, {self.criteria.get_major_issues_count()} major, {self.criteria.get_minor_issues_count()} minor",
            f"⚠️ Escalation Priority: {self.escalation_priority.upper()}",
            ""
        ]
        
        # Price Issues Feedback with specific suggestions
        if self.criteria.price_issues:
            feedback_parts.append("💰 PRICE ACCURACY ISSUES:")
            for issue in self.criteria.price_issues:
                feedback_parts.append(f"  - {issue.explanation}")
                if issue.correction_suggestion:
                    feedback_parts.append(f"    💡 Suggestion: {issue.correction_suggestion}")
                if issue.product_sku:
                    feedback_parts.append(f"    🔍 Verify SKU: {issue.product_sku}")
            feedback_parts.append("")
        
        # Policy Issues Feedback with fabrication alerts
        if self.criteria.policy_issues:
            feedback_parts.append("📋 POLICY AUTHENTICITY ISSUES:")
            for issue in self.criteria.policy_issues:
                if issue.is_fabricated:
                    feedback_parts.append(f"  - 🚨 FABRICATED: {issue.explanation}")
                else:
                    feedback_parts.append(f"  - {issue.explanation}")
                if issue.correction_suggestion:
                    feedback_parts.append(f"    💡 Suggestion: {issue.correction_suggestion}")
                if issue.source_document:
                    feedback_parts.append(f"    📄 Reference: {issue.source_document}")
            feedback_parts.append("")
        
        # Relevance Issues Feedback with coverage analysis
        if self.criteria.relevance_issues:
            feedback_parts.append("🎯 TOPIC RELEVANCE ISSUES:")
            for issue in self.criteria.relevance_issues:
                feedback_parts.append(f"  - {issue.explanation}")
                if issue.correction_suggestion:
                    feedback_parts.append(f"    💡 Suggestion: {issue.correction_suggestion}")
                if issue.missing_aspects:
                    feedback_parts.append(f"    📝 Missing: {', '.join(issue.missing_aspects)}")
                if issue.empathy_score is not None and issue.empathy_score < 0.5:
                    feedback_parts.append(f"    💝 Add empathy statements (current score: {issue.empathy_score:.1f})")
            feedback_parts.append("")
        
        # Escalation warning for critical issues
        if self.escalation_priority in ["immediate", "high"]:
            feedback_parts.append("🚨 ESCALATION REQUIRED:")
            feedback_parts.append(f"   Priority: {self.escalation_priority.upper()}")
            if any(issue.is_fabricated for issue in self.criteria.policy_issues):
                feedback_parts.append("   Reason: Fabricated policy detected")
            feedback_parts.append("")
        
        feedback_parts.append("💡 CORRECTION INSTRUCTIONS:")
        feedback_parts.append("Please revise the draft to address the above issues.")
        feedback_parts.append("Focus on critical and major issues first.")
        
        return "\n".join(feedback_parts)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "criteria": {
                "price_accuracy_pass": True,
                "policy_authenticity_pass": False,
                "topic_relevance_pass": True,
                "overall_pass": False,
                "critical_issues_count": 1
            },
            "timestamp": "2024-01-15T10:30:00",
            "verification_reasoning": "Policy authenticity failed due to fabricated warranty terms",
            "execution_time_seconds": 2.5,
            "llm_tokens_used": 1250
        }
    })
