"""
Test suite for VerificationResult Pydantic model với binary pass/fail status

Tests cover:
- Binary verification decisions (PASS/FAIL)
- Structured issue tracking
- Feedback generation
- Pydantic validation
"""

import pytest
from datetime import datetime
from backend.verification.models.verification import (
    VerificationResult,
    RubricCriteria,
    PriceIssue,
    PolicyIssue,
    RelevanceIssue,
    IssueSeverity
)


class TestVerificationResult:
    """Test VerificationResult binary pass/fail functionality"""
    
    def test_binary_pass_all_criteria(self):
        """Test PASS when all criteria pass"""
        criteria = RubricCriteria(
            price_accuracy_pass=True,
            policy_authenticity_pass=True,
            topic_relevance_pass=True
        )
        
        result = VerificationResult(
            criteria=criteria,
            verification_reasoning="All checks passed",
            execution_time_seconds=1.5,
            llm_tokens_used=800
        )
        
        assert result.is_approved is True
        assert result.requires_correction is False
        assert result.criteria.overall_pass is True
        assert result.criteria.critical_issues_count == 0
    
    def test_binary_fail_single_criteria(self):
        """Test FAIL when any single criteria fails"""
        criteria = RubricCriteria(
            price_accuracy_pass=False,  # This fails
            policy_authenticity_pass=True,
            topic_relevance_pass=True,
            price_issues=[
                PriceIssue(
                    product_name="iPhone 15",
                    mentioned_price="30,000,000 VND",
                    actual_price="29,990,000 VND",
                    deviation_percent=0.03,
                    severity=IssueSeverity.MINOR,
                    explanation="Giá sai lệch nhỏ"
                )
            ]
        )
        
        result = VerificationResult(
            criteria=criteria,
            verification_reasoning="Price accuracy failed",
            execution_time_seconds=2.0,
            llm_tokens_used=1200
        )
        
        assert result.is_approved is False
        assert result.requires_correction is True
        assert result.criteria.overall_pass is False
        assert result.criteria.critical_issues_count == 0
    
    def test_critical_issue_escalation(self):
        """Test escalation when critical issues detected"""
        criteria = RubricCriteria(
            price_accuracy_pass=False,
            policy_authenticity_pass=False,
            topic_relevance_pass=True,
            price_issues=[
                PriceIssue(
                    product_name="iPhone 15",
                    mentioned_price="50,000,000 VND",
                    actual_price="30,000,000 VND",
                    deviation_percent=66.7,
                    severity=IssueSeverity.CRITICAL,
                    explanation="Giá sai lệch nghiêm trọng"
                )
            ],
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="Bảo hành 5 năm miễn phí",
                    policy_type="warranty",
                    is_fabricated=True,
                    is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL,
                    explanation="Chính sách bịa đặt hoàn toàn"
                )
            ]
        )
        
        result = VerificationResult(
            criteria=criteria,
            verification_reasoning="Multiple critical issues detected",
            execution_time_seconds=3.0,
            llm_tokens_used=1800
        )
        
        assert result.requires_escalation is True
        assert result.criteria.critical_issues_count == 2
        assert result.is_approved is False
    
    def test_structured_feedback_generation(self):
        """Test structured correction feedback generation"""
        criteria = RubricCriteria(
            price_accuracy_pass=False,
            policy_authenticity_pass=True,
            topic_relevance_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="Samsung Galaxy S24",
                    mentioned_price="25,000,000 VND",
                    actual_price="24,990,000 VND",
                    deviation_percent=0.04,
                    severity=IssueSeverity.MINOR,
                    explanation="Giá cần cập nhật chính xác"
                )
            ],
            relevance_issues=[
                RelevanceIssue(
                    objection_intent="So sánh camera iPhone vs Samsung",
                    response_coverage=0.4,
                    missing_aspects=["Chất lượng ảnh", "Tính năng AI"],
                    severity=IssueSeverity.MAJOR,
                    explanation="Response thiếu so sánh chi tiết về camera"
                )
            ]
        )
        
        result = VerificationResult(
            criteria=criteria,
            verification_reasoning="Price and relevance issues detected",
            execution_time_seconds=2.5,
            llm_tokens_used=1500
        )
        
        feedback = result.get_correction_feedback()
        
        # Check feedback contains expected sections
        assert "🔄 VERIFICATION FAILED" in feedback
        assert "💰 PRICE ACCURACY ISSUES:" in feedback
        assert "🎯 TOPIC RELEVANCE ISSUES:" in feedback
        assert "💡 CORRECTION INSTRUCTIONS:" in feedback
        assert "Giá cần cập nhật chính xác" in feedback
        assert "Response thiếu so sánh chi tiết về camera" in feedback
    
    def test_pydantic_validation(self):
        """Test Pydantic validation rules"""
        # Test valid model creation
        criteria = RubricCriteria(
            price_accuracy_pass=True,
            policy_authenticity_pass=True,
            topic_relevance_pass=True
        )
        
        result = VerificationResult(
            criteria=criteria,
            verification_reasoning="Test validation",
            execution_time_seconds=1.0,
            llm_tokens_used=500
        )
        
        # Verify computed fields
        assert isinstance(result.timestamp, datetime)
        assert result.execution_time_seconds >= 0.0
        assert result.llm_tokens_used >= 0
        
        # Test validation constraints
        with pytest.raises(ValueError):
            VerificationResult(
                criteria=criteria,
                verification_reasoning="",  # Too short
                execution_time_seconds=-1.0,  # Negative
                llm_tokens_used=-100  # Negative
            )
    
    def test_issue_severity_counting(self):
        """Test automatic counting of critical issues"""
        criteria = RubricCriteria(
            price_accuracy_pass=False,
            policy_authenticity_pass=False,
            topic_relevance_pass=False,
            price_issues=[
                PriceIssue(
                    product_name="Product A",
                    severity=IssueSeverity.CRITICAL,
                    explanation="Critical price issue"
                ),
                PriceIssue(
                    product_name="Product B", 
                    severity=IssueSeverity.MINOR,
                    explanation="Minor price issue"
                )
            ],
            policy_issues=[
                PolicyIssue(
                    mentioned_policy="Fake policy",
                    policy_type="warranty",
                    is_fabricated=True,
                    is_inaccurate=False,
                    severity=IssueSeverity.CRITICAL,
                    explanation="Critical policy issue"
                )
            ],
            relevance_issues=[
                RelevanceIssue(
                    objection_intent="Test intent",
                    response_coverage=0.2,
                    severity=IssueSeverity.MAJOR,
                    explanation="Major relevance issue"
                )
            ]
        )
        
        # Should count 2 critical issues (1 price + 1 policy)
        assert criteria.critical_issues_count == 2
        
        # Overall should fail since all criteria failed
        assert criteria.overall_pass is False


if __name__ == "__main__":
    pytest.main([__file__])
