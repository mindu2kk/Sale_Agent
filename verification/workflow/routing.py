"""
Workflow Routing Logic

Conditional edge routing cho LangGraph StateGraph:
- Binary decision routing after verification
- Retry vs escalation logic after correction
- Issue severity-based routing decisions
- Performance-optimized routing với early termination
"""

from typing import Literal
from ..models import WorkflowState, VerificationResult, IssueSeverity
from ..config import VerificationConfig
from ..utils.early_termination import CriticalIssueDetector, should_terminate_immediately


class WorkflowRouter:
    """
    Conditional routing logic cho StateGraph edges
    
    Implements binary decision routing based on verification results
    và issue severity classification.
    """
    
    def __init__(self, config: VerificationConfig):
        """
        Initialize workflow router
        
        Args:
            config: Verification configuration
        """
        self.config = config
        self._critical_detector = CriticalIssueDetector()
    
    def route_after_verification(self, state: WorkflowState) -> Literal["approved", "correction", "escalation"]:
        """
        Route workflow after verification node
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node: "approved", "correction", or "escalation"
        """
        
        verification_result = state.get("verification_result")
        
        if verification_result is None:
            # No verification result - escalate
            return "escalation"
        
        # Check if approved
        if verification_result.is_approved:
            return "approved"
        
        # Check immediate_termination flag set by CriticalIssueDetector (Task 5.4.1)
        if getattr(verification_result, "immediate_termination", False):
            return "escalation"

        # Check for immediate escalation conditions (uses CriticalIssueDetector internally)
        if self._should_escalate_immediately(verification_result, state):
            return "escalation"
        
        # Check retry limits
        if state["retry_count"] >= state["max_retries"]:
            return "escalation"
        
        # Route to correction for retry
        return "correction"
    
    def route_after_correction(self, state: WorkflowState) -> Literal["retry", "escalation"]:
        """
        Route workflow after correction node
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node: "retry" or "escalation"
        """
        
        # Check retry limits
        if state["retry_count"] >= state["max_retries"]:
            return "escalation"
        
        # Check for escalation conditions
        verification_result = state.get("verification_result")
        if verification_result and self._should_escalate_after_correction(verification_result, state):
            return "escalation"
        
        # Continue with retry
        return "retry"
    
    def _should_escalate_immediately(self, 
                                   verification_result: VerificationResult, 
                                   state: WorkflowState) -> bool:
        """
        Determine if workflow should escalate immediately after verification.

        Uses CriticalIssueDetector (Task 5.4.1): any single critical issue
        triggers immediate escalation, bypassing the correction loop.
        Only applies when config.critical_issue_escalation is True.
        """
        # Only use CriticalIssueDetector when critical escalation is enabled
        if getattr(self.config, 'critical_issue_escalation', True):
            termination_decision = self._critical_detector.check_verification_result(verification_result)
            if termination_decision.should_terminate:
                return True
        
        # Escalate if too many total issues (indicates systemic problem)
        total_issues = self._count_total_issues(verification_result)
        if total_issues >= 5:
            return True
        
        # Escalate if fabricated policies detected (compliance risk)
        fabricated_policies = sum(
            1 for issue in verification_result.criteria.policy_issues
            if issue.is_fabricated and issue.severity == IssueSeverity.CRITICAL
        )
        if fabricated_policies > 0:
            return True
        
        return False
    
    def _should_escalate_after_correction(self, 
                                        verification_result: VerificationResult,
                                        state: WorkflowState) -> bool:
        """
        Determine if workflow should escalate after correction attempt
        
        Args:
            verification_result: Previous verification results
            state: Current workflow state
            
        Returns:
            True if escalation required after correction
        """
        
        # Escalate if we've had multiple retries with critical issues
        if (state["retry_count"] >= 2 and 
            verification_result.criteria.critical_issues_count > 0):
            return True
        
        # Escalate if issue complexity suggests human intervention needed
        if self._is_complex_issue_pattern(verification_result):
            return True
        
        return False
    
    def _count_total_issues(self, verification_result: VerificationResult) -> int:
        """Count total issues across all criteria"""
        return (
            len(verification_result.criteria.price_issues) +
            len(verification_result.criteria.policy_issues) +
            len(verification_result.criteria.relevance_issues)
        )
    
    def _is_complex_issue_pattern(self, verification_result: VerificationResult) -> bool:
        """
        Detect complex issue patterns that suggest human intervention needed
        
        Args:
            verification_result: Verification results to analyze
            
        Returns:
            True if complex patterns detected
        """
        
        # Pattern 1: Multiple criteria failed simultaneously
        failed_criteria_count = sum([
            not verification_result.criteria.price_accuracy_pass,
            not verification_result.criteria.policy_authenticity_pass,
            not verification_result.criteria.topic_relevance_pass
        ])
        
        if failed_criteria_count >= 3:  # All criteria failed
            return True
        
        # Pattern 2: Mix of critical and major issues across criteria
        has_critical_price = any(
            issue.severity == IssueSeverity.CRITICAL 
            for issue in verification_result.criteria.price_issues
        )
        has_critical_policy = any(
            issue.severity == IssueSeverity.CRITICAL
            for issue in verification_result.criteria.policy_issues
        )
        has_major_relevance = any(
            issue.severity == IssueSeverity.MAJOR
            for issue in verification_result.criteria.relevance_issues
        )
        
        if sum([has_critical_price, has_critical_policy, has_major_relevance]) >= 2:
            return True
        
        # Pattern 3: High deviation in price issues (>50%)
        high_price_deviations = sum(
            1 for issue in verification_result.criteria.price_issues
            if issue.deviation_percent and issue.deviation_percent > 50.0
        )
        
        if high_price_deviations >= 2:
            return True
        
        # Pattern 4: Multiple fabricated policies
        fabricated_count = sum(
            1 for issue in verification_result.criteria.policy_issues
            if issue.is_fabricated
        )
        
        if fabricated_count >= 2:
            return True
        
        # Pattern 5: Very low relevance coverage (<30%)
        low_coverage_issues = sum(
            1 for issue in verification_result.criteria.relevance_issues
            if issue.response_coverage < 0.3
        )
        
        if low_coverage_issues > 0:
            return True
        
        return False
    
    def get_routing_decision_summary(self, 
                                   state: WorkflowState, 
                                   decision: str) -> str:
        """
        Generate human-readable routing decision summary
        
        Args:
            state: Current workflow state
            decision: Routing decision made
            
        Returns:
            Summary of routing decision reasoning
        """
        
        verification_result = state.get("verification_result")
        
        if decision == "approved":
            return "✅ Verification passed - workflow approved"
        
        elif decision == "escalation":
            reasons = []
            
            if state["retry_count"] >= state["max_retries"]:
                reasons.append(f"Maximum retries ({state['max_retries']}) exceeded")
            
            if verification_result:
                if verification_result.criteria.critical_issues_count > 0:
                    reasons.append(f"Critical issues detected ({verification_result.criteria.critical_issues_count})")
                
                if self._is_complex_issue_pattern(verification_result):
                    reasons.append("Complex issue pattern requires human intervention")
                
                total_issues = self._count_total_issues(verification_result)
                if total_issues >= 5:
                    reasons.append(f"Too many issues ({total_issues}) indicate systemic problem")
            
            if not reasons:
                reasons.append("Escalation triggered by routing logic")
            
            return f"🚨 Escalating to human review: {'; '.join(reasons)}"
        
        elif decision == "correction":
            retry_num = state["retry_count"] + 1
            max_retries = state["max_retries"]
            
            issue_summary = ""
            if verification_result:
                total_issues = self._count_total_issues(verification_result)
                critical_issues = verification_result.criteria.critical_issues_count
                
                if critical_issues > 0:
                    issue_summary = f" ({critical_issues} critical, {total_issues} total issues)"
                else:
                    issue_summary = f" ({total_issues} issues)"
            
            return f"🔄 Routing to correction - retry {retry_num}/{max_retries}{issue_summary}"
        
        elif decision == "retry":
            return f"🔄 Retrying research with correction feedback (attempt {state['retry_count'] + 1})"
        
        else:
            return f"❓ Unknown routing decision: {decision}"
    
    def get_routing_metrics(self, state: WorkflowState) -> dict:
        """
        Get routing decision metrics for observability
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of routing metrics
        """
        
        verification_result = state.get("verification_result")
        
        metrics = {
            "retry_count": state["retry_count"],
            "max_retries": state["max_retries"],
            "workflow_status": state["workflow_status"]
        }
        
        if verification_result:
            metrics.update({
                "verification_passed": verification_result.is_approved,
                "critical_issues_count": verification_result.criteria.critical_issues_count,
                "total_issues_count": self._count_total_issues(verification_result),
                "price_accuracy_pass": verification_result.criteria.price_accuracy_pass,
                "policy_authenticity_pass": verification_result.criteria.policy_authenticity_pass,
                "topic_relevance_pass": verification_result.criteria.topic_relevance_pass,
                "complex_pattern_detected": self._is_complex_issue_pattern(verification_result)
            })
        
        return metrics