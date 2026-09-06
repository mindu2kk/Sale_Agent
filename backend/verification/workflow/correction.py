"""
Self-Correction Node Implementation

Structured issue-based correction feedback generation:
- Binary verification result analysis
- Specific correction instructions cho each issue type
- Prompt engineering cho Research Agent retry
- Issue severity-based feedback prioritization
"""

from typing import Dict, Any, List, Optional
from ..models import VerificationResult, PriceIssue, PolicyIssue, RelevanceIssue, IssueSeverity, FeedbackReport
from ..config import VerificationConfig


class SelfCorrectionNode:
    """
    Self-Correction Node cho structured issue feedback
    
    Generates specific correction instructions based on
    binary verification failures và issue classification.
    """
    
    def __init__(self, config: VerificationConfig):
        """
        Initialize Self-Correction Node
        
        Args:
            config: Verification configuration
        """
        self.config = config
    
    def generate_correction_feedback(self,
                                   original_objection: str,
                                   failed_draft: str,
                                   verification_result: VerificationResult,
                                   feedback_report: Optional[FeedbackReport] = None) -> str:
        """
        Generate structured correction feedback cho Research Agent.

        When a pre-built FeedbackReport is provided (from
        VerificationAgent.generate_structured_feedback()), its correction_prompt
        is used as the core of the feedback, supplemented with the retry
        instructions and quality checklist.  This avoids duplicating the
        severity-prioritised issue analysis logic.

        Args:
            original_objection: Original customer objection
            failed_draft: Draft response that failed verification
            verification_result: Detailed verification results
            feedback_report: Optional pre-built FeedbackReport from VerificationAgent

        Returns:
            Structured correction prompt for Research Agent retry
        """

        if verification_result.is_approved:
            return "✅ No corrections needed - verification passed"

        # If a structured FeedbackReport was provided, use its correction_prompt
        # as the primary issue analysis section.
        if feedback_report is not None and not feedback_report.is_approved:
            feedback_sections = [
                feedback_report.correction_prompt,
                self._build_retry_instructions_section(original_objection, failed_draft),
                self._build_quality_checklist_section(),
            ]
            return "\n\n".join(section for section in feedback_sections if section)

        # Fallback: build feedback from scratch using the verification_result directly
        feedback_sections = [
            self._build_header_section(verification_result),
            self._build_issue_analysis_section(verification_result),
            self._build_specific_corrections_section(verification_result),
            self._build_retry_instructions_section(original_objection, failed_draft),
            self._build_quality_checklist_section()
        ]

        return "\n\n".join(section for section in feedback_sections if section)
    
    def _build_header_section(self, verification_result: VerificationResult) -> str:
        """Build correction feedback header"""
        
        critical_count = verification_result.criteria.critical_issues_count
        total_issues = (
            len(verification_result.criteria.price_issues) +
            len(verification_result.criteria.policy_issues) +
            len(verification_result.criteria.relevance_issues)
        )
        
        header = [
            "🔄 VERIFICATION FAILED - CORRECTION REQUIRED",
            "=" * 50,
            "",
            f"📊 Issue Summary: {total_issues} total issues detected"
        ]
        
        if critical_count > 0:
            header.append(f"⚠️  CRITICAL: {critical_count} critical issues require immediate attention")
        
        header.extend([
            "",
            "❌ Failed Criteria:",
            f"  • Price Accuracy: {'❌ FAILED' if not verification_result.criteria.price_accuracy_pass else '✅ PASSED'}",
            f"  • Policy Authenticity: {'❌ FAILED' if not verification_result.criteria.policy_authenticity_pass else '✅ PASSED'}",
            f"  • Topic Relevance: {'❌ FAILED' if not verification_result.criteria.topic_relevance_pass else '✅ PASSED'}"
        ])
        
        return "\n".join(header)
    
    def _build_issue_analysis_section(self, verification_result: VerificationResult) -> str:
        """Build detailed issue analysis section"""
        
        analysis_parts = ["📋 DETAILED ISSUE ANALYSIS:"]
        
        # Price Issues Analysis
        if verification_result.criteria.price_issues:
            analysis_parts.append("\n💰 PRICE ACCURACY ISSUES:")
            for i, issue in enumerate(verification_result.criteria.price_issues, 1):
                severity_icon = self._get_severity_icon(issue.severity)
                analysis_parts.extend([
                    f"  {i}. {severity_icon} {issue.product_name}",
                    f"     • Mentioned: {issue.mentioned_price or 'Not specified'}",
                    f"     • Actual: {issue.actual_price or 'Unknown'}",
                    f"     • Deviation: {issue.deviation_percent:.1f}%" if issue.deviation_percent else "     • Deviation: Unknown",
                    f"     • Issue: {issue.explanation}",
                    ""
                ])
        
        # Policy Issues Analysis
        if verification_result.criteria.policy_issues:
            analysis_parts.append("📋 POLICY AUTHENTICITY ISSUES:")
            for i, issue in enumerate(verification_result.criteria.policy_issues, 1):
                severity_icon = self._get_severity_icon(issue.severity)
                analysis_parts.extend([
                    f"  {i}. {severity_icon} {issue.policy_type.title()} Policy",
                    f"     • Mentioned: \"{issue.mentioned_policy}\"",
                    f"     • Problem: {'Fabricated policy' if issue.is_fabricated else 'Inaccurate policy'}",
                    f"     • Correct Policy: {issue.correct_policy or 'See official documents'}",
                    f"     • Issue: {issue.explanation}",
                    ""
                ])
        
        # Relevance Issues Analysis
        if verification_result.criteria.relevance_issues:
            analysis_parts.append("🎯 TOPIC RELEVANCE ISSUES:")
            for i, issue in enumerate(verification_result.criteria.relevance_issues, 1):
                severity_icon = self._get_severity_icon(issue.severity)
                analysis_parts.extend([
                    f"  {i}. {severity_icon} Response Coverage: {issue.response_coverage:.1%}",
                    f"     • Objection Intent: {issue.objection_intent}",
                    f"     • Missing Aspects: {', '.join(issue.missing_aspects) if issue.missing_aspects else 'None identified'}",
                    f"     • Off-topic Content: {', '.join(issue.off_topic_content) if issue.off_topic_content else 'None identified'}",
                    f"     • Issue: {issue.explanation}",
                    ""
                ])
        
        return "\n".join(analysis_parts)
    
    def _build_specific_corrections_section(self, verification_result: VerificationResult) -> str:
        """Build specific correction instructions"""
        
        corrections = ["🛠️  SPECIFIC CORRECTIONS REQUIRED:"]
        
        # Price Correction Instructions
        if verification_result.criteria.price_issues:
            corrections.append("\n💰 Price Accuracy Corrections:")
            for issue in verification_result.criteria.price_issues:
                if issue.severity == IssueSeverity.CRITICAL:
                    corrections.append(f"  🚨 CRITICAL: {self._get_price_correction_instruction(issue)}")
                else:
                    corrections.append(f"  • {self._get_price_correction_instruction(issue)}")
        
        # Policy Correction Instructions
        if verification_result.criteria.policy_issues:
            corrections.append("\n📋 Policy Authenticity Corrections:")
            for issue in verification_result.criteria.policy_issues:
                if issue.severity == IssueSeverity.CRITICAL:
                    corrections.append(f"  🚨 CRITICAL: {self._get_policy_correction_instruction(issue)}")
                else:
                    corrections.append(f"  • {self._get_policy_correction_instruction(issue)}")
        
        # Relevance Correction Instructions
        if verification_result.criteria.relevance_issues:
            corrections.append("\n🎯 Topic Relevance Corrections:")
            for issue in verification_result.criteria.relevance_issues:
                if issue.severity == IssueSeverity.CRITICAL:
                    corrections.append(f"  🚨 CRITICAL: {self._get_relevance_correction_instruction(issue)}")
                else:
                    corrections.append(f"  • {self._get_relevance_correction_instruction(issue)}")
        
        return "\n".join(corrections)
    
    def _build_retry_instructions_section(self, objection: str, failed_draft: str) -> str:
        """Build retry instructions for Research Agent"""
        
        instructions = [
            "🔄 RETRY INSTRUCTIONS:",
            "",
            "When generating the corrected response, you MUST:",
            "",
            "1. 📊 Address ALL issues listed above in priority order (Critical → Major → Minor)",
            "2. 🔍 Cross-check ALL price information against the internal database",
            "3. 📋 Verify ALL policy statements against official documents",
            "4. 🎯 Ensure response directly addresses the customer's specific objection",
            "5. ✅ Include proper citations and sources for all claims",
            "",
            "💡 IMPROVEMENT STRATEGIES:",
            "• Use specific product names and exact pricing from database",
            "• Quote official policy documents with section references", 
            "• Structure response to directly answer objection components",
            "• Add empathy statements to improve customer connection",
            "• Remove any generic or off-topic content",
            "",
            f"📝 ORIGINAL OBJECTION TO ADDRESS:",
            f'"{objection}"',
            "",
            "🎯 Focus your response on solving the customer's specific concern while ensuring 100% accuracy."
        ]
        
        return "\n".join(instructions)
    
    def _build_quality_checklist_section(self) -> str:
        """Build quality checklist for final verification"""
        
        checklist = [
            "✅ QUALITY CHECKLIST - Verify before submitting:",
            "",
            "Price Accuracy:",
            "  □ All prices match internal database exactly",
            "  □ Product names and SKUs are correct",
            "  □ Currency and formatting are consistent",
            "",
            "Policy Authenticity:",
            "  □ All policies quoted from official documents",
            "  □ No fabricated or assumed policy statements",
            "  □ Proper citations and references included",
            "",
            "Topic Relevance:",
            "  □ Response directly addresses customer objection",
            "  □ All objection components are covered",
            "  □ No off-topic or irrelevant content",
            "  □ Empathy and understanding demonstrated",
            "",
            "Overall Quality:",
            "  □ Response is clear and professional",
            "  □ Information is complete and actionable",
            "  □ Customer concern is fully resolved"
        ]
        
        return "\n".join(checklist)
    
    def _get_severity_icon(self, severity: IssueSeverity) -> str:
        """Get icon for issue severity"""
        icons = {
            IssueSeverity.CRITICAL: "🚨",
            IssueSeverity.MAJOR: "⚠️",
            IssueSeverity.MINOR: "ℹ️"
        }
        return icons.get(severity, "•")
    
    def _get_price_correction_instruction(self, issue: PriceIssue) -> str:
        """Generate specific price correction instruction"""
        
        if issue.actual_price:
            return f"Update {issue.product_name} price from '{issue.mentioned_price}' to '{issue.actual_price}'"
        else:
            return f"Verify and correct {issue.product_name} pricing using internal database"
    
    def _get_policy_correction_instruction(self, issue: PolicyIssue) -> str:
        """Generate specific policy correction instruction"""
        
        if issue.is_fabricated:
            return f"Remove fabricated {issue.policy_type} policy and replace with official policy from documents"
        elif issue.correct_policy:
            return f"Replace inaccurate {issue.policy_type} policy with: '{issue.correct_policy}'"
        else:
            return f"Verify {issue.policy_type} policy against official documents and correct"
    
    def _get_relevance_correction_instruction(self, issue: RelevanceIssue) -> str:
        """Generate specific relevance correction instruction"""
        
        instructions = []
        
        if issue.missing_aspects:
            instructions.append(f"Address missing aspects: {', '.join(issue.missing_aspects)}")
        
        if issue.off_topic_content:
            instructions.append(f"Remove off-topic content: {', '.join(issue.off_topic_content)}")
        
        if issue.response_coverage < 0.5:
            instructions.append(f"Significantly expand response to better address '{issue.objection_intent}' objection")
        
        return "; ".join(instructions) if instructions else "Improve response relevance to objection"
    
    def should_escalate_immediately(self, verification_result: VerificationResult) -> bool:
        """
        Determine if workflow should escalate immediately
        
        Returns:
            True if immediate escalation required
        """
        
        # Escalate on critical issues if configured
        if (self.config.critical_issue_escalation and 
            verification_result.criteria.critical_issues_count > 0):
            return True
        
        # Escalate if too many total issues
        total_issues = (
            len(verification_result.criteria.price_issues) +
            len(verification_result.criteria.policy_issues) +
            len(verification_result.criteria.relevance_issues)
        )
        
        if total_issues >= 5:  # Threshold for too many issues
            return True
        
        return False