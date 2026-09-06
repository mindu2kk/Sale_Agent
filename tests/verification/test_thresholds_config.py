"""
Unit Tests: Verification Thresholds Configuration

Tests for the verification thresholds configuration system supporting
critical/major/minor issue classification in binary verification.

Supports Task 1.3.1: Design verification thresholds config cho critical/major/minor issues
"""

import pytest
from unittest.mock import patch, mock_open
from backend.verification.config.thresholds_config import (
    VerificationThresholdsConfig,
    PriceAccuracyThresholds,
    PolicyAuthenticityThresholds,
    TopicRelevanceThresholds,
    EscalationThresholds,
    IssueSeverity,
    get_default_thresholds_config,
    enhanced_load_thresholds_config,
    _convert_yaml_to_pydantic_structure,
    _get_thresholds_env_overrides
)


class TestPriceAccuracyThresholds:
    """Test price accuracy thresholds configuration"""
    
    def test_default_thresholds(self):
        """Test default price accuracy thresholds"""
        thresholds = PriceAccuracyThresholds()
        
        assert thresholds.minor_threshold_percent == 5.0
        assert thresholds.major_threshold_percent == 15.0
        assert thresholds.critical_threshold_percent == 30.0
        assert thresholds.pass_tolerance_percent == 1.0
        assert thresholds.missing_price_severity == IssueSeverity.MAJOR
    
    def test_price_deviation_classification(self):
        """Test price deviation severity classification"""
        thresholds = PriceAccuracyThresholds()
        
        # Test different deviation levels
        assert thresholds.classify_price_deviation(0.5) == IssueSeverity.MINOR
        assert thresholds.classify_price_deviation(3.0) == IssueSeverity.MINOR
        assert thresholds.classify_price_deviation(8.0) == IssueSeverity.MAJOR
        assert thresholds.classify_price_deviation(20.0) == IssueSeverity.MAJOR
        assert thresholds.classify_price_deviation(35.0) == IssueSeverity.CRITICAL
    
    def test_pass_criteria(self):
        """Test price pass/fail criteria"""
        thresholds = PriceAccuracyThresholds()
        
        # Within tolerance should pass
        assert thresholds.should_pass_price_check(0.5) == True
        assert thresholds.should_pass_price_check(1.0) == True
        
        # Outside tolerance should fail
        assert thresholds.should_pass_price_check(1.5) == False
        assert thresholds.should_pass_price_check(5.0) == False
    
    def test_threshold_validation(self):
        """Test threshold validation rules"""
        # Valid thresholds should work
        thresholds = PriceAccuracyThresholds(
            minor_threshold_percent=5.0,
            major_threshold_percent=15.0,
            critical_threshold_percent=30.0
        )
        assert thresholds.minor_threshold_percent == 5.0
        
        # Invalid thresholds should raise validation error
        with pytest.raises(ValueError):
            PriceAccuracyThresholds(
                minor_threshold_percent=15.0,
                major_threshold_percent=10.0,  # Should be > minor
                critical_threshold_percent=30.0
            )
        
        with pytest.raises(ValueError):
            PriceAccuracyThresholds(
                minor_threshold_percent=5.0,
                major_threshold_percent=15.0,
                critical_threshold_percent=10.0  # Should be > major
            )


class TestPolicyAuthenticityThresholds:
    """Test policy authenticity thresholds configuration"""
    
    def test_default_thresholds(self):
        """Test default policy authenticity thresholds"""
        thresholds = PolicyAuthenticityThresholds()
        
        assert thresholds.fabricated_policy_severity == IssueSeverity.CRITICAL
        assert thresholds.inaccurate_policy_severity == IssueSeverity.MAJOR
        assert thresholds.incomplete_policy_severity == IssueSeverity.MINOR
        assert thresholds.citation_required == True
    
    def test_policy_issue_classification(self):
        """Test policy issue severity classification"""
        thresholds = PolicyAuthenticityThresholds()
        
        # Fabricated policy should be critical
        severity = thresholds.classify_policy_issue(
            is_fabricated=True,
            is_inaccurate=False,
            is_incomplete=False,
            policy_type="warranty",
            has_citation=False
        )
        assert severity == IssueSeverity.CRITICAL
        
        # Inaccurate policy should be major
        severity = thresholds.classify_policy_issue(
            is_fabricated=False,
            is_inaccurate=True,
            is_incomplete=False,
            policy_type="return",
            has_citation=True
        )
        assert severity == IssueSeverity.MAJOR
        
        # Missing citation for warranty should be critical (escalated)
        severity = thresholds.classify_policy_issue(
            is_fabricated=False,
            is_inaccurate=False,
            is_incomplete=False,
            policy_type="warranty",
            has_citation=False
        )
        assert severity == IssueSeverity.CRITICAL
        
        # Incomplete policy should be minor
        severity = thresholds.classify_policy_issue(
            is_fabricated=False,
            is_inaccurate=False,
            is_incomplete=True,
            policy_type="service",
            has_citation=True
        )
        assert severity == IssueSeverity.MINOR
    
    def test_pass_criteria(self):
        """Test policy pass/fail criteria"""
        thresholds = PolicyAuthenticityThresholds()
        
        # Valid policy should pass
        assert thresholds.should_pass_policy_check(
            is_fabricated=False,
            is_inaccurate=False,
            has_required_citation=True
        ) == True
        
        # Fabricated policy should fail
        assert thresholds.should_pass_policy_check(
            is_fabricated=True,
            is_inaccurate=False,
            has_required_citation=True
        ) == False
        
        # Inaccurate policy should fail
        assert thresholds.should_pass_policy_check(
            is_fabricated=False,
            is_inaccurate=True,
            has_required_citation=True
        ) == False
        
        # Missing citation should fail
        assert thresholds.should_pass_policy_check(
            is_fabricated=False,
            is_inaccurate=False,
            has_required_citation=False
        ) == False


class TestTopicRelevanceThresholds:
    """Test topic relevance thresholds configuration"""
    
    def test_default_thresholds(self):
        """Test default topic relevance thresholds"""
        thresholds = TopicRelevanceThresholds()
        
        assert thresholds.minor_coverage_threshold == 0.8
        assert thresholds.major_coverage_threshold == 0.5
        assert thresholds.critical_coverage_threshold == 0.3
        assert thresholds.pass_coverage_threshold == 0.7
        assert thresholds.empathy_required == True
    
    def test_relevance_issue_classification(self):
        """Test topic relevance issue severity classification"""
        thresholds = TopicRelevanceThresholds()
        
        # High coverage should be minor or no issue
        severity = thresholds.classify_relevance_issue(
            coverage_ratio=0.9,
            empathy_score=0.8,
            off_topic_ratio=0.0
        )
        assert severity == IssueSeverity.MINOR
        
        # Medium coverage should be major
        severity = thresholds.classify_relevance_issue(
            coverage_ratio=0.6,
            empathy_score=0.5,
            off_topic_ratio=0.1
        )
        assert severity == IssueSeverity.MAJOR
        
        # Low coverage should be critical
        severity = thresholds.classify_relevance_issue(
            coverage_ratio=0.2,
            empathy_score=0.3,
            off_topic_ratio=0.2
        )
        assert severity == IssueSeverity.CRITICAL
        
        # Completely irrelevant should be critical
        severity = thresholds.classify_relevance_issue(
            coverage_ratio=0.05,
            empathy_score=0.0,
            off_topic_ratio=0.8
        )
        assert severity == IssueSeverity.CRITICAL
        
        # High off-topic content should be critical
        severity = thresholds.classify_relevance_issue(
            coverage_ratio=0.7,
            empathy_score=0.5,
            off_topic_ratio=0.6
        )
        assert severity == IssueSeverity.CRITICAL
    
    def test_pass_criteria(self):
        """Test topic relevance pass/fail criteria"""
        thresholds = TopicRelevanceThresholds()
        
        # Good coverage and empathy should pass
        assert thresholds.should_pass_relevance_check(
            coverage_ratio=0.8,
            empathy_score=0.7
        ) == True
        
        # Low coverage should fail
        assert thresholds.should_pass_relevance_check(
            coverage_ratio=0.6,
            empathy_score=0.7
        ) == False
        
        # Low empathy should fail
        assert thresholds.should_pass_relevance_check(
            coverage_ratio=0.8,
            empathy_score=0.3
        ) == False
    
    def test_threshold_validation(self):
        """Test threshold validation rules"""
        # Valid thresholds should work
        thresholds = TopicRelevanceThresholds(
            minor_coverage_threshold=0.8,
            major_coverage_threshold=0.5,
            critical_coverage_threshold=0.3
        )
        assert thresholds.minor_coverage_threshold == 0.8
        
        # Invalid thresholds should raise validation error
        with pytest.raises(ValueError):
            TopicRelevanceThresholds(
                minor_coverage_threshold=0.5,
                major_coverage_threshold=0.8,  # Should be < minor
                critical_coverage_threshold=0.3
            )


class TestEscalationThresholds:
    """Test escalation thresholds configuration"""
    
    def test_default_thresholds(self):
        """Test default escalation thresholds"""
        thresholds = EscalationThresholds()
        
        assert thresholds.max_critical_issues_before_escalation == 2
        assert thresholds.max_major_issues_before_escalation == 5
        assert thresholds.fabricated_policy_immediate_escalation == True
        assert thresholds.early_termination_enabled == True
    
    def test_immediate_escalation_logic(self):
        """Test immediate escalation decision logic"""
        thresholds = EscalationThresholds()
        
        # Fabricated policy should trigger immediate escalation
        assert thresholds.should_escalate_immediately(
            critical_count=1,
            major_count=0,
            total_count=1,
            has_fabricated_policy=True
        ) == True
        
        # Critical price deviation should trigger escalation
        assert thresholds.should_escalate_immediately(
            critical_count=1,
            major_count=0,
            total_count=1,
            has_critical_price_deviation=True
        ) == True
        
        # Too many critical issues should trigger escalation
        assert thresholds.should_escalate_immediately(
            critical_count=3,
            major_count=0,
            total_count=3
        ) == True
        
        # Too many major issues should trigger escalation
        assert thresholds.should_escalate_immediately(
            critical_count=0,
            major_count=6,
            total_count=6
        ) == True
        
        # Few issues should not trigger escalation
        assert thresholds.should_escalate_immediately(
            critical_count=1,
            major_count=2,
            total_count=3
        ) == False
    
    def test_early_termination_logic(self):
        """Test early termination decision logic"""
        thresholds = EscalationThresholds()
        
        # Multiple critical issues should trigger termination
        assert thresholds.should_terminate_early(critical_count=3) == True
        
        # Single critical should not trigger termination (by default)
        assert thresholds.should_terminate_early(critical_count=1) == False
        
        # No critical issues should not trigger termination
        assert thresholds.should_terminate_early(critical_count=0) == False
        
        # Test with stop_on_first_critical enabled
        thresholds.stop_on_first_critical = True
        assert thresholds.should_terminate_early(critical_count=1) == True
    
    def test_retry_limits_by_severity(self):
        """Test retry limits based on issue severity"""
        thresholds = EscalationThresholds()
        
        # Critical issues should have limited retries
        max_retries = thresholds.get_max_retries_for_severity(
            critical_count=1,
            major_count=0,
            minor_count=0
        )
        assert max_retries == 1
        
        # Major issues should have more retries
        max_retries = thresholds.get_max_retries_for_severity(
            critical_count=0,
            major_count=2,
            minor_count=0
        )
        assert max_retries == 3
        
        # Minor issues should have most retries
        max_retries = thresholds.get_max_retries_for_severity(
            critical_count=0,
            major_count=0,
            minor_count=3
        )
        assert max_retries == 5


class TestVerificationThresholdsConfig:
    """Test complete verification thresholds configuration"""
    
    def test_default_configuration(self):
        """Test default configuration creation"""
        config = get_default_thresholds_config()
        
        assert isinstance(config.price_accuracy, PriceAccuracyThresholds)
        assert isinstance(config.policy_authenticity, PolicyAuthenticityThresholds)
        assert isinstance(config.topic_relevance, TopicRelevanceThresholds)
        assert isinstance(config.escalation, EscalationThresholds)
        
        # Check verification weights sum to 1.0
        weights = config.verification_weights
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_environment_overrides(self):
        """Test environment-specific configuration overrides"""
        config = get_default_thresholds_config()
        
        # Test development environment
        dev_config = config.get_environment_config("development")
        assert dev_config.price_accuracy.pass_tolerance_percent == 2.0
        assert dev_config.escalation.max_critical_issues_before_escalation == 5
        
        # Test production environment
        prod_config = config.get_environment_config("production")
        assert prod_config.price_accuracy.pass_tolerance_percent == 0.5
        assert prod_config.escalation.max_critical_issues_before_escalation == 1
        
        # Test unknown environment (should return original)
        unknown_config = config.get_environment_config("unknown")
        assert unknown_config == config
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = get_default_thresholds_config()
        
        # Default configuration should have no warnings
        warnings = config.validate_configuration()
        assert len(warnings) == 0
        
        # Create invalid configuration
        config.price_accuracy.pass_tolerance_percent = 10.0  # Higher than minor threshold
        warnings = config.validate_configuration()
        assert len(warnings) > 0
        assert any("tolerance" in warning.lower() for warning in warnings)
    
    def test_weights_validation(self):
        """Test verification weights validation"""
        # Valid weights should work
        config = VerificationThresholdsConfig(
            verification_weights={
                "price_accuracy": 0.4,
                "policy_authenticity": 0.3,
                "topic_relevance": 0.3
            }
        )
        assert config.verification_weights["price_accuracy"] == 0.4
        
        # Invalid weights should raise error
        with pytest.raises(ValueError):
            VerificationThresholdsConfig(
                verification_weights={
                    "price_accuracy": 0.5,
                    "policy_authenticity": 0.3,
                    "topic_relevance": 0.3  # Sum = 1.1, should be 1.0
                }
            )


class TestYAMLConfigurationLoading:
    """Test YAML configuration loading functionality"""
    
    def test_yaml_to_pydantic_conversion(self):
        """Test conversion from YAML structure to Pydantic model structure"""
        yaml_data = {
            "price_accuracy": {
                "thresholds": {
                    "minor_threshold_percent": 3.0,
                    "major_threshold_percent": 10.0,
                    "critical_threshold_percent": 25.0
                },
                "pass_criteria": {
                    "tolerance_percent": 0.8,
                    "allow_promotional_pricing": True
                }
            },
            "escalation": {
                "count_thresholds": {
                    "max_critical_before_escalation": 1
                },
                "immediate_triggers": {
                    "fabricated_policy_detected": True
                }
            }
        }
        
        pydantic_data = _convert_yaml_to_pydantic_structure(yaml_data)
        
        # Check price accuracy conversion
        assert pydantic_data["price_accuracy"]["minor_threshold_percent"] == 3.0
        assert pydantic_data["price_accuracy"]["pass_tolerance_percent"] == 0.8
        
        # Check escalation conversion
        assert pydantic_data["escalation"]["max_critical_issues_before_escalation"] == 1
        assert pydantic_data["escalation"]["fabricated_policy_immediate_escalation"] == True
    
    @patch.dict('os.environ', {
        'VERIFICATION_THRESHOLDS_PRICE_TOLERANCE': '0.3',
        'VERIFICATION_THRESHOLDS_ESCALATION_MAX_CRITICAL': '1',
        'VERIFICATION_THRESHOLDS_EARLY_TERMINATION': 'true'
    })
    def test_environment_variable_overrides(self):
        """Test environment variable overrides"""
        overrides = _get_thresholds_env_overrides()
        
        assert overrides["price_accuracy"]["pass_tolerance_percent"] == 0.3
        assert overrides["escalation"]["max_critical_issues_before_escalation"] == 1
        assert overrides["escalation"]["early_termination_enabled"] == True
    
    @patch('builtins.open', mock_open(read_data="""
price_accuracy:
  thresholds:
    minor_threshold_percent: 4.0
    major_threshold_percent: 12.0
    critical_threshold_percent: 25.0
  pass_criteria:
    tolerance_percent: 0.8
"""))
    @patch('pathlib.Path.exists', return_value=True)
    def test_yaml_file_loading(self, mock_exists):
        """Test loading configuration from YAML file"""
        with patch('yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = {
                "price_accuracy": {
                    "thresholds": {
                        "minor_threshold_percent": 4.0,
                        "major_threshold_percent": 12.0,
                        "critical_threshold_percent": 25.0
                    },
                    "pass_criteria": {
                        "tolerance_percent": 0.8
                    }
                }
            }
            
            config = enhanced_load_thresholds_config("test_config.yaml")
            
            assert config.price_accuracy.minor_threshold_percent == 4.0
            assert config.price_accuracy.major_threshold_percent == 12.0
            assert config.price_accuracy.pass_tolerance_percent == 0.8


class TestIntegrationScenarios:
    """Test integration scenarios with realistic use cases"""
    
    def test_complete_verification_scenario(self):
        """Test complete verification scenario with all thresholds"""
        config = get_default_thresholds_config()
        
        # Scenario: High price deviation + fabricated policy + low relevance
        price_deviation = 35.0  # Critical
        policy_fabricated = True  # Critical
        relevance_coverage = 0.2  # Critical
        
        # Check individual classifications
        price_severity = config.price_accuracy.classify_price_deviation(price_deviation)
        policy_severity = config.policy_authenticity.classify_policy_issue(
            is_fabricated=policy_fabricated,
            is_inaccurate=False,
            is_incomplete=False,
            policy_type="warranty",
            has_citation=False
        )
        relevance_severity = config.topic_relevance.classify_relevance_issue(
            coverage_ratio=relevance_coverage,
            empathy_score=0.3
        )
        
        assert price_severity == IssueSeverity.CRITICAL
        assert policy_severity == IssueSeverity.CRITICAL
        assert relevance_severity == IssueSeverity.CRITICAL
        
        # Check escalation logic
        should_escalate = config.escalation.should_escalate_immediately(
            critical_count=3,
            major_count=0,
            total_count=3,
            has_fabricated_policy=True,
            has_critical_price_deviation=True
        )
        assert should_escalate == True
        
        # Check early termination
        should_terminate = config.escalation.should_terminate_early(critical_count=3)
        assert should_terminate == True
    
    def test_borderline_cases(self):
        """Test borderline threshold cases"""
        config = get_default_thresholds_config()
        
        # Test exact threshold values
        assert config.price_accuracy.classify_price_deviation(5.0) == IssueSeverity.MAJOR
        assert config.price_accuracy.classify_price_deviation(4.99) == IssueSeverity.MINOR
        
        assert config.topic_relevance.classify_relevance_issue(0.5) == IssueSeverity.MAJOR
        assert config.topic_relevance.classify_relevance_issue(0.49) == IssueSeverity.CRITICAL
        
        # Test pass/fail boundaries
        assert config.price_accuracy.should_pass_price_check(1.0) == True
        assert config.price_accuracy.should_pass_price_check(1.01) == False
        
        assert config.topic_relevance.should_pass_relevance_check(0.7) == True
        assert config.topic_relevance.should_pass_relevance_check(0.69) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
