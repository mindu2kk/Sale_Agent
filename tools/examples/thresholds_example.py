"""
Example Usage: Verification Thresholds Configuration

Demonstrates how to use the verification thresholds configuration system
for critical/major/minor issue classification in the binary verification system.

Supports Task 1.3.1: Design verification thresholds config cho critical/major/minor issues
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.verification.config.thresholds_config import (
    VerificationThresholdsConfig,
    PriceAccuracyThresholds,
    PolicyAuthenticityThresholds,
    TopicRelevanceThresholds,
    EscalationThresholds,
    IssueSeverity,
    get_default_thresholds_config,
    enhanced_load_thresholds_config,
    save_thresholds_config
)


def example_basic_usage():
    """Basic usage example - loading and using default configuration"""
    print("=== Basic Thresholds Configuration Usage ===")

    # Load default configuration
    config = get_default_thresholds_config()

    # Access price accuracy thresholds
    price_thresholds = config.price_accuracy
    print(f"Price tolerance for PASS: ±{price_thresholds.pass_tolerance_percent}%")
    print(f"Critical price deviation threshold: >{price_thresholds.critical_threshold_percent}%")

    # Test price deviation classification
    test_deviations = [0.5, 3.0, 10.0, 25.0, 35.0]
    for deviation in test_deviations:
        severity = price_thresholds.classify_price_deviation(deviation)
        should_pass = price_thresholds.should_pass_price_check(deviation)
        print(f"  {deviation}% deviation → {severity.value} severity, Pass: {should_pass}")

    print()


def example_policy_classification():
    """Example of policy authenticity issue classification"""
    print("=== Policy Authenticity Classification ===")

    config = get_default_thresholds_config()
    policy_thresholds = config.policy_authenticity

    # Test different policy scenarios
    test_scenarios = [
        {
            "name": "Fabricated warranty policy",
            "is_fabricated": True,
            "is_inaccurate": False,
            "is_incomplete": False,
            "policy_type": "warranty",
            "has_citation": False
        },
        {
            "name": "Inaccurate return policy",
            "is_fabricated": False,
            "is_inaccurate": True,
            "is_incomplete": False,
            "policy_type": "return",
            "has_citation": True
        },
        {
            "name": "Incomplete service policy",
            "is_fabricated": False,
            "is_inaccurate": False,
            "is_incomplete": True,
            "policy_type": "service",
            "has_citation": True
        },
        {
            "name": "Missing citation for warranty",
            "is_fabricated": False,
            "is_inaccurate": False,
            "is_incomplete": False,
            "policy_type": "warranty",
            "has_citation": False
        }
    ]

    for scenario in test_scenarios:
        severity = policy_thresholds.classify_policy_issue(
            is_fabricated=scenario["is_fabricated"],
            is_inaccurate=scenario["is_inaccurate"],
            is_incomplete=scenario["is_incomplete"],
            policy_type=scenario["policy_type"],
            has_citation=scenario["has_citation"]
        )

        should_pass = policy_thresholds.should_pass_policy_check(
            is_fabricated=scenario["is_fabricated"],
            is_inaccurate=scenario["is_inaccurate"],
            has_required_citation=scenario["has_citation"]
        )

        print(f"  {scenario['name']} → {severity.value} severity, Pass: {should_pass}")

    print()


def example_relevance_assessment():
    """Example of topic relevance issue classification"""
    print("=== Topic Relevance Assessment ===")

    config = get_default_thresholds_config()
    relevance_thresholds = config.topic_relevance

    # Test different coverage scenarios
    test_scenarios = [
        {"coverage": 0.95, "empathy": 0.8, "off_topic": 0.0, "description": "Excellent response"},
        {"coverage": 0.75, "empathy": 0.6, "off_topic": 0.1, "description": "Good response"},
        {"coverage": 0.60, "empathy": 0.4, "off_topic": 0.2, "description": "Adequate response"},
        {"coverage": 0.40, "empathy": 0.3, "off_topic": 0.3, "description": "Poor response"},
        {"coverage": 0.15, "empathy": 0.1, "off_topic": 0.6, "description": "Very poor response"},
        {"coverage": 0.05, "empathy": 0.0, "off_topic": 0.8, "description": "Completely irrelevant"}
    ]

    for scenario in test_scenarios:
        severity = relevance_thresholds.classify_relevance_issue(
            coverage_ratio=scenario["coverage"],
            empathy_score=scenario["empathy"],
            off_topic_ratio=scenario["off_topic"]
        )

        should_pass = relevance_thresholds.should_pass_relevance_check(
            coverage_ratio=scenario["coverage"],
            empathy_score=scenario["empathy"]
        )

        print(f"  {scenario['description']} ({scenario['coverage']:.0%} coverage) → {severity.value} severity, Pass: {should_pass}")

    print()


def example_escalation_logic():
    """Example of escalation threshold logic"""
    print("=== Escalation Logic Examples ===")

    config = get_default_thresholds_config()
    escalation_thresholds = config.escalation

    # Test different issue scenarios
    test_scenarios = [
        {
            "name": "Single critical price deviation",
            "critical_count": 1,
            "major_count": 0,
            "total_count": 1,
            "has_fabricated_policy": False,
            "has_critical_price_deviation": True,
            "is_completely_irrelevant": False
        },
        {
            "name": "Multiple critical issues",
            "critical_count": 3,
            "major_count": 2,
            "total_count": 5,
            "has_fabricated_policy": False,
            "has_critical_price_deviation": False,
            "is_completely_irrelevant": False
        },
        {
            "name": "Fabricated policy detected",
            "critical_count": 1,
            "major_count": 0,
            "total_count": 1,
            "has_fabricated_policy": True,
            "has_critical_price_deviation": False,
            "is_completely_irrelevant": False
        },
        {
            "name": "Many major issues",
            "critical_count": 0,
            "major_count": 6,
            "total_count": 8,
            "has_fabricated_policy": False,
            "has_critical_price_deviation": False,
            "is_completely_irrelevant": False
        },
        {
            "name": "Only minor issues",
            "critical_count": 0,
            "major_count": 0,
            "total_count": 3,
            "has_fabricated_policy": False,
            "has_critical_price_deviation": False,
            "is_completely_irrelevant": False
        }
    ]

    for scenario in test_scenarios:
        should_escalate = escalation_thresholds.should_escalate_immediately(
            critical_count=scenario["critical_count"],
            major_count=scenario["major_count"],
            total_count=scenario["total_count"],
            has_fabricated_policy=scenario["has_fabricated_policy"],
            has_critical_price_deviation=scenario["has_critical_price_deviation"],
            is_completely_irrelevant=scenario["is_completely_irrelevant"]
        )

        should_terminate = escalation_thresholds.should_terminate_early(
            critical_count=scenario["critical_count"]
        )

        max_retries = escalation_thresholds.get_max_retries_for_severity(
            critical_count=scenario["critical_count"],
            major_count=scenario["major_count"],
            minor_count=scenario["total_count"] - scenario["critical_count"] - scenario["major_count"]
        )

        print(f"  {scenario['name']}:")
        print(f"    Escalate: {should_escalate}, Terminate: {should_terminate}, Max retries: {max_retries}")

    print()


def example_environment_overrides():
    """Example of environment-specific configuration overrides"""
    print("=== Environment-Specific Overrides ===")

    config = get_default_thresholds_config()

    # Show default configuration
    print("Default configuration:")
    print(f"  Price tolerance: {config.price_accuracy.pass_tolerance_percent}%")
    print(f"  Max critical issues before escalation: {config.escalation.max_critical_issues_before_escalation}")

    # Apply development environment overrides
    dev_config = config.get_environment_config("development")
    print("\nDevelopment environment:")
    print(f"  Price tolerance: {dev_config.price_accuracy.pass_tolerance_percent}%")
    print(f"  Max critical issues before escalation: {dev_config.escalation.max_critical_issues_before_escalation}")

    # Apply production environment overrides
    prod_config = config.get_environment_config("production")
    print("\nProduction environment:")
    print(f"  Price tolerance: {prod_config.price_accuracy.pass_tolerance_percent}%")
    print(f"  Max critical issues before escalation: {prod_config.escalation.max_critical_issues_before_escalation}")

    print()


def example_configuration_validation():
    """Example of configuration validation"""
    print("=== Configuration Validation ===")

    config = get_default_thresholds_config()

    # Validate configuration
    warnings = config.validate_configuration()

    if warnings:
        print("Configuration warnings found:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Configuration validation passed - no warnings")

    print()


def example_custom_configuration():
    """Example of creating custom configuration"""
    print("=== Custom Configuration Example ===")

    # Create custom thresholds
    custom_price_thresholds = PriceAccuracyThresholds(
        minor_threshold_percent=3.0,      # Stricter minor threshold
        major_threshold_percent=10.0,     # Stricter major threshold
        critical_threshold_percent=20.0,  # Stricter critical threshold
        pass_tolerance_percent=0.5,       # Very strict pass tolerance
        missing_price_severity=IssueSeverity.CRITICAL  # Missing price is critical
    )

    custom_escalation_thresholds = EscalationThresholds(
        max_critical_issues_before_escalation=1,  # Escalate on first critical
        early_termination_enabled=True,
        stop_on_first_critical=True,              # Stop immediately on critical
        max_retries_with_critical=0               # No retries with critical issues
    )

    # Create custom configuration
    custom_config = VerificationThresholdsConfig(
        price_accuracy=custom_price_thresholds,
        escalation=custom_escalation_thresholds
    )

    print("Custom configuration created:")
    print(f"  Price pass tolerance: {custom_config.price_accuracy.pass_tolerance_percent}%")
    print(f"  Critical price threshold: {custom_config.price_accuracy.critical_threshold_percent}%")
    print(f"  Stop on first critical: {custom_config.escalation.stop_on_first_critical}")
    print(f"  Max retries with critical: {custom_config.escalation.max_retries_with_critical}")

    # Test price classification with custom thresholds
    test_deviation = 15.0
    severity = custom_config.price_accuracy.classify_price_deviation(test_deviation)
    print(f"  {test_deviation}% deviation → {severity.value} (vs 'major' in default config)")

    print()


def example_yaml_configuration():
    """Example of loading configuration from YAML file"""
    print("=== YAML Configuration Loading ===")

    try:
        # Load configuration from YAML file
        config = enhanced_load_thresholds_config("backend/verification/config/thresholds.yaml")

        print("Successfully loaded configuration from YAML:")
        print(f"  Price accuracy thresholds: {config.price_accuracy.minor_threshold_percent}% / {config.price_accuracy.major_threshold_percent}% / {config.price_accuracy.critical_threshold_percent}%")
        print(f"  Policy citation required: {config.policy_authenticity.citation_required}")
        print(f"  Topic relevance pass threshold: {config.topic_relevance.pass_coverage_threshold}")
        print(f"  Early termination enabled: {config.escalation.early_termination_enabled}")

        # Show verification weights
        weights = config.verification_weights
        print(f"  Verification weights: Price {weights['price_accuracy']:.1%}, Policy {weights['policy_authenticity']:.1%}, Relevance {weights['topic_relevance']:.1%}")

    except Exception as e:
        print(f"Failed to load YAML configuration: {e}")
        print("Using default configuration instead")

    print()


def example_save_configuration():
    """Example of saving configuration to file"""
    print("=== Save Configuration Example ===")

    # Create a custom configuration
    config = get_default_thresholds_config()

    # Modify some settings
    config.price_accuracy.pass_tolerance_percent = 0.8
    config.escalation.max_critical_issues_before_escalation = 1

    try:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "custom_thresholds.yaml"
            save_thresholds_config(config, str(output_path))
            loaded_config = enhanced_load_thresholds_config(str(output_path))
        print(f"Verified: Price tolerance = {loaded_config.price_accuracy.pass_tolerance_percent}%")

    except Exception as e:
        print(f"Failed to save configuration: {e}")

    print()


def main():
    """Run all examples"""
    print("Verification Thresholds Configuration Examples")
    print("=" * 50)
    print()

    example_basic_usage()
    example_policy_classification()
    example_relevance_assessment()
    example_escalation_logic()
    example_environment_overrides()
    example_configuration_validation()
    example_custom_configuration()
    example_yaml_configuration()
    example_save_configuration()

    print("All examples completed!")


if __name__ == "__main__":
    main()
