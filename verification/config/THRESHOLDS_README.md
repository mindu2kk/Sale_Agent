# Verification Thresholds Configuration

Comprehensive configuration system for critical/major/minor issue classification in the binary verification system.

**Supports Task 1.3.1**: Design verification thresholds config cho critical/major/minor issues

## Overview

The verification thresholds configuration system provides configurable thresholds for classifying issues as critical, major, or minor severity levels across three verification criteria:

1. **Price Accuracy** - Configurable price deviation thresholds
2. **Policy Authenticity** - Severity classification for policy issues  
3. **Topic Relevance** - Coverage and empathy thresholds
4. **Escalation Rules** - Issue count and early termination logic

## Architecture

```
verification/config/
├── thresholds_config.py     # Pydantic models and configuration logic
├── thresholds.yaml          # YAML configuration file
├── thresholds_example.py    # Usage examples
├── test_thresholds_config.py # Unit tests
└── THRESHOLDS_README.md     # This documentation
```

## Configuration Structure

### Price Accuracy Thresholds

```python
class PriceAccuracyThresholds:
    minor_threshold_percent: float = 5.0      # 0-5% = minor
    major_threshold_percent: float = 15.0     # 5-15% = major  
    critical_threshold_percent: float = 30.0  # >30% = critical
    pass_tolerance_percent: float = 1.0       # ±1% for PASS
    missing_price_severity: IssueSeverity = MAJOR
    currency_mismatch_severity: IssueSeverity = MINOR
```

**Business Rules:**
- Price deviations ≤1% → PASS
- Price deviations 1-5% → MINOR issue
- Price deviations 5-15% → MAJOR issue  
- Price deviations >30% → CRITICAL issue
- Missing prices → MAJOR severity by default
- High-value items (>10M VND) have stricter tolerance (0.5%)

### Policy Authenticity Thresholds

```python
class PolicyAuthenticityThresholds:
    fabricated_policy_severity: IssueSeverity = CRITICAL
    inaccurate_policy_severity: IssueSeverity = MAJOR
    incomplete_policy_severity: IssueSeverity = MINOR
    missing_citation_severity: IssueSeverity = MAJOR
    citation_required: bool = True
    source_verification_required: bool = True
```

**Business Rules:**
- Fabricated policies → Always CRITICAL
- Inaccurate policies → MAJOR (escalated to CRITICAL for warranty/return)
- Missing citations → MAJOR (escalated to CRITICAL for critical policy types)
- Incomplete policies → MINOR
- Policy type specific severity mapping (warranty=CRITICAL, service=MINOR)

### Topic Relevance Thresholds

```python
class TopicRelevanceThresholds:
    minor_coverage_threshold: float = 0.8     # 80%+ coverage = minor issues
    major_coverage_threshold: float = 0.5     # 50-80% coverage = major
    critical_coverage_threshold: float = 0.3  # <30% coverage = critical
    pass_coverage_threshold: float = 0.7      # 70% minimum for PASS
    empathy_required: bool = True
    empathy_score_threshold: float = 0.5
```

**Business Rules:**
- Coverage ≥70% + empathy ≥0.5 → PASS
- Coverage 80%+ → Minor issues only
- Coverage 50-80% → Major issues
- Coverage <30% → Critical issues
- Completely irrelevant (<10% coverage) → Always CRITICAL
- High off-topic content (>50%) → CRITICAL

### Escalation Thresholds

```python
class EscalationThresholds:
    max_critical_issues_before_escalation: int = 2
    max_major_issues_before_escalation: int = 5
    fabricated_policy_immediate_escalation: bool = True
    critical_price_deviation_escalation: bool = True
    early_termination_enabled: bool = True
    stop_on_multiple_critical: bool = True
    multiple_critical_threshold: int = 3
```

**Business Rules:**
- ≥3 critical issues → Immediate escalation + early termination
- Fabricated policy → Immediate escalation
- Critical price deviation (>30%) → Immediate escalation
- ≥5 major issues → Escalation
- Retry limits based on severity: Critical=1, Major=3, Minor=5

## Usage Examples

### Basic Usage

```python
from verification.config.thresholds_config import get_default_thresholds_config

# Load default configuration
config = get_default_thresholds_config()

# Classify price deviation
deviation = 8.0  # 8% price deviation
severity = config.price_accuracy.classify_price_deviation(deviation)
should_pass = config.price_accuracy.should_pass_price_check(deviation)
print(f"{deviation}% deviation → {severity.value} severity, Pass: {should_pass}")
# Output: 8.0% deviation → major severity, Pass: False
```

### Policy Issue Classification

```python
# Classify policy authenticity issue
severity = config.policy_authenticity.classify_policy_issue(
    is_fabricated=False,
    is_inaccurate=True,
    is_incomplete=False,
    policy_type="warranty",
    has_citation=False
)
print(f"Inaccurate warranty policy without citation → {severity.value}")
# Output: Inaccurate warranty policy without citation → critical
```

### Escalation Logic

```python
# Check escalation requirements
should_escalate = config.escalation.should_escalate_immediately(
    critical_count=2,
    major_count=1,
    total_count=3,
    has_fabricated_policy=True
)
print(f"Should escalate: {should_escalate}")
# Output: Should escalate: True (due to fabricated policy)
```

### Environment-Specific Configuration

```python
# Apply environment overrides
dev_config = config.get_environment_config("development")
prod_config = config.get_environment_config("production")

print(f"Dev tolerance: {dev_config.price_accuracy.pass_tolerance_percent}%")
print(f"Prod tolerance: {prod_config.price_accuracy.pass_tolerance_percent}%")
# Output: Dev tolerance: 2.0%
#         Prod tolerance: 0.5%
```

## YAML Configuration

### Loading from YAML

```python
from verification.config.thresholds_config import enhanced_load_thresholds_config

# Load from YAML file
config = enhanced_load_thresholds_config("verification/config/thresholds.yaml")
```

### YAML Structure

```yaml
price_accuracy:
  thresholds:
    minor_threshold_percent: 5.0
    major_threshold_percent: 15.0
    critical_threshold_percent: 30.0
  pass_criteria:
    tolerance_percent: 1.0
    allow_promotional_pricing: true

policy_authenticity:
  severity_rules:
    fabricated_policy: "critical"
    inaccurate_policy: "major"
    missing_citation: "major"
  citation_requirements:
    citation_required: true
    source_verification_required: true

topic_relevance:
  coverage_thresholds:
    minor_coverage_threshold: 0.8
    major_coverage_threshold: 0.5
    critical_coverage_threshold: 0.3
  pass_criteria:
    min_coverage_ratio: 0.7
  empathy_requirements:
    empathy_required: true
    min_empathy_score: 0.5

escalation:
  count_thresholds:
    max_critical_before_escalation: 2
    max_major_before_escalation: 5
  immediate_triggers:
    fabricated_policy_detected: true
    critical_price_deviation: true
  early_termination:
    enabled: true
    stop_on_multiple_critical: true
    multiple_critical_threshold: 3
```

## Environment Variables

Override configuration using environment variables:

```bash
# Price accuracy settings
export VERIFICATION_THRESHOLDS_PRICE_TOLERANCE=0.5
export VERIFICATION_THRESHOLDS_PRICE_CRITICAL=25.0

# Policy settings  
export VERIFICATION_THRESHOLDS_POLICY_CITATION_REQUIRED=true

# Relevance settings
export VERIFICATION_THRESHOLDS_RELEVANCE_COVERAGE=0.75

# Escalation settings
export VERIFICATION_THRESHOLDS_ESCALATION_MAX_CRITICAL=1
export VERIFICATION_THRESHOLDS_EARLY_TERMINATION=true
```

## Environment-Specific Overrides

### Development Environment
- More lenient price tolerance (2.0% vs 1.0%)
- Higher critical thresholds (50% vs 30%)
- More critical issues allowed before escalation (5 vs 2)
- Early termination disabled for comprehensive testing
- Longer processing timeouts for debugging

### Production Environment  
- Stricter price tolerance (0.5% vs 1.0%)
- Immediate escalation on first critical issue
- Very strict tolerance for high-value items (0.25%)
- Faster timeouts for performance
- Enhanced monitoring and alerting

### Testing Environment
- Early termination disabled for comprehensive testing
- No retries to speed up test execution
- Alerting thresholds disabled
- Minimal resource limits

## Integration with Verification System

### VerificationAgent Integration

```python
from verification.config.thresholds_config import get_thresholds_config

class VerificationAgent:
    def __init__(self):
        self.thresholds = get_thresholds_config()
    
    def verify_draft(self, state: WorkflowState) -> VerificationResult:
        # Use thresholds for classification
        price_issues = self._check_price_accuracy(state, self.thresholds.price_accuracy)
        policy_issues = self._check_policy_authenticity(state, self.thresholds.policy_authenticity)
        relevance_issues = self._check_topic_relevance(state, self.thresholds.topic_relevance)
        
        # Apply escalation logic
        should_escalate = self.thresholds.escalation.should_escalate_immediately(
            critical_count=len([i for i in price_issues + policy_issues + relevance_issues 
                              if i.severity == IssueSeverity.CRITICAL]),
            major_count=len([i for i in price_issues + policy_issues + relevance_issues 
                           if i.severity == IssueSeverity.MAJOR]),
            total_count=len(price_issues + policy_issues + relevance_issues)
        )
```

### StateGraph Workflow Integration

```python
class VerificationWorkflow:
    def __init__(self):
        self.thresholds = get_thresholds_config()
    
    def route_after_verification(self, state: WorkflowState) -> str:
        verification_result = state["verification_result"]
        
        # Use thresholds for routing decisions
        if verification_result.is_approved:
            return "approved"
        
        # Check escalation thresholds
        if self.thresholds.escalation.should_escalate_immediately(
            critical_count=verification_result.criteria.critical_issues_count,
            major_count=verification_result.criteria.get_major_issues_count(),
            total_count=len(verification_result.criteria.price_issues + 
                          verification_result.criteria.policy_issues + 
                          verification_result.criteria.relevance_issues)
        ):
            return "escalation"
        
        return "correction"
```

## Performance Considerations

### Caching
- Configuration loaded once and cached globally
- Threshold calculations are lightweight (O(1))
- YAML parsing only on startup or reload
- Environment variable overrides cached

### Memory Usage
- Pydantic models are memory efficient
- Configuration validation at load time
- Lazy loading of environment-specific configs
- Automatic cleanup of unused configurations

### Scalability
- Thread-safe configuration access
- No database dependencies
- Fast threshold lookups
- Configurable resource limits

## Monitoring and Alerting

### Threshold-Based Alerts
- Critical issue rate >10% → Alert
- Escalation rate >5% → Alert  
- Fabrication detection >2% → Alert
- Verification failure rate >30% → Alert

### Performance Monitoring
- Average verification time >15s → Alert
- Timeout rate >5% → Alert
- Cache hit rate <70% → Alert
- Retry rate >20% → Alert

### Quality Metrics
- Accuracy rate <95% → Alert
- False positive rate >5% → Alert
- False negative rate >2% → Alert

## Testing

### Unit Tests
```bash
# Run threshold configuration tests
pytest verification/tests/test_thresholds_config.py -v

# Run with coverage
pytest verification/tests/test_thresholds_config.py --cov=verification.config.thresholds_config
```

### Integration Tests
```bash
# Test with real YAML configuration
python verification/config/thresholds_example.py

# Test environment overrides
VERIFICATION_THRESHOLDS_PRICE_TOLERANCE=0.3 python verification/config/thresholds_example.py
```

### Property-Based Tests
- Threshold consistency properties
- Configuration validation properties  
- Escalation logic properties
- Environment override properties

## Best Practices

### Configuration Management
1. **Use YAML for complex configurations** - More readable than JSON
2. **Validate on load** - Catch configuration errors early
3. **Environment-specific overrides** - Different settings per environment
4. **Version control configurations** - Track configuration changes
5. **Document business rules** - Clear reasoning for thresholds

### Threshold Setting
1. **Start conservative** - Begin with stricter thresholds
2. **Monitor and adjust** - Use metrics to tune thresholds
3. **Business alignment** - Ensure thresholds match business requirements
4. **Gradual changes** - Avoid sudden threshold changes
5. **A/B testing** - Test threshold changes with subset of traffic

### Performance Optimization
1. **Cache configurations** - Avoid repeated loading
2. **Lazy evaluation** - Only compute when needed
3. **Batch operations** - Process multiple items together
4. **Resource limits** - Prevent resource exhaustion
5. **Monitoring** - Track performance metrics

## Troubleshooting

### Common Issues

**Configuration not loading:**
```python
# Check file path and permissions
import os
from pathlib import Path

config_path = "verification/config/thresholds.yaml"
if not Path(config_path).exists():
    print(f"Config file not found: {config_path}")

# Check YAML syntax
import yaml
try:
    with open(config_path) as f:
        yaml.safe_load(f)
    print("YAML syntax is valid")
except yaml.YAMLError as e:
    print(f"YAML syntax error: {e}")
```

**Validation errors:**
```python
# Check threshold consistency
config = get_default_thresholds_config()
warnings = config.validate_configuration()
if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")
```

**Environment overrides not working:**
```python
# Check environment variables
import os
env_vars = {k: v for k, v in os.environ.items() 
           if k.startswith('VERIFICATION_THRESHOLDS_')}
print("Environment overrides:", env_vars)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Load configuration with debug info
config = enhanced_load_thresholds_config("verification/config/thresholds.yaml")
```

## Future Enhancements

### Planned Features
1. **Dynamic threshold adjustment** - ML-based threshold optimization
2. **A/B testing support** - Compare different threshold configurations
3. **Real-time monitoring** - Live threshold performance dashboards
4. **Configuration versioning** - Track and rollback configuration changes
5. **Business rule engine** - More complex threshold logic

### Integration Opportunities
1. **Metrics collection** - Detailed threshold performance metrics
2. **Alerting systems** - Integration with monitoring platforms
3. **Configuration management** - Integration with config management tools
4. **Machine learning** - Automated threshold optimization
5. **Business intelligence** - Threshold impact analysis

## References

- [Requirements Document](../requirements.md) - Business requirements for verification thresholds
- [Design Document](../design.md) - Technical design for binary verification system
- [Verification Models](../models/verification.py) - Data models for verification results
- [Configuration System](config.py) - Main verification configuration system
- [Task List](../tasks.md) - Implementation tasks and progress tracking