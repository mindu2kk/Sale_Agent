# Verification Agent Configuration System

This directory contains comprehensive configuration files for the Verification Agent's binary verification system and LangGraph StateGraph workflow.

## Configuration Files Overview

### Core Configuration Files

1. **`verification_config.yaml`** - Main binary verification settings
   - Price accuracy thresholds and tolerance settings
   - Policy authenticity validation rules
   - Topic relevance coverage requirements
   - Critical issue handling and escalation rules
   - Performance optimization settings

2. **`workflow_config.yaml`** - StateGraph workflow orchestration settings
   - Node configuration (research, verification, correction, escalation)
   - Conditional edge routing logic
   - Workflow state management
   - Error handling and recovery strategies
   - Performance and scaling settings

3. **`prompts.yaml`** - Verification prompt templates
   - Master verification prompt with binary criteria
   - Individual check prompts (price, policy, relevance)
   - Correction feedback generation prompts
   - Escalation notification templates
   - Environment-specific prompt overrides

### Environment-Specific Configurations

4. **`environments/development.yaml`** - Development environment overrides
   - Relaxed thresholds for testing and debugging
   - Enhanced logging and debugging features
   - Longer timeouts for development workflow
   - Test data support and mock configurations

5. **`environments/production.yaml`** - Production environment overrides
   - Strict verification thresholds for business use
   - Optimized performance settings
   - Enhanced security and monitoring
   - Auto-scaling and high availability settings

6. **`environments/test.yaml`** - Testing environment overrides
   - Minimal configuration for unit tests and CI/CD
   - Disabled persistence and caching
   - Fast execution with minimal logging
   - Mock services and in-memory databases

### Utility Files

7. **`config_loader.py`** - Configuration loading and management utility
   - YAML configuration file loading with validation
   - Environment-specific override merging
   - Environment variable override support
   - Configuration caching and validation

8. **`example_usage.py`** - Example usage and demonstration script
   - Shows how to load configurations for different environments
   - Demonstrates configuration validation and caching
   - Examples of accessing specific configuration values

## Binary Verification System

The configuration supports a **binary PASS/FAIL verification approach** instead of traditional 0-10 scoring:

### Price Accuracy Verification
- **PASS**: Price deviation ≤ tolerance threshold (default ±1%)
- **FAIL**: Price deviation > tolerance OR missing price when required
- Configurable severity levels (minor, major, critical)
- Support for promotional pricing and currency conversion

### Policy Authenticity Verification
- **PASS**: All policies verified with proper citations from official documents
- **FAIL**: Any fabricated, inaccurate, or uncited policies detected
- Forbidden phrase detection for fabricated content
- Required source verification against internal policy database

### Topic Relevance Verification
- **PASS**: Coverage ratio ≥ threshold (default 70%) AND addresses main concern
- **FAIL**: Coverage ratio < threshold OR misses main objection point
- Intent detection for different objection types
- Empathy detection and bonus scoring

## Configuration Usage

### Basic Usage

```python
from verification.config.config_loader import load_config_for_environment

# Load all configurations for development environment
configs = load_config_for_environment('development')

# Access specific settings
verification_config = configs['verification']
price_tolerance = verification_config['price_accuracy']['tolerance_percent']
max_retries = verification_config['retry_settings']['max_retries']
```

### Environment-Specific Loading

```python
from verification.config.config_loader import get_config_loader

loader = get_config_loader()

# Load for specific environments
dev_config = loader.load_verification_config('development')
prod_config = loader.load_verification_config('production')
test_config = loader.load_verification_config('test')
```

### Environment Variable Overrides

Set environment variables to override configuration values:

```bash
export VERIFICATION_PRICE_TOLERANCE=2.0
export VERIFICATION_MAX_RETRIES=5
export VERIFICATION_LOG_LEVEL=DEBUG
export VERIFICATION_ENVIRONMENT=production
```

Supported environment variables:
- `VERIFICATION_PRICE_TOLERANCE` - Price accuracy tolerance percentage
- `VERIFICATION_MAX_RETRIES` - Maximum retry attempts
- `VERIFICATION_PARALLEL_CHECKS` - Enable/disable parallel verification
- `VERIFICATION_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `VERIFICATION_CACHE_ENABLED` - Enable/disable caching
- `VERIFICATION_TIMEOUT` - Verification timeout in seconds
- `VERIFICATION_ENVIRONMENT` - Environment name (dev, prod, test)

## Configuration Structure

### Verification Configuration Structure

```yaml
price_accuracy:
  tolerance_percent: 1.0          # ±1% default tolerance
  critical_threshold_percent: 30.0 # >30% deviation = critical
  severity_thresholds:
    minor: 5.0                    # 0-5% = minor issue
    major: 15.0                   # 5-15% = major issue
    critical: 30.0                # >30% = critical issue

policy_authenticity:
  citation_required: true
  forbidden_phrases:
    vietnamese: ["tự bịa", "không có trong hệ thống"]
    english: ["fabricated", "made up"]

topic_relevance:
  min_coverage_ratio: 0.7         # 70% minimum coverage
  empathy_detection:
    enabled: true
    bonus_points: 0.1

critical_issues:
  escalation_triggers:
    price_deviation_over_30_percent: true
    fabricated_policy_detected: true
    multiple_critical_issues: true
  early_termination:
    enabled: true
    stop_on_multiple_critical: true

retry_settings:
  max_retries: 3
  retry_backoff_seconds: 1.0
  exponential_backoff: true
```

### Workflow Configuration Structure

```yaml
workflow:
  execution:
    max_execution_time_seconds: 300
    enable_persistence: true
    enable_checkpoints: true

nodes:
  research_node:
    timeout_seconds: 60
    retry_attempts: 2
  verification_node:
    timeout_seconds: 30
    retry_attempts: 1
    settings:
      parallel_checks: true
      early_termination_enabled: true
  correction_node:
    timeout_seconds: 45
    retry_attempts: 1

edges:
  verification_routing:
    conditions:
      approved:
        condition: "verification_result.is_approved == true"
        target: "END"
      critical_escalation:
        condition: "verification_result.critical_issues_count >= 3"
        target: "escalation"
        priority: 1
```

## Environment Differences

| Setting | Development | Production | Test |
|---------|-------------|------------|------|
| Price Tolerance | ±2.0% | ±0.5% | ±5.0% |
| Max Retries | 5 | 2 | 1 |
| Log Level | DEBUG | WARNING | ERROR |
| Timeout | 60s | 15s | 10s |
| Caching | Enabled | Enabled | Disabled |
| Persistence | Enabled | Enabled | Disabled |

## Configuration Validation

The configuration system includes comprehensive validation:

```python
from verification.config.config_loader import validate_all_configs

# Validate all loaded configurations
configs = load_config_for_environment('production')
is_valid = validate_all_configs(configs)

if not is_valid:
    print("Configuration validation failed!")
```

Validation checks include:
- Required sections and fields presence
- Value type and range validation
- Cross-field consistency checks
- Environment-specific constraint validation

## Performance Considerations

### Configuration Loading Performance
- **Caching**: Configurations are cached after first load
- **Lazy Loading**: Only requested configuration types are loaded
- **Validation**: Validation is performed once during loading
- **Memory Usage**: Configurations are shared across instances

### Runtime Performance Impact
- **Environment Variables**: Checked only during configuration loading
- **File Watching**: Not implemented (configurations are static after load)
- **Hot Reloading**: Available via `reload_config()` method

## Security Considerations

### Configuration Security
- **Sensitive Data**: No sensitive data stored in configuration files
- **Environment Variables**: Used for runtime overrides only
- **File Permissions**: Configuration files should be read-only in production
- **Validation**: All inputs are validated to prevent injection attacks

### Production Security
- **API Keys**: Stored separately from configuration files
- **Database Credentials**: Managed via environment variables or secrets management
- **Rate Limiting**: Configured per environment
- **Input Sanitization**: Enabled in production configuration

## Troubleshooting

### Common Issues

1. **Configuration File Not Found**
   ```
   ConfigurationError: Configuration file not found: verification/config/verification_config.yaml
   ```
   - Ensure configuration files exist in the correct directory
   - Check file permissions and accessibility

2. **Invalid YAML Syntax**
   ```
   ConfigurationError: Invalid YAML in verification_config.yaml: ...
   ```
   - Validate YAML syntax using online validators
   - Check indentation and special characters

3. **Environment Variable Override Not Working**
   - Verify environment variable names match expected format
   - Check variable types (string, int, float, bool)
   - Ensure variables are set before loading configuration

4. **Validation Errors**
   ```
   ConfigurationError: Missing required section: price_accuracy
   ```
   - Check configuration file completeness
   - Verify all required sections and fields are present
   - Review environment-specific overrides

### Debug Mode

Enable debug logging to troubleshoot configuration issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Configuration loading will now show detailed debug information
configs = load_config_for_environment('development')
```

## Best Practices

### Configuration Management
1. **Version Control**: Keep all configuration files in version control
2. **Environment Separation**: Use separate files for each environment
3. **Documentation**: Document all configuration changes
4. **Validation**: Always validate configurations before deployment

### Performance Optimization
1. **Caching**: Use configuration caching in production
2. **Lazy Loading**: Load only required configuration sections
3. **Environment Variables**: Use for runtime overrides only
4. **File Size**: Keep configuration files reasonably sized

### Security Best Practices
1. **No Secrets**: Never store secrets in configuration files
2. **Validation**: Always validate configuration inputs
3. **Permissions**: Set appropriate file permissions
4. **Monitoring**: Monitor configuration changes in production

## Migration Guide

### From JSON to YAML Configuration

If migrating from JSON configuration files:

1. Convert JSON structure to YAML format
2. Update configuration loading code to use new loader
3. Add environment-specific overrides
4. Implement configuration validation
5. Test all environments thoroughly

### Adding New Configuration Options

1. Add new fields to appropriate YAML files
2. Update validation logic in `config_loader.py`
3. Add environment variable mappings if needed
4. Update documentation and examples
5. Test with all environments