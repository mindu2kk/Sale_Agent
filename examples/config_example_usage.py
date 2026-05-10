#!/usr/bin/env python3
"""
Example Usage of Verification Agent Configuration System

This script demonstrates how to load and use the configuration files
for different environments and scenarios.
"""

import os
import sys
from pathlib import Path

# Add the verification module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verification.config.config_loader import (
    get_config_loader,
    load_config_for_environment,
    validate_all_configs,
    ConfigurationError
)


def example_basic_usage():
    """Example: Basic configuration loading"""
    print("=== Basic Configuration Loading ===")
    
    try:
        # Load configuration for development environment
        configs = load_config_for_environment('development')
        
        print(f"Loaded {len(configs)} configuration sections:")
        for section_name in configs.keys():
            print(f"  - {section_name}")
        
        # Access specific configuration values
        verification_config = configs['verification']
        price_tolerance = verification_config['price_accuracy']['tolerance_percent']
        max_retries = verification_config['retry_settings']['max_retries']
        
        print(f"\nDevelopment settings:")
        print(f"  Price tolerance: ±{price_tolerance}%")
        print(f"  Max retries: {max_retries}")
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")


def example_environment_comparison():
    """Example: Compare settings across environments"""
    print("\n=== Environment Comparison ===")
    
    environments = ['development', 'production', 'test']
    
    for env in environments:
        try:
            configs = load_config_for_environment(env)
            verification_config = configs['verification']
            
            price_tolerance = verification_config['price_accuracy']['tolerance_percent']
            log_level = verification_config['logging']['level']
            
            print(f"\n{env.upper()} environment:")
            print(f"  Price tolerance: ±{price_tolerance}%")
            print(f"  Log level: {log_level}")
            
        except ConfigurationError as e:
            print(f"  Error loading {env}: {e}")


def example_workflow_configuration():
    """Example: Working with workflow configuration"""
    print("\n=== Workflow Configuration ===")
    
    try:
        loader = get_config_loader()
        workflow_config = loader.load_workflow_config('production')
        
        # Access workflow settings
        max_execution_time = workflow_config['workflow']['execution']['max_execution_time_seconds']
        research_timeout = workflow_config['nodes']['research_node']['timeout_seconds']
        verification_timeout = workflow_config['nodes']['verification_node']['timeout_seconds']
        
        print(f"Production workflow timeouts:")
        print(f"  Max execution time: {max_execution_time}s")
        print(f"  Research node timeout: {research_timeout}s")
        print(f"  Verification node timeout: {verification_timeout}s")
        
        # Check if parallel verification is enabled
        parallel_checks = workflow_config['nodes']['verification_node']['settings']['parallel_checks']
        print(f"  Parallel verification: {parallel_checks}")
        
    except ConfigurationError as e:
        print(f"Workflow configuration error: {e}")


def example_prompts_configuration():
    """Example: Working with prompts configuration"""
    print("\n=== Prompts Configuration ===")
    
    try:
        loader = get_config_loader()
        prompts_config = loader.load_prompts_config('development')
        
        # Access prompt templates
        master_prompt = prompts_config['verification_prompts']['master_verification']
        template = master_prompt['template']
        variables = master_prompt['variables']
        
        print(f"Master verification prompt:")
        print(f"  Template length: {len(template)} characters")
        print(f"  Required variables: {', '.join(variables)}")
        
        # Check prompt settings
        prompt_settings = prompts_config['prompt_settings']
        max_length = prompt_settings['global']['max_prompt_length']
        use_json = prompt_settings['output_format']['prefer_json']
        
        print(f"\nPrompt settings:")
        print(f"  Max prompt length: {max_length}")
        print(f"  Prefer JSON output: {use_json}")
        
    except ConfigurationError as e:
        print(f"Prompts configuration error: {e}")


def example_environment_variables():
    """Example: Using environment variable overrides"""
    print("\n=== Environment Variable Overrides ===")
    
    # Set some environment variables
    os.environ['VERIFICATION_PRICE_TOLERANCE'] = '2.5'
    os.environ['VERIFICATION_MAX_RETRIES'] = '5'
    os.environ['VERIFICATION_LOG_LEVEL'] = 'DEBUG'
    
    try:
        # Load configuration with environment variable overrides
        configs = load_config_for_environment('development')
        verification_config = configs['verification']
        
        # Check if environment variables were applied
        price_tolerance = verification_config['price_accuracy']['tolerance_percent']
        max_retries = verification_config['retry_settings']['max_retries']
        log_level = verification_config['logging']['level']
        
        print(f"Configuration with environment overrides:")
        print(f"  Price tolerance: ±{price_tolerance}% (from env var)")
        print(f"  Max retries: {max_retries} (from env var)")
        print(f"  Log level: {log_level} (from env var)")
        
    except ConfigurationError as e:
        print(f"Environment override error: {e}")
    
    # Clean up environment variables
    for var in ['VERIFICATION_PRICE_TOLERANCE', 'VERIFICATION_MAX_RETRIES', 'VERIFICATION_LOG_LEVEL']:
        if var in os.environ:
            del os.environ[var]


def example_configuration_validation():
    """Example: Configuration validation"""
    print("\n=== Configuration Validation ===")
    
    try:
        # Load all configurations
        configs = load_config_for_environment('production')
        
        # Validate all configurations
        is_valid = validate_all_configs(configs)
        print(f"Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Validate individual configuration sections
        loader = get_config_loader()
        
        for config_type, config_data in configs.items():
            try:
                loader.validate_config(config_data, config_type)
                print(f"  {config_type} configuration: VALID")
            except ConfigurationError as e:
                print(f"  {config_type} configuration: INVALID - {e}")
        
    except ConfigurationError as e:
        print(f"Validation error: {e}")


def example_configuration_caching():
    """Example: Configuration caching and reloading"""
    print("\n=== Configuration Caching ===")
    
    try:
        loader = get_config_loader()
        
        # Load configuration (will be cached)
        print("Loading configuration (first time)...")
        config1 = loader.load_verification_config('development')
        
        # Load same configuration again (from cache)
        print("Loading configuration (from cache)...")
        config2 = loader.load_verification_config('development')
        
        # Check if they're the same object (cached)
        print(f"Configurations are identical: {config1 is config2}")
        
        # Clear cache and reload
        print("Clearing cache and reloading...")
        loader.clear_cache()
        config3 = loader.load_verification_config('development')
        
        print(f"After cache clear, identical: {config1 is config3}")
        print(f"But content is same: {config1 == config3}")
        
    except ConfigurationError as e:
        print(f"Caching error: {e}")


def main():
    """Run all configuration examples"""
    print("Verification Agent Configuration System Examples")
    print("=" * 50)
    
    # Run all examples
    example_basic_usage()
    example_environment_comparison()
    example_workflow_configuration()
    example_prompts_configuration()
    example_environment_variables()
    example_configuration_validation()
    example_configuration_caching()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()