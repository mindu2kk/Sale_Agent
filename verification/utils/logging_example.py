"""
Example Usage of Enhanced Structured Logging với Correlation IDs

Demonstrates:
- Basic logging setup
- Correlation ID tracking
- Workflow observability
- Performance monitoring
- Error handling với context
"""

import time
import asyncio
from datetime import datetime

from ..config import VerificationConfig, LogLevel
from .logging_setup import setup_verification_logging, get_workflow_logger
from .logging import (
    correlation_context,
    workflow_context,
    performance_tracking,
    CorrelationIDGenerator
)


def basic_logging_example():
    """Basic logging setup và usage example"""
    print("=== Basic Logging Example ===")
    
    # Setup logging for development environment
    configurator = setup_verification_logging("development")
    
    # Get logger for specific component
    logger = configurator.get_logger("verification.example")
    
    # Basic logging
    logger.info("Starting basic logging example")
    logger.debug("Debug information", extra_data="debug_value")
    logger.warning("Warning message", warning_type="example")
    
    print("Basic logging completed. Check logs/verification_workflow.log")


def correlation_tracking_example():
    """Correlation ID tracking example"""
    print("\n=== Correlation Tracking Example ===")
    
    configurator = setup_verification_logging("development")
    logger = configurator.get_logger("verification.correlation")
    
    # Generate correlation ID
    correlation_id = CorrelationIDGenerator.generate_correlation_id()
    workflow_id = CorrelationIDGenerator.generate_workflow_id()
    
    print(f"Generated Correlation ID: {correlation_id}")
    print(f"Generated Workflow ID: {workflow_id}")
    
    # Use correlation context
    with correlation_context(correlation_id=correlation_id, workflow_id=workflow_id):
        logger.info("Message with correlation context")
        logger.info("Another message - same correlation ID")
        
        # Nested operations maintain correlation
        simulate_verification_process(logger)
    
    print("Correlation tracking completed")


def simulate_verification_process(logger):
    """Simulate verification process với correlation tracking"""
    logger.info("Starting verification process")
    
    # Simulate verification steps
    steps = ["price_check", "policy_check", "relevance_check"]
    
    for step in steps:
        with performance_tracking(step, logger):
            logger.info(f"Executing {step}")
            time.sleep(0.1)  # Simulate work
            logger.info(f"Completed {step}")
    
    logger.info("Verification process completed")


def workflow_observability_example():
    """Workflow observability example"""
    print("\n=== Workflow Observability Example ===")
    
    configurator = setup_verification_logging("development")
    logger = configurator.get_logger("verification.workflow")
    
    workflow_id = CorrelationIDGenerator.generate_workflow_id("example_objection")
    
    # Use workflow context
    with workflow_context(workflow_id, logger):
        logger.log_workflow_start(workflow_id, "Customer objection about pricing")
        
        # Simulate workflow phases
        phases = [
            ("research", "Researching product information"),
            ("verification", "Verifying draft response"),
            ("correction", "Applying corrections"),
            ("completion", "Finalizing response")
        ]
        
        for i, (phase, description) in enumerate(phases):
            logger.info(f"Phase: {phase} - {description}")
            
            # Simulate phase work
            with performance_tracking(f"phase_{phase}", logger):
                time.sleep(0.2)
                
                # Log performance metrics
                logger.log_performance_metrics({
                    "phase": phase,
                    "execution_time": 0.2,
                    "tokens_used": 50 + i * 25,
                    "api_calls": 1,
                    "memory_usage": 1024 + i * 512
                })
        
        logger.log_workflow_end(workflow_id, "completed", 1.0)
    
    print("Workflow observability completed")


def error_handling_example():
    """Error handling với correlation context example"""
    print("\n=== Error Handling Example ===")
    
    configurator = setup_verification_logging("development")
    logger = configurator.get_logger("verification.errors")
    
    correlation_id = CorrelationIDGenerator.generate_correlation_id()
    
    with correlation_context(correlation_id=correlation_id):
        try:
            # Simulate an error
            logger.info("Attempting risky operation")
            raise ValueError("Simulated verification error")
            
        except ValueError as e:
            # Log error với comprehensive context
            error_context = {
                "operation": "verification",
                "input_data": "sample_objection",
                "step": "price_accuracy_check",
                "retry_count": 1
            }
            
            logger.log_error_with_context(e, error_context)
            logger.warning("Error handled, attempting recovery")
    
    print("Error handling completed")


def performance_monitoring_example():
    """Performance monitoring example"""
    print("\n=== Performance Monitoring Example ===")
    
    configurator = setup_verification_logging("development")
    logger = configurator.get_logger("verification.performance")
    
    # Simulate multiple operations với performance tracking
    operations = ["llm_call", "db_query", "cache_lookup", "validation"]
    
    for operation in operations:
        with performance_tracking(operation, logger):
            # Simulate different execution times
            if operation == "llm_call":
                time.sleep(0.5)  # LLM calls are slower
            elif operation == "db_query":
                time.sleep(0.2)  # DB queries moderate
            else:
                time.sleep(0.05)  # Cache/validation fast
            
            logger.info(f"Completed {operation}")
    
    # Export performance metrics
    metrics = logger.export_metrics()
    print(f"Exported {len(metrics)} performance metrics")
    
    print("Performance monitoring completed")


async def async_logging_example():
    """Async logging example"""
    print("\n=== Async Logging Example ===")
    
    configurator = setup_verification_logging("development")
    logger = configurator.get_logger("verification.async")
    
    correlation_id = CorrelationIDGenerator.generate_correlation_id()
    
    with correlation_context(correlation_id=correlation_id):
        # Simulate async operations
        async def async_operation(name: str, delay: float):
            logger.info(f"Starting async operation: {name}")
            await asyncio.sleep(delay)
            logger.info(f"Completed async operation: {name}")
            return f"result_{name}"
        
        # Run multiple async operations
        tasks = [
            async_operation("research", 0.3),
            async_operation("verification", 0.2),
            async_operation("formatting", 0.1)
        ]
        
        results = await asyncio.gather(*tasks)
        logger.info(f"All async operations completed: {results}")
    
    print("Async logging completed")


def comprehensive_example():
    """Comprehensive example combining all features"""
    print("\n=== Comprehensive Example ===")
    
    configurator = setup_verification_logging("development")
    logger = configurator.get_logger("verification.comprehensive")
    
    # Generate IDs
    correlation_id = CorrelationIDGenerator.generate_correlation_id()
    workflow_id = CorrelationIDGenerator.generate_workflow_id("comprehensive_test")
    
    print(f"Correlation ID: {correlation_id}")
    print(f"Workflow ID: {workflow_id}")
    
    # Complete workflow simulation
    with workflow_context(workflow_id, logger):
        with correlation_context(correlation_id=correlation_id):
            
            # Start workflow
            objection = "Your product is too expensive compared to competitors"
            logger.log_workflow_start(workflow_id, objection)
            
            try:
                # Research phase
                logger.update_workflow_phase(workflow_id, "research")
                with performance_tracking("research_phase", logger):
                    logger.info("Researching competitor pricing")
                    time.sleep(0.3)
                    
                    logger.log_performance_metrics({
                        "phase": "research",
                        "execution_time": 0.3,
                        "tokens_used": 120,
                        "api_calls": 2,
                        "cache_hits": 1
                    })
                
                # Verification phase
                logger.update_workflow_phase(workflow_id, "verification")
                timer_id = logger.log_verification_start(objection, "Draft response about pricing")
                
                # Simulate verification result
                from unittest.mock import Mock
                mock_result = Mock()
                mock_result.is_approved = True
                mock_result.criteria = Mock()
                mock_result.criteria.critical_issues_count = 0
                mock_result.verification_reasoning = "All checks passed"
                
                logger.log_verification_result(mock_result, timer_id)
                
                # Completion
                logger.update_workflow_phase(workflow_id, "completion")
                logger.log_workflow_end(workflow_id, "completed", 1.0)
                
            except Exception as e:
                logger.log_error_with_context(e, {
                    "workflow_id": workflow_id,
                    "phase": "comprehensive_example",
                    "objection": objection
                })
                logger.log_workflow_end(workflow_id, "failed", 0.5)
    
    print("Comprehensive example completed")


def main():
    """Run all examples"""
    print("Enhanced Structured Logging Examples")
    print("=" * 50)
    
    # Run examples
    basic_logging_example()
    correlation_tracking_example()
    workflow_observability_example()
    error_handling_example()
    performance_monitoring_example()
    
    # Run async example
    asyncio.run(async_logging_example())
    
    # Run comprehensive example
    comprehensive_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check the following log files:")
    print("- logs/verification_workflow.log")
    print("- logs/verification_errors.log")
    print("- logs/verification_metrics.log")


if __name__ == "__main__":
    main()