"""
Example usage of ExecutionStep and WorkflowMetrics models

Demonstrates how to use the enhanced performance tracking models
với StateGraph workflow integration và real-time monitoring.
"""

from verification.models.execution import (
    ExecutionStep, 
    WorkflowMetrics, 
    WorkflowExecutionLog, 
    WorkflowTracker,
    ExecutionStatus
)
from datetime import datetime
import asyncio


async def simulate_verification_workflow():
    """
    Simulate a complete verification workflow với performance tracking
    
    This example shows how to:
    1. Track individual execution steps
    2. Collect comprehensive metrics
    3. Monitor workflow performance
    4. Handle errors and warnings
    5. Generate optimization recommendations
    """
    
    print("🚀 Starting Verification Workflow Simulation...")
    print("=" * 60)
    
    # Initialize workflow tracker
    tracker = WorkflowTracker(
        max_concurrent_workflows=5,
        performance_alert_threshold=10.0
    )
    
    # Create workflow execution log
    workflow_log = WorkflowExecutionLog(
        workflow_id="demo_verification_workflow",
        user_id="sales_agent_demo",
        session_id="demo_session_123"
    )
    
    print(f"📋 Workflow ID: {workflow_log.workflow_id}")
    print(f"🔗 Correlation ID: {workflow_log.correlation_id}")
    print(f"👤 User: {workflow_log.user_id}")
    
    # Start tracking
    tracker.start_workflow(workflow_log)
    
    # Simulate Step 1: Research Agent
    print("\n🔍 Step 1: Research Agent Execution")
    research_step = ExecutionStep(
        node_name="research",
        execution_time=3.2,
        status=ExecutionStatus.SUCCESS,
        input_summary="objection: 'iPhone 15 Pro quá đắt so với Samsung'",
        output_summary="draft: 'iPhone 15 Pro giá 29.990.000đ với tính năng...'",
        llm_tokens_input=650,
        llm_tokens_output=420,
        llm_cost_usd=0.012,
        memory_usage_mb=52.3,
        cpu_usage_percent=15.2
    )
    
    # Add custom metrics
    research_step.add_metric("db_queries", 2)
    research_step.add_metric("cache_hits", 1)
    research_step.add_metric("product_matches", 3)
    
    workflow_log.add_step(research_step)
    print(f"   ✅ Research completed in {research_step.execution_time}s")
    print(f"   🔗 Step Correlation ID: {research_step.correlation_id}")
    
    # Simulate Step 2: Verification Agent (with failure)
    print("\n🔍 Step 2: Verification Agent Execution")
    verification_step = ExecutionStep(
        node_name="verification",
        execution_time=2.8,
        status=ExecutionStatus.FAILED,
        input_summary="draft: iPhone 15 Pro giá 29.990.000đ",
        output_summary="verification: FAIL - price mismatch detected",
        error_details="Price deviation: 12% higher than internal DB (26.790.000đ)",
        error_type="price_validation_error",
        llm_tokens_input=480,
        llm_tokens_output=320,
        llm_cost_usd=0.008,
        memory_usage_mb=48.7,
        cpu_usage_percent=18.5
    )
    
    verification_step.add_metric("price_checks", 1)
    verification_step.add_metric("policy_checks", 2)
    verification_step.add_metric("relevance_score", 0.85)
    verification_step.add_metric("issues_found", 1)
    
    workflow_log.add_step(verification_step)
    workflow_log.add_error(
        "price_validation_error", 
        "Price mismatch: 29.990.000đ vs 26.790.000đ (12% deviation)",
        "verification"
    )
    
    print(f"   ❌ Verification failed in {verification_step.execution_time}s")
    print(f"   🚨 Error: {verification_step.error_details}")
    
    # Simulate Step 3: Self-Correction
    print("\n🔍 Step 3: Self-Correction Execution")
    correction_step = ExecutionStep(
        node_name="correction",
        execution_time=1.9,
        status=ExecutionStatus.SUCCESS,
        input_summary="correction: fix price from 29.990.000đ to 26.790.000đ",
        output_summary="corrected draft: iPhone 15 Pro giá 26.790.000đ",
        llm_tokens_input=380,
        llm_tokens_output=290,
        llm_cost_usd=0.007,
        memory_usage_mb=45.1,
        cpu_usage_percent=12.8
    )
    
    correction_step.add_metric("corrections_made", 1)
    correction_step.add_metric("retry_attempt", 1)
    
    workflow_log.add_step(correction_step)
    workflow_log.add_warning(
        "performance_warning",
        "Correction required due to price validation failure",
        "correction"
    )
    
    print(f"   ✅ Correction completed in {correction_step.execution_time}s")
    
    # Simulate Step 4: Re-verification
    print("\n🔍 Step 4: Re-verification Execution")
    reverification_step = ExecutionStep(
        node_name="verification",
        execution_time=2.1,
        status=ExecutionStatus.SUCCESS,
        input_summary="corrected draft: iPhone 15 Pro giá 26.790.000đ",
        output_summary="verification: PASS - all checks successful",
        llm_tokens_input=420,
        llm_tokens_output=280,
        llm_cost_usd=0.006,
        memory_usage_mb=46.8,
        cpu_usage_percent=14.2
    )
    
    reverification_step.add_metric("price_checks", 1)
    reverification_step.add_metric("policy_checks", 2)
    reverification_step.add_metric("relevance_score", 0.92)
    reverification_step.add_metric("issues_found", 0)
    
    workflow_log.add_step(reverification_step)
    print(f"   ✅ Re-verification passed in {reverification_step.execution_time}s")
    
    # Create comprehensive metrics
    total_execution_time = sum(step.execution_time for step in workflow_log.steps)
    total_tokens = sum(step.get_total_llm_tokens() for step in workflow_log.steps)
    total_cost = sum(step.llm_cost_usd or 0 for step in workflow_log.steps)
    
    metrics = WorkflowMetrics(
        total_execution_time=total_execution_time,
        min_step_time=min(step.execution_time for step in workflow_log.steps),
        max_step_time=max(step.execution_time for step in workflow_log.steps),
        total_steps=len(workflow_log.steps),
        successful_steps=len([s for s in workflow_log.steps if s.is_successful()]),
        failed_steps=len([s for s in workflow_log.steps if s.is_failed()]),
        timeout_steps=0,
        total_retries=1,
        nodes_executed=["research", "verification", "correction", "verification"],
        node_execution_counts={
            "research": 1,
            "verification": 2,
            "correction": 1
        },
        node_average_times={
            "research": 3.2,
            "verification": 2.45,  # Average of 2.8 and 2.1
            "correction": 1.9
        },
        critical_issues_found=0,
        major_issues_found=1,  # Price mismatch
        minor_issues_found=0,
        total_issues_found=1,
        llm_tokens_used=total_tokens,
        llm_tokens_input=sum(step.llm_tokens_input or 0 for step in workflow_log.steps),
        llm_tokens_output=sum(step.llm_tokens_output or 0 for step in workflow_log.steps),
        cost_estimate=total_cost,
        peak_memory_usage_mb=max(step.memory_usage_mb or 0 for step in workflow_log.steps),
        average_cpu_usage_percent=sum(step.cpu_usage_percent or 0 for step in workflow_log.steps) / len(workflow_log.steps),
        cache_hits=4,
        cache_misses=2,
        db_queries_count=5,
        external_api_calls=2,
        verification_pass_rate=0.5,  # 1 pass out of 2 attempts initially
        escalation_rate=0.0
    )
    
    # Update workflow log with metrics
    workflow_log.metrics = metrics
    workflow_log.update_status("completed")
    
    # Complete tracking
    completed_log = tracker.complete_workflow(workflow_log.workflow_id, "completed")
    
    # Display comprehensive results
    print("\n" + "=" * 60)
    print("📊 WORKFLOW EXECUTION RESULTS")
    print("=" * 60)
    
    print("\n📋 EXECUTION SUMMARY")
    print("-" * 30)
    print(workflow_log.get_execution_summary())
    
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 30)
    print(metrics.get_performance_summary())
    
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    recommendations = metrics.get_optimization_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n🔗 CORRELATION TRACKING")
    print("-" * 30)
    for i, step in enumerate(workflow_log.steps, 1):
        status_emoji = "✅" if step.is_successful() else "❌"
        print(f"{i}. {step.node_name:12} | {step.correlation_id} | {status_emoji} {step.status.value}")
    
    print("\n🚨 ERROR & WARNING LOG")
    print("-" * 30)
    if workflow_log.errors:
        print("Errors:")
        for error in workflow_log.errors:
            print(f"  ❌ {error['error_type']}: {error['error_message']}")
    
    if workflow_log.warnings:
        print("Warnings:")
        for warning in workflow_log.warnings:
            print(f"  ⚠️  {warning['warning_type']}: {warning['warning_message']}")
    
    print("\n🖥️  SYSTEM STATUS")
    print("-" * 20)
    print(tracker.get_system_status())
    
    print("\n🎯 KEY INSIGHTS")
    print("-" * 20)
    print(f"• Workflow completed with {metrics.total_retries} retry")
    print(f"• Performance grade: {metrics.performance_grade}")
    print(f"• Cost efficiency: ${metrics.cost_per_success:.4f} per success")
    print(f"• Memory peak: {metrics.peak_memory_usage_mb:.1f}MB")
    print(f"• Average CPU: {metrics.average_cpu_usage_percent:.1f}%")
    
    print("\n🎉 Workflow simulation completed successfully!")
    return workflow_log, metrics, tracker


async def demonstrate_concurrent_workflows():
    """Demonstrate concurrent workflow tracking"""
    
    print("\n" + "=" * 60)
    print("🔄 CONCURRENT WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    tracker = WorkflowTracker(max_concurrent_workflows=3)
    
    # Create multiple workflows
    workflows = []
    for i in range(4):  # Create 4 workflows (1 over limit)
        metrics = WorkflowMetrics(
            total_execution_time=5.0 + i,
            total_steps=2,
            successful_steps=2,
            failed_steps=0,
            timeout_steps=0,
            total_retries=0,
            nodes_executed=["research", "verification"],
            critical_issues_found=0,
            major_issues_found=0,
            minor_issues_found=0,
            total_issues_found=0,
            llm_tokens_used=1000,
            llm_tokens_input=600,
            llm_tokens_output=400,
            cost_estimate=0.015,
            cache_hits=5,
            cache_misses=1,
            db_queries_count=2,
            external_api_calls=1,
            verification_pass_rate=1.0,
            escalation_rate=0.0
        )
        
        workflow = WorkflowExecutionLog(
            workflow_id=f"concurrent_workflow_{i}",
            metrics=metrics,
            user_id=f"user_{i}",
            session_id=f"session_{i}"
        )
        
        workflows.append(workflow)
        tracker.start_workflow(workflow)
        
        print(f"Started workflow {i+1}: {workflow.workflow_id}")
        print(f"  Current load: {tracker.current_load}")
        print(f"  Overloaded: {'Yes' if tracker.is_overloaded() else 'No'}")
        
        if tracker.is_overloaded():
            print(f"  ⚠️  System overloaded! Consider queuing or scaling.")
    
    print(f"\n📊 System Status:")
    print(tracker.get_system_status())
    
    # Complete workflows
    print(f"\n🏁 Completing workflows...")
    for i, workflow in enumerate(workflows[:3]):  # Complete first 3
        completed = tracker.complete_workflow(workflow.workflow_id, "completed")
        print(f"Completed workflow {i+1}: {completed.workflow_id if completed else 'Not found'}")
    
    print(f"\n📊 Final System Status:")
    print(tracker.get_system_status())


if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(simulate_verification_workflow())
    asyncio.run(demonstrate_concurrent_workflows())