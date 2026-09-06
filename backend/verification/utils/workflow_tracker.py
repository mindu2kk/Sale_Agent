"""
Workflow Execution Tracker cho StateGraph Observability

Real-time workflow tracking và monitoring:
- Execution step tracking với correlation IDs
- Performance metrics collection
- Error monitoring và alerting
- Workflow status dashboard data
- Execution history với analytics
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json

from ..models.execution import ExecutionStep, WorkflowMetrics, WorkflowExecutionLog, ExecutionStatus
from ..config import VerificationConfig
from .logging import EnhancedVerificationLogger, CorrelationIDGenerator


class WorkflowPhase(str, Enum):
    """Workflow execution phases"""
    INITIALIZATION = "initialization"
    RESEARCH = "research"
    VERIFICATION = "verification"
    CORRECTION = "correction"
    ESCALATION = "escalation"
    COMPLETION = "completion"


@dataclass
class WorkflowTrackingContext:
    """Context for tracking workflow execution"""
    workflow_id: str
    correlation_id: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_phase: WorkflowPhase = WorkflowPhase.INITIALIZATION
    current_node: Optional[str] = None
    progress_percentage: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    
    # Performance tracking
    phase_start_times: Dict[str, datetime] = field(default_factory=dict)
    node_execution_times: Dict[str, float] = field(default_factory=dict)
    
    # Issue tracking
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    critical_issues_count: int = 0
    
    # Resource usage
    llm_tokens_used: int = 0
    api_calls_made: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class WorkflowExecutionTracker:
    """
    Comprehensive workflow execution tracker với real-time monitoring
    """
    
    def __init__(self, config: VerificationConfig, logger: EnhancedVerificationLogger):
        """
        Initialize workflow tracker
        
        Args:
            config: Verification configuration
            logger: Enhanced verification logger
        """
        self.config = config
        self.logger = logger
        
        # Active workflow tracking
        self._active_workflows: Dict[str, WorkflowTrackingContext] = {}
        self._workflow_lock = threading.Lock()
        
        # Execution history
        self._execution_history: List[WorkflowExecutionLog] = []
        self._history_lock = threading.Lock()
        
        # Performance metrics
        self._performance_metrics: Dict[str, List[float]] = {
            "total_execution_time": [],
            "verification_time": [],
            "research_time": [],
            "correction_time": []
        }
        
        # Event callbacks
        self._event_callbacks: Dict[str, List[Callable]] = {}
        
        # Cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
    
    def start_workflow_tracking(self, workflow_id: str, objection_text: str, 
                              correlation_id: Optional[str] = None) -> WorkflowTrackingContext:
        """
        Start tracking a new workflow
        
        Args:
            workflow_id: Unique workflow identifier
            objection_text: Input objection text
            correlation_id: Optional correlation ID
            
        Returns:
            WorkflowTrackingContext for the workflow
        """
        
        if correlation_id is None:
            correlation_id = CorrelationIDGenerator.generate_correlation_id()
        
        context = WorkflowTrackingContext(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            max_retries=self.config.max_retries
        )
        
        with self._workflow_lock:
            self._active_workflows[workflow_id] = context
        
        # Log workflow start
        self.logger.info(
            f"Started tracking workflow: {workflow_id}",
            workflow_event="tracking_started",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            objection_length=len(objection_text),
            max_retries=context.max_retries
        )
        
        # Trigger event callbacks
        self._trigger_event("workflow_started", context, {"objection_text": objection_text})
        
        return context
    
    def update_workflow_phase(self, workflow_id: str, phase: WorkflowPhase, 
                            node_name: Optional[str] = None) -> None:
        """
        Update workflow phase và current node
        
        Args:
            workflow_id: Workflow identifier
            phase: New workflow phase
            node_name: Optional current node name
        """
        
        with self._workflow_lock:
            if workflow_id not in self._active_workflows:
                self.logger.warning(f"Workflow not found for phase update: {workflow_id}")
                return
            
            context = self._active_workflows[workflow_id]
            
            # Record phase timing
            if context.current_phase != phase:
                now = datetime.now(timezone.utc)
                
                # End previous phase timing
                if context.current_phase.value in context.phase_start_times:
                    phase_duration = (now - context.phase_start_times[context.current_phase.value]).total_seconds()
                    self.logger.debug(
                        f"Phase {context.current_phase.value} completed in {phase_duration:.2f}s",
                        workflow_id=workflow_id,
                        phase=context.current_phase.value,
                        phase_duration=phase_duration
                    )
                
                # Start new phase timing
                context.phase_start_times[phase.value] = now
                context.current_phase = phase
            
            if node_name:
                context.current_node = node_name
            
            # Update progress percentage based on phase
            progress_map = {
                WorkflowPhase.INITIALIZATION: 10.0,
                WorkflowPhase.RESEARCH: 30.0,
                WorkflowPhase.VERIFICATION: 60.0,
                WorkflowPhase.CORRECTION: 80.0,
                WorkflowPhase.ESCALATION: 90.0,
                WorkflowPhase.COMPLETION: 100.0
            }
            context.progress_percentage = progress_map.get(phase, context.progress_percentage)
        
        self.logger.debug(
            f"Workflow phase updated: {workflow_id} -> {phase.value}",
            workflow_event="phase_update",
            workflow_id=workflow_id,
            new_phase=phase.value,
            current_node=node_name,
            progress_percentage=context.progress_percentage
        )
        
        # Trigger event callbacks
        self._trigger_event("phase_changed", context, {"phase": phase, "node_name": node_name})
    
    def track_node_execution(self, workflow_id: str, node_name: str, 
                           execution_time: float, status: ExecutionStatus,
                           input_summary: str = "", output_summary: str = "",
                           error_details: Optional[str] = None,
                           metrics: Optional[Dict[str, Any]] = None) -> ExecutionStep:
        """
        Track individual node execution
        
        Args:
            workflow_id: Workflow identifier
            node_name: Name of executed node
            execution_time: Execution time in seconds
            status: Execution status
            input_summary: Summary of input state
            output_summary: Summary of output state
            error_details: Error details if failed
            metrics: Additional metrics
            
        Returns:
            ExecutionStep record
        """
        
        with self._workflow_lock:
            if workflow_id not in self._active_workflows:
                self.logger.warning(f"Workflow not found for node tracking: {workflow_id}")
                return None
            
            context = self._active_workflows[workflow_id]
            
            # Create execution step
            step = ExecutionStep(
                timestamp=datetime.now(timezone.utc).isoformat(),
                node_name=node_name,
                execution_time=execution_time,
                status=status,
                input_summary=input_summary[:200],  # Truncate for performance
                output_summary=output_summary[:200],
                error_details=error_details,
                metrics=metrics or {},
                correlation_id=context.correlation_id
            )
            
            # Update context
            context.node_execution_times[node_name] = execution_time
            context.current_node = node_name
            
            # Update resource usage from metrics
            if metrics:
                context.llm_tokens_used += metrics.get("llm_tokens", 0)
                context.api_calls_made += metrics.get("api_calls", 0)
                context.cache_hits += metrics.get("cache_hits", 0)
                context.cache_misses += metrics.get("cache_misses", 0)
        
        # Log node execution
        self.logger.info(
            f"Node executed: {node_name} - {status.value} ({execution_time:.2f}s)",
            workflow_event="node_executed",
            workflow_id=workflow_id,
            node_name=node_name,
            execution_time=execution_time,
            status=status.value,
            error_details=error_details
        )
        
        # Trigger event callbacks
        self._trigger_event("node_executed", context, {"step": step})
        
        return step
    
    def track_verification_result(self, workflow_id: str, verification_result: Any,
                                execution_time: float) -> None:
        """
        Track verification result và issues found
        
        Args:
            workflow_id: Workflow identifier
            verification_result: Verification result object
            execution_time: Verification execution time
        """
        
        with self._workflow_lock:
            if workflow_id not in self._active_workflows:
                return
            
            context = self._active_workflows[workflow_id]
            
            # Extract issues from verification result
            if hasattr(verification_result, 'criteria'):
                criteria = verification_result.criteria
                
                # Count critical issues
                if hasattr(criteria, 'critical_issues_count'):
                    context.critical_issues_count = criteria.critical_issues_count
                
                # Track individual issues
                issues = []
                
                if hasattr(criteria, 'price_issues'):
                    for issue in criteria.price_issues:
                        issues.append({
                            "type": "price_accuracy",
                            "severity": getattr(issue, 'severity', 'unknown'),
                            "description": str(issue)
                        })
                
                if hasattr(criteria, 'policy_issues'):
                    for issue in criteria.policy_issues:
                        issues.append({
                            "type": "policy_authenticity",
                            "severity": getattr(issue, 'severity', 'unknown'),
                            "description": str(issue)
                        })
                
                if hasattr(criteria, 'relevance_issues'):
                    for issue in criteria.relevance_issues:
                        issues.append({
                            "type": "topic_relevance",
                            "severity": getattr(issue, 'severity', 'unknown'),
                            "description": str(issue)
                        })
                
                context.issues_found.extend(issues)
        
        # Log verification result
        is_approved = getattr(verification_result, 'is_approved', False)
        self.logger.info(
            f"Verification result tracked: {'APPROVED' if is_approved else 'REJECTED'}",
            workflow_event="verification_tracked",
            workflow_id=workflow_id,
            verification_approved=is_approved,
            critical_issues=context.critical_issues_count,
            total_issues=len(context.issues_found),
            execution_time=execution_time
        )
        
        # Trigger event callbacks
        self._trigger_event("verification_completed", context, {
            "verification_result": verification_result,
            "execution_time": execution_time
        })
    
    def track_retry_attempt(self, workflow_id: str, retry_reason: str) -> None:
        """
        Track retry attempt
        
        Args:
            workflow_id: Workflow identifier
            retry_reason: Reason for retry
        """
        
        with self._workflow_lock:
            if workflow_id not in self._active_workflows:
                return
            
            context = self._active_workflows[workflow_id]
            context.retry_count += 1
        
        self.logger.warning(
            f"Retry attempt {context.retry_count}/{context.max_retries}: {retry_reason}",
            workflow_event="retry_tracked",
            workflow_id=workflow_id,
            retry_count=context.retry_count,
            max_retries=context.max_retries,
            retry_reason=retry_reason
        )
        
        # Trigger event callbacks
        self._trigger_event("retry_attempted", context, {"retry_reason": retry_reason})
    
    def finish_workflow_tracking(self, workflow_id: str, final_status: str,
                                final_response: str = "") -> WorkflowExecutionLog:
        """
        Finish tracking workflow và generate execution log
        
        Args:
            workflow_id: Workflow identifier
            final_status: Final workflow status
            final_response: Final response text
            
        Returns:
            WorkflowExecutionLog with complete execution data
        """
        
        with self._workflow_lock:
            if workflow_id not in self._active_workflows:
                self.logger.warning(f"Workflow not found for completion: {workflow_id}")
                return None
            
            context = self._active_workflows[workflow_id]
            end_time = datetime.now(timezone.utc)
            total_execution_time = (end_time - context.start_time).total_seconds()
            
            # Calculate metrics
            metrics = WorkflowMetrics(
                total_execution_time=total_execution_time,
                average_step_time=sum(context.node_execution_times.values()) / max(len(context.node_execution_times), 1),
                total_retries=context.retry_count,
                total_steps=len(context.node_execution_times),
                successful_steps=len([t for t in context.node_execution_times.values() if t > 0]),
                failed_steps=0,  # Will be calculated from execution steps
                nodes_executed=list(context.node_execution_times.keys()),
                success_rate=1.0 if final_status == "completed" else 0.0,
                critical_issues_found=context.critical_issues_count,
                total_issues_found=len(context.issues_found),
                llm_tokens_used=context.llm_tokens_used,
                cost_estimate=context.llm_tokens_used * 0.00001,  # Rough estimate
                cache_hits=context.cache_hits,
                cache_misses=context.cache_misses
            )
            
            # Create execution log
            execution_log = WorkflowExecutionLog(
                workflow_id=workflow_id,
                correlation_id=context.correlation_id,
                start_time=context.start_time,
                end_time=end_time,
                steps=[],  # Will be populated from tracked steps
                metrics=metrics,
                final_status=final_status,
                config_snapshot={
                    "max_retries": context.max_retries,
                    "log_level": self.config.log_level.value,
                    "detailed_logging": self.config.detailed_logging
                }
            )
            
            # Remove from active workflows
            del self._active_workflows[workflow_id]
        
        # Add to execution history
        with self._history_lock:
            self._execution_history.append(execution_log)
            
            # Keep only recent history (configurable limit)
            max_history = getattr(self.config, 'max_execution_history', 1000)
            if len(self._execution_history) > max_history:
                self._execution_history = self._execution_history[-max_history:]
        
        # Update performance metrics
        self._update_performance_metrics(metrics)
        
        # Log workflow completion
        self.logger.info(
            f"Workflow tracking completed: {workflow_id} - {final_status}",
            workflow_event="tracking_completed",
            workflow_id=workflow_id,
            final_status=final_status,
            total_execution_time=total_execution_time,
            retry_count=context.retry_count,
            critical_issues=context.critical_issues_count,
            efficiency_score=metrics.efficiency_score
        )
        
        # Trigger event callbacks
        self._trigger_event("workflow_completed", context, {
            "execution_log": execution_log,
            "final_status": final_status
        })
        
        return execution_log
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current workflow status
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Dictionary with workflow status data
        """
        
        with self._workflow_lock:
            if workflow_id not in self._active_workflows:
                return None
            
            context = self._active_workflows[workflow_id]
            
            return {
                "workflow_id": workflow_id,
                "correlation_id": context.correlation_id,
                "current_phase": context.current_phase.value,
                "current_node": context.current_node,
                "progress_percentage": context.progress_percentage,
                "retry_count": context.retry_count,
                "max_retries": context.max_retries,
                "start_time": context.start_time.isoformat(),
                "elapsed_time": (datetime.now(timezone.utc) - context.start_time).total_seconds(),
                "critical_issues_count": context.critical_issues_count,
                "total_issues_count": len(context.issues_found),
                "llm_tokens_used": context.llm_tokens_used,
                "api_calls_made": context.api_calls_made,
                "cache_hit_rate": context.cache_hits / max(context.cache_hits + context.cache_misses, 1)
            }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get real-time dashboard data
        
        Returns:
            Dictionary with dashboard metrics
        """
        
        with self._workflow_lock:
            active_workflows = list(self._active_workflows.values())
        
        with self._history_lock:
            recent_completions = [
                log for log in self._execution_history
                if log.end_time and log.end_time > datetime.now(timezone.utc) - timedelta(hours=1)
            ]
        
        # Calculate statistics
        total_active = len(active_workflows)
        
        phase_counts = {}
        for context in active_workflows:
            phase = context.current_phase.value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        avg_progress = sum(c.progress_percentage for c in active_workflows) / max(total_active, 1)
        
        # Recent performance metrics
        recent_avg_time = sum(log.metrics.total_execution_time for log in recent_completions) / max(len(recent_completions), 1)
        recent_success_rate = sum(1 for log in recent_completions if log.final_status == "completed") / max(len(recent_completions), 1)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_workflows": {
                "total": total_active,
                "by_phase": phase_counts,
                "average_progress": avg_progress
            },
            "recent_performance": {
                "completions_last_hour": len(recent_completions),
                "average_execution_time": recent_avg_time,
                "success_rate": recent_success_rate
            },
            "system_metrics": {
                "total_workflows_tracked": len(self._execution_history),
                "performance_metrics_count": sum(len(metrics) for metrics in self._performance_metrics.values())
            }
        }
    
    def register_event_callback(self, event_name: str, callback: Callable) -> None:
        """
        Register callback for workflow events
        
        Args:
            event_name: Event name (workflow_started, phase_changed, etc.)
            callback: Callback function
        """
        
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        
        self._event_callbacks[event_name].append(callback)
    
    def _trigger_event(self, event_name: str, context: WorkflowTrackingContext, 
                      event_data: Dict[str, Any]) -> None:
        """Trigger event callbacks"""
        
        if event_name in self._event_callbacks:
            for callback in self._event_callbacks[event_name]:
                try:
                    callback(context, event_data)
                except Exception as e:
                    self.logger.error(f"Error in event callback {event_name}: {e}")
    
    def _update_performance_metrics(self, metrics: WorkflowMetrics) -> None:
        """Update performance metrics history"""
        
        self._performance_metrics["total_execution_time"].append(metrics.total_execution_time)
        
        # Keep only recent metrics (last 1000 entries)
        for metric_name in self._performance_metrics:
            if len(self._performance_metrics[metric_name]) > 1000:
                self._performance_metrics[metric_name] = self._performance_metrics[metric_name][-1000:]
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        
        def cleanup_worker():
            while not self._stop_cleanup.wait(300):  # Run every 5 minutes
                try:
                    # Cleanup old workflows
                    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                    
                    with self._workflow_lock:
                        stale_workflows = [
                            wf_id for wf_id, context in self._active_workflows.items()
                            if context.start_time < cutoff_time
                        ]
                        
                        for wf_id in stale_workflows:
                            self.logger.warning(f"Cleaning up stale workflow: {wf_id}")
                            del self._active_workflows[wf_id]
                    
                    # Cleanup old execution history
                    with self._history_lock:
                        self._execution_history = [
                            log for log in self._execution_history
                            if not log.end_time or log.end_time > cutoff_time
                        ]
                    
                except Exception as e:
                    self.logger.error(f"Error in cleanup thread: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def export_execution_data(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export execution data cho analytics
        
        Args:
            output_path: Optional path to save data
            
        Returns:
            Dictionary with execution data
        """
        
        with self._history_lock:
            execution_logs = [asdict(log) for log in self._execution_history]
        
        with self._workflow_lock:
            active_workflows = [asdict(context) for context in self._active_workflows.values()]
        
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_logs": execution_logs,
            "active_workflows": active_workflows,
            "performance_metrics": self._performance_metrics,
            "dashboard_data": self.get_dashboard_data()
        }
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        return export_data
    
    def shutdown(self) -> None:
        """Shutdown tracker và cleanup resources"""
        
        self._stop_cleanup.set()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Export final data
        self.export_execution_data("logs/final_execution_data.json")
        
        self.logger.info("Workflow tracker shutdown completed")