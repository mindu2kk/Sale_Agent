"""
Main LangGraph StateGraph Workflow Implementation

StateGraph orchestration cho verification workflow với:
- Async node execution với error handling
- Binary conditional routing logic
- State persistence và observability
- Performance optimization với early termination
- Structured logging với correlation IDs cho all async workflow steps
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END

from ..models import WorkflowState, ExecutionStep, ExecutionStatus
from ..agent import VerificationAgent
from ..config import VerificationConfig
from .correction import SelfCorrectionNode
from .routing import WorkflowRouter
from .persistence import WorkflowPersistenceManager
from ..utils.logging import (
    setup_verification_logger,
    set_correlation_context,
    get_correlation_context,
    clear_correlation_context,
    CorrelationIDGenerator,
)
from ..utils.distributed_tracing import get_tracer
from ..utils.critical_alert_manager import get_critical_alert_manager
from ..utils.graceful_shutdown import get_shutdown_manager

logger = logging.getLogger(__name__)


class VerificationWorkflow:
    """
    Main LangGraph StateGraph workflow cho verification process
    
    Orchestrates Research → Verification → Correction loop
    với binary decisions và structured issue handling.
    """
    
    def __init__(self,
                 research_agent,
                 verification_agent: VerificationAgent,
                 config: VerificationConfig,
                 persistence_manager: Optional[WorkflowPersistenceManager] = None):
        """
        Initialize verification workflow

        Args:
            research_agent: Sales Research Agent instance
            verification_agent: Verification Agent instance
            config: Workflow configuration
            persistence_manager: Optional persistence manager for checkpointing.
                                  If provided, state is saved after each node
                                  execution and can be resumed after failures.
        """
        self.research_agent = research_agent
        self.verification_agent = verification_agent
        self.config = config
        self.persistence_manager = persistence_manager

        # Initialize workflow components
        self.correction_node = SelfCorrectionNode(config)
        self.router = WorkflowRouter(config)

        # Structured logger with correlation ID support for all workflow steps
        self._logger = setup_verification_logger("verification.workflow", config)

        # Build StateGraph
        self.graph = self._build_graph()

        # Performance tracking
        self._active_workflows: Dict[str, WorkflowState] = {}
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph StateGraph với nodes và conditional edges
        """
        
        # Create StateGraph
        workflow = StateGraph(WorkflowState)
        
        # Add workflow nodes
        workflow.add_node("research", self._execute_research_node)
        workflow.add_node("verification", self._execute_verification_node)
        workflow.add_node("correction", self._execute_correction_node)
        workflow.add_node("escalation", self._execute_escalation_node)
        
        # Set entry point
        workflow.set_entry_point("research")

        # research always proceeds to verification
        workflow.add_edge("research", "verification")

        # Add conditional edges
        workflow.add_conditional_edges(
            "verification",
            self._route_after_verification,
            {
                "approved": END,
                "correction": "correction",
                "escalation": "escalation"
            }
        )
        
        workflow.add_conditional_edges(
            "correction", 
            self._route_after_correction,
            {
                "retry": "research",
                "escalation": "escalation"
            }
        )
        
        # Escalation always ends workflow
        workflow.add_edge("escalation", END)
        
        return workflow.compile()
    
    async def execute_workflow(self, 
                             objection_text: str,
                             customer_context: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """
        Execute complete verification workflow
        
        Args:
            objection_text: Customer objection to process
            customer_context: Optional customer context
            
        Returns:
            Final workflow state với results
        """
        
        # Initialize workflow state
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # --- Resume from checkpoint if available ---
        initial_state: WorkflowState
        if self.persistence_manager and self.persistence_manager.exists(workflow_id):
            resumed = self.persistence_manager.load(workflow_id)
            if resumed is not None:
                initial_state = resumed  # type: ignore[assignment]
            else:
                initial_state = self._build_initial_state(workflow_id, objection_text, customer_context)
        else:
            initial_state = self._build_initial_state(workflow_id, objection_text, customer_context)

        # Set correlation context for this workflow (propagates to all async nodes)
        correlation_id = initial_state.get("correlation_id", CorrelationIDGenerator.generate_correlation_id())
        set_correlation_context(correlation_id, workflow_id)

        # Track active workflow
        self._active_workflows[workflow_id] = initial_state

        # Register this workflow task with the shutdown manager for graceful cleanup
        try:
            current_task = asyncio.current_task()
            if current_task is not None:
                get_shutdown_manager().register_task(current_task)
        except Exception:
            pass  # Non-critical — don't block workflow execution

        self._logger.info(
            f"Workflow execution started: {workflow_id}",
            workflow_event="workflow_start",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            objection_length=len(objection_text),
        )

        tracer = get_tracer()
        try:
            async with tracer.start_workflow_span(
                workflow_id,
                correlation_id,
                objection_length=len(objection_text),
            ):
                # Execute StateGraph workflow
                final_state = await self._execute_graph_async(initial_state)
                # Finalize workflow
                final_state["end_time"] = datetime.now().isoformat()

                # Attach trace data to execution log
                trace_data = tracer.export_trace(workflow_id)
                if trace_data:
                    final_state.setdefault("execution_log", [])
                    # Store trace as a metadata entry in the last execution step's metrics
                    if final_state["execution_log"]:
                        final_state["execution_log"][-1].metrics["trace"] = trace_data

                self._logger.info(
                    f"Workflow execution completed: {workflow_id}",
                    workflow_event="workflow_end",
                    workflow_id=workflow_id,
                    correlation_id=correlation_id,
                    final_status=final_state.get("workflow_status"),
                    retry_count=final_state.get("retry_count", 0),
                )

                # Delete checkpoint on successful completion
                if self.persistence_manager:
                    self.persistence_manager.delete(workflow_id)

                return final_state

        except Exception as e:
            self._logger.error(
                f"Workflow execution failed: {workflow_id} - {e}",
                workflow_event="workflow_error",
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                error_type=type(e).__name__,
            )
            # Handle workflow errors — checkpoint is preserved for resume
            return self._handle_workflow_error(initial_state, e)

        finally:
            tracer.clear_trace(workflow_id)
            # Cleanup active tracking
            if workflow_id in self._active_workflows:
                del self._active_workflows[workflow_id]
            # Clear correlation context
            clear_correlation_context()

    def _build_initial_state(
        self,
        workflow_id: str,
        objection_text: str,
        customer_context: Optional[Dict[str, Any]],
    ) -> WorkflowState:
        """Build a fresh initial WorkflowState dict."""
        return {  # type: ignore[return-value]
            "objection_text": objection_text,
            "customer_context": customer_context or {},
            "draft_response": "",
            "tools_used": [],
            "research_reasoning": "",
            "research_sources": [],
            "verification_result": None,
            "correction_feedback": None,
            "retry_count": 0,
            "max_retries": self.config.max_retries,
            "final_response": "",
            "workflow_status": "initialized",
            "execution_log": [],
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "resource_usage": {
                "cpu_time_seconds": 0.0,
                "memory_peak_mb": 0.0,
                "llm_tokens_total": 0,
                "llm_cost_usd": 0.0,
                "db_queries_count": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            },
            "error_log": [],
            "config": self.config.model_dump(),
            "workflow_id": workflow_id,
            "correlation_id": f"corr_{uuid.uuid4().hex[:12]}",
        }

    def _checkpoint(self, state: WorkflowState) -> None:
        """
        Auto-checkpoint: save state after each node execution.

        Silently skips if no persistence_manager is configured.
        Errors are caught and logged as warnings so they never crash the workflow.
        """
        if self.persistence_manager is None:
            return
        try:
            self.persistence_manager.save(state)
        except Exception as exc:
            logger.warning("Auto-checkpoint failed: %s", exc)

    async def _execute_graph_async(self, initial_state: WorkflowState) -> WorkflowState:
        """Execute StateGraph với async support"""
        
        # Convert sync graph execution to async
        loop = asyncio.get_event_loop()
        
        def run_graph():
            return self.graph.invoke(initial_state)
        
        # Execute with timeout
        try:
            final_state = await asyncio.wait_for(
                loop.run_in_executor(None, run_graph),
                timeout=self.config.async_timeout_seconds * 3  # Allow extra time for full workflow
            )
            return final_state
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Workflow timeout after {self.config.async_timeout_seconds * 3}s")
    
    def _execute_research_node(self, state: WorkflowState) -> WorkflowState:
        """
        Execute Sales Research Agent node với async support.

        Wraps the synchronous SalesResearchAgent.run() call, injecting any
        correction feedback into the objection prompt on retry attempts, and
        records a structured ExecutionStep in the workflow log.

        Supports both the canonical SalesResearchAgent interface (``run()``)
        and legacy interfaces that expose ``process_objection()`` or
        ``generate_response()``, so the workflow is not tightly coupled to a
        single agent implementation.
        """
        start_time = datetime.now()
        workflow_id = state.get("workflow_id", "unknown")
        correlation_id = state.get("correlation_id", "unknown")

        # Ensure correlation context is set for this node execution
        set_correlation_context(correlation_id, workflow_id)

        self._logger.info(
            "Research node started",
            workflow_event="node_start",
            node_name="research",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            retry_count=state.get("retry_count", 0),
            timestamp=start_time.isoformat(),
        )

        tracer = get_tracer()
        with tracer.start_span_sync(
            "node:research",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            retry_count=state.get("retry_count", 0),
        ) as node_span:
            try:
                state["workflow_status"] = "researching"

                # Build the prompt: on retries, prepend correction feedback so the
                # Research Agent knows exactly what to fix.
                objection = state["objection_text"]
                correction_feedback = state.get("correction_feedback")
                if correction_feedback and state.get("retry_count", 0) > 0:
                    prompt = (
                        f"{correction_feedback}\n\n"
                        f"---\n"
                        f"ORIGINAL OBJECTION:\n{objection}"
                    )
                else:
                    prompt = objection

                # --- Dispatch to the correct agent interface ---
                if hasattr(self.research_agent, "run"):
                    # Canonical SalesResearchAgent interface
                    agent_result = self.research_agent.run(prompt)
                    state["draft_response"] = agent_result.draft_response
                    state["tools_used"] = agent_result.tools_used
                    state["research_reasoning"] = getattr(agent_result, "research_reasoning", "")
                    state["research_sources"] = getattr(agent_result, "research_sources", [])

                elif hasattr(self.research_agent, "process_objection"):
                    # Legacy interface
                    result = self.research_agent.process_objection(
                        objection,
                        correction_feedback=correction_feedback,
                    )
                    state["draft_response"] = result.get("response", "")
                    state["tools_used"] = result.get("tools_used", [])
                    state["research_reasoning"] = result.get("reasoning", "")
                    state["research_sources"] = result.get("sources", [])

                else:
                    # Minimal fallback
                    state["draft_response"] = self.research_agent.generate_response(prompt)
                    state["tools_used"] = ["research_agent"]
                    state["research_reasoning"] = "Research agent response generated"
                    state["research_sources"] = []

                execution_time = (datetime.now() - start_time).total_seconds()

                # Truncate summaries to stay within ExecutionStep field limits
                objection_preview = objection[:97] + "..." if len(objection) > 100 else objection
                draft_preview = state["draft_response"][:97] + "..." if len(state["draft_response"]) > 100 else state["draft_response"]

                execution_step = ExecutionStep(
                    timestamp=datetime.now().isoformat(),
                    node_name="research",
                    execution_time=execution_time,
                    status=ExecutionStatus.SUCCESS,
                    input_summary=f"objection: '{objection_preview}'",
                    output_summary=f"draft: '{draft_preview}'",
                    metrics={
                        "retry_count": state.get("retry_count", 0),
                        "tools_used": state["tools_used"],
                        "has_correction_feedback": bool(correction_feedback),
                        "correlation_id": correlation_id,
                        "span_id": node_span.span_id,
                        "trace_id": node_span.trace_id,
                    },
                )
                state["execution_log"].append(execution_step)

                self._logger.info(
                    "Research node completed",
                    workflow_event="node_end",
                    node_name="research",
                    workflow_id=workflow_id,
                    correlation_id=correlation_id,
                    status="success",
                    execution_time=execution_time,
                    draft_length=len(state["draft_response"]),
                    tools_used=state["tools_used"],
                )

                self._checkpoint(state)
                return state

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                self._logger.error(
                    f"Research node failed: {e}",
                    workflow_event="node_error",
                    node_name="research",
                    workflow_id=workflow_id,
                    correlation_id=correlation_id,
                    error_type=type(e).__name__,
                    execution_time=execution_time,
                )
                return self._handle_node_error(state, "research", e, start_time)
    
    def _execute_verification_node(self, state: WorkflowState) -> WorkflowState:
        """
        Execute Verification Agent node with parallel verification and early termination.

        Runs price accuracy, policy authenticity, and topic relevance checks in
        parallel via VerificationAgent.verify_draft() (which uses asyncio.gather()
        internally).  Early termination is handled inside the agent when critical
        issues are detected.

        State updates:
        - ``verification_result``: populated with the VerificationResult
        - ``workflow_status``: set to "approved" or "correction_needed"
        - ``final_response``: set to draft_response when approved
        - ``execution_log``: an ExecutionStep with timing info is appended

        Error handling:
        - Any exception is caught and routed through ``_handle_node_error()``
        """
        import asyncio

        start_time = datetime.now()
        workflow_id = state.get("workflow_id", "unknown")
        correlation_id = state.get("correlation_id", "unknown")

        # Ensure correlation context is set for this node execution
        set_correlation_context(correlation_id, workflow_id)

        self._logger.info(
            "Verification node started",
            workflow_event="node_start",
            node_name="verification",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            retry_count=state.get("retry_count", 0),
            draft_length=len(state.get("draft_response", "")),
            timestamp=start_time.isoformat(),
        )

        tracer = get_tracer()
        with tracer.start_span_sync(
            "node:verification",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            retry_count=state.get("retry_count", 0),
        ) as node_span:
            try:
                # Update workflow status to indicate verification is in progress
                state["workflow_status"] = "verifying"

                draft_preview = state.get("draft_response", "")
                input_summary = (
                    f"draft: '{draft_preview[:97]}...'"
                    if len(draft_preview) > 100
                    else f"draft: '{draft_preview}'"
                )

                # ------------------------------------------------------------------
                # Run async verification in the current or a new event loop.
                # VerificationAgent.verify_draft() internally runs price / policy /
                # relevance checks in parallel with asyncio.gather() and supports
                # early termination when critical issues are found.
                # ------------------------------------------------------------------
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    # Already inside an async context (e.g. pytest-asyncio) —
                    # schedule on the running loop via a thread-safe future.
                    import concurrent.futures
                    future = asyncio.run_coroutine_threadsafe(
                        self.verification_agent.verify_draft(state), loop
                    )
                    verification_result = future.result(
                        timeout=self.config.async_timeout_seconds
                    )
                else:
                    verification_result = loop.run_until_complete(
                        self.verification_agent.verify_draft(state)
                    )

                # ------------------------------------------------------------------
                # Update state with verification result
                # ------------------------------------------------------------------
                state["verification_result"] = verification_result

                if verification_result.is_approved:
                    state["workflow_status"] = "approved"
                    state["final_response"] = state.get("draft_response", "")
                else:
                    state["workflow_status"] = "correction_needed"

                # ------------------------------------------------------------------
                # Fire critical alerts (fire-and-forget — never blocks workflow)
                # ------------------------------------------------------------------
                try:
                    alert_manager = get_critical_alert_manager()
                    alert_manager.check_and_alert(
                        verification_result,
                        correlation_id=correlation_id,
                        workflow_id=workflow_id,
                    )
                except Exception as _alert_exc:
                    logger.warning(
                        "Critical alert dispatch failed (non-blocking): %s", _alert_exc
                    )

                # ------------------------------------------------------------------
                # Build output summary for the execution log
                # ------------------------------------------------------------------
                critical_count = verification_result.criteria.critical_issues_count
                overall_status = "APPROVED" if verification_result.is_approved else "FAILED"
                output_summary = (
                    f"result: {overall_status} "
                    f"(critical={critical_count}, "
                    f"price={'PASS' if verification_result.criteria.price_accuracy_pass else 'FAIL'}, "
                    f"policy={'PASS' if verification_result.criteria.policy_authenticity_pass else 'FAIL'}, "
                    f"relevance={'PASS' if verification_result.criteria.topic_relevance_pass else 'FAIL'})"
                )
                # Truncate to ExecutionStep.output_summary max_length=200
                if len(output_summary) > 200:
                    output_summary = output_summary[:197] + "..."

                execution_time = (datetime.now() - start_time).total_seconds()

                # Build step latency metrics from VerificationResult
                step_latency_metrics: dict = {}
                if verification_result.step_latencies:
                    step_latency_metrics = verification_result.step_latencies

                execution_step = ExecutionStep(
                    timestamp=datetime.now().isoformat(),
                    node_name="verification",
                    execution_time=execution_time,
                    status=ExecutionStatus.SUCCESS,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    metrics={
                        "overall_pass": verification_result.criteria.overall_pass,
                        "critical_issues": critical_count,
                        "price_accuracy_pass": verification_result.criteria.price_accuracy_pass,
                        "policy_authenticity_pass": verification_result.criteria.policy_authenticity_pass,
                        "topic_relevance_pass": verification_result.criteria.topic_relevance_pass,
                        "tokens_used": verification_result.llm_tokens_used,
                        "verification_time_seconds": verification_result.execution_time_seconds,
                        "retry_count": state.get("retry_count", 0),
                        "early_termination_triggered": (
                            critical_count > 0 and self.config.early_termination
                        ),
                        "correlation_id": correlation_id,
                        "span_id": node_span.span_id,
                        "trace_id": node_span.trace_id,
                        **step_latency_metrics,
                    },
                )

                state["execution_log"].append(execution_step)

                self._logger.info(
                    f"Verification node completed: {overall_status}",
                    workflow_event="node_end",
                    node_name="verification",
                    workflow_id=workflow_id,
                    correlation_id=correlation_id,
                    status="success",
                    execution_time=execution_time,
                    is_approved=verification_result.is_approved,
                    critical_issues=critical_count,
                    price_pass=verification_result.criteria.price_accuracy_pass,
                    policy_pass=verification_result.criteria.policy_authenticity_pass,
                    relevance_pass=verification_result.criteria.topic_relevance_pass,
                )

                self._checkpoint(state)
                return state

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                self._logger.error(
                    f"Verification node failed: {e}",
                    workflow_event="node_error",
                    node_name="verification",
                    workflow_id=workflow_id,
                    correlation_id=correlation_id,
                    error_type=type(e).__name__,
                    execution_time=execution_time,
                )
                return self._handle_node_error(state, "verification", e, start_time)
    
    def _execute_correction_node(self, state: WorkflowState) -> WorkflowState:
        """
        Execute Self-Correction node
        """
        start_time = datetime.now()
        workflow_id = state.get("workflow_id", "unknown")
        correlation_id = state.get("correlation_id", "unknown")

        # Ensure correlation context is set for this node execution
        set_correlation_context(correlation_id, workflow_id)

        self._logger.info(
            "Correction node started",
            workflow_event="node_start",
            node_name="correction",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            retry_count=state.get("retry_count", 0),
            timestamp=start_time.isoformat(),
        )

        try:
            # Update workflow status
            state["workflow_status"] = "correcting"
            
            # Increment retry count
            state["retry_count"] += 1
            
            # Generate correction feedback
            if state["verification_result"]:
                correction_feedback = self.correction_node.generate_correction_feedback(
                    state["objection_text"],
                    state["draft_response"],
                    state["verification_result"]
                )
                state["correction_feedback"] = correction_feedback

            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log execution step
            execution_step = ExecutionStep(
                timestamp=datetime.now().isoformat(),
                node_name="correction",
                execution_time=execution_time,
                status=ExecutionStatus.SUCCESS,
                input_summary=f"retry #{state['retry_count']}",
                output_summary="correction feedback generated",
                metrics={
                    "retry_count": state["retry_count"],
                    "max_retries": state["max_retries"],
                    "correlation_id": correlation_id,
                }
            )
            
            state["execution_log"].append(execution_step)

            self._logger.info(
                "Correction node completed",
                workflow_event="node_end",
                node_name="correction",
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                status="success",
                execution_time=execution_time,
                retry_count=state["retry_count"],
                max_retries=state["max_retries"],
            )

            self._checkpoint(state)
            return state

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._logger.error(
                f"Correction node failed: {e}",
                workflow_event="node_error",
                node_name="correction",
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                error_type=type(e).__name__,
                execution_time=execution_time,
            )
            return self._handle_node_error(state, "correction", e, start_time)
    
    def _execute_escalation_node(self, state: WorkflowState) -> WorkflowState:
        """
        Execute Human Escalation node
        """
        start_time = datetime.now()
        workflow_id = state.get("workflow_id", "unknown")
        correlation_id = state.get("correlation_id", "unknown")

        # Ensure correlation context is set for this node execution
        set_correlation_context(correlation_id, workflow_id)

        self._logger.warning(
            "Escalation node started - workflow escalating to human review",
            workflow_event="node_start",
            node_name="escalation",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            retry_count=state.get("retry_count", 0),
            timestamp=start_time.isoformat(),
        )

        try:
            # Update workflow status
            state["workflow_status"] = "escalated"
            
            # Generate escalation summary
            escalation_reason = self._generate_escalation_reason(state)
            state["final_response"] = f"ESCALATED TO HUMAN REVIEW\n\nReason: {escalation_reason}"

            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log execution step
            execution_step = ExecutionStep(
                timestamp=datetime.now().isoformat(),
                node_name="escalation",
                execution_time=execution_time,
                status=ExecutionStatus.SUCCESS,
                input_summary="workflow escalated",
                output_summary="human review required",
                metrics={
                    "escalation_reason": escalation_reason[:100],
                    "retry_count": state.get("retry_count", 0),
                    "correlation_id": correlation_id,
                }
            )
            
            state["execution_log"].append(execution_step)

            self._logger.warning(
                f"Escalation node completed: {escalation_reason}",
                workflow_event="node_end",
                node_name="escalation",
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                status="escalated",
                execution_time=execution_time,
                escalation_reason=escalation_reason,
            )

            self._checkpoint(state)
            return state

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._logger.error(
                f"Escalation node failed: {e}",
                workflow_event="node_error",
                node_name="escalation",
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                error_type=type(e).__name__,
                execution_time=execution_time,
            )
            return self._handle_node_error(state, "escalation", e, start_time)
    
    def _route_after_verification(self, state: WorkflowState) -> str:
        """
        Conditional routing after verification node
        """
        return self.router.route_after_verification(state)
    
    def _route_after_correction(self, state: WorkflowState) -> str:
        """
        Conditional routing after correction node
        """
        return self.router.route_after_correction(state)
    
    def _handle_node_error(self, 
                          state: WorkflowState, 
                          node_name: str, 
                          error: Exception,
                          start_time: datetime) -> WorkflowState:
        """
        Handle node execution errors with structured logging and correlation IDs
        """
        workflow_id = state.get("workflow_id", "unknown")
        correlation_id = state.get("correlation_id", "unknown")
        execution_time = (datetime.now() - start_time).total_seconds()

        # Log error step
        execution_step = ExecutionStep(
            timestamp=datetime.now().isoformat(),
            node_name=node_name,
            execution_time=execution_time,
            status=ExecutionStatus.FAILED,
            input_summary="error occurred",
            output_summary=f"error: {str(error)[:100]}",
            error_details=str(error),
            metrics={"correlation_id": correlation_id},
        )
        
        state["execution_log"].append(execution_step)
        state["workflow_status"] = "failed"
        state["final_response"] = f"Workflow failed at {node_name}: {str(error)}"

        self._logger.error(
            f"Node error in {node_name}: {error}",
            workflow_event="node_error",
            node_name=node_name,
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            error_type=type(error).__name__,
            execution_time=execution_time,
        )
        
        return state
    
    def _handle_workflow_error(self, state: WorkflowState, error: Exception) -> WorkflowState:
        """
        Handle overall workflow errors with structured logging and correlation IDs
        """
        workflow_id = state.get("workflow_id", "unknown")
        correlation_id = state.get("correlation_id", "unknown")

        state["workflow_status"] = "failed"
        state["final_response"] = f"Workflow execution failed: {str(error)}"
        state["end_time"] = datetime.now().isoformat()
        
        # Log error
        execution_step = ExecutionStep(
            timestamp=datetime.now().isoformat(),
            node_name="workflow",
            execution_time=0.0,
            status=ExecutionStatus.FAILED,
            input_summary="workflow error",
            output_summary=str(error)[:100],
            error_details=str(error),
            metrics={"correlation_id": correlation_id},
        )
        
        state["execution_log"].append(execution_step)

        self._logger.error(
            f"Workflow error: {error}",
            workflow_event="workflow_error",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            error_type=type(error).__name__,
        )
        
        return state
    
    def _generate_escalation_reason(self, state: WorkflowState) -> str:
        """
        Generate human-readable escalation reason
        """
        
        reasons = []
        
        if state["retry_count"] >= state["max_retries"]:
            reasons.append(f"Maximum retries ({state['max_retries']}) exceeded")
        
        if state["verification_result"] and state["verification_result"].criteria.critical_issues_count > 0:
            reasons.append(f"Critical issues detected ({state['verification_result'].criteria.critical_issues_count})")
        
        if not reasons:
            reasons.append("Workflow escalation triggered")
        
        return "; ".join(reasons)
    
    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of active workflows
        """
        
        summaries = {}
        
        for workflow_id, state in self._active_workflows.items():
            summaries[workflow_id] = {
                "status": state["workflow_status"],
                "retry_count": state["retry_count"],
                "start_time": state["start_time"],
                "objection_preview": state["objection_text"][:100] + "..." if len(state["objection_text"]) > 100 else state["objection_text"]
            }
        
        return summaries