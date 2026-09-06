from typing import List, Optional, Dict, Literal, Union, Any
from pydantic import BaseModel, Field, ConfigDict

# --- Enums & Literals ---
Priority = Literal["P0", "P1", "P2", "P3"]

PhaseName = Literal[
    "perceive",
    "plan",
    "preflight",
    "retrieve_execute",
    "postflight",
    "recover",
    "commit"
]

EventStatus = Literal["started", "succeeded", "failed", "recovered", "blocked", "safe_degraded", "asked_clarification"]

CategoryType = Literal["phone", "laptop", "accessory", "unknown"]

TrustLevel = Literal["high", "medium", "low"]

FreshnessState = Literal["fresh", "stale", "unknown"]

EvidenceSource = Literal["catalog", "policy", "manual", "runtime"]

SkillLifecycle = Literal["draft", "tested", "active", "deprecated", "removed"]

AnswerMode = Literal[
    "factual",
    "comparison",
    "consultative",
    "recommendation",
    "strong_claim",
    "clarification_with_facts",
    "ai_unavailable_with_catalog_facts",
    "hard_fail",
]

StrictnessMode = Literal[
    "low",
    "medium",
    "high",
    "very_high",
]

# --- Base Models ---

class BudgetUsed(BaseModel):
    elapsed_ms: int = Field(alias="elapsedMs")
    phase_events: int = Field(alias="phaseEvents")
    candidate_count: int = Field(alias="candidateCount")
    retries: int = Field()

class PreflightResult(BaseModel):
    passed: bool
    decision: Literal["approved", "ask_clarification", "rejected", "safe_degrade"]
    reason: Optional[str] = None
    trace_event: str
    blocked_guard: Optional[str] = None
    recovery_action: Optional["RecoveryAction"] = Field(None, alias="recoveryAction")
    clarification_message: Optional[str] = None

class PhaseEvent(BaseModel):
    event_id: str = Field(alias="eventId")
    run_id: str = Field(alias="runId")
    phase: PhaseName
    event_type: str = Field(alias="eventType")
    timestamp: str
    status: EventStatus
    input_hash: Optional[str] = Field(None, alias="inputHash")
    output_hash: Optional[str] = Field(None, alias="outputHash")
    reason: Optional[str] = None
    budget_used: Optional[BudgetUsed] = Field(None, alias="budgetUsed")

class ExecutionBudget(BaseModel):
    max_phase_events: int = Field(alias="maxPhaseEvents")
    max_candidates: int = Field(alias="maxCandidates")
    max_elapsed_ms: int = Field(alias="maxElapsedMs")
    max_retries: int = Field(alias="maxRetries")
    max_llm_calls: Optional[int] = Field(None, alias="maxLlmCalls")

class TerminalEvent(BaseModel):
    event_id: str = Field(alias="eventId")
    status: EventStatus
    reason: str

class HarnessMetrics(BaseModel):
    total_elapsed_ms: int = Field(alias="totalElapsedMs")
    # Add any other metrics as needed based on implementation

class HarnessRun(BaseModel):
    run_id: str = Field(alias="runId")
    request_id: str = Field(alias="requestId")
    started_at: str = Field(alias="startedAt")
    ended_at: Optional[str] = Field(None, alias="endedAt")
    user_message_hash: str = Field(alias="userMessageHash")
    catalog_revision: str = Field(alias="catalogRevision")
    phases: List[PhaseEvent]
    budget: ExecutionBudget
    terminal_event: Optional[TerminalEvent] = Field(None, alias="terminalEvent")
    metrics: Optional[HarnessMetrics] = None

class BudgetConfig(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None
    currency: str

class RejectedProduct(BaseModel):
    product_id: str = Field(alias="productId")
    reason: str

class BeliefState(BaseModel):
    version: int
    category: Optional[CategoryType] = None
    budget: Optional[BudgetConfig] = None
    active_product_ids: List[str] = Field(default_factory=list, alias="activeProductIds")
    candidate_product_ids: List[str] = Field(default_factory=list, alias="candidateProductIds")
    rejected_product_ids: List[RejectedProduct] = Field(default_factory=list, alias="rejectedProductIds")
    preferences: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)
    confidence: float
    freshness: FreshnessState
    catalog_revision: str = Field(alias="catalogRevision")

class EvidenceRef(BaseModel):
    evidence_id: str = Field(alias="evidenceId")
    source: EvidenceSource
    url: Optional[str] = None
    product_id: Optional[str] = Field(None, alias="productId")
    field: str
    value: Union[str, int, float, bool]
    fetched_at: str = Field(alias="fetchedAt")
    catalog_revision: str = Field(alias="catalogRevision")
    trust: TrustLevel
    freshness: FreshnessState

class CandidateItem(BaseModel):
    product_id: str = Field(alias="productId")
    sku: Optional[str] = None
    name: str
    verified_fields: List[EvidenceRef] = Field(default_factory=list, alias="verifiedFields")
    score: Optional[float] = None
    score_reasons: Optional[List[str]] = Field(None, alias="scoreReasons")

class CandidateSet(BaseModel):
    category: str
    candidates: List[CandidateItem]
    retrieval_method: str = Field(alias="retrievalMethod")
    catalog_revision: str = Field(alias="catalogRevision")

class ConversationPlan(BaseModel):
    intent: Literal[
        "product_detail",
        "compare_products",
        "recommend_by_need",
        "catalog_ranking",
        "budget_filter",
        "policy_question",
        "clarification"
    ]
    skill_name: str = Field(alias="skillName")
    objective: str
    expected_category: Optional[str] = Field(None, alias="expectedCategory")
    required_evidence_fields: List[str] = Field(default_factory=list, alias="requiredEvidenceFields")
    constraints: List[str] = Field(default_factory=list)
    should_ask_clarification: bool = Field(alias="shouldAskClarification")

class RecommendationData(BaseModel):
    product_id: str = Field(alias="productId")
    reasons: List[str]
    confidence: float
    score_margin: Optional[float] = Field(None, alias="scoreMargin")

class DecisionPacket(BaseModel):
    recommendation: Optional[RecommendationData] = None
    abstained: bool
    tradeoffs: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list, alias="evidenceIds")

# --- Recovery Actions ---
class BaseRecoveryAction(BaseModel):
    type: str

class AskClarificationAction(BaseRecoveryAction):
    type: Literal["ask_clarification"] = "ask_clarification"
    question: str

class RegenerateWithConstraintsAction(BaseRecoveryAction):
    type: Literal["regenerate_with_constraints"] = "regenerate_with_constraints"
    constraints: List[str]

class DeterministicFallbackAction(BaseRecoveryAction):
    type: Literal["deterministic_fallback"] = "deterministic_fallback"
    skill_name: str = Field(alias="skillName")

class SafeDegradeAction(BaseRecoveryAction):
    type: Literal["safe_degrade"] = "safe_degrade"
    message: str

class EscalateAction(BaseRecoveryAction):
    type: Literal["escalate"] = "escalate"
    reason: str

RecoveryAction = Union[
    AskClarificationAction,
    RegenerateWithConstraintsAction,
    DeterministicFallbackAction,
    SafeDegradeAction,
    EscalateAction
]

class VerificationFailure(BaseModel):
    code: str
    severity: Literal["blocker", "warning"]
    message: str

class VerificationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    passed: bool
    failures: List[VerificationFailure] = Field(default_factory=list)
    recovery_action: Optional[RecoveryAction] = Field(None, alias="recoveryAction")

class DecisionGateResult(BaseModel):
    allowed: bool
    abstained: bool
    reason: Optional[str] = None
    required_user_criterion: bool
    has_verifiable_advantage: bool
    confidence_ok: bool
    margin_ok: bool
    evidence_ok: bool
    differentiating_reasons_ok: bool
    recommended_recovery: Optional[RecoveryAction] = Field(None, alias="recommendedRecovery")
    trace_event: Optional[str] = Field(None, alias="traceEvent")

class SkillDefinition(BaseModel):
    name: str
    version: str
    lifecycle: SkillLifecycle
    owner: str
    input_contract: str = Field(alias="inputContract")
    output_contract: str = Field(alias="outputContract")
    allowed_tools: List[str] = Field(default_factory=list, alias="allowedTools")
    tests: List[str] = Field(default_factory=list)
    compatibility: List[str] = Field(default_factory=list)
    changelog: List[str] = Field(default_factory=list)
