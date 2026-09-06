"""Versioned capability contracts used by the advisor harness."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.services.conversation import ConversationPlan


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    version: str
    intents: tuple[str, ...]
    risk: str
    requires_products: bool = False
    minimum_products: int = 0
    maximum_candidates: int = 8
    allows_generation: bool = False
    input_contract: dict[str, str] = field(default_factory=dict)
    output_contract: dict[str, str] = field(default_factory=dict)
    owner: str = "core-team"
    compatibility: tuple[str, ...] = ("web", "mobile", "api")


class SkillRegistry:
    """Resolve an intent to one explicit, inspectable capability contract."""

    def __init__(self) -> None:
        skills = (
            SkillDefinition(
                name="product-detail",
                version="1.0.0",
                intents=(
                    "product_detail",
                    "select_previous_candidate",
                    "product_detail_followup",
                    "exact_product_detail",
                    "product_correction",
                    "focused_product_analysis",
                ),
                risk="low",
                requires_products=True,
                minimum_products=1,
                maximum_candidates=1,
                input_contract={"product_code": "str"},
                output_contract={"text": "str", "product_codes": "list[str]"},
                owner="catalog-team",
            ),
            SkillDefinition(
                name="product-comparison",
                version="1.1.0",
                intents=("brand_comparison", "refine_preferences"),
                risk="medium",
                requires_products=True,
                minimum_products=2,
                maximum_candidates=4,
                input_contract={"product_codes": "list[str]", "preferences": "dict"},
                output_contract={"text": "str", "product_codes": "list[str]", "decision_packet": "dict"},
                owner="decision-engine-team",
            ),
            SkillDefinition(
                name="price-causality",
                version="1.0.0",
                intents=("price_causality",),
                risk="medium",
                requires_products=True,
                minimum_products=2,
                maximum_candidates=2,
                input_contract={"product_codes": "list[str]"},
                output_contract={"text": "str", "product_codes": "list[str]", "decision_packet": "dict"},
                owner="decision-engine-team",
            ),
            SkillDefinition(
                name="price-objection",
                version="1.0.0",
                intents=("price_objection",),
                risk="medium",
                requires_products=True,
                minimum_products=1,
                maximum_candidates=4,
                input_contract={"product_code": "str"},
                output_contract={"text": "str", "product_codes": "list[str]"},
                owner="catalog-team",
            ),
            SkillDefinition(
                name="replacement-search",
                version="1.0.0",
                intents=("cheaper_alternatives", "reject_candidate"),
                risk="medium",
                requires_products=False,
                minimum_products=0,
                maximum_candidates=8,
                input_contract={"rejected_code": "str", "constraints": "QueryConstraints"},
                output_contract={"text": "str", "product_codes": "list[str]"},
                owner="search-team",
            ),
            SkillDefinition(
                name="value-ranking",
                version="1.0.0",
                intents=("value_ranking",),
                risk="medium",
                requires_products=False,
                minimum_products=0,
                maximum_candidates=40,
                input_contract={"constraints": "QueryConstraints", "use_case": "str"},
                output_contract={"text": "str", "product_codes": "list[str]", "decision_packet": "dict"},
                owner="value-engine-team",
            ),
            SkillDefinition(
                name="catalog-ranking",
                version="1.0.0",
                intents=("catalog_ranking",),
                risk="medium",
                requires_products=False,
                minimum_products=0,
                maximum_candidates=16,
                input_contract={"constraints": "QueryConstraints", "goal": "str"},
                output_contract={"text": "str", "product_codes": "list[str]", "decision_packet": "dict"},
                owner="value-engine-team",
            ),
            SkillDefinition(
                name="policy-grounding",
                version="1.0.0",
                intents=("policy",),
                risk="high",
                requires_products=False,
                minimum_products=0,
                maximum_candidates=4,
                input_contract={"query": "str", "context_code": "str"},
                output_contract={"text": "str", "sources": "list[dict]"},
                owner="policy-team",
            ),
            SkillDefinition(
                name="catalog-search",
                version="1.0.0",
                intents=("catalog_search",),
                risk="low",
                requires_products=False,
                minimum_products=0,
                maximum_candidates=8,
                allows_generation=True,
                input_contract={"query": "str", "constraints": "QueryConstraints"},
                output_contract={"text": "str", "product_codes": "list[str]"},
                owner="ai-research-team",
            ),
            SkillDefinition(
                name="clarification",
                version="1.0.0",
                intents=("clarify",),
                risk="low",
                requires_products=False,
                minimum_products=0,
                maximum_candidates=0,
                input_contract={"reason": "str"},
                output_contract={"text": "str"},
                owner="core-team",
            ),
            SkillDefinition(
                name="general-knowledge",
                version="1.0.0",
                intents=("general_explanation",),
                risk="low",
                requires_products=False,
                minimum_products=0,
                maximum_candidates=8,
                allows_generation=True,
                input_contract={"query": "str"},
                output_contract={"text": "str"},
                owner="ai-research-team",
            ),
        )
        self._by_intent = {
            intent: skill for skill in skills for intent in skill.intents
        }

    def resolve(self, plan: ConversationPlan) -> SkillDefinition:
        return self._by_intent.get(
            plan.dialogue_act,
            SkillDefinition(
                name="safe-clarification",
                version="1.0.0",
                intents=(plan.dialogue_act,),
                risk="high",
                input_contract={"reason": "str"},
                output_contract={"text": "str"},
                owner="core-team"
            ),
        )


skill_registry = SkillRegistry()
