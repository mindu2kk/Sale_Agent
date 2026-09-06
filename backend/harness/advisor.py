"""Facade that makes the harness the control plane of an advisor request."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any

from backend.services.catalog import CatalogProduct, CatalogService
from backend.services.conversation import ConversationPlan, ConversationPlanner, DecisionContext
from backend.harness.context import ContextLifecycleManager, ContextSlice
from backend.harness.governance import (
    GovernanceViolation,
    PostflightPolicy,
    PreflightPolicy,
    RecoveryPolicy,
)
from backend.harness.runtime import HarnessRun, HarnessRuntime, harness_runtime
from backend.harness.skills import SkillDefinition, SkillRegistry, skill_registry


@dataclass
class HarnessSession:
    run: HarnessRun
    context: ContextSlice
    plan: ConversationPlan
    skill: SkillDefinition
    preflight: list[GovernanceViolation]
    recovery_action: str


class AdvisorHarness:
    def __init__(
        self,
        *,
        runtime: HarnessRuntime = harness_runtime,
        registry: SkillRegistry = skill_registry,
    ) -> None:
        self.runtime = runtime
        self.registry = registry
        self.context_manager = ContextLifecycleManager()
        self.preflight_policy = PreflightPolicy()
        self.postflight_policy = PostflightPolicy()

    def begin(
        self,
        *,
        query: str,
        history: list[Any],
        state: DecisionContext,
        catalog: CatalogService,
    ) -> HarnessSession:
        context = self.context_manager.prepare(history, state, catalog)
        run = self.runtime.start(query, state)
        run.context = context.public_summary()
        current_revision = self._catalog_revision(catalog)
        run.environment = {
            "catalog_revision": current_revision,
            "incoming_catalog_revision": state.catalog_revision,
            "state_version": state.state_version,
            "state_expired": state.is_expired(),
        }
        plan = ConversationPlanner(catalog).plan(query, state)
        skill = self.registry.resolve(plan)
        run.skill = {
            "name": skill.name,
            "version": skill.version,
            "risk": skill.risk,
            "maximum_candidates": skill.maximum_candidates,
        }
        self.runtime.record_plan(run, plan)
        preflight = self.preflight_policy.evaluate(
            catalog=catalog,
            plan=plan,
            skill=skill,
            context_codes=(
                [] if plan.starts_new_topic else list(context.product_codes)
            ),
        )
        if (
            state.catalog_revision
            and state.catalog_revision != current_revision
        ):
            preflight.append(
                GovernanceViolation(
                    "catalog_revision_changed",
                    "Catalog đã thay đổi; mọi dữ kiện sản phẩm phải được truy xuất lại.",
                    "warning",
                )
            )
        action = RecoveryPolicy.decide(preflight)
        run.governance["preflight"] = [item.as_dict() for item in preflight]
        run.governance["recovery_action"] = action
        if preflight:
            run.record(
                "guard",
                "Capability contract rejected or constrained the plan.",
                status="failed" if action != "continue" else "warning",
                data={
                    "skill": skill.name,
                    "violations": [item.as_dict() for item in preflight],
                    "recovery_action": action,
                },
            )
        if action != "continue":
            question = (
                "Mình chưa thể khóa đúng sản phẩm và loại máy từ ngữ cảnh hiện tại. "
                "Bạn cho mình đúng SKU hoặc tên hai mẫu cùng loại cần tư vấn nhé."
            )
            plan = replace(
                plan,
                dialogue_act="clarify",
                confidence=1.0,
                response_strategy="deterministic",
                reason="Harness preflight blocked an unsafe execution plan.",
                clarification_question=question,
            )
            skill = self.registry.resolve(plan)
            run.skill = {
                "name": skill.name,
                "version": skill.version,
                "risk": skill.risk,
                "maximum_candidates": skill.maximum_candidates,
            }
            run.record(
                "recovery",
                "Converted an unsafe plan into a bounded clarification.",
                status="recovered",
                data={"recovery_action": action},
            )
        return HarnessSession(run, context, plan, skill, preflight, action)

    @staticmethod
    def _catalog_revision(catalog: CatalogService) -> str:
        material = "|".join(
            (
                str(len(catalog.products)),
                max((item.fetched_at for item in catalog.products), default=""),
                max(
                    (item.price_valid_until for item in catalog.products),
                    default="",
                ),
            )
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]

    def record_retrieval(
        self, session: HarnessSession, products: list[CatalogProduct]
    ) -> list[str]:
        products = products[: session.skill.maximum_candidates]
        session.run.enforce_budget(len(products))
        return self.runtime.record_retrieval(session.run, products)

    def postflight(
        self,
        session: HarnessSession,
        *,
        candidates: list[CatalogProduct],
        answer_codes: list[str],
        verification_approved: bool,
        sources: list[dict[str, Any]],
        decision_trace: dict[str, Any] | None = None,
    ) -> list[GovernanceViolation]:
        violations = self.postflight_policy.evaluate(
            skill=session.skill,
            candidates=candidates,
            answer_codes=answer_codes,
            verification_approved=verification_approved,
            sources=sources,
            decision_trace=decision_trace,
        )
        session.run.governance["postflight"] = [
            item.as_dict() for item in violations
        ]
        return violations


advisor_harness = AdvisorHarness()
