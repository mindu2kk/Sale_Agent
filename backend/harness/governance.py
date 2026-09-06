"""Preflight/postflight policies and deterministic recovery decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from backend.services.catalog import CatalogProduct, CatalogService
from backend.services.conversation import ConversationPlan
from backend.harness.skills import SkillDefinition


@dataclass(frozen=True)
class GovernanceViolation:
    code: str
    message: str
    severity: Literal["warning", "error", "critical"]
    recoverable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "recoverable": self.recoverable,
        }


class PreflightPolicy:
    def evaluate(
        self,
        *,
        catalog: CatalogService,
        plan: ConversationPlan,
        skill: SkillDefinition,
        context_codes: list[str],
    ) -> list[GovernanceViolation]:
        violations: list[GovernanceViolation] = []
        unknown = [
            code for code in plan.product_codes if catalog.get(code) is None
        ]
        if unknown:
            violations.append(
                GovernanceViolation(
                    "unknown_product",
                    "Mã sản phẩm không tồn tại trong catalog: " + ", ".join(unknown),
                    "critical",
                )
            )

        known = [
            catalog.get(code)
            for code in dict.fromkeys([*plan.product_codes, *context_codes])
        ]
        products = [product for product in known if product is not None]
        categories = {product.category for product in products}
        if plan.category and any(
            product.category != plan.category for product in products
        ):
            violations.append(
                GovernanceViolation(
                    "category_drift",
                    "Sản phẩm trong context không cùng loại với kế hoạch.",
                    "critical",
                )
            )
        if skill.minimum_products >= 2 and len(categories) > 1:
            violations.append(
                GovernanceViolation(
                    "cross_category_comparison",
                    "Không được so sánh trực tiếp sản phẩm khác loại.",
                    "critical",
                )
            )
        available_refs = len(dict.fromkeys([*plan.product_codes, *context_codes]))
        if skill.requires_products and available_refs < skill.minimum_products:
            # Brand comparison can resolve concrete products during retrieval.
            if not (skill.minimum_products >= 2 and len(plan.brands) >= 2):
                violations.append(
                    GovernanceViolation(
                        "insufficient_context",
                        f"Skill {skill.name} cần ít nhất {skill.minimum_products} sản phẩm.",
                        "error",
                    )
                )
        return violations


class PostflightPolicy:
    def evaluate(
        self,
        *,
        skill: SkillDefinition,
        candidates: list[CatalogProduct],
        answer_codes: list[str],
        verification_approved: bool,
        sources: list[dict[str, Any]],
        decision_trace: dict[str, Any] | None = None,
    ) -> list[GovernanceViolation]:
        violations: list[GovernanceViolation] = []
        candidate_codes = {product.code for product in candidates}
        if candidate_codes and not set(answer_codes).issubset(candidate_codes):
            violations.append(
                GovernanceViolation(
                    "candidate_escape",
                    "Câu trả lời chứa sản phẩm ngoài tập ứng viên đã khóa.",
                    "critical",
                )
            )
        source_codes = {
            str(item.get("product_code"))
            for item in sources
            if item.get("product_code")
        }
        if answer_codes and not set(answer_codes).issubset(source_codes):
            violations.append(
                GovernanceViolation(
                    "missing_evidence",
                    "Có sản phẩm trong câu trả lời chưa được nguồn dữ liệu bao phủ.",
                    "critical",
                )
            )
        if not verification_approved:
            violations.append(
                GovernanceViolation(
                    "verification_rejected",
                    "Bộ kiểm chứng không phê duyệt câu trả lời.",
                    "critical",
                )
            )
        if skill.requires_products and len(answer_codes) < skill.minimum_products:
            violations.append(
                GovernanceViolation(
                    "incomplete_skill_output",
                    f"Kết quả không đủ dữ liệu cho skill {skill.name}.",
                    "error",
                )
            )
        trace = decision_trace or {}
        scores = trace.get("scores") or []
        recommendation = trace.get("recommendation_code")
        ranking_goal = trace.get("ranking_goal")
        if recommendation and len(answer_codes) >= 3 and len(scores) >= 2:
            confidence = float(scores[0].get("confidence", 0.0))
            margin = float(scores[0].get("score", 0.0)) - float(
                scores[1].get("score", 0.0)
            )
            if ranking_goal not in {"lowest_price", "highest_price"} and (
                confidence < 0.6 or margin < 4.0
            ):
                violations.append(
                    GovernanceViolation(
                        "weak_recommendation",
                        "Kết luận đứng đầu không có đủ bằng chứng phân biệt.",
                        "critical",
                    )
                )
        return violations


class RecoveryPolicy:
    @staticmethod
    def decide(
        violations: list[GovernanceViolation],
    ) -> Literal["continue", "clarify", "deterministic_fallback"]:
        if not violations:
            return "continue"
        if any(item.code in {"unknown_product", "insufficient_context"} for item in violations):
            return "clarify"
        if any(item.severity == "critical" for item in violations):
            return "deterministic_fallback"
        return "continue"
