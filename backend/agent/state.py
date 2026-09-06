"""Thread-scoped agent state for product grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.services.catalog import CatalogProduct, _price_value


@dataclass(frozen=True)
class CandidateRef:
    code: str
    name: str
    brand: str | None = None
    category: str | None = None
    price_value: int | None = None
    summary_specs: tuple[str, ...] = ()

    @classmethod
    def from_product(cls, product: CatalogProduct) -> "CandidateRef":
        return cls(
            code=product.code,
            name=product.name,
            brand=product.brand,
            category=product.category,
            price_value=_price_value(product.price) if product.price else None,
            summary_specs=tuple(product.specs[:4]),
        )


@dataclass(frozen=True)
class ProductConstraints:
    category: str | None = None
    brand: str | None = None
    min_price: int | None = None
    max_price: int | None = None
    cpu_tier: str | None = None
    gpu_type: str | None = None
    ram_gb: int | None = None
    storage_gb: int | None = None
    screen_inches: float | None = None
    use_case: str | None = None


@dataclass(frozen=True)
class QueryFrame:
    intent: str | None = None
    constraints: ProductConstraints = field(default_factory=ProductConstraints)
    requested_attributes: tuple[str, ...] = ()
    exclude_product_codes: tuple[str, ...] = ()
    inherit_from_last_query_frame: bool = False

    @property
    def category(self) -> str | None:
        return self.constraints.category

    @property
    def brand(self) -> str | None:
        return self.constraints.brand

    @property
    def min_price(self) -> int | None:
        return self.constraints.min_price

    @property
    def max_price(self) -> int | None:
        return self.constraints.max_price

    @property
    def cpu_tier(self) -> str | None:
        return self.constraints.cpu_tier

    @property
    def gpu_type(self) -> str | None:
        return self.constraints.gpu_type

    @property
    def ram_gb(self) -> int | None:
        return self.constraints.ram_gb

    @property
    def storage_gb(self) -> int | None:
        return self.constraints.storage_gb

    @property
    def screen_inches(self) -> float | None:
        return self.constraints.screen_inches


@dataclass
class AgentState:
    conversation_id: str = ""
    active_category: str | None = None
    focused_product_code: str | None = None
    focused_product_name: str | None = None
    last_shown_candidates: list[CandidateRef] = field(default_factory=list)
    last_shown_product_codes: list[str] = field(default_factory=list)
    last_constraints: ProductConstraints | None = None
    query_frame: QueryFrame | None = None
    compare_set: list[str] = field(default_factory=list)
    rejected_products: list[str] = field(default_factory=list)
    last_intent: str | None = None
    user_preference: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_products(
        cls,
        products: list[CatalogProduct],
        *,
        conversation_id: str = "",
    ) -> "AgentState":
        candidates = [CandidateRef.from_product(product) for product in products]
        first = candidates[0] if candidates else None
        return cls(
            conversation_id=conversation_id,
            active_category=first.category if first else None,
            focused_product_code=first.code if first else None,
            focused_product_name=first.name if first else None,
            last_shown_candidates=candidates,
        )

    @classmethod
    def from_decision_context(cls, context: Any) -> "AgentState":
        raw_frame = dict(getattr(context, "last_query_frame", {}) or {})
        raw_constraints = dict(raw_frame.get("constraints", {}) or {})
        query_frame = None
        if raw_constraints or raw_frame.get("requested_attributes"):
            query_frame = QueryFrame(
                constraints=ProductConstraints(
                    category=_coerce_optional_str(raw_constraints.get("category")),
                    brand=_coerce_optional_str(raw_constraints.get("brand")),
                    min_price=_coerce_optional_int(raw_constraints.get("min_price")),
                    max_price=_coerce_optional_int(raw_constraints.get("max_price")),
                    cpu_tier=_coerce_optional_str(raw_constraints.get("cpu_tier")),
                    gpu_type=_coerce_optional_str(raw_constraints.get("gpu_type")),
                    ram_gb=_coerce_optional_int(raw_constraints.get("ram_gb")),
                    storage_gb=_coerce_optional_int(raw_constraints.get("storage_gb")),
                    screen_inches=_coerce_optional_float(raw_constraints.get("screen_inches")),
                    use_case=_coerce_optional_str(raw_constraints.get("use_case")),
                ),
                requested_attributes=tuple(
                    item for item in raw_frame.get("requested_attributes", []) if isinstance(item, str)
                ),
                exclude_product_codes=tuple(
                    item for item in raw_frame.get("exclude_product_codes", []) if isinstance(item, str)
                ),
                inherit_from_last_query_frame=bool(raw_frame.get("inherit_from_last_query_frame", False)),
                intent=_coerce_optional_str(raw_frame.get("intent")),
            )
        candidates = [
            CandidateRef(
                code=item.code,
                name=item.name,
                brand=item.brand,
                category=item.category,
                price_value=_coerce_price_value(item.price),
                summary_specs=tuple(
                    spec.strip()
                    for spec in (item.specs_summary or "").split(",")
                    if spec.strip()
                ),
            )
            for item in getattr(context, "last_shown_candidates", []) or []
        ]
        return cls(
            active_category=getattr(context, "category", None)
            or getattr(context, "last_category", None),
            focused_product_code=getattr(context, "focused_product_code", None),
            focused_product_name=getattr(context, "focused_product_name", None),
            last_shown_candidates=candidates,
            last_shown_product_codes=[
                candidate.code for candidate in candidates
            ],
            last_constraints=query_frame.constraints if query_frame else None,
            query_frame=query_frame,
            compare_set=list(getattr(context, "compared_codes", []) or []),
            rejected_products=list(getattr(context, "rejected_codes", {}) or []),
            last_intent=getattr(context, "last_intent", None)
            or getattr(context, "last_sales_intent", None),
            user_preference=dict(getattr(context, "preferences", {}) or {}),
        )

    def remember_candidates(self, candidates: list[CandidateRef]) -> None:
        self.last_shown_candidates = candidates[:12]
        self.last_shown_product_codes = [candidate.code for candidate in self.last_shown_candidates]
        if candidates:
            self.active_category = candidates[0].category or self.active_category

    def set_focus(self, candidate: CandidateRef) -> None:
        self.focused_product_code = candidate.code
        self.focused_product_name = candidate.name
        self.active_category = candidate.category or self.active_category

    def clear_focus(self) -> None:
        self.focused_product_code = None
        self.focused_product_name = None


def _coerce_price_value(value: int | str | None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(char for char in value if char.isdigit())
        return int(digits) if digits else None
    return None


def _coerce_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
