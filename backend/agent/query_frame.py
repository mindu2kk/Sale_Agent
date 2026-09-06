"""QueryFrame construction and state commit helpers."""

from __future__ import annotations

from backend.agent.intent_router import IntentRoute
from backend.agent.state import AgentState, CandidateRef, ProductConstraints, QueryFrame


CONSTRAINT_FIELDS = (
    "category",
    "brand",
    "min_price",
    "max_price",
    "cpu_tier",
    "gpu_type",
    "ram_gb",
    "storage_gb",
    "screen_inches",
    "use_case",
)


def build_query_frame(route: IntentRoute, state: AgentState | None = None) -> QueryFrame:
    """Normalize route constraints into the thread-scoped query frame."""

    constraints = constraints_from_route(route)
    requested_attributes = route.constraints.get("requested_attributes")
    if not isinstance(requested_attributes, tuple):
        requested_attributes = ()

    if (
        inherits_previous_frame(route)
        and route.field_requested is None
        and state
        and state.query_frame
    ):
        requested_attributes = tuple(
            dict.fromkeys((*state.query_frame.requested_attributes, *requested_attributes))
        )

    exclude_product_codes = ()
    if state is not None:
        exclude_product_codes = tuple(sorted(continuation_exclude_codes(route, state)))

    return QueryFrame(
        intent=route.intent,
        constraints=constraints,
        requested_attributes=tuple(requested_attributes),
        exclude_product_codes=exclude_product_codes,
        inherit_from_last_query_frame=inherits_previous_frame(route),
    )


def constraints_from_route(route: IntentRoute) -> ProductConstraints:
    constraints = route.constraints
    return ProductConstraints(
        category=_as_optional_str(constraints.get("category")),
        brand=_as_optional_str(constraints.get("brand")),
        min_price=_as_optional_int(constraints.get("min_price")),
        max_price=_as_optional_int(constraints.get("max_price")),
        cpu_tier=_as_optional_str(constraints.get("cpu_tier")),
        gpu_type=_as_optional_str(constraints.get("gpu_type")),
        ram_gb=_as_optional_int(constraints.get("ram_gb")),
        storage_gb=_as_optional_int(constraints.get("storage_gb")),
        screen_inches=_as_optional_float(constraints.get("screen_inches")),
        use_case=_as_optional_str(constraints.get("use_case")),
    )


def is_query_continuation(route: IntentRoute) -> bool:
    return route.constraints.get("exclude_previous") is True


def inherits_previous_frame(route: IntentRoute) -> bool:
    return (
        route.constraints.get("exclude_previous") is True
        or route.constraints.get("inherits_previous") is True
    )


def continuation_exclude_codes(route: IntentRoute, state: AgentState) -> set[str]:
    if not is_query_continuation(route):
        return set()
    return {candidate.code for candidate in state.last_shown_candidates}


def commit_query_frame(
    state: AgentState,
    frame: QueryFrame,
    shown_candidates: list[CandidateRef],
) -> None:
    state.query_frame = frame
    state.last_constraints = frame.constraints
    if shown_candidates:
        state.remember_candidates(shown_candidates)


def has_material_constraints(frame: QueryFrame) -> bool:
    return any(getattr(frame.constraints, field_name) is not None for field_name in CONSTRAINT_FIELDS)


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _as_optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
