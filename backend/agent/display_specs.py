"""Compatibility exports for backend-selected product display specs."""

from backend.agent.display_spec_selector import (
    DisplaySpec,
    displayed_attribute_fields,
    format_attribute,
    select_display_specs,
)

__all__ = [
    "DisplaySpec",
    "displayed_attribute_fields",
    "format_attribute",
    "select_display_specs",
]
