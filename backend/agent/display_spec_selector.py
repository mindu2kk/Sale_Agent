"""Select product specs that must be visible in answer text and cards."""

from __future__ import annotations

from dataclasses import dataclass

from backend.agent.product_facts import NormalizedProductFacts


FIELD_LABELS = {
    "price_value": "Giá",
    "cpu_tier": "CPU",
    "gpu_type": "GPU",
    "ram_gb": "RAM",
    "storage_gb": "SSD",
    "screen_inches": "Màn hình",
    "battery_wh": "Pin",
    "weight_kg": "Trọng lượng",
    "durability": "Độ bền",
    "warranty": "Bảo hành",
}


DEFAULT_DISPLAY_ORDER = (
    "price_value",
    "cpu_tier",
    "gpu_type",
    "ram_gb",
    "storage_gb",
    "screen_inches",
)


@dataclass(frozen=True)
class DisplaySpec:
    field: str
    label: str
    value: str

    @property
    def text(self) -> str:
        return f"{self.label} {self.value}"


def select_display_specs(
    product: NormalizedProductFacts,
    requested_attributes: tuple[str, ...] = (),
    *,
    max_specs: int = 5,
) -> tuple[DisplaySpec, ...]:
    """Put user-requested attributes first, then fill with useful retail specs."""

    ordered_fields = tuple(dict.fromkeys(requested_attributes + DEFAULT_DISPLAY_ORDER))
    specs: list[DisplaySpec] = []
    for field_name in ordered_fields:
        value = format_attribute(product, field_name)
        if value is None:
            continue
        specs.append(
            DisplaySpec(
                field=field_name,
                label=FIELD_LABELS.get(field_name, field_name),
                value=value,
            )
        )
        if len(specs) >= max_specs:
            break
    return tuple(specs)


def displayed_attribute_fields(
    products: tuple[NormalizedProductFacts, ...],
    requested_attributes: tuple[str, ...],
) -> tuple[str, ...]:
    displayed: list[str] = []
    for attribute in requested_attributes:
        if any(format_attribute(product, attribute) is not None for product in products):
            displayed.append(attribute)
    return tuple(displayed)


def format_attribute(
    product: NormalizedProductFacts,
    field_name: str,
) -> str | None:
    if field_name == "price_value":
        return _format_price(product.price_value)
    if field_name == "cpu_tier":
        return product.cpu_raw or product.cpu_tier
    if field_name == "gpu_type":
        return product.gpu_raw or product.gpu_type
    if field_name == "ram_gb":
        return f"{product.ram_gb}GB" if product.ram_gb is not None else None
    if field_name == "storage_gb":
        return f"{product.storage_gb}GB" if product.storage_gb is not None else None
    if field_name == "screen_inches":
        return _format_screen(product)
    if field_name == "battery_wh":
        return f"{product.battery_wh:g}Wh" if product.battery_wh is not None else None
    if field_name == "weight_kg":
        return f"{product.weight_kg:g}kg" if product.weight_kg is not None else None
    value = getattr(product, field_name, None)
    return str(value) if value is not None else None


def _format_price(value: int | None) -> str | None:
    if value is None:
        return None
    return f"{value:,}".replace(",", ".") + " VNĐ"


def _format_screen(product: NormalizedProductFacts) -> str | None:
    if product.screen_inches is None and product.refresh_hz is None:
        return None
    parts: list[str] = []
    if product.screen_inches is not None:
        parts.append(f"{product.screen_inches:g} inch")
    if product.refresh_hz is not None:
        parts.append(f"{product.refresh_hz}Hz")
    return " ".join(parts)
