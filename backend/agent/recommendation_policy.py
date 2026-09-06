"""Grounded recommendation policy for short retail trade-off advice."""

from __future__ import annotations

from backend.agent.product_facts import NormalizedProductFacts
from backend.agent.state import ProductConstraints


def advisory_tradeoff(
    products: tuple[NormalizedProductFacts, ...],
    constraints: ProductConstraints | None,
    requested_attributes: tuple[str, ...] = (),
) -> str | None:
    if not products:
        return None
    if len(products) == 1:
        product = products[0]
        strengths = _strengths(product, requested_attributes)
        caution = _caution(product)
        return (
            f"Nhận định nhanh: mẫu này khớp tốt ở {strengths}. "
            f"{caution}"
        )

    cheapest = min(
        (product for product in products if product.price_value is not None),
        key=lambda product: product.price_value,
        default=None,
    )
    dedicated = [product for product in products if product.gpu_type == "dedicated"]
    integrated = [product for product in products if product.gpu_type == "integrated"]

    parts: list[str] = []
    if cheapest is not None:
        parts.append(
            f"Nếu ưu tiên giá thấp, {cheapest.name} nổi bật vì giá {_format_price(cheapest.price_value)}."
        )
    if dedicated:
        parts.append(
            f"Nếu cần GPU rời cho game/đồ họa nhẹ, nên ưu tiên {dedicated[0].name}."
        )
    elif integrated and constraints and constraints.category == "Laptop":
        parts.append(
            "Các mẫu đang có GPU tích hợp, hợp văn phòng/học tập hơn là gaming hoặc đồ họa nặng."
        )
    if {"ram_gb", "storage_gb"} & set(requested_attributes):
        parts.append("RAM/SSD trong danh sách là dữ liệu xác nhận từ catalog, không phải suy đoán.")
    return " ".join(parts[:3]) or None


def _strengths(product: NormalizedProductFacts, requested_attributes: tuple[str, ...]) -> str:
    labels = []
    if "ram_gb" in requested_attributes and product.ram_gb is not None:
        labels.append(f"RAM {product.ram_gb}GB")
    if "storage_gb" in requested_attributes and product.storage_gb is not None:
        labels.append(f"SSD {product.storage_gb}GB")
    if "gpu_type" in requested_attributes and product.gpu_raw:
        labels.append(product.gpu_raw)
    if not labels and product.price_value is not None:
        labels.append(f"giá {_format_price(product.price_value)}")
    if not labels and product.cpu_tier:
        labels.append(f"CPU {product.cpu_tier}")
    return ", ".join(labels) if labels else "các thông số catalog hiện có"


def _caution(product: NormalizedProductFacts) -> str:
    if product.gpu_type == "integrated":
        return "Điểm cần cân nhắc là GPU tích hợp sẽ không phù hợp nếu bạn cần gaming/đồ họa nặng."
    missing = [field for field in ("battery_wh", "weight_kg") if getattr(product, field, None) is None]
    if missing:
        return "Catalog còn thiếu dữ liệu pin/trọng lượng nên mình chưa kết luận phần đó."
    return "Các nhận định trên chỉ dựa vào thông số catalog hiện có."


def _format_price(value: int | None) -> str:
    if value is None:
        return "chưa rõ giá"
    return f"{value:,}".replace(",", ".") + " VNĐ"
