"""Grounded product comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass

from backend.agent.display_spec_selector import format_attribute
from backend.agent.product_facts import NormalizedProductFacts


COMPARISON_FIELDS = (
    ("price_value", "Giá"),
    ("cpu_tier", "CPU"),
    ("ram_gb", "RAM"),
    ("storage_gb", "SSD"),
    ("gpu_type", "GPU"),
    ("screen_inches", "Màn hình"),
)


@dataclass(frozen=True)
class ComparisonResult:
    markdown_table: str
    conclusion: str


def build_comparison(products: tuple[NormalizedProductFacts, ...]) -> ComparisonResult:
    selected = products[:4]
    if len(selected) < 2:
        return ComparisonResult("", "Mình cần ít nhất 2 mẫu đã xác định để so sánh.")
    lines = [
        "| Tiêu chí | " + " | ".join(_product_label(product) for product in selected) + " |",
        "|---|" + "|".join("---" for _ in selected) + "|",
    ]
    for field, label in COMPARISON_FIELDS:
        lines.append(
            f"| {label} | "
            + " | ".join(_value(product, field) for product in selected)
            + " |"
        )

    conclusion_parts: list[str] = []
    priced_products = [product for product in selected if product.price_value is not None]
    if len(priced_products) == len(selected):
        cheaper = min(priced_products, key=lambda product: product.price_value or 0)
        conclusion_parts.append(f"Nếu ưu tiên giá thấp, {cheaper.name} lợi hơn.")
    dedicated_gpu_products = [product for product in selected if product.gpu_type == "dedicated"]
    if len(dedicated_gpu_products) == 1:
        conclusion_parts.append(
            f"Nếu cần GPU rời, {dedicated_gpu_products[0].name} đáng ưu tiên hơn."
        )
    if not conclusion_parts:
        conclusion_parts.append("Nếu chỉ văn phòng/học tập, hãy chọn theo giá và kích thước màn hình bạn thích hơn.")
    return ComparisonResult("\n".join(lines), " ".join(conclusion_parts))


def _product_label(product: NormalizedProductFacts) -> str:
    return f"{product.name} ({product.code})"


def _value(product: NormalizedProductFacts, field: str) -> str:
    return format_attribute(product, field) or "Chưa có dữ liệu"
