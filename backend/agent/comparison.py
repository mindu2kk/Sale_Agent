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
    selected = products[:2]
    if len(selected) < 2:
        return ComparisonResult("", "Mình cần ít nhất 2 mẫu đã xác định để so sánh.")
    a, b = selected
    lines = [
        f"| Tiêu chí | {_product_label(a)} | {_product_label(b)} |",
        "|---|---|---|",
    ]
    for field, label in COMPARISON_FIELDS:
        lines.append(f"| {label} | {_value(a, field)} | {_value(b, field)} |")

    conclusion_parts: list[str] = []
    if a.price_value is not None and b.price_value is not None:
        cheaper = a if a.price_value <= b.price_value else b
        conclusion_parts.append(f"Nếu ưu tiên giá thấp, {cheaper.name} lợi hơn.")
    if a.gpu_type != b.gpu_type:
        gpu_pick = a if a.gpu_type == "dedicated" else b if b.gpu_type == "dedicated" else None
        if gpu_pick:
            conclusion_parts.append(f"Nếu cần GPU rời, {gpu_pick.name} đáng ưu tiên hơn.")
    if not conclusion_parts:
        conclusion_parts.append("Nếu chỉ văn phòng/học tập, hãy chọn theo giá và kích thước màn hình bạn thích hơn.")
    return ComparisonResult("\n".join(lines), " ".join(conclusion_parts))


def _product_label(product: NormalizedProductFacts) -> str:
    return f"{product.name} ({product.code})"


def _value(product: NormalizedProductFacts, field: str) -> str:
    return format_attribute(product, field) or "Chưa có dữ liệu"
