"""Context-aware next-best-question generation."""

from __future__ import annotations

from backend.agent.state import ProductConstraints


def next_best_question(
    *,
    response_mode: str,
    constraints: ProductConstraints | None,
    requested_attributes: tuple[str, ...] = (),
    product_count: int = 0,
) -> str | None:
    constraints = constraints or ProductConstraints()
    if response_mode == "no_result":
        if constraints.brand and constraints.gpu_type == "dedicated":
            return "Bạn muốn mình mở rộng theo 1 hướng: bỏ ràng buộc hãng, tăng ngân sách, hay chấp nhận GPU tích hợp?"
        return "Bạn muốn mình nới ngân sách, đổi hãng, hay giảm bớt một tiêu chí cấu hình?"
    if constraints.brand == "Dell" and constraints.gpu_type == "dedicated":
        return "Bạn muốn mình mở rộng sang Asus/MSI/Acer có GPU rời cùng tầm giá không?"
    if constraints.brand == "Dell" and constraints.cpu_tier == "i7":
        return "Bạn muốn ưu tiên i7 giá tốt cho văn phòng, hay cần thêm card rời để làm đồ họa/game nhẹ?"
    if {"ram_gb", "storage_gb"} & set(requested_attributes):
        return "Bạn muốn mình lọc tiếp theo giá dưới 20 triệu, ưu tiên hãng Dell/Asus/MSI, hay chọn mẫu mỏng nhẹ hơn?"
    if product_count >= 2:
        return "Bạn muốn mình so sánh 2 mẫu đáng chú ý nhất theo giá, CPU/GPU và màn hình không?"
    if product_count == 1:
        return "Bạn muốn mình phân tích mẫu này theo nhu cầu học tập, văn phòng hay game/đồ họa nhẹ?"
    return None
