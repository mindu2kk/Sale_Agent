from typing import List, Optional

from backend.services.catalog import CatalogProduct
from backend.harness.response_policy import (
    SalesResponsePolicy,
    infer_sales_response_policy,
)
from backend.harness.types import BeliefState, ConversationPlan, EvidenceRef, AnswerMode


_DURABILITY_FIELDS = frozenset(
    {
        "durability",
        "material",
        "certification",
        "military_standard",
        "mil_std",
        "ruggedness",
        "drop_test",
        "ip_rating",
    }
)
_WARRANTY_FIELDS = frozenset({"warranty", "service", "support", "guarantee"})


def _ev_field(ev) -> Optional[str]:
    if isinstance(ev, dict):
        return ev.get("field")
    return getattr(ev, "field", None)


def _has_evidence_for(evidence_refs: list, field_set: frozenset) -> bool:
    return any((field := _ev_field(ev)) and field.lower() in field_set for ev in evidence_refs)


def _find_candidate_by_code(
    requested_code: Optional[str], candidates: List[CatalogProduct]
) -> Optional[CatalogProduct]:
    if not requested_code:
        return None
    requested = requested_code.lower().strip()
    for candidate in candidates:
        if candidate.code.lower().strip() == requested:
            return candidate
    return None


def _pick_detail_candidate(
    policy: SalesResponsePolicy, candidates: List[CatalogProduct]
) -> Optional[CatalogProduct]:
    return _find_candidate_by_code(policy.requested_product_code, candidates) or (
        candidates[0] if candidates else None
    )


def _candidate_perf_tags(candidate: CatalogProduct) -> List[str]:
    spec_text = " ".join(candidate.specs).lower()
    tags: list[str] = []
    if any(k in spec_text for k in ("rtx", "gtx", "radeon rx", "arc graphics")):
        tags.append("GPU rời")
    if any(k in spec_text for k in ("core i7", "core i9", "ryzen 7", "ryzen 9", "core ultra", "core 7", "core 9", "hx")):
        tags.append("CPU hiệu năng cao")
    if any(k in spec_text for k in ("16gb", "32gb", "64gb")):
        tags.append("RAM lớn")
    return tags


def _build_per_candidate_perf_summary(candidates: List[CatalogProduct]) -> str:
    lines: list[str] = []
    highlighted = False
    for candidate in candidates[:3]:
        tags = _candidate_perf_tags(candidate)
        if tags:
            highlighted = True
            lines.append(f"{candidate.name} có {', '.join(tags)}")
        else:
            lines.append(f"{candidate.name} chưa thấy thông tin hiệu năng nổi bật")
    if not highlighted:
        names = ", ".join(c.name for c in candidates[:3])
        return (
            f"Trong các mẫu hiện có ({names}), mình chưa thấy GPU rời hay CPU hiệu năng cao đủ rõ để kết luận mạnh."
        )
    return "; ".join(lines)


def _detail_body(candidate: CatalogProduct) -> str:
    specs = list(candidate.specs)
    lines = [f"Mình đang xem đúng mẫu: {candidate.name} ({candidate.code}).", ""]
    lines.append("Thông tin catalog hiện có:")
    if candidate.price:
        lines.append(f"- Giá: {candidate.price}")
    added_detail = False
    for label in (
        "CPU",
        "Card đồ hoạ",
        "GPU",
        "RAM",
        "Ổ cứng SSD",
        "Kích thước màn hình",
        "Độ phân giải",
    ):
        value = next((spec for spec in specs if spec.lower().startswith(label.lower())), None)
        if value:
            lines.append(f"- {value}")
            added_detail = True
    if not added_detail and specs:
        lines.extend(f"- {spec}" for spec in specs[:6])
    if len(lines) == 3 and not candidate.price:
        lines.append("- Catalog hiện chưa có đủ thông số cấu hình chi tiết.")
    lines.extend(["", "Đánh giá nhanh:"])
    if any(token in " ".join(specs).lower() for token in ("rtx", "gtx", "radeon", "arc graphics")):
        lines.append("- Phù hợp nếu: cần hiệu năng tốt hơn cho đồ họa hoặc game tầm vừa.")
    else:
        lines.append("- Phù hợp nếu: học tập, văn phòng, họp online và di chuyển nhẹ.")
    lines.append("- Cần cân nhắc nếu: nhu cầu game nặng hoặc đồ họa nặng vì catalog chưa cho thấy GPU rời mạnh.")
    if not specs:
        lines.append("- Thiếu dữ liệu: catalog chưa có đủ thông số chi tiết để kết luận sâu hơn.")
    return "\n".join(lines)


def build_policy_driven_response(
    policy: SalesResponsePolicy,
    candidates: List[CatalogProduct],
    evidence_refs: Optional[List] = None,
) -> str:
    evidence_refs = evidence_refs or []
    names = ", ".join(candidate.name for candidate in candidates[:3])

    opening = ""
    if policy.price_warning_scope == "light":
        opening = "Giá có thể cần kiểm tra lại ở thời điểm mua."
    elif policy.price_warning_scope == "prominent":
        opening = "Thông tin giá hoặc khuyến mãi có thể đang cập nhật nên có thể chênh lệch nhẹ."
    elif policy.intent == "ai_unavailable" and candidates:
        opening = "Hệ thống AI phân tích chuyên sâu đang bận một chút, nhưng dựa trên dữ liệu catalog hiện có,"

    if not candidates:
        if policy.template_key == "product_detail":
            code_hint = f" mã {policy.requested_product_code}" if policy.requested_product_code else ""
            body = f"Mình chưa tìm thấy sản phẩm{code_hint} trong catalog."
        elif policy.template_key == "strong_superlative":
            body = "Mình chưa tìm thấy sản phẩm nào đủ sát yêu cầu này."
        else:
            body = "Mình chưa thể tìm thấy thông tin chính xác."
    elif policy.template_key == "product_detail":
        candidate = _pick_detail_candidate(policy, candidates)
        if candidate is None:
            body = "Mình chưa tìm thấy sản phẩm đúng để phân tích."
        elif policy.requested_product_code and _find_candidate_by_code(policy.requested_product_code, candidates) is None and len(candidates) > 1:
            body = f"Mình chưa tìm thấy sản phẩm mã {policy.requested_product_code} trong catalog."
        else:
            body = _detail_body(candidate)
    elif policy.template_key == "durability_consulting":
        has_dur = _has_evidence_for(evidence_refs, _DURABILITY_FIELDS)
        has_warranty = _has_evidence_for(evidence_refs, _WARRANTY_FIELDS)
        if has_dur:
            body = f"Dựa trên dữ liệu catalog, các mẫu {names} có thông tin về vật liệu/chứng nhận độ bền."
            if has_warranty:
                body += " Thông tin bảo hành cũng có sẵn."
        else:
            body = (
                "Mình chưa thấy dữ liệu độ bền như vật liệu vỏ, chứng nhận chống va đập hoặc chuẩn quân đội trong catalog, nên chưa chốt mẫu nào bền nhất."
            )
            body += (
                " Tuy nhiên, mình có thông tin bảo hành để anh/chị tham khảo."
                if has_warranty
                else " Anh/chị nên kiểm tra thêm chính sách bảo hành nếu đây là tiêu chí quan trọng."
            )
    elif policy.template_key == "performance_consulting":
        body = (
            "Nếu ưu tiên cấu hình mạnh, mình sẽ nhìn vào CPU, GPU, RAM và SSD. "
            + _build_per_candidate_perf_summary(candidates)
        )
    elif policy.template_key == "budget_consulting":
        body = f"trong tầm ngân sách này mình thấy có các mẫu như {names} rất đáng cân nhắc."
    elif policy.template_key == "comparison":
        body = (
            f"Cả hai mẫu {candidates[0].name} và {candidates[1].name} đều có ưu điểm riêng."
            if len(candidates) == 2
            else f"mỗi mẫu {names} đều có thế mạnh riêng."
        )
    elif policy.template_key == "hardware_explanation":
        body = f"Ví dụ như các mẫu {names} đang sử dụng linh kiện này."
    elif policy.template_key == "experience_question":
        body = f"Các mẫu như {names} thường mang lại trải nghiệm rất tốt."
    elif policy.template_key == "phone_consulting":
        body = f"mình có các mẫu điện thoại như {names}."
    elif policy.template_key == "accessory_bundle":
        body = f"mình có các phụ kiện đi kèm như {names}."
    elif policy.template_key == "strong_superlative":
        body = (
            f"Mình chưa đủ dữ liệu để chốt mẫu nào là tốt nhất tuyệt đối, nhưng trong các mẫu hiện có ({names}), mỗi mẫu đều có thế mạnh riêng."
        )
    else:
        body = f"mình thấy có các mẫu như {names} đáng cân nhắc."

    if policy.should_ask_clarification:
        follow_up = "Bạn có thể nói rõ hơn về ngân sách hoặc nhu cầu cụ thể không?"
    elif policy.follow_up_style == "product_detail":
        follow_up = "Anh/chị muốn em so sánh mẫu này với mẫu vừa xem, hay đánh giá theo nhu cầu học tập/văn phòng/game?"
    elif policy.follow_up_style == "durability":
        follow_up = "Anh/chị cần bền để mang đi học/công tác nhiều, hay chủ yếu dùng cố định ở nhà/văn phòng?"
    elif policy.follow_up_style == "performance":
        follow_up = "Anh/chị cần mạnh để chơi game, dựng video, lập trình hay đa nhiệm nhiều?"
    elif policy.follow_up_style == "budget":
        follow_up = "Anh/chị ưu tiên chơi game, học tập/văn phòng hay máy mỏng nhẹ dễ mang đi?"
    elif policy.follow_up_style == "comparison":
        follow_up = "Anh/chị muốn chốt theo giá, hiệu năng, pin, màn hình hay độ gọn nhẹ?"
    elif policy.follow_up_style == "hardware":
        follow_up = "Anh/chị muốn em áp dụng tiêu chí này vào vài mẫu cụ thể trong catalog không?"
    elif candidates:
        follow_up = "Bạn có muốn mình phân tích sâu hơn về mẫu nào không?"
    else:
        follow_up = "Bạn có muốn điều chỉnh lại tiêu chí một chút không?"

    return " ".join(part for part in (opening.strip(), body.strip(), follow_up.strip()) if part)


def build_verified_fallback_response(
    reason: str,
    candidates: List[CatalogProduct],
    evidence_refs: List[EvidenceRef],
    plan: ConversationPlan,
    context: BeliefState,
    mode: AnswerMode = "consultative",
    user_query: str = "",
    ai_available: bool = True,
) -> str:
    del mode
    policy = infer_sales_response_policy(
        user_query=user_query,
        plan=plan,
        context=context,
        candidates=candidates,
        evidence_refs=evidence_refs,
        failure_reason=reason,
        ai_available=ai_available,
    )
    return build_policy_driven_response(policy, candidates, evidence_refs)
