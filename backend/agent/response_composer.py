"""Deterministic response composer for grounded sales-advisor answers."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.agent.comparison import build_comparison
from backend.agent.display_spec_selector import (
    FIELD_LABELS,
    displayed_attribute_fields,
    format_attribute,
    select_display_specs,
)
from backend.agent.evidence import EvidenceLedger
from backend.agent.next_best_question import next_best_question
from backend.agent.product_facts import NormalizedProductFacts
from backend.agent.recommendation_policy import advisory_tradeoff
from backend.agent.state import ProductConstraints


RESPONSE_MODES = {
    "filtered_search_result",
    "query_continuation_result",
    "focused_product_detail",
    "focused_product_field_answer",
    "missing_field",
    "fit_assessment",
    "comparison",
    "no_result",
    "clarifying_question",
    "correction_acknowledged",
    "strong_claim_insufficient_evidence",
    "correction",
    "tradeoff",
    "hardware_explanation",
}


@dataclass(frozen=True)
class UIAction:
    type: str
    product_codes: tuple[str, ...] = ()
    payload: dict[str, object] | None = None


@dataclass(frozen=True)
class ResponseDraftInput:
    response_mode: str
    products: tuple[NormalizedProductFacts, ...] = ()
    evidence_ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    missing_fields: tuple[str, ...] = ()
    constraints: ProductConstraints | None = None
    comparison_result: object | None = None
    ui_actions: tuple[UIAction, ...] = ()
    focused_product_code: str | None = None
    user_query: str = ""
    alternative_brands: tuple[str, ...] = ()
    requested_attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelatedProductDisplay:
    product_code: str
    display_specs: tuple[str, ...] = ()
    matching_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdvisorResponse:
    answer_text: str
    related_product_codes: tuple[str, ...]
    ui_actions: tuple[UIAction, ...]
    answer_mode: str
    missing_fields: tuple[str, ...] = ()
    displayed_attributes: tuple[str, ...] = ()
    related_products: tuple[RelatedProductDisplay, ...] = ()


def compose_response(draft: ResponseDraftInput) -> AdvisorResponse:
    """Build a grounded advisor response without inventing catalog facts."""

    mode = draft.response_mode
    if mode not in RESPONSE_MODES:
        mode = "focused_product_detail" if draft.products else "no_result"

    if mode == "no_result" or (mode == "filtered_search_result" and not draft.products):
        return _compose_no_result(draft)
    if mode == "missing_field":
        return _compose_missing_field(draft)
    if mode == "focused_product_field_answer":
        return _compose_focused_field_answer(draft)
    if mode == "fit_assessment":
        return _compose_fit_assessment(draft)
    if mode == "focused_product_detail":
        return _compose_focused_detail(draft)
    if mode in {"filtered_search_result", "query_continuation_result"}:
        return _compose_filtered_search(draft, mode)
    if mode == "comparison":
        return _compose_comparison(draft)
    if mode in {"correction", "correction_acknowledged"}:
        return _compose_correction(draft, mode)
    if mode == "tradeoff":
        return _compose_tradeoff(draft)
    if mode == "clarifying_question":
        return _response(
            "Bạn đang tìm laptop hay điện thoại, ưu tiên học tập/văn phòng, gaming hay đồ họa, và ngân sách khoảng bao nhiêu?",
            "clarifying_question",
            (),
            draft,
        )
    return _compose_hardware_explanation(draft)


def _compose_focused_detail(draft: ResponseDraftInput) -> AdvisorResponse:
    product = draft.products[0]
    lines = [
        f"Mình đang xem đúng mẫu: {product.name} ({product.code}).",
        "",
        "Thông tin catalog hiện có:",
    ]
    lines.extend(_fact_lines(product, requested_attributes=draft.requested_attributes))
    lines.extend(
        [
            "",
            "Phân tích theo tiêu chí:",
        ]
    )
    lines.extend(_focused_analysis_lines(product))
    follow_up = next_best_question(
        response_mode="focused_product_detail",
        constraints=draft.constraints,
        requested_attributes=draft.requested_attributes,
        product_count=1,
    )
    if follow_up:
        lines.extend(["", follow_up])
    return _response(
        "\n".join(lines),
        "focused_product_detail",
        (product,),
        draft,
        extra_actions=(UIAction("SET_FOCUSED_PRODUCT", (product.code,)),),
    )


def _compose_fit_assessment(draft: ResponseDraftInput) -> AdvisorResponse:
    product = draft.products[0]
    use_case = (draft.constraints.use_case if draft.constraints else None) or "office"
    if use_case == "office":
        lines = _office_fit_assessment_lines(product)
    else:
        lines = _generic_fit_assessment_lines(product, use_case)
    return _response(
        "\n".join(lines),
        "fit_assessment",
        (product,),
        draft,
        extra_actions=(UIAction("SET_FOCUSED_PRODUCT", (product.code,)),),
    )


def _compose_focused_field_answer(draft: ResponseDraftInput) -> AdvisorResponse:
    product = draft.products[0]
    field_name = (draft.requested_attributes or ("",))[0]
    value = _field_value_text(product, field_name)
    if value is None:
        return _compose_missing_field(
            ResponseDraftInput(
                response_mode="missing_field",
                products=draft.products,
                evidence_ledger=draft.evidence_ledger,
                missing_fields=(field_name,) if field_name else draft.missing_fields,
                constraints=draft.constraints,
                ui_actions=draft.ui_actions,
                focused_product_code=draft.focused_product_code,
                user_query=draft.user_query,
                alternative_brands=draft.alternative_brands,
                requested_attributes=draft.requested_attributes,
            )
        )
    text = f"{product.name} ({product.code}) có {_field_label(field_name)}: {value}."
    return _response(
        text,
        "focused_product_field_answer",
        (product,),
        draft,
        extra_actions=(UIAction("SET_FOCUSED_PRODUCT", (product.code,)),),
    )


def _compose_missing_field(draft: ResponseDraftInput) -> AdvisorResponse:
    product = draft.products[0] if draft.products else None
    missing = draft.missing_fields or draft.evidence_ledger.missing_fields
    field_text = ", ".join(_field_label(field_name) for field_name in missing) or "trường dữ liệu này"
    if product is None:
        text = f"Catalog hiện chưa có dữ liệu {field_text}, nên mình chưa thể khẳng định phần này."
        return _response(text, "missing_field", (), draft)

    lines = [
        f"Mình đang xem đúng mẫu {product.name} ({product.code}).",
        f"Catalog hiện chưa có dữ liệu {field_text} của mẫu này, nên mình chưa thể khẳng định phần đó.",
        "",
        "Thông tin hiện có:",
    ]
    lines.extend(_fact_lines(product, requested_attributes=draft.requested_attributes, limit=5))
    return _response(
        "\n".join(lines),
        "missing_field",
        (product,),
        draft,
        missing_fields=missing,
        extra_actions=(UIAction("SET_FOCUSED_PRODUCT", (product.code,)),),
    )


def _compose_filtered_search(draft: ResponseDraftInput, mode: str) -> AdvisorResponse:
    products = draft.products
    constraint_text = _constraints_text(draft.constraints)
    intro = (
        f"Mình tìm thấy {len(products)} mẫu khớp bộ lọc {constraint_text}:"
        if constraint_text
        else f"Mình tìm thấy {len(products)} mẫu phù hợp trong catalog:"
    )
    lines = [intro]
    for product in products:
        lines.append(f"- {product.name} ({product.code}): {_compact_facts(product, draft.requested_attributes)}")
    tradeoff = advisory_tradeoff(products, draft.constraints, draft.requested_attributes)
    follow_up = next_best_question(
        response_mode=mode,
        constraints=draft.constraints,
        requested_attributes=draft.requested_attributes,
        product_count=len(products),
    )
    if tradeoff:
        lines.extend(["", tradeoff])
    if follow_up:
        lines.extend(["", follow_up])
    return _response(
        "\n".join(lines),
        mode,
        products,
        draft,
        extra_actions=(
            UIAction(
                "SHOW_RELATED_PRODUCTS",
                _codes(products),
                payload={"display_specs": draft.requested_attributes} if draft.requested_attributes else None,
            ),
        ),
    )


def _compose_no_result(draft: ResponseDraftInput) -> AdvisorResponse:
    constraint_text = _constraints_text(draft.constraints)
    if constraint_text:
        text = f"Mình chưa thấy mẫu {constraint_text} trong catalog hiện tại."
    else:
        text = "Mình chưa thấy sản phẩm phù hợp trong catalog hiện tại."
    follow_up = next_best_question(
        response_mode="no_result",
        constraints=draft.constraints,
        requested_attributes=draft.requested_attributes,
        product_count=0,
    )
    if follow_up:
        text += "\n" + follow_up
    if draft.alternative_brands:
        text += "\nMình cũng có thể mở rộng sang " + "/".join(draft.alternative_brands) + " trong cùng tầm giá."
    return _response(text, "no_result", (), draft)


def _compose_comparison(draft: ResponseDraftInput) -> AdvisorResponse:
    products = draft.products
    comparison = build_comparison(products)
    lines = [
        "Mình so sánh nhanh theo dữ liệu catalog hiện có:",
        "",
        comparison.markdown_table,
        "",
        comparison.conclusion,
    ]
    return _response(
        "\n".join(line for line in lines if line is not None),
        "comparison",
        products,
        draft,
        extra_actions=(
            UIAction("OFFER_COMPARE", _codes(products)),
            UIAction(
                "SHOW_RELATED_PRODUCTS",
                _codes(products),
                payload={"display_specs": draft.requested_attributes} if draft.requested_attributes else None,
            ),
        ),
    )


def _compose_correction(draft: ResponseDraftInput, mode: str) -> AdvisorResponse:
    product = draft.products[0] if draft.products else None
    if product is None:
        return _response(
            "Mình đã ghi nhận điều chỉnh, nhưng cần bạn chỉ rõ mẫu sản phẩm cụ thể để mình bám đúng.",
            mode,
            (),
            draft,
        )
    lines = [
        f"Mình đã chỉnh lại đúng mẫu: {product.name} ({product.code}).",
        "Mình sẽ giữ focus vào mẫu này, không đổi sang danh sách sản phẩm khác.",
        "",
        "Thông tin catalog hiện có:",
    ]
    lines.extend(_fact_lines(product, requested_attributes=draft.requested_attributes))
    return _response(
        "\n".join(lines),
        mode,
        (product,),
        draft,
        extra_actions=(UIAction("SET_FOCUSED_PRODUCT", (product.code,)),),
    )


def _compose_tradeoff(draft: ResponseDraftInput) -> AdvisorResponse:
    products = draft.products
    lines = ["Nếu xét theo trade-off từ dữ liệu catalog hiện có:"]
    for product in products:
        lines.append(
            f"- {product.name} ({product.code}): mạnh ở {_strength_summary(product)}; "
            f"cần cân nhắc {_caution_summary(product)}."
        )
    return _response("\n".join(lines), "tradeoff", products, draft)


def _compose_hardware_explanation(draft: ResponseDraftInput) -> AdvisorResponse:
    products = draft.products
    lines = [
        "Mình giải thích theo thông số catalog, không gán nhãn vào sản phẩm khác nếu bạn chưa chọn.",
    ]
    for product in products:
        lines.append(f"- {product.name} ({product.code}): {_compact_facts(product, draft.requested_attributes)}")
    return _response("\n".join(lines), "hardware_explanation", products, draft)


def _response(
    answer_text: str,
    mode: str,
    products: tuple[NormalizedProductFacts, ...],
    draft: ResponseDraftInput,
    *,
    missing_fields: tuple[str, ...] | None = None,
    extra_actions: tuple[UIAction, ...] = (),
) -> AdvisorResponse:
    product_codes = _codes(products)
    actions = draft.ui_actions + extra_actions
    if product_codes and not any(action.type == "SHOW_RELATED_PRODUCTS" for action in actions):
        actions = actions + (
            UIAction(
                "SHOW_RELATED_PRODUCTS",
                product_codes,
                payload={"display_specs": draft.requested_attributes} if draft.requested_attributes else None,
            ),
        )
    return AdvisorResponse(
        answer_text=answer_text.strip(),
        related_product_codes=product_codes,
        ui_actions=actions,
        answer_mode=mode,
        missing_fields=tuple(missing_fields if missing_fields is not None else draft.missing_fields),
        displayed_attributes=_displayed_attributes(products, draft.requested_attributes),
        related_products=_related_product_displays(products, draft.requested_attributes),
    )


def _codes(products: tuple[NormalizedProductFacts, ...]) -> tuple[str, ...]:
    return tuple(product.code for product in products)


def _fact_lines(
    product: NormalizedProductFacts,
    *,
    requested_attributes: tuple[str, ...] = (),
    limit: int | None = None,
) -> list[str]:
    facts: list[tuple[str, object | None]] = [
        ("Giá", _format_price(product.price_value)),
        ("CPU", product.cpu_raw or product.cpu_tier),
        ("GPU", product.gpu_raw or product.gpu_type),
        ("RAM", f"{product.ram_gb}GB" if product.ram_gb is not None else None),
        ("SSD", f"{product.storage_gb}GB" if product.storage_gb is not None else None),
        ("Màn hình", _format_screen(product)),
        ("Pin", f"{product.battery_wh:g}Wh" if product.battery_wh is not None else None),
        ("Trọng lượng", f"{product.weight_kg:g}kg" if product.weight_kg is not None else None),
    ]
    fact_attributes = {
        "Giá": "price_value",
        "CPU": "cpu_tier",
        "GPU": "gpu_type",
        "RAM": "ram_gb",
        "SSD": "storage_gb",
        "Màn hình": "screen_inches",
        "Pin": "battery_wh",
        "Trọng lượng": "weight_kg",
    }
    requested = set(requested_attributes)
    facts.sort(key=lambda item: (0 if fact_attributes.get(item[0]) in requested else 1, item[0]))
    lines = [f"- {label}: {value}" for label, value in facts if value not in (None, "")]
    return lines[:limit] if limit is not None else lines


def _compact_facts(
    product: NormalizedProductFacts,
    requested_attributes: tuple[str, ...] = (),
) -> str:
    parts = [spec.text for spec in select_display_specs(product, requested_attributes)]
    return "; ".join(part for part in parts if part) or "catalog hiện chưa có nhiều thông số chi tiết"


def _displayed_attributes(
    products: tuple[NormalizedProductFacts, ...],
    requested_attributes: tuple[str, ...],
) -> tuple[str, ...]:
    return displayed_attribute_fields(products, requested_attributes)


def _related_product_displays(
    products: tuple[NormalizedProductFacts, ...],
    requested_attributes: tuple[str, ...],
) -> tuple[RelatedProductDisplay, ...]:
    displays: list[RelatedProductDisplay] = []
    requested = set(requested_attributes)
    for product in products:
        specs = select_display_specs(product, requested_attributes)
        displays.append(
            RelatedProductDisplay(
                product_code=product.code,
                display_specs=tuple(spec.text for spec in specs),
                matching_facts=tuple(spec.text for spec in specs if spec.field in requested),
            )
        )
    return tuple(displays)


def _fit_summary(product: NormalizedProductFacts) -> str:
    if product.gpu_type == "dedicated":
        return "gaming, đồ họa nhẹ-vừa hoặc công việc cần GPU rời"
    if product.category.casefold() == "laptop":
        return "học tập, văn phòng, họp online và di chuyển nhẹ"
    return "nhu cầu sử dụng hằng ngày"


def _caution_summary(product: NormalizedProductFacts) -> str:
    if product.gpu_type == "integrated":
        return "bạn cần gaming/đồ họa nặng vì GPU tích hợp sẽ là điểm cần cân nhắc"
    missing = [field for field in ("battery_wh", "weight_kg") if getattr(product, field, None) is None]
    if missing:
        return "catalog còn thiếu, chưa có " + ", ".join(_field_label(field) for field in missing)
    return "bạn cần thông tin ngoài catalog như độ bền thực tế hoặc bảo hành chi tiết"


def _focused_analysis_lines(product: NormalizedProductFacts) -> list[str]:
    lines: list[str] = []

    performance_bits: list[str] = []
    if product.cpu_raw or product.cpu_tier:
        performance_bits.append(f"CPU {product.cpu_raw or product.cpu_tier}")
    if product.ram_gb is not None:
        performance_bits.append(f"RAM {product.ram_gb}GB")
    if product.gpu_raw:
        performance_bits.append(f"GPU {product.gpu_raw}")
    elif product.gpu_type == "integrated":
        performance_bits.append("GPU tích hợp")
    elif product.gpu_type is None:
        performance_bits.append("catalog chưa có thông tin GPU")
    lines.append(
        "- Hiệu năng: "
        + (
            ", ".join(performance_bits)
            + "; phù hợp học tập, văn phòng, họp online và đa nhiệm thường ngày."
            if performance_bits
            else "catalog chưa có đủ CPU/RAM/GPU để kết luận sâu."
        )
    )

    display_bits: list[str] = []
    screen = _format_screen(product)
    if screen:
        display_bits.append(screen)
    resolution = _evidence_value(product, "resolution")
    if resolution:
        display_bits.append(str(resolution))
    lines.append(
        "- Màn hình: "
        + (
            ", ".join(display_bits)
            + "; hợp đọc tài liệu, làm việc trình duyệt và chia cửa sổ ở mức vừa."
            if display_bits
            else "catalog chưa có đủ kích thước/độ phân giải để đánh giá."
        )
    )

    mobility_bits: list[str] = []
    if product.weight_kg is not None:
        mobility_bits.append(f"{product.weight_kg:g}kg")
    if product.battery_wh is not None:
        mobility_bits.append(f"pin {product.battery_wh:g}Wh")
    lines.append(
        "- Di động và pin: "
        + (
            ", ".join(mobility_bits)
            + "; đây là nhóm thông số quan trọng nếu mang máy đi học/đi làm hằng ngày."
            if mobility_bits
            else "catalog chưa đủ dữ liệu pin/trọng lượng để đánh giá độ cơ động."
        )
    )

    storage_bits: list[str] = []
    if product.storage_gb is not None:
        storage_bits.append(f"SSD {product.storage_gb}GB")
    if product.ram_gb is not None:
        storage_bits.append(f"RAM {product.ram_gb}GB")
    lines.append(
        "- Bộ nhớ/lưu trữ: "
        + (
            ", ".join(storage_bits)
            + "; ổn cho app học tập, Office, trình duyệt nhiều tab và dữ liệu cá nhân cơ bản."
            if storage_bits
            else "catalog chưa có đủ RAM/SSD để đánh giá khả năng dùng lâu dài."
        )
    )

    lines.append(f"- Phù hợp nếu: bạn cần {_fit_summary(product)}.")
    cautions = _detailed_cautions(product)
    if cautions:
        lines.append("- Cần cân nhắc: " + "; ".join(cautions) + ".")
    return lines


def _office_fit_assessment_lines(product: NormalizedProductFacts) -> list[str]:
    verdict = _office_verdict(product)
    lines = [
        f"Có, {product.name} ({product.code}) {verdict}.",
        "",
        "Vì sao hợp văn phòng:",
    ]
    reasons: list[str] = []
    if product.cpu_raw or product.cpu_tier:
        reasons.append(f"- CPU {product.cpu_raw or product.cpu_tier} đủ mạnh cho Office, trình duyệt, họp online và xử lý nhiều tab.")
    if product.ram_gb is not None:
        reasons.append(f"- RAM {product.ram_gb}GB giúp đa nhiệm văn phòng thoải mái hơn mức cơ bản.")
    if product.storage_gb is not None:
        reasons.append(f"- SSD {product.storage_gb}GB đủ cho hệ điều hành, phần mềm làm việc và tài liệu cá nhân thông thường.")
    screen = _format_screen(product)
    if screen:
        reasons.append(f"- Màn hình {screen} phù hợp làm việc cơ động; kích thước 14 inch dễ mang theo hơn nhóm 15-16 inch.")
    if product.weight_kg is not None:
        reasons.append(f"- Trọng lượng {product.weight_kg:g}kg hợp mang đi làm, đi học hoặc họp ngoài văn phòng.")
    if product.battery_wh is not None:
        reasons.append(f"- Pin {product.battery_wh:g}Wh là điểm cộng cho ngày làm việc có di chuyển.")
    lines.extend(reasons or ["- Catalog hiện có quá ít thông số để kết luận sâu về nhu cầu văn phòng."])

    cautions = _office_cautions(product)
    if cautions:
        lines.extend(["", "Điểm cần cân nhắc:"])
        lines.extend(f"- {item}" for item in cautions)

    lines.extend(["", _office_buying_conclusion(product)])
    return lines


def _generic_fit_assessment_lines(product: NormalizedProductFacts, use_case: str) -> list[str]:
    return [
        f"Mình đang xem đúng mẫu: {product.name} ({product.code}).",
        "",
        f"Với nhu cầu {use_case}, các thông số đáng chú ý là:",
        *_focused_analysis_lines(product),
        "",
        "Kết luận: nên so thêm với 1-2 mẫu cùng tầm nếu nhu cầu của bạn có phần mềm chuyên biệt.",
    ]


def _office_verdict(product: NormalizedProductFacts) -> str:
    if product.ram_gb is not None and product.ram_gb >= 16 and product.storage_gb is not None:
        return "hợp văn phòng tốt"
    return "có thể dùng văn phòng, nhưng cần xem kỹ cấu hình còn thiếu"


def _office_cautions(product: NormalizedProductFacts) -> list[str]:
    cautions: list[str] = []
    if product.price_value is not None and product.price_value >= 30_000_000:
        cautions.append("Giá trên 30 triệu là khá cao nếu chỉ dùng Word, Excel, trình duyệt và họp online.")
    if product.screen_inches is not None and product.screen_inches <= 14:
        cautions.append("Màn 14 inch gọn nhẹ, nhưng nếu làm Excel/bảng tính lớn cả ngày thì màn 15-16 inch sẽ thoải mái hơn.")
    if product.gpu_type == "integrated" or product.gpu_type is None:
        cautions.append("Không nên chọn mẫu này nếu công việc văn phòng của bạn kèm dựng 3D, render nặng hoặc game nặng.")
    return cautions


def _office_buying_conclusion(product: NormalizedProductFacts) -> str:
    if product.price_value is not None and product.price_value >= 30_000_000:
        return (
            "Kết luận: nên mua nếu bạn ưu tiên máy gọn, pin tốt, RAM 16GB và muốn một mẫu văn phòng cao cấp. "
            "Nếu chỉ làm văn phòng cơ bản, nên so thêm mẫu rẻ hơn cùng RAM/SSD trước khi chốt."
        )
    return "Kết luận: nên mua cho văn phòng nếu mức giá phù hợp ngân sách của bạn."


def _detailed_cautions(product: NormalizedProductFacts) -> list[str]:
    cautions: list[str] = []
    if product.gpu_type == "integrated":
        cautions.append("không nên xem là lựa chọn chính cho game nặng hoặc dựng 3D")
    elif product.gpu_type is None:
        cautions.append("catalog chưa có GPU nên chưa kết luận chắc về game/đồ họa")
    if product.price_value is not None and product.price_value >= 30_000_000:
        cautions.append("mức giá trên 30 triệu nên cần so thêm với mẫu cùng tầm trước khi chốt")
    missing = [
        _field_label(field)
        for field in ("battery_wh", "weight_kg")
        if getattr(product, field, None) is None
    ]
    if missing:
        cautions.append("catalog còn thiếu " + ", ".join(missing))
    if not cautions:
        cautions.append("nên kiểm thêm bảo hành, cổng kết nối và trải nghiệm màn hình thực tế")
    return cautions


def _evidence_value(product: NormalizedProductFacts, field_name: str) -> object | None:
    evidence = product.evidence_map.get(field_name)
    return evidence.value if evidence is not None else None


def _strength_summary(product: NormalizedProductFacts) -> str:
    if product.gpu_type == "dedicated":
        return "GPU rời"
    if product.cpu_tier:
        return f"CPU {product.cpu_tier}"
    if product.price_value is not None:
        return "mức giá rõ ràng"
    return "các thông số đang có"


def _constraints_text(constraints: ProductConstraints | None) -> str:
    if constraints is None:
        return ""
    parts: list[str] = []
    if constraints.brand:
        parts.append(constraints.brand)
    if constraints.category:
        parts.append(constraints.category)
    if constraints.cpu_tier:
        parts.append(constraints.cpu_tier)
    if constraints.gpu_type == "dedicated":
        parts.append("card rời")
    elif constraints.gpu_type == "integrated":
        parts.append("GPU tích hợp")
    if constraints.ram_gb:
        parts.append(f"RAM {constraints.ram_gb}GB")
    if constraints.storage_gb:
        parts.append(f"SSD {constraints.storage_gb}GB")
    if constraints.min_price is not None and constraints.max_price is not None:
        parts.append(f"từ {_format_price(constraints.min_price)} đến {_format_price(constraints.max_price)}")
    elif constraints.max_price is not None:
        parts.append(f"dưới {_format_price(constraints.max_price)}")
    elif constraints.min_price is not None:
        parts.append(f"từ {_format_price(constraints.min_price)}")
    if constraints.use_case:
        use_case_labels = {
            "office": "văn phòng và học tập",
            "gaming": "gaming",
            "creative": "đồ họa",
            "portability": "mỏng nhẹ",
        }
        parts.append(f"cho {use_case_labels.get(constraints.use_case, constraints.use_case)}")
    return " ".join(parts)


def _format_price(value: int | None) -> str | None:
    if value is None:
        return None
    return f"{value:,}".replace(",", ".") + " VNĐ"


def _format_screen(product: NormalizedProductFacts) -> str | None:
    if product.screen_inches is None and product.refresh_hz is None:
        return None
    parts = []
    if product.screen_inches is not None:
        parts.append(f"{product.screen_inches:g} inch")
    if product.refresh_hz is not None:
        parts.append(f"{product.refresh_hz}Hz")
    return " ".join(parts)


def _field_value_text(product: NormalizedProductFacts, field_name: str) -> str | None:
    return format_attribute(product, field_name)


def _field_label(field_name: str) -> str:
    labels = {
        "weight_kg": "trọng lượng",
        "battery_wh": "pin",
        "durability": "độ bền",
        "warranty": "bảo hành",
        "price_value": "giá",
        "cpu_tier": "CPU",
        "gpu_type": "GPU",
        "ram_gb": "RAM",
        "storage_gb": "SSD",
        "screen_inches": "màn hình",
    }
    return labels.get(field_name, FIELD_LABELS.get(field_name, field_name))
