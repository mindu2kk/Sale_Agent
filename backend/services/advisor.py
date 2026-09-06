"""Deterministic product-advisory logic grounded in the internal catalog."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Literal

from backend.services.catalog import CatalogProduct, CatalogService
from backend.services.value_engine import CatalogCapabilityRanking, ValueRanking


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD", value.casefold().replace("đ", "d")
    )
    ascii_value = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


def _price_value(product: CatalogProduct) -> int:
    return int(re.sub(r"\D", "", product.price) or "0")


def _format_vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + " VNĐ"


@lru_cache(maxsize=100_000)
def _spec_map(product: CatalogProduct) -> dict[str, str]:
    result: dict[str, str] = {}
    display_patterns = {
        "chip": r"^Chip\s+(.+)$",
        "core": r"^(?:CPU|Core)\s+(.+)$",
        "gpu": r"^(?:Card đồ hoạ|Card đồ họa|GPU)\s+(.+)$",
        "ram": r"^RAM\s+(.+)$",
        "storage": r"^Bộ nhớ trong\s+(.+)$",
        "ssd": r"^(?:Ổ cứng SSD|SSD)\s+(.+)$",
        "screen": r"^(?:Kích thước màn hình|Màn hình)\s+(.+)$",
        "resolution": r"^Độ phân giải\s+(.+)$",
        "os": r"^Hệ điều hành\s+(.+)$",
        "camera": r"^Camera\s+(.+)$",
        "battery": r"^Pin\s+(.+)$",
        "weight": r"^Trá»ng lÆ°á»£ng\s+(.+)$",
        "refresh_rate": r"^Tần số quét\s+(.+)$",
        "water_resistance": r"^Kháng nước\s+(.+)$",
        "warranty": r"^Bảo hành\s+(.+)$",
        "nfc": r"^(?:Kết nối NFC|NFC)\s+(.+)$",
    }
    for spec in product.specs:
        for key, pattern in display_patterns.items():
            match = re.match(pattern, spec, flags=re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
                break
        else:
            normalized = _normalize(spec)
            if not normalized or normalized.endswith(" nan"):
                continue
            for key, aliases in {
                "chip": ("chip ",),
                "core": ("core ",),
                "gpu": ("card do hoa ", "gpu "),
                "ram": ("ram ",),
                "storage": ("bo nho trong ",),
                "ssd": ("o cung ssd ", "ssd "),
                "screen": ("kich thuoc man hinh ", "man hinh "),
                "resolution": ("do phan giai ",),
                "os": ("he dieu hanh ",),
                "camera": ("camera ",),
                "battery": ("pin ", "dung luong pin "),
                "weight": ("trong luong ",),
                "refresh_rate": ("tan so quet ",),
                "water_resistance": ("khang nuoc ",),
                "warranty": ("bao hanh ",),
                "nfc": ("ket noi nfc ", "nfc "),
            }.items():
                alias = next(
                    (item for item in aliases if normalized.startswith(item)),
                    None,
                )
                if alias:
                    result[key] = normalized[len(alias) :].strip()
                    break
    return result


def _number(value: str) -> int:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else 0


@dataclass(frozen=True)
class Alternative:
    product: CatalogProduct
    similarity_score: float
    savings: int
    shared_specs: tuple[str, ...]
    tradeoffs: tuple[str, ...]


@dataclass(frozen=True)
class AdvisoryResult:
    text: str
    product_codes: tuple[str, ...]


AdvisoryIntent = Literal["detail", "price_objection", "alternatives"]


class CatalogAdvisor:
    """Build useful product advice without asking an external LLM to invent value."""

    PRICE_OBJECTION_TERMS = (
        "gia dat",
        "dat qua",
        "mac qua",
        "gia cao",
        "expensive",
        "too expensive",
        "khong dang",
    )
    DETAIL_TERMS = (
        "tu van",
        "chi tiet",
        "noi ki",
        "noi ky",
        "ki hon",
        "ky hon",
        "cau hinh",
        "thong so",
        "noi ro",
        "phan tich ky",
        "dang tien",
        "co gi hay",
        "diem manh",
        "phan tich",
    )
    ALTERNATIVE_TERMS = (
        "re hon",
        "cung tam gia",
        "cung tam",
        "tam gia",
        "mau re hon",
        "san pham re hon",
        "lua chon khac",
        "mau khac",
        "thay the",
        "so sanh",
        "gia tot hon",
        "tiet kiem hon",
        "alternative",
        "compare",
    )

    def __init__(self, catalog: CatalogService) -> None:
        self.catalog = catalog

    def is_price_objection(self, query: str) -> bool:
        normalized = _normalize(query)
        return any(term in normalized for term in self.PRICE_OBJECTION_TERMS)

    def is_detail_request(self, query: str) -> bool:
        normalized = _normalize(query)
        return any(term in normalized for term in self.DETAIL_TERMS)

    def is_alternative_request(self, query: str) -> bool:
        normalized = _normalize(query)
        return any(term in normalized for term in self.ALTERNATIVE_TERMS)

    def classify_intent(self, query: str) -> AdvisoryIntent:
        if self.is_alternative_request(query):
            return "alternatives"
        if self.is_price_objection(query):
            return "price_objection"
        return "detail"

    def answer_missing_field_or_known_fact(
        self,
        query: str,
        product: CatalogProduct,
    ) -> AdvisoryResult | None:
        requested_fields = self._requested_fact_fields(query)
        if not requested_fields:
            return None

        specs = _spec_map(product)
        labels = {
            "weight": "trọng lượng",
            "battery": "pin",
            "warranty": "bảo hành",
            "durability": "độ bền",
        }
        values = {
            "weight": specs.get("weight"),
            "battery": specs.get("battery"),
            "warranty": specs.get("warranty"),
            "durability": None,
        }

        lines = [f"Mình đang xem đúng mẫu {product.name} ({product.code})."]
        available_lines: list[str] = []
        missing_labels: list[str] = []
        for field in requested_fields:
            value = values.get(field)
            if value:
                if field == "weight":
                    available_lines.append(f"- Trọng lượng catalog ghi nhận: {value}.")
                elif field == "battery":
                    available_lines.append(f"- Pin catalog ghi nhận: {value}.")
                elif field == "warranty":
                    available_lines.append(f"- Bảo hành catalog ghi nhận: {value}.")
            else:
                missing_labels.append(labels[field])

        if available_lines:
            lines.append("Thông tin catalog hiện có:")
            lines.extend(available_lines)
        if missing_labels:
            lines.append(
                "Catalog hiện chưa có dữ liệu "
                + "/".join(missing_labels)
                + " của mẫu này, nên mình chưa thể khẳng định phần đó."
            )
        return AdvisoryResult(text="\n".join(lines), product_codes=(product.code,))

    @staticmethod
    def _requested_fact_fields(query: str) -> tuple[str, ...]:
        normalized = _normalize(query)
        detected: list[str] = []
        if any(term in normalized for term in ("bao kg", "nang bao nhieu", "trong luong", "kg")):
            detected.append("weight")
        if any(term in normalized for term in ("pin", "thoi luong")):
            detected.append("battery")
        if any(term in normalized for term in ("bao hanh", "warranty")):
            detected.append("warranty")
        if any(term in normalized for term in ("ben khong", "do ben", "ben bi")):
            detected.append("durability")
        return tuple(dict.fromkeys(detected))

    def answer(
        self,
        query: str,
        product: CatalogProduct,
        *,
        force_price_analysis: bool = False,
        force_intent: AdvisoryIntent | None = None,
    ) -> AdvisoryResult:
        intent: AdvisoryIntent = (
            force_intent
            or ("price_objection" if force_price_analysis else self.classify_intent(query))
        )
        if intent == "detail":
            return self._detail_answer(product)
        if intent == "price_objection":
            return self._price_objection_answer(product)
        return self._alternatives_answer(product)

    def _detail_answer(self, product: CatalogProduct) -> AdvisoryResult:
        specs = _spec_map(product)
        lines = [
            f"Dạ, mình đang nói đúng về {product.name} · SKU {product.code}, giá {product.price}.",
        ]
        validity = self._price_validity_warning(product)
        if validity:
            lines.append(validity)

        lines.append("Cấu hình catalog hiện xác nhận:")
        if product.specs:
            lines.extend(f"- {spec}." for spec in product.specs)
        else:
            lines.append("- Chưa có thông số cấu hình có nguồn.")

        if product.category == "Laptop":
            known_cpu = specs.get("core") or specs.get("chip")
            lines.append("Nhận định có thể đưa ra từ dữ liệu:")
            if known_cpu:
                lines.append(f"- Bộ xử lý được ghi nhận là {known_cpu}.")
            if specs.get("screen"):
                resolution = (
                    f", độ phân giải {specs['resolution']}"
                    if specs.get("resolution")
                    else ""
                )
                lines.append(
                    f"- Màn hình {specs['screen']}{resolution}; kích thước 14 inch thiên về sự gọn nhẹ khi di chuyển."
                )
            if specs.get("ram"):
                lines.append(
                    f"- RAM {specs['ram']} phù hợp làm việc văn phòng, mở nhiều tab và đa nhiệm phổ thông."
                )
            if specs.get("ssd"):
                lines.append(
                    f"- SSD {specs['ssd']} là mức dung lượng cân bằng cho hệ điều hành, ứng dụng và tài liệu; nếu lưu nhiều game hoặc video lớn thì cần tính thêm lưu trữ."
                )
            if specs.get("os"):
                lines.append(f"- Máy được ghi nhận đi kèm {specs['os']}.")
            missing = [
                label
                for key, label in (
                    ("gpu", "card đồ họa"),
                    ("battery", "pin"),
                    ("refresh_rate", "tần số quét"),
                )
                if not specs.get(key)
            ]
            if missing:
                lines.append(
                    "- Catalog chưa có " + ", ".join(missing)
                    + ", nên chưa đủ căn cứ kết luận khả năng đồ họa/chơi game, độ mượt màn hình hoặc thời lượng pin."
                )
        else:
            highlights = self._benefit_lines(specs)
            if highlights:
                lines.append("Nhận định có thể đưa ra từ dữ liệu:")
                lines.extend(f"- {line}" for line in highlights)

        return AdvisoryResult(text="\n".join(lines), product_codes=(product.code,))

    def _price_objection_answer(self, product: CatalogProduct) -> AdvisoryResult:
        specs = _spec_map(product)
        category_products = [
            item for item in self.catalog.products if item.category == product.category
        ]
        product_price = _price_value(product)
        cheaper_count = sum(
            1 for item in category_products if _price_value(item) < product_price
        )
        percentile = (
            round(cheaper_count / max(1, len(category_products) - 1) * 100)
            if category_products
            else 0
        )
        limitations = self._limitations(specs)
        lines = [
            f"Dạ, nhìn riêng con số {product.price} thì có thể thấy cao, "
            f"nhưng {product.name} thực tế vẫn nằm trong nhóm giá thấp của catalog.",
            (
                f"Khoảng {100 - percentile}% sản phẩm {self._category_label(product).lower()} "
                f"đang có giá cao hơn mẫu này."
            ),
        ]
        value_reasons = self._price_value_reasons(specs)
        if value_reasons:
            lines.append("Phần giá trị đáng chú ý nằm ở:")
            lines.extend(f"- {reason}" for reason in value_reasons)

        if limitations:
            lines.append(
                "Đổi lại, đây không phải mẫu dành cho hiệu năng nặng: "
                + "; ".join(limitations)
                + "."
            )
        lines.append(
            "Vì vậy, mức giá này hợp lý nếu nhu cầu chính là liên lạc, mạng xã hội, "
            "xem video và cần máy bền bỉ hằng ngày. Nếu ưu tiên chơi game nặng hoặc camera, "
            "mẫu này không phải lựa chọn mạnh nhất."
        )
        return AdvisoryResult(
            text="\n".join(lines),
            product_codes=(product.code,),
        )

    def _alternatives_answer(self, product: CatalogProduct) -> AdvisoryResult:
        alternatives = self.find_alternatives(product, limit=2)
        lines = [
            f"Dạ, nếu mục tiêu là tiết kiệm hơn so với {product.name} giá {product.price}, "
            "mình tìm được các mẫu cùng phân khúc sau:"
        ]
        if not alternatives:
            lines.append(
                "Hiện catalog chưa có mẫu rẻ hơn đủ tương đồng để so sánh công bằng."
            )
            return AdvisoryResult(text="\n".join(lines), product_codes=(product.code,))

        for alternative in alternatives:
            shared = ", ".join(alternative.shared_specs) or "cùng nhóm cấu hình"
            tradeoff = (
                f"; điểm đánh đổi: {', '.join(alternative.tradeoffs)}"
                if alternative.tradeoffs
                else ""
            )
            lines.append(
                f"- {alternative.product.name} · SKU {alternative.product.code}: "
                f"{alternative.product.price}, tiết kiệm {_format_vnd(alternative.savings)}; "
                f"giữ được {shared}{tradeoff}."
            )
        lines.append(
            "Nếu bạn cho mình biết ưu tiên pin, camera, màn hình hay hiệu năng, "
            "mình sẽ chọn một mẫu phù hợp nhất thay vì chỉ chọn mẫu rẻ nhất."
        )
        return AdvisoryResult(
            text="\n".join(lines),
            product_codes=(product.code, *(item.product.code for item in alternatives)),
        )

    def compare_products(
        self,
        products: list[CatalogProduct],
        *,
        reference_price: int | None = None,
    ) -> AdvisoryResult:
        first, second = products[:2]
        first_price = _price_value(first)
        second_price = _price_value(second)
        lines = [
            "Dạ, nếu đặt hai lựa chọn vào cùng một quyết định mua thì mình chốt như sau:",
        ]

        if reference_price:
            lines.append(
                f"Ngân sách đang xét khoảng {_format_vnd(reference_price)}. "
                f"{first.brand} gần mức này nhất ở {first.price}; "
                f"{second.brand} gần nhất là {second.price}."
            )
            tolerance = max(1_500_000, int(reference_price * 0.12))
            outside = [
                product
                for product in (first, second)
                if abs(_price_value(product) - reference_price) > tolerance
            ]
            if outside:
                lines.append(
                    "Lưu ý: catalog không có "
                    + " và ".join(product.brand for product in outside)
                    + " đúng sát tầm giá này; mình đang dùng mẫu gần nhất để so sánh công bằng."
                )

        for product in (first, second):
            specs = _spec_map(product)
            details: list[str] = []
            cpu = specs.get("core") or specs.get("chip")
            if cpu:
                details.append(f"CPU {cpu}")
            if specs.get("ram"):
                details.append(f"RAM {specs['ram']}")
            if specs.get("ssd"):
                details.append(f"SSD {specs['ssd']}")
            if specs.get("gpu"):
                details.append(f"đồ họa {specs['gpu']}")
            if specs.get("screen"):
                screen = f"màn hình {specs['screen']}"
                if specs.get("resolution"):
                    screen += f" {specs['resolution']}"
                details.append(screen)
            detail_text = "; ".join(details) or "catalog chưa đủ thông số sâu"
            lines.append(
                f"- {product.brand}: {product.name} — {product.price} "
                f"(SKU {product.code}). {detail_text}."
            )

        brands = {product.brand for product in (first, second)}
        if brands == {"Dell", "Apple"}:
            dell = next(product for product in (first, second) if product.brand == "Dell")
            apple = next(product for product in (first, second) if product.brand == "Apple")
            premium = _price_value(apple) - _price_value(dell)
            lines.extend(
                [
                    "Nên chọn Dell nếu bạn muốn giữ ngân sách, cần Windows và ưu tiên tính thực dụng cho học tập/văn phòng.",
                    (
                        "Nên chọn MacBook nếu bạn chủ động muốn macOS hoặc hệ sinh thái Apple "
                        f"và chấp nhận chi thêm khoảng {_format_vnd(abs(premium))}."
                    ),
                    (
                        f"Kết luận: với tiêu chí chính là cùng tầm giá, mình nghiêng về Dell {dell.code}. "
                        f"MacBook {apple.code} chỉ đáng chuyển sang khi macOS/hệ sinh thái Apple là ưu tiên thực sự."
                    ),
                ]
            )
        else:
            cheaper = min((first, second), key=_price_value)
            lines.append(
                f"Kết luận theo dữ liệu hiện có: {cheaper.name} dễ chốt hơn nếu ưu tiên hiệu quả chi phí. "
                "Nếu bạn cho biết phần mềm sử dụng và nhu cầu di chuyển, mình có thể chốt sâu hơn."
            )

        return AdvisoryResult(
            text="\n".join(lines),
            product_codes=(first.code, second.code),
        )

    def performance_value_answer(
        self,
        rankings: list[ValueRanking],
        *,
        budget: int | None,
        profile: str | None,
        office_alternative: ValueRanking | None = None,
    ) -> AdvisoryResult:
        if not rankings:
            return AdvisoryResult(
                text=(
                    "Dạ, mình chưa có đủ sản phẩm và dữ liệu cấu hình để xếp hạng "
                    "hiệu năng/giá một cách đáng tin cậy trong mức ngân sách này."
                ),
                product_codes=(),
            )

        winner = rankings[0]
        profile_labels = {
            "gaming": "chơi game",
            "programming": "lập trình",
            "creative": "đồ họa và dựng nội dung",
            "office": "văn phòng và học tập",
            "overall": "hiệu năng tổng thể",
            None: "hiệu năng tổng thể",
        }
        profile_label = profile_labels.get(profile, "hiệu năng tổng thể")
        budget_text = f" quanh {_format_vnd(budget)}" if budget else ""
        lines = [
            (
                f"Dạ, nếu ưu tiên {profile_label}{budget_text}, "
                f"mình chốt {winner.product.name} (SKU {winner.product.code}) "
                f"là lựa chọn có tỷ lệ hiệu năng/giá tốt nhất trong catalog hiện tại."
            )
        ]
        if profile is None or profile == "overall":
            lines.append(
                "Mình đang hiểu “hiệu năng” theo nghĩa tổng thể gồm CPU, GPU, RAM và SSD; "
                "nếu nhu cầu chỉ là văn phòng thì thứ tự có thể thay đổi."
            )

        lines.append("Vì sao mình chọn mẫu này:")
        for reason in winner.reasons:
            lines.append(f"- {reason}.")
        lines.append(
            f"- Giá {winner.product.price} vẫn nằm sát ngân sách, nên phần hiệu năng nhận lại trên mỗi đồng chi ra tốt hơn các lựa chọn còn lại trong nhóm."
        )

        if winner.tradeoffs:
            lines.append("Điểm cần chấp nhận:")
            lines.extend(f"- {tradeoff}." for tradeoff in winner.tradeoffs)

        if winner.confidence < 0.8:
            lines.append(
                "Mức tin cậy của xếp hạng hiện ở mức khá, chưa phải tuyệt đối, "
                "vì catalog vẫn còn thiếu một số benchmark hoặc thông số linh kiện."
            )

        if office_alternative and office_alternative.product.code != winner.product.code:
            lines.extend(
                [
                    (
                        f"Nếu bạn không chơi game và chủ yếu làm văn phòng, "
                        f"{office_alternative.product.name} (SKU {office_alternative.product.code}) "
                        "là phương án cân bằng hơn nhờ ưu tiên CPU, RAM và tính thực dụng thay vì card rời."
                    ),
                ]
            )

        if len(rankings) > 1:
            runner_up = rankings[1]
            lines.append(
                f"Phương án đứng sau là {runner_up.product.name} giá {runner_up.product.price}; "
                "mẫu này đáng cân nhắc nếu các điểm đánh đổi của lựa chọn đầu không phù hợp."
            )

        lines.append(
            "Bạn dùng máy chủ yếu để chơi game, làm văn phòng hay lập trình? "
            "Mình sẽ chốt lại theo đúng kiểu hiệu năng bạn thực sự cần."
        )
        return AdvisoryResult(
            text="\n".join(lines),
            product_codes=tuple(item.product.code for item in rankings[:3]),
        )

    def catalog_ranking_answer(
        self,
        rankings: list[CatalogCapabilityRanking],
        *,
        goal: str,
        use_case: str | None,
    ) -> AdvisoryResult:
        if not rankings:
            return AdvisoryResult(
                text=(
                    "Mình chưa có đủ sản phẩm cùng loại và dữ liệu cấu hình "
                    "để xếp hạng một cách đáng tin cậy."
                ),
                product_codes=(),
            )

        winner = rankings[0]
        score_margin = (
            winner.score - rankings[1].score if len(rankings) > 1 else 100.0
        )
        close_race = (
            goal in {"max_performance", "best_overall"}
            and score_margin < 4.0
        )
        if goal == "lowest_price":
            opening = (
                f"Mẫu có giá thấp nhất trong nhóm đang xét là "
                f"**{winner.product.name}** (SKU {winner.product.code}), "
                f"giá **{winner.product.price}**."
            )
        elif goal == "highest_price":
            opening = (
                f"Mẫu có giá cao nhất trong nhóm đang xét là "
                f"**{winner.product.name}** (SKU {winner.product.code}), "
                f"giá **{winner.product.price}**."
            )
        elif goal == "max_performance":
            use_case_text = {
                "gaming": "cho game và tác vụ đồ họa nặng",
                "creative": "cho đồ họa và dựng nội dung",
                "programming": "cho khối lượng lập trình nặng",
            }.get(use_case, "theo năng lực phần cứng tổng thể")
            opening = (
                (
                    "Hai mẫu dẫn đầu đang rất sát nhau. "
                    f"Nếu hiểu “khỏe nhất” là **{use_case_text}**, "
                    f"**{winner.product.name}** (SKU {winner.product.code}) "
                    "đang nhỉnh nhẹ theo dữ liệu cấu hình"
                )
                if close_race
                else (
                    f"Nếu hiểu “khỏe nhất” là **{use_case_text}**, mẫu đứng đầu "
                    f"catalog hiện tại là **{winner.product.name}** "
                    f"(SKU {winner.product.code})"
                )
            ) + f", giá **{winner.product.price}**."
        else:
            opening = (
                (
                    "Hai mẫu cao cấp nhất đang gần như ngang điểm. "
                    "Nếu hiểu “xịn nhất” là cấu hình cao cấp và toàn diện nhất "
                    f"theo dữ liệu shop, **{winner.product.name}** "
                    f"(SKU {winner.product.code}) đang nhỉnh nhẹ"
                )
                if close_race
                else (
                    "Nếu hiểu “xịn nhất” là cấu hình cao cấp và toàn diện nhất "
                    f"theo dữ liệu shop, mình chọn **{winner.product.name}** "
                    f"(SKU {winner.product.code})"
                )
            ) + f", giá **{winner.product.price}**."

        lines = [opening]
        if goal not in {"lowest_price", "highest_price"}:
            lines.extend(["", "Vì sao mẫu này đứng đầu:"])
            lines.extend(f"- {reason}." for reason in winner.reasons[:5])

        if len(rankings) > 1:
            lines.extend(["", "Hai lựa chọn sát phía sau:"])
            for index, ranking in enumerate(rankings[1:3], start=2):
                reason = "; ".join(ranking.reasons[:3]) or "dữ liệu cấu hình còn hạn chế"
                lines.append(
                    f"- #{index} **{ranking.product.name}** — "
                    f"{ranking.product.price}: {reason}."
                )

        if winner.cautions:
            lines.extend(["", "Lưu ý để hiểu đúng kết quả:"])
            lines.extend(f"- {caution}." for caution in winner.cautions)
        if goal in {"max_performance", "best_overall"}:
            lines.append(
                "Nếu bạn nói rõ dùng để chơi game, dựng video, AI hay cần pin/độ nhẹ, "
                "mình sẽ chấm lại theo đúng workload thay vì dùng tiêu chí tổng thể."
            )
        return AdvisoryResult(
            text="\n".join(lines),
            product_codes=tuple(item.product.code for item in rankings[:3]),
        )

    def _benefit_lines(self, specs: dict[str, str]) -> list[str]:
        lines: list[str] = []
        if specs.get("screen"):
            refresh = f", tần số quét {specs['refresh_rate']}" if specs.get("refresh_rate") else ""
            lines.append(
                f"Màn hình {specs['screen']}{refresh}, phù hợp xem video và cuộn nội dung."
            )
        if specs.get("battery"):
            lines.append(
                f"Pin {specs['battery']}, hướng tới thời lượng sử dụng cả ngày."
            )
        if specs.get("water_resistance"):
            lines.append(
                f"Chuẩn {specs['water_resistance']} giúp hạn chế rủi ro từ bụi và nước bắn nhẹ."
            )
        performance = []
        if specs.get("chip"):
            performance.append(f"chip {specs['chip']}")
        elif specs.get("core"):
            performance.append(f"CPU {specs['core']}")
        if specs.get("ram"):
            performance.append(f"RAM {specs['ram']}")
        if specs.get("storage"):
            performance.append(f"bộ nhớ {specs['storage']}")
        if performance:
            performance_text = ", ".join(performance)
            lines.append(
                performance_text[:1].upper()
                + performance_text[1:]
                + ", đáp ứng các tác vụ phổ thông."
            )
        if specs.get("camera"):
            lines.append(f"Camera {specs['camera']} phục vụ nhu cầu chụp cơ bản.")
        if specs.get("warranty"):
            lines.append(f"Bảo hành chính hãng {specs['warranty']}.")
        return lines

    def _price_value_reasons(self, specs: dict[str, str]) -> list[str]:
        reasons: list[str] = []
        if specs.get("screen") and specs.get("refresh_rate"):
            reasons.append(
                f"Màn hình {specs['screen']} với tần số quét {specs['refresh_rate']}, "
                "cho thao tác mượt hơn màn hình phổ thông 60Hz."
            )
        if specs.get("battery"):
            reasons.append(f"Pin {specs['battery']} hỗ trợ sử dụng dài trong ngày.")
        if specs.get("water_resistance"):
            reasons.append(
                f"Kháng bụi/nước {specs['water_resistance']}, hữu ích với nhu cầu di chuyển."
            )
        if specs.get("warranty"):
            reasons.append(f"Thời hạn bảo hành {specs['warranty']}.")
        return reasons

    def _fit_summary(
        self, product: CatalogProduct, specs: dict[str, str]
    ) -> str:
        ram = _number(specs.get("ram", ""))
        storage = _number(specs.get("storage", ""))
        if product.category == "Mobile Phone" and ram and ram <= 4:
            summary = (
                "Phù hợp nhất với người dùng phổ thông, học sinh, người lớn tuổi hoặc máy phụ: "
                "gọi điện, nhắn tin, mạng xã hội, xem video và ứng dụng nhẹ."
            )
            if storage and storage <= 64:
                summary += " Không phù hợp nếu cần lưu nhiều game, video hoặc đa nhiệm nặng."
            return summary
        return (
            "Phù hợp với nhu cầu sử dụng hằng ngày; mức độ phù hợp cuối cùng phụ thuộc "
            "ứng dụng và khối lượng công việc của bạn."
        )

    @staticmethod
    def _price_validity_warning(product: CatalogProduct) -> str | None:
        if product.fetched_at:
            try:
                fetched_at = datetime.fromisoformat(
                    str(product.fetched_at).replace("Z", "+00:00")
                )
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=UTC)
                if datetime.now(UTC) - fetched_at <= timedelta(hours=36):
                    return None
            except (TypeError, ValueError):
                pass
        if not product.price_valid_until:
            return None
        try:
            valid_until = date.fromisoformat(str(product.price_valid_until))
        except (TypeError, ValueError):
            return None
        if valid_until < date.today():
            return (
                f"Lưu ý: giá nguồn chỉ được xác nhận đến {product.price_valid_until}; "
                "cần cập nhật catalog trước khi chốt đơn."
            )
        return None

    def find_alternatives(
        self, product: CatalogProduct, *, limit: int = 2
    ) -> list[Alternative]:
        target_price = _price_value(product)
        target_specs = _spec_map(product)
        candidates: list[Alternative] = []

        for candidate in self.catalog.products:
            candidate_price = _price_value(candidate)
            if (
                candidate.code == product.code
                or candidate.category != product.category
                or candidate_price <= 0
                or candidate_price >= target_price
                or candidate_price < target_price * 0.55
            ):
                continue

            specs = _spec_map(candidate)
            score = 0.0
            shared: list[str] = []
            tradeoffs: list[str] = []

            for key, weight, label in (
                ("chip", 5.0, "cùng chip"),
                ("core", 4.0, "cùng CPU"),
                ("ram", 3.0, "cùng RAM"),
                ("storage", 3.0, "cùng bộ nhớ"),
                ("ssd", 2.0, "cùng SSD"),
                ("screen", 1.5, "màn hình cùng cỡ"),
                ("camera", 1.0, "camera cùng độ phân giải"),
                ("battery", 1.5, "pin cùng dung lượng"),
                ("refresh_rate", 1.0, "cùng tần số quét"),
                ("water_resistance", 1.0, "cùng chuẩn kháng nước"),
                ("nfc", 0.5, "cùng NFC"),
            ):
                target_value = target_specs.get(key)
                candidate_value = specs.get(key)
                if not target_value or not candidate_value:
                    continue
                if target_value == candidate_value:
                    score += weight
                    shared.append(label)
                elif key in {"ram", "storage", "ssd", "battery", "refresh_rate"}:
                    target_number = _number(target_value)
                    candidate_number = _number(candidate_value)
                    if candidate_number >= target_number > 0:
                        score += weight * 0.8
                        comparison_label = {
                            "ram": "RAM không thấp hơn",
                            "storage": "bộ nhớ không thấp hơn",
                            "ssd": "SSD không thấp hơn",
                            "battery": "pin không thấp hơn",
                            "refresh_rate": "tần số quét không thấp hơn",
                        }[key]
                        shared.append(comparison_label)
                    elif candidate_number and target_number:
                        tradeoff_label = {
                            "ram": "RAM",
                            "storage": "bộ nhớ",
                            "ssd": "SSD",
                            "battery": "pin",
                            "refresh_rate": "tần số quét",
                        }[key]
                        tradeoffs.append(f"{tradeoff_label} {candidate_value}")

            if candidate.brand == product.brand:
                score += 0.5
            if score < 3:
                continue
            candidates.append(
                Alternative(
                    product=candidate,
                    similarity_score=score,
                    savings=target_price - candidate_price,
                    shared_specs=tuple(shared[:3]),
                    tradeoffs=tuple(tradeoffs[:2]),
                )
            )

        candidates.sort(
            key=lambda item: (
                -item.similarity_score,
                -item.savings,
                _price_value(item.product),
                item.product.code,
            )
        )
        selected: list[Alternative] = []
        seen_signatures: set[tuple[str, tuple[str, ...]]] = set()
        for item in candidates:
            signature = (item.product.brand, item.product.specs)
            if signature in seen_signatures:
                continue
            selected.append(item)
            seen_signatures.add(signature)
            if len(selected) >= limit:
                break
        return selected

    def _has_synthetic_price_anomaly(self, product: CatalogProduct) -> bool:
        target_specs = _spec_map(product)
        if not target_specs:
            return False
        minimum_price: int | None = None
        maximum_price: int | None = None
        comparable_count = 0
        for candidate in self.catalog.products:
            if candidate.category != product.category:
                continue
            specs = _spec_map(candidate)
            if all(
                not value or specs.get(key) == value
                for key, value in target_specs.items()
            ):
                price = _price_value(candidate)
                if price > 0:
                    comparable_count += 1
                    minimum_price = (
                        price if minimum_price is None else min(minimum_price, price)
                    )
                    maximum_price = (
                        price if maximum_price is None else max(maximum_price, price)
                    )
                    if (
                        comparable_count >= 3
                        and maximum_price / max(1, minimum_price) >= 2.5
                    ):
                        return True
        return False

    def _strengths(
        self, product: CatalogProduct, specs: dict[str, str]
    ) -> list[str]:
        strengths: list[str] = []
        raw_specs = list(product.specs)
        for key, label in (
            ("chip", "chip"),
            ("core", "CPU"),
            ("ram", "RAM"),
            ("storage", "bộ nhớ trong"),
            ("ssd", "SSD"),
            ("screen", "màn hình"),
            ("camera", "camera"),
            ("battery", "pin"),
            ("refresh_rate", "tần số quét"),
            ("water_resistance", "kháng nước"),
            ("warranty", "bảo hành"),
        ):
            value = specs.get(key)
            if value:
                original = next(
                    (
                        spec
                        for spec in raw_specs
                        if _normalize(spec).endswith(value)
                    ),
                    value,
                )
                strengths.append(original)
        return strengths[:3]

    def _limitations(self, specs: dict[str, str]) -> list[str]:
        limitations: list[str] = []
        storage = _number(specs.get("storage", ""))
        ram = _number(specs.get("ram", ""))
        if storage and storage <= 64:
            limitations.append("bộ nhớ 64GB phù hợp nhu cầu cơ bản nhưng nhanh đầy nếu lưu nhiều video hoặc game")
        if ram and ram < 8:
            limitations.append(f"RAM {ram}GB phù hợp ứng dụng nhẹ, hạn chế khi đa nhiệm hoặc chơi game nặng")
        return limitations

    @staticmethod
    def _category_label(product: CatalogProduct) -> str:
        return "Laptop" if product.category == "Laptop" else "Điện thoại"
