"""Deterministic decision engines used before optional LLM phrasing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.services.advisor import AdvisoryResult, _format_vnd, _spec_map
from backend.services.catalog import CatalogProduct, _price_value
from backend.services.value_engine import ValueScoringEngine


@dataclass(frozen=True)
class DecisionScore:
    product: CatalogProduct
    score: float
    confidence: float
    reasons: tuple[str, ...]
    cautions: tuple[str, ...]


@dataclass(frozen=True)
class DecisionPacket:
    answer_type: str
    products: tuple[CatalogProduct, ...]
    scores: tuple[DecisionScore, ...]
    recommendation_code: str | None
    facts: tuple[str, ...]
    warnings: tuple[str, ...]
    decision_confidence: float = 1.0
    abstained: bool = False


@dataclass(frozen=True)
class PacketVerification:
    approved: bool
    issues: tuple[str, ...]


class DecisionPacketVerifier:
    """Hard guard against candidate, SKU, price, and recommendation drift."""

    # Real catalog SKUs are eight digits beginning with zero. Model names such
    # as EP1179TU are also eight characters and must not be treated as unknown
    # SKUs during verification.
    SKU_PATTERN = re.compile(r"\b0\d{7}\b")

    def verify(
        self,
        packet: DecisionPacket,
        advisory: AdvisoryResult,
        *,
        baseline_text: str | None = None,
    ) -> PacketVerification:
        allowed_codes = {product.code for product in packet.products}
        returned_codes = set(advisory.product_codes)
        mentioned_codes = set(self.SKU_PATTERN.findall(advisory.text.upper()))
        issues: list[str] = []

        if not returned_codes.issubset(allowed_codes):
            issues.append("response contains products outside DecisionPacket")
        if not mentioned_codes.issubset(allowed_codes):
            issues.append("response text mentions an unknown SKU")
        if (
            packet.recommendation_code
            and packet.recommendation_code not in allowed_codes
        ):
            issues.append("recommendation is outside DecisionPacket")
        if packet.recommendation_code and len(packet.products) >= 3:
            score_margin = (
                packet.scores[0].score - packet.scores[1].score
                if len(packet.scores) >= 2
                else 0.0
            )
            if packet.decision_confidence < 0.6 or score_margin < 4.0:
                issues.append(
                    "multi-product recommendation lacks discriminating evidence"
                )
        if packet.abstained and packet.recommendation_code:
            issues.append("abstained packet cannot contain a recommendation")
        if (
            baseline_text is not None
            and packet.recommendation_code
            and packet.recommendation_code not in advisory.text
        ):
            issues.append("phrased response omitted the locked recommendation SKU")

        known_price_digits = {
            re.sub(r"\D", "", product.price)
            for product in packet.products
            if re.sub(r"\D", "", product.price)
        }
        for raw_price in re.findall(
            r"\b\d{1,3}(?:\.\d{3}){2,3}\s*(?:VNĐ|VND)?\b",
            advisory.text,
            flags=re.IGNORECASE,
        ):
            digits = re.sub(r"\D", "", raw_price)
            # Price gaps are allowed in causal explanations; exact sale prices
            # must belong to one of the locked products.
            if digits not in known_price_digits and "khoảng" not in advisory.text[
                max(0, advisory.text.find(raw_price) - 12):advisory.text.find(raw_price)
            ].casefold():
                issues.append(f"unverified price mentioned: {raw_price.strip()}")

        if baseline_text is not None:
            number_pattern = re.compile(
                r"\b\d+(?:[.,]\d+)*(?:\s*(?:GB|TB|Hz|inch|VNÄ|VND|W|Wh|mAh))?\b",
                flags=re.IGNORECASE,
            )
            baseline_numbers = {
                re.sub(r"\s+", " ", item).casefold()
                for item in number_pattern.findall(baseline_text)
            }
            phrased_numbers = {
                re.sub(r"\s+", " ", item).casefold()
                for item in number_pattern.findall(advisory.text)
            }
            invented_numbers = phrased_numbers - baseline_numbers
            if invented_numbers:
                issues.append(
                    "phrased response introduced numeric facts: "
                    + ", ".join(sorted(invented_numbers))
                )

        return PacketVerification(
            approved=not issues,
            issues=tuple(dict.fromkeys(issues)),
        )


class DurabilityScorer:
    """Score only durability evidence present on the sourced product page."""

    EVIDENCE = (
        (r"\bmil[- ]?std\b|\bmilitary grade\b", 35, "có tiêu chuẩn độ bền được nguồn sản phẩm đề cập"),
        (r"\bvo kim loai\b|\bnhom\b|\baluminum\b|\bmagnesium\b", 18, "có thông tin vật liệu kim loại"),
        (r"\bbao hanh\s+(?:24|36)\s*thang\b", 20, "có thời hạn bảo hành từ 24 tháng"),
        (r"\bnang cap\b|\b2 khe ram\b|\b2 slot\b", 12, "có thông tin hỗ trợ nâng cấp/bảo trì"),
        (r"\bchong soc\b|\bchiu luc\b", 10, "có thông tin chống sốc/chịu lực cụ thể"),
    )

    def score(self, product: CatalogProduct) -> tuple[float, float, tuple[str, ...]]:
        # Generic marketing prose such as "bền bỉ" is not evidence. Only
        # structured specifications may affect the durability score.
        searchable = _ascii(" ".join(product.specs))
        score = 0.0
        reasons: list[str] = []
        for pattern, points, reason in self.EVIDENCE:
            if re.search(pattern, searchable):
                score += points
                reasons.append(reason)
        confidence = min(0.95, 0.25 + len(reasons) * 0.18)
        return min(100.0, score), confidence, tuple(reasons)


class PreferenceReRanker:
    def __init__(self) -> None:
        self.value_engine = ValueScoringEngine()
        self.durability = DurabilityScorer()

    def rank(
        self,
        products: list[CatalogProduct],
        *,
        preferences: dict[str, float],
        use_case: str | None,
    ) -> list[DecisionScore]:
        profile = use_case if use_case in {"gaming", "programming", "creative", "office"} else "overall"
        value_by_code = {
            ranking.product.code: ranking
            for ranking in self.value_engine.rank(products, profile=profile)
        }
        performance_weight = 0.55
        durability_weight = 0.0
        value_weight = 0.25
        if preferences.get("durability"):
            durability_weight = 0.35
            performance_weight = 0.45
            value_weight = 0.20
        if preferences.get("gaming"):
            performance_weight = 0.70
            durability_weight *= 0.7
        if preferences.get("value"):
            value_weight = 0.40
            performance_weight = max(0.40, performance_weight - 0.15)

        scored: list[DecisionScore] = []
        max_value = max(
            (ranking.value_score for ranking in value_by_code.values()),
            default=1.0,
        )
        for product in products:
            ranking = value_by_code.get(product.code)
            durability, durability_confidence, durability_reasons = self.durability.score(product)
            performance = ranking.performance_score if ranking else 0.0
            normalized_value = (
                ranking.value_score / max_value * 100 if ranking else 0.0
            )
            score = (
                performance * performance_weight
                + durability * durability_weight
                + normalized_value * value_weight
            )
            reasons = list(ranking.reasons if ranking else ())
            reasons.extend(durability_reasons)
            cautions = list(ranking.tradeoffs if ranking else ())
            if preferences.get("durability") and not durability_reasons:
                cautions.append("chưa có đủ dữ liệu có nguồn để kết luận độ bền")
            confidence = (
                (ranking.confidence if ranking else 0.35) * 0.75
                + durability_confidence * 0.25
            )
            scored.append(
                DecisionScore(
                    product=product,
                    score=round(score, 2),
                    confidence=round(confidence, 2),
                    reasons=tuple(dict.fromkeys(reasons))[:4],
                    cautions=tuple(dict.fromkeys(cautions))[:3],
                )
            )
        return sorted(
            scored,
            key=lambda item: (
                -item.score,
                -item.confidence,
                _price_value(item.product.price),
            ),
        )


class ProductComparisonEngine:
    def __init__(self) -> None:
        self.reranker = PreferenceReRanker()

    def compare(
        self,
        products: list[CatalogProduct],
        *,
        preferences: dict[str, float],
        use_case: str | None,
        budget: int | None,
    ) -> tuple[AdvisoryResult, DecisionPacket]:
        products = products[:4]
        scores = self.reranker.rank(
            products,
            preferences=preferences,
            use_case=use_case,
        )
        if len(scores) < 2:
            text = "Mình chưa có đủ hai sản phẩm đúng ngữ cảnh để so sánh công bằng."
            return (
                AdvisoryResult(text=text, product_codes=tuple(p.code for p in products)),
                DecisionPacket("comparison", tuple(products), tuple(scores), None, (), (text,)),
            )

        if not preferences and not use_case:
            scores = sorted(
                scores,
                key=lambda item: (
                    _price_value(item.product.price),
                    -item.confidence,
                ),
            )
        winner = scores[0]
        runner_up = scores[1]
        explicit_priorities = bool(preferences) or bool(use_case)
        score_margin = winner.score - runner_up.score
        should_recommend = (
            explicit_priorities
            and winner.confidence >= 0.62
            and score_margin >= 4.0
        )
        if len(scores) == 2 and not explicit_priorities:
            verified_price_advantage = (
                _price_value(runner_up.product.price)
                > _price_value(winner.product.price)
            )
            should_recommend = winner.confidence >= 0.49 and (
                bool(winner.reasons) or verified_price_advantage
            )

        preference_labels = {
            "performance": "hiệu năng",
            "durability": "độ bền",
            "battery": "pin",
            "display": "màn hình",
            "portability": "tính di động",
            "gaming": "chơi game",
            "value": "giá trị/giá",
        }
        priorities = [
            preference_labels[key]
            for key, value in preferences.items()
            if value and key in preference_labels
        ]
        priority_text = ", ".join(priorities) or (
            {"gaming": "chơi game", "office": "văn phòng", "programming": "lập trình"}.get(
                use_case or "", ""
            )
        )
        if should_recommend:
            lines = [
                f"Nếu ưu tiên **{priority_text or 'giá trị sử dụng'}**, mình nghiêng về "
                f"{winner.product.brand}: **{winner.product.name}** "
                f"(SKU {winner.product.code}).",
                "",
                "Lý do có thể kiểm chứng:",
            ]
            common_reasons = set.intersection(
                *(set(score.reasons) for score in scores)
            )
            differentiating = [
                reason for reason in winner.reasons if reason not in common_reasons
            ]
            for reason in differentiating[:3]:
                lines.append(f"- {reason}.")
            lines.append(f"- Giá hiện tại: {winner.product.price}.")
        else:
            subject = "Các mẫu này"
            lines = [
                f"{subject} không có một lựa chọn thắng tuyệt đối với dữ liệu hiện có.",
                "Mình tách theo nhu cầu để bạn chọn đúng, thay vì ép một mẫu đứng đầu.",
            ]

        lines.extend(["", "So sánh trực tiếp:"])
        dimensions = (
            ("chip", "chip"),
            ("core", "CPU"),
            ("ram", "RAM"),
            ("storage", "bộ nhớ"),
            ("battery", "pin"),
            ("refresh_rate", "tần số quét"),
            ("camera", "camera"),
            ("water_resistance", "kháng nước"),
            ("warranty", "bảo hành"),
            ("screen", "màn hình"),
        )
        for score in scores:
            specs = _spec_map(score.product)
            facts = [
                f"{label} {specs[key]}"
                for key, label in dimensions
                if specs.get(key)
            ]
            lines.append(
                f"- **{score.product.name}** — {score.product.price}: "
                + "; ".join(facts[:9])
                + "."
            )

        if not should_recommend:
            lines.extend(["", "Chốt theo nhu cầu:"])
            cheapest = min(scores, key=lambda item: _price_value(item.product.price))
            lines.append(
                f"- Muốn tiết kiệm chi phí ban đầu: **{cheapest.product.name}** "
                f"({cheapest.product.price})."
            )
            for key, label in (
                ("storage", "bộ nhớ rộng"),
                ("battery", "pin lớn"),
                ("camera", "camera độ phân giải cao"),
                ("warranty", "bảo hành dài"),
            ):
                best = max(scores, key=lambda item: _numeric_spec(item.product, key))
                value = _spec_map(best.product).get(key)
                if value:
                    lines.append(f"- Ưu tiên {label}: **{best.product.name}** ({value}).")

        missing_chips = [
            score.product.brand
            for score in scores
            if not (
                _spec_map(score.product).get("chip")
                or _spec_map(score.product).get("core")
            )
        ]
        if missing_chips:
            lines.extend([
                "",
                "Giới hạn dữ liệu:",
                "- Chưa có thông tin chip đồng nhất cho "
                + ", ".join(missing_chips)
                + ", nên chưa thể kết luận máy nào mạnh nhất.",
            ])
        if preferences.get("durability") and not any(
            "độ bền" in reason or "vật liệu" in reason or "bảo hành" in reason
            for reason in winner.reasons
        ):
            lines.append(
                "- Catalog chưa có đủ bằng chứng vật liệu/tiêu chuẩn kiểm định để khẳng định máy nào bền hơn tuyệt đối."
            )
        lines.extend([
            "",
            (
                f"Kết luận: **{winner.product.name}** hợp lý hơn theo đúng ưu tiên đã nêu."
                if should_recommend
                else "Kết luận: chưa nên chốt theo hiệu năng/giá khi chip của các mẫu chưa đầy đủ; "
                "hãy chọn theo pin, bộ nhớ, camera và bảo hành ở trên."
            ),
        ])
        packet = DecisionPacket(
            answer_type="comparison",
            products=tuple(score.product for score in scores),
            scores=tuple(scores),
            recommendation_code=winner.product.code if should_recommend else None,
            facts=tuple(winner.reasons) if should_recommend else (),
            warnings=tuple(winner.cautions),
            decision_confidence=winner.confidence,
            abstained=not should_recommend,
        )
        return (
            AdvisoryResult(
                text="\n".join(lines),
                product_codes=tuple(score.product.code for score in scores),
            ),
            packet,
        )


def _numeric_spec(product: CatalogProduct, key: str) -> int:
    value = _spec_map(product).get(key, "")
    match = re.search(r"\d+", value.replace(".", ""))
    return int(match.group()) if match else 0


class PriceCausalityExplainer:
    def explain(self, products: list[CatalogProduct]) -> tuple[AdvisoryResult, DecisionPacket]:
        products = products[:2]
        if len(products) < 2:
            text = (
                "Mình cần đúng hai sản phẩm hoặc hai hãng đang so sánh để giải thích "
                "chênh lệch giá, thay vì suy đoán."
            )
            return (
                AdvisoryResult(text=text, product_codes=tuple(p.code for p in products)),
                DecisionPacket("price_causality", tuple(products), (), None, (), (text,)),
            )
        cheaper, expensive = sorted(
            products, key=lambda product: _price_value(product.price)
        )
        gap = _price_value(expensive.price) - _price_value(cheaper.price)
        cheap_specs = _spec_map(cheaper)
        expensive_specs = _spec_map(expensive)
        differences: list[str] = []
        for key, label in (
            ("core", "CPU"),
            ("chip", "chip"),
            ("gpu", "GPU"),
            ("ram", "RAM"),
            ("ssd", "SSD"),
            ("screen", "màn hình"),
            ("resolution", "độ phân giải"),
            ("os", "hệ điều hành"),
            ("warranty", "bảo hành"),
        ):
            cheap_value = cheap_specs.get(key)
            expensive_value = expensive_specs.get(key)
            if cheap_value and expensive_value and cheap_value != expensive_value:
                differences.append(
                    f"{label}: {cheaper.brand} có {cheap_value}, còn {expensive.brand} có {expensive_value}"
                )
        lines = [
            f"{cheaper.name} rẻ hơn {expensive.name} khoảng **{_format_vnd(gap)}**.",
            "",
        ]
        if differences:
            lines.append("Những khác biệt catalog có thể kiểm chứng:")
            lines.extend(f"- {difference}." for difference in differences[:6])
        else:
            lines.append(
                "Catalog hiện chưa có đủ thông số đồng nhất để quy toàn bộ chênh lệch giá cho hiệu năng."
            )
        lines.extend([
            "",
            "Điểm quan trọng: “cùng hiệu năng” còn phụ thuộc loại tác vụ. Hai máy có thể gần nhau ở CPU "
            "nhưng khác GPU, màn hình, pin, hệ điều hành, vật liệu hoặc dịch vụ đi kèm.",
            "",
            f"Kết luận: mức giá thấp hơn của {cheaper.brand} không tự động có nghĩa là kém hơn. "
            "Nếu các tác vụ bạn dùng cho kết quả tương đương, mẫu rẻ hơn thường có lợi thế về hiệu quả chi phí; "
            "phần chênh còn lại chỉ đáng trả khi các khác biệt nêu trên thực sự có ích cho bạn.",
        ])
        packet = DecisionPacket(
            answer_type="price_causality",
            products=(cheaper, expensive),
            scores=(),
            recommendation_code=cheaper.code,
            facts=tuple(differences),
            warnings=(
                "Không suy diễn vật liệu, độ bền hoặc dịch vụ nếu catalog không có bằng chứng.",
            ),
        )
        return (
            AdvisoryResult(
                text="\n".join(lines),
                product_codes=(cheaper.code, expensive.code),
            ),
            packet,
        )


def _ascii(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
