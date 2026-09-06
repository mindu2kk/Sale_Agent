"""Deterministic performance/value ranking for catalog products."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from backend.services.catalog import CatalogProduct, _price_value


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "component_benchmarks.json"
)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD", value.casefold().replace("đ", "d")
    )
    ascii_value = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


def _number(value: str) -> int:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else 0


def _display_component(value: str) -> str:
    replacements = {
        "core i5 13420h": "Intel Core i5-13420H",
        "core 5 210h": "Intel Core 5 210H",
        "ryzen 7 7445hs": "AMD Ryzen 7 7445HS",
        "ryzen 5 7535hs": "AMD Ryzen 5 7535HS",
        "ryzen 5 7533hs": "AMD Ryzen 5 7533HS",
        "geforce rtx 3050 4gb": "NVIDIA GeForce RTX 3050 4GB",
        "geforce rtx 3050 6gb": "NVIDIA GeForce RTX 3050 6GB",
    }
    return replacements.get(value, value.upper())


def _find_spec(product: CatalogProduct, labels: tuple[str, ...]) -> str:
    normalized_labels = tuple(_normalize(label) for label in labels)
    for spec in product.specs:
        normalized = _normalize(spec)
        for label in normalized_labels:
            if normalized.startswith(label + " "):
                return spec[len(label) + 1 :].strip()
    return ""


@lru_cache(maxsize=1)
def _benchmarks() -> dict:
    with BENCHMARK_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@dataclass(frozen=True)
class ValueRanking:
    product: CatalogProduct
    profile: str
    performance_score: float
    value_score: float
    confidence: float
    reasons: tuple[str, ...]
    tradeoffs: tuple[str, ...]


@dataclass(frozen=True)
class CatalogCapabilityRanking:
    product: CatalogProduct
    mode: str
    score: float
    confidence: float
    reasons: tuple[str, ...]
    cautions: tuple[str, ...]


class ValueScoringEngine:
    PROFILE_WEIGHTS = {
        "gaming": {"cpu": 0.25, "gpu": 0.55, "ram": 0.12, "ssd": 0.08},
        "programming": {"cpu": 0.48, "gpu": 0.04, "ram": 0.34, "ssd": 0.14},
        "creative": {"cpu": 0.30, "gpu": 0.38, "ram": 0.20, "ssd": 0.12},
        "office": {"cpu": 0.42, "gpu": 0.03, "ram": 0.32, "ssd": 0.15, "display": 0.08},
        "overall": {"cpu": 0.38, "gpu": 0.28, "ram": 0.20, "ssd": 0.10, "display": 0.04},
    }

    def rank(
        self,
        products: list[CatalogProduct],
        *,
        profile: str | None = None,
    ) -> list[ValueRanking]:
        effective_profile = profile if profile in self.PROFILE_WEIGHTS else "overall"
        ranked = [
            self._score_product(product, effective_profile)
            for product in products
            if _price_value(product.price) > 0
        ]
        ranked.sort(
            key=lambda item: (
                -item.value_score,
                -item.confidence,
                _price_value(item.product.price),
                item.product.code,
            )
        )
        return ranked

    def _score_product(self, product: CatalogProduct, profile: str) -> ValueRanking:
        searchable = _normalize(" ".join((product.name, *product.specs)))
        cpu_score, cpu_match = self._component_score("cpu", searchable)
        gpu_score, gpu_match = self._component_score("gpu", searchable)

        ram_value = _find_spec(product, ("RAM",))
        ram_gb = _number(ram_value)
        ram_score = min(100.0, ram_gb / 24 * 100) if ram_gb else 0.0

        ssd_value = _find_spec(product, ("Ổ cứng SSD", "SSD"))
        ssd_gb = _number(ssd_value)
        if "tb" in _normalize(ssd_value):
            ssd_gb *= 1024
        ssd_score = min(100.0, ssd_gb / 1024 * 100) if ssd_gb else 0.0

        panel = _find_spec(product, ("Tấm nền",))
        resolution = _find_spec(product, ("Độ phân giải",))
        refresh = _find_spec(product, ("Tần số quét",))
        display_score = 45.0
        if "oled" in _normalize(panel):
            display_score += 30
        elif panel:
            display_score += 12
        if resolution:
            display_score += 10
        if _number(refresh) >= 120:
            display_score += 15
        display_score = min(100.0, display_score)

        components = {
            "cpu": cpu_score,
            "gpu": gpu_score,
            "ram": ram_score,
            "ssd": ssd_score,
            "display": display_score,
        }
        weights = self.PROFILE_WEIGHTS[profile]
        # Missing evidence scores zero instead of being silently ignored.
        # This prevents sparse products from winning merely because fewer
        # components were available to evaluate.
        performance = sum(
            components[key] * weight for key, weight in weights.items()
        )
        price_millions = _price_value(product.price) / 1_000_000
        value_score = performance / max(price_millions, 0.1) * 10

        evidence_points = sum(
            (
                bool(cpu_match),
                bool(gpu_match),
                bool(ram_gb),
                bool(ssd_gb),
            )
        )
        confidence = 0.45 + evidence_points * 0.125
        confidence = min(0.95, confidence)

        reasons: list[str] = []
        tradeoffs: list[str] = []
        if cpu_match:
            reasons.append(
                f"CPU {_display_component(cpu_match)} thuộc nhóm hiệu năng tốt trong tầm giá"
            )
        if gpu_match and gpu_score >= 50:
            reasons.append(
                f"GPU {_display_component(gpu_match)} tạo lợi thế rõ cho game và tác vụ đồ họa"
            )
        if ram_gb >= 16:
            reasons.append(f"RAM {ram_gb}GB đủ tốt cho đa nhiệm")
        if ssd_gb >= 512:
            reasons.append(f"SSD {ssd_gb}GB là mức dung lượng cân bằng")

        if product.category == "Laptop" and profile in {"gaming", "creative", "overall"}:
            if gpu_score and gpu_score < 40:
                tradeoffs.append("đồ họa tích hợp, không phải lựa chọn mạnh cho game nặng")
            if not gpu_match:
                tradeoffs.append("chưa đủ dữ liệu GPU để kết luận sâu")
        if product.category == "Laptop" and not ssd_gb:
            tradeoffs.append("catalog chưa xác nhận dung lượng SSD")
        if "rtx 3050 4gb" in searchable:
            tradeoffs.append("RTX 3050 4GB là GPU rời nhưng VRAM không dư dả cho game mới")

        return ValueRanking(
            product=product,
            profile=profile,
            performance_score=round(performance, 2),
            value_score=round(value_score, 2),
            confidence=round(confidence, 2),
            reasons=tuple(reasons[:3]),
            tradeoffs=tuple(tradeoffs[:2]),
        )

    @staticmethod
    def _component_score(kind: str, searchable: str) -> tuple[float, str]:
        table = _benchmarks()[kind]
        matches = [
            (key, float(score))
            for key, score in table.items()
            if _normalize(key) in searchable
        ]
        if not matches:
            return 0.0, ""
        key, score = max(matches, key=lambda item: (len(item[0]), item[1]))
        return score, key


class CatalogRankingEngine:
    """Rank catalog extrema without confusing capability with value-for-money."""

    MODES = {"max_performance", "best_overall", "lowest_price", "highest_price"}

    def rank(
        self,
        products: list[CatalogProduct],
        *,
        mode: str,
        use_case: str | None = None,
        limit: int = 5,
    ) -> list[CatalogCapabilityRanking]:
        effective_mode = mode if mode in self.MODES else "best_overall"
        valid_products = [
            product for product in products if _price_value(product.price) > 0
        ]
        if effective_mode in {"lowest_price", "highest_price"}:
            reverse = effective_mode == "highest_price"
            ordered = sorted(
                valid_products,
                key=lambda product: (_price_value(product.price), product.code),
                reverse=reverse,
            )
            return [
                CatalogCapabilityRanking(
                    product=product,
                    mode=effective_mode,
                    score=float(_price_value(product.price)),
                    confidence=1.0,
                    reasons=(f"Giá catalog hiện tại là {product.price}",),
                    cautions=(),
                )
                for product in ordered[:limit]
            ]

        ranked = [
            self._score_capability(
                product,
                mode=effective_mode,
                use_case=use_case,
            )
            for product in valid_products
        ]
        ranked.sort(
            key=lambda item: (
                -item.score,
                -item.confidence,
                -_price_value(item.product.price),
                item.product.code,
            )
        )
        return ranked[:limit]

    def _score_capability(
        self,
        product: CatalogProduct,
        *,
        mode: str,
        use_case: str | None,
    ) -> CatalogCapabilityRanking:
        searchable = _normalize(" ".join((product.name, *product.specs)))
        scorer = ValueScoringEngine()
        cpu_score, cpu_match = scorer._component_score("cpu", searchable)
        gpu_score, gpu_match = scorer._component_score("gpu", searchable)
        cpu_score, cpu_match = self._fallback_cpu(searchable, cpu_score, cpu_match)
        gpu_score, gpu_match = self._fallback_gpu(searchable, gpu_score, gpu_match)

        gpu_core_match = re.search(r"\b(\d{1,2})gpu\b", searchable)

        ram_value = _find_spec(product, ("RAM",))
        ram_gb = _number(ram_value)
        ram_score = min(100.0, ram_gb / 64 * 100) if ram_gb else 0.0

        ssd_value = _find_spec(product, ("Ổ cứng SSD", "SSD"))
        ssd_gb = _number(ssd_value)
        if "tb" in _normalize(ssd_value):
            ssd_gb *= 1024
        storage_score = min(100.0, ssd_gb / 4096 * 100) if ssd_gb else 0.0

        panel = _find_spec(product, ("Tấm nền",))
        refresh = _number(_find_spec(product, ("Tần số quét",)))
        display_score = 25.0
        if "oled" in _normalize(panel):
            display_score += 45.0
        elif panel:
            display_score += 20.0
        if refresh >= 240:
            display_score += 30.0
        elif refresh >= 120:
            display_score += 20.0
        display_score = min(100.0, display_score)

        if use_case == "gaming":
            performance_score = (
                cpu_score * 0.30
                + gpu_score * 0.55
                + ram_score * 0.10
                + storage_score * 0.05
            )
        else:
            performance_score = (
                cpu_score * 0.45
                + gpu_score * 0.35
                + ram_score * 0.15
                + storage_score * 0.05
            )
        if mode == "max_performance":
            score = performance_score
        else:
            score = (
                performance_score * 0.68
                + display_score * 0.17
                + ram_score * 0.08
                + storage_score * 0.07
            )

        evidence = sum(
            (bool(cpu_match), bool(gpu_match), bool(ram_gb), bool(ssd_gb), bool(panel))
        )
        confidence = min(0.96, 0.46 + evidence * 0.10)
        reasons: list[str] = []
        if cpu_match:
            reasons.append(f"CPU {self._display_match(cpu_match)} thuộc nhóm đầu catalog")
        if gpu_match:
            reasons.append(f"GPU {self._display_match(gpu_match)} tạo lợi thế hiệu năng rõ")
        elif gpu_core_match:
            reasons.append(
                f"Cấu hình công bố {gpu_core_match.group(1)} lõi GPU, "
                "nhưng chưa quy đổi chéo sang GPU rời"
            )
        if ram_gb:
            reasons.append(f"RAM {ram_gb}GB")
        if ssd_gb:
            reasons.append(
                f"SSD {ssd_gb // 1024}TB" if ssd_gb >= 1024 else f"SSD {ssd_gb}GB"
            )
        if panel:
            display = panel
            if refresh:
                display += f", {refresh}Hz"
            reasons.append(f"Màn hình {display}")

        cautions: list[str] = []
        if not cpu_match:
            cautions.append("catalog chưa có tier CPU đủ rõ")
        if not gpu_match:
            cautions.append(
                "catalog chưa có tier GPU đồng nhất để so sánh chéo nền tảng"
                if gpu_core_match
                else "catalog chưa có tier GPU đủ rõ"
            )
        cautions.append(
            "đây là xếp hạng nội bộ từ cấu hình catalog, không thay thế benchmark thực tế"
        )
        return CatalogCapabilityRanking(
            product=product,
            mode=mode,
            score=round(score, 2),
            confidence=round(confidence, 2),
            reasons=tuple(reasons[:5]),
            cautions=tuple(cautions[:2]),
        )

    @staticmethod
    def _fallback_cpu(
        searchable: str,
        score: float,
        match: str,
    ) -> tuple[float, str]:
        if match:
            return score, match
        tiers = (
            ("core ultra 9", 92.0),
            ("ultra 9", 92.0),
            ("ryzen 9", 90.0),
            ("core i9", 89.0),
            ("m5 max", 98.0),
            ("m5 pro", 95.0),
            ("m5", 90.0),
            ("core ultra 7", 82.0),
            ("ryzen 7", 76.0),
            ("core i7", 75.0),
        )
        for token, tier_score in tiers:
            if token in searchable:
                return tier_score, token
        return 0.0, ""

    @staticmethod
    def _fallback_gpu(
        searchable: str,
        score: float,
        match: str,
    ) -> tuple[float, str]:
        if match:
            return score, match
        tiers = (
            ("rtx 5090", 100.0),
            ("rtx 5080", 94.0),
            ("rtx 5070 ti", 87.0),
            ("rtx 5070", 83.0),
            ("rtx 5060", 77.0),
            ("rtx 4050", 70.0),
        )
        for token, tier_score in tiers:
            if token in searchable:
                return tier_score, token
        return 0.0, ""

    @staticmethod
    def _display_match(value: str) -> str:
        return _display_component(value)
