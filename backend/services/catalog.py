"""Product catalog access and deterministic image resolution."""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CATALOG_PATH = PROJECT_ROOT / "data" / "product_catalog_clean.csv"
REAL_CATALOG_PATH = PROJECT_ROOT / "data" / "product_catalog_real.csv"
CATALOG_PATH = Path(
    os.getenv(
        "PRODUCT_CATALOG_PATH",
        str(REAL_CATALOG_PATH if REAL_CATALOG_PATH.exists() else SYNTHETIC_CATALOG_PATH),
    )
)
IMAGE_DIR = PROJECT_ROOT / "data" / "product_images"
IMAGE_MAP_PATH = PROJECT_ROOT / "data" / "product_images.csv"
GENERIC_LAPTOP_IMAGE = IMAGE_DIR / "_generic_laptop.png"
GENERIC_PHONE_IMAGE = IMAGE_DIR / "_generic_phone.png"


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD", value.casefold().replace("đ", "d")
    )
    ascii_value = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


def _price_value(value: str) -> int:
    digits = re.sub(r"\D", "", value)
    return int(digits) if digits else 0


def _parse_money_amount(raw: str, unit: str | None = None) -> int:
    cleaned = raw.strip().replace(" ", "")
    digits_only = re.sub(r"\D", "", cleaned)
    normalized_unit = _normalize(unit or "")

    if normalized_unit in {"trieu", "tr", "million"}:
        compact = cleaned.replace(",", ".")
        if compact.count(".") > 1:
            return int(digits_only or 0)
        return int(float(compact) * 1_000_000)

    if normalized_unit in {"nghin", "ngan", "k"}:
        compact = cleaned.replace(",", ".")
        if compact.count(".") > 1:
            return int(digits_only or 0)
        return int(float(compact) * 1_000)

    return int(digits_only or 0)


@dataclass(frozen=True)
class PriceIntent:
    mode: str
    target: int | None = None
    minimum: int | None = None
    maximum: int | None = None


@dataclass(frozen=True)
class QueryAnalysis:
    normalized_query: str
    tokens: tuple[str, ...]
    price_intent: PriceIntent | None = None


@dataclass(frozen=True)
class QueryConstraints:
    category: str | None = None
    price_intent: PriceIntent | None = None
    discrete_gpu: bool = False
    brands: tuple[str, ...] = ()
    cpu_filters: tuple[str, ...] = ()
    gpu_filters: tuple[str, ...] = ()
    comparison: bool = False
    goal: str | None = None
    use_case: str | None = None


def _detect_cpu_filters(query: str) -> tuple[str, ...]:
    normalized = _normalize(query)
    patterns = (
        ("intel_core_i9", (r"\bcore\s*i9\b", r"\bi9\b")),
        ("intel_core_i7", (r"\bcore\s*i7\b", r"\bi7\b")),
        ("intel_core_i5", (r"\bcore\s*i5\b", r"\bi5\b")),
        ("intel_core_i3", (r"\bcore\s*i3\b", r"\bi3\b")),
        ("intel_core_ultra", (r"\bcore\s+ultra\b", r"\bultra\b")),
        ("intel_core_7", (r"\bcore\s+7\b",)),
        ("intel_core_5", (r"\bcore\s+5\b",)),
        ("amd_ryzen_7", (r"\bryzen\s*7\b",)),
        ("amd_ryzen_5", (r"\bryzen\s*5\b",)),
    )
    detected: list[str] = []
    for label, regexes in patterns:
        if any(re.search(regex, normalized) for regex in regexes):
            detected.append(label)
    return tuple(detected)


def _detect_gpu_filters(query: str) -> tuple[str, ...]:
    normalized = _normalize(query)
    patterns = (
        ("nvidia_rtx", (r"\brtx\b", r"\bgeforce\s+rtx\b")),
        ("nvidia_gtx", (r"\bgtx\b", r"\bgeforce\s+gtx\b")),
        ("radeon_rx", (r"\bradeon\s+rx\b",)),
        ("intel_graphics", (r"\bintel\s+graphics\b", r"\bintel\s+uhd\b", r"\bintel\s+iris\b")),
        ("intel_arc", (r"\bintel\s+arc\b",)),
    )
    detected: list[str] = []
    for label, regexes in patterns:
        if any(re.search(regex, normalized) for regex in regexes):
            detected.append(label)
    return tuple(detected)


def _parse_price_intent(query: str) -> PriceIntent | None:
    # Product codes in this catalog are 8 digits beginning with 0. They must
    # never be interpreted as a monetary amount when resolving conversation
    # history (for example SKU 00928595 is not 928.595 VNĐ).
    without_dates = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", query)
    without_skus = re.sub(r"(?<!\d)0\d{7}(?!\d)", " ", without_dates)
    decomposed = unicodedata.normalize(
        "NFD", without_skus.casefold().replace("đ", "d")
    )
    ascii_value = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    lowered = re.sub(r"[^a-z0-9.,-]+", " ", ascii_value).strip()
    number = r"\d+(?:[.,]\d+)*"
    unit = r"(?:trieu|tr|million|nghin|ngan|k|vnd|d)\b"

    range_match = re.search(
        rf"(?:tu|khoang|tam|gia)?\s*({number})\s*({unit})?\s*(?:den|toi|-|to)\s*({number})\s*({unit})?",
        lowered,
    )
    if range_match:
        shared_unit = range_match.group(2) or range_match.group(4)
        minimum = _parse_money_amount(
            range_match.group(1), range_match.group(2) or shared_unit
        )
        maximum = _parse_money_amount(
            range_match.group(3),
            range_match.group(4) or shared_unit,
        )
        price_context = lowered[
            max(0, range_match.start() - 18) : range_match.end() + 8
        ]
        has_price_signal = bool(shared_unit) or bool(
            re.search(r"(gia|ngan sach|khoang|tam)", price_context)
        )
        if minimum and maximum and has_price_signal:
            return PriceIntent(
                mode="range",
                minimum=min(minimum, maximum),
                maximum=max(minimum, maximum),
            )

    # Vietnamese users commonly omit the connector: "tầm giá 23 24 triệu".
    shared_unit_range = re.search(
        rf"(?:tu|khoang|tam|gia)\s*({number})\s+({number})\s*({unit})",
        lowered,
    )
    if shared_unit_range:
        minimum = _parse_money_amount(
            shared_unit_range.group(1), shared_unit_range.group(3)
        )
        maximum = _parse_money_amount(
            shared_unit_range.group(2), shared_unit_range.group(3)
        )
        if minimum and maximum:
            return PriceIntent(
                mode="range",
                minimum=min(minimum, maximum),
                maximum=max(minimum, maximum),
            )

    for amount_match in re.finditer(rf"({number})\s*({unit})?", lowered):
        raw_amount = amount_match.group(1)
        raw_unit = amount_match.group(2)
        amount = _parse_money_amount(raw_amount, raw_unit)
        if amount <= 0:
            continue

        if raw_unit is None:
            # Bare small numbers are usually model names, RAM, screen sizes,
            # or malformed ranges. Only accept a VND-sized number.
            if amount < 100_000:
                continue
            context_window = lowered[
                max(0, amount_match.start() - 18) : amount_match.end() + 18
            ]
            if not re.search(r"(gia|ngan sach|duoi|tren|tam|khoang)", context_window):
                continue

        prefix = lowered[: amount_match.start()]
        nearby_prefix = prefix[-28:]
        suffix = lowered[amount_match.end() :]
        if re.search(r"(duoi|toi da|khong qua|under|less than)", prefix):
            return PriceIntent(mode="max", maximum=amount)
        if re.search(r"(tam|khoang|gan)\s*$", nearby_prefix):
            return PriceIntent(mode="target", target=amount)
        if re.search(r"(tren|it nhat|from|over|above)\s*$", nearby_prefix) or re.search(
            r"(tro len|up)", suffix
        ):
            return PriceIntent(mode="min", minimum=amount)
        return PriceIntent(mode="target", target=amount)
    return None


def _extract_specs(context: str) -> list[str]:
    marker = "bao gồm:"
    if marker not in context:
        return []
    raw_specs = context.split(marker, 1)[1].rstrip(".")
    return [
        spec.strip()
        for spec in raw_specs.split(",")
        if spec.strip() and not spec.strip().endswith(" nan")
    ]


def _extract_row_specs(row: dict[str, str]) -> list[str]:
    specs: list[str] = []
    structured = row.get("Structured Specs JSON", "")
    if structured:
        try:
            payload = json.loads(structured)
            if isinstance(payload, dict):
                specs.extend(
                    f"{key} {value}".strip()
                    for key, value in payload.items()
                    if key and value
                )
        except json.JSONDecodeError:
            pass

    evidence_text = row.get("Evidence Facts JSON", "")
    try:
        evidence = json.loads(evidence_text) if evidence_text else []
    except json.JSONDecodeError:
        evidence = []
    combined = " ".join(str(item) for item in evidence)
    patterns = (
        ("Chip", r"(?:chip|vi xử lý|bộ xử lý)\s+([A-Za-z0-9][A-Za-z0-9 +\-]{2,45})"),
        ("RAM", r"\bRAM\s+(\d+\s*(?:GB|TB))"),
        ("Bộ nhớ trong", r"bộ nhớ trong\s+(\d+\s*(?:GB|TB))"),
        (
            "Ổ cứng SSD",
            r"(?:ổ\s+)?SSD(?:\s+PCIe(?:\s+Gen\d+)?)?(?:\s+dung lượng)?\s+(\d+\s*(?:GB|TB))",
        ),
        ("Pin", r"(?:pin|viên pin)[^.\d]{0,45}(\d{3,5}\s*mAh)"),
        ("Pin", r"(?:pin|viên pin)[^.\d]{0,45}(\d+(?:[.,]\d+)?\s*Wh)"),
        ("Tần số quét", r"tần số quét(?: tối đa)?\s+(\d+\s*Hz)"),
        ("Trọng lượng", r"trọng lượng(?: chỉ| khoảng)?\s+(\d+(?:[.,]\d+)?\s*kg)"),
        ("Kháng nước", r"(IP\d{2})"),
        ("Bảo hành", r"bảo hành\s+(\d+\s*tháng)"),
    )
    for label, pattern in patterns:
        matches = {
            re.sub(r"\s+", " ", match).strip()
            for match in re.findall(pattern, combined, flags=re.IGNORECASE)
        }
        # Conflicting facts on the source page are not safe to use in advice.
        if len(matches) == 1:
            specs.append(f"{label} {next(iter(matches))}")

    if not specs:
        specs.extend(_extract_specs(row.get("LLM_Context", "")))
    return list(dict.fromkeys(specs))


@dataclass(frozen=True)
class CatalogProduct:
    code: str
    category: str
    brand: str
    price: str
    context: str
    specs: tuple[str, ...]
    title: str = ""
    source_url: str = ""
    fetched_at: str = ""
    price_valid_until: str = ""
    external_image_url: str = ""
    spec_provenance_json: str = ""

    @property
    def name(self) -> str:
        if self.title:
            return self.title
        category = "Laptop" if self.category == "Laptop" else "Điện thoại"
        return f"{self.brand} {category} · {self.code}"

    def to_dict(self, image_source: str = "generated") -> dict:
        return {
            "id": self.code,
            "code": self.code,
            "name": self.name,
            "category": self.category,
            "brand": self.brand,
            "price": self.price,
            "price_value": _price_value(self.price),
            "description": self.context,
            "specs": list(self.specs),
            "image_url": f"/api/products/{self.code}/image",
            "image_source": image_source,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
            "price_valid_until": self.price_valid_until,
            "spec_provenance": (
                json.loads(self.spec_provenance_json)
                if self.spec_provenance_json
                else {}
            ),
        }


class CatalogService:
    def __init__(self, catalog_path: Path = CATALOG_PATH) -> None:
        self.catalog_path = catalog_path
        self.products = self._load_products()
        self.by_code = {product.code.casefold(): product for product in self.products}
        self.image_map = self._load_image_map()

    def _load_products(self) -> list[CatalogProduct]:
        with self.catalog_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return [
                CatalogProduct(
                    code=row["Product Code"].strip(),
                    category=row["Product"].strip(),
                    brand=row["Brand"].strip(),
                    price=row["Price"].strip(),
                    context=row["LLM_Context"].strip(),
                    specs=tuple(_extract_row_specs(row)),
                    title=row.get("Name", "").strip(),
                    source_url=row.get("Source URL", "").strip(),
                    fetched_at=row.get("Fetched At", "").strip(),
                    price_valid_until=row.get("Price Valid Until", "").strip(),
                    external_image_url=row.get("Image URL", "").strip(),
                    spec_provenance_json=row.get(
                        "Spec Provenance JSON", ""
                    ).strip(),
                )
                for row in reader
            ]

    def _load_image_map(self) -> dict[str, str]:
        if not IMAGE_MAP_PATH.exists():
            return {}
        with IMAGE_MAP_PATH.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return {
                row["Product Code"].strip().casefold(): row["Image URL"].strip()
                for row in reader
                if row.get("Product Code") and row.get("Image URL")
            }

    def get(self, code: str) -> CatalogProduct | None:
        return self.by_code.get(code.casefold())

    def resolve_product(
        self,
        query: str,
        context_product_codes: Iterable[str] = (),
    ) -> CatalogProduct | None:
        """Resolve an explicit SKU first, then the latest valid conversation SKU."""
        for candidate in re.findall(r"\b[A-Z0-9]{8}\b", query.upper()):
            product = self.get(candidate)
            if product is not None:
                return product
        for code in context_product_codes:
            product = self.get(code)
            if product is not None:
                return product
        return None

    def resolve_products(
        self,
        query: str,
        context_product_codes: Iterable[str] = (),
        *,
        limit: int = 8,
    ) -> list[CatalogProduct]:
        """Resolve every explicitly named SKU/model in query order.

        Product names frequently contain model identifiers that look similar
        to SKUs (for example EP1179TU or DC15255). Matching those identifiers
        lets comparison requests name two or more products naturally without
        forcing the customer to copy internal SKUs.
        """
        normalized_query = _normalize(query)
        query_tokens = set(normalized_query.split())
        matches: list[tuple[int, int, CatalogProduct]] = []
        seen: set[str] = set()

        for match in re.finditer(r"\b[A-Z0-9]{8}\b", query.upper()):
            product = self.get(match.group(0))
            if product is not None and product.code not in seen:
                matches.append((match.start(), 10_000, product))
                seen.add(product.code)

        ignored_identifiers = {
            "windows",
            "win11",
            "office",
            "laptop",
            "gaming",
            "mobile",
        }
        for product in self.products:
            if product.code in seen:
                continue
            normalized_name = _normalize(product.name)
            if normalized_name and normalized_name in normalized_query:
                position = normalized_query.find(normalized_name)
                matches.append(
                    (position, 5_000 + len(normalized_name), product)
                )
                continue
            identifiers = {
                token
                for token in normalized_name.split()
                if len(token) >= 5
                and sum(char.isalpha() for char in token) >= 2
                and any(char.isdigit() for char in token)
                and token not in ignored_identifiers
            }
            identifier_hits = identifiers & query_tokens
            if not identifier_hits:
                continue
            positions = [
                normalized_query.find(identifier)
                for identifier in identifier_hits
                if normalized_query.find(identifier) >= 0
            ]
            if not positions:
                continue
            # Prefer products matching several distinctive model tokens.
            score = len(identifier_hits) * 100 + sum(len(token) for token in identifier_hits)
            matches.append((min(positions), score, product))

        matches.sort(key=lambda item: (item[0], -item[1], _price_value(item[2].price)))
        resolved: list[CatalogProduct] = []
        used_positions: set[tuple[int, str]] = set()
        for position, _, product in matches:
            signature = (position, product.brand)
            if signature in used_positions:
                continue
            resolved.append(product)
            used_positions.add(signature)
            if len(resolved) >= limit:
                break

        if resolved:
            return resolved
        for code in context_product_codes:
            product = self.get(code)
            if product is not None and product not in resolved:
                resolved.append(product)
            if len(resolved) >= limit:
                break
        return resolved

    def analyze_query(self, query: str) -> QueryAnalysis:
        normalized_query = _normalize(query)
        return QueryAnalysis(
            normalized_query=normalized_query,
            tokens=tuple(normalized_query.split()),
            price_intent=_parse_price_intent(query),
        )

    def resolve_constraints(
        self,
        query: str,
        history: Iterable[str] = (),
        context_product_codes: Iterable[str] = (),
        session_state: dict | None = None,
    ) -> QueryConstraints:
        session_state = session_state or {}
        analysis = self.analyze_query(query)
        explicit_category = self._detect_category(query)
        category = explicit_category
        price_intent = analysis.price_intent
        discrete_gpu = self._asks_for_discrete_gpu(query)
        brands = self._detect_brands(query)
        cpu_filters = _detect_cpu_filters(query)
        gpu_filters = _detect_gpu_filters(query)
        comparison = len(brands) >= 2 and self._is_comparison_query(query)
        goal = self._detect_goal(query)
        use_case = self._detect_use_case(query)
        continuation = self._should_inherit_price(query) or goal is not None
        inherit_price = price_intent is None and continuation

        state_category = session_state.get("category")
        if category is None and state_category:
            category = str(state_category)
        elif (
            explicit_category is not None
            and state_category
            and explicit_category != state_category
        ):
            # An explicit category switch starts a new shopping topic.
            session_state = {}

        if price_intent is None and inherit_price:
            target = session_state.get("budget_target")
            minimum = session_state.get("budget_minimum")
            maximum = session_state.get("budget_maximum")
            if target:
                price_intent = PriceIntent(mode="target", target=int(target))
            elif minimum is not None and maximum is not None:
                price_intent = PriceIntent(
                    mode="range",
                    minimum=int(minimum),
                    maximum=int(maximum),
                )
            elif maximum is not None:
                price_intent = PriceIntent(mode="max", maximum=int(maximum))
            elif minimum is not None:
                price_intent = PriceIntent(mode="min", minimum=int(minimum))

        if goal is None and continuation:
            goal = session_state.get("goal")
        if use_case is None and continuation:
            use_case = session_state.get("use_case")

        context_product: CatalogProduct | None = None
        if category is None:
            for code in context_product_codes:
                product = self.get(code)
                if product is not None:
                    context_product = product
                    category = product.category
                    break
        else:
            for code in context_product_codes:
                product = self.get(code)
                if product is not None:
                    context_product = product
                    break

        if (
            price_intent is None
            and context_product is not None
            and self._asks_for_same_price(query)
        ):
            price_intent = PriceIntent(
                mode="target",
                target=_price_value(context_product.price),
            )

        for previous_query in reversed(list(history)):
            previous_category = self._detect_category(previous_query)
            previous_price = self.analyze_query(previous_query).price_intent

            if category is None and previous_category is not None:
                category = previous_category
            if (
                price_intent is None
                and inherit_price
                and previous_price is not None
                and (
                    category is None
                    or previous_category is None
                    or previous_category == category
                )
            ):
                price_intent = previous_price
            if category is not None and price_intent is not None:
                break

        return QueryConstraints(
            category=category,
            price_intent=price_intent,
            discrete_gpu=discrete_gpu,
            brands=brands,
            cpu_filters=cpu_filters,
            gpu_filters=gpu_filters,
            comparison=comparison,
            goal=goal,
            use_case=use_case,
        )

    @staticmethod
    def _detect_category(query: str) -> str | None:
        normalized = _normalize(query)
        if any(
            term in normalized
            for term in ("laptop", "may tinh xach tay", "notebook", "ultrabook", "macbook")
        ):
            return "Laptop"
        if any(
            term in normalized
            for term in ("dien thoai", "smartphone", "phone", "iphone")
        ):
            return "Mobile Phone"
        return None

    @staticmethod
    def _asks_for_discrete_gpu(query: str) -> bool:
        normalized = _normalize(query)
        if any(
            term in normalized
            for term in (
                "khong can card roi",
                "khong can gpu roi",
                "card tich hop",
                "khong choi game",
            )
        ):
            return False
        return any(
            term in normalized
            for term in (
                "card roi",
                "card do hoa roi",
                "gpu roi",
                "do hoa roi",
                "rtx",
                "gtx",
                "choi game",
                "gaming",
            )
        )

    @staticmethod
    def _should_inherit_price(query: str) -> bool:
        normalized = _normalize(query)
        return any(
            term in normalized
            for term in (
                "co may nao",
                "co mau nao",
                "con may nao",
                "con mau nao",
                "mau khac",
                "may khac",
                "tam do",
                "muc do",
                "gia do",
                "gia nay",
                "nhu vay",
                "card roi",
                "gpu roi",
                "choi game",
                "gaming",
                "hieu nang tren gia",
                "hieu nang gia",
                "p p",
                "performance per price",
                "dang tien nhat",
                "ngon nhat",
                "khong choi game",
                "van phong",
                "office",
                "hoc tap",
                "lap trinh",
                "coding",
                "do hoa",
                "render",
            )
        )

    @staticmethod
    def _detect_goal(query: str) -> str | None:
        normalized = _normalize(query)
        if any(
            term in normalized
            for term in (
                "hieu nang tren gia",
                "hieu nang gia",
                "ti le hieu nang",
                "performance per price",
                "p p",
                "dang tien nhat",
                "ngon nhat trong tam",
                "gia tri cao nhat",
            )
        ):
            return "performance_per_price"
        return None

    @staticmethod
    def _detect_use_case(query: str) -> str | None:
        normalized = _normalize(query)
        if any(
            term in normalized
            for term in ("khong choi game", "chi van phong", "chu yeu van phong")
        ):
            return "office"
        if any(term in normalized for term in ("choi game", "gaming", "game")):
            return "gaming"
        if any(
            term in normalized
            for term in ("lap trinh", "code", "coding", "developer")
        ):
            return "programming"
        if any(
            term in normalized
            for term in ("do hoa", "render", "edit video", "photoshop", "premiere")
        ):
            return "creative"
        if any(
            term in normalized
            for term in ("van phong", "office", "hoc tap", "ke toan")
        ):
            return "office"
        return None

    def _detect_brands(self, query: str) -> tuple[str, ...]:
        normalized = _normalize(query)
        aliases = {
            "macbook": "Apple",
            "mac book": "Apple",
            "iphone": "Apple",
        }
        positions: dict[str, int] = {}
        for alias, brand in aliases.items():
            position = normalized.find(alias)
            if position >= 0:
                positions[brand] = min(position, positions.get(brand, position))
        for brand in dict.fromkeys(product.brand for product in self.products):
            position = normalized.find(_normalize(brand))
            if position >= 0:
                positions[brand] = min(position, positions.get(brand, position))
        return tuple(
            brand for brand, _ in sorted(positions.items(), key=lambda item: item[1])
        )

    @staticmethod
    def _is_comparison_query(query: str) -> bool:
        normalized = _normalize(query)
        return any(
            term in normalized
            for term in ("hay", "so sanh", "chon", "nen mua", "tot hon")
        )

    @staticmethod
    def _asks_for_same_price(query: str) -> bool:
        normalized = _normalize(query)
        return any(
            term in normalized
            for term in (
                "cung tam gia",
                "cung muc gia",
                "ngang gia",
                "gia tuong duong",
                "tam gia do",
            )
        )

    @staticmethod
    def _has_discrete_gpu(product: CatalogProduct) -> bool:
        searchable = _normalize(
            " ".join((product.name, product.context, *product.specs))
        )
        return any(
            marker in searchable
            for marker in (
                "nvidia geforce",
                "geforce rtx",
                "geforce gtx",
                "radeon rx",
                "intel arc a",
            )
        )

    @staticmethod
    def _matches_cpu_filters(
        product: CatalogProduct,
        cpu_filters: Iterable[str],
    ) -> bool:
        requested = tuple(cpu_filters)
        if not requested:
            return True
        from backend.agent.product_facts import normalize_product

        facts = normalize_product(product)
        requested_tiers = {
            "intel_core_i9": "i9",
            "intel_core_i7": "i7",
            "intel_core_i5": "i5",
            "intel_core_i3": "i3",
            "intel_core_ultra": "Core Ultra",
            "intel_core_7": "Core 7",
            "intel_core_5": "Core 5",
            "amd_ryzen_7": "Ryzen 7",
            "amd_ryzen_5": "Ryzen 5",
        }
        return any(facts.cpu_tier == requested_tiers.get(label) for label in requested)

    @staticmethod
    def _matches_gpu_filters(
        product: CatalogProduct,
        gpu_filters: Iterable[str],
    ) -> bool:
        requested = tuple(gpu_filters)
        if not requested:
            return True
        from backend.agent.product_facts import normalize_product

        facts = normalize_product(product)
        if any(label in {"nvidia_rtx", "nvidia_gtx", "radeon_rx"} for label in requested):
            return facts.gpu_type == "dedicated"
        if "intel_graphics" in requested:
            return facts.gpu_type == "integrated" and bool(
                facts.gpu_raw and "intel" in _normalize(facts.gpu_raw)
            )
        if "intel_arc" in requested:
            return facts.gpu_type == "dedicated" and bool(
                facts.gpu_raw and "intel arc" in _normalize(facts.gpu_raw)
            )
        return False

    def image_source(self, product: CatalogProduct) -> str:
        if self.local_image(product.code):
            return "local"
        if product.external_image_url or product.code.casefold() in self.image_map:
            return "mapped"
        if self.generic_image(product):
            return "aura"
        return "generated"

    def local_image(self, code: str) -> Path | None:
        for extension in (".webp", ".png", ".jpg", ".jpeg"):
            path = IMAGE_DIR / f"{code}{extension}"
            if path.exists():
                return path
        return None

    def mapped_image(self, code: str) -> str | None:
        product = self.get(code)
        if product and product.external_image_url:
            return product.external_image_url
        return self.image_map.get(code.casefold())

    def generic_image(self, product: CatalogProduct) -> Path | None:
        fallback = GENERIC_LAPTOP_IMAGE if product.category == "Laptop" else GENERIC_PHONE_IMAGE
        return fallback if fallback.exists() else None

    def search(
        self,
        query: str = "",
        *,
        category: str | None = None,
        brand: str | None = None,
        limit: int = 12,
        offset: int = 0,
        price_intent: PriceIntent | None = None,
        discrete_gpu: bool = False,
        cpu_filters: Iterable[str] = (),
        gpu_filters: Iterable[str] = (),
        exclude_codes: Iterable[str] = (),
    ) -> list[CatalogProduct]:
        analysis = self.analyze_query(query)
        effective_price_intent = price_intent or analysis.price_intent
        normalized_query = analysis.normalized_query
        tokens = list(analysis.tokens)
        exact_code = self.get(query.strip())
        if exact_code:
            return [exact_code]
        for candidate in re.findall(r"\b[A-Z0-9]{8}\b", query.upper()):
            exact_code = self.get(candidate)
            if exact_code:
                return [exact_code]

        normalized_category = _normalize(category or "")
        normalized_brand = _normalize(brand or "")
        excluded_codes = {code.casefold() for code in exclude_codes}
        candidates: list[tuple[int, int, int, CatalogProduct]] = []

        for index, product in enumerate(self.products):
            if product.code.casefold() in excluded_codes:
                continue
            product_category = _normalize(product.category)
            product_brand = _normalize(product.brand)
            if normalized_category and normalized_category not in product_category:
                continue
            if normalized_brand and normalized_brand != product_brand:
                continue
            if discrete_gpu and not self._has_discrete_gpu(product):
                continue
            if not self._matches_cpu_filters(product, cpu_filters):
                continue
            if not self._matches_gpu_filters(product, gpu_filters):
                continue

            searchable = _normalize(
                f"{product.code} {product.category} {product.brand} {product.context}"
            )
            score = sum(1 for token in tokens if token in searchable)
            if product.code.casefold() in query.casefold():
                score += 20
            if product_brand and product_brand in normalized_query:
                score += 5
            if product_category and product_category in normalized_query:
                score += 3
            if tokens and effective_price_intent is None and not discrete_gpu and score == 0:
                continue
            candidates.append((score, _price_value(product.price), index, product))

        if not candidates:
            return []

        if effective_price_intent is not None:
            def price_key(item: tuple[int, int, int, CatalogProduct]) -> tuple[int, int, int, int]:
                score, price, index, _ = item
                if effective_price_intent.mode == "max":
                    maximum = effective_price_intent.maximum or 0
                    return (
                        0 if price <= maximum else 1,
                        abs(maximum - price),
                        -score,
                        index,
                    )
                if effective_price_intent.mode == "min":
                    minimum = effective_price_intent.minimum or 0
                    return (
                        0 if price >= minimum else 1,
                        abs(price - minimum),
                        -score,
                        index,
                    )
                if effective_price_intent.mode == "range":
                    minimum = effective_price_intent.minimum or 0
                    maximum = effective_price_intent.maximum or minimum
                    midpoint = (minimum + maximum) // 2
                    inside = minimum <= price <= maximum
                    distance = 0 if inside else min(abs(price - minimum), abs(price - maximum))
                    return (0 if inside else 1, distance or abs(price - midpoint), -score, index)

                target = effective_price_intent.target or 0
                soft_window = max(1_500_000, int(target * 0.12))
                close = abs(price - target) <= soft_window
                return (0 if close else 1, abs(price - target), -score, index)

            if effective_price_intent.mode == "range":
                minimum = effective_price_intent.minimum or 0
                maximum = effective_price_intent.maximum or minimum
                inside = [
                    item for item in candidates
                    if minimum <= item[1] <= maximum
                ]
                if inside:
                    candidates = inside
            elif effective_price_intent.mode == "max":
                maximum = effective_price_intent.maximum or 0
                inside = [item for item in candidates if item[1] <= maximum]
                if inside:
                    candidates = inside
            elif effective_price_intent.mode == "min":
                minimum = effective_price_intent.minimum or 0
                inside = [item for item in candidates if item[1] >= minimum]
                if inside:
                    candidates = inside
            elif effective_price_intent.mode == "target":
                target = effective_price_intent.target or 0
                soft_window = max(1_500_000, int(target * 0.12))
                nearby = [
                    item for item in candidates
                    if abs(item[1] - target) <= soft_window
                ]
                if nearby:
                    candidates = nearby
            candidates.sort(key=price_key)
            return [item[3] for item in candidates[offset : offset + limit]]

        candidates.sort(key=lambda item: (-item[0], item[2]))
        if not normalized_query and not normalized_category and not normalized_brand:
            # The storefront's default catalog should feel like a real mixed
            # collection instead of showing an entire sitemap category first.
            laptops = [
                item for item in candidates if item[3].category == "Laptop"
            ]
            phones = [
                item for item in candidates if item[3].category == "Mobile Phone"
            ]
            mixed: list[tuple[int, int, int, CatalogProduct]] = []
            for index in range(max(len(laptops), len(phones))):
                if index < len(laptops):
                    mixed.append(laptops[index])
                if index < len(phones):
                    mixed.append(phones[index])
            candidates = mixed
        return [item[3] for item in candidates[offset : offset + limit]]

    def featured(self, limit: int = 6) -> list[CatalogProduct]:
        targets = [
            ("Dell", "Laptop"),
            ("Asus", "Laptop"),
            ("HP", "Laptop"),
            ("Lenovo", "Laptop"),
            ("Samsung", "Mobile Phone"),
            ("Apple", "Mobile Phone"),
        ]
        selected: list[CatalogProduct] = []
        for brand, category in targets:
            matches = self.search(category=category, brand=brand, limit=1)
            if matches:
                selected.append(matches[0])
            if len(selected) >= limit:
                break
        return selected

    def serialize_many(self, products: Iterable[CatalogProduct]) -> list[dict]:
        return [
            product.to_dict(image_source=self.image_source(product))
            for product in products
        ]


@lru_cache(maxsize=2)
def _get_catalog_version(path: str, modified_ns: int) -> CatalogService:
    del modified_ns
    return CatalogService(Path(path))


def get_catalog() -> CatalogService:
    """Return a cached catalog that refreshes when the CSV is replaced."""
    modified_ns = CATALOG_PATH.stat().st_mtime_ns
    return _get_catalog_version(str(CATALOG_PATH), modified_ns)
