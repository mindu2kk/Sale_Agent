"""Build a small, source-attributed real product catalog from FPT Shop JSON-LD.

The crawler:
- uses public product sitemaps declared in robots.txt;
- respects robots.txt for every product URL;
- rate-limits requests and identifies itself;
- stores structured product facts, not long editorial/marketing copy;
- never overwrites the active synthetic catalog automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://fptshop.com.vn"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
SITEMAPS = {
    "Mobile Phone": f"{BASE_URL}/products/sitemap-dien-thoai.xml",
    "Laptop": f"{BASE_URL}/products/sitemap-may-tinh-xach-tay.xml",
}
USER_AGENT = "DuAnTTCSCatalogBot/1.0 (internal educational catalog; respectful crawl)"
KEY_FACT_TERMS = (
    "chip",
    "ram",
    "bộ nhớ",
    "ổ cứng",
    "ssd",
    "màn hình",
    "camera",
    "pin",
    "sạc",
    "kháng nước",
    "tần số quét",
    "trọng lượng",
    "bảo hành",
    "vật liệu",
    "kim loại",
    "mil-std",
    "cổng kết nối",
    "nâng cấp",
)


def _load_urls(session: requests.Session, sitemap_url: str) -> list[str]:
    response = session.get(sitemap_url, timeout=30)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [
        node.text.strip()
        for node in root.findall(".//s:loc", namespace)
        if node.text
    ]


def _load_disallow_patterns(session: requests.Session) -> list[re.Pattern]:
    response = session.get(ROBOTS_URL, timeout=30)
    response.raise_for_status()
    patterns: list[re.Pattern] = []
    applies_to_all = False
    for raw_line in response.text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().casefold()
        value = value.strip()
        if key == "user-agent":
            applies_to_all = value == "*"
        elif key == "disallow" and applies_to_all and value:
            regex = re.escape(value).replace(r"\*", ".*")
            patterns.append(re.compile(rf"^{regex}"))
    return patterns


def _robots_allows(url: str, disallow_patterns: list[re.Pattern]) -> bool:
    target = urlparse(url)
    path_and_query = target.path + (f"?{target.query}" if target.query else "")
    return not any(pattern.search(path_and_query) for pattern in disallow_patterns)


def _product_json_ld(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("@type") == "Product":
                return candidate
    return None


def _brand_name(payload: dict) -> str:
    brand = payload.get("brand")
    if isinstance(brand, dict):
        return str(brand.get("name", "")).strip()
    return str(brand or "").strip()


def _offer(payload: dict) -> dict:
    offers = payload.get("offers")
    if isinstance(offers, list):
        return next((item for item in offers if isinstance(item, dict)), {})
    return offers if isinstance(offers, dict) else {}


def _structured_specs(payload: dict) -> dict[str, str]:
    specs: dict[str, str] = {}
    properties = payload.get("additionalProperty") or []
    for item in properties:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        value = str(item.get("value", "")).strip()
        if name and value:
            specs[name] = value
    return specs


def _heading_specs(html: str, category: str) -> dict[str, str]:
    """Recover compact specs embedded in the product H1.

    FPT Shop's JSON-LD often exposes only two highlight fields while the H1
    still contains the purchasable variant, e.g. /16GB/512GB/14" FHD/Win11.
    """
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    if not title:
        return {}

    specs: dict[str, str] = {}
    if category == "Laptop":
        memory_storage = re.search(
            r"/\s*(\d+\s*GB(?:\s*\([^)]*\))?)\s*/\s*(\d+\s*(?:GB|TB))(?=\s*/|\s*\(|\s*$)",
            title,
            flags=re.IGNORECASE,
        )
        if memory_storage:
            specs["RAM"] = re.sub(r"\s+", " ", memory_storage.group(1)).strip()
            specs["Ổ cứng SSD"] = re.sub(
                r"\s+", " ", memory_storage.group(2)
            ).strip()

        screen = re.search(
            r"/\s*(\d+(?:[.,]\d+)?)\s*[\"”]\s*([^/]+)",
            title,
            flags=re.IGNORECASE,
        )
        if screen:
            specs["Kích thước màn hình"] = (
                screen.group(1).replace(",", ".") + " inch"
            )
            resolution = screen.group(2).strip()
            if resolution:
                specs["Độ phân giải"] = resolution

        gpu = re.search(
            r"/\s*((?:NVIDIA\s+)?GeForce\s+(?:RTX|GTX)\s*\d+[^/]*)",
            title,
            flags=re.IGNORECASE,
        )
        if gpu:
            specs["Card đồ hoạ"] = re.sub(r"\s+", " ", gpu.group(1)).strip()

        os_match = re.search(
            r"/\s*((?:Windows|Win)\s*11[^/]*)",
            title,
            flags=re.IGNORECASE,
        )
        if os_match:
            specs["Hệ điều hành"] = re.sub(
                r"\s+", " ", os_match.group(1)
            ).strip()
    elif category == "Mobile Phone":
        memory_storage = re.search(
            r"(\d+\s*GB)\s*/\s*(\d+\s*(?:GB|TB))",
            title,
            flags=re.IGNORECASE,
        )
        if memory_storage:
            specs["RAM"] = re.sub(r"\s+", " ", memory_storage.group(1)).strip()
            specs["Bộ nhớ trong"] = re.sub(
                r"\s+", " ", memory_storage.group(2)
            ).strip()
        else:
            storage_values = re.findall(
                r"(?<!\d)(64|128|256|512)\s*GB\b",
                title,
                flags=re.IGNORECASE,
            )
            if storage_values:
                specs["Bộ nhớ trong"] = f"{storage_values[-1]} GB"
    return specs


def _extract_fact_sentences(html: str, limit: int = 10) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        lowered = sentence.casefold()
        if not any(term in lowered for term in KEY_FACT_TERMS):
            continue
        if len(sentence) < 25 or len(sentence) > 360:
            continue
        signature = re.sub(r"\W+", " ", lowered).strip()
        if signature in seen:
            continue
        selected.append(sentence)
        seen.add(signature)
        if len(selected) >= limit:
            break
    return selected


def _evidence_specs(facts: list[str]) -> dict[str, str]:
    """Promote only unambiguous sourced facts into comparable fields."""
    text = " ".join(facts)
    patterns = (
        ("Dung lượng Pin", r"\b(\d{3,5}\s*mAh)\b"),
        ("Dung lượng Pin", r"\b(\d+(?:[.,]\d+)?\s*Wh)\b"),
        ("Tần số quét", r"\b(\d{2,3}\s*Hz)\b"),
        ("Trọng lượng", r"trọng lượng(?: chỉ| khoảng)?\s+(\d+(?:[.,]\d+)?\s*kg)"),
        ("Bảo hành", r"bảo hành\s+(\d+\s*tháng)"),
        ("Tiêu chuẩn độ bền", r"\b(MIL[- ]STD[- ]?\d+[A-Z]?)\b"),
        (
            "Vật liệu",
            r"(?:vỏ|khung|chất liệu|vật liệu)\s+(nhôm|kim loại|magie|magnesium|aluminum)",
        ),
        (
            "Khả năng nâng cấp",
            r"((?:hỗ trợ|có thể)\s+nâng cấp[^.]{0,90}|(?:2|hai)\s+khe\s+RAM)",
        ),
    )
    specs: dict[str, str] = {}
    for label, pattern in patterns:
        matches = {
            re.sub(r"\s+", " ", match).strip()
            for match in re.findall(pattern, text, flags=re.IGNORECASE)
        }
        if len(matches) == 1:
            specs[label] = next(iter(matches))
    ports = sorted(
        set(
            re.findall(
                r"\b(?:Thunderbolt\s*\d*|USB(?:-C| Type-C| 3\.\d)?|HDMI(?:\s*\d\.\d)?|RJ-45)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    if ports:
        specs["Cổng kết nối"] = ", ".join(ports[:8])
    return specs


def _format_vnd(price: int) -> str:
    return f"{price:,}".replace(",", ".") + " VNĐ"


def _compat_context(
    name: str,
    category: str,
    brand: str,
    sku: str,
    price: int,
    specs: dict[str, str],
    facts: list[str],
) -> str:
    spec_text = ", ".join(f"{key}: {value}" for key, value in specs.items())
    fact_text = " ".join(facts[:4])
    return (
        f"Sản phẩm {category} {name}, thương hiệu {brand}, mã sản phẩm {sku}. "
        f"Giá bán ghi nhận là {_format_vnd(price)}. "
        f"Thông số có nguồn gồm: {spec_text or 'chưa có thông số cấu trúc'}. "
        f"Điểm nổi bật từ trang sản phẩm: {fact_text or 'chưa có dữ kiện bổ sung'}."
    )


def crawl(
    output_path: Path,
    *,
    limit: int,
    delay_seconds: float,
    per_category_limit: int | None = None,
) -> list[dict]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.6",
        }
    )
    disallow_patterns = _load_disallow_patterns(session)

    rows: list[dict] = []
    fetched_at = datetime.now(UTC).isoformat()

    for category, sitemap_url in SITEMAPS.items():
        urls = _load_urls(session, sitemap_url)
        category_count = 0
        for url in urls:
            if len(rows) >= limit:
                break
            if per_category_limit is not None and category_count >= per_category_limit:
                break
            if not _robots_allows(url, disallow_patterns):
                continue

            try:
                response = session.get(url, timeout=35)
                response.raise_for_status()
                payload = _product_json_ld(response.text)
                if payload is None:
                    continue

                offer = _offer(payload)
                price = int(float(offer.get("price") or 0))
                sku = str(payload.get("sku") or payload.get("mpn") or "").strip()
                name = str(payload.get("name") or "").strip()
                brand = _brand_name(payload)
                if not sku:
                    sku = hashlib.sha256(url.encode()).hexdigest()[:12].upper()
                if not name or not brand or price <= 0:
                    continue

                specs = _structured_specs(payload)
                for key, value in _heading_specs(response.text, category).items():
                    specs.setdefault(key, value)
                facts = _extract_fact_sentences(response.text)
                for key, value in _evidence_specs(facts).items():
                    specs.setdefault(key, value)
                image = payload.get("image")
                if isinstance(image, list):
                    image = image[0] if image else ""

                rows.append(
                    {
                        "Product Code": sku,
                        "Product": category,
                        "Brand": brand,
                        "Name": name,
                        "Price": _format_vnd(price),
                        "Price Value": price,
                        "Currency": str(offer.get("priceCurrency") or "VND"),
                        "Availability": str(offer.get("availability") or ""),
                        "Price Valid Until": str(offer.get("priceValidUntil") or ""),
                        "Image URL": str(image or ""),
                        "Source URL": url,
                        "Fetched At": fetched_at,
                        "Structured Specs JSON": json.dumps(specs, ensure_ascii=False),
                        "Evidence Facts JSON": json.dumps(facts, ensure_ascii=False),
                        "Spec Provenance JSON": json.dumps(
                            {
                                key: {
                                    "source_url": url,
                                    "fetched_at": fetched_at,
                                    "confidence": "high",
                                }
                                for key in specs
                            },
                            ensure_ascii=False,
                        ),
                        "LLM_Context": _compat_context(
                            name, category, brand, sku, price, specs, facts
                        ),
                    }
                )
                category_count += 1
                print(f"[{len(rows):03d}/{limit}] {category}: {name}")
            except (requests.RequestException, ValueError) as exc:
                print(f"[skip] {url}: {exc}")
            time.sleep(delay_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/product_catalog_real.csv"),
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--per-category-limit", type=int)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    rows = crawl(
        args.output,
        limit=args.limit,
        delay_seconds=max(0.25, args.delay),
        per_category_limit=args.per_category_limit,
    )
    print(f"Wrote {len(rows)} products to {args.output}")


if __name__ == "__main__":
    main()
