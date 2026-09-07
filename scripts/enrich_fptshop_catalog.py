"""Refresh and enrich an existing FPT Shop catalog in place.

This script keeps the catalog's SKU set stable and revisits each existing
Source URL. It merges JSON-LD specs with compact variant specs from the H1,
refreshes source price/name metadata, and writes atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    from scripts.crawl_fptshop_catalog import (
        USER_AGENT,
        _brand_name,
        _extract_fact_sentences,
        _evidence_specs,
        _format_vnd,
        _heading_specs,
        _offer,
        _product_json_ld,
        _structured_specs,
    )
except ModuleNotFoundError:
    from crawl_fptshop_catalog import (
        USER_AGENT,
        _brand_name,
        _extract_fact_sentences,
        _evidence_specs,
        _format_vnd,
        _heading_specs,
        _offer,
        _product_json_ld,
        _structured_specs,
    )


def _json_dict(value: str) -> dict[str, str]:
    try:
        payload = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_list(value: str) -> list[str]:
    try:
        payload = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _heading(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    return heading.get_text(" ", strip=True) if heading else ""


def _compat_context(row: dict[str, str], specs: dict[str, str], facts: list[str]) -> str:
    spec_text = ", ".join(f"{key}: {value}" for key, value in specs.items())
    fact_text = " ".join(facts[:4])
    return (
        f"Sản phẩm {row['Product']} {row['Name']}, thương hiệu {row['Brand']}, "
        f"mã sản phẩm {row['Product Code']}. Giá bán ghi nhận là {row['Price']}. "
        f"Thông số có nguồn gồm: {spec_text or 'chưa có thông số cấu trúc'}. "
        f"Điểm nổi bật từ trang sản phẩm: {fact_text or 'chưa có dữ kiện bổ sung'}."
    )


def enrich(
    path: Path,
    *,
    delay: float,
    limit: int | None = None,
    only_sparse: bool = False,
) -> tuple[int, int]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
        fieldnames = list(rows[0].keys()) if rows else []
    if "Spec Provenance JSON" not in fieldnames:
        fieldnames.append("Spec Provenance JSON")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.6",
        }
    )
    updated = 0
    failed = 0
    processed = 0

    for row in rows:
        if limit is not None and processed >= limit:
            break
        if only_sparse:
            current_specs = _json_dict(row.get("Structured Specs JSON", ""))
            threshold = 4 if row.get("Product") == "Laptop" else 3
            if len(current_specs) >= threshold:
                continue
        url = row.get("Source URL", "").strip()
        if not url:
            continue
        processed += 1
        try:
            response = session.get(url, timeout=35)
            response.raise_for_status()
            payload = _product_json_ld(response.text) or {}
            offer = _offer(payload)

            specs = _json_dict(row.get("Structured Specs JSON", ""))
            specs.update(_structured_specs(payload))
            specs.update(_heading_specs(response.text, row.get("Product", "")))

            facts = list(
                dict.fromkeys(
                    _json_list(row.get("Evidence Facts JSON", ""))
                    + _extract_fact_sentences(response.text)
                )
            )
            for key, value in _evidence_specs(facts).items():
                specs.setdefault(key, value)
            price = int(float(offer.get("price") or row.get("Price Value") or 0))
            if price > 0:
                row["Price Value"] = str(price)
                row["Price"] = _format_vnd(price)
            heading = _heading(response.text)
            if heading:
                row["Name"] = heading.removeprefix("Laptop ").strip()
            brand = _brand_name(payload)
            if brand:
                row["Brand"] = brand
            row["Availability"] = str(
                offer.get("availability") or row.get("Availability", "")
            )
            row["Price Valid Until"] = str(
                offer.get("priceValidUntil") or row.get("Price Valid Until", "")
            )
            row["Fetched At"] = datetime.now(UTC).isoformat()
            row["Structured Specs JSON"] = json.dumps(specs, ensure_ascii=False)
            row["Evidence Facts JSON"] = json.dumps(facts[:10], ensure_ascii=False)
            row["Spec Provenance JSON"] = json.dumps(
                {
                    key: {
                        "source_url": url,
                        "fetched_at": row["Fetched At"],
                        "confidence": "high",
                    }
                    for key in specs
                },
                ensure_ascii=False,
            )
            row["LLM_Context"] = _compat_context(row, specs, facts)
            updated += 1
            print(
                f"[{processed:03d}] {row['Product Code']} "
                f"{row['Product']} specs={len(specs)}"
            )
        except (requests.RequestException, ValueError) as exc:
            failed += 1
            print(f"[failed] {row.get('Product Code', '')}: {exc}")
        time.sleep(max(0.25, delay))

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return updated, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog",
        type=Path,
        nargs="?",
        default=Path("data/product_catalog_real.csv"),
    )
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-sparse", action="store_true")
    args = parser.parse_args()
    updated, failed = enrich(
        args.catalog,
        delay=args.delay,
        limit=args.limit,
        only_sparse=args.only_sparse,
    )
    print(f"Updated: {updated}; failed: {failed}")


if __name__ == "__main__":
    main()
