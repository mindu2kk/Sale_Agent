"""Validate the crawled real catalog before activating it."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_FIELDS = (
    "Product Code",
    "Product",
    "Brand",
    "Name",
    "Price Value",
    "Source URL",
    "Fetched At",
    "LLM_Context",
)


def validate(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    errors: list[str] = []
    warnings: list[str] = []
    codes = [row.get("Product Code", "").strip() for row in rows]

    for field in REQUIRED_FIELDS:
        missing = sum(not row.get(field, "").strip() for row in rows)
        if missing:
            errors.append(f"{field}: missing in {missing} rows")

    duplicates = [code for code, count in Counter(codes).items() if code and count > 1]
    if duplicates:
        errors.append(f"duplicate SKUs: {', '.join(duplicates[:10])}")

    for index, row in enumerate(rows, start=2):
        try:
            price = int(row.get("Price Value", "0"))
        except ValueError:
            price = 0
        if price < 300_000 or price > 250_000_000:
            warnings.append(f"row {index}: suspicious price {price}")
        source = urlparse(row.get("Source URL", ""))
        if source.scheme != "https" or source.netloc != "fptshop.com.vn":
            errors.append(f"row {index}: invalid source URL")
        specs: dict = {}
        try:
            specs = json.loads(row.get("Structured Specs JSON", "{}"))
            facts = json.loads(row.get("Evidence Facts JSON", "[]"))
            if not isinstance(specs, dict) or not isinstance(facts, list):
                raise ValueError
            minimum_specs = 4 if row.get("Product") == "Laptop" else 3
            if len(specs) < minimum_specs:
                warnings.append(
                    f"row {index}: sparse specs ({len(specs)}/{minimum_specs}) "
                    f"for SKU {row.get('Product Code', '')}"
                )
        except (json.JSONDecodeError, ValueError):
            errors.append(f"row {index}: invalid JSON evidence fields")
        provenance_text = row.get("Spec Provenance JSON", "")
        if provenance_text:
            try:
                provenance = json.loads(provenance_text)
                if not isinstance(provenance, dict):
                    raise ValueError
                missing_provenance = set(specs) - set(provenance)
                if missing_provenance:
                    warnings.append(
                        f"row {index}: {len(missing_provenance)} specs lack provenance"
                    )
            except (json.JSONDecodeError, ValueError):
                errors.append(f"row {index}: invalid Spec Provenance JSON")

    print(f"Rows: {len(rows)}")
    print(f"Unique SKUs: {len(set(codes))}")
    print(f"Categories: {dict(Counter(row['Product'] for row in rows))}")
    print(f"Brands: {len(set(row['Brand'] for row in rows))}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings[:20]:
        print(f"  WARN {warning}")
    print(f"Errors: {len(errors)}")
    for error in errors[:20]:
        print(f"  ERROR {error}")
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog",
        type=Path,
        nargs="?",
        default=Path("data/product_catalog_real.csv"),
    )
    args = parser.parse_args()
    raise SystemExit(validate(args.catalog))


if __name__ == "__main__":
    main()
