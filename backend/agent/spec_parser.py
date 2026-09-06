"""Deterministic spec parsing for catalog-grounded product facts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD",
        value.casefold().replace("Ä‘", "d").replace("đ", "d"),
    )
    ascii_value = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9.+-]+", " ", ascii_value).strip()


@dataclass(frozen=True)
class ParsedSpec:
    field: str
    value: str
    source_text: str


FIELD_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cpu", ("cpu", "chip", "core", "bo xu ly", "vi xu ly")),
    ("gpu", ("card do hoa", "gpu")),
    ("ram", ("ram",)),
    ("storage", ("o cung ssd", "ssd", "bo nho trong", "storage")),
    ("screen", ("kich thuoc man hinh", "man hinh")),
    ("resolution", ("do phan giai",)),
    ("refresh_rate", ("tan so quet",)),
    ("battery", ("dung luong pin", "pin", "battery")),
    ("weight", ("trong luong", "weight")),
    ("os", ("he dieu hanh", "os")),
)


def parse_spec_line(spec: str) -> ParsedSpec | None:
    source_text = spec.strip()
    normalized = normalize_text(source_text)
    if not normalized or normalized.endswith(" nan"):
        return None

    for field, aliases in FIELD_ALIASES:
        for alias in aliases:
            if normalized == alias:
                return None
            if normalized.startswith(alias + " "):
                value = source_text[len(source_text) - len(normalized[len(alias) :].strip()) :]
                return ParsedSpec(
                    field=field,
                    value=value.strip(" :-"),
                    source_text=source_text,
                )

    return None


def parse_specs(specs: list[str] | tuple[str, ...]) -> dict[str, ParsedSpec]:
    parsed: dict[str, ParsedSpec] = {}
    for spec in specs:
        item = parse_spec_line(spec)
        if item is not None and item.field not in parsed:
            parsed[item.field] = item
    return parsed


def parse_capacity_gb(value: str) -> int | None:
    compact = value.casefold().replace(" ", "")
    match = re.search(r"(\d+(?:[.,]\d+)?)(tb|gb)", compact)
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    unit = match.group(2)
    if unit == "tb":
        number *= 1024
    return int(number)


def parse_float_with_unit(value: str, unit: str) -> float | None:
    compact = value.casefold().replace(" ", "")
    match = re.search(r"(\d+(?:[.,]\d+)?)" + re.escape(unit), compact)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def parse_int_with_unit(value: str, unit: str) -> int | None:
    parsed = parse_float_with_unit(value, unit)
    return int(parsed) if parsed is not None else None


def infer_cpu_tier(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value)
    patterns = (
        (r"\bcore\s+ultra\s+([579])\b", "Core Ultra {group}"),
        (r"\bcore\s+i([3579])\b", "i{group}"),
        (r"\bi([3579])\b", "i{group}"),
        (r"\bcore\s+([3579])\b", "Core {group}"),
        (r"\bryzen\s+ai\s+([579])\b", "Ryzen AI {group}"),
        (r"\bryzen\s+([3579])\b", "Ryzen {group}"),
    )
    for pattern, template in patterns:
        match = re.search(pattern, normalized)
        if match:
            return template.format(group=match.group(1))
    return None


def infer_gpu_type(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value)
    if any(
        marker in normalized
        for marker in (
            "rtx",
            "gtx",
            "mx",
            "radeon rx",
            "nvidia geforce",
            "geforce",
            "intel arc a",
        )
    ):
        return "dedicated"
    if any(
        marker in normalized
        for marker in (
            "intel graphics",
            "intel uhd",
            "iris xe",
            "radeon graphics",
            "amd radeon graphics",
            "adreno",
        )
    ):
        return "integrated"
    return "unknown"
