"""
Price Extractor Utility

Extracts and normalizes price mentions from text using regex patterns.
Supports Vietnamese (VND, đồng, triệu, nghìn, tỷ, đ) and English (USD, $) formats.

Supports Requirements 4.1: Price extraction from draft_response using regex/NLP
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExtractedPrice:
    """
    A single price mention extracted from text.

    Attributes:
        original_text:    The raw matched string (e.g. "25 triệu", "$500").
        amount_vnd:       Amount normalized to VND (float).  USD prices are
                          converted using a configurable exchange rate.
        currency:         Canonical currency code: "VND" or "USD".
        product_context:  Up to 80 characters of surrounding text that may
                          identify the product this price belongs to.
        start:            Start character offset in the source text.
        end:              End character offset in the source text.
    """
    original_text: str
    amount_vnd: float
    currency: str
    product_context: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Thousand-separator variants used in Vietnamese text:
#   25,000,000  /  25.000.000  /  25 000 000
_SEP = r"(?:[.,\s](?=\d{3}))*"

# Integer or decimal number with optional thousand separators
_NUM = r"\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d+)?"

# Compiled pattern pieces (used in the big pattern list below)
_CONTEXT_WINDOW = 80   # characters on each side for product_context


def _clean_number(raw: str) -> float:
    """
    Parse a raw number string that may use commas or dots as thousand
    separators and/or decimal separator.

    Strategy:
    - If the string ends with a 2-digit decimal part after the last separator
      (e.g. "1,234.56" or "1.234,56") treat the last separator as decimal.
    - Otherwise strip all separators and parse as integer-like float.
    """
    raw = raw.strip()

    # Detect European-style: last separator is comma with 2 decimal digits
    # e.g. "1.234,56"
    if re.search(r",\d{2}$", raw):
        raw = raw.replace(".", "").replace(",", ".")
        return float(raw)

    # Detect US-style: last separator is dot with 2 decimal digits
    # e.g. "1,234.56"
    if re.search(r"\.\d{2}$", raw):
        raw = raw.replace(",", "")
        return float(raw)

    # Otherwise: strip all thousand separators (commas, dots, spaces)
    raw = re.sub(r"[,.\s](?=\d{3})", "", raw)
    # Replace remaining comma/dot (decimal) with dot
    raw = raw.replace(",", ".").replace(" ", "")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _get_context(text: str, start: int, end: int, window: int = _CONTEXT_WINDOW) -> str:
    """Return surrounding text around [start, end] for product identification."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    return text[ctx_start:ctx_end].strip()


# ---------------------------------------------------------------------------
# PriceExtractor
# ---------------------------------------------------------------------------

class PriceExtractor:
    """
    Deterministic regex-based price extractor.

    Handles:
    - Vietnamese: "25 triệu", "25,000,000 VND", "25.000.000đ",
                  "25.000.000 đồng", "500 nghìn", "2 tỷ"
    - English:    "$500", "500 USD", "500 dollars"
    - Price ranges: "25-30 triệu", "$400-$500"

    All amounts are normalized to VND using ``usd_to_vnd_rate``.
    """

    # Default USD → VND exchange rate (approximate; override as needed)
    DEFAULT_USD_RATE = 25_000.0

    def __init__(self, usd_to_vnd_rate: float = DEFAULT_USD_RATE):
        self.usd_to_vnd_rate = usd_to_vnd_rate
        self._patterns = self._build_patterns()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> List[ExtractedPrice]:
        """
        Extract all price mentions from *text*.

        Returns a list of :class:`ExtractedPrice` objects sorted by their
        position in the source text.  Overlapping matches are deduplicated
        (the longer match wins).
        """
        if not text or not text.strip():
            return []

        raw_matches: List[Tuple[int, int, ExtractedPrice]] = []

        for pattern, currency, multiplier in self._patterns:
            for m in pattern.finditer(text):
                amount_raw = m.group("amount")
                base = _clean_number(amount_raw)
                amount_vnd = base * multiplier
                if currency == "USD":
                    amount_vnd = base * self.usd_to_vnd_rate

                price = ExtractedPrice(
                    original_text=m.group(0),
                    amount_vnd=amount_vnd,
                    currency=currency,
                    product_context=_get_context(text, m.start(), m.end()),
                    start=m.start(),
                    end=m.end(),
                )
                raw_matches.append((m.start(), m.end(), price))

        # Handle price ranges: detect "X-Y <unit>" and keep both endpoints
        range_prices = self._extract_ranges(text)
        raw_matches.extend(range_prices)

        # Deduplicate overlapping matches (keep longest span)
        deduped = _deduplicate(raw_matches)

        return sorted(deduped, key=lambda p: p.start)

    # ------------------------------------------------------------------
    # Pattern construction
    # ------------------------------------------------------------------

    def _build_patterns(self) -> List[Tuple[re.Pattern, str, float]]:
        """
        Returns list of (compiled_pattern, currency_code, vnd_multiplier).

        For USD patterns the multiplier is ignored; conversion uses
        ``self.usd_to_vnd_rate`` at match time.
        """
        num = r"(?P<amount>\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d+)?)"

        patterns = [
            # ---- Vietnamese VND explicit --------------------------------
            # 25,000,000 VND  /  25.000.000 VND  /  25000000VND
            (rf"{num}\s*(?:VND|vnđ)", "VND", 1.0),

            # 25,000,000đ  /  25.000.000đ  (đ suffix, no space required)
            (rf"{num}\s*đ(?!\w)", "VND", 1.0),

            # 25,000,000 đồng
            (rf"{num}\s*đồng", "VND", 1.0),

            # ---- Vietnamese shorthand -----------------------------------
            # 25 triệu  /  25.5 triệu
            (rf"{num}\s*triệu", "VND", 1_000_000.0),

            # 500 nghìn  /  500 ngàn
            (rf"{num}\s*(?:nghìn|ngàn)", "VND", 1_000.0),

            # 2 tỷ
            (rf"{num}\s*tỷ", "VND", 1_000_000_000.0),

            # ---- English USD -------------------------------------------
            # $500  /  $1,234.56
            (r"\$\s*(?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d+)?)", "USD", 1.0),

            # 500 USD  /  500 dollars
            (rf"{num}\s*(?:USD|dollars?)", "USD", 1.0),
        ]

        compiled = []
        for pat, currency, mult in patterns:
            try:
                compiled.append((re.compile(pat, re.IGNORECASE | re.UNICODE), currency, mult))
            except re.error:
                pass  # skip malformed patterns gracefully

        return compiled

    # ------------------------------------------------------------------
    # Price range extraction
    # ------------------------------------------------------------------

    def _extract_ranges(self, text: str) -> List[Tuple[int, int, "ExtractedPrice"]]:
        """
        Detect price ranges like "25-30 triệu" or "$400-$500" and return
        both the lower and upper bound as separate ExtractedPrice entries.
        """
        results: List[Tuple[int, int, ExtractedPrice]] = []

        range_patterns = [
            # 25-30 triệu  /  25 - 30 triệu
            (
                r"(?P<lo>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*[-–]\s*"
                r"(?P<hi>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*triệu",
                "VND", 1_000_000.0,
            ),
            # 500-600 nghìn
            (
                r"(?P<lo>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*[-–]\s*"
                r"(?P<hi>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(?:nghìn|ngàn)",
                "VND", 1_000.0,
            ),
            # $400-$500
            (
                r"\$\s*(?P<lo>\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[-–]\s*"
                r"\$\s*(?P<hi>\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
                "USD", 1.0,
            ),
            # 400-500 USD
            (
                r"(?P<lo>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*[-–]\s*"
                r"(?P<hi>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*USD",
                "USD", 1.0,
            ),
        ]

        for pat, currency, mult in range_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE | re.UNICODE):
                lo_raw = m.group("lo")
                hi_raw = m.group("hi")
                lo_base = _clean_number(lo_raw)
                hi_base = _clean_number(hi_raw)

                if currency == "USD":
                    lo_vnd = lo_base * self.usd_to_vnd_rate
                    hi_vnd = hi_base * self.usd_to_vnd_rate
                else:
                    lo_vnd = lo_base * mult
                    hi_vnd = hi_base * mult

                ctx = _get_context(text, m.start(), m.end())

                # Use actual sub-match positions for lo and hi so deduplication
                # treats them as distinct entries
                lo_start = m.start() + m.group(0).index(lo_raw)
                lo_end = lo_start + len(lo_raw)
                hi_start = m.start() + m.group(0).rindex(hi_raw)
                hi_end = hi_start + len(hi_raw)

                results.append((
                    lo_start, lo_end,
                    ExtractedPrice(
                        original_text=lo_raw,
                        amount_vnd=lo_vnd,
                        currency=currency,
                        product_context=ctx,
                        start=lo_start,
                        end=lo_end,
                    )
                ))
                results.append((
                    hi_start, hi_end,
                    ExtractedPrice(
                        original_text=hi_raw,
                        amount_vnd=hi_vnd,
                        currency=currency,
                        product_context=ctx,
                        start=hi_start,
                        end=hi_end,
                    )
                ))

        return results


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def _deduplicate(
    matches: List[Tuple[int, int, ExtractedPrice]]
) -> List[ExtractedPrice]:
    """
    Remove overlapping matches, keeping the one with the longest span.
    When spans are equal, keep the first encountered.
    """
    if not matches:
        return []

    # Sort by start, then by span length descending
    sorted_m = sorted(matches, key=lambda x: (x[0], -(x[1] - x[0])))

    kept: List[ExtractedPrice] = []
    last_end = -1

    for start, end, price in sorted_m:
        if start >= last_end:
            kept.append(price)
            last_end = end

    return kept


# ---------------------------------------------------------------------------
# Module-level convenience function (mirrors text_utils API)
# ---------------------------------------------------------------------------

_default_extractor = PriceExtractor()


def extract_prices_detailed(text: str, usd_to_vnd_rate: float = PriceExtractor.DEFAULT_USD_RATE) -> List[ExtractedPrice]:
    """
    Convenience wrapper around :class:`PriceExtractor`.

    Returns a list of :class:`ExtractedPrice` objects sorted by position.
    """
    extractor = PriceExtractor(usd_to_vnd_rate=usd_to_vnd_rate)
    return extractor.extract(text)
