"""
Product Matcher Utility

Matches product queries against the internal product catalog using:
- Exact SKU (Product Code) match
- Exact product name match
- Fuzzy name matching via difflib (token-based similarity)

Supports Requirements 4.2: Cross-reference extracted prices với Internal DB
using product SKU/name matching.
"""

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ProductMatch:
    """
    Result of a product lookup.

    Attributes:
        product_name:  Product type/category (e.g. "Mobile Phone", "Laptop").
        brand:         Brand name (e.g. "Apple", "Samsung").
        sku:           Product Code from the catalog.
        price_raw:     Raw price string as stored in the CSV (e.g. "29.990.000 VNĐ").
        price_vnd:     Price normalized to a float in VND.
        match_score:   Similarity score in [0.0, 1.0].
        match_type:    One of "exact_sku", "exact_name", "fuzzy_name".
    """
    product_name: str
    brand: str
    sku: str
    price_raw: str
    price_vnd: float
    match_score: float
    match_type: str  # "exact_sku" | "exact_name" | "fuzzy_name"

    @property
    def display_name(self) -> str:
        """Human-readable product label: '<Brand> <Product>'."""
        return f"{self.brand} {self.product_name}".strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PRICE_CLEAN_RE = re.compile(r"[^\d]")


def _parse_price_vnd(raw: str) -> float:
    """
    Parse a Vietnamese price string like '29.990.000 VNĐ' or '7.857.000 VNĐ'
    into a float.  Dots are used as thousand separators.
    """
    digits = _PRICE_CLEAN_RE.sub("", raw)
    try:
        return float(digits)
    except ValueError:
        return 0.0


def _normalize(text: str) -> str:
    """
    Lowercase and strip non-alphanumeric characters (keep spaces).
    Used to create a canonical form for comparison.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_set_ratio(a: str, b: str) -> float:
    """
    Token-set similarity: compare the *intersection* of token sets against
    each individual set, then return the best ratio.

    This lets "iPhone 15" match "Apple iPhone 15 128GB" well because the
    shared tokens ("iphone", "15") dominate the shorter query.
    """
    tokens_a = set(_normalize(a).split())
    tokens_b = set(_normalize(b).split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b

    # Ratio of intersection vs each side
    ratio_a = len(intersection) / len(tokens_a)
    ratio_b = len(intersection) / len(tokens_b)

    # Also compute sequence-level similarity on normalized strings
    seq_ratio = SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

    # Weighted combination: token coverage of the query (ratio_a) is most
    # important, then sequence similarity for ordering close candidates.
    return 0.6 * ratio_a + 0.2 * ratio_b + 0.2 * seq_ratio


# ---------------------------------------------------------------------------
# ProductMatcher
# ---------------------------------------------------------------------------

class ProductMatcher:
    """
    Matches a free-text product query against the internal product catalog.

    Usage::

        matcher = ProductMatcher("data/product_catalog_clean.csv")
        result = matcher.find_product("Samsung Galaxy S23")
        if result:
            print(result.price_vnd)

    The catalog CSV must have at minimum the columns:
        ``Product Code``, ``Product``, ``Brand``, ``Price``

    Parameters
    ----------
    catalog_source:
        Path to the CSV file **or** a list of dicts with keys
        ``Product Code``, ``Product``, ``Brand``, ``Price``.
    threshold:
        Minimum similarity score (0–1) to accept a fuzzy match.
        Defaults to 0.6.
    """

    DEFAULT_THRESHOLD = 0.6

    def __init__(
        self,
        catalog_source: "str | Path | List[Dict]" = "data/product_catalog_clean.csv",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self._products: List[Dict] = []
        self._load(catalog_source)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_product(self, query: str) -> Optional[ProductMatch]:
        """
        Find the best matching product for *query*.

        Matching priority:
        1. Exact SKU match (case-insensitive)
        2. Exact product name match (normalized)
        3. Fuzzy token-based name match above ``self.threshold``

        Returns ``None`` if no match is found above the threshold.
        """
        if not query or not query.strip():
            return None

        query_stripped = query.strip()

        # 1. Exact SKU
        sku_match = self._match_by_sku(query_stripped)
        if sku_match:
            return sku_match

        # 2. Exact name
        exact_match = self._match_by_exact_name(query_stripped)
        if exact_match:
            return exact_match

        # 3. Fuzzy name
        return self._match_by_fuzzy_name(query_stripped)

    def find_all(self, query: str, top_k: int = 5) -> List[ProductMatch]:
        """
        Return up to *top_k* matches sorted by score descending.
        Includes all matches above ``self.threshold``.
        """
        if not query or not query.strip():
            return []

        candidates: List[Tuple[float, ProductMatch]] = []

        for product in self._products:
            score, match_type = self._score_product(query.strip(), product)
            if score >= self.threshold:
                candidates.append((
                    score,
                    self._build_match(product, score, match_type),
                ))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in candidates[:top_k]]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, source: "str | Path | List[Dict]") -> None:
        if isinstance(source, list):
            self._products = source
            return

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Product catalog not found: {path}")

        with path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self._products = list(reader)

    # ------------------------------------------------------------------
    # Match strategies
    # ------------------------------------------------------------------

    def _match_by_sku(self, query: str) -> Optional[ProductMatch]:
        """Return exact SKU match (case-insensitive)."""
        q_upper = query.upper()
        for product in self._products:
            if product.get("Product Code", "").upper() == q_upper:
                return self._build_match(product, 1.0, "exact_sku")
        return None

    def _match_by_exact_name(self, query: str) -> Optional[ProductMatch]:
        """Return exact normalized name match."""
        q_norm = _normalize(query)
        for product in self._products:
            # Build a combined name: "<Brand> <Product>"
            combined = f"{product.get('Brand', '')} {product.get('Product', '')}"
            if _normalize(combined) == q_norm or _normalize(product.get("Product", "")) == q_norm:
                return self._build_match(product, 1.0, "exact_name")
        return None

    def _match_by_fuzzy_name(self, query: str) -> Optional[ProductMatch]:
        """Return best fuzzy match above threshold."""
        best_score = -1.0
        best_product: Optional[Dict] = None

        for product in self._products:
            score = self._fuzzy_score(query, product)
            if score > best_score:
                best_score = score
                best_product = product

        if best_product is not None and best_score >= self.threshold:
            return self._build_match(best_product, best_score, "fuzzy_name")
        return None

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_product(self, query: str, product: Dict) -> Tuple[float, str]:
        """Return (score, match_type) for a single product."""
        q_upper = query.upper()
        if product.get("Product Code", "").upper() == q_upper:
            return 1.0, "exact_sku"

        combined = f"{product.get('Brand', '')} {product.get('Product', '')}"
        if _normalize(combined) == _normalize(query) or _normalize(product.get("Product", "")) == _normalize(query):
            return 1.0, "exact_name"

        score = self._fuzzy_score(query, product)
        return score, "fuzzy_name"

    def _fuzzy_score(self, query: str, product: Dict) -> float:
        """
        Compute fuzzy similarity between *query* and a product entry.

        Compares against:
        - "<Brand> <Product>" combined name
        - "<Product>" alone
        - "<Brand>" alone

        Returns the maximum score.
        """
        brand = product.get("Brand", "")
        name = product.get("Product", "")
        combined = f"{brand} {name}"

        scores = [
            _token_set_ratio(query, combined),
            _token_set_ratio(query, name),
        ]
        if brand:
            scores.append(_token_set_ratio(query, brand))

        return max(scores)

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------

    def _build_match(self, product: Dict, score: float, match_type: str) -> ProductMatch:
        price_raw = product.get("Price", "")
        return ProductMatch(
            product_name=product.get("Product", ""),
            brand=product.get("Brand", ""),
            sku=product.get("Product Code", ""),
            price_raw=price_raw,
            price_vnd=_parse_price_vnd(price_raw),
            match_score=score,
            match_type=match_type,
        )
