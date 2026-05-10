"""
Semantic Similarity Analyzer

Lightweight semantic similarity analysis between objection and response text.
Uses TF-IDF-style keyword overlap + intent detection (no heavy ML dependencies).

Supports Requirements 6: Topic Relevance Assessment
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Intent keyword taxonomy (Vietnamese + English)
# ---------------------------------------------------------------------------

INTENT_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "price": {
        "objection": [
            "đắt", "giá", "expensive", "cost", "budget", "tiền", "money",
            "rẻ", "cheap", "ngân sách", "chi phí", "bao nhiêu", "how much",
            "price", "pricing", "affordable", "giá cả", "giá tiền",
        ],
        "response": [
            "giá", "price", "cost", "đồng", "vnd", "usd", "$", "triệu",
            "nghìn", "discount", "giảm giá", "khuyến mãi", "promotion",
            "affordable", "budget", "tiết kiệm", "save",
        ],
    },
    "feature": {
        "objection": [
            "tính năng", "specs", "performance", "feature", "chức năng",
            "cấu hình", "configuration", "specification", "thông số",
            "camera", "pin", "battery", "màn hình", "screen", "display",
            "processor", "chip", "ram", "storage", "bộ nhớ",
        ],
        "response": [
            "tính năng", "feature", "specs", "performance", "camera",
            "pin", "battery", "màn hình", "screen", "processor", "chip",
            "ram", "storage", "bộ nhớ", "cấu hình", "thông số",
        ],
    },
    "comparison": {
        "objection": [
            "so sánh", "compare", "khác gì", "difference", "vs", "versus",
            "better", "tốt hơn", "hơn", "hay hơn", "nên chọn", "which",
            "between", "giữa", "or", "hay là",
        ],
        "response": [
            "so sánh", "compare", "khác", "difference", "better", "tốt hơn",
            "trong khi", "while", "whereas", "on the other hand", "ngược lại",
            "điểm khác biệt", "ưu điểm", "nhược điểm", "advantage",
        ],
    },
    "policy": {
        "objection": [
            "bảo hành", "đổi trả", "warranty", "return", "policy",
            "hoàn tiền", "refund", "exchange", "đổi máy", "thay thế",
            "chính sách", "điều khoản", "terms", "condition",
        ],
        "response": [
            "bảo hành", "warranty", "đổi trả", "return", "hoàn tiền",
            "refund", "chính sách", "policy", "tháng", "năm", "month",
            "year", "điều kiện", "condition", "áp dụng", "apply",
        ],
    },
}

# Empathy phrases (Vietnamese + English)
EMPATHY_PHRASES: List[str] = [
    # Vietnamese
    "hiểu được", "hiểu rằng", "cảm ơn", "quan tâm", "lo lắng",
    "băn khoăn", "thắc mắc", "chia sẻ", "đồng cảm", "thông cảm",
    "rất tiếc", "xin lỗi", "chúng tôi hiểu", "chúng tôi biết",
    "hoàn toàn hiểu", "hoàn toàn đồng ý",
    # English
    "understand", "appreciate", "concern", "worry", "i see",
    "that's a great question", "thank you for", "i hear you",
    "completely understand", "totally understand", "valid concern",
    "good point", "fair point",
]


@dataclass
class SimilarityResult:
    """Result of semantic similarity analysis between objection and response."""

    similarity_score: float  # 0.0 - 1.0 overall similarity
    detected_intents: List[str]  # intent categories detected in objection
    coverage_ratio: float  # 0.0 - 1.0 how well response covers objection intents
    missing_aspects: List[str]  # aspects from objection not covered in response
    has_empathy: bool  # whether response contains empathy statements
    intent_coverage_detail: Dict[str, float] = field(default_factory=dict)
    # per-intent coverage scores (0-1)


class SemanticSimilarityAnalyzer:
    """
    Lightweight semantic similarity analyzer using keyword overlap + intent detection.

    No heavy ML dependencies — uses TF-IDF-inspired term weighting and
    intent-based coverage analysis.
    """

    def __init__(
        self,
        min_coverage_ratio: float = 0.7,
        empathy_bonus_enabled: bool = True,
    ):
        self.min_coverage_ratio = min_coverage_ratio
        self.empathy_bonus_enabled = empathy_bonus_enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, objection: str, response: str) -> SimilarityResult:
        """
        Analyze semantic similarity between objection and response.

        Args:
            objection: Customer objection text
            response: Draft response text

        Returns:
            SimilarityResult with similarity_score, detected_intents,
            coverage_ratio, missing_aspects, has_empathy
        """
        objection_lower = objection.lower()
        response_lower = response.lower()

        # 1. Detect intents from objection
        detected_intents = self._detect_intents(objection_lower)

        # 2. Calculate per-intent coverage
        intent_coverage = self._calculate_intent_coverage(
            detected_intents, objection_lower, response_lower
        )

        # 3. Overall coverage ratio
        coverage_ratio = self._compute_coverage_ratio(
            detected_intents, intent_coverage, objection_lower, response_lower
        )

        # 4. Lexical similarity (TF-IDF-style word overlap)
        lexical_sim = self._lexical_similarity(objection_lower, response_lower)

        # 5. Combined similarity score
        if detected_intents:
            # Weight intent coverage more heavily when intents are detected
            similarity_score = 0.6 * coverage_ratio + 0.4 * lexical_sim
        else:
            similarity_score = lexical_sim

        # 6. Empathy detection
        has_empathy = self._has_empathy(response_lower)

        # 7. Apply empathy bonus to coverage (capped at 1.0)
        if self.empathy_bonus_enabled and has_empathy:
            coverage_ratio = min(1.0, coverage_ratio * 1.1)
            similarity_score = min(1.0, similarity_score * 1.05)

        # 8. Identify missing aspects
        missing_aspects = self._identify_missing_aspects(
            detected_intents, intent_coverage, objection_lower, response_lower
        )

        return SimilarityResult(
            similarity_score=round(similarity_score, 4),
            detected_intents=detected_intents,
            coverage_ratio=round(coverage_ratio, 4),
            missing_aspects=missing_aspects,
            has_empathy=has_empathy,
            intent_coverage_detail={k: round(v, 4) for k, v in intent_coverage.items()},
        )

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_intents(self, objection_lower: str) -> List[str]:
        """Detect intent categories present in the objection."""
        detected = []
        for intent, kw_groups in INTENT_KEYWORDS.items():
            objection_kws = kw_groups["objection"]
            if any(kw in objection_lower for kw in objection_kws):
                detected.append(intent)
        return detected

    # ------------------------------------------------------------------
    # Coverage calculation
    # ------------------------------------------------------------------

    def _calculate_intent_coverage(
        self,
        detected_intents: List[str],
        objection_lower: str,
        response_lower: str,
    ) -> Dict[str, float]:
        """
        For each detected intent, calculate how well the response covers it.

        Coverage = matched response keywords / total response keywords for intent,
        but boosted so that having even a few relevant keywords gives meaningful credit.
        """
        coverage: Dict[str, float] = {}
        for intent in detected_intents:
            kw_groups = INTENT_KEYWORDS[intent]
            obj_kws = kw_groups["objection"]
            resp_kws = kw_groups["response"]

            # Keywords from this intent that appear in the objection
            obj_matched = [kw for kw in obj_kws if kw in objection_lower]
            # Response keywords for this intent that appear in the response
            resp_matched = [kw for kw in resp_kws if kw in response_lower]

            if not obj_matched:
                coverage[intent] = 0.0
                continue

            if not resp_kws:
                coverage[intent] = 0.0
                continue

            # Use a sqrt-scaled ratio so that having a few keywords gives good credit.
            # e.g. 3/15 keywords = 0.2 raw → sqrt(0.2) ≈ 0.45
            raw_ratio = len(resp_matched) / len(resp_kws)
            scaled = math.sqrt(raw_ratio) if raw_ratio > 0 else 0.0
            coverage[intent] = min(1.0, scaled)

        return coverage

    def _compute_coverage_ratio(
        self,
        detected_intents: List[str],
        intent_coverage: Dict[str, float],
        objection_lower: str,
        response_lower: str,
    ) -> float:
        """
        Compute overall coverage ratio.

        If intents detected: take the max of (average intent coverage) and
        (lexical similarity), so that a highly similar response always gets
        credit even when intent keyword lists are sparse.
        If no intents: fall back to lexical overlap.
        """
        lexical = self._lexical_similarity(objection_lower, response_lower)

        if not detected_intents:
            return lexical

        scores = [intent_coverage.get(intent, 0.0) for intent in detected_intents]
        intent_avg = sum(scores) / len(scores)

        # Use the higher of intent-based coverage and lexical similarity
        return max(intent_avg, lexical)

    # ------------------------------------------------------------------
    # Lexical similarity (TF-IDF-inspired word overlap)
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, keeping Vietnamese characters."""
        # Keep alphanumeric + Vietnamese unicode characters
        tokens = re.findall(
            r"[a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+",
            text,
        )
        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency for a list of tokens."""
        if not tokens:
            return {}
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = len(tokens)
        return {t: count / total for t, count in freq.items()}

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        """
        Compute lexical similarity using TF-weighted Jaccard overlap.

        Returns a score in [0, 1].
        """
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        tf1 = self._compute_tf(tokens1)
        tf2 = self._compute_tf(tokens2)

        vocab = set(tf1.keys()) | set(tf2.keys())

        # Weighted Jaccard: sum(min(tf1, tf2)) / sum(max(tf1, tf2))
        numerator = sum(min(tf1.get(w, 0.0), tf2.get(w, 0.0)) for w in vocab)
        denominator = sum(max(tf1.get(w, 0.0), tf2.get(w, 0.0)) for w in vocab)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    # ------------------------------------------------------------------
    # Empathy detection
    # ------------------------------------------------------------------

    def _has_empathy(self, response_lower: str) -> bool:
        """Check if response contains empathy statements."""
        return any(phrase in response_lower for phrase in EMPATHY_PHRASES)

    # ------------------------------------------------------------------
    # Missing aspects identification
    # ------------------------------------------------------------------

    def _identify_missing_aspects(
        self,
        detected_intents: List[str],
        intent_coverage: Dict[str, float],
        objection_lower: str,
        response_lower: str,
    ) -> List[str]:
        """
        Identify aspects from the objection that are not covered in the response.

        Returns human-readable descriptions of missing aspects.
        """
        missing: List[str] = []

        intent_labels = {
            "price": "Price information not addressed",
            "feature": "Product features/specs not covered",
            "comparison": "Comparison details not provided",
            "policy": "Policy information (warranty/return) not addressed",
        }

        for intent in detected_intents:
            coverage = intent_coverage.get(intent, 0.0)
            if coverage < 0.3:
                label = intent_labels.get(intent, f"{intent} not addressed")
                missing.append(label)

        # Also check for specific keywords in objection that are absent in response
        objection_tokens = set(self._tokenize(objection_lower))
        response_tokens = set(self._tokenize(response_lower))

        # Significant words in objection not found in response (length > 3 to skip stopwords)
        significant_missing = [
            w for w in objection_tokens
            if w not in response_tokens and len(w) > 3
        ]

        # Only report if there are many missing significant words
        if len(significant_missing) > 5 and not missing:
            missing.append(
                f"Response does not address key terms: {', '.join(list(significant_missing)[:5])}"
            )

        return missing
