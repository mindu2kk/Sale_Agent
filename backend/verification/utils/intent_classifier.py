"""
Intent Classifier for Objection Types

Classifies customer objections into structured intent categories with
confidence scoring. Supports Requirements 6: Topic Relevance Assessment.

Intent categories (Vietnamese + English):
- price: Price concerns, cost questions, budget inquiries
- feature: Product specs, performance, technical details
- comparison: Comparing products or brands
- policy: Warranty, return, exchange, service terms
- availability: Stock, delivery, lead time
- support: Technical support, after-sales service
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

# Each intent maps to:
#   "keywords"  — terms that signal this intent in the objection
#   "weight"    — base confidence contribution per keyword match (0–1)
#   "patterns"  — regex patterns for stronger signals (multi-word phrases)
INTENT_TAXONOMY: Dict[str, Dict] = {
    "price": {
        "keywords": [
            # Vietnamese
            "giá", "đắt", "rẻ", "tiền", "chi phí", "ngân sách", "bao nhiêu",
            "giá cả", "giá tiền", "giảm giá", "khuyến mãi", "ưu đãi",
            "tốn", "mắc", "phí", "thanh toán", "trả góp",
            # English
            "price", "cost", "expensive", "cheap", "budget", "money",
            "how much", "pricing", "affordable", "discount", "promotion",
            "payment", "installment",
        ],
        "weight": 0.15,
        "patterns": [
            r"giá\s+(?:bao nhiêu|như thế nào|thế nào)",
            r"(?:đắt|mắc)\s+(?:quá|lắm|vậy)",
            r"how\s+much\s+(?:does|is|cost)",
            r"(?:price|cost)\s+(?:too|very)\s+(?:high|expensive)",
            r"(?:có|được)\s+(?:giảm giá|khuyến mãi|ưu đãi)",
        ],
    },
    "feature": {
        "keywords": [
            # Vietnamese — products
            "laptop", "máy tính", "điện thoại", "tablet", "máy tính bảng",
            "gaming", "game", "đồ họa", "render", "chơi game",
            "tầm", "loại", "dòng", "model", "sản phẩm", "mua",
            "cần", "muốn", "tìm", "giới thiệu", "gợi ý", "tư vấn",
            # Vietnamese — specs
            "tính năng", "chức năng", "cấu hình", "thông số", "hiệu năng",
            "camera", "pin", "màn hình", "bộ nhớ", "ram", "chip", "cpu",
            "gpu", "loa", "âm thanh", "kết nối", "wifi", "bluetooth",
            "sạc", "dung lượng", "độ phân giải", "tốc độ",
            # English
            "laptop", "phone", "tablet", "gaming", "recommend", "suggest",
            "need", "want", "looking for", "best",
            "feature", "spec", "performance", "specification", "display",
            "screen", "battery", "storage", "processor", "resolution",
            "connectivity", "charging", "capacity",
        ],
        "weight": 0.15,
        "patterns": [
            r"(?:cần|muốn|tìm|mua)\s+(?:laptop|điện thoại|tablet|máy tính)",
            r"laptop\s+(?:gaming|đồ họa|văn phòng|sinh viên)",
            r"tầm\s+\d+\s+(?:triệu|tr|million)",
            r"(?:gợi ý|tư vấn|giới thiệu)\s+(?:cho tôi|mình|em)",
            r"tính năng\s+(?:gì|như thế nào|có gì)",
            r"(?:camera|pin|màn hình)\s+(?:như thế nào|tốt không|bao nhiêu)",
            r"(?:specs?|specifications?)\s+(?:of|for)",
            r"how\s+(?:good|fast|long)\s+is\s+the",
            r"(?:recommend|suggest)\s+(?:a|an|the|me|some)",
        ],
    },
    "comparison": {
        "keywords": [
            # Vietnamese
            "so sánh", "khác gì", "hơn", "tốt hơn", "hay hơn", "nên chọn",
            "giữa", "hay là", "hoặc", "so với", "đối với", "thay vì",
            # English
            "compare", "comparison", "difference", "vs", "versus", "better",
            "between", "which", "or", "alternative", "instead of",
        ],
        "weight": 0.15,
        "patterns": [
            r"so sánh\s+\w+\s+(?:vs?\.?|và|với)\s+\w+",
            r"\w+\s+(?:vs?\.?|versus)\s+\w+",
            r"(?:nên|should)\s+(?:chọn|mua|buy|choose)\s+(?:cái nào|which)",
            r"(?:khác|different)\s+(?:gì|nhau|how)",
        ],
    },
    "policy": {
        "keywords": [
            # Vietnamese
            "bảo hành", "đổi trả", "hoàn tiền", "đổi máy", "chính sách",
            "điều khoản", "điều kiện", "quy định", "thủ tục", "hỗ trợ",
            "sửa chữa", "thay thế", "bảo trì",
            # English
            "warranty", "return", "refund", "exchange", "policy", "terms",
            "condition", "guarantee", "repair", "replacement", "service",
            "support", "maintenance",
        ],
        "weight": 0.15,
        "patterns": [
            r"(?:bảo hành|warranty)\s+(?:bao lâu|mấy tháng|mấy năm|how long)",
            r"(?:đổi trả|return)\s+(?:như thế nào|trong bao lâu|policy)",
            r"(?:hoàn tiền|refund)\s+(?:được không|có không|policy)",
            r"(?:chính sách|policy)\s+(?:đổi trả|bảo hành|return|warranty)",
        ],
    },
    "availability": {
        "keywords": [
            # Vietnamese
            "còn hàng", "hết hàng", "có sẵn", "giao hàng", "vận chuyển",
            "thời gian", "bao giờ", "khi nào", "nhận hàng", "đặt hàng",
            "tồn kho", "nhập hàng",
            # English
            "in stock", "out of stock", "available", "delivery", "shipping",
            "when", "how long", "lead time", "order", "backorder",
        ],
        "weight": 0.15,
        "patterns": [
            r"(?:còn|có)\s+hàng\s+(?:không|chưa)",
            r"(?:giao hàng|delivery)\s+(?:bao lâu|trong bao lâu|how long)",
            r"(?:khi nào|when)\s+(?:có|available|in stock)",
            r"(?:đặt hàng|order)\s+(?:được không|như thế nào|how)",
        ],
    },
    "support": {
        "keywords": [
            # Vietnamese
            "hỗ trợ kỹ thuật", "trung tâm bảo hành", "hotline", "liên hệ",
            "tư vấn", "hướng dẫn", "cài đặt", "sử dụng", "lỗi", "vấn đề",
            "khắc phục", "giải quyết",
            # English
            "technical support", "service center", "contact", "help",
            "guide", "setup", "install", "issue", "problem", "fix", "resolve",
        ],
        "weight": 0.15,
        "patterns": [
            r"(?:hỗ trợ|support)\s+(?:kỹ thuật|technical|24/7)",
            r"(?:lỗi|issue|problem)\s+(?:như thế nào|how to|fix)",
            r"(?:liên hệ|contact)\s+(?:ở đâu|như thế nào|how|where)",
        ],
    },
}

# Human-readable labels for each intent
INTENT_LABELS: Dict[str, str] = {
    "price": "Price / Cost Inquiry",
    "feature": "Product Features / Specs",
    "comparison": "Product Comparison",
    "policy": "Warranty / Return Policy",
    "availability": "Stock / Delivery",
    "support": "Technical Support",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntentScore:
    """Confidence score for a single detected intent."""
    intent: str
    confidence: float          # 0.0 – 1.0
    matched_keywords: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)
    label: str = ""

    def __post_init__(self):
        self.label = INTENT_LABELS.get(self.intent, self.intent)


@dataclass
class ClassificationResult:
    """
    Full classification result for an objection.

    Attributes:
        primary_intent: Highest-confidence intent (or "general_inquiry" if none).
        intents: All detected intents sorted by confidence descending.
        is_multi_intent: True when ≥2 intents detected above threshold.
        raw_text: Original objection text (lowercased).
    """
    primary_intent: str
    intents: List[IntentScore]
    is_multi_intent: bool
    raw_text: str

    @property
    def intent_names(self) -> List[str]:
        """Ordered list of detected intent names."""
        return [s.intent for s in self.intents]

    @property
    def top_confidence(self) -> float:
        """Confidence of the primary intent (0 if none detected)."""
        return self.intents[0].confidence if self.intents else 0.0

    def get_intent(self, name: str) -> Optional[IntentScore]:
        """Return the IntentScore for a specific intent name, or None."""
        return next((s for s in self.intents if s.intent == name), None)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Rule-based intent classifier for customer objections.

    Uses keyword matching + regex pattern scoring to assign confidence
    scores to each intent category. Supports Vietnamese and English text.

    Args:
        confidence_threshold: Minimum confidence to include an intent in results.
            Default 0.2 — keeps intents with at least one strong keyword match.
        max_intents: Maximum number of intents to return per classification.
            Default 3 — covers most multi-intent objections without noise.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.2,
        max_intents: int = 3,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_intents = max_intents
        # Pre-compile regex patterns for performance
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {
            intent: [
                (re.compile(p, re.IGNORECASE), p)
                for p in data["patterns"]
            ]
            for intent, data in INTENT_TAXONOMY.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, objection: str) -> ClassificationResult:
        """
        Classify an objection into one or more intent categories.

        Algorithm:
        1. Lowercase and normalize the objection text.
        2. For each intent, score keyword matches (additive, capped at 0.6).
        3. For each intent, score pattern matches (+0.25 per match, capped at 0.4).
        4. Combine keyword + pattern scores (capped at 1.0).
        5. Filter by confidence_threshold, sort descending, take top max_intents.
        6. Return ClassificationResult with primary_intent and full list.

        Args:
            objection: Raw customer objection text.

        Returns:
            ClassificationResult with detected intents and confidence scores.
        """
        text = objection.lower().strip()

        scores: List[IntentScore] = []
        for intent, data in INTENT_TAXONOMY.items():
            score = self._score_intent(intent, data, text)
            if score.confidence >= self.confidence_threshold:
                scores.append(score)

        # Sort by confidence descending
        scores.sort(key=lambda s: s.confidence, reverse=True)
        top_scores = scores[: self.max_intents]

        primary = top_scores[0].intent if top_scores else "general_inquiry"
        is_multi = len(top_scores) >= 2

        return ClassificationResult(
            primary_intent=primary,
            intents=top_scores,
            is_multi_intent=is_multi,
            raw_text=text,
        )

    def classify_batch(self, objections: List[str]) -> List[ClassificationResult]:
        """Classify a list of objections. Returns results in the same order."""
        return [self.classify(obj) for obj in objections]

    def get_primary_intent(self, objection: str) -> str:
        """Convenience method — returns only the primary intent name."""
        return self.classify(objection).primary_intent

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_intent(
        self, intent: str, data: Dict, text: str
    ) -> IntentScore:
        """
        Compute a confidence score for a single intent against the text.

        Keyword score: each matched keyword contributes `weight` points,
        capped at 0.6 to prevent keyword stuffing from dominating.

        Pattern score: each matched regex pattern contributes 0.25 points,
        capped at 0.4 (patterns are stronger signals than single keywords).

        Final confidence = min(1.0, keyword_score + pattern_score).
        """
        weight: float = data["weight"]
        keywords: List[str] = data["keywords"]

        # --- Keyword matching ---
        matched_kws: List[str] = [kw for kw in keywords if kw in text]
        keyword_score = min(0.6, len(matched_kws) * weight)

        # --- Pattern matching ---
        matched_pats: List[str] = []
        for pattern, raw_pat in self._compiled_patterns[intent]:
            if pattern.search(text):
                matched_pats.append(raw_pat)
        pattern_score = min(0.4, len(matched_pats) * 0.25)

        confidence = min(1.0, keyword_score + pattern_score)

        return IntentScore(
            intent=intent,
            confidence=round(confidence, 4),
            matched_keywords=matched_kws,
            matched_patterns=matched_pats,
        )
