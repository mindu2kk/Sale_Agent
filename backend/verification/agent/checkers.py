"""
Specialized Verification Checkers

Binary checkers cho Price/Policy/Relevance verification:
- PriceAccuracyChecker: Cross-check prices với Internal DB
- PolicyAuthenticityChecker: Verify policies against official documents  
- TopicRelevanceChecker: Semantic analysis cho objection coverage
"""

import re
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from ..models import PriceIssue, PolicyIssue, RelevanceIssue, IssueSeverity
from ..config import VerificationConfig
from ..config.thresholds_config import (
    PriceAccuracyThresholds,
    PolicyAuthenticityThresholds,
    TopicRelevanceThresholds,
    VerificationThresholdsConfig,
)
from ..utils.product_matcher import ProductMatcher
from ..utils.price_extractor import PriceExtractor, ExtractedPrice
from ..utils.early_termination import EarlyTerminationManager
from ..utils.semantic_similarity import SemanticSimilarityAnalyzer
from ..utils.intent_classifier import IntentClassifier
from ..utils.cache import PolicyDocumentCache, get_policy_document_cache, ProductPriceLookupCache, get_product_price_cache


class PriceAccuracyChecker:
    """Binary price accuracy verification với DB cross-checking"""

    # Default path to the product catalog CSV
    DEFAULT_CATALOG_PATH = "data/product_catalog_clean.csv"

    def __init__(self, llm, rag_pipeline, config: VerificationConfig,
                 catalog_path: str = DEFAULT_CATALOG_PATH,
                 thresholds: Optional[PriceAccuracyThresholds] = None,
                 thresholds_config: Optional[VerificationThresholdsConfig] = None,
                 price_cache: Optional[ProductPriceLookupCache] = None):
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.config = config
        self._thresholds = thresholds or PriceAccuracyThresholds(
            pass_tolerance_percent=config.price_tolerance_percent,
            critical_threshold_percent=config.price_critical_threshold,
        )
        # Early termination manager — uses thresholds_config if provided
        self._early_termination = EarlyTerminationManager(thresholds_config)
        self._price_extractor = PriceExtractor()
        try:
            self._product_matcher = ProductMatcher(catalog_path)
        except FileNotFoundError:
            self._product_matcher = None
        # LRU cache for product price lookups (task 5.1.2)
        self._price_cache: ProductPriceLookupCache = price_cache or get_product_price_cache()

    def check_price_accuracy(self, draft: str, objection: str) -> Tuple[bool, List[PriceIssue]]:
        """
        Binary price accuracy check: PASS/FAIL với structured PriceIssue.

        Algorithm:
        1. Use PriceExtractor to extract all price mentions from draft.
        2. If no prices found and objection mentions price → FAIL (MAJOR).
        3. For each extracted price, use ProductMatcher to find the product in DB.
        4. Compare mentioned price vs DB price using PriceAccuracyThresholds.
        5. After each price verification, check early termination via EarlyTerminationManager.
        6. If a CRITICAL issue is found and early termination is enabled, stop immediately.
        7. Worst-case scoring: if ANY price fails, overall check fails.

        Returns:
            (is_pass, issues_list)
        """
        extracted_prices: List[ExtractedPrice] = self._price_extractor.extract(draft)

        if not extracted_prices:
            # No prices found — check if objection mentions price
            if self._objection_mentions_price(objection):
                issue = PriceIssue(
                    product_name="Unknown",
                    severity=IssueSeverity.MAJOR,
                    explanation="Objection mentions price but draft contains no pricing information",
                    correction_suggestion="Add accurate pricing information from the product catalog",
                )
                return False, [issue]
            return True, []

        issues: List[PriceIssue] = []
        overall_pass = True

        for extracted in extracted_prices:
            is_accurate, issue = self._verify_extracted_price(extracted)
            if not is_accurate and issue is not None:
                overall_pass = False
                issues.append(issue)

                # Check early termination after each failed price verification
                termination = self._early_termination.should_terminate(issues)
                if termination.should_terminate:
                    # Stop processing remaining prices — return issues found so far
                    break

        return overall_pass, issues

    def _verify_extracted_price(
        self, extracted: ExtractedPrice
    ) -> Tuple[bool, Optional[PriceIssue]]:
        """
        Verify a single ExtractedPrice against the product catalog.

        Uses ProductMatcher with the product_context from the extracted price
        to find the matching product, then compares prices.

        Strategy: try multiple sub-queries from the context to maximize the
        chance of finding the right product (the full context may be too long
        and dilute the fuzzy score).
        """
        if self._product_matcher is None:
            # No catalog available — cannot verify, treat as pass to avoid false positives
            return True, None

        context = extracted.product_context.strip() if extracted.product_context else ""
        if not context:
            # No context to identify product — skip verification
            return True, None

        match = self._find_product_from_context(context, extracted.amount_vnd)
        if match is None:
            # Product not found in catalog — cannot verify
            return False, PriceIssue(
                product_name="Unknown Product",
                mentioned_price=extracted.original_text,
                severity=IssueSeverity.MAJOR,
                explanation="Cannot verify price — product not found in catalog",
                correction_suggestion="Ensure product name is mentioned clearly near the price",
            )

        # Compare mentioned price (normalized to VND) vs catalog price (VND)
        mentioned_vnd = extracted.amount_vnd
        actual_vnd = match.price_vnd

        if actual_vnd == 0:
            deviation = 100.0
        else:
            deviation = abs(mentioned_vnd - actual_vnd) / actual_vnd * 100.0

        # Binary decision: PASS if within tolerance
        if self._thresholds.should_pass_price_check(deviation):
            return True, None

        # Classify severity using thresholds
        severity = self._thresholds.classify_price_deviation(deviation)

        issue = PriceIssue(
            product_name=match.display_name,
            product_sku=match.sku,
            mentioned_price=extracted.original_text,
            actual_price=match.price_raw,
            deviation_percent=round(deviation, 2),
            currency=extracted.currency,
            severity=severity,
            explanation=(
                f"Price deviation {deviation:.1f}% exceeds tolerance "
                f"{self._thresholds.pass_tolerance_percent}% for {match.display_name}"
            ),
            correction_suggestion=(
                f"Update price to {match.price_raw} (SKU: {match.sku})"
            ),
        )
        return False, issue

    def _find_product_from_context(self, context: str, amount_vnd: float = 0.0):
        """
        Try to find a product from a (potentially long) context string.

        Strategy:
        1. Check the price cache first (keyed by context + amount_vnd) — return immediately on hit.
        2. Try the full context directly, passing amount_vnd as a hint.
        3. If that fails, try progressively shorter sub-strings (first N words).
        4. Use a lower threshold (0.4) for sub-string queries to be more lenient.
        5. Store the result (including None for "not found") in the cache.
        """
        # Use (context, amount_vnd) as cache key to distinguish multiple prices
        # that share the same context string (e.g. when PriceExtractor uses full draft)
        cache_key = f"{context}|{amount_vnd:.0f}" if amount_vnd else context

        # Check cache first
        cached = self._price_cache.get(cache_key)
        if cached is not None:
            found, product_match = cached
            return product_match  # None if negative cache, ProductMatch if found

        # Try full context first
        match = self._product_matcher.find_product(context)
        if match is not None:
            self._price_cache.put(cache_key, match)
            return match

        # Try shorter sub-strings: first 2, 3, 4, 5 words
        words = context.split()
        for n in [2, 3, 4, 5]:
            if n >= len(words):
                break
            sub_query = " ".join(words[:n])
            match = self._product_matcher.find_product(sub_query)
            if match is not None:
                self._price_cache.put(cache_key, match)
                return match

        # Try with a lower threshold using find_all
        original_threshold = self._product_matcher.threshold
        try:
            self._product_matcher.threshold = 0.4
            results = self._product_matcher.find_all(context, top_k=1)
            if results:
                self._price_cache.put(cache_key, results[0])
                return results[0]
        finally:
            self._product_matcher.threshold = original_threshold

        # Not found — store negative cache entry
        self._price_cache.put(cache_key, None)
        return None

    def _objection_mentions_price(self, objection: str) -> bool:
        """Check if objection mentions price-related terms"""
        price_keywords = [
            'giá', 'price', 'cost', 'đắt', 'expensive', 'rẻ', 'cheap',
            'tiền', 'money', 'budget', 'ngân sách', 'chi phí'
        ]
        return any(keyword in objection.lower() for keyword in price_keywords)


class PolicyAuthenticityChecker:
    """Binary policy authenticity verification"""
    
    def __init__(self, llm, rag_pipeline, config: VerificationConfig,
                 thresholds: Optional[PolicyAuthenticityThresholds] = None,
                 thresholds_config: Optional[VerificationThresholdsConfig] = None,
                 policy_cache: Optional[PolicyDocumentCache] = None):
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.config = config
        self._thresholds = thresholds or PolicyAuthenticityThresholds()
        self._early_termination = EarlyTerminationManager(thresholds_config)
        # LRU cache for policy document lookups (task 5.1.1)
        self._policy_cache: PolicyDocumentCache = policy_cache or get_policy_document_cache()
    
    def check_policy_authenticity(self, draft: str) -> Tuple[bool, List[PolicyIssue]]:
        """
        Binary policy authenticity check: PASS/FAIL với structured issues.

        Severity classification:
        - CRITICAL: Policy is completely fabricated (not found in DB at all)
        - MAJOR: Policy exists in DB but stated terms are significantly wrong
        - MINOR: Policy exists and is mostly correct but missing details

        After each failed verification, checks early termination via
        EarlyTerminationManager. CRITICAL issues (fabricated policies) trigger
        early termination when configured.
        """
        # Extract policy statements
        policy_statements = self._extract_policy_statements(draft)
        
        if not policy_statements:
            return True, []  # No policies mentioned, no issue
        
        issues = []
        overall_pass = True
        
        for statement in policy_statements:
            is_authentic, issue = self._verify_policy_statement(statement)
            if not is_authentic and issue is not None:
                overall_pass = False
                issues.append(issue)

                # Check early termination after each failed policy verification
                termination = self._early_termination.should_terminate(issues)
                if termination.should_terminate:
                    break
        
        return overall_pass, issues
    
    # ---------------------------------------------------------------------------
    # Policy keyword taxonomy (Requirement 5.1)
    # Each entry: policy_type → {keywords, duration_patterns, claim_patterns}
    # ---------------------------------------------------------------------------
    _POLICY_TAXONOMY: Dict[str, Dict[str, Any]] = {
        'warranty': {
            'keywords': [
                # Vietnamese
                'bảo hành', 'bảo đảm', 'bảo trì', 'chứng nhận bảo hành',
                # English
                'warranty', 'warrantee', 'guarantee', 'guaranteed',
            ],
            # Patterns that extract a concrete duration claim, e.g. "bảo hành 12 tháng"
            'duration_patterns': [
                r'(?:bảo hành|warranty|guarantee)\s+(\d+)\s*(năm|tháng|ngày|year|month|day)s?',
                r'(\d+)\s*(năm|tháng|ngày|year|month|day)s?\s+(?:bảo hành|warranty|guarantee)',
            ],
            # Patterns that signal a specific policy claim (free service, coverage scope, etc.)
            'claim_patterns': [
                r'(?:bảo hành|warranty)\s+(?:toàn|full|complete|trọn)',
                r'(?:miễn phí|free)\s+(?:bảo hành|warranty|sửa chữa|repair)',
                r'(?:bảo hành|warranty)\s+(?:chính hãng|official|authorized)',
            ],
        },
        'return': {
            'keywords': [
                # Vietnamese
                'đổi trả', 'hoàn tiền', 'trả hàng', 'đổi hàng', 'hoàn lại',
                'chính sách đổi', 'chính sách trả',
                # English
                'return', 'refund', 'exchange', 'money back', 'money-back',
            ],
            'duration_patterns': [
                r'(?:đổi trả|hoàn tiền|return|refund)\s+(?:trong|within)\s+(\d+)\s*(ngày|tháng|day|month)s?',
                r'(\d+)\s*(ngày|tháng|day|month)s?\s+(?:đổi trả|hoàn tiền|return|refund)',
            ],
            'claim_patterns': [
                r'(?:hoàn tiền|refund)\s+(?:100%|toàn bộ|full)',
                r'(?:đổi trả|return)\s+(?:miễn phí|free|không mất phí)',
                r'(?:không|no)\s+(?:câu hỏi|question)',
            ],
        },
        'exchange': {
            'keywords': [
                # Vietnamese
                'đổi máy', 'thay máy', 'đổi mới', 'thay thế sản phẩm',
                # English
                'product exchange', 'device exchange', 'swap', 'replacement unit',
            ],
            'duration_patterns': [
                r'(?:đổi máy|thay máy|exchange)\s+(?:trong|within)\s+(\d+)\s*(ngày|tháng|day|month)s?',
            ],
            'claim_patterns': [
                r'(?:đổi|thay)\s+(?:máy mới|new device|new unit)',
                r'(?:1:1|one.to.one)\s+(?:exchange|replacement)',
            ],
        },
        'service': {
            'keywords': [
                # Vietnamese
                'sửa chữa', 'thay thế linh kiện', 'dịch vụ kỹ thuật',
                'hỗ trợ kỹ thuật', 'trung tâm bảo hành',
                # English
                'repair', 'service center', 'technical support', 'maintenance',
                'authorized service', 'after-sales service',
            ],
            'duration_patterns': [
                r'(?:sửa chữa|repair)\s+(?:trong|within)\s+(\d+)\s*(ngày|giờ|day|hour)s?',
            ],
            'claim_patterns': [
                r'(?:sửa chữa|repair)\s+(?:miễn phí|free)',
                r'(?:thay thế|replace)\s+(?:linh kiện|parts?)\s+(?:miễn phí|free)',
                r'(?:on.?site|tại nhà)\s+(?:service|repair|sửa)',
            ],
        },
        'support': {
            'keywords': [
                # Vietnamese
                'hỗ trợ', 'chăm sóc khách hàng', 'tư vấn', 'hotline',
                # English
                'support', 'customer care', 'helpdesk', 'help desk', '24/7',
            ],
            'duration_patterns': [],
            'claim_patterns': [
                r'(?:hỗ trợ|support)\s+(?:24/7|24 giờ|24 hours?)',
                r'(?:miễn phí|free)\s+(?:hỗ trợ|support)',
            ],
        },
    }

    # Sentence boundary splitter — handles Vietnamese and English punctuation
    _SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?;।\n])\s+')

    def _extract_policy_statements(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract policy-related statements from text using keyword patterns and NLP.

        Algorithm:
        1. Split text into sentences using punctuation boundaries.
        2. For each sentence, scan against the full keyword taxonomy.
        3. When a keyword matches, extract:
           - The full sentence as context (better than fixed char windows).
           - The policy type and triggering keyword.
           - Any duration claim (e.g. "12 tháng") via duration_patterns.
           - Any specific claim (e.g. "miễn phí") via claim_patterns.
           - A confidence score based on keyword specificity + claim presence.
        4. Deduplicate: if two matches share the same sentence span, keep the
           one with the higher confidence score.

        Returns:
            List of statement dicts with keys:
              text, type, keyword, position, duration, claims, confidence
        """
        sentences = self._split_into_sentences(text)
        raw_matches: List[Dict[str, Any]] = []

        for policy_type, taxonomy in self._POLICY_TAXONOMY.items():
            for keyword in taxonomy['keywords']:
                kw_pattern = re.compile(re.escape(keyword), re.IGNORECASE)

                for sentence, (sent_start, sent_end) in sentences:
                    kw_match = kw_pattern.search(sentence)
                    if kw_match is None:
                        continue

                    # Absolute position in original text
                    abs_start = sent_start
                    abs_end = sent_end

                    # Extract duration claim from the sentence
                    duration = self._extract_duration(sentence, taxonomy['duration_patterns'])

                    # Extract specific claims from the sentence
                    claims = self._extract_claims(sentence, taxonomy['claim_patterns'])

                    # Confidence: base 0.5, +0.2 for multi-word keyword, +0.15 per claim
                    confidence = 0.5
                    if ' ' in keyword:
                        confidence += 0.2
                    confidence += min(0.3, len(claims) * 0.15)
                    if duration:
                        confidence += 0.1
                    confidence = round(min(1.0, confidence), 3)

                    raw_matches.append({
                        'text': sentence.strip(),
                        'type': policy_type,
                        'keyword': keyword,
                        'position': (abs_start, abs_end),
                        'duration': duration,
                        'claims': claims,
                        'confidence': confidence,
                    })

        return self._deduplicate_statements(raw_matches)

    def _split_into_sentences(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Split text into (sentence, (start, end)) tuples using punctuation boundaries.

        Falls back to the whole text as a single sentence if no boundaries found.
        """
        sentences: List[Tuple[str, Tuple[int, int]]] = []
        current_pos = 0

        for part in self._SENTENCE_SPLIT_RE.split(text):
            start = current_pos
            end = start + len(part)
            stripped = part.strip()
            if stripped:
                sentences.append((stripped, (start, end)))
            current_pos = end + 1  # +1 for the whitespace consumed by split

        if not sentences:
            sentences = [(text.strip(), (0, len(text)))]

        return sentences

    def _extract_duration(self, sentence: str, patterns: List[str]) -> Optional[Dict[str, str]]:
        """
        Extract a duration claim (amount + unit) from a sentence.

        Returns dict with 'amount' and 'unit', or None if not found.
        """
        for pattern in patterns:
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m and len(m.groups()) >= 2:
                return {'amount': m.group(1), 'unit': m.group(2), 'full_match': m.group(0)}
        return None

    def _extract_claims(self, sentence: str, patterns: List[str]) -> List[str]:
        """
        Extract specific policy claims from a sentence using claim_patterns.

        Returns list of matched claim strings.
        """
        claims: List[str] = []
        for pattern in patterns:
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                claims.append(m.group(0).strip())
        return claims

    def _deduplicate_statements(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate matches that refer to the same sentence span.

        When two matches share the same (start, end) position, keep the one
        with the higher confidence score. This prevents the same sentence from
        being reported multiple times for different keywords of the same type.
        """
        # Group by position
        best: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for match in matches:
            pos = match['position']
            if pos not in best or match['confidence'] > best[pos]['confidence']:
                best[pos] = match

        # Return sorted by position in original text
        return sorted(best.values(), key=lambda m: m['position'][0])
    
    def _verify_policy_statement(self, statement: Dict[str, Any]) -> Tuple[bool, Optional[PolicyIssue]]:
        """
        Verify single policy statement against official documents.

        Severity classification logic:
        - CRITICAL: Policy is completely fabricated — not found in DB at all
          (correct_policy is None after DB lookup, or forbidden phrase detected)
        - MAJOR: Policy exists in DB but stated terms are significantly wrong
          (correct_policy is returned from DB, meaning a real policy was found
          but the claim doesn't match it)
        - MINOR: Policy exists and is mostly correct but missing details
          (is_incomplete=True, no fabrication or major inaccuracy)

        Uses PolicyAuthenticityThresholds.classify_policy_issue() for consistent
        severity classification aligned with thresholds configuration.
        """
        policy_type = statement.get('type', 'service')

        # Check for forbidden phrases (fabricated policies) — CRITICAL
        if self._contains_forbidden_phrases(statement['text']):
            severity = self._thresholds.classify_policy_issue(
                is_fabricated=True,
                is_inaccurate=False,
                is_incomplete=False,
                policy_type=policy_type,
                has_citation=False,
            )
            return False, PolicyIssue(
                mentioned_policy=statement['text'],
                policy_type=policy_type,
                is_fabricated=True,
                is_inaccurate=False,
                severity=severity,
                explanation="Policy contains forbidden phrases indicating fabrication",
                correction_suggestion="Remove fabricated policy claims and use only verified official policies",
            )

        # Verify against official policy documents in DB
        is_verified, correct_policy = self._lookup_policy_in_db(statement)

        if is_verified:
            return True, None

        # Determine fabrication vs inaccuracy based on whether DB returned a reference
        # correct_policy is None  → nothing found in DB → completely fabricated (CRITICAL)
        # correct_policy is str   → found in DB but claim doesn't match → inaccurate (MAJOR)
        is_fabricated = correct_policy is None
        is_inaccurate = correct_policy is not None

        # Check for incomplete policy (exists in DB, claim partially matches)
        # We treat "inaccurate" as MAJOR and "fabricated" as CRITICAL per design spec.
        # is_incomplete is a softer variant — not triggered here since _lookup_policy_in_db
        # returns binary verified/not-verified. Future enhancement can add partial matching.
        is_incomplete = False

        severity = self._thresholds.classify_policy_issue(
            is_fabricated=is_fabricated,
            is_inaccurate=is_inaccurate,
            is_incomplete=is_incomplete,
            policy_type=policy_type,
            has_citation=correct_policy is not None,
        )

        if is_fabricated:
            explanation = "Policy statement not found in official documents — likely fabricated"
            correction = "Remove this policy claim or replace with verified policy from official documents"
        else:
            explanation = "Policy statement inaccurate compared to official documents"
            correction = (
                f"Update policy to match official document: {correct_policy[:200]}..."
                if correct_policy and len(correct_policy) > 200
                else f"Update policy to match official document: {correct_policy}"
            )

        return False, PolicyIssue(
            mentioned_policy=statement['text'],
            policy_type=policy_type,
            is_fabricated=is_fabricated,
            is_inaccurate=is_inaccurate,
            is_incomplete=is_incomplete,
            correct_policy=correct_policy,
            severity=severity,
            explanation=explanation,
            correction_suggestion=correction,
        )
    
    def _contains_forbidden_phrases(self, text: str) -> bool:
        """Check if text contains forbidden phrases"""
        return any(phrase in text.lower() for phrase in self.config.policy_forbidden_phrases)
    
    # ---------------------------------------------------------------------------
    # Policy document lookup constants
    # ---------------------------------------------------------------------------

    # Minimum similarity ratio (0–1) for a claim to be considered "matched"
    # in a retrieved document chunk.
    _CLAIM_MATCH_THRESHOLD = 0.55

    # Number of document chunks to retrieve per policy query.
    _RETRIEVAL_TOP_K = 5

    # Policy-type → search query templates (used to build targeted RAG queries)
    _POLICY_QUERY_TEMPLATES: Dict[str, str] = {
        'warranty':  'chính sách bảo hành warranty policy terms',
        'return':    'chính sách đổi trả hoàn tiền return refund policy',
        'exchange':  'chính sách đổi máy thay thế sản phẩm exchange replacement policy',
        'service':   'dịch vụ sửa chữa bảo trì service repair policy',
        'support':   'hỗ trợ khách hàng customer support policy',
    }

    def _lookup_policy_in_db(self, statement: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Verify a policy statement against official documents via RAG retrieval.

        Algorithm:
        1. Build a targeted search query from the statement's type, keyword,
           duration, and claims.
        2. Retrieve the top-K most relevant policy document chunks via
           ``self.rag_pipeline.retriever.retrieve()``.
        3. Filter chunks to those tagged as policy documents
           (``source_type == "policy_pdf"`` or ``file_name`` contains a policy
           keyword).  Fall back to all chunks if none are tagged.
        4. For each claim in the statement, check whether it appears (exactly
           or approximately) in any retrieved chunk.
        5. If a duration claim is present, verify it against the retrieved text.
        6. Return:
           - ``(True, None)``  — all claims verified against official documents.
           - ``(False, correct_policy)`` — claim not found; ``correct_policy``
             is the most relevant retrieved chunk text (or ``None`` if nothing
             was retrieved at all, indicating a fabricated policy).

        Falls back to ``(True, None)`` when the RAG pipeline is unavailable
        (e.g. during unit tests without a live retriever).
        """
        # Graceful fallback when no RAG pipeline is configured
        if self.rag_pipeline is None:
            return True, None
        # Read the concrete instance attribute first. Using plain getattr here
        # is unsafe with MagicMock because it auto-creates a fake attribute and
        # can accidentally shadow the configured retriever.
        policy_retriever = vars(self.rag_pipeline).get('policy_retriever')
        retriever = policy_retriever or getattr(
            self.rag_pipeline, 'retriever', None
        )
        if retriever is None:
            return True, None

        # ------------------------------------------------------------------
        # Step 1: Build targeted search query
        # ------------------------------------------------------------------
        policy_type: str = statement.get('type', 'warranty')
        keyword: str = statement.get('keyword', '')
        duration: Optional[Dict[str, str]] = statement.get('duration')
        claims: List[str] = statement.get('claims', [])
        statement_text: str = statement.get('text', '')

        base_query = self._POLICY_QUERY_TEMPLATES.get(policy_type, keyword)

        # Enrich query with duration and claim context for better retrieval
        query_parts = [base_query]
        if duration:
            query_parts.append(duration.get('full_match', ''))
        if claims:
            query_parts.append(' '.join(claims[:2]))  # top 2 claims only
        if keyword and keyword not in base_query:
            query_parts.append(keyword)

        search_query = ' '.join(filter(None, query_parts))

        # ------------------------------------------------------------------
        # Cache lookup — avoid redundant RAG retrieval for the same query
        # ------------------------------------------------------------------
        cached = self._policy_cache.get(search_query)
        if cached is not None:
            return cached

        # ------------------------------------------------------------------
        # Step 2: Retrieve relevant document chunks
        # ------------------------------------------------------------------
        try:
            retrieved_nodes = retriever.retrieve(
                search_query, top_k=self._RETRIEVAL_TOP_K
            )
        except Exception:
            # Retrieval failure → cannot verify → treat as unverified
            return False, None

        if not retrieved_nodes:
            # Nothing retrieved → policy not found in DB → fabricated
            return False, None

        # ------------------------------------------------------------------
        # Step 3: Filter to policy document chunks
        # ------------------------------------------------------------------
        policy_nodes = [
            nws for nws in retrieved_nodes
            if self._is_policy_node(nws)
        ]
        # Fall back to all retrieved nodes if none are tagged as policy docs
        candidate_nodes = policy_nodes if policy_nodes else retrieved_nodes

        # ------------------------------------------------------------------
        # Step 4 & 5: Match claims and duration against retrieved text
        # ------------------------------------------------------------------
        best_chunk_text: Optional[str] = candidate_nodes[0].node.text if candidate_nodes else None

        # If no specific claims or duration, presence of relevant chunks is
        # sufficient to consider the policy statement verified.
        if not claims and not duration:
            result = (True, None)
            self._policy_cache.put(search_query, result)
            return result

        # Check each claim against the retrieved chunks
        unverified_claims: List[str] = []
        for claim in claims:
            if not self._claim_found_in_nodes(claim, candidate_nodes):
                unverified_claims.append(claim)

        # Check duration claim if present
        duration_verified = True
        if duration:
            duration_str = duration.get('full_match', '')
            amount = duration.get('amount', '')
            unit = duration.get('unit', '')
            numeric_check = f"{amount} {unit}".strip()
            # Duration must match exactly (numbers must be precise — no fuzzy matching)
            if duration_str and not self._exact_substring_in_nodes(duration_str, candidate_nodes):
                if numeric_check and not self._exact_substring_in_nodes(numeric_check, candidate_nodes):
                    duration_verified = False

        if unverified_claims or not duration_verified:
            # Return the best matching chunk as the "correct policy" reference
            result = (False, best_chunk_text)
            self._policy_cache.put(search_query, result)
            return result

        result = (True, None)
        self._policy_cache.put(search_query, result)
        return result

    # ---------------------------------------------------------------------------
    # Policy lookup helpers
    # ---------------------------------------------------------------------------

    _POLICY_FILE_KEYWORDS = ('policy', 'warranty', 'return', 'exchange', 'bảo hành', 'đổi trả')

    def _is_policy_node(self, nws: Any) -> bool:
        """Return True if a NodeWithScore comes from a policy document."""
        metadata = getattr(nws.node, 'metadata', {}) or {}
        source_type = metadata.get('source_type', '')
        file_name = metadata.get('file_name', '').lower()
        if source_type == 'policy_pdf':
            return True
        return any(kw in file_name for kw in self._POLICY_FILE_KEYWORDS)

    def _claim_found_in_nodes(self, claim: str, nodes: List[Any]) -> bool:
        """
        Check whether *claim* appears in any of the retrieved node texts.

        Uses a two-pass strategy:
        1. Case-insensitive substring match (exact).
        2. Token-overlap ratio ≥ ``_CLAIM_MATCH_THRESHOLD`` (fuzzy).
        """
        from difflib import SequenceMatcher

        claim_lower = claim.lower().strip()
        if not claim_lower:
            return True  # Empty claim — nothing to verify

        for nws in nodes:
            node_text = (getattr(nws.node, 'text', '') or '').lower()

            # Pass 1: exact substring
            if claim_lower in node_text:
                return True

            # Pass 2: token-overlap ratio
            claim_tokens = set(claim_lower.split())
            node_tokens = set(node_text.split())
            if not claim_tokens:
                continue
            overlap = len(claim_tokens & node_tokens) / len(claim_tokens)
            if overlap >= self._CLAIM_MATCH_THRESHOLD:
                return True

        return False

    def _exact_substring_in_nodes(self, text: str, nodes: List[Any]) -> bool:
        """
        Check whether *text* appears as a case-insensitive exact substring
        in any of the retrieved node texts.  No fuzzy matching — used for
        numeric duration checks where token overlap would be misleading.
        """
        text_lower = text.lower().strip()
        if not text_lower:
            return True
        for nws in nodes:
            node_text = (getattr(nws.node, 'text', '') or '').lower()
            if text_lower in node_text:
                return True
        return False


class TopicRelevanceChecker:
    """
    Binary topic relevance verification using SemanticSimilarityAnalyzer.

    Implements check(draft, objection) -> (bool, List[RelevanceIssue]).
    Returns PASS (True) if coverage_ratio >= min_coverage_ratio (default 0.7).
    """

    def __init__(
        self,
        llm,
        config: VerificationConfig,
        thresholds: Optional[TopicRelevanceThresholds] = None,
        thresholds_config: Optional[VerificationThresholdsConfig] = None,
    ):
        self.llm = llm
        self.config = config
        self._thresholds = thresholds or TopicRelevanceThresholds(
            pass_coverage_threshold=config.relevance_min_coverage,
            empathy_required=config.relevance_empathy_bonus,
        )
        self._analyzer = SemanticSimilarityAnalyzer(
            min_coverage_ratio=self._thresholds.pass_coverage_threshold,
            empathy_bonus_enabled=self._thresholds.empathy_required,
        )
        self._intent_classifier = IntentClassifier()

    # ------------------------------------------------------------------
    # Primary public interface (matches design spec signature)
    # ------------------------------------------------------------------

    def check(self, draft: str, objection: str) -> Tuple[bool, List[RelevanceIssue]]:
        """
        Binary topic relevance check: PASS/FAIL with structured RelevanceIssue list.

        Returns:
            (True, [])  — coverage_ratio >= min_coverage_ratio → PASS
            (False, [RelevanceIssue])  — coverage below threshold → FAIL
        """
        return self.check_topic_relevance(draft, objection)

    def check_topic_relevance(self, draft: str, objection: str) -> Tuple[bool, List[RelevanceIssue]]:
        """
        Binary topic relevance check using SemanticSimilarityAnalyzer + IntentClassifier.

        Algorithm:
        1. Run IntentClassifier.classify(objection) for structured intent detection.
        2. Run SemanticSimilarityAnalyzer.analyze(objection, draft) for coverage.
        3. If coverage_ratio >= pass_coverage_threshold → PASS (True, []).
        4. Otherwise run off-topic detection and build structured RelevanceIssue.
        5. Return (False, [issue]).
        """
        # An exact shared SKU is authoritative evidence that the response is
        # addressing the same catalog item. This is stronger than lexical
        # overlap, especially for concise price/spec answers.
        objection_skus = set(re.findall(r"\b[A-Z0-9]{8}\b", objection.upper()))
        draft_skus = set(re.findall(r"\b[A-Z0-9]{8}\b", draft.upper()))
        if objection_skus & draft_skus:
            return True, []
        if self._is_price_budget_answer(objection, draft):
            return True, []

        # Structured intent classification
        classification = self._intent_classifier.classify(objection)

        result = self._analyzer.analyze(objection, draft)

        coverage_ratio = result.coverage_ratio
        min_coverage = self._thresholds.pass_coverage_threshold

        # Binary PASS decision
        if coverage_ratio >= min_coverage:
            return True, []

        # Off-topic detection with structured feedback generation
        issue = self._build_relevance_issue(
            objection=objection,
            draft=draft,
            classification=classification,
            similarity_result=result,
            coverage_ratio=coverage_ratio,
            min_coverage=min_coverage,
        )

        return False, [issue]

    def _is_price_budget_answer(self, objection: str, draft: str) -> bool:
        objection_lower = objection.lower()
        draft_lower = draft.lower()
        mentions_price_intent = any(
            token in objection_lower
            for token in (
                "giá",
                "gia",
                "triệu",
                "trieu",
                "vnđ",
                "vnd",
                "ngân sách",
                "ngan sach",
                "budget",
                "chi phí",
                "chi phi",
            )
        ) or bool(re.search(r"\d+(?:[.,]\d+)*\s*(triệu|tr|vnđ|vnd|đ)?", objection_lower))
        if not mentions_price_intent:
            return False

        draft_has_price = bool(re.search(r"\d+(?:[.,]\d+)*(?:\s*(vnđ|vnd|đ))?", draft_lower))
        draft_has_budget_resolution = any(
            phrase in draft_lower
            for phrase in (
                "không có sản phẩm",
                "khong co san pham",
                "chưa có mẫu",
                "chua co mau",
                "trong tầm",
                "trong tam",
                "ngân sách",
                "ngan sach",
            )
        )
        return draft_has_price and draft_has_budget_resolution

    # ------------------------------------------------------------------
    # Off-topic detection and structured feedback generation
    # ------------------------------------------------------------------

    def _classify_severity_by_coverage(self, coverage_ratio: float) -> "IssueSeverity":
        """
        Classify severity based on coverage ratio thresholds.

        Severity mapping (per task 2.3.4 spec):
          coverage < 0.3  → critical  (completely off-topic)
          0.3 <= coverage < 0.6 → major  (mostly off-topic)
          0.6 <= coverage < pass_threshold → minor  (partially off-topic)
        """
        from ..models.verification import IssueSeverity as ModelSeverity
        if coverage_ratio < 0.3:
            return ModelSeverity.CRITICAL
        if coverage_ratio < 0.6:
            return ModelSeverity.MAJOR
        return ModelSeverity.MINOR

    def _detect_off_topic_content(
        self,
        draft: str,
        detected_intents: List[str],
    ) -> List[str]:
        """
        Identify sentences/phrases in the draft that are off-topic relative to
        the detected objection intents.

        Strategy:
        1. Split draft into sentences.
        2. For each sentence, check if it contains any keyword from the detected intents.
        3. Sentences with no intent keyword overlap are flagged as off-topic.
        4. Return up to 3 representative off-topic snippets (truncated to 80 chars).
        """
        from ..utils.semantic_similarity import INTENT_KEYWORDS

        if not detected_intents:
            return []

        # Collect all relevant keywords for detected intents
        relevant_keywords: List[str] = []
        for intent in detected_intents:
            kw_groups = INTENT_KEYWORDS.get(intent, {})
            relevant_keywords.extend(kw_groups.get("objection", []))
            relevant_keywords.extend(kw_groups.get("response", []))

        if not relevant_keywords:
            return []

        # Split draft into sentences
        sentence_split_re = re.compile(r'(?<=[.!?;।\n])\s+')
        sentences = [s.strip() for s in sentence_split_re.split(draft) if s.strip()]
        if not sentences:
            sentences = [draft.strip()] if draft.strip() else []

        off_topic: List[str] = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            has_relevant_content = any(kw in sentence_lower for kw in relevant_keywords)
            if not has_relevant_content and len(sentence) > 20:
                snippet = sentence[:80] + ("..." if len(sentence) > 80 else "")
                off_topic.append(snippet)

        return off_topic[:3]

    def _build_missing_aspects_from_intents(
        self,
        classification,
        similarity_result,
        objection: str,
    ) -> List[str]:
        """
        Build a comprehensive list of missing aspects by combining:
        1. Missing aspects from SemanticSimilarityAnalyzer (keyword-based).
        2. Unaddressed intents from IntentClassifier with human-readable labels.
        """
        from ..utils.intent_classifier import INTENT_LABELS

        missing: List[str] = list(similarity_result.missing_aspects)

        # Add human-readable labels for intents with low coverage
        intent_coverage = similarity_result.intent_coverage_detail
        for intent_score in classification.intents:
            intent_name = intent_score.intent
            coverage = intent_coverage.get(intent_name, 0.0)
            if coverage < 0.3:
                label = INTENT_LABELS.get(intent_name, intent_name)
                readable = f"{label} not adequately addressed"
                if readable not in missing:
                    missing.append(readable)

        return missing

    def _build_objection_intent_description(self, classification) -> str:
        """
        Build a human-readable description of what the customer was asking about,
        using the IntentClassifier result.
        """
        from ..utils.intent_classifier import INTENT_LABELS

        if not classification.intents:
            return "general_inquiry"

        primary = classification.primary_intent
        label = INTENT_LABELS.get(primary, primary)

        if classification.is_multi_intent and len(classification.intents) >= 2:
            secondary = classification.intents[1].intent
            secondary_label = INTENT_LABELS.get(secondary, secondary)
            return f"{label} and {secondary_label}"

        return label

    def _build_relevance_issue(
        self,
        objection: str,
        draft: str,
        classification,
        similarity_result,
        coverage_ratio: float,
        min_coverage: float,
    ) -> "RelevanceIssue":
        """
        Build a structured RelevanceIssue with off-topic detection and feedback.

        Combines IntentClassifier + SemanticSimilarityAnalyzer results to produce:
        - objection_intent: human-readable description of what customer asked
        - response_coverage: float 0-1 from semantic analysis
        - missing_aspects: specific aspects not addressed
        - off_topic_content: sentences in draft unrelated to the objection
        - severity: critical/major/minor based on coverage ratio
        - correction_suggestion: actionable guidance
        """
        detected_intents = (
            classification.intent_names if classification.intents
            else similarity_result.detected_intents
        )

        # Human-readable intent description
        objection_intent = self._build_objection_intent_description(classification)

        # Identify off-topic content in the draft
        off_topic_content = self._detect_off_topic_content(draft, detected_intents)

        # Build comprehensive missing aspects list
        missing_aspects = self._build_missing_aspects_from_intents(
            classification, similarity_result, objection
        )

        # Severity based on coverage ratio (task 2.3.4 spec)
        severity = self._classify_severity_by_coverage(coverage_ratio)

        # Empathy score
        empathy_score = 1.0 if similarity_result.has_empathy else 0.0

        # Build explanation
        severity_label = {
            "critical": "completely off-topic",
            "major": "mostly off-topic",
            "minor": "partially off-topic",
        }.get(severity.value, "off-topic")

        explanation = (
            f"Response is {severity_label} — coverage {coverage_ratio:.1%} is below "
            f"the minimum {min_coverage:.1%} threshold. "
            f"Customer was asking about: {objection_intent}."
        )
        if off_topic_content:
            explanation += f" Off-topic content detected in response."

        # Build correction suggestion
        correction_parts = []
        if missing_aspects:
            correction_parts.append(
                "Address missing aspects: " + "; ".join(missing_aspects[:3])
            )
        if off_topic_content:
            correction_parts.append(
                "Remove or replace off-topic content unrelated to the objection"
            )
        if not similarity_result.has_empathy:
            correction_parts.append("Add empathy statements to acknowledge the customer's concern")
        if not correction_parts:
            correction_parts.append("Improve overall relevance to the customer's objection")

        correction_suggestion = ". ".join(correction_parts) + "."

        return RelevanceIssue(
            objection_intent=objection_intent,
            detected_intents=detected_intents,
            response_coverage=coverage_ratio,
            missing_aspects=missing_aspects,
            off_topic_content=off_topic_content,
            empathy_score=empathy_score,
            severity=severity,
            explanation=explanation,
            correction_suggestion=correction_suggestion,
        )

    # ------------------------------------------------------------------
    # Convenience helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _has_empathy_statements(self, draft: str) -> bool:
        """Check if response contains empathy statements."""
        from ..utils.semantic_similarity import EMPATHY_PHRASES
        return any(phrase in draft.lower() for phrase in EMPATHY_PHRASES)
