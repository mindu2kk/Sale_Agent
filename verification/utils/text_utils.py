"""
Text Processing Utilities

Helper functions cho text analysis:
- Price extraction với regex patterns
- Policy statement detection
- Semantic similarity calculation
- Text preprocessing utilities
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher


def extract_prices(text: str) -> List[Dict[str, Any]]:
    """
    Extract price information from text
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of price dictionaries với extracted information
    """
    
    # Vietnamese price patterns
    vnd_patterns = [
        (r'(\d{1,3}(?:[.,]\d{3})*)\s*(?:VND|vnđ|đồng)', 'VND'),
        (r'(\d{1,3}(?:[.,]\d{3})*)\s*triệu', 'VND_MILLION'),
        (r'(\d{1,3}(?:[.,]\d{3})*)\s*nghìn', 'VND_THOUSAND'),
        (r'(\d{1,3}(?:[.,]\d{3})*)\s*tỷ', 'VND_BILLION')
    ]
    
    # USD patterns
    usd_patterns = [
        (r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', 'USD'),
        (r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*USD', 'USD')
    ]
    
    extracted_prices = []
    
    # Process all patterns
    for pattern, currency_type in vnd_patterns + usd_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            price_info = {
                'raw_text': match.group(0),
                'amount': match.group(1),
                'currency_type': currency_type,
                'position': match.span(),
                'normalized_amount': _normalize_amount(match.group(1), currency_type)
            }
            extracted_prices.append(price_info)
    
    return extracted_prices


def extract_policies(text: str) -> List[Dict[str, Any]]:
    """
    Extract policy-related statements from text
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of policy dictionaries với extracted information
    """
    
    policy_patterns = {
        'warranty': {
            'keywords': ['bảo hành', 'warranty', 'guarantee', 'warrantee'],
            'context_patterns': [
                r'bảo hành\s+(\d+)\s*(năm|tháng|ngày)',
                r'warranty\s+(\d+)\s*(year|month|day)s?',
                r'(\d+)\s*(năm|tháng|ngày)\s+bảo hành'
            ]
        },
        'return': {
            'keywords': ['đổi trả', 'hoàn tiền', 'return', 'refund', 'exchange'],
            'context_patterns': [
                r'đổi trả\s+trong\s+(\d+)\s*(ngày|tháng)',
                r'return\s+within\s+(\d+)\s*(day|month)s?',
                r'(\d+)\s*(ngày|tháng)\s+đổi trả'
            ]
        },
        'service': {
            'keywords': ['sửa chữa', 'thay thế', 'repair', 'replace', 'service', 'maintenance'],
            'context_patterns': [
                r'sửa chữa\s+(miễn phí|free)',
                r'thay thế\s+(miễn phí|free)',
                r'free\s+(repair|replacement|service)'
            ]
        }
    }
    
    extracted_policies = []
    
    for policy_type, config in policy_patterns.items():
        # Find keyword matches
        for keyword in config['keywords']:
            pattern = rf'.{{0,50}}{re.escape(keyword)}.{{0,100}}'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                policy_info = {
                    'text': match.group(0).strip(),
                    'type': policy_type,
                    'keyword': keyword,
                    'position': match.span(),
                    'context': _extract_policy_context(match.group(0), config['context_patterns'])
                }
                extracted_policies.append(policy_info)
    
    return extracted_policies


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    
    # Preprocess texts
    clean_text1 = _preprocess_text(text1)
    clean_text2 = _preprocess_text(text2)
    
    # Calculate sequence similarity
    sequence_similarity = SequenceMatcher(None, clean_text1, clean_text2).ratio()
    
    # Calculate word overlap similarity
    words1 = set(clean_text1.split())
    words2 = set(clean_text2.split())
    
    if not words1 and not words2:
        word_similarity = 1.0
    elif not words1 or not words2:
        word_similarity = 0.0
    else:
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        word_similarity = len(intersection) / len(union)
    
    # Combine similarities (weighted average)
    combined_similarity = (sequence_similarity * 0.3) + (word_similarity * 0.7)
    
    return combined_similarity


def extract_product_names(text: str) -> List[str]:
    """
    Extract product names from text
    
    Args:
        text: Input text
        
    Returns:
        List of extracted product names
    """
    
    # Common product patterns
    product_patterns = [
        r'iPhone\s+\d+(?:\s+Pro(?:\s+Max)?)?',
        r'Samsung\s+Galaxy\s+\w+(?:\s+\d+)?',
        r'MacBook\s+(?:Air|Pro)(?:\s+\d+)?',
        r'iPad\s+(?:Air|Pro|Mini)?(?:\s+\d+)?',
        r'AirPods\s+(?:Pro|Max)?(?:\s+\d+)?'
    ]
    
    extracted_products = []
    
    for pattern in product_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            product_name = match.group(0).strip()
            if product_name not in extracted_products:
                extracted_products.append(product_name)
    
    return extracted_products


def detect_objection_intent(objection: str) -> Dict[str, Any]:
    """
    Detect main intent of customer objection
    
    Args:
        objection: Customer objection text
        
    Returns:
        Dictionary với intent classification
    """
    
    intent_patterns = {
        'price_concern': {
            'keywords': ['đắt', 'giá', 'expensive', 'cost', 'budget', 'tiền', 'money'],
            'weight': 1.0
        },
        'feature_question': {
            'keywords': ['tính năng', 'specs', 'performance', 'feature', 'chức năng'],
            'weight': 1.0
        },
        'comparison_request': {
            'keywords': ['so sánh', 'compare', 'khác gì', 'difference', 'vs', 'versus'],
            'weight': 1.2
        },
        'policy_inquiry': {
            'keywords': ['bảo hành', 'đổi trả', 'warranty', 'return', 'policy'],
            'weight': 1.1
        },
        'availability_check': {
            'keywords': ['có sẵn', 'available', 'stock', 'kho', 'còn hàng'],
            'weight': 0.9
        }
    }
    
    objection_lower = objection.lower()
    intent_scores = {}
    
    for intent, config in intent_patterns.items():
        score = 0
        matched_keywords = []
        
        for keyword in config['keywords']:
            if keyword in objection_lower:
                score += config['weight']
                matched_keywords.append(keyword)
        
        if score > 0:
            intent_scores[intent] = {
                'score': score,
                'matched_keywords': matched_keywords
            }
    
    # Determine primary intent
    if intent_scores:
        primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['score'])
        confidence = intent_scores[primary_intent]['score'] / sum(s['score'] for s in intent_scores.values())
    else:
        primary_intent = 'general_inquiry'
        confidence = 0.5
    
    return {
        'primary_intent': primary_intent,
        'confidence': confidence,
        'all_intents': intent_scores,
        'is_complex': len(intent_scores) > 2  # Multiple intents detected
    }


def _normalize_amount(amount_str: str, currency_type: str) -> float:
    """
    Normalize amount string to float value
    
    Args:
        amount_str: Amount string
        currency_type: Currency type identifier
        
    Returns:
        Normalized float value
    """
    
    # Remove formatting
    clean_amount = re.sub(r'[,.](?=\d{3})', '', amount_str)
    clean_amount = clean_amount.replace(',', '.')
    
    try:
        base_amount = float(clean_amount)
    except ValueError:
        return 0.0
    
    # Apply multipliers
    if currency_type == 'VND_MILLION':
        return base_amount * 1_000_000
    elif currency_type == 'VND_THOUSAND':
        return base_amount * 1_000
    elif currency_type == 'VND_BILLION':
        return base_amount * 1_000_000_000
    else:
        return base_amount


def _extract_policy_context(text: str, patterns: List[str]) -> Optional[Dict[str, Any]]:
    """
    Extract structured context from policy text
    
    Args:
        text: Policy text
        patterns: Regex patterns to match
        
    Returns:
        Extracted context or None
    """
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                return {
                    'duration': groups[0],
                    'unit': groups[1],
                    'full_match': match.group(0)
                }
    
    return None


def _preprocess_text(text: str) -> str:
    """
    Preprocess text for similarity calculation
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text