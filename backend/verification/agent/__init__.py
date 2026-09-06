"""
Verification Agent Core Implementation

Binary verification logic với async optimization:
- VerificationAgent: Main agent class với parallel checking
- Price/Policy/Relevance checkers với structured issue detection
- Async LLM integration với early termination
- Pydantic result models với type safety
"""

from .verification_agent import VerificationAgent
from .checkers import (
    PriceAccuracyChecker,
    PolicyAuthenticityChecker, 
    TopicRelevanceChecker
)

__all__ = [
    "VerificationAgent",
    "PriceAccuracyChecker",
    "PolicyAuthenticityChecker",
    "TopicRelevanceChecker"
]