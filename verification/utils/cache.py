"""
Caching Utilities cho Verification Agent

LRU cache implementation với:
- TTL expiration support
- Memory-efficient storage
- Performance metrics tracking
- Thread-safe operations
"""

import time
import threading
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CacheEntry:
    """Cache entry với TTL support"""
    
    value: Any
    created_at: float
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        
        return time.time() - self.created_at > self.ttl_seconds
    
    def access(self) -> Any:
        """Access entry and update stats"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class VerificationCache:
    """
    Thread-safe LRU cache với TTL support
    
    Optimized cho verification results caching
    với automatic cleanup và performance tracking.
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl_seconds: Optional[float] = 3600):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of entries
            default_ttl_seconds: Default TTL for entries
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        
        # Thread-safe storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        
        # Cleanup tracking
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        
        with self._lock:
            # Periodic cleanup
            self._maybe_cleanup()
            
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Update stats and return
            self._hits += 1
            return entry.access()
    
    def put(self, 
            key: str, 
            value: Any, 
            ttl_seconds: Optional[float] = None) -> None:
        """
        Put value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL override (uses default if None)
        """
        
        with self._lock:
            # Use default TTL if not specified
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl_seconds
            )
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict if over capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted, False if not found
        """
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        
        with self._lock:
            self._cache.clear()
            self._reset_stats()
    
    def size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "total_requests": total_requests
            }
    
    def _maybe_cleanup(self) -> None:
        """Perform periodic cleanup of expired entries"""
        
        current_time = time.time()
        
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        # Find expired entries
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self._cache[key]
            self._expirations += 1
        
        self._last_cleanup = current_time
    
    def _reset_stats(self) -> None:
        """Reset performance statistics"""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage (approximate)"""
        
        with self._lock:
            # Rough estimation
            entry_overhead = 200  # bytes per entry (approximate)
            total_entries = len(self._cache)
            estimated_bytes = total_entries * entry_overhead
            
            return {
                "estimated_bytes": estimated_bytes,
                "estimated_mb": estimated_bytes / (1024 * 1024),
                "entries": total_entries,
                "overhead_per_entry": entry_overhead
            }


# Global cache instance
_global_cache: Optional[VerificationCache] = None


def get_global_cache() -> VerificationCache:
    """Get global cache instance (singleton)"""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = VerificationCache()
    
    return _global_cache


def configure_global_cache(max_size: int = 1000, 
                          default_ttl_seconds: Optional[float] = 3600) -> VerificationCache:
    """Configure global cache instance"""
    global _global_cache
    
    _global_cache = VerificationCache(max_size, default_ttl_seconds)
    return _global_cache


class PolicyDocumentCache:
    """
    LRU cache chuyên biệt cho policy document lookups.

    Wraps VerificationCache với:
    - Cache key generation từ policy statement attributes
    - Dedicated TTL cho policy documents (default 1 giờ)
    - Singleton instance per PolicyAuthenticityChecker
    """

    # Default TTL: 1 hour — policy documents change infrequently
    DEFAULT_TTL_SECONDS: float = 3600.0

    # Default max entries: 256 policy queries is more than enough
    DEFAULT_MAX_SIZE: int = 256

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._cache = VerificationCache(
            max_size=max_size,
            default_ttl_seconds=ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, search_query: str) -> Optional[Tuple[bool, Optional[str]]]:
        """
        Retrieve cached lookup result for a policy search query.

        Returns:
            (is_verified, correct_policy) tuple, or None on cache miss.
        """
        return self._cache.get(self._make_key(search_query))

    def put(self, search_query: str, result: Tuple[bool, Optional[str]]) -> None:
        """Store a lookup result keyed by the search query."""
        self._cache.put(self._make_key(search_query), result)

    def invalidate(self, search_query: str) -> bool:
        """Remove a specific entry (e.g. after policy document update)."""
        return self._cache.delete(self._make_key(search_query))

    def clear(self) -> None:
        """Flush all cached policy lookups."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Expose underlying cache statistics."""
        return self._cache.get_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(search_query: str) -> str:
        """Normalise query to a stable cache key."""
        return search_query.strip().lower()


# ---------------------------------------------------------------------------
# Module-level singleton for PolicyDocumentCache
# ---------------------------------------------------------------------------

_policy_document_cache: Optional[PolicyDocumentCache] = None


def get_policy_document_cache() -> PolicyDocumentCache:
    """Return the module-level PolicyDocumentCache singleton."""
    global _policy_document_cache
    if _policy_document_cache is None:
        _policy_document_cache = PolicyDocumentCache()
    return _policy_document_cache


def configure_policy_document_cache(
    max_size: int = PolicyDocumentCache.DEFAULT_MAX_SIZE,
    ttl_seconds: float = PolicyDocumentCache.DEFAULT_TTL_SECONDS,
) -> PolicyDocumentCache:
    """Replace the module-level PolicyDocumentCache singleton."""
    global _policy_document_cache
    _policy_document_cache = PolicyDocumentCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _policy_document_cache


class ProductPriceLookupCache:
    """
    LRU cache chuyên biệt cho product price lookups.

    Wraps VerificationCache với:
    - Cache key generation từ normalized query string
    - Dedicated TTL cho product prices (default 5 phút — prices change more frequently)
    - Stores Tuple[bool, Optional[ProductMatch]] to distinguish:
        (True, ProductMatch)  → product found
        (False, None)         → negative cache (product not found in catalog)
    - Singleton instance via get_product_price_cache()
    """

    # Default TTL: 5 minutes — prices change more frequently than policies
    DEFAULT_TTL_SECONDS: float = 300.0

    # Default max entries
    DEFAULT_MAX_SIZE: int = 512

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._cache = VerificationCache(
            max_size=max_size,
            default_ttl_seconds=ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[Tuple[bool, Any]]:
        """
        Retrieve cached lookup result for a product price query.

        Returns:
            (found, product_match) tuple on cache hit, or None on cache miss.
            - (True, ProductMatch)  → product was found in catalog
            - (False, None)         → negative cache (product not found)
        """
        return self._cache.get(self._make_key(query))

    def put(self, query: str, result: Optional[Any]) -> None:
        """
        Store a lookup result keyed by the query.

        Args:
            query:  The product context string used for lookup.
            result: The ProductMatch if found, or None if not found.
        """
        if result is not None:
            self._cache.put(self._make_key(query), (True, result))
        else:
            self._cache.put(self._make_key(query), (False, None))

    def invalidate(self, query: str) -> bool:
        """Remove a specific entry (e.g. after catalog update)."""
        return self._cache.delete(self._make_key(query))

    def clear(self) -> None:
        """Flush all cached price lookups."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Expose underlying cache statistics."""
        return self._cache.get_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(query: str) -> str:
        """Normalise query to a stable cache key (strip + lowercase)."""
        return query.strip().lower()


# ---------------------------------------------------------------------------
# Module-level singleton for ProductPriceLookupCache
# ---------------------------------------------------------------------------

_product_price_cache: Optional[ProductPriceLookupCache] = None


def get_product_price_cache() -> ProductPriceLookupCache:
    """Return the module-level ProductPriceLookupCache singleton."""
    global _product_price_cache
    if _product_price_cache is None:
        _product_price_cache = ProductPriceLookupCache()
    return _product_price_cache


def configure_product_price_cache(
    max_size: int = ProductPriceLookupCache.DEFAULT_MAX_SIZE,
    ttl_seconds: float = ProductPriceLookupCache.DEFAULT_TTL_SECONDS,
) -> ProductPriceLookupCache:
    """Replace the module-level ProductPriceLookupCache singleton."""
    global _product_price_cache
    _product_price_cache = ProductPriceLookupCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _product_price_cache
