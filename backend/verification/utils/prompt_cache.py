"""
Prompt Template Cache với Version Control

Caches rendered prompt templates keyed by (template_name, version, variables_hash)
to avoid re-rendering on every call. Supports:
- Version control via SHA256 hash of template content
- TTL-based expiration
- Thread-safe concurrent access
- Cache hit/miss statistics
- Auto-invalidation when template content changes
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptCacheEntry:
    """A single cached rendered prompt."""

    rendered: str
    template_version: str       # SHA256 of template content at render time
    created_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self) -> str:
        self.access_count += 1
        self.last_accessed = time.time()
        return self.rendered


# ---------------------------------------------------------------------------
# Core cache class
# ---------------------------------------------------------------------------

class PromptTemplateCache:
    """
    Thread-safe LRU cache for rendered prompt templates with version control.

    Cache key: (template_name, template_version, variables_hash)
    - template_version  = SHA256 of the raw template string
    - variables_hash    = SHA256 of the JSON-serialised variables dict

    When a template's content changes its version hash changes, so all
    previously cached renders for that template are automatically stale
    (they will never be looked up again under the new version key).

    Usage::

        cache = PromptTemplateCache(max_size=256, default_ttl_seconds=3600)

        version = PromptTemplateCache.compute_version("Hello {name}!")
        vars_hash = PromptTemplateCache.compute_variables_hash({"name": "World"})

        rendered = cache.get("greeting", version, vars_hash)
        if rendered is None:
            rendered = "Hello World!"
            cache.put("greeting", version, vars_hash, rendered)
    """

    DEFAULT_MAX_SIZE: int = 512
    DEFAULT_TTL_SECONDS: float = 3600.0  # 1 hour

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        default_ttl_seconds: Optional[float] = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds

        # Ordered dict for LRU eviction
        self._store: OrderedDict[Tuple[str, str, str], PromptCacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        template_name: str,
        template_version: str,
        variables_hash: str,
    ) -> Optional[str]:
        """
        Return the cached rendered string, or None on miss/expiry.

        Args:
            template_name:    Logical name of the template.
            template_version: SHA256 of the raw template content.
            variables_hash:   SHA256 of the variables dict.
        """
        key = (template_name, template_version, variables_hash)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._store[key]
                self._expirations += 1
                self._misses += 1
                return None
            # LRU: move to end
            self._store.move_to_end(key)
            self._hits += 1
            return entry.touch()

    def put(
        self,
        template_name: str,
        template_version: str,
        variables_hash: str,
        rendered: str,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """
        Store a rendered template.

        Args:
            template_name:    Logical name of the template.
            template_version: SHA256 of the raw template content.
            variables_hash:   SHA256 of the variables dict.
            rendered:         The fully rendered prompt string.
            ttl_seconds:      Override TTL; uses default_ttl_seconds if None.
        """
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        key = (template_name, template_version, variables_hash)
        entry = PromptCacheEntry(
            rendered=rendered,
            template_version=template_version,
            ttl_seconds=ttl_seconds,
        )
        with self._lock:
            self._store[key] = entry
            self._store.move_to_end(key)
            # Evict oldest entries when over capacity
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
                self._evictions += 1

    def invalidate_template(self, template_name: str) -> int:
        """
        Remove all cached entries for *template_name* regardless of version.

        Returns the number of entries removed.
        """
        with self._lock:
            keys_to_remove = [k for k in self._store if k[0] == template_name]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def invalidate_version(self, template_name: str, template_version: str) -> int:
        """
        Remove cached entries for a specific (name, version) pair.

        Returns the number of entries removed.
        """
        with self._lock:
            keys_to_remove = [
                k for k in self._store
                if k[0] == template_name and k[1] == template_version
            ]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def clear(self) -> None:
        """Flush all entries and reset statistics."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "total_requests": total,
            }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_version(template_content: str) -> str:
        """Return a SHA256 hex digest of *template_content* (the version tag)."""
        return hashlib.sha256(template_content.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_variables_hash(variables: Dict[str, Any]) -> str:
        """
        Return a stable SHA256 hex digest of *variables*.

        The dict is serialised to JSON with sorted keys so that
        ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` produce the same hash.
        """
        serialised = json.dumps(variables, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_prompt_template_cache: Optional[PromptTemplateCache] = None


def get_prompt_template_cache() -> PromptTemplateCache:
    """Return the module-level PromptTemplateCache singleton."""
    global _prompt_template_cache
    if _prompt_template_cache is None:
        _prompt_template_cache = PromptTemplateCache()
    return _prompt_template_cache


def configure_prompt_template_cache(
    max_size: int = PromptTemplateCache.DEFAULT_MAX_SIZE,
    default_ttl_seconds: Optional[float] = PromptTemplateCache.DEFAULT_TTL_SECONDS,
) -> PromptTemplateCache:
    """Replace the module-level PromptTemplateCache singleton."""
    global _prompt_template_cache
    _prompt_template_cache = PromptTemplateCache(
        max_size=max_size,
        default_ttl_seconds=default_ttl_seconds,
    )
    return _prompt_template_cache
