"""Simple OrderedDict-based query cache with TTL and LRU max_size eviction."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class QueryCache:
    """In-memory LRU cache with TTL expiry and max_size eviction via OrderedDict."""

    def __init__(self, ttl: float = 300.0, max_size: int = 100) -> None:
        self._ttl = ttl
        self._max_size = max_size
        # OrderedDict preserves insertion order → O(1) oldest-entry eviction
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        """Return cached value if present and not expired, else None."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts >= self._ttl:
            del self._store[key]
            return None
        logger.debug("Cache hit for query: %s", key)
        return value

    def set(self, key: str, value: str) -> None:
        """Store value; evict the oldest entry (LRU) if at max_size."""
        if key in self._store:
            # Remove so we can re-insert at the end (most-recently-used)
            del self._store[key]
        elif len(self._store) >= self._max_size:
            # Evict the first (oldest) entry
            self._store.popitem(last=False)
        self._store[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all entries."""
        self._store.clear()
