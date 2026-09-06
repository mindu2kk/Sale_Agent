"""
Cache Invalidation Manager cho Verification Agent

Provides selective cache invalidation triggered by configuration updates.
Uses an observer/event pattern so caches are notified when specific config
sections change, without requiring a full system restart.

Supports Requirement 10.5 (configuration changes take effect without restart)
and Requirement 9.4 (caching for policy documents, price lookups, prompts).

Config change types and their affected caches:
- THRESHOLDS  → price cache, policy cache (verification criteria changed)
- POLICY_DATA → policy document cache
- PRODUCT_DATA → product price cache
- PROMPTS     → prompt template cache
- ALL         → all caches
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config change event types
# ---------------------------------------------------------------------------

class ConfigChangeType(str, Enum):
    """Types of configuration changes that can trigger cache invalidation."""
    THRESHOLDS = "thresholds"       # Verification threshold changes
    POLICY_DATA = "policy_data"     # Policy source / document changes
    PRODUCT_DATA = "product_data"   # Product catalog / price data changes
    PROMPTS = "prompts"             # Prompt template changes
    ALL = "all"                     # Full config reload


@dataclass
class ConfigChangeEvent:
    """Represents a configuration change event."""
    change_type: ConfigChangeType
    changed_keys: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cache adapter protocol
# ---------------------------------------------------------------------------

class CacheAdapter:
    """
    Thin adapter wrapping a cache instance with a uniform clear() interface.

    Subclass or instantiate directly with a callable *clear_fn*.
    """

    def __init__(self, name: str, clear_fn: Callable[[], None]) -> None:
        self.name = name
        self._clear_fn = clear_fn

    def clear(self) -> None:
        try:
            self._clear_fn()
            logger.info("Cache invalidated: %s", self.name)
        except Exception as exc:
            logger.error("Failed to invalidate cache '%s': %s", self.name, exc)


# ---------------------------------------------------------------------------
# Invalidation mapping
# ---------------------------------------------------------------------------

# Maps each ConfigChangeType to the set of cache names it should invalidate.
_INVALIDATION_MAP: Dict[ConfigChangeType, Set[str]] = {
    ConfigChangeType.THRESHOLDS: {"policy_cache", "price_cache"},
    ConfigChangeType.POLICY_DATA: {"policy_cache"},
    ConfigChangeType.PRODUCT_DATA: {"price_cache"},
    ConfigChangeType.PROMPTS: {"prompt_cache"},
    ConfigChangeType.ALL: {"policy_cache", "price_cache", "prompt_cache"},
}


# ---------------------------------------------------------------------------
# CacheInvalidationManager
# ---------------------------------------------------------------------------

class CacheInvalidationManager:
    """
    Manages selective cache invalidation in response to configuration updates.

    Usage::

        manager = CacheInvalidationManager()
        manager.register_cache("policy_cache", policy_doc_cache.clear)
        manager.register_cache("price_cache", price_cache.clear)
        manager.register_cache("prompt_cache", prompt_cache.clear)

        # When config changes:
        manager.on_config_changed(ConfigChangeEvent(ConfigChangeType.PRODUCT_DATA))

    The manager is thread-safe and can be used from multiple threads.
    """

    def __init__(self) -> None:
        self._caches: Dict[str, CacheAdapter] = {}
        self._listeners: List[Callable[[ConfigChangeEvent], None]] = []
        self._lock = threading.RLock()
        self._invalidation_count: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_cache(
        self,
        name: str,
        clear_fn: Callable[[], None],
    ) -> None:
        """
        Register a cache with the manager.

        Args:
            name:     Logical cache name (must match keys in _INVALIDATION_MAP).
            clear_fn: Zero-argument callable that clears the cache.
        """
        with self._lock:
            self._caches[name] = CacheAdapter(name, clear_fn)
            logger.debug("Registered cache: %s", name)

    def unregister_cache(self, name: str) -> bool:
        """Remove a registered cache. Returns True if it existed."""
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                logger.debug("Unregistered cache: %s", name)
                return True
            return False

    def add_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> None:
        """
        Add an observer that is called after invalidation for every event.

        Useful for logging, metrics, or triggering cache re-warming.
        """
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> bool:
        """Remove a previously added listener. Returns True if found."""
        with self._lock:
            try:
                self._listeners.remove(listener)
                return True
            except ValueError:
                return False

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def on_config_changed(self, event: ConfigChangeEvent) -> List[str]:
        """
        Handle a configuration change event.

        Determines which caches are affected by *event.change_type* and
        clears them. Notifies all registered listeners after invalidation.

        Args:
            event: The configuration change event.

        Returns:
            List of cache names that were invalidated.
        """
        affected_names = _INVALIDATION_MAP.get(event.change_type, set())
        invalidated: List[str] = []

        with self._lock:
            for name in affected_names:
                adapter = self._caches.get(name)
                if adapter is not None:
                    adapter.clear()
                    invalidated.append(name)
                else:
                    logger.debug(
                        "Cache '%s' not registered; skipping invalidation for %s",
                        name,
                        event.change_type,
                    )

            self._invalidation_count += len(invalidated)
            listeners = list(self._listeners)

        # Notify listeners outside the lock to avoid deadlocks
        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                logger.error("Cache invalidation listener error: %s", exc)

        if invalidated:
            logger.info(
                "Config change '%s' invalidated caches: %s",
                event.change_type,
                invalidated,
            )
        return invalidated

    def invalidate_all(self) -> List[str]:
        """Unconditionally clear all registered caches."""
        return self.on_config_changed(ConfigChangeEvent(ConfigChangeType.ALL))

    def invalidate(self, change_type: ConfigChangeType) -> List[str]:
        """Convenience wrapper: invalidate caches for *change_type*."""
        return self.on_config_changed(ConfigChangeEvent(change_type))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def registered_caches(self) -> List[str]:
        """Return names of all registered caches."""
        with self._lock:
            return list(self._caches.keys())

    def invalidation_count(self) -> int:
        """Total number of individual cache invalidations performed."""
        with self._lock:
            return self._invalidation_count

    def get_affected_caches(self, change_type: ConfigChangeType) -> Set[str]:
        """Return the set of cache names affected by *change_type*."""
        return set(_INVALIDATION_MAP.get(change_type, set()))


# ---------------------------------------------------------------------------
# Integration helpers: build a manager wired to the existing singletons
# ---------------------------------------------------------------------------

def build_default_manager() -> CacheInvalidationManager:
    """
    Create a CacheInvalidationManager pre-wired to the module-level cache
    singletons (policy, price, prompt).

    Import is deferred to avoid circular imports.
    """
    from .cache import get_policy_document_cache, get_product_price_cache
    from .prompt_cache import get_prompt_template_cache

    manager = CacheInvalidationManager()
    manager.register_cache("policy_cache", get_policy_document_cache().clear)
    manager.register_cache("price_cache", get_product_price_cache().clear)
    manager.register_cache("prompt_cache", get_prompt_template_cache().clear)
    return manager


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[CacheInvalidationManager] = None
_manager_lock = threading.Lock()


def get_cache_invalidation_manager() -> CacheInvalidationManager:
    """Return the module-level CacheInvalidationManager singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = build_default_manager()
    return _manager


def reset_cache_invalidation_manager() -> None:
    """Reset the singleton (useful in tests)."""
    global _manager
    with _manager_lock:
        _manager = None
