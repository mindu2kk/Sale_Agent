"""
Tests for CacheInvalidationManager (Task 5.1.5)

Validates:
- Selective invalidation based on ConfigChangeType
- Observer/listener notification
- Integration with existing cache singletons
- ConfigLoader emits events on reload
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from verification.utils.cache_invalidation import (
    CacheAdapter,
    CacheInvalidationManager,
    ConfigChangeEvent,
    ConfigChangeType,
    build_default_manager,
    get_cache_invalidation_manager,
    reset_cache_invalidation_manager,
)


# -----------------------------------------------