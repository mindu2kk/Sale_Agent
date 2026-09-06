"""Project-wide test compatibility helpers."""

from __future__ import annotations

import asyncio
import warnings

import pytest
from hypothesis import HealthCheck, settings


settings.register_profile(
    "project",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("project")


@pytest.fixture(autouse=True)
def ensure_legacy_event_loop() -> None:
    """Keep synchronous asyncio tests compatible with Python 3.13."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            asyncio.get_event_loop()
            created_loop = None
        except RuntimeError:
            created_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(created_loop)

    yield

    if created_loop is not None:
        if not created_loop.is_closed():
            created_loop.close()
        asyncio.set_event_loop(None)
