"""
Tests for PromptTemplateCache (task 5.1.3) and CachedPromptTemplates.

Covers:
- Cache hit returns same rendered template
- Cache miss triggers rendering and stores result
- Version change invalidates cache (stale version never matched)
- TTL expiration works correctly
- Thread safety (basic concurrent access)
- Cache statistics (hits, misses, hit_rate)
- invalidate_template() removes all entries for a name
- CachedPromptTemplates.render() integrates with PromptTemplateManager
- CachedPromptTemplates.reload() clears cache
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from verification.utils.prompt_cache import (
    PromptTemplateCache,
    configure_prompt_template_cache,
    get_prompt_template_cache,
)
from verification.config.prompt_templates import (
    CachedPromptTemplates,
    PromptTemplateError,
    get_cached_prompt_manager,
)

PROMPTS_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cache() -> PromptTemplateCache:
    return PromptTemplateCache(max_size=64, default_ttl_seconds=60.0)


@pytest.fixture()
def cached_mgr() -> CachedPromptTemplates:
    return CachedPromptTemplates(PROMPTS_PATH)


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------

TEMPLATE_A = "Hello {name}!"
TEMPLATE_B = "Hi {name}!"  # different content → different version

VARS = {"name": "World"}
VARS_ALT = {"name": "Alice"}


# ---------------------------------------------------------------------------
# PromptTemplateCache — version / hash helpers
# ---------------------------------------------------------------------------

def test_compute_version_is_deterministic():
    v1 = PromptTemplateCache.compute_version(TEMPLATE_A)
    v2 = PromptTemplateCache.compute_version(TEMPLATE_A)
    assert v1 == v2


def test_compute_version_differs_for_different_content():
    v1 = PromptTemplateCache.compute_version(TEMPLATE_A)
    v2 = PromptTemplateCache.compute_version(TEMPLATE_B)
    assert v1 != v2


def test_compute_variables_hash_is_deterministic():
    h1 = PromptTemplateCache.compute_variables_hash(VARS)
    h2 = PromptTemplateCache.compute_variables_hash(VARS)
    assert h1 == h2


def test_compute_variables_hash_order_independent():
    h1 = PromptTemplateCache.compute_variables_hash({"a": 1, "b": 2})
    h2 = PromptTemplateCache.compute_variables_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_compute_variables_hash_differs_for_different_vars():
    h1 = PromptTemplateCache.compute_variables_hash(VARS)
    h2 = PromptTemplateCache.compute_variables_hash(VARS_ALT)
    assert h1 != h2


# ---------------------------------------------------------------------------
# PromptTemplateCache — basic get/put
# ---------------------------------------------------------------------------

def test_cache_miss_returns_none(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)
    assert cache.get("greeting", version, vars_hash) is None


def test_cache_hit_returns_stored_value(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)
    rendered = "Hello World!"

    cache.put("greeting", version, vars_hash, rendered)
    result = cache.get("greeting", version, vars_hash)
    assert result == rendered


def test_cache_hit_returns_same_object_content(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)
    rendered = "Hello World!"

    cache.put("greeting", version, vars_hash, rendered)
    r1 = cache.get("greeting", version, vars_hash)
    r2 = cache.get("greeting", version, vars_hash)
    assert r1 == r2 == rendered


# ---------------------------------------------------------------------------
# PromptTemplateCache — version change invalidates cache
# ---------------------------------------------------------------------------

def test_version_change_causes_cache_miss(cache):
    """Old version key is never matched after template content changes."""
    version_old = PromptTemplateCache.compute_version(TEMPLATE_A)
    version_new = PromptTemplateCache.compute_version(TEMPLATE_B)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)

    cache.put("greeting", version_old, vars_hash, "Hello World!")

    # Looking up with new version → miss (old entry still exists but under old key)
    assert cache.get("greeting", version_new, vars_hash) is None
    # Old version still retrievable
    assert cache.get("greeting", version_old, vars_hash) == "Hello World!"


def test_invalidate_template_removes_all_versions(cache):
    version_old = PromptTemplateCache.compute_version(TEMPLATE_A)
    version_new = PromptTemplateCache.compute_version(TEMPLATE_B)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)

    cache.put("greeting", version_old, vars_hash, "Hello World!")
    cache.put("greeting", version_new, vars_hash, "Hi World!")

    removed = cache.invalidate_template("greeting")
    assert removed == 2
    assert cache.get("greeting", version_old, vars_hash) is None
    assert cache.get("greeting", version_new, vars_hash) is None


def test_invalidate_template_does_not_affect_other_templates(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)

    cache.put("greeting", version, vars_hash, "Hello World!")
    cache.put("farewell", version, vars_hash, "Goodbye World!")

    cache.invalidate_template("greeting")
    assert cache.get("farewell", version, vars_hash) == "Goodbye World!"


def test_invalidate_version_removes_specific_version(cache):
    version_old = PromptTemplateCache.compute_version(TEMPLATE_A)
    version_new = PromptTemplateCache.compute_version(TEMPLATE_B)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)

    cache.put("greeting", version_old, vars_hash, "Hello World!")
    cache.put("greeting", version_new, vars_hash, "Hi World!")

    removed = cache.invalidate_version("greeting", version_old)
    assert removed == 1
    assert cache.get("greeting", version_old, vars_hash) is None
    assert cache.get("greeting", version_new, vars_hash) == "Hi World!"


# ---------------------------------------------------------------------------
# PromptTemplateCache — TTL expiration
# ---------------------------------------------------------------------------

def test_ttl_expiration(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)

    # Store with very short TTL
    cache.put("greeting", version, vars_hash, "Hello World!", ttl_seconds=0.05)
    assert cache.get("greeting", version, vars_hash) == "Hello World!"

    time.sleep(0.1)  # wait for expiry
    assert cache.get("greeting", version, vars_hash) is None


def test_no_ttl_entry_never_expires(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)

    cache.put("greeting", version, vars_hash, "Hello World!", ttl_seconds=None)
    time.sleep(0.05)
    assert cache.get("greeting", version, vars_hash) == "Hello World!"


# ---------------------------------------------------------------------------
# PromptTemplateCache — LRU eviction
# ---------------------------------------------------------------------------

def test_lru_eviction_when_over_capacity():
    small_cache = PromptTemplateCache(max_size=2, default_ttl_seconds=None)
    version = PromptTemplateCache.compute_version(TEMPLATE_A)

    h1 = PromptTemplateCache.compute_variables_hash({"n": "1"})
    h2 = PromptTemplateCache.compute_variables_hash({"n": "2"})
    h3 = PromptTemplateCache.compute_variables_hash({"n": "3"})

    small_cache.put("t", version, h1, "r1")
    small_cache.put("t", version, h2, "r2")
    small_cache.put("t", version, h3, "r3")  # should evict h1

    assert small_cache.size() == 2
    assert small_cache.get("t", version, h1) is None  # evicted
    assert small_cache.get("t", version, h2) == "r2"
    assert small_cache.get("t", version, h3) == "r3"


# ---------------------------------------------------------------------------
# PromptTemplateCache — statistics
# ---------------------------------------------------------------------------

def test_stats_initial_state(cache):
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 0.0
    assert stats["size"] == 0


def test_stats_after_miss(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)
    cache.get("greeting", version, vars_hash)
    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0


def test_stats_after_hit(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)
    cache.put("greeting", version, vars_hash, "Hello World!")
    cache.get("greeting", version, vars_hash)
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 1.0


def test_stats_hit_rate_calculation(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)

    cache.get("greeting", version, vars_hash)  # miss
    cache.put("greeting", version, vars_hash, "Hello World!")
    cache.get("greeting", version, vars_hash)  # hit
    cache.get("greeting", version, vars_hash)  # hit

    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert abs(stats["hit_rate"] - 2 / 3) < 1e-9


def test_clear_resets_stats(cache):
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)
    cache.put("greeting", version, vars_hash, "Hello World!")
    cache.get("greeting", version, vars_hash)
    cache.clear()
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["size"] == 0


# ---------------------------------------------------------------------------
# PromptTemplateCache — thread safety
# ---------------------------------------------------------------------------

def test_concurrent_put_and_get():
    """Multiple threads writing and reading should not raise or corrupt data."""
    shared_cache = PromptTemplateCache(max_size=200, default_ttl_seconds=None)
    errors: list = []

    def worker(thread_id: int) -> None:
        try:
            version = PromptTemplateCache.compute_version(f"template_{thread_id}")
            vars_hash = PromptTemplateCache.compute_variables_hash({"id": thread_id})
            name = f"tmpl_{thread_id}"
            shared_cache.put(name, version, vars_hash, f"rendered_{thread_id}")
            result = shared_cache.get(name, version, vars_hash)
            assert result == f"rendered_{thread_id}"
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"


def test_concurrent_invalidate_and_get():
    """Invalidation under concurrent reads should not raise."""
    shared_cache = PromptTemplateCache(max_size=200, default_ttl_seconds=None)
    version = PromptTemplateCache.compute_version(TEMPLATE_A)
    vars_hash = PromptTemplateCache.compute_variables_hash(VARS)
    shared_cache.put("greeting", version, vars_hash, "Hello World!")

    errors: list = []

    def reader() -> None:
        try:
            for _ in range(20):
                shared_cache.get("greeting", version, vars_hash)
        except Exception as exc:
            errors.append(exc)

    def invalidator() -> None:
        try:
            for _ in range(5):
                shared_cache.invalidate_template("greeting")
                shared_cache.put("greeting", version, vars_hash, "Hello World!")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(5)]
    threads.append(threading.Thread(target=invalidator))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

def test_get_prompt_template_cache_returns_singleton():
    c1 = get_prompt_template_cache()
    c2 = get_prompt_template_cache()
    assert c1 is c2


def test_configure_prompt_template_cache_replaces_singleton():
    new_cache = configure_prompt_template_cache(max_size=10, default_ttl_seconds=30.0)
    assert get_prompt_template_cache() is new_cache
    # Restore default for other tests
    configure_prompt_template_cache()


# ---------------------------------------------------------------------------
# CachedPromptTemplates — integration with PromptTemplateManager
# ---------------------------------------------------------------------------

def test_cached_render_returns_correct_output(cached_mgr):
    rendered = cached_mgr.render(
        "price_accuracy_check",
        objection_text="iPhone 15 giá bao nhiêu?",
        draft_response="iPhone 15 giá 25,000,000 VND",
        db_data="iPhone 15: 24,990,000 VND",
        price_tolerance="1",
        critical_threshold="30",
    )
    assert "iPhone 15 giá bao nhiêu?" in rendered
    assert "24,990,000 VND" in rendered


def test_cached_render_second_call_is_cache_hit(cached_mgr):
    kwargs = dict(
        objection_text="test",
        draft_response="draft",
        db_data="db",
        price_tolerance="1",
        critical_threshold="30",
    )
    cached_mgr.render("price_accuracy_check", **kwargs)
    stats_before = cached_mgr.get_cache_stats()

    cached_mgr.render("price_accuracy_check", **kwargs)
    stats_after = cached_mgr.get_cache_stats()

    assert stats_after["hits"] == stats_before["hits"] + 1


def test_cached_render_different_vars_are_separate_entries(cached_mgr):
    base_kwargs = dict(
        draft_response="draft",
        db_data="db",
        price_tolerance="1",
        critical_threshold="30",
    )
    r1 = cached_mgr.render("price_accuracy_check", objection_text="Q1", **base_kwargs)
    r2 = cached_mgr.render("price_accuracy_check", objection_text="Q2", **base_kwargs)
    assert r1 != r2


def test_cached_render_missing_variable_raises(cached_mgr):
    with pytest.raises(PromptTemplateError):
        cached_mgr.render("price_accuracy_check", objection_text="only this")


def test_reload_clears_cache(cached_mgr):
    kwargs = dict(
        objection_text="test",
        draft_response="draft",
        db_data="db",
        price_tolerance="1",
        critical_threshold="30",
    )
    cached_mgr.render("price_accuracy_check", **kwargs)
    assert cached_mgr.get_cache_stats()["size"] > 0

    cached_mgr.reload()
    assert cached_mgr.get_cache_stats()["size"] == 0


def test_invalidate_template_removes_entries(cached_mgr):
    kwargs = dict(
        objection_text="test",
        draft_response="draft",
        db_data="db",
        price_tolerance="1",
        critical_threshold="30",
    )
    cached_mgr.render("price_accuracy_check", **kwargs)
    assert cached_mgr.get_cache_stats()["size"] >= 1

    removed = cached_mgr.invalidate_template("price_accuracy_check")
    assert removed >= 1
    # After invalidation, next render is a miss → re-renders
    stats_before = cached_mgr.get_cache_stats()
    cached_mgr.render("price_accuracy_check", **kwargs)
    stats_after = cached_mgr.get_cache_stats()
    assert stats_after["misses"] == stats_before["misses"] + 1


def test_version_changes_when_template_content_changes(cached_mgr):
    """Simulate a template content change: old cached render is not returned."""
    raw_original = cached_mgr.get_template("price_accuracy_check")
    version_original = PromptTemplateCache.compute_version(raw_original)

    # Simulate changed template content
    raw_modified = raw_original + "\n# modified"
    version_modified = PromptTemplateCache.compute_version(raw_modified)

    assert version_original != version_modified

    vars_hash = PromptTemplateCache.compute_variables_hash({"x": 1})
    # Manually store under old version
    cached_mgr.cache.put("price_accuracy_check", version_original, vars_hash, "old render")

    # Looking up with new version → miss
    result = cached_mgr.cache.get("price_accuracy_check", version_modified, vars_hash)
    assert result is None


def test_get_cache_stats_returns_dict(cached_mgr):
    stats = cached_mgr.get_cache_stats()
    assert isinstance(stats, dict)
    for key in ("hits", "misses", "hit_rate", "size", "max_size"):
        assert key in stats


def test_list_templates_works(cached_mgr):
    names = cached_mgr.list_templates()
    assert "price_accuracy_check" in names
    assert names == sorted(names)


def test_get_cached_prompt_manager_singleton():
    m1 = get_cached_prompt_manager()
    m2 = get_cached_prompt_manager()
    assert m1 is m2
