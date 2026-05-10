"""
PromptTemplateManager — loads and renders binary verification prompt templates.

Templates are stored in prompts.yaml and support simple {variable} substitution.
Loaded templates are cached in-memory to avoid repeated disk reads.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


class PromptTemplateError(Exception):
    """Raised when a template is missing or a required variable is not supplied."""


class PromptTemplateManager:
    """
    Loads prompt templates from a YAML file and renders them with variable substitution.

    Usage::

        manager = PromptTemplateManager()
        rendered = manager.render("price_accuracy_check", draft_response="...", db_data="...")

    The YAML file may contain templates at the top level *or* nested under a parent key.
    Each template entry must have a ``template`` field with the prompt text.
    Variables are expressed as ``{variable_name}`` placeholders.

    Templates are cached after the first load so repeated calls are cheap.
    """

    def __init__(self, prompts_path: Optional[str | Path] = None) -> None:
        self._path: Path = Path(prompts_path) if prompts_path else _DEFAULT_PROMPTS_PATH
        self._cache: Dict[str, str] = {}
        self._raw: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_template(self, name: str) -> str:
        """Return the raw (un-rendered) template string for *name*.

        Raises :class:`PromptTemplateError` if the template is not found.
        """
        if name in self._cache:
            return self._cache[name]

        self._ensure_loaded()
        template_text = self._resolve_template(name)
        self._cache[name] = template_text
        return template_text

    def render(self, name: str, **variables: Any) -> str:
        """Render template *name* by substituting *variables*.

        All ``{placeholder}`` tokens in the template are replaced with the
        corresponding keyword argument values.  Missing variables raise
        :class:`PromptTemplateError`.
        """
        template_text = self.get_template(name)
        return self._substitute(name, template_text, variables)

    def list_templates(self) -> list[str]:
        """Return the names of all available top-level templates."""
        self._ensure_loaded()
        assert self._raw is not None
        names: list[str] = []
        for key, value in self._raw.items():
            if isinstance(value, dict) and "template" in value:
                names.append(key)
        return sorted(names)

    def reload(self) -> None:
        """Clear the cache and force a fresh load from disk on next access."""
        self._cache.clear()
        self._raw = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._raw is not None:
            return
        if not self._path.exists():
            raise PromptTemplateError(
                f"Prompts file not found: {self._path}"
            )
        with open(self._path, "r", encoding="utf-8") as fh:
            self._raw = yaml.safe_load(fh) or {}

    def _resolve_template(self, name: str) -> str:
        """Walk the YAML structure to find a template by name."""
        assert self._raw is not None

        # Direct top-level key
        if name in self._raw:
            entry = self._raw[name]
            if isinstance(entry, dict) and "template" in entry:
                return entry["template"]
            raise PromptTemplateError(
                f"Template '{name}' found but has no 'template' field."
            )

        # Search one level deep (e.g. verification_prompts.master_verification)
        for section_value in self._raw.values():
            if not isinstance(section_value, dict):
                continue
            if name in section_value:
                entry = section_value[name]
                if isinstance(entry, dict) and "template" in entry:
                    return entry["template"]

        raise PromptTemplateError(
            f"Template '{name}' not found in {self._path}. "
            f"Available top-level templates: {self.list_templates()}"
        )

    @staticmethod
    def _substitute(name: str, template: str, variables: Dict[str, Any]) -> str:
        """Replace {placeholder} tokens; raise on missing variables."""
        # Collect all placeholder names in the template (skip {{ }} escapes)
        placeholders = set(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", template))
        missing = placeholders - set(variables.keys())
        if missing:
            raise PromptTemplateError(
                f"Template '{name}' requires variables {sorted(missing)} "
                f"but they were not provided."
            )
        # Use str.format_map so that {{ }} literal braces are preserved
        return template.format_map(variables)


# ---------------------------------------------------------------------------
# Module-level singleton (optional convenience)
# ---------------------------------------------------------------------------

_default_manager: Optional[PromptTemplateManager] = None


def get_prompt_manager(prompts_path: Optional[str | Path] = None) -> PromptTemplateManager:
    """Return a shared :class:`PromptTemplateManager` instance.

    If *prompts_path* is provided the first time this is called, that path is
    used; subsequent calls without a path reuse the existing instance.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptTemplateManager(prompts_path)
    return _default_manager


# ---------------------------------------------------------------------------
# CachedPromptTemplates — wrapper with version-controlled caching
# ---------------------------------------------------------------------------

from verification.utils.prompt_cache import PromptTemplateCache
from verification.utils.prompt_compressor import PromptCompressor, CompressionResult


class CachedPromptTemplates:
    """
    A wrapper around :class:`PromptTemplateManager` that caches rendered
    prompt templates with version control and optional prompt compression.

    Version is derived from the SHA256 hash of the raw template content, so
    any edit to a template automatically invalidates its cached renders.

    Usage::

        cached = CachedPromptTemplates(compression_level="light")
        rendered = cached.render(
            "price_accuracy_check",
            objection_text="...",
            draft_response="...",
            db_data="...",
            price_tolerance="1",
            critical_threshold="30",
        )

    Cache statistics are available via :meth:`get_cache_stats`.
    Compression statistics are available via :meth:`get_last_compression_result`.
    """

    def __init__(
        self,
        prompts_path: Optional[str | Path] = None,
        cache: Optional[PromptTemplateCache] = None,
        cache_ttl_seconds: Optional[float] = 3600.0,
        cache_max_size: int = 512,
        compression_level: str = "none",
        compressor: Optional[PromptCompressor] = None,
    ) -> None:
        self._manager = PromptTemplateManager(prompts_path)
        self._cache = cache or PromptTemplateCache(
            max_size=cache_max_size,
            default_ttl_seconds=cache_ttl_seconds,
        )
        self._compressor = compressor or PromptCompressor(level=compression_level)
        self._last_compression: Optional[CompressionResult] = None

    # ------------------------------------------------------------------
    # Public API (mirrors PromptTemplateManager)
    # ------------------------------------------------------------------

    def get_template(self, name: str) -> str:
        """Return the raw template string (delegates to inner manager)."""
        return self._manager.get_template(name)

    def render(self, name: str, **variables: Any) -> str:
        """
        Render *name* with *variables*, applying compression and caching.

        Cache key: (name, version_of_template_content, hash_of_variables)
        Compression is applied after rendering (before caching).
        """
        raw = self._manager.get_template(name)
        version = PromptTemplateCache.compute_version(raw)
        vars_hash = PromptTemplateCache.compute_variables_hash(dict(variables))

        cached = self._cache.get(name, version, vars_hash)
        if cached is not None:
            return cached

        rendered = self._manager.render(name, **variables)

        # Apply compression
        result = self._compressor.compress(rendered)
        self._last_compression = result
        compressed = result.compressed

        self._cache.put(name, version, vars_hash, compressed)
        return compressed

    def get_last_compression_result(self) -> Optional[CompressionResult]:
        """Return the :class:`CompressionResult` from the most recent render call."""
        return self._last_compression

    def list_templates(self) -> list[str]:
        """Return sorted list of available template names."""
        return self._manager.list_templates()

    def reload(self) -> None:
        """
        Reload templates from disk and invalidate the entire cache.

        Call this after prompts.yaml is updated so stale renders are evicted.
        """
        self._manager.reload()
        self._cache.clear()

    def invalidate_template(self, name: str) -> int:
        """
        Force-invalidate all cache entries for a specific template name.

        Returns the number of entries removed.
        """
        return self._cache.invalidate_template(name)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        return self._cache.get_stats()

    # ------------------------------------------------------------------
    # Convenience: expose the underlying cache / compressor for advanced use
    # ------------------------------------------------------------------

    @property
    def cache(self) -> PromptTemplateCache:
        return self._cache

    @property
    def compressor(self) -> PromptCompressor:
        return self._compressor


# ---------------------------------------------------------------------------
# Module-level singleton for CachedPromptTemplates
# ---------------------------------------------------------------------------

_cached_manager: Optional["CachedPromptTemplates"] = None


def get_cached_prompt_manager(
    prompts_path: Optional[str | Path] = None,
    compression_level: str = "none",
) -> "CachedPromptTemplates":
    """Return a shared :class:`CachedPromptTemplates` instance."""
    global _cached_manager
    if _cached_manager is None:
        _cached_manager = CachedPromptTemplates(
            prompts_path, compression_level=compression_level
        )
    return _cached_manager
