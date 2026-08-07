"""Shared cache helpers for warm-path pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from threading import Lock
from typing import Any

from cachetools import TTLCache

from core.config import get_settings

logger = logging.getLogger(__name__)

_memory_caches: dict[str, TTLCache] = {}
_cache_lock = Lock()


def stable_cache_key(payload: Any) -> str:
    """Create a stable hash key for structured cache payloads."""

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


class PipelineCacheService:
    """In-process TTL cache for warm-path pipeline artifacts (same-process repeat/retry requests)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def get_json(self, namespace: str, cache_key: str) -> Any | None:
        """Get a cached JSON-compatible payload."""

        return self._get_memory_cache(namespace).get(cache_key)

    def set_json(self, namespace: str, cache_key: str, payload: Any, ttl_seconds: int | None = None) -> None:
        """Persist a JSON-compatible payload in cache."""

        self._get_memory_cache(namespace)[cache_key] = payload

    def get_ttl(self, namespace: str) -> int:
        """Expose the configured TTL for a namespace."""

        return self._default_ttl(namespace)

    def _get_memory_cache(self, namespace: str) -> TTLCache:
        ttl = self._default_ttl(namespace)
        with _cache_lock:
            cache = _memory_caches.get(namespace)
            if cache is None or cache.ttl != ttl:
                cache = TTLCache(maxsize=256, ttl=ttl)
                _memory_caches[namespace] = cache
            return cache

    def _default_ttl(self, namespace: str) -> int:
        if namespace.startswith("llm_"):
            return self.settings.CACHE_LLM_TTL_SECONDS
        if namespace.startswith("retrieval_"):
            return self.settings.CACHE_RETRIEVAL_TTL_SECONDS
        if namespace.startswith("github_"):
            return self.settings.CACHE_GITHUB_ARTIFACT_TTL_SECONDS
        return max(
            self.settings.CACHE_LLM_TTL_SECONDS,
            self.settings.CACHE_GITHUB_ARTIFACT_TTL_SECONDS,
            self.settings.CACHE_RETRIEVAL_TTL_SECONDS,
        )


def compact_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop empty values before hashing or caching a mapping."""

    result: dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, "", [], {}, ()):
            continue
        result[key] = value
    return result


def now_epoch_seconds() -> int:
    """Return the current epoch seconds for benchmark output."""

    return int(time.time())


def clear_memory_caches() -> None:
    """Clear in-process pipeline caches, mainly for tests."""

    with _cache_lock:
        _memory_caches.clear()
