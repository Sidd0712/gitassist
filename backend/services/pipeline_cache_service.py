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
from services.rag.store_service import get_rag_store

logger = logging.getLogger(__name__)

_memory_caches: dict[str, TTLCache] = {}
_cache_lock = Lock()


def stable_cache_key(payload: Any) -> str:
    """Create a stable hash key for structured cache payloads."""

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


class PipelineCacheService:
    """Cross-request cache backed by Postgres when available, with in-process fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._store = None
        self._store_checked = False

    def get_json(self, namespace: str, cache_key: str) -> Any | None:
        """Get a cached JSON-compatible payload."""

        memory_cache = self._get_memory_cache(namespace)
        cached = memory_cache.get(cache_key)
        if cached is not None:
            return cached

        store = self._get_store()
        if store is None:
            return None

        try:
            payload = store.get_cache_entry(namespace, cache_key)
        except Exception as exc:
            logger.debug("Persistent cache read failed for %s:%s: %s", namespace, cache_key, exc)
            return None

        if payload is not None:
            memory_cache[cache_key] = payload
        return payload

    def set_json(self, namespace: str, cache_key: str, payload: Any, ttl_seconds: int | None = None) -> None:
        """Persist a JSON-compatible payload in cache."""

        memory_cache = self._get_memory_cache(namespace)
        memory_cache[cache_key] = payload

        store = self._get_store()
        if store is None:
            return

        try:
            store.set_cache_entry(namespace, cache_key, payload, ttl_seconds or self._default_ttl(namespace))
        except Exception as exc:
            logger.debug("Persistent cache write failed for %s:%s: %s", namespace, cache_key, exc)

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

    def _get_store(self):
        if self._store_checked:
            return self._store

        self._store_checked = True
        if self.settings.RAG_STORE_BACKEND.lower() != "postgres" or not self.settings.DATABASE_URL:
            self._store = None
            return None

        try:
            self._store = get_rag_store()
        except Exception as exc:
            logger.debug("Pipeline cache store unavailable: %s", exc)
            self._store = None
        return self._store

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
