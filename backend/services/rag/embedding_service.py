"""Embedding service with explicit remote configuration and local fallback mode."""

from __future__ import annotations

import hashlib
import logging
import math
import re

from langchain_openai import OpenAIEmbeddings

from core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings for repo chunks and retrieval queries."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = self._build_remote_client()

    @property
    def embedding_model_name(self) -> str:
        if self._client is None:
            return "local-hash-256"
        return self.settings.EMBEDDING_MODEL

    def _build_remote_client(self) -> OpenAIEmbeddings | None:
        """Create a remote embedding client only when embedding config is explicit."""

        base_url = self.settings.EMBEDDING_BASE_URL.strip()
        api_key = self.settings.EMBEDDING_API_KEY.strip()

        # Do not silently inherit chat settings for embeddings.
        # If the user wants remote embeddings, they should configure them explicitly.
        if not base_url and not api_key:
            logger.info("Embedding config not set explicitly; using local hash embeddings.")
            return None

        kwargs = {
            "model": self.settings.EMBEDDING_MODEL,
            "api_key": api_key or "placeholder",
            "max_retries": self.settings.EMBEDDING_MAX_RETRIES,
            "timeout": self.settings.EMBEDDING_TIMEOUT_SECONDS,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if self.settings.EMBEDDING_DIMENSIONS > 0:
            kwargs["dimensions"] = self.settings.EMBEDDING_DIMENSIONS

        try:
            return OpenAIEmbeddings(**kwargs)
        except TypeError:
            kwargs.pop("dimensions", None)
            return OpenAIEmbeddings(**kwargs)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""

        if not texts:
            return []
        if self._client is not None:
            try:
                return await self._client.aembed_documents(texts)
            except Exception as exc:  # pragma: no cover - remote fallback path
                logger.warning("Remote embeddings failed, falling back to local hash vectors: %s", exc)
        return [self._hash_embedding(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

        if self._client is not None:
            try:
                return await self._client.aembed_query(text)
            except Exception as exc:  # pragma: no cover - remote fallback path
                logger.warning("Remote query embedding failed, falling back to local hash vector: %s", exc)
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str, dimensions: int = 256) -> list[float]:
        """Create a deterministic local vector using token hashing."""

        vector = [0.0] * dimensions
        tokens = re.findall(r"[A-Za-z0-9_./-]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]
