"""Lightweight deterministic embeddings for low-memory deployments."""

from __future__ import annotations

import hashlib
import logging
import math
import re

from core.config import get_settings

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class EmbeddingService:
    """Generate compact hashed embeddings without loading Torch models."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.dimension = self.settings.PGVECTOR_DIMENSION

    @property
    def embedding_model_name(self) -> str:
        return self.settings.EMBEDDING_MODEL

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using deterministic hashing."""

        if not texts:
            return []
        return [self._embed_text(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query using the same hashing space as documents."""

        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        token_count = 0

        for token in TOKEN_PATTERN.findall((text or "").lower()):
            token_count += 1
            bucket = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16) % self.dimension
            vector[bucket] += 1.0

        if token_count == 0:
            return vector

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [round(value / norm, 8) for value in vector]
