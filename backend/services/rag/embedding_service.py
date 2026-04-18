"""Embedding service using local sentence-transformers (not HuggingFace API)."""

from __future__ import annotations

import logging
from threading import Lock

from langchain_huggingface import HuggingFaceEmbeddings

from core.config import get_settings

logger = logging.getLogger(__name__)

_shared_client: HuggingFaceEmbeddings | None = None
_client_lock = Lock()


class EmbeddingService:
    """Generate embeddings for repo chunks and retrieval queries."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = self._get_or_create_client()

    @property
    def embedding_model_name(self) -> str:
        return self.settings.EMBEDDING_MODEL

    def _get_or_create_client(self) -> HuggingFaceEmbeddings:
        """Reuse one embedding model instance per process to avoid duplicate memory pressure."""

        global _shared_client

        if _shared_client is not None:
            return _shared_client

        with _client_lock:
            if _shared_client is not None:
                return _shared_client

            try:
                logger.info("Initializing local sentence-transformers model: %s", self.settings.EMBEDDING_MODEL)
                _shared_client = HuggingFaceEmbeddings(model_name=self.settings.EMBEDDING_MODEL)
                return _shared_client
            except Exception as exc:
                logger.error("Failed to initialize sentence-transformers embeddings: %s", exc)
                raise

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using local sentence-transformers."""

        if not texts:
            return []
        try:
            # sentence-transformers embeddings are sync
            return self._client.embed_documents(texts)
        except Exception as exc:
            logger.error("sentence-transformers embeddings failed: %s", exc)
            raise

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query using local sentence-transformers."""

        try:
            # sentence-transformers embeddings are sync
            return self._client.embed_query(text)
        except Exception as exc:
            logger.error("sentence-transformers query embedding failed: %s", exc)
            raise
