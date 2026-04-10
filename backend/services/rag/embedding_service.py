"""Embedding service with HuggingFace models."""

from __future__ import annotations

import logging

from langchain_community.embeddings import HuggingFaceEmbeddings

from core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings for repo chunks and retrieval queries."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = self._build_client()

    @property
    def embedding_model_name(self) -> str:
        return self.settings.EMBEDDING_MODEL

    def _build_client(self) -> HuggingFaceEmbeddings:
        """Create a HuggingFace embedding client."""
        
        try:
            logger.info("Initializing HuggingFace embeddings with model: %s", self.settings.EMBEDDING_MODEL)
            kwargs = {
                "model_name": self.settings.EMBEDDING_MODEL,
            }
            if self.settings.LLM_TOKEN.strip():
                kwargs["huggingfacehub_api_token"] = self.settings.LLM_TOKEN
            
            return HuggingFaceEmbeddings(**kwargs)
        except Exception as exc:
            logger.error("Failed to initialize HuggingFace embeddings: %s", exc)
            raise

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using HuggingFace."""

        if not texts:
            return []
        try:
            # HuggingFace embeddings are sync
            return self._client.embed_documents(texts)
        except Exception as exc:
            logger.error("HuggingFace embeddings failed: %s", exc)
            raise

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query using HuggingFace."""

        try:
            # HuggingFace embeddings are sync
            return self._client.embed_query(text)
        except Exception as exc:
            logger.error("HuggingFace query embedding failed: %s", exc)
            raise
