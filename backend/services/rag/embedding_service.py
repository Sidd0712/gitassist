from __future__ import annotations

import asyncio
import logging
import threading

from sentence_transformers import SentenceTransformer

from core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The model name to store in settings.EMBEDDING_MODEL (update your .env):
#   EMBEDDING_MODEL=all-MiniLM-L6-v2
#   PGVECTOR_DIMENSION=384
# ---------------------------------------------------------------------------
_EXPECTED_DIM = 384


class EmbeddingService:
    """
    Semantic embedding service — drop-in replacement for the hashing service.

    Public interface is identical to the original:
        service = EmbeddingService()
        vectors = await service.embed_documents(["code snippet", "another"])
        query_vec = await service.embed_query("find authentication logic")
    """
    _instance = None  # class-level variable, shared across all instantiations

    def __new__(cls, *args, **kwargs):
        # __new__ runs before __init__ every time someone calls EmbeddingService()
        # If an instance already exists, return that same one instead of creating new
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, '_initialized'):
            return
            
        self.settings = get_settings()
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self._initialized = True

        if self.settings.PGVECTOR_DIMENSION != _EXPECTED_DIM:
            logger.warning(
                "PGVECTOR_DIMENSION is set to %d but all-MiniLM-L6-v2 outputs %d. "
                "Update your .env: PGVECTOR_DIMENSION=384",
                self.settings.PGVECTOR_DIMENSION,
                _EXPECTED_DIM,
            )

        self.dimension = self.settings.PGVECTOR_DIMENSION

    # ------------------------------------------------------------------
    # Public property — same as original
    # ------------------------------------------------------------------

    @property
    def embedding_model_name(self) -> str:
        """Returns the model identifier from settings. Set EMBEDDING_MODEL=all-MiniLM-L6-v2"""
        return self.settings.EMBEDDING_MODEL

    # ------------------------------------------------------------------
    # Public async API — identical signatures to original
    # ------------------------------------------------------------------

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document texts (code chunks, READMEs, etc).

        Why async + asyncio.to_thread:
          The original was async but did CPU work inline, which blocks the
          event loop. We offload the model's matrix operations to a thread
          so FastAPI can keep serving other requests during indexing.

        Args:
            texts: List of strings to embed (e.g. code chunks from ChunkingService)

        Returns:
            List of vectors in the same order. Each vector is list[float] length 384.
        """
        if not texts:
            return []

        # asyncio.to_thread runs _embed_batch in a threadpool executor.
        # This keeps the async event loop unblocked while the model runs.
        return await asyncio.to_thread(self._embed_batch, texts)

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string (used by RetrievalService before vector search).

        Uses the exact same embedding space as embed_documents, so query vectors
        and document vectors are directly comparable via cosine similarity.

        Args:
            text: The query string (e.g. "find routing and middleware setup")

        Returns:
            Single vector: list[float] of length 384
        """
        if not text or not text.strip():
            logger.warning("embed_query() called with empty text — returning zero vector")
            return [0.0] * self.dimension

        results = await asyncio.to_thread(self._embed_batch, [text.strip()])
        return results[0]

    # ------------------------------------------------------------------
    # Private: lazy model loader
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """
        Loads the model on first call, then reuses the same instance forever.

        Thread-safe via double-checked locking:
          - First check (outside lock): fast path for when model is already loaded
          - Second check (inside lock): guards against two threads both seeing
            _model=None and both trying to load it simultaneously
        """
        if self._model is None:
            with self._lock:
                if self._model is None:  # re-check after acquiring lock
                    logger.info(
                        "Loading sentence-transformer model '%s' — one-time cost ~2-5s...",
                        self.embedding_model_name,
                    )
                    # device="cpu" is explicit for Render (no GPU available).
                    # The model name comes from settings so you can swap models
                    # just by changing EMBEDDING_MODEL in .env without touching code.
                    self._model = SentenceTransformer(
                        self.embedding_model_name,
                        device="cpu",
                    )
                    logger.info("Model loaded. Output dimension: %d", _EXPECTED_DIM)

        return self._model

    # ------------------------------------------------------------------
    # Private: actual embedding logic
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Runs the model on a list of texts and returns normalized vectors.

        Why batch instead of one-by-one:
          Neural networks process matrices, not individual rows.
          Encoding 32 texts at once is ~10x faster than 32 separate .encode() calls.
          This matters during repo indexing where a single repo can have 200+ chunks.

        normalize_embeddings=True:
          Forces all vectors to unit length (magnitude = 1.0).
          This makes cosine similarity equivalent to a dot product, which is
          faster to compute and is what pgvector optimises for.
          Without this, longer documents score artificially higher just because
          they contain more tokens.
        """
        model = self._get_model()

        # Sanitise inputs — empty strings produce degenerate vectors
        cleaned = [t.strip() if t and t.strip() else " " for t in texts]

        # batch_size=32 is safe for Render's 512MB RAM limit.
        # If you upgrade to a higher-memory tier, you can raise this to 64
        # for faster bulk indexing.
        vectors = model.encode(
            cleaned,
            batch_size=32,
            normalize_embeddings=True,  # unit vectors for cosine similarity
            show_progress_bar=False,    # suppress tqdm noise in server logs
        )

        # vectors is a 2D numpy array shape (len(texts), 384).
        # .tolist() converts each row to a plain Python list[float],
        # which is what psycopg and your RAGStore expect.
        return [v.tolist() for v in vectors]