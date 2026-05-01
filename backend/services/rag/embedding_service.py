from __future__ import annotations

import asyncio
import logging

import cohere

from core.config import get_settings

logger = logging.getLogger(__name__)

# Cohere's batch limit per API call.
# embed-english-light-v3.0 supports up to 96 texts per call.
# We use 50 to stay safely under and keep individual payloads small.
_BATCH_SIZE = 50
_EXPECTED_DIM = 384


class EmbeddingService:
    """
    Cohere-backed semantic embedding service.

    Drop-in replacement for both the original hashing service and the
    local MiniLM service — public interface is identical:

        service = EmbeddingService()
        vectors = await service.embed_documents(["code snippet", "another"])
        query_vec = await service.embed_query("find authentication logic")

    Why Cohere instead of a local model:
      Running a local model (MiniLM, etc.) on Render's free tier requires
      ~120MB+ of RAM just for the model weights. Combined with FastAPI,
      Postgres connections, and active requests, this pushes the 512MB
      limit and causes OOM crashes. Cohere handles all computation on
      their servers — your Render instance stays lean (~300MB total).

    Free tier: 1000 API calls/month. At ~15-35 calls per research request
    and monthly resets, this covers roughly 30-65 full requests/month for free.
    After that, cost is ~$0.10/1M tokens — negligible at this scale.
    """

    _instance = None  # class-level, shared across all instantiations

    def __new__(cls, *args, **kwargs):
        # Singleton pattern — __new__ runs before __init__ on every EmbeddingService()
        # call. Returning the existing instance means the Cohere client is only
        # created once, no matter how many files import and instantiate this class.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Guard: __init__ runs every time EmbeddingService() is called (even on the
        # singleton), so we use a flag to make sure the real setup only runs once.
        if hasattr(self, '_initialized'):
            return
        
        # AsyncClient is used throughout — Cohere's async client is non-blocking,
        # so embed calls don't hold up the FastAPI event loop while waiting for
        # the HTTP response from Cohere's servers.
        self._client = cohere.AsyncClientV2(api_key=self.settings.COHERE_API_KEY)

        self._initialized = True

    # ------------------------------------------------------------------
    # Public property — same as original
    # ------------------------------------------------------------------

    @property
    def embedding_model_name(self) -> str:
        """Returns the model identifier from settings.
        Set EMBEDDING_MODEL=embed-english-light-v3.0 in your .env."""
        return self.settings.EMBEDDING_MODEL

    # ------------------------------------------------------------------
    # Public async API — identical signatures to the original service
    # ------------------------------------------------------------------

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document texts (code chunks, READMEs, etc).

        Uses input_type="search_document" — this is Cohere-specific context
        that tells the model these are passages being stored for later retrieval,
        not a query. Using the correct input_type measurably improves the quality
        of semantic similarity comparisons at search time.

        Args:
            texts: List of strings to embed (e.g. code chunks from ChunkingService)

        Returns:
            List of float vectors in the same order. Each vector is length 384.
        """
        if not texts:
            return []

        return await self._embed_batched(texts, input_type="search_document")

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string (used by RetrievalService before vector search).

        Uses input_type="search_query" — the counterpart to "search_document".
        Cohere trains the model so that query vectors and document vectors are
        comparable even though they use different input_type tags. This asymmetric
        encoding is what makes retrieval quality better than symmetric models.

        Args:
            text: The query string (e.g. "find routing and middleware setup")

        Returns:
            Single vector: list[float] of length 384
        """
        if not text or not text.strip():
            logger.warning("embed_query() called with empty text — returning zero vector")
            return [0.0] * self.dimension

        results = await self._embed_batched([text.strip()], input_type="search_query")
        return results[0]

    # ------------------------------------------------------------------
    # Private: batching logic
    # ------------------------------------------------------------------

    async def _embed_batched(
        self, texts: list[str], input_type: str
    ) -> list[list[float]]:
        """
        Splits texts into batches and embeds them, then reassembles in order.

        Why batch at all:
          Cohere's API accepts up to 96 texts per call. During repo indexing
          a single file can produce 200+ chunks. Batching into groups of 50
          means we make ceil(200/50) = 4 API calls instead of 200 — dramatically
          fewer network round trips and less chance of hitting rate limits.

        Why gather instead of sequential awaits:
          asyncio.gather fires all batch calls concurrently rather than waiting
          for each one to finish before starting the next. For 4 batches this
          can cut total latency by ~3x since Cohere processes them in parallel.

        Args:
            texts:      The full list of strings to embed.
            input_type: "search_document" for chunks, "search_query" for queries.

        Returns:
            Flat list of vectors in the same order as the input texts.
        """
        # Sanitise: replace empty strings with a single space.
        # Cohere rejects empty strings with a 400 error.
        cleaned = [t.strip() if t and t.strip() else " " for t in texts]

        # Split into batches of _BATCH_SIZE
        batches = [
            cleaned[i : i + _BATCH_SIZE]
            for i in range(0, len(cleaned), _BATCH_SIZE)
        ]

        # Fire all batch API calls concurrently
        batch_results = await asyncio.gather(
            *[self._call_cohere(batch, input_type) for batch in batches]
        )

        # Flatten: batch_results is list[list[list[float]]], we want list[list[float]]
        return [vec for batch in batch_results for vec in batch]

    async def _call_cohere(
        self, texts: list[str], input_type: str
    ) -> list[list[float]]:
        """
        Makes a single Cohere embed API call and returns the float vectors.

        embedding_types=["float"]:
          Cohere v2 can return embeddings in multiple formats (float, int8, binary).
          We explicitly request float — the same type pgvector expects — to avoid
          any format mismatch. Without this, the API may return a default that
          needs extra conversion.

        Args:
            texts:      A single batch (≤50 strings) to embed.
            input_type: "search_document" or "search_query".

        Returns:
            List of float vectors, one per input text.
        """
        response = await self._client.embed(
            texts=texts,
            model=self.embedding_model_name,  # from settings: embed-english-light-v3.0
            input_type=input_type,            # "search_document" or "search_query"
            embedding_types=["float"],        # we only need float vectors for pgvector
        )

        # response.embeddings.float_ is a list of lists (one per input text).
        # We convert each inner list to a plain Python list[float] — psycopg
        # and pgvector both expect standard Python types, not numpy arrays or
        # Cohere wrapper objects.
        return [list(embedding) for embedding in response.embeddings.float_]
