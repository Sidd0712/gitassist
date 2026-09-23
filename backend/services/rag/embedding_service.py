from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque

import cohere
import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)

# BGE models are asymmetrically trained like Cohere's, but fastembed does NOT apply
# this automatically. The worker (gh_embed_worker.py) prepends this to query texts
# only — kept here too as the single source of truth both sides must agree on.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Cohere's batch limit per API call.
# embed-english-light-v3.0 supports up to 96 texts per call.
# The free tier's 1000 calls/month is metered by call count, not tokens, so
# packing each call to the documented max directly buys more effective quota —
# there's no per-call token cap tight enough at our chunk size (RAG_CHUNK_TOKENS)
# to make a smaller batch worthwhile.
_BATCH_SIZE = 96
_EXPECTED_DIM = 384

# Discovered live (2026-09): a Cohere Trial key enforces 40 calls/minute — a
# call-rate limit, not just the documented monthly 1000-call cap. Concurrent
# multi-repo indexing (INDEXER_MAX_CONCURRENCY) plus per-repo batch
# parallelism (_embed_batched's asyncio.gather) can fire far more than 40
# embed calls within the same second, so relying on retry-after-429 alone
# means real, relevant repos fail to index under normal multi-repo load (seen
# live: 3 concurrently-indexing repos all exhausted their retries and failed).
# A shared sliding-window limiter throttles calls *before* they're sent,
# rather than only reacting after Cohere has already rejected them.
_MAX_CALLS_PER_MINUTE = 35  # stays under the observed 40/min trial-key ceiling


class EmbeddingService:
    """
    Dual-provider semantic embedding service, selected via EMBEDDING_PROVIDER.

    Public interface is identical regardless of provider:

        service = EmbeddingService()
        vectors = await service.embed_documents(["code snippet", "another"])
        query_vec = await service.embed_query("find authentication logic")

    Providers:
      "github_actions" (default) — dispatches the batch to a GitHub Actions
      workflow (backend/scripts/gh_embed_worker.py) that runs fastembed +
      BAAI/bge-small-en-v1.5 on a GitHub-hosted runner and writes the result
      back to a Postgres job row this service polls. No rate limit, no
      monthly cap, no RAM cost to this process — Cohere's Trial key (5-40
      calls/min per Cohere's own docs and observed 429 bodies, 1000
      calls/month) caused real indexing failures under multi-repo/multi-user
      load; a production Cohere key would cost real money to fix the same
      problem. Tradeoff: per-job dispatch/queue/runner-startup latency,
      measured live 2026-09 at ~15-35s beyond the embedding compute itself.

      "cohere" — hosted API, zero dispatch latency, but rate-limited and
      metered as above. Kept as a fallback path, switchable via
      EMBEDDING_PROVIDER without a code change.
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

        self.settings = get_settings()
        self.provider = self.settings.EMBEDDING_PROVIDER

        if self.provider != "github_actions":
            # AsyncClient is used throughout — Cohere's async client is non-blocking,
            # so embed calls don't hold up the FastAPI event loop while waiting for
            # the HTTP response from Cohere's servers.
            self._client = cohere.AsyncClientV2(api_key=self.settings.COHERE_API_KEY)

            # Shared across every call this singleton makes, from any repo's
            # indexing task or any concurrent request -- a single process-wide
            # rate limit, matching how Cohere actually enforces it (per API key,
            # not per caller).
            self._call_timestamps: deque[float] = deque()
            self._rate_limit_lock = asyncio.Lock()

        self._initialized = True

    async def _throttle_for_rate_limit(self) -> None:
        """Block until sending one more call would stay under the per-minute cap."""

        async with self._rate_limit_lock:
            while True:
                now = time.monotonic()
                while self._call_timestamps and now - self._call_timestamps[0] >= 60:
                    self._call_timestamps.popleft()
                if len(self._call_timestamps) < _MAX_CALLS_PER_MINUTE:
                    self._call_timestamps.append(now)
                    return
                wait = 60 - (now - self._call_timestamps[0]) + 0.1
                logger.info("Cohere call-rate limiter: pausing %.1fs to stay under %d calls/min", wait, _MAX_CALLS_PER_MINUTE)
                await asyncio.sleep(max(wait, 0.1))

    # ------------------------------------------------------------------
    # Public property — same as original
    # ------------------------------------------------------------------

    @property
    def embedding_model_name(self) -> str:
        """
        Returns the active model identifier. This is part of the indexing cache
        key (see CorpusService.is_indexed) — switching EMBEDDING_PROVIDER changes
        this value, which naturally forces every repo to re-index under the new
        vector space instead of silently mixing incompatible vectors.
        """
        if self.provider == "github_actions":
            return self.settings.LOCAL_EMBEDDING_MODEL
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

        if self.provider == "github_actions":
            return await self._embed_via_github_actions(texts, is_query=False)
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
            return [0.0] * _EXPECTED_DIM

        if self.provider == "github_actions":
            results = await self._embed_via_github_actions([text.strip()], is_query=True)
        else:
            results = await self._embed_batched([text.strip()], input_type="search_query")
        return results[0]

    async def embed_queries_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of semantic intent strings as search_query vectors.

        IMPORTANT: Use this method — not embed_documents — when embedding
        concept families, capability labels, or any text that represents
        "what we are looking for" rather than "content we have found".

        Background: Cohere's retrieval model is asymmetrically trained.
        Query vectors (search_query) and document vectors (search_document)
        live in different subspaces that are compatible with each other.
        Comparing two search_document vectors (e.g. a concept label embedded
        as a document vs a repo chunk embedded as a document) produces
        depressed cosine scores in the 0.30–0.48 range instead of the
        expected 0.55–0.80. This was the root cause of every capability
        showing as "Missing" — all scores fell below the 0.52 threshold.

        Args:
            texts: List of intent/concept strings to embed as queries.
                   Typical callers: _compute_concept_family_embeddings,
                   rank_repo_evidence concept family block.

        Returns:
            List of float vectors in the same order. Each vector is length 384.
        """
        if not texts:
            return []
        if self.provider == "github_actions":
            return await self._embed_via_github_actions(texts, is_query=True)
        return await self._embed_batched(texts, input_type="search_query")

    # ------------------------------------------------------------------
    # Private: GitHub Actions worker dispatch
    # ------------------------------------------------------------------

    async def _embed_via_github_actions(self, texts: list[str], is_query: bool) -> list[list[float]]:
        """
        Runs one full batch (however large) as a single workflow job, rather
        than splitting into Cohere-style sub-batches — there's no per-call
        size limit to work around, only a per-job dispatch/queue/startup tax
        that's cheaper to pay once than N times.
        """
        from services.rag.store_service import get_rag_store

        store = get_rag_store()
        job_id = str(uuid.uuid4())
        store.create_embedding_job(job_id, texts, is_query)

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"https://api.github.com/repos/{self.settings.GITHUB_ACTIONS_OWNER}/"
                f"{self.settings.GITHUB_ACTIONS_REPO}/actions/workflows/"
                f"{self.settings.GITHUB_ACTIONS_WORKFLOW_FILE}/dispatches",
                headers={
                    "Authorization": f"token {self.settings.GITHUB_ACTIONS_TRIGGER_TOKEN}",
                    "Accept": "application/vnd.github+json",
                },
                json={"ref": "main", "inputs": {"job_id": job_id}},
            )
            if response.status_code != 204:
                store.delete_embedding_job(job_id)
                raise RuntimeError(
                    f"Failed to dispatch embedding workflow: {response.status_code} {response.text}"
                )

        deadline = time.monotonic() + self.settings.GITHUB_ACTIONS_JOB_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            await asyncio.sleep(self.settings.GITHUB_ACTIONS_POLL_INTERVAL_SECONDS)
            job = store.get_embedding_job(job_id)
            if job is None:
                continue
            if job["status"] == "complete":
                store.delete_embedding_job(job_id)
                return job["result"]
            if job["status"] == "failed":
                store.delete_embedding_job(job_id)
                raise RuntimeError(f"Embedding worker reported failure: {job['error']}")

        store.delete_embedding_job(job_id)
        raise TimeoutError(
            f"Embedding job {job_id} did not complete within "
            f"{self.settings.GITHUB_ACTIONS_JOB_TIMEOUT_SECONDS}s"
        )

    # ------------------------------------------------------------------
    # Private: batching logic (Cohere)
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
        Makes a single Cohere embed API call with retry logic for rate limits.

        A process-wide sliding-window limiter (_throttle_for_rate_limit) keeps
        calls under the per-minute cap proactively. Retries below are a
        backstop for the trial key's per-minute window still being exceeded
        (e.g. a burst just before this process started, or a lower limit than
        expected) — backoff is long enough to let a full minute window clear,
        not the few-second backoff that's adequate for a generic 429 but not
        a strict per-minute cap.
        """
        from cohere.errors.too_many_requests_error import TooManyRequestsError

        max_attempts = 4
        for attempt in range(max_attempts):
            await self._throttle_for_rate_limit()
            try:
                response = await self._client.embed(
                    texts=texts,
                    model=self.embedding_model_name,
                    input_type=input_type,
                    embedding_types=["float"],
                )
                return [list(embedding) for embedding in response.embeddings.float_]

            except TooManyRequestsError:
                if attempt == max_attempts - 1:
                    logger.error("Cohere rate limit exceeded after %d attempts — giving up", max_attempts)
                    raise
                wait = 20 * (attempt + 1)  # 20s, 40s, 60s
                logger.warning(
                    "Cohere rate limit hit (attempt %d/%d), retrying in %ds...",
                    attempt + 1, max_attempts, wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError("Cohere embed call failed unexpectedly.")