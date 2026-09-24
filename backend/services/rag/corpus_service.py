"""Whole-repo indexing (worker side) and index lookups/queueing (API side)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from core.config import get_settings
from models.schemas import RepoFetchPlan, RepoSearchResult
from services.github_service import fetch_repo_files_for_indexing, fetch_repo_tree, select_index_paths
from services.rag.chunking_service import ChunkingService
from services.rag.embedding_service import EmbeddingService
from services.rag.store_service import get_rag_store

logger = logging.getLogger(__name__)

# Chunks written per batch. Each batch makes the repo more searchable
# ('partial'), so a big repo is chat-ready long before it finishes.
_WRITE_BATCH = 256


class CorpusService:
    """Index repositories and look up / queue their indexes."""

    def __init__(self) -> None:
        self.store = get_rag_store()
        self.chunker = ChunkingService()
        self.embedding_service = EmbeddingService()

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_service.embedding_model_name

    @property
    def chunking_version(self) -> str:
        return self.chunker.chunking_version

    def is_indexed(self, repository: RepoSearchResult) -> bool:
        return self.store.is_indexed(repository, self.embedding_model_name, self.chunking_version)

    def load_indexed_repository(self, repository: RepoSearchResult) -> RepoSearchResult | None:
        """The stored repo if it's searchable yet (partially or fully indexed)."""

        return self.store.load_repository(
            repository.full_name,
            repository.commit_sha,
            self.embedding_model_name,
            self.chunking_version,
        )

    def enqueue(self, repository: RepoSearchResult, path_terms: list[str]) -> None:
        """Hand a repo commit to the indexing worker (no-op if already queued/indexed)."""

        self.store.enqueue_index_job(repository, path_terms, self.embedding_model_name, self.chunking_version)

    def index_status(self, repos: list[tuple[str, str]]) -> dict[str, dict]:
        return self.store.index_status(repos, self.embedding_model_name, self.chunking_version)

    async def index_repository(
        self,
        repository: RepoSearchResult,
        path_terms: list[str],
        embed: Callable[[list[str]], list[list[float]]],
    ) -> int:
        """Fetch, chunk, embed and store a whole repo commit; returns the chunk count.

        `embed` is the worker's synchronous in-process embedder. It runs in a
        thread so the event loop stays free for downloads.
        """

        settings = get_settings()
        model, version = self.embedding_model_name, self.chunking_version

        tree = await fetch_repo_tree(repository.full_name, repository.commit_sha or repository.default_branch)
        paths, _skipped, _chars = select_index_paths(repository.full_name, tree, path_terms)
        files = await fetch_repo_files_for_indexing(RepoFetchPlan(repository=repository, selected_paths=paths))

        chunks = self.chunker.chunk_repository(repository, files)
        if len(chunks) > settings.RAG_REPO_MAX_CHUNKS:
            logger.info(
                "%s: capping %d chunks at %d (lowest-priority files dropped)",
                repository.full_name, len(chunks), settings.RAG_REPO_MAX_CHUNKS,
            )
            chunks = chunks[: settings.RAG_REPO_MAX_CHUNKS]

        self.store.evict_to_budget(len(chunks), keep=(repository.full_name, repository.commit_sha))
        self.store.begin_repository_index(repository, model, version)

        # Every chunk's TEXT lands right away (embedding=NULL), in priority
        # order (README, manifests, source, ... tests, scripts — see
        # select_index_paths). The repo is keyword-searchable in chat within
        # seconds, long before embedding — the slow part — has run at all.
        await asyncio.to_thread(self.store.insert_chunk_texts, repository, chunks, model, version)

        # Writes to Neon take ~40% as long as embedding, so each backfill
        # batch is written while the next one embeds. Chunks stay in the
        # same priority order, so dense search fills in for the
        # highest-value files first.
        pending_write: asyncio.Task | None = None
        for offset in range(0, len(chunks), _WRITE_BATCH):
            batch = chunks[offset : offset + _WRITE_BATCH]
            started = time.monotonic()
            vectors = await asyncio.to_thread(embed, [ChunkingService.embedding_text(chunk) for chunk in batch])
            embed_seconds = time.monotonic() - started
            if pending_write is not None:
                await pending_write
            pending_write = asyncio.create_task(
                asyncio.to_thread(self.store.backfill_embeddings, [c.chunk_id for c in batch], vectors)
            )
            logger.info(
                "%s: %d/%d chunks embedded (%.1f chunks/s)",
                repository.full_name, offset + len(batch), len(chunks), len(batch) / max(embed_seconds, 1e-6),
            )
        if pending_write is not None:
            await pending_write
        self.store.finish_repository_index(repository, model, version)

        logger.info("Indexed %s with %d files and %d chunks", repository.full_name, len(files), len(chunks))
        return len(chunks)
