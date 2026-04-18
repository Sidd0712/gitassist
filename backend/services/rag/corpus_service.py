"""Background-safe corpus maintenance for shared Postgres storage."""

from __future__ import annotations

import logging

from models.schemas import RepoFetchPlan, RepoSearchResult, ShallowRepoEvidence
from services.github_service import fetch_repo_files_for_indexing
from services.rag.chunking_service import ChunkingService
from services.rag.embedding_service import EmbeddingService
from services.rag.store_service import ClaimedIndexJob, get_rag_store

logger = logging.getLogger(__name__)


class CorpusService:
    """Queue and process repository indexing without blocking the request path."""

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
        return self.store.load_repository(
            repository.full_name,
            repository.commit_sha,
            self.embedding_model_name,
            self.chunking_version,
        )

    def enqueue_index_job(self, plan: RepoFetchPlan, shallow: ShallowRepoEvidence) -> bool:
        return self.store.enqueue_index_job(
            plan,
            shallow,
            self.embedding_model_name,
            self.chunking_version,
        )

    def claim_index_job(self, worker_id: str) -> ClaimedIndexJob | None:
        return self.store.claim_index_job(worker_id, self.embedding_model_name, self.chunking_version)

    async def process_index_job(self, job: ClaimedIndexJob) -> RepoSearchResult:
        """Fetch, chunk, embed, and persist a claimed indexing job."""

        repository = job.plan.repository
        files = await fetch_repo_files_for_indexing(job.plan)
        if not files:
            logger.warning("No files fetched for indexing: %s", repository.full_name)
            self.store.replace_repository_index(
                repository=repository,
                chunks=[],
                embeddings=[],
                embedding_model=self.embedding_model_name,
                chunking_version=self.chunking_version,
            )
            self.store.mark_index_job_completed(
                job,
                self.embedding_model_name,
                self.chunking_version,
                chunk_count=0,
            )
            return repository

        chunks = self.chunker.chunk_repository(repository, files)
        if not chunks:
            logger.warning("No chunks generated for repository: %s", repository.full_name)
            self.store.replace_repository_index(
                repository=repository,
                chunks=[],
                embeddings=[],
                embedding_model=self.embedding_model_name,
                chunking_version=self.chunking_version,
            )
            self.store.mark_index_job_completed(
                job,
                self.embedding_model_name,
                self.chunking_version,
                chunk_count=0,
            )
            return repository

        embeddings = await self.embedding_service.embed_documents([chunk.text for chunk in chunks])
        self.store.replace_repository_index(
            repository=repository,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=self.embedding_model_name,
            chunking_version=self.chunking_version,
        )
        self.store.mark_index_job_completed(
            job,
            self.embedding_model_name,
            self.chunking_version,
            chunk_count=len(chunks),
        )
        logger.info(
            "Indexed %s with %d files and %d chunks",
            repository.full_name,
            len(files),
            len(chunks),
        )
        return repository

    async def run_next_index_job(self, worker_id: str) -> bool:
        """Claim and process a single background indexing job."""

        job = self.claim_index_job(worker_id)
        if job is None:
            return False

        try:
            await self.process_index_job(job)
        except Exception as exc:
            self.store.mark_index_job_failed(
                job,
                self.embedding_model_name,
                self.chunking_version,
                str(exc),
            )
            logger.exception("Background indexing failed for %s", job.plan.repository.full_name)
        return True
