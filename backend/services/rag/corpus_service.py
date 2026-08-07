"""Repository indexing: fetch, chunk, embed, and persist into Postgres."""

from __future__ import annotations

import logging

from models.schemas import RepoFetchPlan, RepoSearchResult
from services.github_service import fetch_repo_files_for_indexing
from services.rag.chunking_service import ChunkingService
from services.rag.embedding_service import EmbeddingService
from services.rag.store_service import get_rag_store

logger = logging.getLogger(__name__)


class CorpusService:
    """Index repository fetch plans directly, in-process."""

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

    async def index_repository_plan(self, plan: RepoFetchPlan) -> RepoSearchResult:
        """Fetch, chunk, embed, and persist a repository fetch plan immediately."""

        existing = self.load_indexed_repository(plan.repository)
        if existing is not None:
            return existing

        repository = plan.repository
        files = await fetch_repo_files_for_indexing(plan)
        if not files:
            logger.warning("No files fetched for indexing: %s", repository.full_name)
            self.store.replace_repository_index(
                repository=repository,
                chunks=[],
                embeddings=[],
                embedding_model=self.embedding_model_name,
                chunking_version=self.chunking_version,
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
            return repository

        embeddings = await self.embedding_service.embed_documents([chunk.text for chunk in chunks])
        self.store.replace_repository_index(
            repository=repository,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=self.embedding_model_name,
            chunking_version=self.chunking_version,
        )
        logger.info(
            "Indexed %s with %d files and %d chunks",
            repository.full_name,
            len(files),
            len(chunks),
        )
        return repository
