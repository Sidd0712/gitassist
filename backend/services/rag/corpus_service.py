"""Index selected repositories into the local RAG corpus."""

from __future__ import annotations

import logging

from models.schemas import RepoFetchPlan, RepoSearchResult
from services.github_service import fetch_repo_files_for_indexing
from services.rag.chunking_service import ChunkingService
from services.rag.embedding_service import EmbeddingService
from services.rag.store_service import get_rag_store

logger = logging.getLogger(__name__)


class CorpusService:
    """Create and maintain the persistent retrieval corpus."""

    def __init__(self) -> None:
        self.store = get_rag_store()
        self.chunker = ChunkingService()
        self.embedding_service = EmbeddingService()

    async def ensure_indexed(self, plan: RepoFetchPlan) -> RepoSearchResult:
        """Index a repository if its current commit is not already stored."""

        repository = plan.repository
        embedding_model = self.embedding_service.embedding_model_name
        chunking_version = self.chunker.chunking_version

        if self.store.is_indexed(repository, embedding_model, chunking_version):
            logger.info("Using cached corpus for %s@%s", repository.full_name, repository.commit_sha[:12])
            return repository

        files = await fetch_repo_files_for_indexing(plan)
        if not files:
            logger.warning("No files fetched for indexing: %s", repository.full_name)
            return repository

        chunks = self.chunker.chunk_repository(repository, files)
        if not chunks:
            logger.warning("No chunks generated for repository: %s", repository.full_name)
            return repository

        embeddings = await self.embedding_service.embed_documents([chunk.text for chunk in chunks])
        self.store.save_repository_artifacts(repository, files, chunks)
        self.store.replace_repository_index(
            repository=repository,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
            chunking_version=chunking_version,
        )
        logger.info(
            "Indexed %s with %d files and %d chunks",
            repository.full_name,
            len(files),
            len(chunks),
        )
        return repository
