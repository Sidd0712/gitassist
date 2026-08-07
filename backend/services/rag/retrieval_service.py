"""Dense retrieval over persisted repo chunks."""

from __future__ import annotations

import logging
from collections import defaultdict

from models.schemas import RepoSearchResult, RetrievalHit, RetrievalPlan, RetrievalQuery, RetrievalWeights
from services.pipeline_cache_service import PipelineCacheService, stable_cache_key
from services.rag.embedding_service import EmbeddingService
from services.rag.store_service import get_rag_store

logger = logging.getLogger(__name__)

_cache = PipelineCacheService()


class RetrievalService:
    """Dense embedding retrieval service with repo-prior weighting and result diversification."""

    def __init__(self) -> None:
        self.store = get_rag_store()
        self.embedding_service = EmbeddingService()

    async def retrieve(
        self,
        plan: RetrievalPlan,
        repositories: list[RepoSearchResult],
    ) -> dict[str, list[RetrievalHit]]:
        """Retrieve evidence hits for each analysis section."""

        if not repositories or not plan.queries:
            return {}

        allowed_repos = {repo.full_name: repo.commit_sha for repo in repositories}
        max_repo_score = max((repo.relevance_score for repo in repositories), default=1.0) or 1.0
        repo_priors = {
            repo.full_name: (repo.relevance_score / max_repo_score if max_repo_score else 0.0) for repo in repositories
        }
        repo_cache_key = [(repo.full_name, repo.commit_sha) for repo in repositories]
        section_hits: dict[str, list[RetrievalHit]] = {}
        pending_queries: list[tuple[RetrievalQuery, str]] = []

        for query in plan.queries:
            cache_key = stable_cache_key(
                {
                    "repositories": repo_cache_key,
                    "section": query.section,
                    "query": query.query,
                    "top_k": query.top_k,
                    "weights": query.weights.model_dump(),
                }
            )
            cached = _cache.get_json("retrieval_hits", cache_key)
            if isinstance(cached, list):
                hits = [RetrievalHit(**item) for item in cached if isinstance(item, dict)]
                existing_hits = section_hits.get(query.section, [])
                section_hits[query.section] = self._merge_section_hits(existing_hits, hits, query.top_k)
                continue
            pending_queries.append((query, cache_key))

        if not pending_queries:
            return section_hits

        query_embeddings = await self.embedding_service.embed_documents([query.query for query, _cache_key in pending_queries])

        for (query, cache_key), query_embedding in zip(pending_queries, query_embeddings):
            hits = self._score_hits(
                dense_hits=self.store.dense_search(query_embedding, allowed_repos, query.top_k),
                repo_priors=repo_priors,
                top_k=query.top_k,
                weights=query.weights,
            )
            _cache.set_json("retrieval_hits", cache_key, [hit.model_dump() for hit in hits])
            existing_hits = section_hits.get(query.section, [])
            section_hits[query.section] = self._merge_section_hits(existing_hits, hits, query.top_k)

        self._warn_on_silent_repos(repositories, section_hits)
        return section_hits

    def _warn_on_silent_repos(
        self,
        repositories: list[RepoSearchResult],
        section_hits: dict[str, list[RetrievalHit]],
    ) -> None:
        """Flag repos that contributed zero hits across every section.

        dense_search() only ever matches rows tagged with the currently configured
        EMBEDDING_MODEL (store_service.py) — correct behavior if a repo was indexed
        under a different model/commit than expected, but it fails *silently*: the
        repo just never appears in results, indistinguishable from "genuinely no
        relevant content" without checking here. This doesn't guess which case it
        is, it just makes the silence loud enough to notice.
        """

        contributing = {hit.repo_full_name for hits in section_hits.values() for hit in hits}
        silent = [repo.full_name for repo in repositories if repo.full_name not in contributing]
        if silent:
            logger.warning(
                "Repos contributed zero retrieval hits across all sections (check embedding_model/"
                "commit_sha match in repo_indexes — this can mean the index is stale or was built "
                "under a different model than settings.EMBEDDING_MODEL): %s",
                silent,
            )

    def _score_hits(
        self,
        *,
        dense_hits: list[dict],
        repo_priors: dict[str, float],
        top_k: int,
        weights: RetrievalWeights,
    ) -> list[RetrievalHit]:
        scored: list[RetrievalHit] = []
        for entry in dense_hits:
            repo_prior = repo_priors.get(entry["repo_full_name"], 0.0)
            dense_score = entry.get("dense_score", 0.0)
            score = weights.dense_weight * dense_score + weights.repo_weight * repo_prior
            scored.append(
                RetrievalHit(
                    chunk_id=entry["chunk_id"],
                    repo_full_name=entry["repo_full_name"],
                    path=entry["path"],
                    chunk_role=entry["chunk_role"],
                    language=entry.get("language"),
                    start_line=entry.get("start_line"),
                    end_line=entry.get("end_line"),
                    score=round(score, 5),
                    dense_score=round(dense_score, 5),
                    repo_prior=round(repo_prior, 5),
                    reason="strong semantic match" if dense_score >= 0.5 else "semantic relevance score",
                    text=entry.get("text", ""),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return self._diversify(scored, top_k)

    def _diversify(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        selected: list[RetrievalHit] = []
        repo_counts: defaultdict[str, int] = defaultdict(int)
        path_counts: defaultdict[str, int] = defaultdict(int)
        per_repo_cap = max(2, top_k // 2)

        for hit in hits:
            if repo_counts[hit.repo_full_name] >= per_repo_cap:
                continue
            path_key = f"{hit.repo_full_name}:{hit.path}"
            if path_counts[path_key] >= 2:
                continue
            selected.append(hit)
            repo_counts[hit.repo_full_name] += 1
            path_counts[path_key] += 1
            if len(selected) >= top_k:
                break
        return selected

    def _merge_section_hits(
        self,
        existing_hits: list[RetrievalHit],
        new_hits: list[RetrievalHit],
        top_k: int,
    ) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {hit.chunk_id: hit for hit in existing_hits}
        for hit in new_hits:
            current = merged.get(hit.chunk_id)
            if current is None or hit.score > current.score:
                merged[hit.chunk_id] = hit
        combined = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return self._diversify(combined, max(top_k, len(existing_hits), len(new_hits)))
