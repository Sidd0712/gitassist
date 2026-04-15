"""Hybrid retrieval over persisted repo chunks."""

from __future__ import annotations

import logging
from collections import defaultdict

from models.schemas import RepoSearchResult, RetrievalHit, RetrievalPlan
from services.rag.embedding_service import EmbeddingService
from services.rag.store_service import get_rag_store

logger = logging.getLogger(__name__)


class RetrievalService:
    """Hybrid retrieval service using dense, lexical, and metadata signals."""

    def __init__(self) -> None:
        self.store = get_rag_store()
        self.embedding_service = EmbeddingService()

    async def retrieve(
        self,
        plan: RetrievalPlan,
        repositories: list[RepoSearchResult],
    ) -> dict[str, list[RetrievalHit]]:
        """Retrieve evidence hits for each analysis section."""

        if not repositories:
            return {}

        allowed_repos = {repo.full_name: repo.commit_sha for repo in repositories}
        max_repo_score = max((repo.relevance_score for repo in repositories), default=1.0) or 1.0
        repo_priors = {
            repo.full_name: (repo.relevance_score / max_repo_score if max_repo_score else 0.0) for repo in repositories
        }

        section_hits: dict[str, list[RetrievalHit]] = {}
        for query in plan.queries:
            query_embedding = await self.embedding_service.embed_query(query.query)
            dense_hits = self.store.dense_search(query_embedding, allowed_repos, query.top_k)
            lexical_hits = self.store.lexical_search(query.query, allowed_repos, query.top_k)
            merged_hits = await self._merge_hits(
                section=query.section,
                query_text=query.query,
                dense_hits=dense_hits,
                lexical_hits=lexical_hits,
                repo_priors=repo_priors,
                preferred_roles=query.preferred_roles,
                top_k=query.top_k,
            )
            existing_hits = section_hits.get(query.section, [])
            section_hits[query.section] = self._merge_section_hits(existing_hits, merged_hits, query.top_k)

        return section_hits

    async def _merge_hits(
        self,
        *,
        section: str,
        query_text: str,
        dense_hits: list[dict],
        lexical_hits: list[dict],
        repo_priors: dict[str, float],
        preferred_roles: list[str],
        top_k: int,
    ) -> list[RetrievalHit]:
        merged: dict[str, dict] = {}

        for dense in dense_hits:
            merged[dense["chunk_id"]] = {
                **dense,
                "lexical_score": 0.0,
            }

        for lexical in lexical_hits:
            entry = merged.setdefault(
                lexical["chunk_id"],
                {
                    **lexical,
                    "dense_score": 0.0,
                },
            )
            entry["lexical_score"] = max(entry.get("lexical_score", 0.0), lexical.get("lexical_score", 0.0))
            entry.setdefault("repo_score", lexical.get("repo_score", 0.0))
            entry.setdefault("text", lexical.get("text", ""))
            entry.setdefault("path", lexical.get("path"))
            entry.setdefault("chunk_role", lexical.get("chunk_role"))
            entry.setdefault("language", lexical.get("language"))
            entry.setdefault("start_line", lexical.get("start_line"))
            entry.setdefault("end_line", lexical.get("end_line"))

        # Get AI-determined weights for this query type (cached per instance)
        if not hasattr(self, '_retrieval_weights_cache'):
            self._retrieval_weights_cache = {}
        
        cache_key = f"{section}:{query_text[:50]}"
        if cache_key not in self._retrieval_weights_cache:
            from services.llm_client import get_llm_client
            llm = get_llm_client()
            try:
                weights = await llm.determine_retrieval_weights(section, query_text)
                self._retrieval_weights_cache[cache_key] = weights
            except Exception as exc:
                logger.warning("Failed to get AI retrieval weights for %s: %s. Using defaults.", section, exc)
                self._retrieval_weights_cache[cache_key] = {
                    "dense_weight": 0.5,
                    "lexical_weight": 0.25,
                    "repo_weight": 0.15,
                    "role_weight": 0.1,
                }
        
        weights = self._retrieval_weights_cache[cache_key]
        
        scored: list[RetrievalHit] = []
        for entry in merged.values():
            repo_prior = repo_priors.get(entry["repo_full_name"], 0.0)
            role_prior = self._role_prior(entry["chunk_role"], preferred_roles, section, entry["path"])
            score = (
                weights.get("dense_weight", 0.45) * entry.get("dense_score", 0.0)
                + weights.get("lexical_weight", 0.25) * entry.get("lexical_score", 0.0)
                + weights.get("repo_weight", 0.20) * repo_prior
                + weights.get("role_weight", 0.10) * role_prior
            )
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
                    dense_score=round(entry.get("dense_score", 0.0), 5),
                    lexical_score=round(entry.get("lexical_score", 0.0), 5),
                    repo_prior=round(repo_prior, 5),
                    role_prior=round(role_prior, 5),
                    reason=self._build_reason(entry, role_prior),
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

    def _role_prior(self, chunk_role: str, preferred_roles: list[str], section: str, path: str) -> float:
        role_prior = 1.0 if chunk_role in preferred_roles else 0.25
        lowered_path = path.lower()
        if section == "architecture_diagram" and any(
            token in lowered_path for token in ("router", "route", "service", "controller", "app", "main", "server")
        ):
            role_prior = max(role_prior, 0.9)
        if section == "tech_stack" and any(
            token in lowered_path for token in ("package", "requirements", "pyproject", "docker", "compose", "cargo", "go.mod")
        ):
            role_prior = max(role_prior, 1.0)
        if section == "learning_path" and any(token in lowered_path for token in ("readme", "docs", "example", "tutorial")):
            role_prior = max(role_prior, 1.0)
        return min(role_prior, 1.0)

    def _build_reason(self, entry: dict, role_prior: float) -> str:
        reasons: list[str] = []
        if entry.get("dense_score", 0.0) >= 0.5:
            reasons.append("strong semantic match")
        if entry.get("lexical_score", 0.0) >= 0.4:
            reasons.append("strong lexical overlap")
        if role_prior >= 0.9:
            reasons.append("preferred chunk role")
        if not reasons:
            reasons.append("hybrid relevance score")
        return ", ".join(reasons)
