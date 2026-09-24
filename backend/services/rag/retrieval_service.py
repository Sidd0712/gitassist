"""Retrieval over persisted repo chunks: dense for the report, hybrid for chat."""

from __future__ import annotations

import asyncio
import logging
import posixpath
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from cachetools import TTLCache

from models.schemas import ChatMessage, RepoSearchResult, RetrievalHit, RetrievalPlan, RetrievalQuery, RetrievalWeights
from services.pipeline_cache_service import PipelineCacheService, stable_cache_key
from services.rag.embedding_service import EmbeddingService
from services.rag.store_service import get_rag_store

logger = logging.getLogger(__name__)

_cache = PipelineCacheService()
# Repo file lists change only while a repo is still indexing; a short TTL keeps
# chat from re-reading thousands of paths on every turn.
_repo_paths_cache: TTLCache = TTLCache(maxsize=256, ttl=120)


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
                    "preferred_roles": query.preferred_roles,
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

        query_embeddings = await self.embedding_service.embed_queries_batch(
            [query.query for query, _cache_key in pending_queries]
        )

        for (query, cache_key), query_embedding in zip(pending_queries, query_embeddings):
            hits = self._score_hits(
                dense_hits=self.store.dense_search(
                    query_embedding, allowed_repos, query.top_k,
                    embedding_model=self.embedding_service.embedding_model_name,
                ),
                repo_priors=repo_priors,
                top_k=query.top_k,
                weights=query.weights,
                preferred_roles=query.preferred_roles,
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
        preferred_roles: list[str] | None = None,
    ) -> list[RetrievalHit]:
        preferred_roles = preferred_roles or []
        scored: list[RetrievalHit] = []
        for entry in dense_hits:
            repo_prior = repo_priors.get(entry["repo_full_name"], 0.0)
            dense_score = entry.get("dense_score", 0.0)
            chunk_role = entry["chunk_role"]
            # Role bonus nudges a chunk toward the caller's declared preferred
            # roles (e.g. "source" for an implementation question) without
            # letting role match override genuine semantic relevance: it decays
            # from 0.15 for the most-preferred role toward ~0 for the least, and
            # is 0 for a role not in the list at all (== unchanged prior behavior).
            role_bonus = 0.0
            if preferred_roles and chunk_role in preferred_roles:
                role_rank = preferred_roles.index(chunk_role)
                role_bonus = 0.15 * (1 - role_rank / len(preferred_roles))
            score = weights.dense_weight * dense_score + weights.repo_weight * repo_prior + role_bonus
            reason = "strong semantic match" if dense_score >= 0.5 else "semantic relevance score"
            if role_bonus > 0:
                reason += f" (preferred role: {chunk_role})"
            scored.append(
                RetrievalHit(
                    chunk_id=entry["chunk_id"],
                    repo_full_name=entry["repo_full_name"],
                    path=entry["path"],
                    chunk_role=chunk_role,
                    language=entry.get("language"),
                    start_line=entry.get("start_line"),
                    end_line=entry.get("end_line"),
                    score=round(score, 5),
                    dense_score=round(dense_score, 5),
                    repo_prior=round(repo_prior, 5),
                    reason=reason,
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


# ── Chat retrieval ──────────────────────────────────────────────────────────
# Hybrid (dense + full-text) search fused with Reciprocal Rank Fusion, then
# stitched into whole-file segments so the model reads functions, not
# fragments, plus a map of every scoped repo so overview questions work and
# the model can never claim a repo "has no code".

_RRF_K = 60
_CHAT_POOL = 15  # dense_search returns ~4x this per query
_CHAT_LEXICAL_POOL = 40
_CHAT_MAX_HITS = 16
_CHAT_EXPAND_TOP = 6
_REPO_MAP_KEY_FILES = 25

_INTENT_PATTERNS = {
    "code": r"\b(code|show|implement\w*|function|class|method|logic|algorithm|snippet|how (?:does|do|is|are)|where (?:is|are|does)|handles?|works?)\b",
    "setup": r"\b(install\w*|set ?up|run(?:ning)?|deploy\w*|docker|environment|env|configur\w*|getting started|start)\b",
    "stack": r"\b(stack|dependenc\w*|librar\w*|packages?|frameworks?|built with|tech)\b",
    "compare": r"\b(compare|comparison|which repo\w*|best|differen\w*|versus|vs\.?|better)\b",
    "architecture": r"\b(architecture|structure[ds]?|organi[sz]ed|folders?|overview|design|layout|modules?)\b",
}
_INTENT_ROLES = {
    "code": ["source", "entrypoint"],
    "setup": ["documentation", "config"],
    "stack": ["config", "documentation"],
    "architecture": ["entrypoint", "source", "documentation"],
    "compare": ["documentation", "entrypoint", "source"],
}
_DEFAULT_ROLES = ["source", "entrypoint", "documentation"]
_LEXICAL_STOPWORDS = {
    "the", "and", "for", "are", "was", "how", "what", "where", "when", "which", "who", "why", "does", "did",
    "can", "could", "would", "should", "this", "that", "these", "those", "with", "from", "into", "about",
    "show", "code", "repo", "repos", "repository", "repositories", "file", "files", "use", "used", "uses",
    "using", "implement", "implemented", "implementation", "work", "works", "make", "like", "have", "has",
    "you", "your", "they", "them", "their", "there", "here", "some", "any", "all", "its", "not", "but",
    "get", "got", "let", "please", "tell", "explain", "give", "want", "need", "idea", "project", "app",
}
_FOLLOW_UP_HINT = re.compile(r"\b(it|its|this|that|they|them|those|these|there|same|also|more|else)\b", re.I)


@dataclass
class ChatEvidence:
    """What the chat model reads: stitched code segments plus a map of each repo."""

    segments: list[dict] = field(default_factory=list)
    hits: list[RetrievalHit] = field(default_factory=list)
    repo_maps: list[dict] = field(default_factory=list)
    intent: str = "general"
    targeted_repo: str | None = None


def detect_chat_intent(question: str) -> str:
    lowered = question.lower()
    for intent in ("compare", "stack", "setup", "architecture", "code"):
        if re.search(_INTENT_PATTERNS[intent], lowered):
            return intent
    return "general"


def lexical_terms(question: str) -> list[str]:
    """Identifiers and meaningful words from the question, safe for to_tsquery."""

    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question):
        term = raw.lower()
        if term in _LEXICAL_STOPWORDS:
            continue
        terms.append(term)
        # verifyToken → also "verify", "token" so prose questions still hit.
        parts = [p.lower() for p in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", raw) if len(p) > 2]
        if len(parts) > 1:
            terms.extend(p for p in parts if p not in _LEXICAL_STOPWORDS)
    return list(dict.fromkeys(terms))[:12]


def _targeted_repo(question: str, repositories: list[RepoSearchResult]) -> str | None:
    lowered = question.lower()
    for repo in repositories:
        full = repo.full_name.lower()
        short = full.split("/", 1)[-1]
        if full in lowered or (len(short) > 3 and re.search(rf"\b{re.escape(short)}\b", lowered)):
            return repo.full_name
    return None


def _mentioned_paths(question: str, repo_paths: dict[str, list[str]]) -> list[str]:
    """Indexed paths the question names, by full path or file name."""

    candidates = {m.lower().strip("`'\".,:;()") for m in re.findall(r"[\w./-]+\.\w{1,6}|[\w-]+/[\w./-]+", question)}
    if not candidates:
        return []
    found: list[str] = []
    for paths in repo_paths.values():
        for path in paths:
            lowered = path.lower()
            if lowered in candidates or posixpath.basename(lowered) in candidates:
                found.append(path)
    return list(dict.fromkeys(found))[:4]


def _stitch_segments(chunks: list[dict], scores: dict[str, float]) -> list[dict]:
    """Merge overlapping/adjacent chunks of the same file into continuous segments."""

    by_file: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_file[(chunk["repo_full_name"], chunk["path"])].append(chunk)

    segments: list[dict] = []
    for (repo, path), file_chunks in by_file.items():
        file_chunks.sort(key=lambda c: (c["start_line"] or 0))
        current: dict | None = None
        for chunk in file_chunks:
            start, end = chunk["start_line"] or 0, chunk["end_line"] or 0
            lines = chunk["text"].split("\n")
            score = scores.get(chunk["chunk_id"], 0.0)
            if current and start <= current["end_line"] + 1:
                if end > current["end_line"]:
                    skip = current["end_line"] - start + 1  # overlapping lines already present
                    current["lines"].extend(lines[max(0, skip):])
                    current["end_line"] = end
                current["score"] = max(current["score"], score)
                current["chunk_ids"].append(chunk["chunk_id"])
                continue
            current = {
                "repo_full_name": repo,
                "path": path,
                "language": chunk.get("language"),
                "start_line": start,
                "end_line": end,
                "lines": list(lines),
                "score": score,
                "chunk_ids": [chunk["chunk_id"]],
            }
            segments.append(current)

    for segment in segments:
        segment["text"] = "\n".join(segment.pop("lines"))
    segments.sort(key=lambda s: s["score"], reverse=True)
    return segments


def _repo_map(repo: RepoSearchResult, paths: list[str], readme_opening: str) -> dict:
    from services.github_service import _path_priority  # local: avoids an import cycle at module load

    top_dirs = Counter(p.split("/", 1)[0] + "/" for p in paths if "/" in p)
    key_files = sorted(paths, key=lambda p: _path_priority(p, set()), reverse=True)[:_REPO_MAP_KEY_FILES]
    return {
        "full_name": repo.full_name,
        "description": repo.description or "",
        "language": repo.language or "",
        "stars": repo.stars,
        "dependencies": repo.dependencies[:25],
        "file_count": len(paths),
        "top_level_dirs": [f"{d} ({n} files)" for d, n in top_dirs.most_common(12)],
        "root_files": [p for p in paths if "/" not in p][:20],
        "key_files": key_files,
        "readme_opening": readme_opening[:900],
    }


class ChatRetriever:
    """Hybrid retrieval tuned for open-ended questions about specific repos."""

    def __init__(self) -> None:
        self.store = get_rag_store()
        self.embedding_service = EmbeddingService()

    async def retrieve(
        self,
        question: str,
        messages: list[ChatMessage],
        repositories: list[RepoSearchResult],
        context_chars: int,
    ) -> ChatEvidence:
        intent = detect_chat_intent(question)
        targeted = _targeted_repo(question, repositories)
        scoped = [r for r in repositories if r.full_name == targeted] if targeted else repositories
        allowed = {r.full_name: r.commit_sha for r in scoped}

        # The question itself, plus (for short follow-ups like "and how is it
        # stored?") the question joined with the previous user turn.
        queries = [question.strip()]
        previous_user = next((m.content for m in reversed(messages) if m.role == "user"), None)
        if previous_user and (len(question) < 120 or _FOLLOW_UP_HINT.search(question)):
            queries.append(f"{previous_user.strip()[:300]}\n{question.strip()}")
        vectors = await self.embedding_service.embed_queries_batch(queries)

        return await asyncio.to_thread(
            self._gather, question, intent, targeted, scoped, allowed, vectors, context_chars
        )

    def _gather(
        self,
        question: str,
        intent: str,
        targeted: str | None,
        scoped: list[RepoSearchResult],
        allowed: dict[str, str],
        vectors: list[list[float]],
        context_chars: int,
    ) -> ChatEvidence:
        model = self.embedding_service.embedding_model_name
        roles = _INTENT_ROLES.get(intent, _DEFAULT_ROLES)
        wants_tests = bool(re.search(r"\b(tests?|spec|examples?|demo)\b", question.lower()))

        ranked_lists = [self.store.dense_search(v, allowed, _CHAT_POOL, embedding_model=model) for v in vectors]
        ranked_lists.append(self.store.lexical_search(lexical_terms(question), allowed, _CHAT_LEXICAL_POOL, model))

        fused: dict[str, float] = defaultdict(float)
        rows: dict[str, dict] = {}
        for weight, ranked in zip([1.0] * len(vectors) + [0.8], ranked_lists):
            for rank, row in enumerate(ranked):
                fused[row["chunk_id"]] += weight / (_RRF_K + rank + 1)
                rows.setdefault(row["chunk_id"], row)
        for chunk_id, row in rows.items():
            role = row["chunk_role"]
            if role in roles:
                fused[chunk_id] += 0.006 * (1 - roles.index(role) / len(roles))
            elif role in ("test", "example") and not wants_tests:
                fused[chunk_id] -= 0.003
            elif role == "documentation" and intent in ("code", "stack"):
                # Plans, blog posts and design docs describe code; for "show me
                # the code" questions the code itself should win.
                fused[chunk_id] -= 0.004

        ordered = sorted(rows.values(), key=lambda r: fused[r["chunk_id"]], reverse=True)
        selected = self._diversify(ordered, intent, len(allowed))

        # Files the question names go in first, whatever their similarity.
        repo_paths = self._repo_paths(allowed, model)
        named = self.store.chunks_for_paths(allowed, _mentioned_paths(question, repo_paths), model, per_path=4)
        top_score = fused[selected[0]["chunk_id"]] if selected else 1.0
        for row in named:
            fused[row["chunk_id"]] = max(fused[row["chunk_id"]], top_score + 0.01)

        neighbours = self.store.neighbor_chunks(
            [
                (r["repo_full_name"], r["commit_sha"], r["path"], r["start_line"] or 0)
                for r in selected[:_CHAT_EXPAND_TOP]
            ],
            model,
        )
        for row in neighbours:  # context only: ranked just below the hit it surrounds
            fused.setdefault(row["chunk_id"], 0.0)

        segments = _stitch_segments([*named, *selected, *neighbours], fused)
        packed, used = [], 0
        for segment in segments:
            size = len(segment["text"]) + 120
            if used + size > context_chars and packed:
                continue
            packed.append(segment)
            used += size

        readme_rows = self.store.chunks_for_paths(
            allowed,
            [p for paths in repo_paths.values() for p in paths if "/" not in p and p.lower().startswith("readme")],
            model,
            per_path=1,
        )
        readmes = {row["repo_full_name"]: row["text"] for row in readme_rows}
        repo_maps = [_repo_map(r, repo_paths.get(r.full_name, []), readmes.get(r.full_name, "")) for r in scoped]

        named_ids = {row["chunk_id"] for row in named}
        hits = [
            RetrievalHit(
                chunk_id=row["chunk_id"],
                repo_full_name=row["repo_full_name"],
                path=row["path"],
                chunk_role=row["chunk_role"],
                language=row.get("language"),
                start_line=row.get("start_line"),
                end_line=row.get("end_line"),
                score=round(fused[row["chunk_id"]], 5),
                dense_score=round(row.get("dense_score", 0.0), 5),
                reason="named in question" if row["chunk_id"] in named_ids else f"hybrid match ({intent})",
                text=row["text"],
            )
            for row in [*named, *selected]
        ]
        logger.info(
            "Chat retrieval: intent=%s target=%s dense=%s lexical=%d selected=%d segments=%d/%d chars=%d",
            intent, targeted, [len(r) for r in ranked_lists[:-1]], len(ranked_lists[-1]),
            len(selected), len(packed), len(segments), used,
        )
        return ChatEvidence(segments=packed, hits=hits, repo_maps=repo_maps, intent=intent, targeted_repo=targeted)

    def _diversify(self, ordered: list[dict], intent: str, repo_count: int) -> list[dict]:
        """Cap hits per file, and per repo unless one repo is targeted; compare questions round-robin."""

        per_repo_cap = 3 if intent == "compare" else max(6, _CHAT_MAX_HITS // max(1, repo_count) + 2)
        if repo_count <= 1:
            per_repo_cap = _CHAT_MAX_HITS
        repo_counts: Counter[str] = Counter()
        path_counts: Counter[tuple[str, str]] = Counter()
        selected: list[dict] = []
        for row in ordered:
            repo, path = row["repo_full_name"], row["path"]
            if repo_counts[repo] >= per_repo_cap or path_counts[(repo, path)] >= 3:
                continue
            selected.append(row)
            repo_counts[repo] += 1
            path_counts[(repo, path)] += 1
            if len(selected) >= _CHAT_MAX_HITS:
                break
        return selected

    def _repo_paths(self, allowed: dict[str, str], model: str) -> dict[str, list[str]]:
        missing = {name: sha for name, sha in allowed.items() if (name, sha) not in _repo_paths_cache}
        if missing:
            fetched = self.store.repo_paths(missing, model)
            for name, sha in missing.items():
                _repo_paths_cache[(name, sha)] = fetched.get(name, [])
        return {name: _repo_paths_cache[(name, sha)] for name, sha in allowed.items()}
