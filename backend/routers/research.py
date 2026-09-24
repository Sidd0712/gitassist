"""Research router with staged GitHub ingestion and grounded RAG synthesis."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections import defaultdict
from time import perf_counter

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.config import get_settings
from models.schemas import (
    AnalysisResponse,
    Citation,
    IdeaRequest,
    IndexStatusRequest,
    IndexStatusResponse,
    RepoChatEvidenceHit,
    RepoChatRequest,
    RepoChatResponse,
    RepoFetchPlan,
    RepoIndexStatus,
    RepoSearchResult,
    RepoSnapshot,
)
from services.github_service import (
    build_repo_fetch_plan,
    fetch_shallow_repo_evidence_batch,
    fetch_repo_snapshot,
    path_terms_for_intent,
    search_repo_candidates,
)
from services.llm_service import (
    answer_repo_chat,
    build_clarification_questions,
    extract_keywords,
    generate_analysis,
    plan_retrieval_queries,
)
from services.pipeline_budget import RequestBudget
from services.rag.corpus_service import CorpusService
from services.rag.embedding_service import EmbeddingUnavailable
from services.rag.ranking_service import rank_repo_evidence
from services.rag.retrieval_service import ChatRetriever, RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["research"])

_PROGRESS_EXTRACTING = "Extracting intent from your idea..."
_PROGRESS_SEARCHING = "Searching GitHub for relevant repositories..."
_PROGRESS_ANALYSING = "Analysing top candidates..."
_PROGRESS_INDEXING = "Indexing real-world code (this takes a moment)..."
_PROGRESS_RETRIEVING = "Retrieving relevant code sections..."
_PROGRESS_GENERATING = "Generating your research report..."


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ── Auth + rate limiting ────────────────────────────────────────────────────
# Single-operator tool: a shared API key gate plus a simple per-IP rolling-hour
# counter is enough to stop a leaked URL from running up GitHub/Groq/Cohere
# usage. Neither is meant to hold up under multi-instance/multi-tenant load.

_rate_limit_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _rate_limiter(bucket_name: str, limit_setting: str):
    """Per-IP rolling-hour limit, with a separate bucket per route family."""

    def enforce(http_request: Request) -> None:
        limit = getattr(get_settings(), limit_setting)
        client_ip = http_request.client.host if http_request.client else "unknown"
        now = time.monotonic()
        bucket = _rate_limit_buckets[(bucket_name, client_ip)]
        while bucket and bucket[0] < now - 3600:
            bucket.pop(0)
        if len(bucket) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        bucket.append(now)

    return enforce


_guarded = [Depends(_require_api_key), Depends(_rate_limiter("research", "RATE_LIMIT_PER_HOUR"))]
_chat_guarded = [Depends(_require_api_key), Depends(_rate_limiter("chat", "CHAT_RATE_LIMIT_PER_HOUR"))]


# ── Whole-repo indexing is queued, never run here ───────────────────────────
# Render has 512 MB; one 604 MB repo zip OOM-killed it mid-request. Repos are
# now queued in Postgres (repo_index_jobs) and indexed by the worker on the
# developer's machine. The report never waits: repos still being indexed are
# grounded by their shallow-rerank evidence, and chat picks them up as soon as
# the worker marks them 'partial'.


def _queue_for_indexing(
    corpus_service: CorpusService,
    plan: RepoFetchPlan,
    keywords,
    shallow,
) -> None:
    try:
        corpus_service.enqueue(plan.repository, path_terms_for_intent(keywords, shallow.matched_capabilities if shallow else None))
    except Exception:
        # A queueing hiccup must never fail the report; the next request re-queues.
        logger.exception("Could not queue %s for indexing", plan.repository.full_name)


async def research_idea(request: IdeaRequest) -> AnalysisResponse:
    """Plain async helper — run the full research pipeline and return AnalysisResponse.

    Unit tests call this directly so the signature and return type must stay stable.
    """

    settings = get_settings()
    budget = RequestBudget(total_seconds=settings.PIPELINE_REQUEST_BUDGET_SECONDS)

    try:
        logger.info("=" * 72)
        logger.info("NEW RESEARCH REQUEST: %s", request.idea[:120])
        logger.info("=" * 72)

        keyword_started = perf_counter()
        keywords = await extract_keywords(request.idea, request.clarification_answers)
        logger.info("Keyword extraction completed in %.2fs", budget.record_stage("keyword_extraction", keyword_started))

        clarification_questions = build_clarification_questions(keywords)
        has_meaningful_answers = bool(
            request.clarification_answers and any(value and value.strip() for value in request.clarification_answers.values())
        )
        if clarification_questions and not has_meaningful_answers:
            logger.info("Returning %d clarification questions before repository search", len(clarification_questions))
            return AnalysisResponse(
                idea_summary=keywords.summary or request.idea,
                keywords=keywords,
                clarification_questions=clarification_questions,
                assumptions=keywords.assumptions,
                status="needs_clarification",
            )

        search_started = perf_counter()
        candidates = await search_repo_candidates(keywords)
        for candidate in candidates:
            _apply_baseline_fit_metrics(candidate)
        logger.info(
            "Candidate search completed in %.2fs with %d repos",
            budget.record_stage("candidate_search", search_started),
            len(candidates),
        )
        if not candidates:
            analysis = await generate_analysis(request.idea, keywords, [], {})
            analysis.repo_descriptions = ["No strong GitHub reference repositories were found for this idea yet."]
            logger.info("Research completed in %.2fs (no candidates)", perf_counter() - budget.started_at)
            return analysis

        rerank_started = perf_counter()
        rerank_pool = candidates[: settings.RAG_README_RERANK_LIMIT]
        shallow_evidence = await fetch_shallow_repo_evidence_batch(rerank_pool, keywords)
        ranked_evidence = await rank_repo_evidence(request.idea, keywords, shallow_evidence)
        logger.info(
            "Shallow rerank completed in %.2fs for %d repositories",
            budget.record_stage("shallow_rerank", rerank_started),
            len(ranked_evidence),
        )

        selected = [evidence.repository for evidence in ranked_evidence[: settings.RAG_DEEP_INDEX_REPO_LIMIT]]
        selected_evidence_map = {evidence.repository.full_name: evidence for evidence in ranked_evidence}
        if not selected:
            selected = candidates[: settings.RAG_DEEP_INDEX_REPO_LIMIT]
        logger.info(
            "Selected %d repositories for deep analysis from %d search candidates",
            len(selected),
            len(candidates),
        )

        corpus_service = CorpusService()
        index_started = perf_counter()
        indexed_repositories = []
        cache_hit_repositories = []
        inline_plans = []

        snapshot_pairs = await _fetch_selected_snapshots(selected)
        for repository, snapshot in snapshot_pairs:
            shallow = selected_evidence_map.get(repository.full_name)
            plan = build_repo_fetch_plan(snapshot, keywords, shallow)
            if _skip_if_not_indexable(plan):
                continue
            cached_repository = corpus_service.load_indexed_repository(plan.repository)
            if cached_repository is not None:
                merged = _merge_ranked_repository_metrics(cached_repository, plan.repository)
                indexed_repositories.append(merged)
                cache_hit_repositories.append(merged)
                continue

            _queue_for_indexing(corpus_service, plan, keywords, shallow)
            inline_plans.append(plan)
            # Still reported now, grounded by shallow-rerank evidence.
            indexed_repositories.append(plan.repository)

        logger.info(
            "Deep index preparation completed in %.2fs; %d repos ready (%d queued for the indexing worker)",
            budget.record_stage("snapshot_and_index_stage", index_started),
            len(indexed_repositories),
            len(inline_plans),
        )

        if not indexed_repositories:
            analysis = await generate_analysis(request.idea, keywords, [], {})
            analysis.status = "error"
            analysis.error = "Deep repository indexing did not complete for any shortlisted repository."
            analysis.repo_descriptions = [
                "Deep repository analysis could not finish for the shortlisted repositories. Try again or reduce the scope of the idea."
            ]
            logger.info("Research completed in %.2fs (no deep repos ready)", perf_counter() - budget.started_at)
            return analysis

        section_hits = {}
        # Only repos already indexed under this exact commit/model/chunking-version
        # combo (cache hits) have any chunks to search yet — freshly-dispatched
        # repos are still embedding in the background. Skip retrieval entirely
        # when there's nothing to retrieve against; it would just pay the
        # embedding-provider dispatch cost for guaranteed zero hits.
        if cache_hit_repositories and budget.has_time_for(settings.PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS):
            retrieval_plan_started = perf_counter()
            retrieval_plan = await plan_retrieval_queries(request.idea, keywords, cache_hit_repositories)
            logger.info("Retrieval plan completed in %.2fs", budget.record_stage("retrieval_planning", retrieval_plan_started))

            retrieval_started = perf_counter()
            retrieval_service = RetrievalService()
            try:
                section_hits = await retrieval_service.retrieve(retrieval_plan, cache_hit_repositories)
            except EmbeddingUnavailable as exc:
                logger.warning("Skipping retrieval, embeddings unavailable: %s", exc)
            logger.info("Retrieval completed in %.2fs", budget.record_stage("retrieval", retrieval_started))
        elif not cache_hit_repositories:
            logger.info("Skipping retrieval: all %d selected repos are still indexing in background", len(inline_plans))
        else:
            logger.info(
                "Skipping deep retrieval because only %.2fs remain in the request budget",
                budget.remaining_seconds(),
            )

        generation_started = perf_counter()
        analysis = await generate_analysis(
            request.idea,
            keywords,
            indexed_repositories,
            section_hits,
            repo_context=_repo_context(selected_evidence_map),
        )
        logger.info("Grounded generation completed in %.2fs", budget.record_stage("generation", generation_started))

        for repository in analysis.repositories:
            repository.files = []

        if analysis.status == "complete" and not analysis.error:
            analysis.error = _low_confidence_caveat(analysis.repositories)

        logger.info(
            "Research completed in %.2fs with stage timings: %s",
            perf_counter() - budget.started_at,
            budget.stage_durations,
        )
        return analysis

    except Exception as exc:
        logger.exception("Research pipeline failed")
        raise HTTPException(status_code=500, detail=f"Research pipeline error: {exc}") from exc


async def _research_sse_generator(request: IdeaRequest, http_request: Request):
    """Async generator that streams SSE events for the research pipeline."""

    settings = get_settings()
    budget = RequestBudget(total_seconds=settings.PIPELINE_REQUEST_BUDGET_SECONDS)

    async def _client_gone() -> bool:
        if await http_request.is_disconnected():
            logger.info("Client disconnected mid-stream; stopping research pipeline early")
            return True
        return False

    try:
        logger.info("=" * 72)
        logger.info("NEW RESEARCH REQUEST (SSE): %s", request.idea[:120])
        logger.info("=" * 72)

        # ── SILENT PREFLIGHT ─────────────────────────────────────────────────
        keyword_started = perf_counter()
        keywords = await extract_keywords(request.idea, request.clarification_answers)
        logger.info("Keyword extraction completed in %.2fs", budget.record_stage("keyword_extraction", keyword_started))

        clarification_questions = build_clarification_questions(keywords)
        has_meaningful_answers = bool(
            request.clarification_answers and any(value and value.strip() for value in request.clarification_answers.values())
        )
        if clarification_questions and not has_meaningful_answers:
            logger.info("Returning %d clarification questions (SSE result event)", len(clarification_questions))
            result = AnalysisResponse(
                idea_summary=keywords.summary or request.idea,
                keywords=keywords,
                clarification_questions=clarification_questions,
                assumptions=keywords.assumptions,
                status="needs_clarification",
            )
            yield _sse({"type": "result", "data": json.loads(result.model_dump_json())})
            return

        if await _client_gone():
            return

        # ── MAIN PIPELINE ─────────────────────────────────────────────────────
        # Step 1 — already done during preflight; emit retroactively
        yield _sse({"type": "progress", "message": _PROGRESS_EXTRACTING})

        # Step 2 — search
        yield _sse({"type": "progress", "message": _PROGRESS_SEARCHING})
        search_started = perf_counter()
        candidates = await search_repo_candidates(keywords)
        for candidate in candidates:
            _apply_baseline_fit_metrics(candidate)
        logger.info(
            "Candidate search completed in %.2fs with %d repos",
            budget.record_stage("candidate_search", search_started),
            len(candidates),
        )

        if not candidates:
            yield _sse({"type": "progress", "message": _PROGRESS_GENERATING})
            analysis = await generate_analysis(request.idea, keywords, [], {})
            analysis.repo_descriptions = ["No strong GitHub reference repositories were found for this idea yet."]
            logger.info("Research completed in %.2fs (no candidates, SSE)", perf_counter() - budget.started_at)
            yield _sse({"type": "result", "data": json.loads(analysis.model_dump_json())})
            return

        if await _client_gone():
            return

        # Step 3 — rerank / analyse candidates
        yield _sse({"type": "progress", "message": _PROGRESS_ANALYSING})
        rerank_started = perf_counter()
        rerank_pool = candidates[: settings.RAG_README_RERANK_LIMIT]
        shallow_evidence = await fetch_shallow_repo_evidence_batch(rerank_pool, keywords)
        ranked_evidence = await rank_repo_evidence(request.idea, keywords, shallow_evidence)
        logger.info(
            "Shallow rerank completed in %.2fs for %d repositories",
            budget.record_stage("shallow_rerank", rerank_started),
            len(ranked_evidence),
        )

        selected = [evidence.repository for evidence in ranked_evidence[: settings.RAG_DEEP_INDEX_REPO_LIMIT]]
        selected_evidence_map = {evidence.repository.full_name: evidence for evidence in ranked_evidence}
        if not selected:
            selected = candidates[: settings.RAG_DEEP_INDEX_REPO_LIMIT]

        if await _client_gone():
            return

        # Step 4 — indexing
        yield _sse({"type": "progress", "message": _PROGRESS_INDEXING})
        corpus_service = CorpusService()
        index_started = perf_counter()
        indexed_repositories = []
        cache_hit_repositories = []
        inline_plans = []

        snapshot_pairs = await _fetch_selected_snapshots(selected)
        for repository, snapshot in snapshot_pairs:
            shallow = selected_evidence_map.get(repository.full_name)
            plan = build_repo_fetch_plan(snapshot, keywords, shallow)
            if _skip_if_not_indexable(plan):
                continue
            cached_repository = corpus_service.load_indexed_repository(plan.repository)
            if cached_repository is not None:
                merged = _merge_ranked_repository_metrics(cached_repository, plan.repository)
                indexed_repositories.append(merged)
                cache_hit_repositories.append(merged)
                continue

            _queue_for_indexing(corpus_service, plan, keywords, shallow)
            inline_plans.append(plan)
            indexed_repositories.append(plan.repository)

        logger.info(
            "Deep index preparation completed in %.2fs; %d repos ready (%d queued for the indexing worker)",
            budget.record_stage("snapshot_and_index_stage", index_started),
            len(indexed_repositories),
            len(inline_plans),
        )

        if not indexed_repositories:
            analysis = await generate_analysis(request.idea, keywords, [], {})
            analysis.status = "error"
            analysis.error = "Deep repository indexing did not complete for any shortlisted repository."
            analysis.repo_descriptions = [
                "Deep repository analysis could not finish for the shortlisted repositories. Try again or reduce the scope of the idea."
            ]
            logger.info("Research completed in %.2fs (no deep repos ready, SSE)", perf_counter() - budget.started_at)
            yield _sse({"type": "result", "data": json.loads(analysis.model_dump_json())})
            return

        if await _client_gone():
            return

        # Step 5 — retrieval
        section_hits = {}
        # Same reasoning as the plain research_idea() path: only cache-hit repos
        # have chunks to search yet, so skip retrieval entirely rather than pay
        # its dispatch cost for guaranteed zero hits against freshly-dispatched repos.
        if cache_hit_repositories and budget.has_time_for(settings.PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS):
            yield _sse({"type": "progress", "message": _PROGRESS_RETRIEVING})
            retrieval_plan_started = perf_counter()
            retrieval_plan = await plan_retrieval_queries(request.idea, keywords, cache_hit_repositories)
            logger.info("Retrieval plan completed in %.2fs", budget.record_stage("retrieval_planning", retrieval_plan_started))

            retrieval_started = perf_counter()
            retrieval_service = RetrievalService()
            try:
                section_hits = await retrieval_service.retrieve(retrieval_plan, cache_hit_repositories)
            except EmbeddingUnavailable as exc:
                logger.warning("Skipping retrieval, embeddings unavailable: %s", exc)
            logger.info("Retrieval completed in %.2fs", budget.record_stage("retrieval", retrieval_started))
        elif not cache_hit_repositories:
            logger.info("Skipping retrieval: all %d selected repos are still indexing in background", len(inline_plans))
        else:
            logger.info(
                "Skipping deep retrieval because only %.2fs remain in the request budget",
                budget.remaining_seconds(),
            )

        if await _client_gone():
            return

        # Step 6 — generation
        yield _sse({"type": "progress", "message": _PROGRESS_GENERATING})
        generation_started = perf_counter()
        analysis = await generate_analysis(
            request.idea,
            keywords,
            indexed_repositories,
            section_hits,
            repo_context=_repo_context(selected_evidence_map),
        )
        logger.info("Grounded generation completed in %.2fs", budget.record_stage("generation", generation_started))

        for repository in analysis.repositories:
            repository.files = []

        if analysis.status == "complete" and not analysis.error:
            analysis.error = _low_confidence_caveat(analysis.repositories)

        logger.info(
            "Research completed in %.2fs with stage timings: %s",
            perf_counter() - budget.started_at,
            budget.stage_durations,
        )
        yield _sse({"type": "result", "data": json.loads(analysis.model_dump_json())})

    except Exception as exc:
        logger.exception("Research pipeline failed in SSE stream")
        yield _sse({"type": "error", "message": str(exc)})


@router.post("/research", dependencies=_guarded)
async def research_idea_stream(request: IdeaRequest, http_request: Request) -> StreamingResponse:
    """Stream research pipeline progress as Server-Sent Events."""
    return StreamingResponse(
        _research_sse_generator(request, http_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/research/chat", response_model=RepoChatResponse, dependencies=_chat_guarded)
async def chat_about_repositories(request: RepoChatRequest) -> RepoChatResponse:
    """Answer a chat question grounded in the scoped repos' indexed code."""

    settings = get_settings()
    scoped_repositories = _load_scoped_repositories(request)
    try:
        evidence = await ChatRetriever().retrieve(
            request.question,
            request.messages,
            scoped_repositories,
            context_chars=settings.CHAT_CONTEXT_CHARS,
        )
    except EmbeddingUnavailable as exc:
        logger.warning("Repo chat unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Repo chat is temporarily unavailable because the code-search worker is offline. Please try again later.",
        ) from exc

    response_payload = await answer_repo_chat(request.question, request.idea_summary, request.messages, evidence)
    commits = {repo.full_name: repo.commit_sha for repo in scoped_repositories}
    answer_text = str(response_payload.get("answer", ""))
    retrieved = [segment for segment in evidence.segments if segment["repo_full_name"] in commits]
    # Sources = the files the answer actually cites; fall back to the top
    # retrieved sections when it cites none (e.g. an overview answer).
    cited = [segment for segment in retrieved if segment["path"] in answer_text]
    sources = (cited or retrieved[:4])[:8]

    return RepoChatResponse(
        answer=str(response_payload.get("answer", "")).strip(),
        citations=[
            Citation(
                repo_full_name=segment["repo_full_name"],
                path=segment["path"],
                start_line=segment["start_line"],
                end_line=segment["end_line"],
                reason="retrieved for this answer",
                commit_sha=commits[segment["repo_full_name"]],
            )
            for segment in sources
        ],
        evidence_hits=[
            RepoChatEvidenceHit(
                repo_full_name=hit.repo_full_name,
                path=hit.path,
                start_line=hit.start_line,
                end_line=hit.end_line,
                reason=hit.reason,
                snippet=_truncate_hit_text(hit.text),
                score=round(hit.score, 5),
            )
            for hit in evidence.hits[:6]
        ],
        follow_up_suggestions=response_payload.get("follow_up_suggestions", [])[:3],
        scoped_repo_count=len(scoped_repositories),
    )


@router.post("/research/index-status", response_model=IndexStatusResponse, dependencies=[Depends(_require_api_key)])
async def index_status(request: IndexStatusRequest) -> IndexStatusResponse:
    """Per-repo indexing progress, polled by the chat panel (not rate-limited)."""

    settings = get_settings()
    corpus_service = CorpusService()
    repos = [(repo.full_name, repo.commit_sha) for repo in request.repositories[:10]]
    statuses = await asyncio.to_thread(corpus_service.index_status, repos)
    worker_online = await asyncio.to_thread(
        corpus_service.store.embedding_worker_alive, settings.LOCAL_WORKER_HEARTBEAT_MAX_AGE_SECONDS
    )
    return IndexStatusResponse(
        worker_online=worker_online,
        repositories=[
            RepoIndexStatus(full_name=name, **statuses.get(name, {"state": "not_queued", "chunk_count": 0}))
            for name, _sha in repos
        ],
    )


async def _fetch_selected_snapshots(
    selected: list[RepoSearchResult],
) -> list[tuple[RepoSearchResult, RepoSnapshot]]:
    if not selected:
        return []

    snapshots = await asyncio.gather(*(fetch_repo_snapshot(repository) for repository in selected), return_exceptions=True)
    pairs: list[tuple[RepoSearchResult, RepoSnapshot]] = []
    for repository, snapshot in zip(selected, snapshots):
        if isinstance(snapshot, Exception):
            logger.warning("Skipping snapshot for %s: %s", repository.full_name, snapshot)
            continue
        pairs.append((repository, snapshot))
    return pairs


def _load_scoped_repositories(request: RepoChatRequest) -> list[RepoSearchResult]:
    if not request.scope_repositories:
        raise HTTPException(status_code=400, detail="At least one scoped repository is required for repo chat.")

    corpus_service = CorpusService()
    scoped_repositories = corpus_service.store.load_repositories(
        [(repo.full_name, repo.commit_sha) for repo in request.scope_repositories],
        corpus_service.embedding_model_name,
        corpus_service.chunking_version,
    )
    if not scoped_repositories:
        raise HTTPException(
            status_code=409,
            detail="None of these repositories are searchable yet. They are queued for indexing; try again shortly.",
        )
    return scoped_repositories


def _repo_context(evidence_map: dict) -> dict[str, dict]:
    """README excerpt and known file paths per repo, from shallow rerank evidence."""

    return {
        name: {
            "readme_excerpt": evidence.readme[:1500],
            "paths": list(dict.fromkeys(
                [*evidence.highlighted_paths, *evidence.sampled_paths, *(f.path for f in evidence.manifest_files)]
            )),
        }
        for name, evidence in evidence_map.items()
    }


def _merge_ranked_repository_metrics(repository: RepoSearchResult, ranked_repository: RepoSearchResult) -> RepoSearchResult:
    merged = repository.model_copy(deep=True)
    merged.description = ranked_repository.description or merged.description
    merged.html_url = ranked_repository.html_url or merged.html_url
    merged.language = ranked_repository.language or merged.language
    merged.stars = ranked_repository.stars or merged.stars
    merged.topics = ranked_repository.topics or merged.topics
    merged.updated_at = ranked_repository.updated_at or merged.updated_at
    merged.archived = ranked_repository.archived
    merged.default_branch = ranked_repository.default_branch or merged.default_branch
    merged.commit_sha = ranked_repository.commit_sha or merged.commit_sha
    merged.relevance_score = ranked_repository.relevance_score
    merged.semantic_meta_score = ranked_repository.semantic_meta_score
    merged.semantic_readme_score = ranked_repository.semantic_readme_score
    merged.query_hit_count = ranked_repository.query_hit_count
    merged.rank_reasons = ranked_repository.rank_reasons[:]
    merged.reference_type = ranked_repository.reference_type
    merged.fit_score = ranked_repository.fit_score
    merged.fit_summary = ranked_repository.fit_summary
    merged.covered_primary = ranked_repository.covered_primary[:]
    merged.missing_primary = ranked_repository.missing_primary[:]
    merged.dependencies = ranked_repository.dependencies[:] or merged.dependencies
    _apply_baseline_fit_metrics(merged)
    return merged


def _skip_if_not_indexable(plan: RepoFetchPlan) -> bool:
    """True if this repo has no index-eligible files and should be skipped entirely.

    A repo with zero selected_paths (an empty or placeholder repo that only
    matched on weak metadata similarity) would otherwise be handed to the
    indexer, produce an empty index, and -- if it was the only repo selected --
    make the whole request error out with no indication of why.
    """
    if plan.selected_paths:
        return False
    logger.warning(
        "Skipping %s: no index-eligible files found (empty or placeholder repo)",
        plan.repository.full_name,
    )
    return True


def _apply_baseline_fit_metrics(repository: RepoSearchResult) -> None:
    if repository.fit_score <= 0 and repository.relevance_score > 0:
        repository.fit_score = round(repository.relevance_score, 4)
    if repository.reference_type == "candidate" and repository.fit_score > 0 and not repository.fit_summary:
        if repository.query_hit_count and repository.semantic_meta_score > 0:
            repository.fit_summary = (
                "Selected from GitHub candidate search with strong metadata alignment "
                f"({repository.semantic_meta_score:.2f}) across {repository.query_hit_count} query families."
            )
        else:
            repository.fit_summary = "Selected from GitHub candidate search based on metadata and query relevance."


# Below this, a repo's fit score reflects "the best of a weak pool," not a
# genuine match -- GitHub's metadata-only search can come back thin for niche
# ideas whose vocabulary doesn't match how real matching repos describe
# themselves. Chosen as a floor clearly below the percentile-relative scores
# normal matches land at, not tuned against a specific dataset.
_MIN_CONFIDENT_FIT_SCORE = 0.35


def _low_confidence_caveat(repositories: list[RepoSearchResult]) -> str | None:
    """Return a user-facing caveat if the whole repo pool is a weak match.

    Percentile-relative scoring always crowns a "best" candidate even when
    every candidate is a poor fit -- this catches that case with an absolute
    floor instead, so a niche idea gets an honest "we didn't find strong
    references" instead of a confidently-labeled weak match.
    """
    if not repositories:
        return None
    best_fit = max((repo.fit_score or repo.relevance_score) for repo in repositories)
    if best_fit >= _MIN_CONFIDENT_FIT_SCORE:
        return None
    return (
        "The repositories found are a weak match for this idea (GitHub's search "
        "only looks at repo names/descriptions/READMEs, not code content, so niche "
        "or novel ideas can come back thin). Treat the results below as loose "
        "inspiration rather than close references, or try rephrasing the idea with "
        "more specific technical terms."
    )


def _truncate_hit_text(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "GitAssist AI"}
