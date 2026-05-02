"""Research router with staged GitHub ingestion and grounded RAG synthesis."""

from __future__ import annotations

import asyncio
import json
import logging
from time import perf_counter

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.config import get_settings
from models.schemas import (
    AnalysisResponse,
    IdeaRequest,
    RepoChatEvidenceHit,
    RepoChatRequest,
    RepoChatResponse,
    RepoSearchResult,
    RepoSnapshot,
    RetrievalPlan,
)
from services.github_service import (
    build_repo_fetch_plan,
    fetch_shallow_repo_evidence_batch,
    fetch_repo_snapshot,
    search_repo_candidates,
)
from services.llm_service import (
    answer_repo_chat,
    build_clarification_questions,
    build_repo_chat_retrieval_query,
    extract_keywords,
    generate_analysis,
    plan_retrieval_queries,
)
from services.pipeline_budget import RequestBudget
from services.rag.citation_service import citations_from_hits
from services.rag.corpus_service import CorpusService
from services.rag.ranking_service import rank_repo_evidence
from services.rag.retrieval_service import RetrievalService

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
        queued_jobs = 0
        inline_plans = []

        snapshot_pairs = await _fetch_selected_snapshots(selected)
        for repository, snapshot in snapshot_pairs:
            shallow = selected_evidence_map.get(repository.full_name)
            plan = build_repo_fetch_plan(snapshot, keywords, shallow)
            cached_repository = corpus_service.load_indexed_repository(plan.repository)
            if cached_repository is not None:
                indexed_repositories.append(_merge_ranked_repository_metrics(cached_repository, plan.repository))
                continue

            if settings.INDEXING_MODE.lower() == "background":
                if len(inline_plans) < max(1, settings.RAG_INLINE_BOOTSTRAP_REPO_LIMIT):
                    inline_plans.append(plan)
                    continue
                if corpus_service.enqueue_index_job_placeholder(plan):
                    queued_jobs += 1
                continue

            inline_plans.append(plan)

        if inline_plans:
            newly_indexed = await _index_selected_plans(corpus_service, inline_plans)
            indexed_repositories.extend(newly_indexed)

        logger.info(
            "Deep index preparation completed in %.2fs; %d repos ready and %d background jobs queued",
            budget.record_stage("snapshot_and_index_stage", index_started),
            len(indexed_repositories),
            queued_jobs,
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
        if budget.has_time_for(settings.PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS):
            retrieval_plan_started = perf_counter()
            retrieval_plan = await plan_retrieval_queries(request.idea, keywords, indexed_repositories)
            logger.info("Retrieval plan completed in %.2fs", budget.record_stage("retrieval_planning", retrieval_plan_started))

            retrieval_started = perf_counter()
            retrieval_service = RetrievalService()
            section_hits = await retrieval_service.retrieve(retrieval_plan, indexed_repositories)
            logger.info("Retrieval completed in %.2fs", budget.record_stage("retrieval", retrieval_started))
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
        )
        logger.info("Grounded generation completed in %.2fs", budget.record_stage("generation", generation_started))

        for repository in analysis.repositories:
            repository.files = []

        logger.info(
            "Research completed in %.2fs with stage timings: %s",
            perf_counter() - budget.started_at,
            budget.stage_durations,
        )
        return analysis

    except Exception as exc:
        logger.exception("Research pipeline failed")
        raise HTTPException(status_code=500, detail=f"Research pipeline error: {exc}") from exc


async def _research_sse_generator(request: IdeaRequest):
    """Async generator that streams SSE events for the research pipeline."""

    settings = get_settings()
    budget = RequestBudget(total_seconds=settings.PIPELINE_REQUEST_BUDGET_SECONDS)

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

        # Step 4 — indexing
        yield _sse({"type": "progress", "message": _PROGRESS_INDEXING})
        corpus_service = CorpusService()
        index_started = perf_counter()
        indexed_repositories = []
        queued_jobs = 0
        inline_plans = []

        snapshot_pairs = await _fetch_selected_snapshots(selected)
        for repository, snapshot in snapshot_pairs:
            shallow = selected_evidence_map.get(repository.full_name)
            plan = build_repo_fetch_plan(snapshot, keywords, shallow)
            cached_repository = corpus_service.load_indexed_repository(plan.repository)
            if cached_repository is not None:
                indexed_repositories.append(_merge_ranked_repository_metrics(cached_repository, plan.repository))
                continue

            if settings.INDEXING_MODE.lower() == "background":
                if len(inline_plans) < max(1, settings.RAG_INLINE_BOOTSTRAP_REPO_LIMIT):
                    inline_plans.append(plan)
                    continue
                if corpus_service.enqueue_index_job_placeholder(plan):
                    queued_jobs += 1
                continue

            inline_plans.append(plan)

        if inline_plans:
            newly_indexed = await _index_selected_plans(corpus_service, inline_plans)
            indexed_repositories.extend(newly_indexed)

        logger.info(
            "Deep index preparation completed in %.2fs; %d repos ready and %d background jobs queued",
            budget.record_stage("snapshot_and_index_stage", index_started),
            len(indexed_repositories),
            queued_jobs,
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

        # Step 5 — retrieval
        section_hits = {}
        if budget.has_time_for(settings.PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS):
            yield _sse({"type": "progress", "message": _PROGRESS_RETRIEVING})
            retrieval_plan_started = perf_counter()
            retrieval_plan = await plan_retrieval_queries(request.idea, keywords, indexed_repositories)
            logger.info("Retrieval plan completed in %.2fs", budget.record_stage("retrieval_planning", retrieval_plan_started))

            retrieval_started = perf_counter()
            retrieval_service = RetrievalService()
            section_hits = await retrieval_service.retrieve(retrieval_plan, indexed_repositories)
            logger.info("Retrieval completed in %.2fs", budget.record_stage("retrieval", retrieval_started))
        else:
            logger.info(
                "Skipping deep retrieval because only %.2fs remain in the request budget",
                budget.remaining_seconds(),
            )

        # Step 6 — generation
        yield _sse({"type": "progress", "message": _PROGRESS_GENERATING})
        generation_started = perf_counter()
        analysis = await generate_analysis(
            request.idea,
            keywords,
            indexed_repositories,
            section_hits,
        )
        logger.info("Grounded generation completed in %.2fs", budget.record_stage("generation", generation_started))

        for repository in analysis.repositories:
            repository.files = []

        logger.info(
            "Research completed in %.2fs with stage timings: %s",
            perf_counter() - budget.started_at,
            budget.stage_durations,
        )
        yield _sse({"type": "result", "data": json.loads(analysis.model_dump_json())})

    except Exception as exc:
        logger.exception("Research pipeline failed in SSE stream")
        yield _sse({"type": "error", "message": str(exc)})


@router.post("/research")
async def research_idea_stream(request: IdeaRequest) -> StreamingResponse:
    """Stream research pipeline progress as Server-Sent Events."""
    return StreamingResponse(
        _research_sse_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/research/chat", response_model=RepoChatResponse)
async def chat_about_repositories(request: RepoChatRequest) -> RepoChatResponse:
    """Answer a grounded chat question using only the indexed repositories in scope."""

    scoped_repositories = _load_scoped_repositories(request)
    retrieval_query = build_repo_chat_retrieval_query(
        request.question,
        request.idea_summary,
        request.messages,
    )
    retrieval_service = RetrievalService()
    section_hits = await retrieval_service.retrieve(
        plan=_retrieval_query_to_plan(retrieval_query),
        repositories=scoped_repositories,
    )
    hits = section_hits.get("chat_answer", [])
    response_payload = await answer_repo_chat(
        request.question,
        request.idea_summary,
        request.messages,
        scoped_repositories,
        hits,
    )

    return RepoChatResponse(
        answer=str(response_payload.get("answer", "")).strip(),
        citations=citations_from_hits(hits, limit=6),
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
            for hit in hits[:6]
        ],
        follow_up_suggestions=[
            suggestion
            for suggestion in response_payload.get("follow_up_suggestions", [])
            if isinstance(suggestion, str) and suggestion.strip()
        ][:3],
        scoped_repo_count=len(scoped_repositories),
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


async def _index_selected_plans(corpus_service: CorpusService, plans) -> list[RepoSearchResult]:
    if not plans:
        return []

    settings = get_settings()
    semaphore = asyncio.Semaphore(max(1, min(settings.INDEXER_MAX_CONCURRENCY, len(plans))))

    async def run(plan):
        async with semaphore:
            try:
                return await corpus_service.index_repository_plan(plan)
            except Exception as exc:
                logger.warning("Deep indexing failed for %s: %s", plan.repository.full_name, exc)
                return None

    results = await asyncio.gather(*(run(plan) for plan in plans))
    return [repository for repository in results if repository is not None]


def _load_scoped_repositories(request: RepoChatRequest) -> list[RepoSearchResult]:
    if not request.scope_repositories:
        raise HTTPException(status_code=400, detail="At least one scoped repository is required for repo chat.")

    corpus_service = CorpusService()
    scoped_repositories: list[RepoSearchResult] = []
    for scoped_repo in request.scope_repositories:
        repository = RepoSearchResult(
            full_name=scoped_repo.full_name,
            commit_sha=scoped_repo.commit_sha,
            html_url=f"https://github.com/{scoped_repo.full_name}",
        )
        indexed = corpus_service.load_indexed_repository(repository)
        if indexed is not None:
            scoped_repositories.append(indexed)

    if not scoped_repositories:
        raise HTTPException(
            status_code=409,
            detail="No indexed repositories were available for the current repo chat scope.",
        )

    return scoped_repositories


def _retrieval_query_to_plan(query) -> RetrievalPlan:
    return RetrievalPlan(queries=[query])


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
    _apply_baseline_fit_metrics(merged)
    return merged


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


def _truncate_hit_text(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "GitAssist AI"}
