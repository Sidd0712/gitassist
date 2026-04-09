"""Research router with staged GitHub ingestion and grounded RAG synthesis."""

from __future__ import annotations

import logging
from time import perf_counter

from fastapi import APIRouter, HTTPException

from core.config import get_settings
from models.schemas import AnalysisResponse, IdeaRequest, RepoSnapshot, ShallowRepoEvidence
from services.github_service import (
    build_repo_fetch_plan,
    fetch_repo_snapshot,
    fetch_shallow_repo_evidence,
    search_repo_candidates,
)
from services.llm_service import build_clarification_questions, extract_keywords, generate_analysis, plan_retrieval_queries
from services.rag.corpus_service import CorpusService
from services.rag.ranking_service import rank_repo_evidence
from services.rag.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["research"])


@router.post("/research", response_model=AnalysisResponse)
async def research_idea(request: IdeaRequest) -> AnalysisResponse:
    """Run the project-research pipeline using staged ingestion and hybrid retrieval."""

    started = perf_counter()
    try:
        logger.info("=" * 72)
        logger.info("NEW RESEARCH REQUEST: %s", request.idea[:120])
        logger.info("=" * 72)

        keyword_started = perf_counter()
        keywords = await extract_keywords(request.idea, request.clarification_answers)
        logger.info("Keyword extraction completed in %.2fs", perf_counter() - keyword_started)

        clarification_questions = build_clarification_questions(keywords)
        if clarification_questions and not request.clarification_answers:
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
        logger.info("Candidate search completed in %.2fs with %d repos", perf_counter() - search_started, len(candidates))
        if not candidates:
            analysis = await generate_analysis(request.idea, keywords, [], {})
            analysis.repo_descriptions = ["No strong GitHub reference repositories were found for this idea yet."]
            return AnalysisResponse(
                **analysis.model_dump(),
            )

        ingest_started = perf_counter()
        snapshots: dict[str, RepoSnapshot] = {}
        evidence_items: list[ShallowRepoEvidence] = []
        for candidate in candidates:
            try:
                snapshot = await fetch_repo_snapshot(candidate)
                evidence = await fetch_shallow_repo_evidence(snapshot)
                snapshots[snapshot.full_name] = snapshot
                evidence_items.append(evidence)
            except Exception as exc:
                logger.warning("Skipping candidate %s during shallow ingest: %s", candidate.full_name, exc)
        logger.info("Shallow ingest completed in %.2fs", perf_counter() - ingest_started)

        if not evidence_items:
            analysis = await generate_analysis(request.idea, keywords, [], {})
            analysis.status = "error"
            analysis.error = "GitHub references could not be ingested successfully for this idea."
            analysis.repo_descriptions = ["GitHub references were unavailable, so the recommendations below are idea-first without repo support."]
            return AnalysisResponse(
                **analysis.model_dump(),
            )

        ranking_started = perf_counter()
        ranked = rank_repo_evidence(request.idea, keywords, evidence_items)
        settings = get_settings()
        selected = ranked[: settings.RAG_DEEP_INDEX_REPO_LIMIT]
        logger.info(
            "Reranking completed in %.2fs; selected %d repos for deep indexing",
            perf_counter() - ranking_started,
            len(selected),
        )

        index_started = perf_counter()
        corpus_service = CorpusService()
        selected_repositories = []
        for evidence in selected:
            snapshot = snapshots[evidence.repository.full_name]
            plan = build_repo_fetch_plan(snapshot, evidence, keywords)
            repository = await corpus_service.ensure_indexed(plan)
            selected_repositories.append(repository)
        logger.info("Corpus indexing completed in %.2fs", perf_counter() - index_started)

        retrieval_started = perf_counter()
        retrieval_plan = await plan_retrieval_queries(request.idea, keywords, selected_repositories)
        retrieval_service = RetrievalService()
        section_hits = await retrieval_service.retrieve(retrieval_plan, selected_repositories)
        logger.info("Retrieval completed in %.2fs", perf_counter() - retrieval_started)

        generation_started = perf_counter()
        analysis = await generate_analysis(request.idea, keywords, selected_repositories, section_hits)
        logger.info("Grounded generation completed in %.2fs", perf_counter() - generation_started)

        for repository in analysis.repositories:
            repository.files = []

        logger.info("Research completed in %.2fs", perf_counter() - started)
        return analysis

    except Exception as exc:
        logger.exception("Research pipeline failed")
        raise HTTPException(status_code=500, detail=f"Research pipeline error: {exc}") from exc


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "GitAssist AI"}
