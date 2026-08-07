"""LLM-judged reranking and diversity selection for GitHub repositories."""

from __future__ import annotations

import logging
from collections import Counter

from core.config import get_settings
from models.schemas import ExtractedKeywords, ShallowRepoEvidence
from services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_REFERENCE_TYPES = {"end_to_end", "subsystem", "pattern"}


async def rank_repo_evidence(
    idea: str,
    keywords: ExtractedKeywords,
    evidence_items: list[ShallowRepoEvidence],
) -> list[ShallowRepoEvidence]:
    """Ask the LLM to judge and select the best reference repositories."""

    if not evidence_items:
        return []

    settings = get_settings()
    llm = get_llm_client()
    candidates = [_build_digest(evidence) for evidence in evidence_items]

    try:
        judged = await llm.rank_repositories(
            idea,
            keywords.model_dump(),
            candidates,
            limit=settings.RAG_OUTPUT_REPO_LIMIT,
        )
    except Exception as exc:
        logger.warning("LLM ranking failed, falling back to relevance-score order: %s", exc)
        judged = []

    by_name = {evidence.repository.full_name: evidence for evidence in evidence_items}
    primary_capabilities = set(keywords.primary_capabilities or keywords.capabilities[:3])
    selected = _apply_judged_ranking(judged, by_name, primary_capabilities, settings.RAG_MAX_PER_LANGUAGE)

    if not selected:
        selected = _fallback_ranking(evidence_items, settings.RAG_MAX_PER_LANGUAGE)

    selected = selected[: settings.RAG_OUTPUT_REPO_LIMIT]
    logger.info("Reranked %d repositories for idea '%s'", len(selected), idea[:80])
    return selected


def _build_digest(evidence: ShallowRepoEvidence) -> dict:
    repo = evidence.repository
    return {
        "full_name": repo.full_name,
        "stars": repo.stars,
        "language": repo.language or "",
        "description": (repo.description or "")[:300],
        "readme_excerpt": evidence.readme[:800],
        "manifest_files": [file.path for file in evidence.manifest_files[:6]],
    }


def _apply_judged_ranking(
    judged: list[dict],
    by_name: dict[str, ShallowRepoEvidence],
    primary_capabilities: set[str],
    max_per_language: int,
) -> list[ShallowRepoEvidence]:
    selected: list[ShallowRepoEvidence] = []
    seen: set[str] = set()
    language_counts: Counter[str] = Counter()

    for item in judged:
        full_name = str(item.get("full_name", "")).strip()
        evidence = by_name.get(full_name)
        if evidence is None or full_name in seen:
            continue

        language = (evidence.repository.language or "Unknown").strip() or "Unknown"
        if language_counts[language] >= max_per_language:
            continue

        repo = evidence.repository
        try:
            fit_score = max(0.0, min(1.0, float(item.get("fit_score", 0.0))))
        except (TypeError, ValueError):
            fit_score = 0.0
        reference_type = item.get("reference_type")
        matched = [str(c).strip() for c in item.get("matched_capabilities", []) if str(c).strip()][:6]

        repo.fit_score = fit_score
        repo.relevance_score = fit_score
        repo.reference_type = reference_type if reference_type in _REFERENCE_TYPES else "candidate"
        repo.fit_summary = str(item.get("fit_summary", "")).strip()[:400]
        repo.covered_primary = [c for c in matched if c in primary_capabilities]
        repo.missing_primary = [c for c in primary_capabilities if c not in repo.covered_primary]
        repo.rank_reasons = [repo.fit_summary] if repo.fit_summary else []
        evidence.matched_capabilities = matched

        selected.append(evidence)
        seen.add(full_name)
        language_counts[language] += 1

    return selected


def _fallback_ranking(
    evidence_items: list[ShallowRepoEvidence],
    max_per_language: int,
) -> list[ShallowRepoEvidence]:
    """Deterministic order by existing relevance score when the LLM call fails."""

    selected: list[ShallowRepoEvidence] = []
    language_counts: Counter[str] = Counter()
    ordered = sorted(evidence_items, key=lambda evidence: evidence.repository.relevance_score, reverse=True)

    for evidence in ordered:
        language = (evidence.repository.language or "Unknown").strip() or "Unknown"
        if language_counts[language] >= max_per_language:
            continue
        repo = evidence.repository
        if not repo.fit_summary:
            repo.fit_summary = "Selected from GitHub candidate search based on metadata and query relevance."
        if repo.fit_score <= 0 and repo.relevance_score > 0:
            repo.fit_score = repo.relevance_score
        selected.append(evidence)
        language_counts[language] += 1

    return selected
