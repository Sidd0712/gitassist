"""Helpers for turning retrieval hits into response citations."""

from __future__ import annotations

from models.schemas import AnalysisEvidence, Citation, RetrievalHit


def citations_from_hits(hits: list[RetrievalHit], limit: int = 4) -> list[Citation]:
    """Build deduplicated citations from retrieval hits."""

    citations: list[Citation] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()

    for hit in hits:
        key = (hit.repo_full_name, hit.path, hit.start_line, hit.end_line)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                repo_full_name=hit.repo_full_name,
                path=hit.path,
                start_line=hit.start_line,
                end_line=hit.end_line,
                reason=hit.reason,
            )
        )
        if len(citations) >= limit:
            break
    return citations


def build_analysis_evidence(section_hits: dict[str, list[RetrievalHit]]) -> AnalysisEvidence:
    """Create grouped evidence for the final response."""

    return AnalysisEvidence(
        repo_descriptions=citations_from_hits(section_hits.get("repo_descriptions", [])),
        learning_path=citations_from_hits(section_hits.get("learning_path", [])),
        architecture_diagram=citations_from_hits(section_hits.get("architecture_diagram", [])),
        tech_stack=citations_from_hits(section_hits.get("tech_stack", [])),
    )
