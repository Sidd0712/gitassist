"""Semantic reranking, deduplication, and diversity selection for GitHub repositories."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from core.config import get_settings
from models.schemas import ExtractedKeywords, ShallowRepoEvidence
from services.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

LOW_VALUE_PATTERNS = (
    "awesome-",
    "boilerplate",
    "challenge",
    "cheatsheet",
    "course",
    "exercise",
    "leetcode",
    "roadmap",
    "scaffold",
    "starter",
    "template",
    "test",
    "tutorial",
)


async def rank_repo_evidence(
    idea: str,
    keywords: ExtractedKeywords,
    evidence_items: list[ShallowRepoEvidence],
) -> list[ShallowRepoEvidence]:
    """Score, deduplicate, diversify, and select repositories by idea fit."""

    if not evidence_items:
        return []

    settings = get_settings()
    embedding_service = EmbeddingService()
    intent_text = _intent_text(keywords, idea)
    intent_embedding = await embedding_service.embed_query(intent_text)
    repo_texts = [_repo_text(evidence) for evidence in evidence_items]
    repo_embeddings = await embedding_service.embed_documents(repo_texts)
    max_query_hits = max((evidence.repository.query_hit_count for evidence in evidence_items), default=1)

    for evidence, repo_embedding, repo_text in zip(evidence_items, repo_embeddings, repo_texts):
        repo = evidence.repository
        lowered = repo_text.lower()
        primary_capabilities = keywords.primary_capabilities or keywords.capabilities[:3]
        secondary_capabilities = keywords.secondary_capabilities or keywords.capabilities[3:6]
        trivial_capabilities = keywords.trivial_capabilities or []

        covered_primary = _matched_capabilities(primary_capabilities, lowered)
        covered_secondary = _matched_capabilities(secondary_capabilities, lowered)
        matched_trivial = _matched_capabilities(trivial_capabilities, lowered)
        missing_primary = [capability for capability in primary_capabilities if capability not in covered_primary]

        semantic_readme_score = _cosine_similarity(intent_embedding, repo_embedding)
        weighted_coverage = _weighted_coverage(keywords, lowered)
        semantic_meta_score = repo.semantic_meta_score
        query_diversity_score = repo.query_hit_count / max(max_query_hits, 1)
        doc_quality = _doc_quality_score(evidence.readme, evidence.manifest_files)
        quality_score = _repo_quality_score(repo)
        low_value_penalty = _low_value_penalty(repo.full_name, repo.description or "", evidence.readme)
        scope_penalty = _scope_mismatch_penalty(keywords, lowered)

        final_score = (
            0.45 * semantic_readme_score
            + 0.20 * semantic_meta_score
            + 0.20 * weighted_coverage
            + 0.07 * doc_quality
            + 0.03 * query_diversity_score
            + 0.03 * quality_score
            - low_value_penalty
            - scope_penalty
        )

        evidence.matched_capabilities = _dedupe_preserve([*covered_primary, *covered_secondary, *matched_trivial])
        evidence.matched_keywords = _matched_keywords(keywords.domain_terms or keywords.keywords, lowered)
        evidence.matched_frameworks = _matched_keywords(keywords.frameworks, lowered)
        evidence.matched_stack_families = _matched_keywords(keywords.tech_terms or keywords.likely_stack_families, lowered)
        evidence.capability_coverage = round(weighted_coverage, 4)
        evidence.score = max(0.0, round(final_score, 4))
        evidence.score_reasons = _build_reasons(
            repo,
            covered_primary,
            covered_secondary,
            semantic_readme_score,
            query_diversity_score,
            doc_quality,
            low_value_penalty,
            scope_penalty,
        )

        repo.semantic_readme_score = round(semantic_readme_score, 5)
        repo.fit_score = evidence.score
        repo.relevance_score = evidence.score
        repo.covered_primary = covered_primary
        repo.missing_primary = missing_primary
        repo.reference_type = _classify_reference_type(covered_primary, weighted_coverage, doc_quality)
        repo.fit_summary = _build_fit_summary(repo.reference_type, covered_primary, covered_secondary, missing_primary)
        repo.rank_reasons = evidence.score_reasons

    ranked_pairs = sorted(
        zip(evidence_items, repo_embeddings),
        key=lambda item: item[0].repository.fit_score,
        reverse=True,
    )
    deduped_pairs = _dedupe_ranked_pairs(ranked_pairs, settings.RAG_DEDUP_THRESHOLD)
    selected = _select_diverse_covering(
        [evidence for evidence, _embedding in deduped_pairs],
        output_limit=settings.RAG_OUTPUT_REPO_LIMIT,
        max_per_language=settings.RAG_MAX_PER_LANGUAGE,
        primary_capabilities=keywords.primary_capabilities or keywords.capabilities[:3],
    )

    logger.info("Reranked %d repositories for idea '%s'", len(selected), idea[:80])
    return selected


def _intent_text(keywords: ExtractedKeywords, idea: str) -> str:
    return " ".join(
        part
        for part in (
            keywords.core_intent or keywords.summary or idea,
            " ".join(keywords.primary_capabilities[:3]),
            " ".join(keywords.secondary_capabilities[:4]),
            " ".join(keywords.domain_terms[:8]),
        )
        if part
    )


def _repo_text(evidence: ShallowRepoEvidence) -> str:
    repo = evidence.repository
    manifest_text = " ".join(file.content[:1200] for file in evidence.manifest_files[:3])
    return "\n".join(
        [
            repo.full_name,
            repo.description or "",
            " ".join(repo.topics),
            evidence.readme[:12000],
            manifest_text,
            " ".join(evidence.highlighted_paths[:12]),
        ]
    )


def _matched_capabilities(capabilities: list[str], lowered_text: str) -> list[str]:
    matched: list[str] = []
    for capability in capabilities:
        if _capability_signal(capability, lowered_text) >= 0.55:
            matched.append(capability)
    return matched


def _matched_keywords(terms: list[str], lowered_text: str) -> list[str]:
    matched: list[str] = []
    for term in terms:
        normalized = term.lower()
        if normalized and normalized in lowered_text:
            matched.append(term)
    return _dedupe_preserve(matched)[:6]


def _capability_signal(capability: str, lowered_text: str, keywords: ExtractedKeywords) -> float:
    """Score how well a capability matches the repository text."""
    score = 0.0
    
    # Direct capability name match
    if capability.lower() in lowered_text:
        score += 0.7
    
    # Check related keywords from model extraction
    for keyword in keywords.keywords + keywords.domain_terms + keywords.tech_terms:
        if keyword.lower() in lowered_text:
            score += 0.3
    
    return min(score, 1.0)


def _weighted_coverage(keywords: ExtractedKeywords, lowered_text: str) -> float:
    weights = keywords.capability_weights or {
        capability: 1.0 for capability in (keywords.primary_capabilities or keywords.capabilities[:3])
    }
    if not weights:
        return 0.0

    total_weight = sum(weights.values()) or 1.0
    covered_weight = sum(weight * _capability_signal(capability, lowered_text, keywords) for capability, weight in weights.items())
    return min(covered_weight / total_weight, 1.0)


def _doc_quality_score(readme: str, manifest_files: list) -> float:
    score = 0.0
    length = len(readme)
    if length > 1800:
        score += 0.65
    elif length > 700:
        score += 0.45
    elif length > 250:
        score += 0.2

    lowered = readme.lower()
    if any(token in lowered for token in ("installation", "usage", "architecture", "setup", "features")):
        score += 0.2
    if manifest_files:
        score += 0.15
    return min(score, 1.0)


def _repo_quality_score(repo) -> float:
    star_score = min(math.log10(repo.stars + 10) / 4.0, 1.0)
    freshness_boost = 0.1 if repo.updated_at else 0.0
    archive_penalty = 0.25 if repo.archived else 0.0
    return max(0.0, star_score + freshness_boost - archive_penalty)


def _low_value_penalty(full_name: str, description: str, readme: str) -> float:
    lowered = f"{full_name} {description} {readme[:800]}".lower()
    penalty = 0.0
    if any(pattern in lowered for pattern in LOW_VALUE_PATTERNS):
        penalty += 0.18
    if len(readme) < 120:
        penalty += 0.08
    return min(penalty, 0.35)


def _scope_mismatch_penalty(keywords: ExtractedKeywords, repo_text: str) -> float:
    ai_expected = "ai features" in keywords.capabilities or any(
        integration.lower() == "llm provider" for integration in keywords.likely_integrations
    )
    if ai_expected:
        return 0.0

    ai_patterns = (
        r"(^|[^a-z])ai([^a-z]|$)",
        r"(^|[^a-z])llm([^a-z]|$)",
        r"(^|[^a-z])gpt([^a-z]|$)",
        r"(^|[^a-z])openai([^a-z]|$)",
        r"(^|[^a-z])rag([^a-z]|$)",
        r"(^|[^a-z])embedding(s)?([^a-z]|$)",
        r"(^|[^a-z])assistant([^a-z]|$)",
        r"(^|[^a-z])vector([^a-z]|$)",
        r"(^|[^a-z])prompt(s|ing)?([^a-z]|$)",
    )
    hits = sum(1 for pattern in ai_patterns if re.search(pattern, repo_text))
    if hits >= 4:
        return 0.35
    if hits >= 2:
        return 0.18
    if hits >= 1:
        return 0.08
    return 0.0


def _classify_reference_type(covered_primary: list[str], weighted_coverage: float, doc_quality: float) -> str:
    if len(covered_primary) >= 2 and weighted_coverage >= 0.55 and doc_quality >= 0.35:
        return "end_to_end"
    if covered_primary or weighted_coverage >= 0.3:
        return "subsystem"
    return "pattern"


def _build_fit_summary(
    reference_type: str,
    covered_primary: list[str],
    covered_secondary: list[str],
    missing_primary: list[str],
) -> str:
    fragments: list[str] = [f"It behaves like a {reference_type.replace('_', ' ')} reference"]
    if covered_primary:
        fragments.append(f"covering {', '.join(covered_primary[:3])}")
    elif covered_secondary:
        fragments.append(f"showing useful patterns for {', '.join(covered_secondary[:2])}")
    if missing_primary:
        fragments.append(f"while missing {', '.join(missing_primary[:2])}")
    return " ".join(fragments) + "."


def _build_reasons(
    repo,
    covered_primary: list[str],
    covered_secondary: list[str],
    semantic_readme_score: float,
    query_diversity_score: float,
    doc_quality: float,
    low_value_penalty: float,
    scope_penalty: float,
) -> list[str]:
    reasons: list[str] = []
    if covered_primary:
        reasons.append(f"covers primary capabilities: {', '.join(covered_primary[:3])}")
    elif covered_secondary:
        reasons.append(f"covers supporting capabilities: {', '.join(covered_secondary[:3])}")
    reasons.append(f"README semantic score {semantic_readme_score:.2f}")
    if repo.query_hit_count:
        reasons.append(f"matched {repo.query_hit_count} search families")
    if doc_quality >= 0.45:
        reasons.append("good implementation docs or manifests")
    if low_value_penalty > 0:
        reasons.append("discounted as likely template/tutorial/test-style repo")
    if scope_penalty > 0:
        reasons.append("discounted for emphasizing implementation scope outside the resolved idea")
    if query_diversity_score >= 0.6:
        reasons.append("appeared across multiple idea-focused query families")
    return reasons[:6]


def _dedupe_ranked_pairs(
    ranked_pairs: list[tuple[ShallowRepoEvidence, list[float]]],
    threshold: float,
) -> list[tuple[ShallowRepoEvidence, list[float]]]:
    deduped: list[tuple[ShallowRepoEvidence, list[float]]] = []
    for evidence, embedding in ranked_pairs:
        duplicate = False
        for selected_evidence, selected_embedding in deduped:
            if _cosine_similarity(embedding, selected_embedding) >= threshold:
                duplicate = True
                logger.debug(
                    "Deduped %s as too similar to %s",
                    evidence.repository.full_name,
                    selected_evidence.repository.full_name,
                )
                break
        if not duplicate:
            deduped.append((evidence, embedding))
    return deduped


def _select_diverse_covering(
    evidence_items: list[ShallowRepoEvidence],
    *,
    output_limit: int,
    max_per_language: int,
    primary_capabilities: list[str],
) -> list[ShallowRepoEvidence]:
    selected: list[ShallowRepoEvidence] = []
    covered_primary: set[str] = set()
    language_counts: Counter[str] = Counter()
    remaining = evidence_items[:]
    needs_end_to_end = True

    while remaining and len(selected) < output_limit:
        best: ShallowRepoEvidence | None = None
        best_score = float("-inf")
        for evidence in remaining:
            repo = evidence.repository
            language = (repo.language or "Unknown").strip() or "Unknown"
            if language_counts[language] >= max_per_language:
                continue

            uncovered = len([capability for capability in repo.covered_primary if capability not in covered_primary])
            end_to_end_bonus = 0.15 if needs_end_to_end and repo.reference_type == "end_to_end" else 0.0
            missing_penalty = 0.04 * len(repo.missing_primary[:2])
            language_penalty = 0.03 * language_counts[language]
            selection_score = repo.fit_score + (0.12 * uncovered) + end_to_end_bonus - missing_penalty - language_penalty

            if selection_score > best_score:
                best = evidence
                best_score = selection_score

        if best is None:
            best = remaining[0]

        selected.append(best)
        covered_primary.update(best.repository.covered_primary)
        language = (best.repository.language or "Unknown").strip() or "Unknown"
        language_counts[language] += 1
        needs_end_to_end = needs_end_to_end and best.repository.reference_type != "end_to_end"
        remaining = [evidence for evidence in remaining if evidence.repository.full_name != best.repository.full_name]

        if primary_capabilities and covered_primary.issuperset(primary_capabilities[: len(primary_capabilities)]):
            remaining.sort(key=lambda evidence: evidence.repository.fit_score, reverse=True)

    return selected


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result
