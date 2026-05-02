"""Semantic reranking, deduplication, and diversity selection for GitHub repositories."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter

from core.config import get_settings
from models.schemas import ExtractedKeywords, ShallowRepoEvidence
from services.github_service import build_concept_families
from services.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_FINAL_RANKING_WEIGHTS = {
    "sampled_code_semantic": 0.24,
    "readme_semantic": 0.16,
    "metadata_semantic": 0.08,
    "star_quality": 0.24,
    "concept_family_coverage": 0.15,
    "domain_alignment": 0.08,
    "doc_quality": 0.03,
    "query_diversity": 0.02,
}


async def rank_repo_evidence(
    idea: str,
    keywords: ExtractedKeywords,
    evidence_items: list[ShallowRepoEvidence],
) -> list[ShallowRepoEvidence]:
    """Score, deduplicate, and select repositories by end-to-end semantic fit."""

    if not evidence_items:
        return []

    settings = get_settings()
    embedding_service = EmbeddingService()
    intent_text = _intent_text(keywords, idea)
    intent_embedding = await embedding_service.embed_query(intent_text)
    repo_texts = [_repo_text(evidence) for evidence in evidence_items]
    metadata_texts = [_metadata_text(evidence.repository) for evidence in evidence_items]
    readme_texts = [_readme_semantic_text(evidence) for evidence in evidence_items]
    sampled_code_texts = [_sampled_code_text(evidence) for evidence in evidence_items]
    repo_embeddings, metadata_embeddings, readme_embeddings, sampled_code_embeddings = await asyncio.gather(
        embedding_service.embed_documents(repo_texts),
        embedding_service.embed_documents(metadata_texts),
        embedding_service.embed_documents(readme_texts),
        embedding_service.embed_documents(sampled_code_texts),
    )
    max_query_hits = max((evidence.repository.query_hit_count for evidence in evidence_items), default=1)
    concept_families = await build_concept_families(
        keywords, idea_context=keywords.core_intent or keywords.summary, limit=6
    )
    concept_family_embeddings = await embedding_service.embed_documents(
        [
            " ".join(
                [
                    str(family.get("label", "")),
                    *[str(alias) for alias in family.get("aliases", [])[:3]],
                ]
            ).strip()
            for family in concept_families
        ]
    ) if concept_families else []
    concept_embedding_map = {
        str(family["canonical"]): embedding
        for family, embedding in zip(concept_families, concept_family_embeddings)
    }
    weights = dict(_FINAL_RANKING_WEIGHTS)

    primary_lookup = {
        _normalize_phrase(capability): capability
        for capability in (keywords.primary_capabilities or keywords.capabilities[:3])
    }
    secondary_lookup = {
        _normalize_phrase(capability): capability
        for capability in (keywords.secondary_capabilities or keywords.capabilities[3:6])
    }

    for evidence, repo_embedding, metadata_embedding, readme_embedding, sampled_code_embedding, repo_text in zip(
        evidence_items,
        repo_embeddings,
        metadata_embeddings,
        readme_embeddings,
        sampled_code_embeddings,
        repo_texts,
    ):
        repo = evidence.repository
        lowered = repo_text.lower()
        normalized_text = _normalize_phrase(repo_text)
        trivial_capabilities = keywords.trivial_capabilities or []
        concept_scores = _concept_family_scores(concept_families, normalized_text, repo_embedding, concept_embedding_map)
        covered_primary = [
            capability
            for canonical, capability in primary_lookup.items()
            if concept_scores.get(canonical, 0.0) >= 0.52
        ]
        covered_secondary = [
            capability
            for canonical, capability in secondary_lookup.items()
            if concept_scores.get(canonical, 0.0) >= 0.52
        ]
        matched_trivial = _matched_capabilities(trivial_capabilities, lowered, keywords)
        primary_capabilities = list(primary_lookup.values())
        missing_primary = [capability for capability in primary_capabilities if capability not in covered_primary]

        semantic_readme_score = _cosine_similarity(intent_embedding, readme_embedding)
        sampled_code_semantic_score = _cosine_similarity(intent_embedding, sampled_code_embedding)
        weighted_coverage = _weighted_coverage_from_scores(keywords, concept_scores)
        semantic_meta_score = repo.semantic_meta_score or _cosine_similarity(intent_embedding, metadata_embedding)
        query_diversity_score = repo.query_hit_count / max(max_query_hits, 1)
        domain_alignment = _domain_alignment_score(keywords, normalized_text)
        doc_quality = _doc_quality_score(evidence.readme, evidence.manifest_files)
        quality_score = _repo_quality_score(repo)
        low_value_penalty = min(0.18, _low_value_penalty(repo.full_name, repo.description or "", evidence.readme) * 1.5)
        scope_penalty = min(0.20, _scope_mismatch_penalty(keywords, lowered))
        low_semantic_code_penalty = (
            0.15
            if evidence.sampled_files and sampled_code_semantic_score < settings.RAG_SEMANTIC_MIN_RELEVANCE
            else 0.0
        )
        reference_type = _classify_reference_type(
            covered_primary,
            weighted_coverage,
            doc_quality,
            sampled_code_semantic_score,
            semantic_readme_score,
        )
        end_to_end_bonus = 0.10 if reference_type == "end_to_end" else 0.05 if reference_type == "subsystem" else 0.0
        concept_breadth_bonus = 0.05 if len(covered_primary) >= 2 else 0.0

        final_score = (
            weights["sampled_code_semantic"] * sampled_code_semantic_score
            + weights["readme_semantic"] * semantic_readme_score
            + weights["metadata_semantic"] * semantic_meta_score
            + weights["star_quality"] * quality_score
            + weights["concept_family_coverage"] * weighted_coverage
            + weights["domain_alignment"] * domain_alignment
            + weights["doc_quality"] * doc_quality
            + weights["query_diversity"] * query_diversity_score
            + end_to_end_bonus
            + concept_breadth_bonus
            - low_value_penalty
            - scope_penalty
            - low_semantic_code_penalty
        )

        evidence.matched_capabilities = _dedupe_preserve([*covered_primary, *covered_secondary, *matched_trivial])
        evidence.matched_keywords = _matched_keywords(keywords.domain_terms or keywords.keywords, lowered)
        evidence.matched_frameworks = _matched_keywords(keywords.frameworks, lowered)
        evidence.matched_stack_families = _matched_keywords(keywords.tech_terms or keywords.likely_stack_families, lowered)
        evidence.concept_family_matches = [
            str(family["label"])
            for family in concept_families
            if concept_scores.get(str(family["canonical"]), 0.0) >= 0.52
        ][:6]
        evidence.capability_coverage = round(weighted_coverage, 4)
        evidence.semantic_code_score = round(sampled_code_semantic_score, 5)
        evidence.score = max(0.0, round(final_score, 4))
        evidence.score_reasons = _build_reasons(
            repo,
            covered_primary,
            covered_secondary,
            semantic_readme_score,
            sampled_code_semantic_score,
            query_diversity_score,
            domain_alignment,
            doc_quality,
            low_value_penalty,
            scope_penalty,
            low_semantic_code_penalty,
        )

        repo.semantic_readme_score = round(semantic_readme_score, 5)
        repo.fit_score = evidence.score
        repo.relevance_score = evidence.score
        repo.covered_primary = covered_primary
        repo.missing_primary = missing_primary
        repo.reference_type = reference_type
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
    sampled_code_text = " ".join(file.content[:1600] for file in evidence.sampled_files[:3])
    return "\n".join(
        [
            repo.full_name,
            repo.description or "",
            " ".join(repo.topics),
            evidence.readme[:12000],
            manifest_text,
            sampled_code_text,
            " ".join(evidence.highlighted_paths[:12]),
            " ".join(evidence.sampled_paths[:6]),
        ]
    )


def _metadata_text(repo) -> str:
    return "\n".join(
        [
            repo.full_name,
            repo.description or "",
            " ".join(repo.topics),
            repo.language or "",
        ]
    )


def _readme_semantic_text(evidence: ShallowRepoEvidence) -> str:
    manifest_text = " ".join(file.content[:1200] for file in evidence.manifest_files[:3])
    return "\n".join(
        [
            evidence.readme[:12000],
            manifest_text,
            " ".join(evidence.highlighted_paths[:12]),
        ]
    )


def _sampled_code_text(evidence: ShallowRepoEvidence) -> str:
    sampled_text = " ".join(file.content[:1600] for file in evidence.sampled_files[:3])
    return "\n".join(
        [
            sampled_text,
            " ".join(evidence.sampled_paths[:6]),
        ]
    )


def _normalize_phrase(value: str) -> str:
    return " ".join(part for part in value.lower().replace("_", " ").replace("-", " ").split() if part)


def _concept_family_scores(
    concept_families: list[dict[str, object]],
    normalized_text: str,
    repo_embedding: list[float],
    concept_embedding_map: dict[str, list[float]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for family in concept_families:
        canonical = str(family["canonical"])
        aliases = [_normalize_phrase(str(alias)) for alias in family.get("aliases", []) if str(alias).strip()]
        literal_hits = sum(1 for alias in aliases if alias and alias in normalized_text)
        literal_score = min(1.0, literal_hits / max(1, min(len(aliases), 2)))
        semantic_score = _cosine_similarity(repo_embedding, concept_embedding_map.get(canonical, []))
        scores[canonical] = max(literal_score, semantic_score)
    return scores


def _weighted_coverage_from_scores(keywords: ExtractedKeywords, concept_scores: dict[str, float]) -> float:
    weights = {
        _normalize_phrase(capability): weight
        for capability, weight in (keywords.capability_weights or {
            capability: 1.0 for capability in (keywords.primary_capabilities or keywords.capabilities[:3])
        }).items()
    }
    if not weights:
        return 0.0

    total_weight = sum(weights.values()) or 1.0
    covered_weight = sum(weights.get(canonical, 0.0) * concept_scores.get(canonical, 0.0) for canonical in weights)
    return min(covered_weight / total_weight, 1.0)


def _domain_alignment_score(keywords: ExtractedKeywords, lowered_text: str) -> float:
    terms = [
        keywords.product_type,
        *keywords.domain_terms[:6],
        *keywords.likely_components[:4],
    ]
    normalized_terms = [
        _normalize_phrase(term)
        for term in terms
        if isinstance(term, str) and term.strip()
    ]
    if not normalized_terms:
        return 0.0
    normalized_text = _normalize_phrase(lowered_text)
    hits = sum(1 for term in normalized_terms if term and term in normalized_text)
    return hits / len(normalized_terms)


def _matched_capabilities(capabilities: list[str], lowered_text: str, keywords: ExtractedKeywords) -> list[str]:
    matched: list[str] = []
    for capability in capabilities:
        if _capability_signal(capability, lowered_text, keywords) >= 0.55:
            matched.append(capability)
    return matched


def _matched_keywords(terms: list[str], lowered_text: str) -> list[str]:
    matched: list[str] = []
    for term in terms:
        normalized = term.lower()
        if normalized and normalized in lowered_text:
            matched.append(term)
    return _dedupe_preserve(matched)[:6]


async def _capability_signal_ai(capability: str, repo_text: str, keywords: ExtractedKeywords) -> float:
    """Score how well a capability matches the repository text using AI."""
    from services.llm_client import get_llm_client
    
    llm = get_llm_client()
    keywords_context = {
        "keywords": keywords.keywords,
        "domain_terms": keywords.domain_terms,
        "tech_terms": keywords.tech_terms,
    }
    
    try:
        score = await llm.evaluate_capability_match(capability, repo_text[:2000], keywords_context)
        return score
    except Exception as exc:
        logger.warning("AI capability matching failed for %s: %s. Using simple fallback.", capability, exc)
        # Simple fallback without hardcoded weights
        if capability.lower() in repo_text.lower():
            return 0.7
        return 0.0

def _capability_signal(capability: str, lowered_text: str, keywords: ExtractedKeywords) -> float:
    """Simple synchronous wrapper for capability matching - used in sync contexts."""
    # For synchronous callers, use simple keyword matching
    # This will be replaced when we make all callers async
    score = 0.0
    if capability.lower() in lowered_text:
        score += 0.7
    for keyword in keywords.keywords + keywords.domain_terms + keywords.tech_terms:
        if keyword.lower() in lowered_text:
            score += 0.2
            break
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
    """Evaluate documentation quality without hardcoded length thresholds."""
    score = 0.0
    length = len(readme)
    
    # Proportional length scoring instead of hardcoded thresholds
    if length > 0:
        score += min(0.65, length / 3000)  # Gradual increase up to 0.65 at ~3000 chars
    
    lowered = readme.lower()
    # Look for documentation quality indicators
    quality_indicators = ["installation", "usage", "architecture", "setup", "features", "getting started", "documentation"]
    indicator_count = sum(1 for indicator in quality_indicators if indicator in lowered)
    score += min(0.25, indicator_count * 0.08)
    
    if manifest_files:
        score += 0.15
    
    return min(score, 1.0)


def _repo_quality_score(repo) -> float:
    star_score = min(math.log10(repo.stars + 10) / 4.0, 1.0)
    freshness_boost = 0.1 if repo.updated_at else 0.0
    archive_penalty = 0.25 if repo.archived else 0.0
    return max(0.0, star_score + freshness_boost - archive_penalty)


def _low_value_penalty(full_name: str, description: str, readme: str) -> float:
    """Evaluate if repository is low-value (tutorial, template, etc.) - simplified without hardcoded patterns."""
    penalty = 0.0
    
    # Very short README is a quality signal regardless of content
    if len(readme) < 120:
        penalty += 0.08
    
    # Use simple heuristics instead of hardcoded patterns
    lowered = f"{full_name} {description} {readme[:800]}".lower()
    
    # Count generic tutorial/template indicators
    tutorial_indicators = sum(
        1
        for word in ["tutorial", "example", "template", "boilerplate", "starter", "demo", "sample"]
        if word in lowered
    )
    if tutorial_indicators >= 2:
        penalty += 0.12
    elif tutorial_indicators == 1:
        penalty += 0.06

    docs_only_indicators = sum(1 for word in ["docs", "documentation", "guide"] if word in lowered)
    if docs_only_indicators >= 2 and "src/" not in lowered and "app/" not in lowered:
        penalty += 0.08

    return min(penalty, 0.25)


def _scope_mismatch_penalty(keywords: ExtractedKeywords, repo_text: str) -> float:
    """Penalize repos with heavy AI focus when not needed - simplified without hardcoded regex."""
    ai_expected = "ai features" in keywords.capabilities or any(
        integration.lower() == "llm provider" for integration in keywords.likely_integrations
    )
    if ai_expected:
        return 0.0

    # Simple keyword counting instead of complex regex patterns
    ai_keywords = ["ai", "llm", "gpt", "openai", "rag", "embedding", "assistant", "vector", "prompt"]
    hits = sum(1 for keyword in ai_keywords if f" {keyword} " in f" {repo_text.lower()} ")
    
    if hits >= 4:
        return 0.35
    if hits >= 2:
        return 0.18
    if hits >= 1:
        return 0.08
    return 0.0


def _classify_reference_type(
    covered_primary: list[str],
    weighted_coverage: float,
    doc_quality: float,
    sampled_code_semantic_score: float,
    semantic_readme_score: float,
) -> str:
    if (
        len(covered_primary) >= 2
        and weighted_coverage >= 0.55
        and doc_quality >= 0.25
        and (
            sampled_code_semantic_score >= 0.34
            or semantic_readme_score >= 0.42
        )
    ):
        return "end_to_end"
    if covered_primary or weighted_coverage >= 0.35 or sampled_code_semantic_score >= get_settings().RAG_SEMANTIC_MIN_RELEVANCE:
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
    sampled_code_semantic_score: float,
    query_diversity_score: float,
    domain_alignment: float,
    doc_quality: float,
    low_value_penalty: float,
    scope_penalty: float,
    low_semantic_code_penalty: float,
) -> list[str]:
    reasons: list[str] = []
    if covered_primary:
        reasons.append(f"covers primary capabilities: {', '.join(covered_primary[:3])}")
    elif covered_secondary:
        reasons.append(f"covers supporting capabilities: {', '.join(covered_secondary[:3])}")
    reasons.append(f"README semantic score {semantic_readme_score:.2f}")
    if sampled_code_semantic_score > 0:
        reasons.append(f"sampled code semantic score {sampled_code_semantic_score:.2f}")
    if repo.query_hit_count:
        reasons.append(f"matched {repo.query_hit_count} search families")
    if domain_alignment >= 0.25:
        reasons.append(f"domain alignment {domain_alignment:.2f}")
    if doc_quality >= 0.45:
        reasons.append("good implementation docs or manifests")
    if low_value_penalty > 0:
        reasons.append("discounted as likely template/tutorial/test-style repo")
    if scope_penalty > 0:
        reasons.append("discounted for emphasizing implementation scope outside the resolved idea")
    if low_semantic_code_penalty > 0:
        reasons.append("discounted because sampled code did not strongly match the resolved product intent")
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
    remaining = sorted(evidence_items, key=lambda evidence: evidence.repository.fit_score, reverse=True)
    preferred_slots = min(3, output_limit)

    def _can_add(repo) -> bool:
        language = (repo.language or "Unknown").strip() or "Unknown"
        return language_counts[language] < max_per_language

    def _selection_score(evidence: ShallowRepoEvidence, prefer_end_to_end: bool) -> float:
        repo = evidence.repository
        uncovered = len([capability for capability in repo.covered_primary if capability not in covered_primary])
        language = (repo.language or "Unknown").strip() or "Unknown"
        language_penalty = 0.03 * language_counts[language]
        missing_penalty = 0.04 * len(repo.missing_primary[:2])
        reference_bonus = 0.12 if prefer_end_to_end and repo.reference_type == "end_to_end" else 0.03 if repo.reference_type == "subsystem" else 0.0
        if not prefer_end_to_end and repo.reference_type == "end_to_end":
            reference_bonus = 0.04
        return repo.fit_score + (0.08 * uncovered) + reference_bonus - missing_penalty - language_penalty

    while remaining and len(selected) < preferred_slots:
        best: ShallowRepoEvidence | None = None
        best_score = float("-inf")
        for evidence in remaining:
            repo = evidence.repository
            if not _can_add(repo):
                continue
            selection_score = _selection_score(evidence, prefer_end_to_end=True)
            if selection_score > best_score:
                best = evidence
                best_score = selection_score

        if best is None:
            break

        selected.append(best)
        covered_primary.update(best.repository.covered_primary)
        language = (best.repository.language or "Unknown").strip() or "Unknown"
        language_counts[language] += 1
        remaining = [evidence for evidence in remaining if evidence.repository.full_name != best.repository.full_name]

    while remaining and len(selected) < output_limit:
        best: ShallowRepoEvidence | None = None
        best_score = float("-inf")
        for evidence in remaining:
            repo = evidence.repository
            if not _can_add(repo):
                continue
            selection_score = _selection_score(evidence, prefer_end_to_end=False)
            if selection_score > best_score:
                best = evidence
                best_score = selection_score

        if best is None:
            break

        selected.append(best)
        covered_primary.update(best.repository.covered_primary)
        language = (best.repository.language or "Unknown").strip() or "Unknown"
        language_counts[language] += 1
        remaining = [evidence for evidence in remaining if evidence.repository.full_name != best.repository.full_name]

    if len(selected) < output_limit:
        already_selected = {evidence.repository.full_name for evidence in selected}
        for evidence in evidence_items:
            if len(selected) >= output_limit:
                break
            if evidence.repository.full_name in already_selected:
                continue
            selected.append(evidence)
            already_selected.add(evidence.repository.full_name)

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
