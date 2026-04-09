"""Capability-first reranking for GitHub repositories."""

from __future__ import annotations

import logging
import math
import re

from models.schemas import ExtractedKeywords, ShallowRepoEvidence

logger = logging.getLogger(__name__)

LOW_VALUE_PATTERNS = (
    "test",
    "submission",
    "challenge",
    "leetcode",
    "boilerplate",
    "starter",
    "template",
    "scaffold",
    "tutorial",
    "course",
    "exercise",
)


def rank_repo_evidence(
    idea: str,
    keywords: ExtractedKeywords,
    evidence_items: list[ShallowRepoEvidence],
) -> list[ShallowRepoEvidence]:
    """Score and sort shallow repo evidence by capability fit."""

    ranked: list[ShallowRepoEvidence] = []
    product_terms = _tokenize(f"{keywords.product_type} {keywords.summary}")
    capability_terms = {term.lower() for term in keywords.capabilities}
    keyword_terms = {term.lower() for term in keywords.keywords}
    stack_terms = {term.lower() for term in keywords.likely_stack_families}
    framework_terms = {term.lower() for term in keywords.frameworks}

    for evidence in evidence_items:
        repo = evidence.repository
        text = " ".join(
            [
                repo.full_name,
                repo.description or "",
                " ".join(repo.topics),
                evidence.readme[:5000],
                " ".join(file.path for file in evidence.manifest_files),
                " ".join(file.content[:2000] for file in evidence.manifest_files),
                " ".join(evidence.highlighted_paths[:30]),
            ]
        ).lower()

        matched_capabilities = sorted(term for term in capability_terms if term and term in text)
        matched_keywords = sorted(term for term in keyword_terms if term and term in text)
        matched_frameworks = sorted(term for term in framework_terms if term and term in text)
        matched_stack = sorted(term for term in stack_terms if term and term in text)

        capability_score = min(len(matched_capabilities) / max(len(capability_terms), 1), 1.0)
        keyword_score = min(len(matched_keywords) / max(len(keyword_terms), 1), 1.0)
        framework_score = min(len(matched_frameworks) / max(len(framework_terms), 1), 1.0) if framework_terms else 0.0
        stack_score = min(len(matched_stack) / max(len(stack_terms), 1), 1.0) if stack_terms else 0.0
        product_score = len(product_terms & _tokenize(text)) / max(len(product_terms), 1)
        doc_score = 1.0 if len(evidence.readme) > 1200 else 0.6 if len(evidence.readme) > 300 else 0.1
        manifest_score = min(len(evidence.manifest_files) / 5.0, 1.0)
        star_score = min(math.log10(repo.stars + 10) / 4.0, 1.0)
        architecture_bonus = _architecture_bonus(keywords, evidence.highlighted_paths)
        low_value_penalty = _low_value_penalty(repo.full_name, repo.description or "", evidence.readme)
        scope_penalty = _scope_mismatch_penalty(keywords, text)
        archived_penalty = 0.20 if repo.archived else 0.0

        score = (
            0.42 * capability_score
            + 0.14 * product_score
            + 0.10 * keyword_score
            + 0.08 * framework_score
            + 0.08 * stack_score
            + 0.07 * doc_score
            + 0.05 * manifest_score
            + 0.03 * architecture_bonus
            + 0.03 * star_score
            - low_value_penalty
            - scope_penalty
            - archived_penalty
        )

        reference_type = _classify_reference_type(capability_score, product_score, len(matched_capabilities))
        fit_summary = _build_fit_summary(repo.full_name, reference_type, matched_capabilities, matched_frameworks, low_value_penalty)
        reasons: list[str] = []
        if matched_capabilities:
            reasons.append(f"capabilities: {', '.join(matched_capabilities[:3])}")
        if matched_frameworks:
            reasons.append(f"frameworks: {', '.join(matched_frameworks[:3])}")
        if matched_stack:
            reasons.append(f"stack patterns: {', '.join(matched_stack[:2])}")
        if doc_score >= 0.6:
            reasons.append("useful README/manifests for implementation details")
        if low_value_penalty > 0:
            reasons.append("penalized as likely template/tutorial/test repo")
        if scope_penalty > 0:
            reasons.append("discounted because it emphasizes AI-heavy implementation outside the resolved scope")

        evidence.matched_capabilities = matched_capabilities
        evidence.matched_keywords = matched_keywords
        evidence.matched_frameworks = matched_frameworks
        evidence.matched_stack_families = matched_stack
        evidence.capability_coverage = round(capability_score, 4)
        evidence.score = max(0.0, round(score, 4))
        evidence.score_reasons = reasons

        repo.relevance_score = evidence.score
        repo.fit_score = evidence.score
        repo.reference_type = reference_type
        repo.fit_summary = fit_summary
        repo.rank_reasons = reasons
        ranked.append(evidence)

    ranked.sort(key=lambda item: item.score, reverse=True)
    logger.info("Reranked %d repositories for idea '%s'", len(ranked), idea[:80])
    return ranked


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_./-]+", text.lower()) if len(token) > 2}


def _architecture_bonus(keywords: ExtractedKeywords, highlighted_paths: list[str]) -> float:
    path_text = " ".join(highlighted_paths).lower()
    bonus = 0.0
    if any(cap in keywords.capabilities for cap in ("ingredient inventory", "recipe recommendation", "meal planning")) and any(
        token in path_text for token in ("recipe", "ingredient", "pantry", "meal", "grocery", "nutrition")
    ):
        bonus += 0.5
    if any(cap in keywords.capabilities for cap in ("realtime collaboration", "chat messaging")) and any(
        token in path_text for token in ("socket", "ws", "realtime", "presence")
    ):
        bonus += 0.5
    if "canvas rendering" in keywords.capabilities and any(token in path_text for token in ("canvas", "draw", "whiteboard")):
        bonus += 0.4
    if any(cap in keywords.capabilities for cap in ("booking and scheduling", "marketplace matching")) and any(
        token in path_text for token in ("booking", "schedule", "marketplace", "listing", "search")
    ):
        bonus += 0.4
    return min(bonus, 1.0)


def _low_value_penalty(full_name: str, description: str, readme: str) -> float:
    lowered = f"{full_name} {description}".lower()
    penalty = 0.0
    if any(pattern in lowered for pattern in LOW_VALUE_PATTERNS):
        penalty += 0.18
    if len(readme) < 200:
        penalty += 0.08
    if "generated by" in readme.lower():
        penalty += 0.10
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
    if hits >= 3:
        return 0.28
    if hits >= 2:
        return 0.18
    if hits >= 1:
        return 0.08
    return 0.0


def _classify_reference_type(capability_score: float, product_score: float, matched: int) -> str:
    if capability_score >= 0.45 and product_score >= 0.18 and matched >= 2:
        return "end_to_end"
    if capability_score >= 0.25 and matched >= 1:
        return "subsystem"
    return "pattern"


def _build_fit_summary(
    full_name: str,
    reference_type: str,
    matched_capabilities: list[str],
    matched_frameworks: list[str],
    low_value_penalty: float,
) -> str:
    fragments: list[str] = [f"It behaves like a {reference_type.replace('_', ' ')} reference"]
    if matched_capabilities:
        fragments.append(f"covering {', '.join(matched_capabilities[:3])}")
    if matched_frameworks:
        fragments.append(f"and showing {', '.join(matched_frameworks[:2])}")
    if low_value_penalty > 0:
        fragments.append("but it was discounted for likely being a smaller or thinner reference")
    return " ".join(fragments) + "."
