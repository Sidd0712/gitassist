"""Model-based idea analysis using Groq API with Llama 3.3."""

from __future__ import annotations

import logging

from models.schemas import (
    AnalysisResponse,
    ClarificationQuestion,
    ExtractedKeywords,
    LearningStep,
    RepoSearchResult,
    RetrievalHit,
    RetrievalPlan,
    RetrievalQuery,
    TechRecommendation,
)
from services.llm_client import get_llm_client
from services.rag.citation_service import build_analysis_evidence

logger = logging.getLogger(__name__)


async def extract_keywords(idea: str, clarification_answers: dict[str, str] | None = None) -> ExtractedKeywords:
    """Extract structured technical intent from the user's idea using LLM."""

    llm = get_llm_client()
    extracted = await llm.extract_keywords(idea, clarification_answers)
    
    result = ExtractedKeywords(
        keywords=extracted.get("keywords", [])[:12],
        frameworks=extracted.get("frameworks", [])[:8],
        languages=extracted.get("languages", [])[:4],
        summary=extracted.get("summary", idea)[:200],
        core_intent=extracted.get("core_intent", "")[:200],
        product_type=extracted.get("product_type", "")[:100],
        target_users=extracted.get("target_users", [])[:4],
        capabilities=extracted.get("primary_capabilities", []) + extracted.get("secondary_capabilities", [])[:12],
        primary_capabilities=extracted.get("primary_capabilities", [])[:3],
        secondary_capabilities=extracted.get("secondary_capabilities", [])[:6],
        trivial_capabilities=extracted.get("trivial_capabilities", [])[:8],
        capability_weights=_normalize_capability_weights(extracted.get("capability_weights", {})),
        domain_terms=(extracted.get("domain_terms") or extracted.get("keywords", []))[:12],
        tech_terms=(extracted.get("tech_terms") or (extracted.get("frameworks", []) + extracted.get("languages", [])))[:10],
        constraints=extracted.get("constraints", [])[:8],
        likely_components=extracted.get("likely_components", [])[:10],
        likely_integrations=extracted.get("likely_integrations", [])[:8],
        likely_stack_families=extracted.get("likely_stack_families", [])[:10],
        ambiguities=_convert_ambiguities(extracted.get("ambiguities", [])),
        assumptions=extracted.get("assumptions", [])[:8],
    )

    logger.info(
        "Intent extraction complete: %d capabilities, %d frameworks",
        len(result.capabilities),
        len(result.frameworks),
    )
    return result


def build_clarification_questions(keywords: ExtractedKeywords) -> list[ClarificationQuestion]:
    """Build clarification questions directly from model-generated ambiguities."""

    questions: list[ClarificationQuestion] = []
    seen_axes: set[str] = set()

    for ambiguity in keywords.ambiguities:
        if ambiguity.resolved or ambiguity.severity != "high":
            continue

        axis = ambiguity.axis.strip()
        if not axis or axis.lower() in seen_axes:
            continue

        options = _normalize_clarification_options(ambiguity.options)
        if len(options) < 2:
            logger.warning("Skipping ambiguity '%s' because the model did not return enough usable options", axis)
            continue

        question_text = (ambiguity.question or ambiguity.reason).strip()
        if not question_text:
            logger.warning("Skipping ambiguity '%s' because the model did not return a usable question", axis)
            continue

        question = ClarificationQuestion(
            key=axis,
            question=question_text,
            options=options,
            reason=ambiguity.reason,
        )
        questions.append(question)
        seen_axes.add(axis.lower())
        if len(questions) >= 4:
            break

    return questions


async def plan_retrieval_queries(
    idea: str,
    keywords: ExtractedKeywords,
    repositories: list[RepoSearchResult],
) -> RetrievalPlan:
    """Create retrieval plan using model."""

    llm = get_llm_client()
    
    repo_dicts = [
        {
            "full_name": repo.full_name,
            "relevance_score": repo.relevance_score,
        }
        for repo in repositories
    ]
    
    plan_dict = await llm.plan_retrieval_queries(idea, keywords.model_dump(), repo_dicts)
    
    queries = []
    for q in plan_dict.get("queries", []):
        queries.append(
            RetrievalQuery(
                section=q.get("section", "repo_descriptions"),
                query=q.get("query", ""),
                preferred_roles=[],
                top_k=q.get("top_k", 8),
            )
        )
    
    return RetrievalPlan(queries=queries)


async def generate_analysis(
    idea: str,
    keywords: ExtractedKeywords,
    repositories: list[RepoSearchResult],
    section_hits: dict[str, list[RetrievalHit]],
) -> AnalysisResponse:
    """Generate analysis using models."""

    llm = get_llm_client()
    
    repo_dicts = [
        {
            "full_name": repo.full_name,
            "relevance_score": repo.relevance_score,
            "url": repo.html_url,
        }
        for repo in repositories
    ]
    
    # Generate all sections in parallel concept
    repo_descriptions = await llm.generate_repo_descriptions(
        idea,
        keywords.model_dump(),
        repo_dicts,
        {},
    )
    
    learning_path_data = await llm.generate_learning_path(idea, keywords.model_dump(), repo_dicts)
    learning_path = [
        LearningStep(
            step_number=item.get("step", i + 1),
            title=item.get("title", ""),
            description=item.get("description", ""),
            milestone=item.get("milestone", ""),
            concepts=item.get("concepts", [])[:6],
            resources=item.get("resources", [])[:6],
        )
        for i, item in enumerate(learning_path_data)
    ]
    
    architecture_diagram = await llm.generate_architecture_diagram(idea, keywords.model_dump(), repo_dicts)
    
    tech_stack_data = await llm.generate_tech_stack(idea, keywords.model_dump(), repo_dicts)
    tech_stack = [
        TechRecommendation(
            name=item.get("technology", ""),
            category=item.get("layer", ""),
            why_recommended=item.get("reasoning", ""),
            supported_by=item.get("supported_by", [])[:6],
            pros=item.get("pros", []),
            cons=item.get("cons", []),
        )
        for item in tech_stack_data
    ]

    return AnalysisResponse(
        idea_summary=keywords.summary or idea,
        keywords=keywords,
        assumptions=keywords.assumptions,
        repositories=repositories,
        repo_descriptions=repo_descriptions,
        learning_path=learning_path,
        architecture_diagram=architecture_diagram,
        tech_stack=tech_stack,
        evidence=build_analysis_evidence(section_hits),
        status="complete",
    )


def _convert_ambiguities(ambiguities: list[dict]) -> list:
    """Convert model ambiguities to schema objects."""
    from models.schemas import AmbiguityFlag
    
    result = []
    for amb in ambiguities:
        result.append(
            AmbiguityFlag(
                axis=amb.get("axis", "unknown"),
                question=amb.get("question", ""),
                reason=amb.get("reason", ""),
                options=_normalize_clarification_options(amb.get("options", [])),
                severity=amb.get("severity", "medium"),
                resolved=False,
                answer=None,
            )
        )
    return result


def _normalize_clarification_options(options: object) -> list[str]:
    """Sanitize, deduplicate, and cap model-generated clarification options."""

    if isinstance(options, str):
        raw_options = [options]
    elif isinstance(options, list):
        raw_options = [option for option in options if isinstance(option, str)]
    else:
        raw_options = []

    normalized: list[str] = []
    seen: set[str] = set()

    for option in raw_options:
        cleaned = option.strip().lstrip("-*").strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
        if len(normalized) >= 4:
            break

    return normalized


def _normalize_capability_weights(raw_weights: object) -> dict[str, float]:
    """Coerce capability weights into a clean string->float mapping."""

    if not isinstance(raw_weights, dict):
        return {}

    normalized: dict[str, float] = {}
    for key, value in raw_weights.items():
        if not isinstance(key, str):
            continue
        try:
            normalized_key = key.strip()
            if not normalized_key:
                continue
            normalized[normalized_key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized
