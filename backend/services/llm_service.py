"""Model-based idea analysis using Groq API with warm-path caching."""

from __future__ import annotations

import asyncio
import logging

from models.schemas import (
    AnalysisResponse,
    ChatMessage,
    ClarificationQuestion,
    ExtractedKeywords,
    LearningStep,
    RepoSearchResult,
    RetrievalHit,
    RetrievalPlan,
    RetrievalQuery,
    RetrievalWeights,
    TechRecommendation,
)
from services.llm_client import get_llm_client
from services.pipeline_cache_service import PipelineCacheService, compact_mapping, stable_cache_key
from services.rag.citation_service import build_analysis_evidence
from services.rag.query_planning_service import build_default_retrieval_plan

logger = logging.getLogger(__name__)

_cache = PipelineCacheService()
_DEFAULT_RANKING_WEIGHTS = {
    "readme_semantic": 0.45,
    "metadata_semantic": 0.2,
    "capability_coverage": 0.2,
    "doc_quality": 0.07,
    "query_diversity": 0.04,
    "star_quality": 0.04,
}
_DEFAULT_RETRIEVAL_WEIGHTS = RetrievalWeights()


async def extract_keywords(idea: str, clarification_answers: dict[str, str] | None = None) -> ExtractedKeywords:
    """Extract structured technical intent from the user's idea using LLM."""

    cache_key = stable_cache_key(
        {
            "idea": idea.strip(),
            "clarification_answers": compact_mapping(clarification_answers or {}),
        }
    )
    cached = _cache.get_json("llm_extract_keywords", cache_key)
    if isinstance(cached, dict):
        extracted = cached
    else:
        llm = get_llm_client()
        extracted = await llm.extract_keywords(idea, clarification_answers)
        _cache.set_json("llm_extract_keywords", cache_key, extracted)

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

        questions.append(
            ClarificationQuestion(
                key=axis,
                question=question_text,
                options=options,
                reason=ambiguity.reason,
            )
        )
        seen_axes.add(axis.lower())
        if len(questions) >= 4:
            break

    return questions


async def get_ranking_weights(idea: str, repositories_count: int) -> dict[str, float]:
    """Load cached ranking weights or compute them once for the current idea."""

    cache_key = stable_cache_key(
        {
            "idea": idea.strip(),
            "repositories_count": repositories_count,
        }
    )
    cached = _cache.get_json("llm_ranking_weights", cache_key)
    if isinstance(cached, dict):
        return _normalize_ranking_weights(cached)

    llm = get_llm_client()
    try:
        weights = await llm.calculate_ranking_weights(idea, repositories_count)
    except Exception as exc:
        logger.warning("Failed to get AI ranking weights: %s. Using defaults.", exc)
        return dict(_DEFAULT_RANKING_WEIGHTS)

    normalized = _normalize_ranking_weights(weights)
    _cache.set_json("llm_ranking_weights", cache_key, normalized)
    return normalized


async def plan_retrieval_queries(
    idea: str,
    keywords: ExtractedKeywords,
    repositories: list[RepoSearchResult],
) -> RetrievalPlan:
    """Create retrieval plan using the model, with deterministic fallback."""

    if not repositories:
        return build_default_retrieval_plan(idea, keywords, top_k=8)

    repo_dicts = [
        {
            "full_name": repo.full_name,
            "commit_sha": repo.commit_sha,
            "relevance_score": repo.relevance_score,
        }
        for repo in repositories
    ]
    cache_key = stable_cache_key(
        {
            "idea": idea.strip(),
            "keywords": compact_mapping(
                {
                    "summary": keywords.summary,
                    "primary_capabilities": keywords.primary_capabilities,
                    "secondary_capabilities": keywords.secondary_capabilities,
                    "domain_terms": keywords.domain_terms,
                    "frameworks": keywords.frameworks,
                    "languages": keywords.languages,
                }
            ),
            "repositories": repo_dicts,
        }
    )

    cached = _cache.get_json("llm_retrieval_plan", cache_key)
    if isinstance(cached, dict):
        return _retrieval_plan_from_dict(cached, keywords, idea)

    llm = get_llm_client()
    try:
        plan_dict = await llm.plan_retrieval_queries(idea, keywords.model_dump(), repo_dicts)
    except Exception as exc:
        logger.warning("Falling back to deterministic retrieval plan: %s", exc)
        return build_default_retrieval_plan(idea, keywords, top_k=8)

    _cache.set_json("llm_retrieval_plan", cache_key, plan_dict)
    return _retrieval_plan_from_dict(plan_dict, keywords, idea)


def build_repo_chat_retrieval_query(
    question: str,
    idea_summary: str,
    messages: list[ChatMessage],
) -> RetrievalQuery:
    """Build a deterministic retrieval query for repo-scoped chat."""

    recent_context: list[str] = []
    for message in messages[-4:]:
        role = "User" if message.role == "user" else "Assistant"
        recent_context.append(f"{role}: {message.content.strip()[:240]}")

    query_parts: list[str] = []
    if idea_summary.strip():
        query_parts.append(f"Idea: {idea_summary.strip()}")
    if recent_context:
        query_parts.append("Recent context:\n" + "\n".join(recent_context))
    query_parts.append(f"Question: {question.strip()}")

    return RetrievalQuery(
        section="chat_answer",
        query="\n".join(query_parts),
        preferred_roles=_chat_preferred_roles(question),
        top_k=6,
        weights=RetrievalWeights(dense_weight=0.45, lexical_weight=0.25, repo_weight=0.1, role_weight=0.2),
    )


async def answer_repo_chat(
    question: str,
    idea_summary: str,
    messages: list[ChatMessage],
    repositories: list[RepoSearchResult],
    hits: list[RetrievalHit],
) -> dict[str, object]:
    """Generate a grounded chat answer over the indexed repo scope."""

    if not hits:
        return {
            "answer": (
                "I do not have enough indexed evidence in the current repo scope to answer that yet. "
                "Try asking about a specific repository, file, architecture area, or dependency."
            ),
            "follow_up_suggestions": [
                "Which repo is closest to the core architecture?",
                "Show me the most relevant files for this feature.",
                "What dependencies define the stack in these repos?",
            ],
        }

    llm = get_llm_client()
    repo_dicts = [
        {
            "full_name": repo.full_name,
            "description": repo.description or "",
            "fit_summary": repo.fit_summary,
            "reference_type": repo.reference_type,
        }
        for repo in repositories[:6]
    ]
    evidence = {
        "repositories": repo_dicts,
        "hits": [
            {
                "repo": hit.repo_full_name,
                "path": hit.path,
                "lines": [hit.start_line, hit.end_line],
                "reason": hit.reason,
                "snippet": _truncate_text(hit.text, 420),
            }
            for hit in hits[:6]
        ],
    }

    try:
        result = await llm.answer_repo_chat(
            question,
            idea_summary,
            [message.model_dump() for message in messages],
            repo_dicts,
            evidence,
        )
    except Exception as exc:
        logger.warning("Repo chat generation failed, falling back to a conservative answer: %s", exc)
        first_hit = hits[0]
        return {
            "answer": (
                f"I found relevant indexed evidence in {first_hit.repo_full_name} ({first_hit.path}), "
                "but I could not complete a full grounded answer right now."
            ),
            "follow_up_suggestions": [
                "Summarize the architecture from the top retrieved files.",
                "Which files should I inspect first?",
            ],
        }

    answer = str(result.get("answer", "")).strip()
    if not answer:
        answer = (
            "I found relevant indexed evidence, but the answer was not confident enough to return cleanly. "
            "Please ask a narrower repo or file-specific question."
        )

    follow_up_suggestions = [
        suggestion.strip()
        for suggestion in result.get("follow_up_suggestions", [])
        if isinstance(suggestion, str) and suggestion.strip()
    ][:3]

    return {
        "answer": answer,
        "follow_up_suggestions": follow_up_suggestions,
    }


async def generate_analysis(
    idea: str,
    keywords: ExtractedKeywords,
    repositories: list[RepoSearchResult],
    section_hits: dict[str, list[RetrievalHit]],
) -> AnalysisResponse:
    """Generate the final analysis using compact evidence packs and parallel sections."""

    llm = get_llm_client()
    repo_dicts = [
        {
            "full_name": repo.full_name,
            "relevance_score": repo.relevance_score,
            "url": repo.html_url,
            "description": repo.description or "",
            "fit_summary": repo.fit_summary,
            "reference_type": repo.reference_type,
        }
        for repo in repositories[:8]
    ]
    evidence_pack = _build_generation_evidence(repositories, section_hits)

    repo_descriptions_data, learning_path_data, architecture_diagram, tech_stack_data = await asyncio.gather(
        llm.generate_repo_descriptions(
            idea,
            keywords.model_dump(),
            repo_dicts,
            evidence_pack["repo_descriptions"],
        ),
        llm.generate_learning_path(
            idea,
            keywords.model_dump(),
            repo_dicts,
            evidence_pack["learning_path"],
        ),
        llm.generate_architecture_diagram(
            idea,
            keywords.model_dump(),
            repo_dicts,
            evidence_pack["architecture_diagram"],
        ),
        llm.generate_tech_stack(
            idea,
            keywords.model_dump(),
            repo_dicts,
            evidence_pack["tech_stack"],
        ),
    )

    learning_path = [
        LearningStep(
            step_number=item.get("step", index + 1),
            title=item.get("title", ""),
            description=item.get("description", ""),
            milestone=item.get("milestone", ""),
            concepts=item.get("concepts", [])[:6],
            resources=item.get("resources", [])[:6],
        )
        for index, item in enumerate(learning_path_data)
    ]
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
        repo_descriptions=repo_descriptions_data,
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
    for ambiguity in ambiguities:
        result.append(
            AmbiguityFlag(
                axis=ambiguity.get("axis", "unknown"),
                question=ambiguity.get("question", ""),
                reason=ambiguity.get("reason", ""),
                options=_normalize_clarification_options(ambiguity.get("options", [])),
                severity=ambiguity.get("severity", "medium"),
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
    """Coerce capability weights into a clean string-to-float mapping."""

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


def _normalize_ranking_weights(raw_weights: object) -> dict[str, float]:
    normalized = dict(_DEFAULT_RANKING_WEIGHTS)
    if not isinstance(raw_weights, dict):
        return normalized
    for key, value in raw_weights.items():
        if key not in normalized:
            continue
        try:
            normalized[key] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    total = sum(normalized.values()) or 1.0
    return {key: value / total for key, value in normalized.items()}


def _normalize_retrieval_weights(raw_weights: object) -> RetrievalWeights:
    if isinstance(raw_weights, RetrievalWeights):
        return raw_weights
    if not isinstance(raw_weights, dict):
        return _DEFAULT_RETRIEVAL_WEIGHTS.model_copy(deep=True)

    normalized = _DEFAULT_RETRIEVAL_WEIGHTS.model_dump()
    for key in normalized:
        try:
            normalized[key] = max(0.0, float(raw_weights.get(key, normalized[key])))
        except (TypeError, ValueError):
            continue
    total = sum(normalized.values()) or 1.0
    return RetrievalWeights(**{key: value / total for key, value in normalized.items()})


def _retrieval_plan_from_dict(plan_dict: dict, keywords: ExtractedKeywords, idea: str) -> RetrievalPlan:
    queries: list[RetrievalQuery] = []
    allowed_sections = {"repo_descriptions", "learning_path", "architecture_diagram", "tech_stack", "chat_answer"}
    for raw_query in plan_dict.get("queries", []):
        section = raw_query.get("section", "repo_descriptions")
        if section not in allowed_sections:
            section = "repo_descriptions"
        query_text = raw_query.get("query", "").strip()
        if not query_text:
            continue
        queries.append(
            RetrievalQuery(
                section=section,
                query=query_text,
                preferred_roles=[role for role in raw_query.get("preferred_roles", []) if isinstance(role, str)][:4],
                top_k=int(raw_query.get("top_k", 8) or 8),
                weights=_normalize_retrieval_weights(raw_query.get("weights", {})),
            )
        )

    if queries:
        return RetrievalPlan(queries=queries)
    return build_default_retrieval_plan(idea, keywords, top_k=8)


def _build_generation_evidence(
    repositories: list[RepoSearchResult],
    section_hits: dict[str, list[RetrievalHit]],
) -> dict[str, dict]:
    shared_repositories = [
        {
            "full_name": repo.full_name,
            "reference_type": repo.reference_type,
            "fit_summary": repo.fit_summary,
            "score": round(repo.relevance_score, 4),
            "description": (repo.description or "")[:220],
        }
        for repo in repositories[:5]
    ]

    def pack_hits(section: str) -> dict:
        hits = section_hits.get(section, [])
        return {
            "repositories": shared_repositories,
            "hits": [
                {
                    "repo": hit.repo_full_name,
                    "path": hit.path,
                    "reason": hit.reason,
                    "snippet": _truncate_text(hit.text, 360),
                }
                for hit in hits[:4]
            ],
        }

    return {
        "repo_descriptions": pack_hits("repo_descriptions"),
        "learning_path": pack_hits("learning_path"),
        "architecture_diagram": pack_hits("architecture_diagram"),
        "tech_stack": pack_hits("tech_stack"),
    }


def _truncate_text(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _chat_preferred_roles(question: str) -> list[str]:
    lowered = question.lower()

    if any(token in lowered for token in ("dependency", "dependencies", "package", "packages", "requirements", "env", "config")):
        return ["config", "documentation", "entrypoint", "source"]
    if any(token in lowered for token in ("architecture", "flow", "service", "router", "api", "component")):
        return ["entrypoint", "source", "config", "documentation"]
    if any(token in lowered for token in ("setup", "install", "run", "usage", "example", "examples", "how do i")):
        return ["documentation", "config", "entrypoint", "example"]
    return ["documentation", "entrypoint", "source", "config"]
