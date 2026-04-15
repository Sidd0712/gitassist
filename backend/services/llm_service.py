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
        trivial_capabilities=[],
        capability_weights={},
        domain_terms=extracted.get("keywords", [])[:12],
        tech_terms=extracted.get("frameworks", []) + extracted.get("languages", [])[:10],
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
    """Build clarification questions from detected ambiguities."""

    questions: list[ClarificationQuestion] = []
    
    # Define default options for common ambiguity axes
    option_map = {
        "scale": ["MVP/Prototype", "Production-ready", "Enterprise-scale"],
        "platform": ["Web only", "Mobile (iOS/Android)", "Both web and mobile", "Desktop app"],
        "auth": ["Simple email/password", "Social login (Google, etc.)", "Enterprise SSO"],
        "data_storage": ["Simple database", "Real-time sync", "Offline-first"],
        "payments": ["Not needed", "One-time payments", "Subscriptions", "Marketplace with escrow"],
        "feature_priority": keywords.primary_capabilities[:4] if keywords.primary_capabilities else ["All features equally"],
        "deployment": ["Cloud (AWS/GCP/Azure)", "Self-hosted", "Serverless"],
        "realtime": ["Not needed", "Basic updates", "Real-time collaboration"],
    }
    
    for ambiguity in keywords.ambiguities:
        if ambiguity.resolved or ambiguity.severity != "high":
            continue
        
        # Get options for this ambiguity axis, or provide generic options
        options = option_map.get(ambiguity.axis, [
            "Keep it simple", 
            "Standard approach", 
            "Advanced features"
        ])
        
        question = ClarificationQuestion(
            key=ambiguity.axis,
            question=ambiguity.reason,
            options=options,
            reason=ambiguity.reason,
        )
        questions.append(question)
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
                reason=amb.get("reason", ""),
                severity=amb.get("severity", "medium"),
                resolved=False,
                answer=None,
            )
        )
    return result
