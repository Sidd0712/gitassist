"""Helpers for building retrieval plans for grounded synthesis."""

from __future__ import annotations

from models.schemas import ExtractedKeywords, RetrievalPlan, RetrievalQuery


def build_default_retrieval_plan(idea: str, keywords: ExtractedKeywords, top_k: int) -> RetrievalPlan:
    """Create a deterministic fallback retrieval plan."""

    core_terms = ", ".join(
        [term for term in (keywords.summary, *keywords.keywords[:4], *keywords.frameworks[:3], *keywords.languages[:2]) if term]
    )
    scope = core_terms or idea

    return RetrievalPlan(
        queries=[
            RetrievalQuery(
                section="repo_descriptions",
                query=f"{scope}. Focus on repository purpose, main modules, and implementation patterns.",
                preferred_roles=["documentation", "entrypoint", "source", "config"],
                top_k=top_k,
            ),
            RetrievalQuery(
                section="learning_path",
                query=f"{scope}. Focus on setup, examples, tutorials, and onboarding steps.",
                preferred_roles=["documentation", "example", "test", "config"],
                top_k=top_k,
            ),
            RetrievalQuery(
                section="architecture_diagram",
                query=f"{scope}. Focus on architecture, components, services, routers, and system flow.",
                preferred_roles=["entrypoint", "source", "config", "documentation"],
                top_k=top_k,
            ),
            RetrievalQuery(
                section="tech_stack",
                query=f"{scope}. Focus on dependencies, packages, frameworks, runtimes, and deployment configuration.",
                preferred_roles=["config", "documentation", "entrypoint"],
                top_k=top_k,
            ),
        ]
    )
