"""Model-based idea analysis using Groq API with warm-path caching."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
from urllib.parse import urljoin, urlparse

import httpx

from core.config import get_settings
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
_DEFAULT_RETRIEVAL_WEIGHTS = RetrievalWeights()


async def extract_keywords(idea: str, clarification_answers: dict[str, str] | None = None) -> ExtractedKeywords:
    """Extract structured technical intent from the user's idea using LLM."""

    cache_key = stable_cache_key(
        {
            "model": get_settings().LLM_MODEL,
            "idea": " ".join(idea.strip().lower().split()),
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
            "model": get_settings().LLM_MODEL,
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
        weights=RetrievalWeights(dense_weight=0.85, repo_weight=0.15),
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
                "snippet": _truncate_text(hit.text, 600),
            }
            for hit in hits[:8]
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

    validated_resources = await _validate_resource_links(
        [item.get("resources", [])[:6] for item in learning_path_data]
    )
    learning_path = [
        LearningStep(
            step_number=item.get("step", index + 1),
            title=item.get("title", ""),
            description=item.get("description", ""),
            milestone=item.get("milestone", ""),
            concepts=item.get("concepts", [])[:6],
            resources=validated_resources[index],
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

    evidence_types = _compute_evidence_types(repositories, section_hits)
    repositories_with_evidence = [
        repo.model_copy(update={"evidence_type": evidence_types.get(repo.full_name, "shallow_evidence")})
        for repo in repositories
    ]

    return AnalysisResponse(
        idea_summary=keywords.summary or idea,
        keywords=keywords,
        assumptions=keywords.assumptions,
        repositories=repositories_with_evidence,
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
            "declared_dependencies": repo.dependencies[:25],
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
                    "snippet": _truncate_text(hit.text, 900),
                }
                for hit in hits[:6]
            ],
        }

    return {
        "repo_descriptions": pack_hits("repo_descriptions"),
        "learning_path": pack_hits("learning_path"),
        "architecture_diagram": pack_hits("architecture_diagram"),
        "tech_stack": pack_hits("tech_stack"),
    }


def _compute_evidence_types(
    repositories: list[RepoSearchResult],
    section_hits: dict[str, list[RetrievalHit]],
) -> dict[str, str]:
    """Classify what actually grounds each repo's narrative claims.

    Deep-indexed retrieval hits are real evidence; a repo the pipeline only
    ever saw via shallow candidate-ranking (README/manifest) text is weaker
    but not nothing; a repo with neither should never sit next to citations
    looking as verified as one that does. Surfaced on the response so the UI
    can stop presenting all five selected repos as equally evidenced when
    background indexing has only finished for some of them.
    """
    deep_hit_repos = {hit.repo_full_name for hits in section_hits.values() for hit in hits}

    evidence_types: dict[str, str] = {}
    for repo in repositories:
        if repo.full_name in deep_hit_repos:
            evidence_types[repo.full_name] = "deep_retrieval"
        elif repo.description or repo.fit_summary:
            evidence_types[repo.full_name] = "shallow_evidence"
        else:
            evidence_types[repo.full_name] = "no_evidence"
    return evidence_types


_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"']+")
_MAX_LINKS_TO_VALIDATE = 20


async def _validate_resource_links(resource_lists: list[list[str]]) -> list[list[str]]:
    """Drop learning-path resource links that don't actually resolve.

    The model occasionally invents a plausible-looking but nonexistent
    "further reading" URL alongside genuinely correct ones in the same
    response. A bounded, concurrent check catches most of these before a
    user clicks one. Ambiguous results (timeout, DNS failure, 4xx/5xx,
    bot-blocked) are treated as "drop the link, keep the descriptive text"
    rather than assumed valid -- a wrongly-dropped real link costs far less
    trust than a kept fake one.
    """
    all_urls: list[str] = []
    for resources in resource_lists:
        for resource in resources:
            match = _URL_PATTERN.search(resource)
            if match:
                all_urls.append(match.group(0).rstrip(".,;:)]'\""))

    unique_urls = list(dict.fromkeys(all_urls))[:_MAX_LINKS_TO_VALIDATE]
    if not unique_urls:
        return resource_lists

    validity = await _check_urls(unique_urls)

    def clean(resource: str) -> str:
        match = _URL_PATTERN.search(resource)
        if not match:
            return resource
        url = match.group(0).rstrip(".,;:)]'\"")
        if validity.get(url, True):
            return resource
        stripped = (resource[: match.start()] + resource[match.end():]).strip(" -–—:")
        return stripped or resource

    return [[clean(resource) for resource in resources] for resources in resource_lists]


_MAX_LINK_REDIRECTS = 5


async def _is_public_http_url(url: str) -> bool:
    """Reject URLs that would make this server request its own network.

    Resource links are LLM output shaped by untrusted README text, so a
    hostile repo could plant e.g. a cloud-metadata or localhost URL; every
    address the hostname resolves to must be globally routable.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parsed.hostname, parsed.port or None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            return False
    return bool(infos)


async def _check_urls(urls: list[str]) -> dict[str, bool]:
    results: dict[str, bool] = {}

    async def fetch_status(client: httpx.AsyncClient, url: str, method: str) -> int | None:
        # Redirects are followed by hand so every hop is re-checked; a public
        # URL could otherwise 302 to an internal address.
        for _ in range(_MAX_LINK_REDIRECTS + 1):
            if not await _is_public_http_url(url):
                return None
            response = await client.request(method, url, timeout=2.5)
            location = response.headers.get("location")
            if response.status_code in (301, 302, 303, 307, 308) and location:
                url = urljoin(url, location)
                continue
            return response.status_code
        return None

    async def check(client: httpx.AsyncClient, url: str) -> None:
        for method in ("HEAD", "GET"):
            try:
                status = await fetch_status(client, url, method)
            except httpx.HTTPError:
                continue
            if status is None:
                break
            if status == 405 and method == "HEAD":
                continue
            results[url] = status < 400
            return
        results[url] = False

    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            headers={"User-Agent": "GitAssistAI-LinkValidator/1.0"},
        ) as client:
            await asyncio.gather(*(check(client, url) for url in urls))
    except Exception as exc:
        logger.warning("Resource link validation failed outright, keeping all links unchanged: %s", exc)
        return {url: True for url in urls}

    dropped = sum(1 for ok in results.values() if not ok)
    if dropped:
        logger.info("Resource link validation: dropped %d/%d unresolved URLs", dropped, len(urls))
    return results


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
