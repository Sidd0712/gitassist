"""GitHub service for candidate search, staged repo ingestion, and deep file fetches."""

from __future__ import annotations

import asyncio
import base64
import logging
import math
from collections import defaultdict

import httpx
from cachetools import TTLCache

from core.config import get_settings
from models.schemas import ExtractedKeywords, RepoFetchPlan, RepoFile, RepoSearchResult, RepoSnapshot, RepoTreeEntry, ShallowRepoEvidence
from services.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_cache: TTLCache = TTLCache(maxsize=512, ttl=get_settings().GITHUB_CACHE_TTL)

# Common paths to skip during file fetching
SKIP_PATH_PARTS = {
    "node_modules", "vendor", ".git", "dist", "build", "coverage",
    "test", "tests", "__pycache__", ".next", ".nuxt", "target",
    "bin", "obj", "packages", ".vscode", ".idea"
}

# Important configuration and documentation files
IMPORTANT_FILENAMES = {
    "package.json", "requirements.txt", "cargo.toml", "go.mod",
    "gemfile", "composer.json", "pom.xml", "build.gradle",
    "dockerfile", "docker-compose.yml", ".env.example",
    "config.yml", "config.yaml", "tsconfig.json"
}

# Source code file extensions to fetch
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".r", ".m", ".vue", ".svelte", ".dart",
    ".sh", ".bash", ".sql", ".graphql", ".proto", ".yaml", ".yml",
    ".json", ".xml", ".md", ".txt", ".toml", ".ini", ".cfg"
}


def _headers() -> dict[str, str]:
    settings = get_settings()
    headers = {"Accept": "application/vnd.github+json"}
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
    return headers


def _should_fetch(path: str, size: int) -> bool:
    settings = get_settings()
    lowered = path.lower()
    if any(token in lowered for token in SKIP_PATH_PARTS):
        return False
    if size > settings.RAG_MAX_FILE_SIZE:
        return False

    filename = lowered.rsplit("/", 1)[-1]
    if filename in IMPORTANT_FILENAMES or filename.startswith("readme"):
        return True

    extension = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    return extension in SOURCE_EXTENSIONS


def _path_priority(path: str, idea_terms: set[str]) -> int:
    lowered = path.lower()
    filename = lowered.rsplit("/", 1)[-1]
    score = 0
    if filename.startswith("readme"):
        score += 100
    if filename in IMPORTANT_FILENAMES:
        score += 90
    if any(token in lowered for token in ("/docs/", "/doc/", "guide", "tutorial")):
        score += 70
    if any(token in lowered for token in ("router", "route", "service", "controller", "handler")):
        score += 60
    if any(token in lowered for token in ("main.", "app.", "server.", "api.", "index.")):
        score += 50
    if any(token in lowered for token in ("example", "demo", "sample")):
        score += 35
    if any(token in lowered for token in ("test", "spec")):
        score += 20
    score += sum(5 for term in idea_terms if term in lowered)
    return score


async def build_search_queries(keywords: ExtractedKeywords) -> list[str]:
    """Build multiple GitHub search queries using AI."""
    
    from services.llm_client import get_llm_client
    
    settings = get_settings()
    llm = get_llm_client()
    
    idea_context = f"{keywords.core_intent or keywords.summary}. Capabilities: {', '.join(keywords.primary_capabilities or keywords.capabilities[:3])}"
    
    # Generate queries for primary capabilities using AI
    queries: list[str] = []
    
    for capability in (keywords.primary_capabilities or keywords.capabilities[:3]):
        try:
            capability_queries = await llm.generate_search_queries(capability, idea_context, num_queries=2)
            queries.extend(capability_queries)
        except Exception as exc:
            logger.warning("Failed to generate AI queries for %s: %s", capability, exc)
            # Simple fallback query without hardcoded terms
            queries.append(f"{capability} {keywords.product_type or ''}")
    
    # Add broad domain query
    if keywords.domain_terms:
        queries.append(" ".join(keywords.domain_terms[:3]))
    
    # Add language-qualified query if specified
    if keywords.languages:
        qualifier = " ".join([keywords.product_type or "", keywords.core_intent or ""][:2]).strip()
        if qualifier:
            queries.append(f"{qualifier} language:{keywords.languages[0]}")
    
    # Deduplicate
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(part for part in query.split() if part).strip()
        if not normalized or normalized.lower() in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized.lower())
        if len(deduped) >= settings.RAG_QUERY_LIMIT:
            break
    
    return deduped or [keywords.summary]  # Ultimate fallback to avoid empty list


def _clean_query_terms(terms: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.strip().lower().replace(",", "")
        if not normalized or normalized in {"and", "for", "the", "with", "api", "backend", "client"}:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def _query_join(terms: list[str]) -> str:
    return " ".join(term for term in terms if term).strip()


def _language_bucket(language: str | None) -> str:
    return (language or "Unknown").strip() or "Unknown"


# _candidate_quality_penalty and _candidate_quality_score removed - using AI evaluation


def _intent_search_text(keywords: ExtractedKeywords) -> str:
    terms = [
        keywords.core_intent or keywords.summary,
        " ".join(keywords.primary_capabilities[:3]),
        " ".join(keywords.secondary_capabilities[:3]),
        " ".join(keywords.domain_terms[:6]),
    ]
    return " ".join(term for term in terms if term).strip()


def _repo_metadata_text(repo: RepoSearchResult) -> str:
    return "\n".join(
        [
            repo.full_name,
            repo.description or "",
            " ".join(repo.topics),
        ]
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


async def search_repo_candidates(keywords: ExtractedKeywords) -> list[RepoSearchResult]:
    """Search GitHub using multiple query variants and merge candidates."""

    settings = get_settings()
    queries = await build_search_queries(keywords)
    if not queries:
        return []

    merged: dict[str, RepoSearchResult] = {}
    query_map: defaultdict[str, set[str]] = defaultdict(set)
    embedding_service = EmbeddingService()
    intent_text = _intent_search_text(keywords)

    async with httpx.AsyncClient(timeout=30.0) as client:
        semaphore = asyncio.Semaphore(settings.RAG_MAX_SEARCH_CONCURRENCY)

        async def fetch_query(query: str) -> tuple[str, list[dict]]:
            cache_key = f"search:{query}:{settings.RAG_SEARCH_PER_QUERY}"
            if cache_key in _cache:
                return query, _cache[cache_key]
            async with semaphore:
                response = await client.get(
                    f"{settings.GITHUB_API_BASE}/search/repositories",
                    headers=_headers(),
                    params={
                        "q": query,
                        "per_page": settings.RAG_SEARCH_PER_QUERY,
                    },
                )
                response.raise_for_status()
                items = response.json().get("items", [])
                _cache[cache_key] = items
                return query, items

        query_results = await asyncio.gather(*(fetch_query(query) for query in queries), return_exceptions=True)
        for result in query_results:
            if isinstance(result, Exception):
                logger.warning("GitHub query failed during candidate discovery: %s", result)
                continue
            query, items = result
            for item in items:
                existing = merged.get(item["full_name"])
                repo = RepoSearchResult(
                    full_name=item["full_name"],
                    description=item.get("description"),
                    html_url=item["html_url"],
                    stars=item.get("stargazers_count", 0),
                    language=item.get("language"),
                    topics=item.get("topics", []),
                    archived=item.get("archived", False),
                    updated_at=item.get("updated_at"),
                )
                if existing is None or repo.stars > existing.stars:
                    merged[repo.full_name] = repo
                query_map[repo.full_name].add(query)

    if not merged:
        return []

    candidates = list(merged.values())
    query_hit_max = max((len(query_map[repo.full_name]) for repo in candidates), default=1)
    intent_embedding = await embedding_service.embed_query(intent_text)
    metadata_embeddings = await embedding_service.embed_documents([_repo_metadata_text(repo) for repo in candidates])

    # Use AI to determine optimal scoring weights
    from services.llm_client import get_llm_client
    llm = get_llm_client()
    
    try:
        weights = await llm.calculate_ranking_weights(keywords.summary or keywords.core_intent, len(candidates))
    except Exception as exc:
        logger.warning("Failed to get AI ranking weights: %s. Using semantic-first defaults.", exc)
        weights = {
            "readme_semantic": 0.7,
            "metadata_semantic": 0.15,
            "capability_coverage": 0.1,
            "doc_quality": 0.0,
            "query_diversity": 0.05,
            "star_quality": 0.0,
        }
    
    preliminary: list[RepoSearchResult] = []
    for repo, embedding in zip(candidates, metadata_embeddings):
        repo.query_hit_count = len(query_map[repo.full_name])
        repo.semantic_meta_score = round(_cosine_similarity(intent_embedding, embedding), 5)
        query_score = repo.query_hit_count / max(query_hit_max, 1)
        
        # Simple quality indicators without hardcoded patterns
        is_archived = repo.archived
        has_recent_activity = repo.updated_at is not None
        star_score = min(math.log10(repo.stars + 10) / 4.0, 1.0)
        quality_score = star_score * (0.5 if is_archived else 1.0) * (1.1 if has_recent_activity else 0.9)
        
        coverage_overlap = len(
            {
                term.lower()
                for term in [*keywords.primary_capabilities[:3], *keywords.domain_terms[:6]]
                if term and term.lower() in _repo_metadata_text(repo).lower()
            }
        ) / max(len(keywords.primary_capabilities[:3]) + len(keywords.domain_terms[:6]), 1)
        
        # Use AI-determined weights
        repo.relevance_score = round(
            max(
                0.0,
                (weights.get("metadata_semantic", 0.6) * repo.semantic_meta_score)
                + (weights.get("query_diversity", 0.2) * query_score)
                + (weights.get("capability_coverage", 0.1) * coverage_overlap)
                + (weights.get("star_quality", 0.1) * quality_score),
            ),
            5,
        )
        repo.rank_reasons = [
            f"metadata semantic score {repo.semantic_meta_score:.2f}",
            f"matched {repo.query_hit_count} query families",
        ]
        preliminary.append(repo)

    preliminary.sort(key=lambda repo: repo.relevance_score, reverse=True)
    
    # Apply diversity constraints from config (not hardcoded)
    language_counts: defaultdict[str, int] = defaultdict(int)
    shortlisted: list[RepoSearchResult] = []
    max_per_language = settings.RAG_MAX_PER_LANGUAGE if hasattr(settings, 'RAG_MAX_PER_LANGUAGE') else 999
    
    for repo in preliminary:
        bucket = _language_bucket(repo.language)
        if language_counts[bucket] >= max_per_language:
            continue
        shortlisted.append(repo)
        language_counts[bucket] += 1
        if len(shortlisted) >= settings.RAG_CANDIDATE_REPO_LIMIT:
            break

    candidates = shortlisted
    logger.info("GitHub candidate search produced %d repositories across %d queries", len(candidates), len(queries))
    return candidates


async def fetch_repo_snapshot(repo: RepoSearchResult) -> RepoSnapshot:
    """Fetch repo metadata, commit SHA, and tree."""

    settings = get_settings()
    cache_key = f"snapshot:{repo.full_name}"
    if cache_key in _cache:
        return RepoSnapshot(**_cache[cache_key])

    async with httpx.AsyncClient(timeout=30.0) as client:
        repo_response = await client.get(
            f"{settings.GITHUB_API_BASE}/repos/{repo.full_name}",
            headers=_headers(),
        )
        repo_response.raise_for_status()
        repo_data = repo_response.json()

        default_branch = repo_data.get("default_branch") or repo.default_branch or "HEAD"
        commit_sha = repo.commit_sha or default_branch
        commit_response = await client.get(
            f"{settings.GITHUB_API_BASE}/repos/{repo.full_name}/commits/{default_branch}",
            headers=_headers(),
        )
        if commit_response.status_code == 200:
            commit_sha = commit_response.json().get("sha", "") or commit_sha
        else:
            logger.warning(
                "Could not resolve commit SHA for %s on branch %s (status %d); using fallback key",
                repo.full_name,
                default_branch,
                commit_response.status_code,
            )

        tree_response = await client.get(
            f"{settings.GITHUB_API_BASE}/repos/{repo.full_name}/git/trees/{default_branch}",
            headers=_headers(),
            params={"recursive": 1},
        )
        tree_entries: list[RepoTreeEntry] = []
        if tree_response.status_code == 200:
            tree_entries = [
                RepoTreeEntry(path=item["path"], type=item.get("type", "blob"), size=item.get("size", 0))
                for item in tree_response.json().get("tree", [])
                if item.get("type") == "blob"
            ]
        else:
            logger.warning(
                "Could not fetch tree for %s on branch %s (status %d)",
                repo.full_name,
                default_branch,
                tree_response.status_code,
            )

    snapshot_payload = repo.model_dump()
    snapshot_payload.update(
        {
            "description": repo_data.get("description") or repo.description,
            "html_url": repo_data.get("html_url") or repo.html_url,
            "stars": repo_data.get("stargazers_count", repo.stars),
            "language": repo_data.get("language") or repo.language,
            "topics": repo_data.get("topics", repo.topics),
            "default_branch": default_branch,
            "updated_at": repo_data.get("updated_at"),
            "archived": repo_data.get("archived", False),
            "commit_sha": commit_sha,
            "tree": [entry.model_dump() for entry in tree_entries],
        }
    )
    snapshot = RepoSnapshot(**snapshot_payload)
    _cache[cache_key] = snapshot.model_dump()
    return snapshot


def _root_manifest_candidates(language: str | None) -> list[str]:
    base = ["package.json", "pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml", "Dockerfile", "docker-compose.yml"]
    lowered = (language or "").lower()
    if lowered in {"typescript", "javascript"}:
        return ["package.json", "pnpm-lock.yaml", "package-lock.json", "Dockerfile", "docker-compose.yml"]
    if lowered == "python":
        return ["pyproject.toml", "requirements.txt", "poetry.lock", "Dockerfile", "docker-compose.yml"]
    if lowered == "go":
        return ["go.mod", "Dockerfile", "docker-compose.yml"]
    if lowered in {"rust", "java", "kotlin"}:
        return ["Cargo.toml", "pom.xml", "build.gradle", "Dockerfile", "docker-compose.yml"]
    return base


async def fetch_shallow_repo_evidence(repository: RepoSearchResult) -> ShallowRepoEvidence:
    """Fetch README and root manifests for semantic reranking."""

    readme_candidates = ["README.md", "readme.md", "README", "docs/README.md"]
    manifest_candidates = _root_manifest_candidates(repository.language)[:5]
    manifest_files: list[RepoFile] = []
    readme_text = ""
    readme_path: str | None = None
    highlighted_paths: list[str] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        readme_cache_key = f"readme:{repository.full_name}"
        if readme_cache_key in _cache:
            readme_text = _cache[readme_cache_key]
            readme_path = "README.md" if readme_text else None
        else:
            response = await client.get(
                f"{get_settings().GITHUB_API_BASE}/repos/{repository.full_name}/readme",
                headers={**_headers(), "Accept": "application/vnd.github.raw+json"},
            )
            if response.status_code == 200:
                readme_text = response.text
                readme_path = "README.md"
                _cache[readme_cache_key] = readme_text
        for path in readme_candidates:
            if readme_text:
                break
            readme_text = await _fetch_file_content(client, repository.full_name, path) or ""
            if readme_text:
                readme_path = path
                break
        for path in manifest_candidates:
            content = await _fetch_file_content(client, repository.full_name, path)
            if content:
                manifest_files.append(RepoFile(path=path, content=content, size=len(content)))
                highlighted_paths.append(path)

    if readme_path:
        highlighted_paths.insert(0, readme_path)

    return ShallowRepoEvidence(
        repository=repository.model_copy(deep=True),
        readme_path=readme_path,
        readme=readme_text,
        manifest_files=manifest_files,
        highlighted_paths=highlighted_paths[:10],
    )


async def fetch_shallow_repo_evidence_batch(repositories: list[RepoSearchResult]) -> list[ShallowRepoEvidence]:
    """Fetch shallow evidence for a shortlist with bounded concurrency."""

    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.RAG_MAX_SEARCH_CONCURRENCY)

    async def fetch(repository: RepoSearchResult) -> ShallowRepoEvidence | None:
        async with semaphore:
            try:
                return await fetch_shallow_repo_evidence(repository)
            except Exception as exc:
                logger.warning("Skipping candidate %s during shallow ingest: %s", repository.full_name, exc)
                return None

    results = await asyncio.gather(*(fetch(repository) for repository in repositories))
    return [item for item in results if item is not None]


def build_repo_fetch_plan(snapshot: RepoSnapshot, shallow: ShallowRepoEvidence, intent: ExtractedKeywords) -> RepoFetchPlan:
    """Select the highest-value files for deep indexing."""

    settings = get_settings()
    idea_terms = {
        term.lower()
        for term in [
            *intent.domain_terms[:8],
            *intent.primary_capabilities[:4],
            *intent.secondary_capabilities[:4],
            *intent.keywords[:6],
            *intent.likely_components[:4],
            *shallow.matched_capabilities[:4],
        ]
        if len(term) > 2
    }
    max_files = min(settings.RAG_MAX_FILES_PER_REPO, 30)
    max_chars = min(settings.RAG_MAX_CHARS_PER_REPO, 150_000)
    prioritized = sorted(
        (
            entry
            for entry in snapshot.tree
            if entry.type == "blob" and _should_fetch(entry.path, entry.size)
        ),
        key=lambda entry: (_path_priority(entry.path, idea_terms), entry.size * -1),
        reverse=True,
    )

    selected_paths: list[str] = []
    skipped_paths: list[str] = []
    estimated_chars = 0
    selected_set: set[str] = set()

    readme_seed = RepoFile(
        path=shallow.readme_path or "README.md",
        content=shallow.readme,
        size=len(shallow.readme),
    )
    for repo_file in [*shallow.manifest_files, readme_seed]:
        if repo_file.content and repo_file.path not in selected_set:
            selected_paths.append(repo_file.path)
            selected_set.add(repo_file.path)
            estimated_chars += repo_file.size

    for entry in prioritized:
        if entry.path in selected_set:
            continue
        if len(selected_paths) >= max_files:
            skipped_paths.append(entry.path)
            continue
        if estimated_chars + entry.size > max_chars:
            skipped_paths.append(entry.path)
            continue
        selected_paths.append(entry.path)
        selected_set.add(entry.path)
        estimated_chars += entry.size

    rationale = [
        "Prioritized README, manifests, and paths aligned with the resolved product capabilities first.",
        "Skipped generated/binary/oversized files and capped total indexed characters.",
    ]

    return RepoFetchPlan(
        repository=shallow.repository.model_copy(deep=True),
        selected_paths=selected_paths,
        skipped_paths=skipped_paths[:20],
        estimated_chars=estimated_chars,
        rationale=rationale,
    )


async def fetch_repo_files_for_indexing(plan: RepoFetchPlan) -> list[RepoFile]:
    """Fetch the planned files for indexing."""

    files: list[RepoFile] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for path in plan.selected_paths:
            content = await _fetch_file_content(client, plan.repository.full_name, path)
            if not content:
                continue
            files.append(RepoFile(path=path, content=content, size=len(content)))
    logger.info("Fetched %d planned files for %s", len(files), plan.repository.full_name)
    return files


async def _fetch_file_content(client: httpx.AsyncClient, full_name: str, path: str) -> str | None:
    """Fetch and decode a repository file via the contents API."""

    settings = get_settings()
    cache_key = f"file:{full_name}:{path}"
    if cache_key in _cache:
        return _cache[cache_key]

    response = await client.get(
        f"{settings.GITHUB_API_BASE}/repos/{full_name}/contents/{path}",
        headers=_headers(),
    )
    if response.status_code != 200:
        return None

    data = response.json()
    if data.get("encoding") != "base64" or not data.get("content"):
        return None
    try:
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive decode guard
        return None
    _cache[cache_key] = content
    return content
