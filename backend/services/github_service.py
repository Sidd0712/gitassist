"""GitHub service for candidate search, staged repo ingestion, and deep file fetches."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import posixpath
import zipfile
from collections import defaultdict

from cachetools import TTLCache

from core.config import get_settings
from models.schemas import ExtractedKeywords, RepoFetchPlan, RepoFile, RepoSearchResult, RepoSnapshot, RepoTreeEntry, ShallowRepoEvidence
from services.github_client_service import get_github_client
from services.pipeline_cache_service import PipelineCacheService, stable_cache_key
from services.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_cache: TTLCache = TTLCache(maxsize=512, ttl=get_settings().GITHUB_CACHE_TTL)
_persistent_cache = PipelineCacheService()
_SEARCH_RANKING_WEIGHTS = {
    "metadata_semantic": 0.65,
    "query_diversity": 0.2,
    "capability_coverage": 0.1,
    "star_quality": 0.05,
}

SKIP_PATH_PARTS = {
    "node_modules", "vendor", ".git", "dist", "build", "coverage",
    "test", "tests", "__pycache__", ".next", ".nuxt", "target",
    "bin", "obj", "packages", ".vscode", ".idea",
}

IMPORTANT_FILENAMES = {
    "package.json", "requirements.txt", "cargo.toml", "go.mod",
    "gemfile", "composer.json", "pom.xml", "build.gradle",
    "dockerfile", "docker-compose.yml", ".env.example",
    "config.yml", "config.yaml", "tsconfig.json",
}

SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".r", ".m", ".vue", ".svelte", ".dart",
    ".sh", ".bash", ".sql", ".graphql", ".proto", ".yaml", ".yml",
    ".json", ".xml", ".md", ".txt", ".toml", ".ini", ".cfg",
}


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    settings = get_settings()
    headers = {"Accept": accept}
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


def _normalize_archive_path(member_name: str) -> str:
    normalized = member_name.replace("\\", "/").strip("/")
    if not normalized:
        return ""
    parts = normalized.split("/", 1)
    if len(parts) == 1:
        return ""
    return posixpath.normpath(parts[1]).lstrip("./")


def _looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    text_bytes = sum(1 for byte in data[:1024] if 9 <= byte <= 13 or 32 <= byte <= 126)
    return text_bytes / max(len(data[:1024]), 1) < 0.7


async def build_search_queries(keywords: ExtractedKeywords) -> list[str]:
    """Build multiple GitHub search queries using one model call."""

    settings = get_settings()
    cache_key = stable_cache_key(
        {
            "summary": keywords.summary,
            "core_intent": keywords.core_intent,
            "product_type": keywords.product_type,
            "primary_capabilities": keywords.primary_capabilities,
            "secondary_capabilities": keywords.secondary_capabilities,
            "domain_terms": keywords.domain_terms,
            "languages": keywords.languages,
        }
    )
    cached = _persistent_cache.get_json("llm_search_queries", cache_key)
    if isinstance(cached, list):
        return [query for query in cached if isinstance(query, str)]

    from services.llm_client import get_llm_client

    llm = get_llm_client()
    idea_context = f"{keywords.core_intent or keywords.summary}. Capabilities: {', '.join(keywords.primary_capabilities or keywords.capabilities[:3])}"
    queries: list[str] = []

    try:
        queries = await llm.generate_search_queries_batch(
            idea_context=idea_context,
            capabilities=keywords.primary_capabilities or keywords.capabilities[:4],
            domain_terms=keywords.domain_terms or keywords.keywords,
            languages=keywords.languages,
            max_queries=settings.RAG_QUERY_LIMIT,
        )
    except Exception as exc:
        logger.warning("Failed to generate batched AI search queries: %s", exc)

    if not queries:
        for capability in keywords.primary_capabilities or keywords.capabilities[:3]:
            try:
                queries.extend(await llm.generate_search_queries(capability, idea_context, num_queries=2))
            except Exception as exc:
                logger.warning("Failed to generate fallback queries for %s: %s", capability, exc)
                queries.append(f"{capability} {keywords.product_type or ''}".strip())

    if keywords.domain_terms:
        queries.append(" ".join(keywords.domain_terms[:3]))
    if keywords.languages:
        qualifier = " ".join([keywords.product_type or "", keywords.core_intent or ""][:2]).strip()
        if qualifier:
            queries.append(f"{qualifier} language:{keywords.languages[0]}")

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

    final_queries = deduped or [keywords.summary]
    _persistent_cache.set_json("llm_search_queries", cache_key, final_queries)
    return final_queries


def _language_bucket(language: str | None) -> str:
    return (language or "Unknown").strip() or "Unknown"


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
    client = get_github_client()
    semaphore = asyncio.Semaphore(settings.RAG_MAX_SEARCH_CONCURRENCY)

    async def fetch_query(query: str) -> tuple[str, list[dict]]:
        cache_payload = {"query": query, "per_page": settings.RAG_SEARCH_PER_QUERY}
        cache_key = stable_cache_key(cache_payload)
        memory_key = f"search:{query}:{settings.RAG_SEARCH_PER_QUERY}"
        if memory_key in _cache:
            return query, _cache[memory_key]
        cached = _persistent_cache.get_json("github_search", cache_key)
        if isinstance(cached, list):
            _cache[memory_key] = cached
            return query, cached

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
            _cache[memory_key] = items
            _persistent_cache.set_json("github_search", cache_key, items)
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
    weights = dict(_SEARCH_RANKING_WEIGHTS)

    preliminary: list[RepoSearchResult] = []
    for repo, embedding in zip(candidates, metadata_embeddings):
        repo.query_hit_count = len(query_map[repo.full_name])
        repo.semantic_meta_score = round(_cosine_similarity(intent_embedding, embedding), 5)
        query_score = repo.query_hit_count / max(query_hit_max, 1)
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
    language_counts: defaultdict[str, int] = defaultdict(int)
    shortlisted: list[RepoSearchResult] = []

    for repo in preliminary:
        bucket = _language_bucket(repo.language)
        if language_counts[bucket] >= settings.RAG_MAX_PER_LANGUAGE:
            continue
        shortlisted.append(repo)
        language_counts[bucket] += 1
        if len(shortlisted) >= settings.RAG_CANDIDATE_REPO_LIMIT:
            break

    logger.info("GitHub candidate search produced %d repositories across %d queries", len(shortlisted), len(queries))
    return shortlisted


async def fetch_repo_snapshot(repo: RepoSearchResult) -> RepoSnapshot:
    """Fetch repo metadata, commit SHA, and tree."""

    settings = get_settings()
    cache_key = stable_cache_key({"repo": repo.full_name, "updated_at": repo.updated_at})
    memory_key = f"snapshot:{repo.full_name}"
    if memory_key in _cache:
        return RepoSnapshot(**_cache[memory_key])
    cached = _persistent_cache.get_json("github_snapshot", cache_key)
    if isinstance(cached, dict):
        _cache[memory_key] = cached
        return RepoSnapshot(**cached)

    client = get_github_client()
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
    payload = snapshot.model_dump()
    _cache[memory_key] = payload
    _persistent_cache.set_json("github_snapshot", cache_key, payload)
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

    cache_key = stable_cache_key(
        {
            "repo": repository.full_name,
            "updated_at": repository.updated_at,
            "language": repository.language,
        }
    )
    cached = _persistent_cache.get_json("github_shallow", cache_key)
    if isinstance(cached, dict):
        return ShallowRepoEvidence(**cached)

    readme_candidates = ["README.md", "readme.md", "README", "docs/README.md"]
    manifest_candidates = _root_manifest_candidates(repository.language)[:5]
    manifest_files: list[RepoFile] = []
    readme_text = ""
    readme_path: str | None = None
    highlighted_paths: list[str] = []
    client = get_github_client()

    readme_cache_key = f"readme:{repository.full_name}"
    if readme_cache_key in _cache:
        readme_text = _cache[readme_cache_key]
        readme_path = "README.md" if readme_text else None
    else:
        response = await client.get(
            f"{get_settings().GITHUB_API_BASE}/repos/{repository.full_name}/readme",
            headers=_headers("application/vnd.github.raw+json"),
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

    async def fetch_manifest(path: str) -> tuple[str, str | None]:
        return path, await _fetch_file_content(client, repository.full_name, path)

    manifest_results = await asyncio.gather(*(fetch_manifest(path) for path in manifest_candidates))
    for path, content in manifest_results:
        if not content:
            continue
        manifest_files.append(RepoFile(path=path, content=content, size=len(content)))
        highlighted_paths.append(path)

    if readme_path:
        highlighted_paths.insert(0, readme_path)

    evidence = ShallowRepoEvidence(
        repository=repository.model_copy(deep=True),
        readme_path=readme_path,
        readme=readme_text,
        manifest_files=manifest_files,
        highlighted_paths=highlighted_paths[:10],
    )
    _persistent_cache.set_json("github_shallow", cache_key, evidence.model_dump())
    return evidence


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


def build_repo_fetch_plan(
    snapshot: RepoSnapshot,
    intent: ExtractedKeywords | ShallowRepoEvidence,
    shallow: ShallowRepoEvidence | ExtractedKeywords | None = None,
) -> RepoFetchPlan:
    """Select the highest-value files for deep indexing."""

    if isinstance(intent, ShallowRepoEvidence) and isinstance(shallow, ExtractedKeywords):
        intent, shallow = shallow, intent
    if not isinstance(intent, ExtractedKeywords):
        raise TypeError("build_repo_fetch_plan expects ExtractedKeywords as the resolved intent.")

    settings = get_settings()
    idea_terms = {
        term.lower()
        for term in [
            *intent.domain_terms[:8],
            *intent.primary_capabilities[:4],
            *intent.secondary_capabilities[:4],
            *intent.keywords[:6],
            *intent.likely_components[:4],
            *(shallow.matched_capabilities[:4] if shallow else []),
        ]
        if len(term) > 2
    }
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

    for entry in prioritized:
        if entry.path in selected_set:
            continue
        if len(selected_paths) >= settings.RAG_MAX_FILES_PER_REPO:
            skipped_paths.append(entry.path)
            continue
        if estimated_chars + entry.size > settings.RAG_MAX_CHARS_PER_REPO:
            skipped_paths.append(entry.path)
            continue
        selected_paths.append(entry.path)
        selected_set.add(entry.path)
        estimated_chars += entry.size

    rationale = [
        "Selected files directly from the full Git tree instead of probing only README or manifest files.",
        "Prioritized paths aligned with the resolved capabilities, then included additional eligible source/config/docs files until the per-repo caps were reached.",
    ]

    repo_payload = snapshot.model_dump(exclude={"tree", "search_queries", "files"})
    ranked_overrides = shallow.repository.model_dump(
        include={
            "reference_type",
            "fit_score",
            "fit_summary",
            "covered_primary",
            "missing_primary",
            "rank_reasons",
            "relevance_score",
            "semantic_meta_score",
            "semantic_readme_score",
            "query_hit_count",
        },
    ) if shallow else {}
    repository = RepoSearchResult(**{**repo_payload, **ranked_overrides})

    return RepoFetchPlan(
        repository=repository,
        selected_paths=selected_paths,
        skipped_paths=skipped_paths[:20],
        estimated_chars=estimated_chars,
        rationale=rationale,
    )


async def fetch_repo_files_for_indexing(plan: RepoFetchPlan) -> list[RepoFile]:
    """Fetch planned files by downloading one repo archive and extracting selected paths."""

    settings = get_settings()
    client = get_github_client()
    archive_ref = plan.repository.commit_sha or plan.repository.default_branch or "HEAD"
    response = await client.get(
        f"{settings.GITHUB_API_BASE}/repos/{plan.repository.full_name}/zipball/{archive_ref}",
        headers=_headers("application/vnd.github+json"),
    )
    response.raise_for_status()

    archive_bytes = response.content
    if len(archive_bytes) > settings.RAG_ARCHIVE_MAX_BYTES:
        raise ValueError(
            f"Archive for {plan.repository.full_name} exceeded the max download size of {settings.RAG_ARCHIVE_MAX_BYTES} bytes."
        )

    selected_lookup = {path.replace("\\", "/"): index for index, path in enumerate(plan.selected_paths)}
    extracted_bytes = 0
    extracted_files = 0
    files_by_index: dict[int, RepoFile] = {}

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative_path = _normalize_archive_path(member.filename)
            if not relative_path:
                continue
            normalized_path = relative_path.replace("\\", "/")
            target_index = selected_lookup.get(normalized_path)
            if target_index is None:
                continue
            if member.file_size > settings.RAG_MAX_FILE_SIZE:
                continue

            extracted_files += 1
            if extracted_files > settings.RAG_ARCHIVE_MAX_FILES:
                break

            with archive.open(member) as file_handle:
                raw = file_handle.read()

            extracted_bytes += len(raw)
            if extracted_bytes > settings.RAG_ARCHIVE_MAX_EXTRACTED_BYTES:
                logger.warning(
                    "Stopped extracting %s after %d bytes to stay within archive limits",
                    plan.repository.full_name,
                    extracted_bytes,
                )
                break
            if _looks_binary(raw):
                continue

            content = raw.decode("utf-8", errors="replace")
            files_by_index[target_index] = RepoFile(
                path=normalized_path,
                content=content,
                size=len(content),
            )

    ordered_files = [files_by_index[index] for index in sorted(files_by_index)]
    logger.info(
        "Fetched %d planned files for %s from one archive download",
        len(ordered_files),
        plan.repository.full_name,
    )
    return ordered_files


async def _fetch_file_content(client, full_name: str, path: str) -> str | None:
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
    except Exception:
        return None
    _cache[cache_key] = content
    return content
