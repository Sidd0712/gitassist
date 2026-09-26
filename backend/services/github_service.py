"""GitHub service for candidate search, staged repo ingestion, and deep file fetches."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import posixpath
import re
import tomllib
import zipfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import yaml
from cachetools import TTLCache

from core.config import get_settings
from models.schemas import (
    ExtractedKeywords,
    RepoFetchPlan,
    RepoFile,
    RepoSearchResult,
    RepoSnapshot,
    RepoTreeEntry,
    ShallowRepoEvidence,
)
from services.github_client_service import get_github_client
from services.pipeline_cache_service import PipelineCacheService, stable_cache_key
from services.rag.embedding_service import EmbeddingService, EmbeddingUnavailable

logger = logging.getLogger(__name__)

_cache: TTLCache = TTLCache(maxsize=512, ttl=get_settings().GITHUB_CACHE_TTL)
_RAW_BASE = "https://raw.githubusercontent.com"
_persistent_cache = PipelineCacheService()

# ── Ranking weights ────────────────────────────────────────────────────────────
_SEARCH_RANKING_WEIGHTS = {
    "metadata_semantic": 0.28,
    "star_quality": 0.30,
    "broad_query_hits": 0.16,
    "concept_family_coverage": 0.14,
    "query_diversity": 0.12,
}

# Bump whenever query strategy or scoring logic changes.
# Invalidates all cached github_search and llm_search_query_specs entries.
_SEARCH_CACHE_VERSION = 5

# Matched against whole directory names, never substrings: substring matching
# used to drop every monorepo's packages/ dir, combine.py ("bin"), distance.py
# ("dist") and objects/ ("obj"). Tests are kept on purpose — they're ranked
# low but often show real usage, which chat questions benefit from.
SKIP_DIRS = {
    "node_modules", "vendor", ".git", "dist", "build", "coverage", "__pycache__",
    ".next", ".nuxt", ".svelte-kit", "target", "bin", "obj", ".vscode", ".idea",
    ".venv", "venv", "site-packages", ".tox", ".mypy_cache", ".pytest_cache",
    "bower_components", ".gradle", ".cache", "__snapshots__",
}
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "cargo.lock",
    "composer.lock", "gemfile.lock", "go.sum", "uv.lock", "pipfile.lock", "bun.lockb",
}
_GENERATED_SUFFIXES = (".min.js", ".min.css", ".map", "_pb2.py", "_pb2_grpc.py", ".pb.go", ".d.ts.map", ".bundle.js")
_DATA_EXTENSIONS = {".json", ".yaml", ".yml", ".xml", ".csv", ".tsv"}

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
    ".html", ".css", ".scss", ".less", ".rst", ".ipynb", ".lua", ".ex", ".exs",
    ".clj", ".hs", ".ml", ".zig", ".jl", ".tf", ".prisma",
}
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".rb", ".php", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".swift", ".kt", ".scala", ".r", ".m", ".vue", ".svelte", ".dart",
    ".sql", ".graphql", ".proto", ".ipynb", ".lua", ".ex", ".exs", ".clj", ".hs", ".ml",
    ".zig", ".jl", ".tf", ".prisma",
}
_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "e2e", "testing"}
_EXAMPLE_DIRS = {"example", "examples", "demo", "demos", "sample", "samples", "playground", "benchmarks"}
_DOC_DIRS = {"docs", "doc", "documentation", "website"}
_SCRIPT_DIRS = {"scripts", "script", "tools", ".github", "ci"}

_IMPL_ROUTE_TOKENS = (
    "router", "route", "service", "controller", "handler", "middleware", "resolver"
)
_IMPL_ENTRY_TOKENS = ("main.", "app.", "server.", "api.", "index.")
_DOC_TOKENS = ("/docs/", "/doc/", "guide", "tutorial", "example", "demo", "sample")

# Minimum cosine similarity to consider a repo as "covering" a concept family
# during candidate search (metadata-only context: name + description + topics).
#
# WHY 0.42 and not 0.52:
# This threshold is evaluated against sparse repo metadata embeddings — just
# the repo name, a one-line description, and a few topic tags. Even with the
# correct Cohere input_type (search_query vs search_document), sparse metadata
# text produces lower cosine similarity than rich text (README + source code).
# The stricter 0.52 threshold is used in ranking_service.py where the full
# README + sampled code is available. Setting 0.52 here caused all candidate
# repos to show capabilities as "Missing" even when the description clearly
# mentioned them.
_CAPABILITY_COVERAGE_THRESHOLD = 0.42


@dataclass(frozen=True)
class SearchQuerySpec:
    query: str
    kind: str  # "broad" | "concept_family" | "language"
    concept_family: str | None = None


# ── Path helpers ───────────────────────────────────────────────────────────────

def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    settings = get_settings()
    headers = {"Accept": accept}
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
    return headers


def _extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1] if "." in filename else ""


def _max_file_size(path: str) -> int:
    # Notebooks carry saved outputs (often base64 plots); only their cells get indexed.
    limit = get_settings().RAG_MAX_FILE_SIZE
    return limit * 10 if path.lower().endswith(".ipynb") else limit


def _should_fetch(path: str, size: int) -> bool:
    settings = get_settings()
    parts = path.lower().split("/")
    filename = parts[-1]
    if any(part in SKIP_DIRS for part in parts[:-1]) or filename in SKIP_FILENAMES:
        return False
    if size > _max_file_size(path) or filename.endswith(_GENERATED_SUFFIXES):
        return False
    if filename in IMPORTANT_FILENAMES or filename.startswith("readme"):
        return True
    extension = _extension(filename)
    if extension in _DATA_EXTENSIONS and size > settings.RAG_MAX_DATA_FILE_SIZE:
        return False
    return extension in SOURCE_EXTENSIONS


def _path_priority(path: str, idea_terms: set[str]) -> int:
    """Order files for whole-repo indexing; higher is indexed first.

    Indexing is progressive and capped (RAG_REPO_MAX_CHUNKS), so this order
    decides what chat can see first and what survives the cap for huge repos:
    root README → root manifests → source → docs → config → examples → tests
    → scripts. Idea-term matches and shallower paths win within a tier.
    """
    lowered = path.lower()
    parts = lowered.split("/")
    filename, dirs = parts[-1], set(parts[:-1])
    extension = _extension(filename)

    if len(parts) == 1 and filename.startswith("readme"):
        tier = 1000
    elif len(parts) <= 2 and filename in IMPORTANT_FILENAMES:
        tier = 900
    elif dirs & _TEST_DIRS or ".test." in filename or ".spec." in filename or filename.startswith("test_"):
        tier = 200
    elif dirs & _EXAMPLE_DIRS:
        tier = 250
    elif dirs & _SCRIPT_DIRS or extension in {".sh", ".bash"}:
        tier = 150
    elif extension in _CODE_EXTENSIONS:
        tier = 600
        if any(token in lowered for token in _IMPL_ROUTE_TOKENS) or filename.startswith(_IMPL_ENTRY_TOKENS):
            tier += 50
    elif extension in {".md", ".rst", ".txt"} or dirs & _DOC_DIRS:
        tier = 400
    else:
        tier = 300

    idea_bonus = min(90, 30 * sum(1 for term in idea_terms if term in lowered))
    return tier + idea_bonus - 5 * min(len(parts) - 1, 10)


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


# ── String utilities ───────────────────────────────────────────────────────────

def _normalize_phrase(value: str) -> str:
    return " ".join(
        p for p in value.lower().replace("_", " ").replace("-", " ").split() if p
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        norm = " ".join(value.split()).strip()
        low = norm.lower()
        if norm and low not in seen:
            seen.add(low)
            result.append(norm)
    return result


# ── LLM-driven concept family expansion (no hardcoded alias tables) ───────────

async def _expand_concept_aliases(
    capability_labels: list[str],
    idea_context: str,
) -> dict[str, list[str]]:
    """
    Ask the LLM for synonyms and related search terms for each capability.

    Replaces the previous hardcoded _CONCEPT_FAMILY_ALIASES / _TOKEN_ALIASES
    tables. The LLM generates aliases appropriate for whatever domain is
    being searched — a recipe app, a whiteboard tool, a freelance marketplace,
    or anything else.

    Falls back to bare labels if the LLM call fails.
    """
    if not capability_labels:
        return {}

    from services.llm_client import get_llm_client
    llm = get_llm_client()

    try:
        alias_map: dict[str, list[str]] = await llm.expand_capability_aliases(
            capabilities=capability_labels,
            idea_context=idea_context,
            aliases_per_capability=4,
        )
        # Ensure every label is present and includes itself
        for label in capability_labels:
            if label not in alias_map:
                alias_map[label] = [label]
            elif label not in alias_map[label]:
                alias_map[label].insert(0, label)
        return alias_map
    except Exception as exc:
        logger.warning("Alias expansion failed, falling back to bare labels: %s", exc)
        return {label: [label] for label in capability_labels}


async def build_concept_families(
    keywords: ExtractedKeywords,
    *,
    idea_context: str,
    limit: int = 6,
) -> list[dict[str, object]]:
    """
    Build concept families from extracted keywords with LLM-generated aliases.

    Each family:
      canonical — normalised key used in embeddings and coverage maps
      label     — human-readable, sent to the LLM
      aliases   — LLM synonyms used to enrich the embedding text

    No hardcoded domain tables. Works for any idea submitted to the pipeline.
    """
    seeds = _dedupe_strings([
        *[cap.replace("_", " ") for cap in keywords.primary_capabilities[:4]],
        *keywords.domain_terms[:4],
        *keywords.likely_components[:3],
    ])[:limit]

    if not seeds:
        return []

    alias_map = await _expand_concept_aliases(seeds, idea_context)

    families: list[dict[str, object]] = []
    seen: set[str] = set()
    for label in seeds:
        canonical = _normalize_phrase(label)
        if not canonical or canonical in seen:
            continue
        raw_aliases = alias_map.get(label, [label])
        aliases = _dedupe_strings([str(a) for a in raw_aliases if str(a).strip()])[:5]
        families.append({"canonical": canonical, "label": label, "aliases": aliases})
        seen.add(canonical)

    return families


# ── Query spec serialisation ───────────────────────────────────────────────────

def _serialize_query_specs(specs: list[SearchQuerySpec]) -> list[dict[str, str]]:
    return [
        {"query": s.query, "kind": s.kind, "concept_family": s.concept_family or ""}
        for s in specs
    ]


def _deserialize_query_specs(payload: object) -> list[SearchQuerySpec]:
    if not isinstance(payload, list):
        return []
    specs: list[SearchQuerySpec] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        kind = str(item.get("kind", "")).strip()
        concept_family = str(item.get("concept_family", "")).strip() or None
        if query and kind:
            specs.append(SearchQuerySpec(query=query, kind=kind, concept_family=concept_family))
    return specs


# ── Fallback / composition helpers ────────────────────────────────────────────

# GitHub search qualifiers (repo:, user:, org:, etc.) take exact values, not
# globs. The LLM occasionally hallucinates wildcard syntax like `repo:*/**`,
# which GitHub's search API rejects outright with a 422 — the whole query then
# silently contributes zero candidates (asyncio.gather swallows the exception).
_INVALID_WILDCARD_QUALIFIER = re.compile(r"\b\w+:\S*\*\S*")
# The LLM is asked for 2-5 word queries but doesn't reliably comply; long,
# keyword-stuffed queries match real repo metadata worse than short ones.
_MAX_QUERY_WORDS = 8


def _sanitize_search_query(query: str) -> str:
    """Fix up an LLM-generated GitHub search query before it's ever sent.

    Strips qualifier tokens with invalid wildcard values and caps free-text
    length, rather than letting a malformed query fail the whole search call.
    Valid qualifier tokens (language:python, in:readme, etc.) are always kept
    regardless of where they fall in the word-count cap — truncating from the
    end would otherwise silently drop a deliberately appended qualifier.
    """
    if not query:
        return query
    cleaned = _INVALID_WILDCARD_QUALIFIER.sub("", query)
    tokens = cleaned.split()
    qualifiers = [t for t in tokens if ":" in t]
    free_text = [t for t in tokens if ":" not in t]
    if len(free_text) > _MAX_QUERY_WORDS:
        free_text = free_text[:_MAX_QUERY_WORDS]
    return " ".join(free_text + qualifiers).strip()


def _fallback_broad_queries(keywords: ExtractedKeywords) -> list[str]:
    product_type = keywords.product_type or keywords.summary
    domain_cluster = " ".join(keywords.domain_terms[:3])
    primary_cluster = " ".join(
        cap.replace("_", " ")
        for cap in (keywords.primary_capabilities or keywords.capabilities[:3])
    )
    return _dedupe_strings([
        " ".join(p for p in [product_type, domain_cluster] if p).strip(),
        keywords.core_intent or keywords.summary,
        " ".join(p for p in [product_type, primary_cluster] if p).strip(),
    ])


def _compose_language_query(keywords: ExtractedKeywords) -> str:
    qualifier = " ".join(
        p for p in [keywords.product_type or "", keywords.core_intent or ""] if p
    ).strip() or keywords.summary
    if not keywords.languages:
        return qualifier
    return f"{qualifier} language:{keywords.languages[0]}".strip()


def _compose_concept_family_query(
    family: dict[str, object],
    idea_context: str,
    llm_query: str,
) -> str:
    """
    Build a GitHub search query by blending the LLM's suggestion with
    LLM-generated aliases. No hardcoded domain terms anywhere in this path.
    """
    aliases = [str(a) for a in family.get("aliases", []) if str(a).strip()]
    base = llm_query.strip() or idea_context[:40].strip()
    if not base:
        base = " ".join(p for p in [str(family.get("label", "")), idea_context] if p).strip()
    for alias in aliases[:2]:
        if alias.lower() not in base.lower():
            base = f"{base} {alias}".strip()
    return base


# ── Main query spec builder ────────────────────────────────────────────────────

async def _build_search_query_specs(keywords: ExtractedKeywords) -> list[SearchQuerySpec]:
    """
    Build the typed GitHub search query set for one pipeline request.

    Strategy:
      1. 3 broad end-to-end queries              → full-platform repos
      2. 1 alias-enriched query per concept family → subsystem repos
      3. 1 language-qualified query               → language-specific repos

    Concept family aliases are generated by the LLM (build_concept_families),
    not looked up in a hardcoded table. Thresholds used downstream are computed
    dynamically from the candidate score distribution.
    """
    settings = get_settings()
    cache_key = stable_cache_key({
        "v": _SEARCH_CACHE_VERSION,
        "model": settings.LLM_MODEL,
        "summary": keywords.summary,
        "core_intent": keywords.core_intent,
        "product_type": keywords.product_type,
        "primary_capabilities": keywords.primary_capabilities,
        "secondary_capabilities": keywords.secondary_capabilities,
        "domain_terms": keywords.domain_terms,
        "likely_components": keywords.likely_components,
        "languages": keywords.languages,
    })
    cached = _deserialize_query_specs(
        _persistent_cache.get_json("llm_search_query_specs", cache_key)
    )
    if cached:
        return cached[: settings.RAG_QUERY_LIMIT]

    from services.llm_client import get_llm_client
    llm = get_llm_client()

    idea_context = keywords.core_intent or keywords.summary or keywords.product_type or ""
    concept_families = await build_concept_families(
        keywords, idea_context=idea_context, limit=4
    )

    specs: list[SearchQuerySpec] = []
    seen: set[str] = set()

    def _add(query: str, kind: str, concept_family: str | None = None) -> None:
        norm = _sanitize_search_query(" ".join(query.split()).strip())
        if norm and norm.lower() not in seen:
            specs.append(SearchQuerySpec(query=norm, kind=kind, concept_family=concept_family))
            seen.add(norm.lower())

    # 1. Broad end-to-end queries
    try:
        broad_queries = await llm.generate_search_queries_batch(
            idea_context=idea_context,
            capabilities=(keywords.primary_capabilities or keywords.capabilities[:4])[:3],
            domain_terms=keywords.domain_terms or keywords.keywords,
            languages=keywords.languages,
            max_queries=3,
        )
    except Exception as exc:
        logger.warning("Broad query generation failed: %s", exc)
        broad_queries = []

    for q in broad_queries[:3]:
        _add(q, "broad")
    for fallback in _fallback_broad_queries(keywords):
        if sum(1 for s in specs if s.kind == "broad") >= 3:
            break
        _add(fallback, "broad")

    # 2. One query per concept family (LLM-assisted, alias-enriched)
    for family in concept_families:
        llm_query = ""
        try:
            llm_queries = await llm.generate_search_queries(
                capability=str(family["label"]),
                idea_context=idea_context,
                num_queries=1,
            )
            if llm_queries:
                llm_query = llm_queries[0]
        except Exception as exc:
            logger.warning(
                "Concept-family query failed for %s: %s", family["label"], exc
            )
        _add(
            _compose_concept_family_query(family, idea_context, llm_query),
            "concept_family",
            str(family["canonical"]),
        )

    # 3. Language-qualified query
    lang_query = _compose_language_query(keywords)
    if keywords.languages and lang_query:
        _add(lang_query, "language")
    else:
        for fallback in _fallback_broad_queries(keywords):
            if len(specs) >= settings.RAG_QUERY_LIMIT:
                break
            _add(fallback, "broad")

    if not specs:
        specs = [SearchQuerySpec(
            query=keywords.summary or idea_context or "open source platform",
            kind="broad",
        )]

    final_specs = specs[: settings.RAG_QUERY_LIMIT]
    _persistent_cache.set_json(
        "llm_search_query_specs", cache_key, _serialize_query_specs(final_specs)
    )
    logger.info(
        "Built %d search query specs for: %s",
        len(final_specs), keywords.core_intent or keywords.summary,
    )
    return final_specs


async def build_search_queries(keywords: ExtractedKeywords) -> list[str]:
    """Return just the query strings (for callers that don't need spec metadata)."""
    return [s.query for s in await _build_search_query_specs(keywords)]


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _language_bucket(language: str | None) -> str:
    return (language or "Unknown").strip() or "Unknown"


def _intent_search_text(keywords: ExtractedKeywords) -> str:
    terms = [
        keywords.core_intent or keywords.summary,
        " ".join(keywords.primary_capabilities[:3]),
        " ".join(keywords.secondary_capabilities[:3]),
        " ".join(keywords.domain_terms[:6]),
    ]
    return " ".join(t for t in terms if t).strip()


def _repo_metadata_text(repo: RepoSearchResult) -> str:
    return "\n".join([repo.full_name, repo.description or "", " ".join(repo.topics)])


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _star_score(stars: int) -> float:
    """0 → 0.0 | 50 → ~0.49 | 500 → ~0.73 | 5000+ → 1.0"""
    if stars <= 0:
        return 0.0
    return min(math.log10(stars) / math.log10(5000), 1.0)


def _metadata_domain_alignment(metadata_text: str, keywords: ExtractedKeywords) -> float:
    """
    Fraction of product-type / domain / likely-component terms that appear
    literally in repo metadata. No hardcoded domain words — entirely derived
    from the extracted keywords for this request.
    """
    norm_meta = _normalize_phrase(metadata_text)
    terms = [
        keywords.product_type,
        *keywords.domain_terms[:6],
        *keywords.likely_components[:4],
    ]
    norm_terms = [_normalize_phrase(t) for t in terms if isinstance(t, str) and t.strip()]
    if not norm_terms:
        return 0.0
    return sum(1 for t in norm_terms if t and t in norm_meta) / len(norm_terms)


# ── Concept family embeddings & semantic coverage ─────────────────────────────

async def _compute_concept_family_embeddings(
    concept_families: list[dict[str, object]],
    embedding_service: EmbeddingService,
) -> dict[str, list[float]]:
    """
    Embed each concept family as: label + LLM-generated aliases joined into one text.
    No hardcoded aliases — they come from build_concept_families / _expand_concept_aliases.
    """
    if not concept_families:
        return {}
    texts = [
        " ".join([
            str(f.get("label", "")),
            *[str(a) for a in f.get("aliases", [])[:3]],
        ]).strip()
        for f in concept_families
    ]
    # IMPORTANT: use embed_queries_batch (search_query input_type), NOT embed_documents.
    # Concept family texts are semantic intent descriptions — "what we are looking for".
    # They must be compared against search_document repo metadata embeddings using
    # Cohere's asymmetric retrieval pairing. Using embed_documents here would make
    # both sides search_document, depressing scores below the coverage threshold.
    embeddings = await embedding_service.embed_queries_batch(texts)
    return {str(f["canonical"]): emb for f, emb in zip(concept_families, embeddings)}


def _semantic_concept_coverage(
    repo_embedding: list[float],
    concept_embeddings: dict[str, list[float]],
) -> tuple[float, set[str]]:
    """
    Return (coverage_fraction, covered_concepts).
    A concept is covered when cosine similarity ≥ _CAPABILITY_COVERAGE_THRESHOLD.
    """
    if not concept_embeddings or not repo_embedding:
        return 0.0, set()
    covered: set[str] = set()
    for concept, emb in concept_embeddings.items():
        if _cosine_similarity(repo_embedding, emb) >= _CAPABILITY_COVERAGE_THRESHOLD:
            covered.add(concept)
    return len(covered) / len(concept_embeddings), covered


# ── Dynamic shortlist selection ────────────────────────────────────────────────

def _percentile_threshold(values: list[float], *, pct: float) -> float:
    """
    Compute the value at the given percentile of a score distribution.

    Using the actual distribution rather than a hardcoded constant means
    the threshold adapts automatically: niche domains where all scores are
    low will still produce results; popular domains with many strong repos
    will have a higher bar.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * pct) - 1)
    return sorted_vals[idx]


def _select_candidate_shortlist(
    scored: list[RepoSearchResult],
    *,
    limit: int,
    max_per_language: int,
    broad_query_scores: dict[str, float],
    concept_family_scores: dict[str, float],
    semantic_scores: dict[str, float],
) -> list[RepoSearchResult]:
    """
    Two-pass selection:

    Pass 1 — end-to-end preference:
      Reserve up to 2 slots for repos that appeared in broad queries AND have
      above-median semantic or concept-family relevance. Ensures at least one
      full-platform reference makes it through regardless of star count.

    Pass 2 — fill remaining slots by descending relevance_score with language
      diversity enforcement.

    All thresholds are percentiles of the actual candidate score distribution
    for this request — no hardcoded constants.
    """
    if not scored:
        return []

    semantic_threshold = _percentile_threshold(
        list(semantic_scores.values()), pct=0.40
    )
    broad_threshold = _percentile_threshold(
        list(broad_query_scores.values()), pct=0.40
    )
    concept_threshold = _percentile_threshold(
        list(concept_family_scores.values()), pct=0.30
    )

    selected: list[RepoSearchResult] = []
    selected_names: set[str] = set()
    lang_counts: defaultdict[str, int] = defaultdict(int)

    def _can_add(repo: RepoSearchResult) -> bool:
        return lang_counts[_language_bucket(repo.language)] < max_per_language

    def _add(repo: RepoSearchResult) -> None:
        selected.append(repo)
        selected_names.add(repo.full_name)
        lang_counts[_language_bucket(repo.language)] += 1

    # Pass 1: end-to-end platform repos
    end_to_end = [
        r for r in scored
        if broad_query_scores.get(r.full_name, 0.0) >= broad_threshold
        and (
            semantic_scores.get(r.full_name, 0.0) >= semantic_threshold
            or concept_family_scores.get(r.full_name, 0.0) >= concept_threshold
        )
    ]
    for repo in end_to_end:
        if len(selected) >= min(2, limit):
            break
        if repo.full_name not in selected_names and _can_add(repo):
            _add(repo)

    # Pass 2: fill remaining slots
    for repo in scored:
        if len(selected) >= limit:
            break
        if repo.full_name not in selected_names and _can_add(repo):
            _add(repo)

    return selected


# ── Main candidate search ──────────────────────────────────────────────────────

async def search_repo_candidates(keywords: ExtractedKeywords) -> list[RepoSearchResult]:
    """
    Search GitHub using capability-targeted queries, score candidates with
    star-biased semantic signals, and return a complementary shortlist.

    Design:
      - sort=stars+order=desc on every GitHub request → established repos first
      - Concept families and their aliases are LLM-generated per request
      - Score thresholds derived from this request's distribution (no constants)
      - Off-domain repos penalised relative to score floor (not a fixed cap)
    """
    settings = get_settings()
    query_specs = await _build_search_query_specs(keywords)
    if not query_specs:
        return []

    merged: dict[str, RepoSearchResult] = {}
    query_hits: defaultdict[str, set[str]] = defaultdict(set)
    broad_hits: defaultdict[str, set[str]] = defaultdict(set)
    concept_hits: defaultdict[str, set[str]] = defaultdict(set)

    embedding_service = EmbeddingService()
    intent_text = _intent_search_text(keywords)
    client = get_github_client()
    semaphore = asyncio.Semaphore(settings.RAG_MAX_SEARCH_CONCURRENCY)

    async def fetch_query(spec: SearchQuerySpec) -> tuple[SearchQuerySpec, list[dict]]:
        cache_payload = {
            "v": _SEARCH_CACHE_VERSION,
            "query": spec.query,
            "per_page": settings.RAG_SEARCH_PER_QUERY,
            "sort": "stars",
        }
        ck = stable_cache_key(cache_payload)
        mk = f"search:v{_SEARCH_CACHE_VERSION}:{spec.query}:{settings.RAG_SEARCH_PER_QUERY}"
        if mk in _cache:
            return spec, _cache[mk]
        cached = _persistent_cache.get_json("github_search", ck)
        if isinstance(cached, list):
            _cache[mk] = cached
            return spec, cached
        async with semaphore:
            response = await client.get(
                f"{settings.GITHUB_API_BASE}/search/repositories",
                headers=_headers(),
                params={
                    "q": spec.query,
                    "per_page": settings.RAG_SEARCH_PER_QUERY,
                    "sort": "stars",
                    "order": "desc",
                },
            )
            response.raise_for_status()
            items = response.json().get("items", [])
            _cache[mk] = items
            _persistent_cache.set_json("github_search", ck, items)
            return spec, items

    def _merge_search_results(results: list) -> None:
        for result in results:
            if isinstance(result, Exception):
                logger.warning("GitHub query failed: %s", result)
                continue
            spec, items = result
            for item in items:
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
                existing = merged.get(repo.full_name)
                if existing is None or repo.stars > existing.stars:
                    merged[repo.full_name] = repo
                query_hits[repo.full_name].add(spec.query)
                if spec.kind in {"broad", "language"}:
                    broad_hits[repo.full_name].add(spec.query)
                if spec.concept_family:
                    concept_hits[repo.full_name].add(spec.concept_family)

    results = await asyncio.gather(
        *(fetch_query(s) for s in query_specs), return_exceptions=True
    )
    _merge_search_results(results)

    # Low-candidate-pool fallback: GitHub's /search/repositories only matches
    # metadata (name/description/README-adjacent/topics), so niche ideas whose
    # targeted queries don't share vocabulary with how real matching repos
    # describe themselves can come back nearly empty. Retry with a couple of
    # maximally broad, deterministic (no extra LLM call) queries before
    # accepting a thin pool -- cheap insurance against confidently selecting
    # the one unrelated repo that happened to match.
    _MIN_CANDIDATE_POOL = 5
    if len(merged) < _MIN_CANDIDATE_POOL:
        tried = {s.query.lower() for s in query_specs}
        broadening_specs = [
            SearchQuerySpec(query=q, kind="broad_fallback")
            for q in _fallback_broad_queries(keywords) + [keywords.product_type]
            if q and _sanitize_search_query(q).lower() not in tried
        ]
        # Dedupe while preserving order, cap at 2 extra queries to stay cheap.
        seen_fallback: set[str] = set()
        deduped_fallback = []
        for spec in broadening_specs:
            key = spec.query.lower()
            if key not in seen_fallback:
                seen_fallback.add(key)
                deduped_fallback.append(spec)
        broadening_specs = deduped_fallback[:2]
        if broadening_specs:
            logger.info(
                "Candidate pool thin (%d repos) after %d queries; firing %d broader fallback queries",
                len(merged), len(query_specs), len(broadening_specs),
            )
            fallback_results = await asyncio.gather(
                *(fetch_query(s) for s in broadening_specs), return_exceptions=True
            )
            _merge_search_results(fallback_results)

    if not merged:
        return []

    candidates = list(merged.values())
    idea_context = keywords.core_intent or keywords.summary or ""
    concept_families = await build_concept_families(
        keywords, idea_context=idea_context, limit=4
    )

    max_query_hits = max((len(query_hits[r.full_name]) for r in candidates), default=1)
    broad_total = max(1, sum(1 for s in query_specs if s.kind in {"broad", "language"}))
    concept_total = max(
        1, len({s.concept_family for s in query_specs if s.concept_family})
    )

    try:
        intent_embedding, concept_embeddings = await asyncio.gather(
            embedding_service.embed_query(intent_text),
            _compute_concept_family_embeddings(concept_families, embedding_service),
        )
        metadata_embeddings = await embedding_service.embed_documents(
            [_repo_metadata_text(r) for r in candidates]
        )
    except EmbeddingUnavailable as exc:
        # Semantic scores fall to 0 and ranking leans on stars, query hits,
        # query-derived concept coverage, and domain alignment — weaker, but a
        # report beats an error when the embedding worker is offline.
        logger.warning("Candidate scoring without embeddings: %s", exc)
        intent_embedding, concept_embeddings = [], {}
        metadata_embeddings = [[] for _ in candidates]

    weights = dict(_SEARCH_RANKING_WEIGHTS)
    broad_query_scores: dict[str, float] = {}
    concept_family_scores: dict[str, float] = {}
    semantic_scores: dict[str, float] = {}
    # Stash domain signals per repo for the post-pass off-domain penalty
    domain_signals: dict[str, tuple[float, float]] = {}  # repo → (alignment, concept_score)

    preliminary: list[RepoSearchResult] = []

    for repo, repo_emb in zip(candidates, metadata_embeddings):
        repo.query_hit_count = len(query_hits[repo.full_name])
        repo.semantic_meta_score = round(_cosine_similarity(intent_embedding, repo_emb), 5)

        query_div = repo.query_hit_count / max(max_query_hits, 1)
        star_qual = (
            _star_score(repo.stars)
            * (0.5 if repo.archived else 1.0)
            * (1.1 if repo.updated_at else 0.9)
        )
        broad_score = len(broad_hits[repo.full_name]) / broad_total
        q_concept = len(concept_hits[repo.full_name]) / concept_total
        sem_concept, sem_concepts = _semantic_concept_coverage(repo_emb, concept_embeddings)
        concept_score = max(q_concept, sem_concept)
        domain_align = _metadata_domain_alignment(_repo_metadata_text(repo), keywords)

        candidate_score = (
            weights["metadata_semantic"] * repo.semantic_meta_score
            + weights["star_quality"] * star_qual
            + weights["broad_query_hits"] * broad_score
            + weights["concept_family_coverage"] * concept_score
            + weights["query_diversity"] * query_div
        )

        repo.relevance_score = round(max(candidate_score, 0.0), 5)
        repo.rank_reasons = [
            f"semantic {repo.semantic_meta_score:.2f}",
            f"stars {repo.stars} (quality {star_qual:.2f})",
            f"broad hits {broad_score:.2f}",
            f"concept coverage {concept_score:.2f}"
            f" ({', '.join(sorted(sem_concepts or concept_hits[repo.full_name])) or 'none'})",
            f"query diversity {query_div:.2f}",
        ]
        broad_query_scores[repo.full_name] = broad_score
        concept_family_scores[repo.full_name] = concept_score
        semantic_scores[repo.full_name] = repo.semantic_meta_score
        domain_signals[repo.full_name] = (domain_align, concept_score)
        preliminary.append(repo)

    # ── Dynamic off-domain penalty ────────────────────────────────────────────
    # Repos with zero signal on all three dimensions get capped at the 25th
    # percentile of the score distribution — relative, not hardcoded.
    all_scores = [r.relevance_score for r in preliminary]
    score_floor = _percentile_threshold(all_scores, pct=0.25)

    for repo in preliminary:
        domain_align, concept_score = domain_signals[repo.full_name]
        if (
            repo.semantic_meta_score < settings.RAG_SEMANTIC_MIN_RELEVANCE
            and concept_score <= 0.0
            and domain_align <= 0.0
        ):
            repo.relevance_score = min(repo.relevance_score, score_floor)

    preliminary.sort(key=lambda r: r.relevance_score, reverse=True)

    shortlisted = _select_candidate_shortlist(
        preliminary,
        limit=settings.RAG_CANDIDATE_REPO_LIMIT,
        max_per_language=settings.RAG_MAX_PER_LANGUAGE,
        broad_query_scores=broad_query_scores,
        concept_family_scores=concept_family_scores,
        semantic_scores=semantic_scores,
    )

    logger.info(
        "Candidate search: %d repos from %d queries → %d shortlisted",
        len(candidates), len(query_specs), len(shortlisted),
    )
    return shortlisted


# ── Repo snapshot ──────────────────────────────────────────────────────────────

async def fetch_repo_snapshot(repo: RepoSearchResult) -> RepoSnapshot:
    """Fetch repo metadata, commit SHA, and file tree."""

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
    repo_resp = await client.get(
        f"{settings.GITHUB_API_BASE}/repos/{repo.full_name}", headers=_headers()
    )
    repo_resp.raise_for_status()
    repo_data = repo_resp.json()

    default_branch = repo_data.get("default_branch") or repo.default_branch or "HEAD"
    commit_sha = repo.commit_sha or default_branch

    commit_resp = await client.get(
        f"{settings.GITHUB_API_BASE}/repos/{repo.full_name}/commits/{default_branch}",
        headers=_headers(),
    )
    if commit_resp.status_code == 200:
        commit_sha = commit_resp.json().get("sha", "") or commit_sha
    else:
        logger.warning(
            "Could not resolve commit SHA for %s on %s (status %d)",
            repo.full_name, default_branch, commit_resp.status_code,
        )

    tree_resp = await client.get(
        f"{settings.GITHUB_API_BASE}/repos/{repo.full_name}/git/trees/{default_branch}",
        headers=_headers(),
        params={"recursive": 1},
    )
    tree_entries: list[RepoTreeEntry] = []
    if tree_resp.status_code == 200:
        tree_entries = [
            RepoTreeEntry(
                path=item["path"],
                type=item.get("type", "blob"),
                size=item.get("size", 0),
            )
            for item in tree_resp.json().get("tree", [])
            if item.get("type") == "blob"
        ]
    else:
        logger.warning(
            "Could not fetch tree for %s on %s (status %d)",
            repo.full_name, default_branch, tree_resp.status_code,
        )

    payload = repo.model_dump()
    payload.update({
        "description": repo_data.get("description") or repo.description,
        "html_url": repo_data.get("html_url") or repo.html_url,
        "stars": repo_data.get("stargazers_count", repo.stars),
        "language": repo_data.get("language") or repo.language,
        "topics": repo_data.get("topics", repo.topics),
        "default_branch": default_branch,
        "updated_at": repo_data.get("updated_at"),
        "archived": repo_data.get("archived", False),
        "commit_sha": commit_sha,
        "tree": [e.model_dump() for e in tree_entries],
    })
    snapshot = RepoSnapshot(**payload)
    _cache[memory_key] = payload
    _persistent_cache.set_json("github_snapshot", cache_key, payload)
    return snapshot


# ── Shallow evidence ───────────────────────────────────────────────────────────

_MAX_DEPENDENCIES = 40


def _requirement_name(spec: str) -> str:
    return re.split(r"[<>=!~\[;@\s]", spec.strip(), maxsplit=1)[0]


def _parse_manifest_dependencies(path: str, content: str) -> list[str]:
    """Pull declared package/image names out of one root manifest.

    Lockfiles are skipped on purpose: they list every transitive package,
    which buries the handful of choices the repo's author actually made.
    """
    name = posixpath.basename(path).lower()
    deps: list[str] = []
    try:
        if name == "package.json":
            data = json.loads(content)
            for section in ("dependencies", "devDependencies"):
                deps.extend((data.get(section) or {}).keys())
        elif name == "requirements.txt":
            for line in content.splitlines():
                line = line.split("#", 1)[0].strip()
                if line and not line.startswith("-"):
                    deps.append(_requirement_name(line))
        elif name == "pyproject.toml":
            data = tomllib.loads(content)
            deps.extend(_requirement_name(s) for s in data.get("project", {}).get("dependencies", []))
            poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            deps.extend(k for k in poetry if k.lower() != "python")
        elif name == "cargo.toml":
            deps.extend(tomllib.loads(content).get("dependencies", {}).keys())
        elif name == "go.mod":
            for match in re.finditer(r"^\s*(?:require\s+)?([\w.\-]+(?:/[\w.\-]+)+)\s+v[\w.\-+]+", content, re.M):
                deps.append(match.group(1))
        elif name == "pom.xml":
            deps.extend(re.findall(r"<dependency>.*?<artifactId>\s*([^<\s]+)\s*</artifactId>", content, re.S))
        elif name in ("build.gradle", "build.gradle.kts"):
            deps.extend(
                m.group(1)
                for m in re.finditer(
                    r"(?:implementation|api|compileOnly|runtimeOnly)\s*\(?\s*['\"][^:'\"]+:([^:'\"]+)", content
                )
            )
        elif name == "dockerfile":
            for match in re.finditer(r"^\s*FROM\s+(?:--\S+\s+)*([^\s:@]+)", content, re.M | re.I):
                if match.group(1).lower() != "scratch":
                    deps.append(match.group(1))
        elif name in ("docker-compose.yml", "docker-compose.yaml"):
            services = (yaml.safe_load(content) or {}).get("services") or {}
            for service in services.values():
                if isinstance(service, dict) and isinstance(service.get("image"), str):
                    deps.append(re.split(r"[:@]", service["image"], maxsplit=1)[0])
    except (ValueError, TypeError, AttributeError, tomllib.TOMLDecodeError, yaml.YAMLError):
        return []
    return [d for d in deps if isinstance(d, str) and d]


def _with_manifest_dependencies(evidence: ShallowRepoEvidence) -> ShallowRepoEvidence:
    deps: list[str] = []
    for manifest in evidence.manifest_files:
        deps.extend(_parse_manifest_dependencies(manifest.path, manifest.content))
    evidence.repository.dependencies = list(dict.fromkeys(deps))[:_MAX_DEPENDENCIES]
    return evidence


def _root_manifest_candidates(language: str | None) -> list[str]:
    base = [
        "package.json", "pyproject.toml", "requirements.txt", "go.mod",
        "Cargo.toml", "Dockerfile", "docker-compose.yml",
    ]
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


def _is_implementation_source_path(path: str) -> bool:
    ext = "." + path.lower().rsplit(".", 1)[-1] if "." in path else ""
    return ext in {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
        ".kt", ".scala", ".vue", ".svelte", ".dart", ".sh",
    }


def _idea_terms_from_keywords(
    keywords: ExtractedKeywords | None,
    matched_capabilities: list[str] | None = None,
) -> set[str]:
    if keywords is None:
        return set()
    return {
        term.lower()
        for term in [
            *keywords.domain_terms[:8],
            *keywords.primary_capabilities[:4],
            *keywords.secondary_capabilities[:4],
            *keywords.keywords[:6],
            *keywords.likely_components[:4],
            *((matched_capabilities or [])[:4]),
        ]
        if isinstance(term, str) and len(term) > 2
    }


def _sampled_code_paths(
    snapshot: RepoSnapshot,
    keywords: ExtractedKeywords | None,
    *,
    limit: int,
) -> list[str]:
    idea_terms = _idea_terms_from_keywords(keywords)
    prioritized = sorted(
        (
            entry for entry in snapshot.tree
            if entry.type == "blob"
            and _should_fetch(entry.path, entry.size)
            and _is_implementation_source_path(entry.path)
            and not any(tok in entry.path.lower() for tok in _DOC_TOKENS)
        ),
        key=lambda e: (_path_priority(e.path, idea_terms), e.size * -1),
        reverse=True,
    )
    return [e.path for e in prioritized[:limit]]


async def _fetch_sampled_code_files(
    repository: RepoSearchResult,
    keywords: ExtractedKeywords | None,
    client,
) -> list[RepoFile]:
    if keywords is None:
        return []
    settings = get_settings()
    snapshot = await fetch_repo_snapshot(repository)
    paths = _sampled_code_paths(snapshot, keywords, limit=settings.RAG_SHALLOW_CODE_SAMPLE_COUNT)
    files: list[RepoFile] = []
    for path in paths:
        content = await _fetch_file_content(client, repository.full_name, path, snapshot.commit_sha or "HEAD")
        if not content:
            continue
        truncated = content[: settings.RAG_SHALLOW_CODE_SAMPLE_MAX_CHARS]
        files.append(RepoFile(path=path, content=truncated, size=len(truncated)))
    return files


async def fetch_shallow_repo_evidence(
    repository: RepoSearchResult,
    keywords: ExtractedKeywords | None = None,
) -> ShallowRepoEvidence:
    """Fetch README, manifests, and sampled implementation files for reranking."""

    settings = get_settings()
    cache_key = stable_cache_key({
        "v": _SEARCH_CACHE_VERSION,
        "repo": repository.full_name,
        "updated_at": repository.updated_at,
        "language": repository.language,
        "primary_capabilities": keywords.primary_capabilities[:4] if keywords else [],
        "domain_terms": keywords.domain_terms[:6] if keywords else [],
        "likely_components": keywords.likely_components[:4] if keywords else [],
        "sample_count": settings.RAG_SHALLOW_CODE_SAMPLE_COUNT if keywords else 0,
        "sample_chars": settings.RAG_SHALLOW_CODE_SAMPLE_MAX_CHARS if keywords else 0,
    })
    cached = _persistent_cache.get_json("github_shallow", cache_key)
    if isinstance(cached, dict):
        return _with_manifest_dependencies(ShallowRepoEvidence(**cached))

    client = get_github_client()
    readme_text = ""
    readme_path: str | None = None
    highlighted_paths: list[str] = []
    manifest_files: list[RepoFile] = []

    # Fetch README
    readme_ck = f"readme:{repository.full_name}"
    if readme_ck in _cache:
        readme_text = _cache[readme_ck]
        readme_path = "README.md" if readme_text else None
    else:
        resp = await client.get(
            f"{settings.GITHUB_API_BASE}/repos/{repository.full_name}/readme",
            headers=_headers("application/vnd.github.raw+json"),
        )
        if resp.status_code == 200:
            readme_text = resp.text
            readme_path = "README.md"
            _cache[readme_ck] = readme_text

    for path in ["README.md", "readme.md", "README", "docs/README.md"]:
        if readme_text:
            break
        readme_text = await _fetch_file_content(client, repository.full_name, path) or ""
        if readme_text:
            readme_path = path

    # Fetch manifests in parallel
    manifest_candidates = _root_manifest_candidates(repository.language)[:5]

    async def fetch_manifest(p: str) -> tuple[str, str | None]:
        return p, await _fetch_file_content(client, repository.full_name, p)

    manifest_results = await asyncio.gather(*(fetch_manifest(p) for p in manifest_candidates))
    for path, content in manifest_results:
        if not content:
            continue
        manifest_files.append(RepoFile(path=path, content=content, size=len(content)))
        highlighted_paths.append(path)

    sampled_files = await _fetch_sampled_code_files(repository, keywords, client)
    sampled_paths = [f.path for f in sampled_files]

    if readme_path:
        highlighted_paths.insert(0, readme_path)
    highlighted_paths.extend(sampled_paths)

    evidence = ShallowRepoEvidence(
        repository=repository.model_copy(deep=True),
        readme_path=readme_path,
        readme=readme_text,
        manifest_files=manifest_files,
        sampled_files=sampled_files,
        sampled_paths=sampled_paths,
        highlighted_paths=highlighted_paths[:10],
    )
    _persistent_cache.set_json("github_shallow", cache_key, evidence.model_dump())
    return _with_manifest_dependencies(evidence)


async def fetch_shallow_repo_evidence_batch(
    repositories: list[RepoSearchResult],
    keywords: ExtractedKeywords | None = None,
) -> list[ShallowRepoEvidence]:
    """Fetch shallow evidence for a shortlist with bounded concurrency."""

    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.RAG_MAX_SEARCH_CONCURRENCY)

    async def fetch(repo: RepoSearchResult) -> ShallowRepoEvidence | None:
        async with semaphore:
            try:
                return await fetch_shallow_repo_evidence(repo, keywords)
            except Exception as exc:
                logger.warning("Skipping %s during shallow ingest: %s", repo.full_name, exc)
                return None

    return [r for r in await asyncio.gather(*(fetch(r) for r in repositories)) if r]


# ── Deep indexing ──────────────────────────────────────────────────────────────

def build_repo_fetch_plan(
    snapshot: RepoSnapshot,
    intent: ExtractedKeywords | ShallowRepoEvidence,
    shallow: ShallowRepoEvidence | ExtractedKeywords | None = None,
) -> RepoFetchPlan:
    """Select highest-value files for deep indexing."""

    if isinstance(intent, ShallowRepoEvidence) and isinstance(shallow, ExtractedKeywords):
        intent, shallow = shallow, intent
    if not isinstance(intent, ExtractedKeywords):
        raise TypeError("build_repo_fetch_plan expects ExtractedKeywords as the resolved intent.")

    path_terms = path_terms_for_intent(intent, shallow.matched_capabilities if shallow else None)
    selected_paths, skipped_paths, estimated_chars = select_index_paths(snapshot.full_name, snapshot.tree, path_terms)

    repo_payload = snapshot.model_dump(exclude={"tree", "search_queries", "files"})
    ranked_overrides = shallow.repository.model_dump(
        include={
            "reference_type", "fit_score", "fit_summary",
            "covered_primary", "missing_primary", "rank_reasons",
            "relevance_score", "semantic_meta_score",
            "semantic_readme_score", "query_hit_count",
        },
    ) if shallow else {}

    return RepoFetchPlan(
        repository=RepoSearchResult(**{**repo_payload, **ranked_overrides}),
        selected_paths=selected_paths,
        skipped_paths=skipped_paths[:20],
        estimated_chars=estimated_chars,
        rationale=[
            "Whole repo minus vendored, generated, lock and oversized data files.",
            "Order: root README, manifests, source, docs, config, examples, tests, scripts.",
            "Only the chunk budget (RAG_REPO_MAX_CHUNKS) truncates, from the lowest priority up.",
        ],
    )


def path_terms_for_intent(keywords: ExtractedKeywords, matched_capabilities: list[str] | None = None) -> list[str]:
    """Single words from the idea that can boost matching file paths during indexing."""

    return sorted(_path_terms(_idea_terms_from_keywords(keywords, (matched_capabilities or [])[:4])))


def select_index_paths(
    full_name: str, tree: list[RepoTreeEntry], path_terms: list[str] | set[str]
) -> tuple[list[str], list[str], int]:
    """Whole repo, in priority order, until the chunk budget is spent.

    Returns (selected, skipped, estimated_chars). Budget is ~1 chunk per
    RAG_CHUNK_TARGET_CHARS of source, so only huge repos lose their
    lowest-priority files (tests, scripts) to the cap.
    """

    settings = get_settings()
    terms = set(path_terms)
    total_blobs = sum(1 for e in tree if e.type == "blob")
    prioritized = sorted(
        (e for e in tree if e.type == "blob" and _should_fetch(e.path, e.size)),
        key=lambda e: (_path_priority(e.path, terms), -e.size),
        reverse=True,
    )

    char_budget = settings.RAG_REPO_MAX_CHUNKS * settings.RAG_CHUNK_TARGET_CHARS
    selected: list[str] = []
    skipped: list[str] = []
    estimated_chars = 0
    for entry in prioritized:
        if estimated_chars + entry.size > char_budget:
            skipped.append(entry.path)
            continue
        selected.append(entry.path)
        estimated_chars += entry.size

    logger.info(
        "%s: indexing %d/%d files (%d eligible, %d over the %d-chunk budget)",
        full_name, len(selected), total_blobs, len(prioritized), len(skipped), settings.RAG_REPO_MAX_CHUNKS,
    )
    return selected, skipped, estimated_chars


async def fetch_repo_tree(full_name: str, ref: str) -> list[RepoTreeEntry]:
    """The file tree at an exact commit, so the worker indexes what Render queued."""

    settings = get_settings()
    response = await get_github_client().get(
        f"{settings.GITHUB_API_BASE}/repos/{full_name}/git/trees/{ref}",
        headers=_headers(),
        params={"recursive": 1},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("truncated"):
        logger.warning("Tree for %s@%s is truncated by GitHub; indexing the part that was returned", full_name, ref[:8])
    return [
        RepoTreeEntry(path=item["path"], type="blob", size=item.get("size", 0))
        for item in payload.get("tree", [])
        if item.get("type") == "blob"
    ]


_PATH_TERM_STOPWORDS = {"with", "from", "that", "this", "into", "user", "users", "data", "based", "using", "system", "management"}


def _path_terms(idea_terms: set[str]) -> set[str]:
    """Split idea phrases into single words that can plausibly appear in a file path."""

    return {
        word
        for term in idea_terms
        for word in re.split(r"[^a-z0-9]+", term.lower())
        if len(word) >= 4 and word not in _PATH_TERM_STOPWORDS
    }


async def fetch_repo_files_for_indexing(
    plan: RepoFetchPlan, on_progress: Callable[[int, int], None] | None = None
) -> list[RepoFile]:
    """Fetch planned files via a single archive download.

    `on_progress(done, total)` is only called on the individual-file fallback
    below — a zip download has no meaningful sub-progress to report, and
    finishes quickly enough that it doesn't need any.
    """

    settings = get_settings()
    client = get_github_client()
    archive_ref = plan.repository.commit_sha or plan.repository.default_branch or "HEAD"
    archive_bytes = await _download_archive_capped(
        client,
        f"{settings.GITHUB_API_BASE}/repos/{plan.repository.full_name}/zipball/{archive_ref}",
        settings.RAG_ARCHIVE_MAX_BYTES,
    )
    if archive_bytes is None:
        logger.info(
            "Archive for %s exceeds %d bytes; fetching %d planned files individually",
            plan.repository.full_name, settings.RAG_ARCHIVE_MAX_BYTES, len(plan.selected_paths),
        )
        return await _fetch_planned_files_raw(client, plan, archive_ref, on_progress)

    selected_lookup = {
        path.replace("\\", "/"): idx for idx, path in enumerate(plan.selected_paths)
    }
    files_by_index: dict[int, RepoFile] = {}

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            rel = _normalize_archive_path(member.filename)
            if not rel:
                continue
            norm = rel.replace("\\", "/")
            idx = selected_lookup.get(norm)
            if idx is None or member.file_size > _max_file_size(norm):
                continue

            with archive.open(member) as fh:
                raw = fh.read()
            if _looks_binary(raw):
                continue

            content = _readable_content(norm, raw.decode("utf-8", errors="replace"))
            if content:
                files_by_index[idx] = RepoFile(path=norm, content=content, size=len(content))

    ordered = [files_by_index[i] for i in sorted(files_by_index)]
    logger.info("Fetched %d files for %s", len(ordered), plan.repository.full_name)
    return ordered


async def _download_archive_capped(client, url: str, max_bytes: int) -> bytes | None:
    """Stream an archive, returning None as soon as it exceeds max_bytes.

    Buffering the whole response before checking its size let a single 600 MB
    repo zip OOM-kill the 512 MB Render instance mid-request.
    """

    async with client.stream("GET", url, headers=_headers("application/vnd.github+json")) as response:
        response.raise_for_status()
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            return None
        buffer = bytearray()
        async for chunk in response.aiter_bytes():
            buffer.extend(chunk)
            if len(buffer) > max_bytes:
                return None
    return bytes(buffer)


async def _fetch_planned_files_raw(
    client, plan: RepoFetchPlan, ref: str, on_progress: Callable[[int, int], None] | None = None
) -> list[RepoFile]:
    """Fallback for oversized repos: download only the planned files.

    This is the slow path a >300MB archive falls back to (see
    fetch_repo_files_for_indexing) — thousands of individual HTTP requests
    that can take minutes. `on_progress` is reported at most every 2% of the
    total so an oversized repo shows real progress instead of a status
    endpoint stuck on "queued, 0 chunks" the whole time.
    """

    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.RAG_MAX_SEARCH_CONCURRENCY)
    total = len(plan.selected_paths)
    done = 0
    report_every = max(1, total // 50)

    async def fetch(path: str) -> RepoFile | None:
        nonlocal done
        async with semaphore:
            content = await _fetch_file_content(client, plan.repository.full_name, path, ref, cache=False)
        done += 1
        if on_progress and (done % report_every == 0 or done == total):
            on_progress(done, total)
        content = _readable_content(path, content) if content else None
        if not content:
            return None
        return RepoFile(path=path, content=content, size=len(content))

    results = await asyncio.gather(*(fetch(path) for path in plan.selected_paths))
    ordered = [item for item in results if item is not None]
    logger.info("Fetched %d files for %s via raw downloads", len(ordered), plan.repository.full_name)
    return ordered


def _readable_content(path: str, content: str) -> str:
    """Turn a fetched file into indexable text; notebooks become their cells."""

    if not path.lower().endswith(".ipynb"):
        return content
    try:
        cells = json.loads(content).get("cells", [])
    except (ValueError, AttributeError):
        return ""
    blocks = []
    for cell in cells:
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        if not source.strip():
            continue
        # Markdown cells become comments so the result still reads as one Python file.
        if cell.get("cell_type") == "markdown":
            source = "\n".join(f"# {line}" for line in source.splitlines())
        blocks.append(source.rstrip())
    return "\n\n".join(blocks)


async def _fetch_file_content(
    client, full_name: str, path: str, ref: str = "HEAD", *, cache: bool = True
) -> str | None:
    """Fetch a single file from raw.githubusercontent.com.

    Raw downloads don't count against the 5,000/hour REST quota, which the
    contents API was burning ~125 calls per research request on. The token
    is still sent because GitHub rate-limits unauthenticated raw requests.
    Whole-repo indexing passes cache=False so thousands of files don't pile
    up in the shared in-memory cache.
    """

    ck = f"file:{full_name}:{ref}:{path}"
    if cache and ck in _cache:
        return _cache[ck]

    settings = get_settings()
    headers = {"Authorization": f"token {settings.GITHUB_TOKEN}"} if settings.GITHUB_TOKEN else {}
    url = f"{_RAW_BASE}/{full_name}/{ref}/{path}"
    resp = await client.get(url, headers=headers)
    # A 429 or transient 5xx from raw.githubusercontent.com under heavy
    # concurrent load (whole-repo indexing fires thousands of these) used to
    # be treated identically to "file doesn't exist" — silently dropping a
    # real file from the index with no signal. Retry those specifically.
    for delay in (0.5, 1.5):
        if resp.status_code != 429 and resp.status_code < 500:
            break
        await asyncio.sleep(delay)
        resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        return None

    content = resp.text
    if not content or "\x00" in content or len(content) > settings.RAG_MAX_FILE_SIZE:
        return None
    if cache:
        _cache[ck] = content
    return content