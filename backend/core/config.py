"""Application configuration via environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Global application settings loaded from .env or environment."""

    # App
    APP_NAME: str = "GitAssist AI"
    DEBUG: bool = True

    # CORS
    FRONTEND_URL: str = "http://localhost:5173"

    # Auth
    API_KEY: str = ""  # If set, required via X-API-Key header on /research and /research/chat
    RATE_LIMIT_PER_HOUR: int = 20  # Max /research calls per IP per rolling hour

    # GitHub
    GITHUB_TOKEN: str = ""
    GITHUB_API_BASE: str = "https://api.github.com"

    # LLM / Groq API
    GROQ_API_KEY: str = ""  # Groq API key (get from console.groq.com)
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was decommissioned 2026-09; verify model availability at console.groq.com before changing
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_RETRIES: int = 3
    LLM_AUX_TIMEOUT_SECONDS: int = 20
    LLM_GENERATION_TIMEOUT_SECONDS: int = 35

    # Embeddings
    # "github_actions" dispatches embedding batches to a GitHub Actions workflow
    # (backend/scripts/gh_embed_worker.py via .github/workflows/embed-worker.yml)
    # running fastembed locally on the runner — free, no per-minute/monthly cap,
    # and no RAM cost to this process, at the price of a per-job dispatch/queue/
    # runner-startup latency (~15-35s measured live 2026-09). "cohere" is kept
    # as a fallback path.
    EMBEDDING_PROVIDER: str = "github_actions"
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"  # 384-dim — matches PGVECTOR_DIMENSION, no schema change needed

    GITHUB_ACTIONS_OWNER: str = "Sidd0712"
    GITHUB_ACTIONS_REPO: str = "gitassist"
    GITHUB_ACTIONS_WORKFLOW_FILE: str = "embed-worker.yml"
    # Needs `repo` + `workflow` scopes (a classic PAT, or a fine-grained PAT with
    # Actions:write + Contents:read on this repo) — separate from GITHUB_TOKEN,
    # which only needs public_repo/search access for GitHub search and is not
    # sufficient to dispatch workflows on a private repo.
    GITHUB_ACTIONS_TRIGGER_TOKEN: str = ""
    GITHUB_ACTIONS_POLL_INTERVAL_SECONDS: float = 2.0
    GITHUB_ACTIONS_JOB_TIMEOUT_SECONDS: float = 120.0

    # Cohere (used only when EMBEDDING_PROVIDER=cohere)
    COHERE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "embed-english-light-v3.0"
    PGVECTOR_DIMENSION: int = 384

    # Cache
    GITHUB_CACHE_TTL: int = 3600  # seconds
    CACHE_LLM_TTL_SECONDS: int = 3600
    CACHE_GITHUB_ARTIFACT_TTL_SECONDS: int = 3600
    CACHE_RETRIEVAL_TTL_SECONDS: int = 1800

    # RAG
    RAG_CANDIDATE_REPO_LIMIT: int = 40
    RAG_README_RERANK_LIMIT: int = 20
    RAG_OUTPUT_REPO_LIMIT: int = 5
    RAG_SEARCH_PER_QUERY: int = 20
    RAG_DEEP_INDEX_REPO_LIMIT: int = 5
    # RAG limits — right-sized for what generation actually reads, not for
    # hypothetical "ingest everything" completeness.
    #
    # Measured (2026-08-07): the 4 generation calls only ever read hits[:6] truncated
    # to ~900 chars each per section (_build_generation_evidence in llm_service.py) —
    # a few thousand characters total, regardless of how much is indexed. Indexing
    # depth beyond that exists purely to make repo-chat's open-ended questions answerable
    # later, which is a real but genuinely smaller need than "ingest every eligible file."
    # Single-provider (Cohere only, 1000 calls/month, call-metered):
    # 150 files × ~5 avg chunks × 5 repos = 3,750 chunks ÷ 96 batch ≈ 40 embed calls
    # + ~11 other calls ≈ 51/request ≈ 19-20 full requests/month on the free tier.
    # Watch the "N/M files in tree are index-eligible" log line for real per-repo usage.
    RAG_SHALLOW_CODE_SAMPLE_COUNT: int = 8  # was 5; more diverse ranking signal
    RAG_SHALLOW_CODE_SAMPLE_MAX_CHARS: int = 6_000
    RAG_SEMANTIC_MIN_RELEVANCE: float = 0.30
    RAG_MAX_FILES_PER_REPO: int = 150
    RAG_MAX_CHARS_PER_REPO: int = 1_800_000
    RAG_MAX_FILE_SIZE: int = 80_000
    RAG_ARCHIVE_MAX_BYTES: int = 25_000_000
    RAG_ARCHIVE_MAX_EXTRACTED_BYTES: int = 20_000_000
    RAG_ARCHIVE_MAX_FILES: int = 500  # was 400 — wider scan window before zip cutoff
    RAG_CHUNK_TOKENS: int = 300
    RAG_CHUNK_OVERLAP: int = 60
    RAG_SECTION_TOP_K: int = 8
    RAG_SECTION_CONTEXT_CHARS: int = 14_000
    RAG_MAX_SEARCH_CONCURRENCY: int = 8
    RAG_QUERY_LIMIT: int = 8
    RAG_MAX_PER_LANGUAGE: int = 4
    # Bumped v1->v2: restored ingestion depth (RAG_MAX_FILES_PER_REPO,
    # RAG_MAX_CHARS_PER_REPO, RAG_SHALLOW_CODE_SAMPLE_COUNT) to code defaults
    # after finding production had drifted to older, more conservative values.
    # A version bump forces every repo to re-index at the new depth instead of
    # silently continuing to serve the old, shallower index from cache.
    RAG_CHUNKING_VERSION: str = "v2"
    RAG_STORE_BACKEND: str = "postgres"
    DATABASE_URL: str = ""
    MAX_INDEXED_REPOS: int = 150  # Oldest-by-last-use repos beyond this cap are evicted at startup
    INDEXER_MAX_CONCURRENCY: int = 3  # Bounds concurrent inline indexing within one request
    RAG_DB_POOL_MIN_SIZE: int = 1
    RAG_DB_POOL_MAX_SIZE: int = 4
    PIPELINE_REQUEST_BUDGET_SECONDS: int = 150  # was 110 — accommodates deeper indexing
    PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS: int = 25
    GITHUB_HTTP_TIMEOUT_SECONDS: int = 30
    GITHUB_MAX_CONNECTIONS: int = 20
    GITHUB_MAX_KEEPALIVE_CONNECTIONS: int = 10

    model_config = {
        "env_file": str(BACKEND_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        """Accept common mode strings like `release` and `debug`."""

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"release", "prod", "production", "false", "0", "no", "off"}:
                return False
            if lowered in {"debug", "dev", "development", "true", "1", "yes", "on"}:
                return True
        return bool(value)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
