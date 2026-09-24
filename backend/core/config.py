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
    CHAT_RATE_LIMIT_PER_HOUR: int = 60  # Separate bucket: chat is the main feature

    # GitHub
    GITHUB_TOKEN: str = ""
    GITHUB_API_BASE: str = "https://api.github.com"

    # LLM / Groq API
    GROQ_API_KEY: str = ""  # Groq API key (get from console.groq.com)
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was decommissioned 2026-09; verify model availability at console.groq.com before changing
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_RETRIES: int = 3
    # Used only when Groq itself fails (rate limit, outage, retired model).
    # Free AI Studio tier, no card: 15 requests/min, 500/day for this model.
    # Empty either one to disable the fallback.
    GOOGLE_API_KEY: str = ""
    LLM_FALLBACK_MODEL: str = "gemini-3.5-flash-lite"
    # Same free limits (15 RPM / 250K TPM / 500 RPD) on a separate quota, tried
    # when the first Gemini model is rate-limited. Empty to disable.
    LLM_SECONDARY_GEMINI_MODEL: str = "gemini-3.1-flash-lite"
    LLM_AUX_TIMEOUT_SECONDS: int = 20
    LLM_GENERATION_TIMEOUT_SECONDS: int = 35

    # Repo chat. Groq's free tier is 8K tokens/min per model, far too small for
    # real code context, so chat goes Gemini-first (250K TPM) and only falls
    # back to Groq with a trimmed context.
    CHAT_CONTEXT_CHARS: int = 60_000
    CHAT_FALLBACK_CONTEXT_CHARS: int = 10_000
    CHAT_MAX_OUTPUT_TOKENS: int = 2500
    CHAT_TIMEOUT_SECONDS: int = 45

    # Embeddings
    # "local_worker": the backend queues texts in Postgres (embedding_jobs) and
    # a worker on the developer's own machine (backend/scripts/local_embed_worker.py)
    # embeds them with fastembed and writes the vectors back. Free, uncapped,
    # and within every provider's terms — GitHub Actions was dropped because its
    # terms prohibit use "as part of a serverless application". When the worker
    # is offline, embedding calls fail fast with EmbeddingUnavailable and every
    # caller degrades (reports still work; indexing/chat pause). "cohere" is
    # kept as a fallback path.
    EMBEDDING_PROVIDER: str = "local_worker"
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"  # 384-dim — matches PGVECTOR_DIMENSION, no schema change needed

    LOCAL_WORKER_POLL_INTERVAL_SECONDS: float = 0.5
    # The worker heartbeats every ~5s; a quiet period longer than this means it's
    # offline (PC asleep/off), so callers fail fast instead of queueing a job
    # nobody will pick up.
    LOCAL_WORKER_HEARTBEAT_MAX_AGE_SECONDS: float = 20.0
    LOCAL_WORKER_QUERY_TIMEOUT_SECONDS: float = 60.0
    # Document jobs are now only short repo-metadata batches from candidate
    # search; whole repos are indexed by the worker's own repo lane.
    LOCAL_WORKER_DOCUMENT_TIMEOUT_SECONDS: float = 120.0

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
    RAG_SHALLOW_CODE_SAMPLE_COUNT: int = 8  # was 5; more diverse ranking signal
    RAG_SHALLOW_CODE_SAMPLE_MAX_CHARS: int = 6_000
    RAG_SEMANTIC_MIN_RELEVANCE: float = 0.30
    # Whole-repo indexing runs on the developer's machine (repo lane of
    # scripts/local_embed_worker.py), so these caps protect Neon's 0.5 GB
    # free tier (~4.6 KB/chunk incl. indexes), not Render's RAM. Files are
    # indexed in priority order, so a capped repo keeps its most useful files.
    RAG_REPO_MAX_CHUNKS: int = 8000
    RAG_CORPUS_MAX_CHUNKS: int = 60_000
    RAG_MAX_FILE_SIZE: int = 150_000
    RAG_MAX_DATA_FILE_SIZE: int = 20_000  # JSON/YAML/CSV-like files above this are data, not config
    RAG_ARCHIVE_MAX_BYTES: int = 300_000_000  # worker-side; larger repos fall back to raw file downloads
    # Chunk size in characters. bge-small reads at most 512 tokens (~1,600
    # chars of code); anything past that is silently never embedded.
    RAG_CHUNK_TARGET_CHARS: int = 1200
    RAG_CHUNK_MAX_CHARS: int = 1600
    RAG_CHUNK_OVERLAP_LINES: int = 3
    INDEX_JOB_MAX_RETRIES: int = 3
    INDEX_JOB_STALE_MINUTES: int = 30  # a 'running' job older than this belonged to a dead worker
    RAG_SECTION_TOP_K: int = 8
    RAG_SECTION_CONTEXT_CHARS: int = 14_000
    RAG_MAX_SEARCH_CONCURRENCY: int = 8
    RAG_QUERY_LIMIT: int = 8
    RAG_MAX_PER_LANGUAGE: int = 4
    # v3: whole-repo indexing, top-level-only JS/TS boundaries, merged small
    # spans, char-bounded chunks. Chunks from other versions are purged at startup.
    RAG_CHUNKING_VERSION: str = "v3"
    RAG_STORE_BACKEND: str = "postgres"
    DATABASE_URL: str = ""
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
