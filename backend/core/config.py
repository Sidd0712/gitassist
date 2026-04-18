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

    # GitHub
    GITHUB_TOKEN: str = ""
    GITHUB_API_BASE: str = "https://api.github.com"
    GITHUB_SEARCH_LIMIT: int = 5  # top N repos to fetch

    # LLM / Groq API
    GROQ_API_KEY: str = ""  # Groq API key (get from console.groq.com)
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Groq's latest 70B model (3.1 was decommissioned)
    EMBEDDING_MODEL: str = "hash-embedding-384-v1"  # Low-memory deterministic embeddings for web/worker use
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_RETRIES: int = 3
    LLM_AUX_TIMEOUT_SECONDS: int = 20
    LLM_GENERATION_TIMEOUT_SECONDS: int = 35

    # Cache
    GITHUB_CACHE_TTL: int = 3600  # seconds
    CACHE_LLM_TTL_SECONDS: int = 3600
    CACHE_GITHUB_ARTIFACT_TTL_SECONDS: int = 3600
    CACHE_RETRIEVAL_TTL_SECONDS: int = 1800

    # RAG
    RAG_CANDIDATE_REPO_LIMIT: int = 18
    RAG_README_RERANK_LIMIT: int = 12
    RAG_OUTPUT_REPO_LIMIT: int = 8
    RAG_SEARCH_PER_QUERY: int = 10
    RAG_DEEP_INDEX_REPO_LIMIT: int = 5
    RAG_INLINE_BOOTSTRAP_REPO_LIMIT: int = 5
    RAG_MAX_FILES_PER_REPO: int = 40
    RAG_MAX_CHARS_PER_REPO: int = 150_000
    RAG_MAX_FILE_SIZE: int = 80_000
    RAG_ARCHIVE_MAX_BYTES: int = 25_000_000
    RAG_ARCHIVE_MAX_EXTRACTED_BYTES: int = 20_000_000
    RAG_ARCHIVE_MAX_FILES: int = 400
    RAG_CHUNK_TOKENS: int = 400
    RAG_CHUNK_OVERLAP: int = 60
    RAG_SECTION_TOP_K: int = 8
    RAG_SECTION_CONTEXT_CHARS: int = 14_000
    RAG_MAX_SEARCH_CONCURRENCY: int = 8
    RAG_QUERY_LIMIT: int = 6
    RAG_DEDUP_THRESHOLD: float = 0.92
    RAG_MAX_PER_LANGUAGE: int = 4
    RAG_CHUNKING_VERSION: str = "v1"
    RAG_STORE_BACKEND: str = "postgres"
    DATABASE_URL: str = ""
    PGVECTOR_DIMENSION: int = 384
    INDEXING_MODE: str = "inline"
    INDEX_JOB_TIMEOUT_SECONDS: int = 900
    INDEX_JOB_MAX_RETRIES: int = 3
    INDEXER_POLL_INTERVAL_SECONDS: int = 5
    INDEXER_MAX_CONCURRENCY: int = 2
    RAG_DB_POOL_MIN_SIZE: int = 1
    RAG_DB_POOL_MAX_SIZE: int = 4
    PIPELINE_REQUEST_BUDGET_SECONDS: int = 110
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
