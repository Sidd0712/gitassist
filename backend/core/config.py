"""Application configuration via environment variables."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings


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

    # LLM / OpenAI-compatible
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_BASE_URL: str = ""  # leave empty for default OpenAI
    LLM_TIMEOUT_SECONDS: int = 20
    LLM_MAX_RETRIES: int = 0
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_DIMENSIONS: int = 0
    EMBEDDING_TIMEOUT_SECONDS: int = 15
    EMBEDDING_MAX_RETRIES: int = 0

    # Cache
    GITHUB_CACHE_TTL: int = 3600  # seconds

    # RAG
    RAG_CANDIDATE_REPO_LIMIT: int = 20
    RAG_SEARCH_PER_QUERY: int = 8
    RAG_DEEP_INDEX_REPO_LIMIT: int = 4
    RAG_MAX_FILES_PER_REPO: int = 60
    RAG_MAX_CHARS_PER_REPO: int = 250_000
    RAG_MAX_FILE_SIZE: int = 120_000
    RAG_CHUNK_TOKENS: int = 400
    RAG_CHUNK_OVERLAP: int = 60
    RAG_SECTION_TOP_K: int = 8
    RAG_SECTION_CONTEXT_CHARS: int = 14_000
    RAG_MAX_SEARCH_CONCURRENCY: int = 4
    RAG_CHUNKING_VERSION: str = "v1"
    RAG_VECTOR_STORE_PATH: str = "backend/data/chroma"
    RAG_SQLITE_PATH: str = "backend/data/retrieval.db"
    RAG_CORPUS_PATH: str = "backend/data/corpora"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

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
