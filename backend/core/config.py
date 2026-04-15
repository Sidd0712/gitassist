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

    # LLM / Groq API
    GROQ_API_KEY: str = ""  # Groq API key (get from console.groq.com)
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Groq's latest 70B model (3.1 was decommissioned)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight free embeddings
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_RETRIES: int = 3

    # Cache
    GITHUB_CACHE_TTL: int = 3600  # seconds

    # RAG
    RAG_CANDIDATE_REPO_LIMIT: int = 30
    RAG_README_RERANK_LIMIT: int = 30
    RAG_OUTPUT_REPO_LIMIT: int = 10
    RAG_SEARCH_PER_QUERY: int = 10
    RAG_DEEP_INDEX_REPO_LIMIT: int = 4
    RAG_MAX_FILES_PER_REPO: int = 60
    RAG_MAX_CHARS_PER_REPO: int = 250_000
    RAG_MAX_FILE_SIZE: int = 120_000
    RAG_CHUNK_TOKENS: int = 400
    RAG_CHUNK_OVERLAP: int = 60
    RAG_SECTION_TOP_K: int = 8
    RAG_SECTION_CONTEXT_CHARS: int = 14_000
    RAG_MAX_SEARCH_CONCURRENCY: int = 4
    RAG_QUERY_LIMIT: int = 10
    RAG_DEDUP_THRESHOLD: float = 0.92
    RAG_MAX_PER_LANGUAGE: int = 4
    RAG_CHUNKING_VERSION: str = "v1"
    RAG_VECTOR_STORE_PATH: str = "data/chroma"
    RAG_SQLITE_PATH: str = "data/retrieval.db"
    RAG_CORPUS_PATH: str = "data/corpora"

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
