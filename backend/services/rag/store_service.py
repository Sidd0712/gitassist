"""Shared Postgres + pgvector persistence for repo corpora."""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

from core.config import get_settings
from models.schemas import RepoSearchResult

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised in integration environments
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None

try:  # pragma: no cover - exercised in integration environments
    from pgvector.psycopg import register_vector
    from psycopg import OperationalError, sql
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - optional runtime dependency
    ConnectionPool = None
    OperationalError = Exception
    dict_row = None
    register_vector = None
    sql = None


def _retry_on_connection_loss(func):
    """Retry a Postgres operation once if the pooled connection died before use.

    Managed Postgres providers (Neon, Supabase, etc.) recycle idle connections
    without warning — psycopg_pool detects and discards the dead connection,
    but the in-flight query still raises OperationalError (e.g. AdminShutdown)
    on first use. One retry acquires a fresh connection from the pool instead
    of failing the whole user request over a transient blip.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except OperationalError as exc:
            logger.warning(
                "Postgres operation %s failed (%s); retrying once with a fresh connection",
                func.__name__,
                exc,
            )
            return func(self, *args, **kwargs)

    return wrapper


def _load_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def _as_vector(values: list[float]) -> Any:
    if np is None:
        return values
    return np.asarray(values, dtype=np.float32)


def _allowed_repo_clause(allowed_repos: dict[str, str]) -> tuple[Any, list[Any]]:
    if sql is None:
        raise RuntimeError("psycopg is not installed")

    clauses = []
    params: list[Any] = []
    for full_name, commit_sha in allowed_repos.items():
        clauses.append(sql.SQL("(full_name = %s AND commit_sha = %s)"))
        params.extend([full_name, commit_sha])
    return sql.SQL(" OR ").join(clauses), params


class RAGStore:
    """Shared Postgres-backed store for repo corpora and chunks."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.dimension = self.settings.PGVECTOR_DIMENSION
        self._pool: ConnectionPool | None = None
        self._ready = False

    @property
    def backend_name(self) -> str:
        return self.settings.RAG_STORE_BACKEND

    def ensure_ready(self) -> None:
        """Initialize the shared Postgres schema and indexes."""

        if self._ready:
            return

        if self.settings.RAG_STORE_BACKEND.lower() != "postgres":
            raise RuntimeError("RAG_STORE_BACKEND must be set to 'postgres'.")
        if not self.settings.DATABASE_URL:
            raise RuntimeError("DATABASE_URL must be set for the Postgres RAG store.")
        if ConnectionPool is None or register_vector is None or dict_row is None or sql is None:
            raise RuntimeError(
                "Postgres RAG dependencies are missing. Install psycopg[binary], psycopg-pool, and pgvector."
            )

        def configure(conn) -> None:
            register_vector(conn)

        self._pool = ConnectionPool(
            conninfo=self.settings.DATABASE_URL,
            min_size=self.settings.RAG_DB_POOL_MIN_SIZE,
            max_size=self.settings.RAG_DB_POOL_MAX_SIZE,
            timeout=30.0,
            kwargs={"autocommit": True, "row_factory": dict_row},
            configure=configure,
            open=True,
        )

        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_indexes (
                    full_name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    index_state TEXT NOT NULL DEFAULT 'completed',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    repo_json JSONB NOT NULL,
                    indexed_at TIMESTAMPTZ,
                    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (full_name, commit_sha, embedding_model, chunking_version)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS corpus_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    path TEXT NOT NULL,
                    chunk_role TEXT NOT NULL,
                    language TEXT,
                    symbol TEXT,
                    heading TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    token_count INTEGER NOT NULL,
                    repo_score DOUBLE PRECISION NOT NULL,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding VECTOR({self.dimension}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS repo_indexes_state_idx
                ON repo_indexes (index_state, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS corpus_chunks_repo_idx
                ON corpus_chunks (full_name, commit_sha, embedding_model, chunking_version)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS corpus_chunks_embedding_hnsw_idx
                ON corpus_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
            # Job queue for the local-worker embedding provider: the backend
            # writes pending texts here, the worker on the developer's machine
            # claims them, embeds, and writes the result back — keeping
            # embedding compute off this process's CPU/RAM budget entirely.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_jobs (
                    job_id UUID PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    is_query BOOLEAN NOT NULL DEFAULT FALSE,
                    input_texts JSONB NOT NULL,
                    result JSONB,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS embedding_jobs_created_idx
                ON embedding_jobs (created_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_workers (
                    worker_id TEXT PRIMARY KEY,
                    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        self._ready = True
        self._evict_stale_repositories()
        self._evict_old_embedding_jobs()

    @_retry_on_connection_loss
    def _evict_stale_repositories(self) -> None:
        """Drop the oldest-by-last-use indexed repos beyond MAX_INDEXED_REPOS.

        No background job runs this — it's cheap enough to run inline at
        startup, and this app never accumulates indexes fast enough to need
        more than that.
        """

        max_repos = self.settings.MAX_INDEXED_REPOS
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                stale = conn.execute(
                    """
                    SELECT full_name, commit_sha, embedding_model, chunking_version
                    FROM repo_indexes
                    ORDER BY last_accessed_at DESC
                    OFFSET %s
                    """,
                    (max_repos,),
                ).fetchall()
                for row in stale:
                    key = (row["full_name"], row["commit_sha"], row["embedding_model"], row["chunking_version"])
                    conn.execute(
                        """
                        DELETE FROM corpus_chunks
                        WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                        """,
                        key,
                    )
                    conn.execute(
                        """
                        DELETE FROM repo_indexes
                        WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                        """,
                        key,
                    )
        if stale:
            logger.info("Evicted %d stale indexed repositories beyond MAX_INDEXED_REPOS=%d", len(stale), max_repos)

    @_retry_on_connection_loss
    def _evict_old_embedding_jobs(self) -> None:
        """
        Delete embedding_jobs rows older than 1 day.

        input_texts/result payloads can be several MB each (a full repo's
        chunk texts); the job row is only needed for the few minutes between
        dispatch and the worker writing its result, so nothing legitimate is
        ever this old — only orphans from a worker that crashed or a request
        that was abandoned before polling picked up the result.
        """

        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute("DELETE FROM embedding_jobs WHERE created_at < NOW() - INTERVAL '1 day'")

    @_retry_on_connection_loss
    def create_embedding_job(self, job_id: str, texts: list[str], is_query: bool) -> None:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                INSERT INTO embedding_jobs (job_id, status, is_query, input_texts)
                VALUES (%s, 'pending', %s, %s::jsonb)
                """,
                (job_id, is_query, json.dumps(texts)),
            )

    @_retry_on_connection_loss
    def get_embedding_job(self, job_id: str) -> dict[str, Any] | None:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            row = conn.execute(
                "SELECT status, result, error FROM embedding_jobs WHERE job_id = %s",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": row["status"],
            "result": _load_json(row["result"], None),
            "error": row["error"],
        }

    @_retry_on_connection_loss
    def delete_embedding_job(self, job_id: str) -> None:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute("DELETE FROM embedding_jobs WHERE job_id = %s", (job_id,))

    @_retry_on_connection_loss
    def claim_embedding_job(self, is_query: bool) -> dict[str, Any] | None:
        """Atomically take the oldest pending job of one kind, or None if idle.

        SKIP LOCKED lets more than one worker process poll safely without ever
        handing the same job to two of them.
        """
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            row = conn.execute(
                """
                UPDATE embedding_jobs SET status = 'running'
                WHERE job_id = (
                    SELECT job_id FROM embedding_jobs
                    WHERE status = 'pending' AND is_query = %s
                    ORDER BY created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING job_id, input_texts
                """,
                (is_query,),
            ).fetchone()
        if row is None:
            return None
        return {"job_id": str(row["job_id"]), "texts": _load_json(row["input_texts"], [])}

    @_retry_on_connection_loss
    def complete_embedding_job(self, job_id: str, vectors: list[list[float]]) -> None:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                UPDATE embedding_jobs
                SET status = 'complete', result = %s::jsonb, completed_at = NOW()
                WHERE job_id = %s
                """,
                (json.dumps(vectors), job_id),
            )

    @_retry_on_connection_loss
    def fail_embedding_job(self, job_id: str, error: str) -> None:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                UPDATE embedding_jobs
                SET status = 'failed', error = %s, completed_at = NOW()
                WHERE job_id = %s
                """,
                (error[:2000], job_id),
            )

    @_retry_on_connection_loss
    def record_embedding_worker_heartbeat(self, worker_id: str) -> None:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                INSERT INTO embedding_workers (worker_id, last_seen) VALUES (%s, NOW())
                ON CONFLICT (worker_id) DO UPDATE SET last_seen = NOW()
                """,
                (worker_id,),
            )

    @_retry_on_connection_loss
    def embedding_worker_alive(self, max_age_seconds: float) -> bool:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            row = conn.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM embedding_workers
                    WHERE last_seen > NOW() - make_interval(secs => %s)
                ) AS alive
                """,
                (max_age_seconds,),
            ).fetchone()
        return bool(row and row["alive"])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._ready = False

    @_retry_on_connection_loss
    def get_stats(self) -> dict[str, int]:
        """Return a lightweight store summary for startup logging."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN index_state = 'completed' THEN 1 ELSE 0 END), 0) AS completed_repos,
                    COALESCE((SELECT COUNT(*) FROM corpus_chunks), 0) AS chunks
                FROM repo_indexes
                """
            ).fetchone()
        return {
            "completed_repos": int(row["completed_repos"]) if row else 0,
            "chunks": int(row["chunks"]) if row else 0,
        }

    @_retry_on_connection_loss
    def is_indexed(self, repository: RepoSearchResult, embedding_model: str, chunking_version: str) -> bool:
        """Check whether a repo commit is already indexed for the active corpus settings."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            row = conn.execute(
                """
                SELECT 1
                FROM repo_indexes
                WHERE full_name = %s
                  AND commit_sha = %s
                  AND embedding_model = %s
                  AND chunking_version = %s
                  AND index_state = 'completed'
                """,
                (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE repo_indexes
                    SET last_accessed_at = NOW(), updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )
        return row is not None

    @_retry_on_connection_loss
    def load_repository(
        self,
        full_name: str,
        commit_sha: str,
        embedding_model: str,
        chunking_version: str,
    ) -> RepoSearchResult | None:
        """Load stored repo metadata for a completed repo index."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            row = conn.execute(
                """
                SELECT repo_json
                FROM repo_indexes
                WHERE full_name = %s
                  AND commit_sha = %s
                  AND embedding_model = %s
                  AND chunking_version = %s
                  AND index_state = 'completed'
                """,
                (full_name, commit_sha, embedding_model, chunking_version),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE repo_indexes
                    SET last_accessed_at = NOW(), updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (full_name, commit_sha, embedding_model, chunking_version),
                )
        if row is None:
            return None
        return RepoSearchResult(**_load_json(row["repo_json"], {}))

    @_retry_on_connection_loss
    def replace_repository_index(
        self,
        repository: RepoSearchResult,
        chunks: list[Any],
        embeddings: list[list[float]],
        embedding_model: str,
        chunking_version: str,
    ) -> None:
        """Upsert a completed repo index and replace its stored chunks in Postgres."""

        self.ensure_ready()
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have a matching embedding vector.")

        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO repo_indexes (
                        full_name, commit_sha, embedding_model, chunking_version,
                        index_state, chunk_count, repo_json, indexed_at, last_accessed_at, updated_at
                    ) VALUES (%s, %s, %s, %s, 'completed', %s, %s::jsonb, NOW(), NOW(), NOW())
                    ON CONFLICT (full_name, commit_sha, embedding_model, chunking_version)
                    DO UPDATE SET
                        index_state = 'completed',
                        chunk_count = EXCLUDED.chunk_count,
                        repo_json = EXCLUDED.repo_json,
                        indexed_at = NOW(),
                        last_accessed_at = NOW(),
                        updated_at = NOW()
                    """,
                    (
                        repository.full_name,
                        repository.commit_sha,
                        embedding_model,
                        chunking_version,
                        len(chunks),
                        json.dumps(repository.model_dump()),
                    ),
                )

                conn.execute(
                    """
                    DELETE FROM corpus_chunks
                    WHERE full_name = %s
                      AND commit_sha = %s
                      AND embedding_model = %s
                      AND chunking_version = %s
                    """,
                    (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )

                if chunks:
                    rows = [
                        (
                            chunk.chunk_id,
                            chunk.repo_full_name,
                            chunk.commit_sha,
                            embedding_model,
                            chunking_version,
                            chunk.path,
                            chunk.chunk_role,
                            chunk.language,
                            chunk.symbol,
                            chunk.heading,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.token_count,
                            chunk.repo_score,
                            chunk.content_hash,
                            chunk.text,
                            _as_vector(embedding),
                        )
                        for chunk, embedding in zip(chunks, embeddings)
                    ]
                    with conn.cursor() as cursor:
                        cursor.executemany(
                            """
                            INSERT INTO corpus_chunks (
                                chunk_id,
                                full_name,
                                commit_sha,
                                embedding_model,
                                chunking_version,
                                path,
                                chunk_role,
                                language,
                                symbol,
                                heading,
                                start_line,
                                end_line,
                                token_count,
                                repo_score,
                                content_hash,
                                text,
                                embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            rows,
                        )

    @_retry_on_connection_loss
    def dense_search(
        self,
        query_embedding: list[float],
        allowed_repos: dict[str, str],
        top_k: int,
        embedding_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense retrieval directly in Postgres using pgvector cosine distance."""

        self.ensure_ready()
        if not query_embedding or not allowed_repos:
            return []

        clause, params = _allowed_repo_clause(allowed_repos)
        query = sql.SQL(
            """
            SELECT
                chunk_id,
                full_name,
                commit_sha,
                path,
                chunk_role,
                language,
                start_line,
                end_line,
                repo_score,
                text,
                1 - (embedding <=> %s) AS dense_score
            FROM corpus_chunks
            WHERE embedding_model = %s
              AND chunking_version = %s
              AND ({allowed_clause})
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        ).format(allowed_clause=clause)

        with self._pool.connection() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                query,
                (
                    _as_vector(query_embedding),
                    embedding_model or self.settings.EMBEDDING_MODEL,
                    self.settings.RAG_CHUNKING_VERSION,
                    *params,
                    _as_vector(query_embedding),
                    # Fetch a wider pool than top_k so RetrievalService can re-rank by
                    # chunk_role (preferred_roles) on top of raw cosine distance —
                    # a role-preferred chunk ranked just outside a tight top_k*2
                    # window would otherwise never get the chance to surface.
                    max(top_k * 4, 20),
                ),
            ).fetchall()

        return [
            {
                "chunk_id": row["chunk_id"],
                "repo_full_name": row["full_name"],
                "path": row["path"],
                "chunk_role": row["chunk_role"],
                "language": row["language"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "repo_score": float(row["repo_score"]),
                "text": row["text"],
                "dense_score": max(0.0, float(row["dense_score"] or 0.0)),
            }
            for row in rows
        ]

_store: RAGStore | None = None


def get_rag_store() -> RAGStore:
    """Return the singleton store instance."""

    global _store
    if _store is None:
        _store = RAGStore()
    return _store
