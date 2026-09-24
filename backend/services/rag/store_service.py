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


_CHUNK_COLUMNS = sql.SQL(
    "chunk_id, full_name, commit_sha, path, chunk_role, language, symbol, start_line, end_line, repo_score, text"
) if sql is not None else None
# Must stay identical to the corpus_chunks_fts_idx expression for the index to be used.
_FTS_DOCUMENT = sql.SQL(
    "to_tsvector('simple', COALESCE(path, '') || ' ' || COALESCE(symbol, '') || ' '"
    " || COALESCE(chunk_role, '') || ' ' || COALESCE(text, ''))"
) if sql is not None else None


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
            # Durable queue of whole-repo indexing work. Render enqueues; the
            # worker on the developer's machine claims and indexes, so jobs
            # simply wait while that machine is off.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_index_jobs (
                    full_name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    claimed_by TEXT,
                    claimed_at TIMESTAMPTZ,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (full_name, commit_sha, embedding_model, chunking_version)
                )
                """
            )
            conn.execute("ALTER TABLE repo_index_jobs ADD COLUMN IF NOT EXISTS repo_json JSONB")
            conn.execute("ALTER TABLE repo_index_jobs ADD COLUMN IF NOT EXISTS path_terms JSONB")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS repo_index_jobs_status_idx ON repo_index_jobs (status, updated_at)"
            )
            # Identifier/keyword search for chat. The expression must match
            # lexical_search()'s exactly for Postgres to use this index.
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS corpus_chunks_fts_idx ON corpus_chunks
                USING gin (to_tsvector('simple', COALESCE(path, '') || ' ' || COALESCE(symbol, '') || ' '
                    || COALESCE(chunk_role, '') || ' ' || COALESCE(text, '')))
                """
            )
        self._ready = True
        self._purge_other_chunking_versions()
        self._evict_old_embedding_jobs()

    @_retry_on_connection_loss
    def _purge_other_chunking_versions(self) -> None:
        """Drop chunks nothing can query anymore (retrieval only matches the current version)."""

        version = self.settings.RAG_CHUNKING_VERSION
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            deleted = conn.execute("DELETE FROM corpus_chunks WHERE chunking_version <> %s", (version,)).rowcount
            conn.execute("DELETE FROM repo_indexes WHERE chunking_version <> %s", (version,))
            conn.execute("DELETE FROM repo_index_jobs WHERE chunking_version <> %s", (version,))
        if deleted:
            logger.info("Purged %d chunks from chunking versions other than %s", deleted, version)

    @_retry_on_connection_loss
    def evict_to_budget(self, incoming_chunks: int, keep: tuple[str, str] | None = None) -> int:
        """Evict least-recently-used repos until incoming_chunks fits RAG_CORPUS_MAX_CHUNKS.

        Keeps Neon's 0.5 GB free tier from filling up; `keep` protects the
        repo about to be written. Returns how many repos were evicted.
        """

        budget = self.settings.RAG_CORPUS_MAX_CHUNKS
        evicted = 0
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            total = conn.execute("SELECT COUNT(*) AS n FROM corpus_chunks").fetchone()["n"]
            while total + incoming_chunks > budget:
                victim = conn.execute(
                    """
                    SELECT full_name, commit_sha, embedding_model, chunking_version, chunk_count
                    FROM repo_indexes
                    WHERE (full_name, commit_sha) IS DISTINCT FROM (%s, %s)
                    ORDER BY last_accessed_at ASC
                    LIMIT 1
                    """,
                    keep or (None, None),
                ).fetchone()
                if victim is None:
                    break
                key = (victim["full_name"], victim["commit_sha"], victim["embedding_model"], victim["chunking_version"])
                with conn.transaction():
                    removed = conn.execute(
                        """
                        DELETE FROM corpus_chunks
                        WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                        """,
                        key,
                    ).rowcount
                    for table in ("repo_indexes", "repo_index_jobs"):  # job row too, so it can be re-queued later
                        conn.execute(
                            sql.SQL(
                                "DELETE FROM {} WHERE full_name = %s AND commit_sha = %s"
                                " AND embedding_model = %s AND chunking_version = %s"
                            ).format(sql.Identifier(table)),
                            key,
                        )
                total -= removed
                evicted += 1
                logger.info("Evicted %s@%s (%d chunks) to stay under %d chunks", key[0], key[1][:8], removed, budget)
        return evicted

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
        """Load stored repo metadata for a searchable (partial or completed) repo index."""

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
                  AND index_state IN ('partial', 'completed')
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

    # ── Whole-repo index jobs ────────────────────────────────────────────────

    @_retry_on_connection_loss
    def enqueue_index_job(
        self, repository: RepoSearchResult, path_terms: list[str], embedding_model: str, chunking_version: str
    ) -> None:
        """Queue a repo commit for the worker; a no-op if it's already queued or indexed."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                INSERT INTO repo_index_jobs (
                    full_name, commit_sha, embedding_model, chunking_version, status, repo_json, path_terms
                ) VALUES (%s, %s, %s, %s, 'queued', %s::jsonb, %s::jsonb)
                ON CONFLICT (full_name, commit_sha, embedding_model, chunking_version) DO NOTHING
                """,
                (
                    repository.full_name,
                    repository.commit_sha,
                    embedding_model,
                    chunking_version,
                    json.dumps(repository.model_dump(exclude={"files"})),
                    json.dumps(path_terms),
                ),
            )

    @_retry_on_connection_loss
    def claim_index_job(self, worker_id: str, embedding_model: str, chunking_version: str) -> dict[str, Any] | None:
        """Take the oldest runnable job: queued, retryable, or abandoned by a dead worker."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            row = conn.execute(
                """
                UPDATE repo_index_jobs SET status = 'running', claimed_by = %s, claimed_at = NOW(), updated_at = NOW()
                WHERE (full_name, commit_sha, embedding_model, chunking_version) = (
                    SELECT full_name, commit_sha, embedding_model, chunking_version FROM repo_index_jobs
                    WHERE embedding_model = %s AND chunking_version = %s AND (
                        status = 'queued'
                        OR (status = 'failed' AND retry_count < %s)
                        OR (status = 'running' AND claimed_at < NOW() - make_interval(mins => %s))
                    )
                    ORDER BY enqueued_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING full_name, commit_sha, repo_json, path_terms
                """,
                (
                    worker_id,
                    embedding_model,
                    chunking_version,
                    self.settings.INDEX_JOB_MAX_RETRIES,
                    self.settings.INDEX_JOB_STALE_MINUTES,
                ),
            ).fetchone()
        if row is None:
            return None
        return {
            "full_name": row["full_name"],
            "commit_sha": row["commit_sha"],
            "repository": _load_json(row["repo_json"], {}),
            "path_terms": _load_json(row["path_terms"], []),
        }

    @_retry_on_connection_loss
    def release_index_jobs(self, worker_id: str) -> int:
        """Re-queue jobs this worker was running when it last stopped."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            return conn.execute(
                "UPDATE repo_index_jobs SET status = 'queued', updated_at = NOW() WHERE status = 'running' AND claimed_by = %s",
                (worker_id,),
            ).rowcount

    @_retry_on_connection_loss
    def finish_index_job(
        self, full_name: str, commit_sha: str, embedding_model: str, chunking_version: str, error: str | None = None
    ) -> None:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                UPDATE repo_index_jobs
                SET status = %s,
                    retry_count = retry_count + %s,
                    last_error = %s,
                    updated_at = NOW()
                WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                """,
                (
                    "failed" if error else "completed",
                    1 if error else 0,
                    error[:2000] if error else None,
                    full_name,
                    commit_sha,
                    embedding_model,
                    chunking_version,
                ),
            )

    @_retry_on_connection_loss
    def index_status(self, repos: list[tuple[str, str]], embedding_model: str, chunking_version: str) -> dict[str, dict]:
        """Per-repo index progress for the UI, keyed by full_name."""

        self.ensure_ready()
        if not repos:
            return {}
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                """
                SELECT r.full_name, i.index_state, i.chunk_count, j.status AS job_status,
                       j.retry_count, j.last_error
                FROM unnest(%s::text[], %s::text[]) AS r(full_name, commit_sha)
                LEFT JOIN repo_indexes i
                  ON i.full_name = r.full_name AND i.commit_sha = r.commit_sha
                 AND i.embedding_model = %s AND i.chunking_version = %s
                LEFT JOIN repo_index_jobs j
                  ON j.full_name = r.full_name AND j.commit_sha = r.commit_sha
                 AND j.embedding_model = %s AND j.chunking_version = %s
                """,
                (
                    [name for name, _ in repos],
                    [sha for _, sha in repos],
                    embedding_model,
                    chunking_version,
                    embedding_model,
                    chunking_version,
                ),
            ).fetchall()

        status: dict[str, dict] = {}
        for row in rows:
            state = row["index_state"]
            if state not in ("partial", "completed", "indexing"):
                if row["job_status"] == "failed" and row["retry_count"] >= self.settings.INDEX_JOB_MAX_RETRIES:
                    state = "failed"
                elif row["job_status"] in ("queued", "running", "failed"):
                    state = "queued"
                else:
                    state = "not_queued"
            status[row["full_name"]] = {
                "state": state,
                "chunk_count": int(row["chunk_count"] or 0),
                "error": row["last_error"] if state == "failed" else None,
            }
        return status

    # ── Progressive chunk writes (worker side) ──────────────────────────────

    @_retry_on_connection_loss
    def begin_repository_index(self, repository: RepoSearchResult, embedding_model: str, chunking_version: str) -> None:
        """Reset a repo commit to an empty 'indexing' state before chunks stream in."""

        self.ensure_ready()
        key = (repository.full_name, repository.commit_sha, embedding_model, chunking_version)
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                conn.execute(
                    """
                    DELETE FROM corpus_chunks
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    key,
                )
                conn.execute(
                    """
                    INSERT INTO repo_indexes (
                        full_name, commit_sha, embedding_model, chunking_version,
                        index_state, chunk_count, repo_json, last_accessed_at, updated_at
                    ) VALUES (%s, %s, %s, %s, 'indexing', 0, %s::jsonb, NOW(), NOW())
                    ON CONFLICT (full_name, commit_sha, embedding_model, chunking_version)
                    DO UPDATE SET index_state = 'indexing', chunk_count = 0,
                                  repo_json = EXCLUDED.repo_json, updated_at = NOW()
                    """,
                    (*key, json.dumps(repository.model_dump(exclude={"files"}))),
                )

    @_retry_on_connection_loss
    def finish_repository_index(self, repository: RepoSearchResult, embedding_model: str, chunking_version: str) -> None:
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                UPDATE repo_indexes SET index_state = 'completed', indexed_at = NOW(), updated_at = NOW()
                WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                """,
                (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
            )

    @_retry_on_connection_loss
    def append_chunks(
        self,
        repository: RepoSearchResult,
        chunks: list[Any],
        embeddings: list[list[float]],
        embedding_model: str,
        chunking_version: str,
    ) -> None:
        """Insert one batch of chunks and mark the repo searchable ('partial')."""

        self.ensure_ready()
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have a matching embedding vector.")
        if not chunks:
            return

        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE repo_indexes
                    SET index_state = 'partial', chunk_count = chunk_count + %s, updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (len(chunks), repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )
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
                            chunk_id, full_name, commit_sha, embedding_model, chunking_version,
                            path, chunk_role, language, symbol, heading, start_line, end_line,
                            token_count, repo_score, content_hash, text, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING
                        """,
                        rows,
                    )

    # ── Retrieval ────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_row(row: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {
            "chunk_id": row["chunk_id"],
            "repo_full_name": row["full_name"],
            "commit_sha": row["commit_sha"],
            "path": row["path"],
            "chunk_role": row["chunk_role"],
            "language": row["language"],
            "symbol": row["symbol"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "repo_score": float(row["repo_score"]),
            "text": row["text"],
            **extra,
        }

    @_retry_on_connection_loss
    def dense_search(
        self,
        query_embedding: list[float],
        allowed_repos: dict[str, str],
        top_k: int,
        embedding_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense retrieval in Postgres using pgvector cosine distance.

        HNSW normally scans ~ef_search candidates and *then* applies the repo
        filter, so a query scoped to 5 repos in a 60k-chunk corpus can come
        back nearly empty. pgvector 0.8's iterative scan keeps walking the
        graph until enough rows pass the filter.
        """

        self.ensure_ready()
        if not query_embedding or not allowed_repos:
            return []

        clause, params = _allowed_repo_clause(allowed_repos)
        query = sql.SQL(
            """
            SELECT {columns}, 1 - (embedding <=> %s) AS dense_score
            FROM corpus_chunks
            WHERE embedding_model = %s
              AND chunking_version = %s
              AND ({allowed_clause})
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        ).format(columns=_CHUNK_COLUMNS, allowed_clause=clause)

        vector = _as_vector(query_embedding)
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                conn.execute("SET LOCAL hnsw.ef_search = 100")
                rows = conn.execute(
                    query,
                    (
                        vector,
                        embedding_model or self.settings.EMBEDDING_MODEL,
                        self.settings.RAG_CHUNKING_VERSION,
                        *params,
                        vector,
                        # A wider pool than top_k so RetrievalService can re-rank by role.
                        max(top_k * 4, 20),
                    ),
                ).fetchall()

        return [self._chunk_row(row, dense_score=max(0.0, float(row["dense_score"] or 0.0))) for row in rows]

    @_retry_on_connection_loss
    def lexical_search(
        self,
        terms: list[str],
        allowed_repos: dict[str, str],
        limit: int,
        embedding_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword/identifier search on the GIN full-text index (any term matches)."""

        self.ensure_ready()
        if not terms or not allowed_repos:
            return []

        clause, params = _allowed_repo_clause(allowed_repos)
        query = sql.SQL(
            """
            SELECT {columns}, ts_rank_cd({document}, q) AS lexical_score
            FROM corpus_chunks, to_tsquery('simple', %s) AS q
            WHERE {document} @@ q
              AND embedding_model = %s
              AND chunking_version = %s
              AND ({allowed_clause})
            ORDER BY lexical_score DESC
            LIMIT %s
            """
        ).format(columns=_CHUNK_COLUMNS, document=_FTS_DOCUMENT, allowed_clause=clause)

        with self._pool.connection() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                query,
                (
                    " | ".join(terms),
                    embedding_model or self.settings.EMBEDDING_MODEL,
                    self.settings.RAG_CHUNKING_VERSION,
                    *params,
                    limit,
                ),
            ).fetchall()
        return [self._chunk_row(row, lexical_score=float(row["lexical_score"] or 0.0)) for row in rows]

    @_retry_on_connection_loss
    def neighbor_chunks(
        self, spans: list[tuple[str, str, str, int]], embedding_model: str
    ) -> list[dict[str, Any]]:
        """The chunk before and after each (repo, sha, path, start_line), in one round trip.

        Ordered by start_line rather than end_line because windowed chunks
        overlap by a few lines; the caller stitches overlaps by line number.
        """

        self.ensure_ready()
        if not spans:
            return []
        version = self.settings.RAG_CHUNKING_VERSION
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                sql.SQL(
                    """
                    SELECT n.* FROM unnest(%s::text[], %s::text[], %s::text[], %s::int[])
                        AS s(s_name, s_sha, s_path, s_start)
                    CROSS JOIN LATERAL (
                        (SELECT {columns} FROM corpus_chunks
                         WHERE full_name = s.s_name AND commit_sha = s.s_sha AND path = s.s_path
                           AND embedding_model = %s AND chunking_version = %s AND start_line < s.s_start
                         ORDER BY start_line DESC LIMIT 1)
                        UNION ALL
                        (SELECT {columns} FROM corpus_chunks
                         WHERE full_name = s.s_name AND commit_sha = s.s_sha AND path = s.s_path
                           AND embedding_model = %s AND chunking_version = %s AND start_line > s.s_start
                         ORDER BY start_line ASC LIMIT 1)
                    ) n
                    """
                ).format(columns=_CHUNK_COLUMNS),
                (
                    [s[0] for s in spans],
                    [s[1] for s in spans],
                    [s[2] for s in spans],
                    [s[3] for s in spans],
                    embedding_model,
                    version,
                    embedding_model,
                    version,
                ),
            ).fetchall()
        return [self._chunk_row(row) for row in rows]

    @_retry_on_connection_loss
    def chunks_for_paths(
        self, allowed_repos: dict[str, str], paths: list[str], embedding_model: str, per_path: int = 3
    ) -> list[dict[str, Any]]:
        """Opening chunks of specific files (README openings, files a question names)."""

        self.ensure_ready()
        if not paths or not allowed_repos:
            return []
        clause, params = _allowed_repo_clause(allowed_repos)
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                sql.SQL(
                    """
                    SELECT {columns} FROM (
                        SELECT *, row_number() OVER (PARTITION BY full_name, path ORDER BY start_line) AS rn
                        FROM corpus_chunks
                        WHERE path = ANY(%s) AND embedding_model = %s AND chunking_version = %s
                          AND ({allowed_clause})
                    ) ranked
                    WHERE rn <= %s
                    ORDER BY full_name, path, start_line
                    """
                ).format(columns=_CHUNK_COLUMNS, allowed_clause=clause),
                (paths, embedding_model, self.settings.RAG_CHUNKING_VERSION, *params, per_path),
            ).fetchall()
        return [self._chunk_row(row) for row in rows]

    @_retry_on_connection_loss
    def repo_paths(self, allowed_repos: dict[str, str], embedding_model: str) -> dict[str, list[str]]:
        """Every indexed file path per repo commit, for the chat repo map (one round trip)."""

        self.ensure_ready()
        if not allowed_repos:
            return {}
        clause, params = _allowed_repo_clause(allowed_repos)
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                sql.SQL(
                    """
                    SELECT full_name, array_agg(DISTINCT path ORDER BY path) AS paths
                    FROM corpus_chunks
                    WHERE embedding_model = %s AND chunking_version = %s AND ({allowed_clause})
                    GROUP BY full_name
                    """
                ).format(allowed_clause=clause),
                (embedding_model, self.settings.RAG_CHUNKING_VERSION, *params),
            ).fetchall()
        return {row["full_name"]: list(row["paths"]) for row in rows}

    @_retry_on_connection_loss
    def load_repositories(
        self, repos: list[tuple[str, str]], embedding_model: str, chunking_version: str
    ) -> list[RepoSearchResult]:
        """Batch form of load_repository: searchable repos, in request order."""

        self.ensure_ready()
        if not repos:
            return []
        names, shas = [r[0] for r in repos], [r[1] for r in repos]
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                """
                UPDATE repo_indexes i SET last_accessed_at = NOW()
                FROM unnest(%s::text[], %s::text[]) AS r(full_name, commit_sha)
                WHERE i.full_name = r.full_name AND i.commit_sha = r.commit_sha
                  AND i.embedding_model = %s AND i.chunking_version = %s
                  AND i.index_state IN ('partial', 'completed')
                RETURNING i.full_name, i.repo_json
                """,
                (names, shas, embedding_model, chunking_version),
            ).fetchall()
        by_name = {row["full_name"]: RepoSearchResult(**_load_json(row["repo_json"], {})) for row in rows}
        return [by_name[name] for name in names if name in by_name]


_store: RAGStore | None = None


def get_rag_store() -> RAGStore:
    """Return the singleton store instance."""

    global _store
    if _store is None:
        _store = RAGStore()
    return _store
