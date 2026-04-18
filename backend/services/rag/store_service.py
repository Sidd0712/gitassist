"""Shared Postgres + pgvector persistence for repo corpora and indexing jobs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from core.config import get_settings
from models.schemas import RepoFetchPlan, RepoSearchResult, ShallowRepoEvidence

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised in integration environments
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None

try:  # pragma: no cover - exercised in integration environments
    from pgvector.psycopg import register_vector
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - optional runtime dependency
    ConnectionPool = None
    dict_row = None
    register_vector = None
    sql = None


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


def _serialize_shallow_evidence(shallow: ShallowRepoEvidence) -> dict[str, Any]:
    return {
        "readme_path": shallow.readme_path,
        "readme_excerpt": shallow.readme[:4000],
        "manifest_paths": [repo_file.path for repo_file in shallow.manifest_files[:10]],
        "highlighted_paths": shallow.highlighted_paths[:10],
        "matched_keywords": shallow.matched_keywords[:8],
        "matched_frameworks": shallow.matched_frameworks[:8],
        "matched_capabilities": shallow.matched_capabilities[:8],
        "matched_stack_families": shallow.matched_stack_families[:8],
        "capability_coverage": shallow.capability_coverage,
        "score": shallow.score,
        "score_reasons": shallow.score_reasons[:6],
    }


def _allowed_repo_clause(allowed_repos: dict[str, str]) -> tuple[Any, list[Any]]:
    if sql is None:
        raise RuntimeError("psycopg is not installed")

    clauses = []
    params: list[Any] = []
    for full_name, commit_sha in allowed_repos.items():
        clauses.append(sql.SQL("(full_name = %s AND commit_sha = %s)"))
        params.extend([full_name, commit_sha])
    return sql.SQL(" OR ").join(clauses), params


@dataclass(slots=True)
class ClaimedIndexJob:
    """A claimed background indexing lease."""

    plan: RepoFetchPlan
    shallow_summary: dict[str, Any]
    retry_count: int = 0


class RAGStore:
    """Shared Postgres-backed store for repo corpora, chunks, and job leases."""

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
                f"""
                CREATE TABLE IF NOT EXISTS repo_indexes (
                    full_name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    index_state TEXT NOT NULL DEFAULT 'queued',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    repo_json JSONB NOT NULL,
                    shallow_evidence_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    selected_paths_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    skipped_paths_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    estimated_chars INTEGER NOT NULL DEFAULT 0,
                    rationale_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    indexed_at TIMESTAMPTZ,
                    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (full_name, commit_sha, embedding_model, chunking_version)
                )
                """
            )
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
                CREATE INDEX IF NOT EXISTS repo_index_jobs_status_idx
                ON repo_index_jobs (status, updated_at ASC)
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
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS corpus_chunks_fts_idx
                ON corpus_chunks
                USING GIN (
                    to_tsvector(
                        'simple',
                        coalesce(path, '') || ' ' || coalesce(symbol, '') || ' ' || coalesce(chunk_role, '') || ' ' || coalesce(text, '')
                    )
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_cache_entries (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload_json JSONB NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS pipeline_cache_entries_expiry_idx
                ON pipeline_cache_entries (expires_at ASC)
                """
            )

        self._ready = True

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._ready = False

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

    def get_cache_entry(self, namespace: str, cache_key: str) -> Any | None:
        """Load a non-expired cached pipeline artifact."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            row = conn.execute(
                """
                SELECT payload_json
                FROM pipeline_cache_entries
                WHERE namespace = %s
                  AND cache_key = %s
                  AND expires_at > NOW()
                """,
                (namespace, cache_key),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    DELETE FROM pipeline_cache_entries
                    WHERE namespace = %s
                      AND cache_key = %s
                      AND expires_at <= NOW()
                    """,
                    (namespace, cache_key),
                )
                return None
        return _load_json(row["payload_json"], None)

    def set_cache_entry(self, namespace: str, cache_key: str, payload: Any, ttl_seconds: int) -> None:
        """Persist a cached artifact with a TTL."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            conn.execute(
                """
                INSERT INTO pipeline_cache_entries (
                    namespace,
                    cache_key,
                    payload_json,
                    expires_at,
                    updated_at
                )
                VALUES (%s, %s, %s::jsonb, NOW() + (%s * INTERVAL '1 second'), NOW())
                ON CONFLICT (namespace, cache_key)
                DO UPDATE SET
                    payload_json = EXCLUDED.payload_json,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = NOW()
                """,
                (
                    namespace,
                    cache_key,
                    json.dumps(payload),
                    ttl_seconds,
                ),
            )

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

    def enqueue_index_job(
        self,
        plan: RepoFetchPlan,
        shallow: ShallowRepoEvidence,
        embedding_model: str,
        chunking_version: str,
    ) -> bool:
        """Persist plan metadata and enqueue a deduplicated background indexing job."""

        self.ensure_ready()
        repository = plan.repository
        shallow_summary = _serialize_shallow_evidence(shallow)
        repo_json = json.dumps(repository.model_dump())
        selected_paths_json = json.dumps(plan.selected_paths)
        skipped_paths_json = json.dumps(plan.skipped_paths)
        rationale_json = json.dumps(plan.rationale)
        shallow_json = json.dumps(shallow_summary)

        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO repo_indexes (
                        full_name,
                        commit_sha,
                        embedding_model,
                        chunking_version,
                        index_state,
                        repo_json,
                        shallow_evidence_json,
                        selected_paths_json,
                        skipped_paths_json,
                        estimated_chars,
                        rationale_json,
                        last_accessed_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, 'queued', %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb, NOW(), NOW())
                    ON CONFLICT (full_name, commit_sha, embedding_model, chunking_version)
                    DO UPDATE SET
                        repo_json = EXCLUDED.repo_json,
                        shallow_evidence_json = EXCLUDED.shallow_evidence_json,
                        selected_paths_json = EXCLUDED.selected_paths_json,
                        skipped_paths_json = EXCLUDED.skipped_paths_json,
                        estimated_chars = EXCLUDED.estimated_chars,
                        rationale_json = EXCLUDED.rationale_json,
                        index_state = CASE
                            WHEN repo_indexes.index_state = 'completed' THEN repo_indexes.index_state
                            ELSE 'queued'
                        END,
                        last_accessed_at = NOW(),
                        updated_at = NOW()
                    """,
                    (
                        repository.full_name,
                        repository.commit_sha,
                        embedding_model,
                        chunking_version,
                        repo_json,
                        shallow_json,
                        selected_paths_json,
                        skipped_paths_json,
                        plan.estimated_chars,
                        rationale_json,
                    ),
                )

                existing = conn.execute(
                    """
                    SELECT status, claimed_at
                    FROM repo_index_jobs
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    FOR UPDATE
                    """,
                    (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                ).fetchone()

                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO repo_index_jobs (
                            full_name, commit_sha, embedding_model, chunking_version, status, updated_at
                        ) VALUES (%s, %s, %s, %s, 'queued', NOW())
                        """,
                        (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                    )
                    return True

                claimed_at = existing.get("claimed_at")
                if existing["status"] == "completed":
                    return False
                if existing["status"] == "running" and claimed_at is not None:
                    conn.execute(
                        """
                        UPDATE repo_indexes
                        SET index_state = 'running', updated_at = NOW()
                        WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                        """,
                        (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                    )
                    return False

                conn.execute(
                    """
                    UPDATE repo_index_jobs
                    SET status = 'queued',
                        claimed_by = NULL,
                        claimed_at = NULL,
                        last_error = NULL,
                        updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )
                conn.execute(
                    """
                    UPDATE repo_indexes
                    SET index_state = 'queued', updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )
                return True

    def claim_index_job(self, worker_id: str, embedding_model: str, chunking_version: str) -> ClaimedIndexJob | None:
        """Claim the next available indexing job using row-level locking."""

        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                row = conn.execute(
                    """
                    SELECT
                        j.full_name,
                        j.commit_sha,
                        j.retry_count,
                        i.repo_json,
                        i.shallow_evidence_json,
                        i.selected_paths_json,
                        i.skipped_paths_json,
                        i.estimated_chars,
                        i.rationale_json
                    FROM repo_index_jobs j
                    JOIN repo_indexes i
                      ON i.full_name = j.full_name
                     AND i.commit_sha = j.commit_sha
                     AND i.embedding_model = j.embedding_model
                     AND i.chunking_version = j.chunking_version
                    WHERE j.embedding_model = %s
                      AND j.chunking_version = %s
                      AND j.retry_count < %s
                      AND (
                          j.status = 'queued'
                          OR j.status = 'failed'
                          OR (
                              j.status = 'running'
                              AND j.claimed_at < NOW() - (%s * INTERVAL '1 second')
                          )
                      )
                    ORDER BY
                        CASE j.status
                            WHEN 'queued' THEN 0
                            WHEN 'failed' THEN 1
                            ELSE 2
                        END,
                        j.updated_at ASC
                    FOR UPDATE OF j SKIP LOCKED
                    LIMIT 1
                    """,
                    (
                        embedding_model,
                        chunking_version,
                        self.settings.INDEX_JOB_MAX_RETRIES,
                        self.settings.INDEX_JOB_TIMEOUT_SECONDS,
                    ),
                ).fetchone()

                if row is None:
                    return None

                conn.execute(
                    """
                    UPDATE repo_index_jobs
                    SET status = 'running',
                        claimed_by = %s,
                        claimed_at = NOW(),
                        updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (worker_id, row["full_name"], row["commit_sha"], embedding_model, chunking_version),
                )
                conn.execute(
                    """
                    UPDATE repo_indexes
                    SET index_state = 'running', updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (row["full_name"], row["commit_sha"], embedding_model, chunking_version),
                )

        repository = RepoSearchResult(**_load_json(row["repo_json"], {}))
        plan = RepoFetchPlan(
            repository=repository,
            selected_paths=_load_json(row["selected_paths_json"], []),
            skipped_paths=_load_json(row["skipped_paths_json"], []),
            estimated_chars=int(row["estimated_chars"] or 0),
            rationale=_load_json(row["rationale_json"], []),
        )
        return ClaimedIndexJob(
            plan=plan,
            shallow_summary=_load_json(row["shallow_evidence_json"], {}),
            retry_count=int(row["retry_count"] or 0),
        )

    def mark_index_job_completed(
        self,
        job: ClaimedIndexJob,
        embedding_model: str,
        chunking_version: str,
        chunk_count: int,
    ) -> None:
        """Mark an indexing job as completed."""

        repository = job.plan.repository
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE repo_indexes
                    SET index_state = 'completed',
                        chunk_count = %s,
                        indexed_at = NOW(),
                        last_accessed_at = NOW(),
                        updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (chunk_count, repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )
                conn.execute(
                    """
                    UPDATE repo_index_jobs
                    SET status = 'completed',
                        claimed_by = NULL,
                        claimed_at = NULL,
                        last_error = NULL,
                        updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )

    def mark_index_job_failed(
        self,
        job: ClaimedIndexJob,
        embedding_model: str,
        chunking_version: str,
        error: str,
    ) -> None:
        """Record a failed indexing attempt while leaving the job retryable."""

        repository = job.plan.repository
        self.ensure_ready()
        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE repo_index_jobs
                    SET status = 'failed',
                        claimed_by = NULL,
                        claimed_at = NULL,
                        retry_count = retry_count + 1,
                        last_error = %s,
                        updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (error[:2000], repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )
                conn.execute(
                    """
                    UPDATE repo_indexes
                    SET index_state = 'failed', updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
                )

    def replace_repository_index(
        self,
        repository: RepoSearchResult,
        chunks: list[Any],
        embeddings: list[list[float]],
        embedding_model: str,
        chunking_version: str,
    ) -> None:
        """Replace all stored chunks for a repo commit in Postgres."""

        self.ensure_ready()
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have a matching embedding vector.")

        with self._pool.connection() as conn:  # type: ignore[union-attr]
            with conn.transaction():
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

                conn.execute(
                    """
                    UPDATE repo_indexes
                    SET repo_json = %s::jsonb,
                        chunk_count = %s,
                        updated_at = NOW()
                    WHERE full_name = %s AND commit_sha = %s AND embedding_model = %s AND chunking_version = %s
                    """,
                    (
                        json.dumps(repository.model_dump()),
                        len(chunks),
                        repository.full_name,
                        repository.commit_sha,
                        embedding_model,
                        chunking_version,
                    ),
                )

    def dense_search(
        self,
        query_embedding: list[float],
        allowed_repos: dict[str, str],
        top_k: int,
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
                    self.settings.EMBEDDING_MODEL,
                    self.settings.RAG_CHUNKING_VERSION,
                    *params,
                    _as_vector(query_embedding),
                    max(top_k * 2, top_k),
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

    def lexical_search(self, query_text: str, allowed_repos: dict[str, str], top_k: int) -> list[dict[str, Any]]:
        """Lexical retrieval directly in Postgres using full-text search."""

        self.ensure_ready()
        if not query_text.strip() or not allowed_repos:
            return []

        clause, params = _allowed_repo_clause(allowed_repos)
        query = sql.SQL(
            """
            WITH search_query AS (
                SELECT plainto_tsquery('simple', %s) AS q
            )
            SELECT
                c.chunk_id,
                c.full_name,
                c.commit_sha,
                c.path,
                c.chunk_role,
                c.language,
                c.start_line,
                c.end_line,
                c.repo_score,
                c.text,
                ts_rank_cd(
                    to_tsvector(
                        'simple',
                        coalesce(c.path, '') || ' ' || coalesce(c.symbol, '') || ' ' || coalesce(c.chunk_role, '') || ' ' || coalesce(c.text, '')
                    ),
                    search_query.q
                ) AS lexical_score
            FROM corpus_chunks c, search_query
            WHERE c.embedding_model = %s
              AND c.chunking_version = %s
              AND ({allowed_clause})
              AND search_query.q @@ to_tsvector(
                    'simple',
                    coalesce(c.path, '') || ' ' || coalesce(c.symbol, '') || ' ' || coalesce(c.chunk_role, '') || ' ' || coalesce(c.text, '')
                )
            ORDER BY lexical_score DESC
            LIMIT %s
            """
        ).format(allowed_clause=clause)

        with self._pool.connection() as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                query,
                (
                    query_text,
                    self.settings.EMBEDDING_MODEL,
                    self.settings.RAG_CHUNKING_VERSION,
                    *params,
                    max(top_k * 2, top_k),
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
                "lexical_score": max(0.0, float(row["lexical_score"] or 0.0)),
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
