"""Persistence layer for repo corpora, metadata, and retrieval indexes."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import get_settings
from models.schemas import CorpusChunk, RepoFile, RepoSearchResult

logger = logging.getLogger(__name__)

try:
    import chromadb
except ImportError:  # pragma: no cover - optional runtime dependency
    chromadb = None


COLLECTION_NAME = "repo_chunks"


def sanitize_repo_name(full_name: str) -> str:
    """Convert owner/repo to a filesystem-safe folder name."""

    return full_name.replace("/", "__")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors."""

    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class RAGStore:
    """Storage service for repo corpora and retrieval indexes."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.corpus_path = Path(self.settings.RAG_CORPUS_PATH)
        self.vector_store_path = Path(self.settings.RAG_VECTOR_STORE_PATH)
        self.sqlite_path = Path(self.settings.RAG_SQLITE_PATH)
        self._fts_enabled = True
        self._collection = None

    def ensure_ready(self) -> None:
        """Create all backing directories and tables."""

        self.corpus_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_indexes (
                    full_name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    repo_json TEXT NOT NULL,
                    PRIMARY KEY (full_name, commit_sha, embedding_model, chunking_version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    path TEXT NOT NULL,
                    chunk_role TEXT NOT NULL,
                    language TEXT,
                    symbol TEXT,
                    heading TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    token_count INTEGER NOT NULL,
                    repo_score REAL NOT NULL,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
                    USING fts5(
                        chunk_id UNINDEXED,
                        full_name,
                        path,
                        symbol,
                        chunk_role,
                        text
                    )
                    """
                )
            except sqlite3.OperationalError:
                self._fts_enabled = False
                logger.warning("SQLite FTS5 is unavailable; lexical retrieval will fall back to LIKE queries.")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_collection(self):
        if chromadb is None:
            return None
        if self._collection is None:
            client = chromadb.PersistentClient(path=str(self.vector_store_path))
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def is_indexed(self, repository: RepoSearchResult, embedding_model: str, chunking_version: str) -> bool:
        """Check whether a repo/commit is already indexed for the current embedding settings."""

        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM repo_indexes
                WHERE full_name = ? AND commit_sha = ? AND embedding_model = ? AND chunking_version = ?
                """,
                (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
            ).fetchone()
        return row is not None

    def save_repository_artifacts(
        self,
        repository: RepoSearchResult,
        files: list[RepoFile],
        chunks: list[CorpusChunk],
    ) -> None:
        """Persist raw repo data so index state is inspectable on disk."""

        repo_dir = self.corpus_path / sanitize_repo_name(repository.full_name) / repository.commit_sha
        repo_dir.mkdir(parents=True, exist_ok=True)

        repo_payload = repository.model_dump()
        repo_payload["files"] = []
        (repo_dir / "repo.json").write_text(json.dumps(repo_payload, indent=2), encoding="utf-8")
        with (repo_dir / "files.jsonl").open("w", encoding="utf-8") as handle:
            for repo_file in files:
                handle.write(json.dumps(repo_file.model_dump()) + "\n")
        with (repo_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.model_dump()) + "\n")

    def replace_repository_index(
        self,
        repository: RepoSearchResult,
        chunks: list[CorpusChunk],
        embeddings: list[list[float]],
        embedding_model: str,
        chunking_version: str,
    ) -> None:
        """Replace all stored chunks/embeddings for a repo commit."""

        self.ensure_ready()

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have a matching embedding vector.")

        repo_payload = repository.model_dump()
        repo_payload["files"] = []
        indexed_at = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM repo_indexes
                WHERE full_name = ? AND commit_sha = ? AND embedding_model = ? AND chunking_version = ?
                """,
                (repository.full_name, repository.commit_sha, embedding_model, chunking_version),
            )
            conn.execute(
                "DELETE FROM embeddings WHERE full_name = ? AND commit_sha = ?",
                (repository.full_name, repository.commit_sha),
            )
            conn.execute(
                "DELETE FROM chunks WHERE full_name = ? AND commit_sha = ?",
                (repository.full_name, repository.commit_sha),
            )
            if self._fts_enabled:
                conn.execute(
                    "DELETE FROM chunk_fts WHERE full_name = ?",
                    (repository.full_name,),
                )

            conn.execute(
                """
                INSERT INTO repo_indexes (full_name, commit_sha, embedding_model, chunking_version, indexed_at, repo_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    repository.full_name,
                    repository.commit_sha,
                    embedding_model,
                    chunking_version,
                    indexed_at,
                    json.dumps(repo_payload),
                ),
            )

            for chunk, embedding in zip(chunks, embeddings):
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, full_name, commit_sha, path, chunk_role, language, symbol, heading,
                        start_line, end_line, token_count, repo_score, content_hash, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.repo_full_name,
                        chunk.commit_sha,
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
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO embeddings (chunk_id, full_name, commit_sha, embedding_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.repo_full_name,
                        chunk.commit_sha,
                        json.dumps(embedding),
                    ),
                )
                if self._fts_enabled:
                    conn.execute(
                        """
                        INSERT INTO chunk_fts (chunk_id, full_name, path, symbol, chunk_role, text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.repo_full_name,
                            chunk.path,
                            chunk.symbol or "",
                            chunk.chunk_role,
                            chunk.text,
                        ),
                    )
            conn.commit()

        collection = self._get_collection()
        if collection is not None and chunks:
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            try:
                collection.delete(ids=chunk_ids)
            except Exception:  # pragma: no cover - Chroma is best effort
                pass

            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {
                        "full_name": chunk.repo_full_name,
                        "commit_sha": chunk.commit_sha,
                        "path": chunk.path,
                        "chunk_role": chunk.chunk_role,
                        "language": chunk.language or "",
                        "symbol": chunk.symbol or "",
                        "start_line": chunk.start_line or 0,
                        "end_line": chunk.end_line or 0,
                        "repo_score": chunk.repo_score,
                    }
                    for chunk in chunks
                ],
            )

    def load_repository(self, full_name: str, commit_sha: str, embedding_model: str, chunking_version: str) -> RepoSearchResult | None:
        """Load stored repo metadata for a specific index."""

        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT repo_json
                FROM repo_indexes
                WHERE full_name = ? AND commit_sha = ? AND embedding_model = ? AND chunking_version = ?
                """,
                (full_name, commit_sha, embedding_model, chunking_version),
            ).fetchone()
        if row is None:
            return None
        return RepoSearchResult(**json.loads(row["repo_json"]))

    def dense_search(
        self,
        query_embedding: list[float],
        allowed_repos: dict[str, str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Dense retrieval over stored embeddings."""

        self.ensure_ready()
        if not query_embedding or not allowed_repos:
            return []

        allowed_names = list(allowed_repos.keys())

        collection = self._get_collection()
        if collection is not None:
            try:
                response = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(top_k * 4, top_k),
                    include=["metadatas", "documents", "distances"],
                )
                docs = response.get("documents", [[]])[0]
                metas = response.get("metadatas", [[]])[0]
                ids = response.get("ids", [[]])[0]
                distances = response.get("distances", [[]])[0]
                results: list[dict[str, Any]] = []
                for chunk_id, document, meta, distance in zip(ids, docs, metas, distances):
                    if meta["full_name"] not in allowed_repos:
                        continue
                    if meta.get("commit_sha") != allowed_repos[meta["full_name"]]:
                        continue
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "repo_full_name": meta["full_name"],
                            "path": meta["path"],
                            "chunk_role": meta["chunk_role"],
                            "language": meta["language"] or None,
                            "start_line": meta["start_line"] or None,
                            "end_line": meta["end_line"] or None,
                            "repo_score": float(meta.get("repo_score", 0.0)),
                            "text": document,
                            "dense_score": max(0.0, 1.0 - float(distance)),
                        }
                    )
                    if len(results) >= top_k * 2:
                        break
                return results
            except Exception as exc:  # pragma: no cover - Chroma is optional
                logger.warning("Dense search via Chroma failed, falling back to SQLite vectors: %s", exc)

        placeholders = ",".join("?" for _ in allowed_names)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.chunk_id, c.full_name, c.commit_sha, c.path, c.chunk_role, c.language, c.start_line, c.end_line,
                    c.repo_score, c.text, e.embedding_json
                FROM embeddings e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                WHERE c.full_name IN ({placeholders})
                """,
                tuple(allowed_names),
            ).fetchall()

        scored: list[dict[str, Any]] = []
        for row in rows:
            if row["full_name"] not in allowed_repos:
                continue
            if row["commit_sha"] != allowed_repos[row["full_name"]]:
                continue
            dense_score = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
            scored.append(
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
                    "dense_score": dense_score,
                }
            )
        scored.sort(key=lambda item: item["dense_score"], reverse=True)
        return scored[: max(top_k * 2, top_k)]

    def lexical_search(self, query: str, allowed_repos: dict[str, str], top_k: int) -> list[dict[str, Any]]:
        """Lexical retrieval over chunk metadata and text."""

        self.ensure_ready()
        if not allowed_repos:
            return []
        allowed_names = list(allowed_repos.keys())
        tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(token) > 2]
        if not tokens:
            return []

        placeholders = ",".join("?" for _ in allowed_names)
        if self._fts_enabled:
            fts_query = " OR ".join(f'"{token}"' for token in tokens[:12])
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT
                        c.chunk_id, c.full_name, c.commit_sha, c.path, c.chunk_role, c.language, c.start_line, c.end_line,
                        c.repo_score, c.text, bm25(chunk_fts) AS rank
                    FROM chunk_fts
                    JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
                    WHERE chunk_fts MATCH ?
                      AND c.full_name IN ({placeholders})
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, *allowed_names, max(top_k * 3, top_k)),
                ).fetchall()
        else:
            like_query = f"%{'%'.join(tokens[:4])}%"
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT
                        chunk_id, full_name, commit_sha, path, chunk_role, language, start_line, end_line,
                        repo_score, text, 1.0 AS rank
                    FROM chunks
                    WHERE full_name IN ({placeholders})
                      AND (path LIKE ? OR text LIKE ?)
                    LIMIT ?
                    """,
                    (*allowed_names, like_query, like_query, max(top_k * 3, top_k)),
                ).fetchall()

        results: list[dict[str, Any]] = []
        max_rank = 1.0
        if rows and self._fts_enabled:
            max_rank = abs(float(rows[-1]["rank"])) or 1.0

        for row in rows:
            if row["full_name"] not in allowed_repos:
                continue
            if row["commit_sha"] != allowed_repos[row["full_name"]]:
                continue
            rank = abs(float(row["rank"])) if self._fts_enabled else 1.0
            lexical_score = 1.0 if not self._fts_enabled else max(0.0, 1.0 - (rank / (max_rank + 1e-6)))
            results.append(
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
                    "lexical_score": lexical_score,
                }
            )
        return results[: max(top_k * 2, top_k)]


_store: RAGStore | None = None


def get_rag_store() -> RAGStore:
    """Return the singleton store instance."""

    global _store
    if _store is None:
        _store = RAGStore()
    return _store
