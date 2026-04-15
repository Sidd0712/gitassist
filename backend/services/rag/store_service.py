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
    import faiss
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    faiss = None
    np = None


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
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = None
        self.faiss_index_path = self.vector_store_path / "faiss.index"
        self.faiss_metadata_path = self.vector_store_path / "faiss_metadata.json"
        self.chunks_metadata = []

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
        
        # Initialize or load FAISS index
        self._load_or_create_faiss_index()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_or_create_faiss_index(self) -> None:
        """Load existing FAISS index or create new one."""
        if faiss is None:
            logger.warning("FAISS not available, vector search will be disabled")
            return
        
        if self.faiss_index_path.exists() and self.faiss_metadata_path.exists():
            try:
                # Load existing index
                self.index = faiss.read_index(str(self.faiss_index_path))
                with open(self.faiss_metadata_path, 'r', encoding='utf-8') as f:
                    self.chunks_metadata = json.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as exc:
                logger.error(f"Failed to load FAISS index: {exc}, creating new index")
                self._create_new_faiss_index()
        else:
            self._create_new_faiss_index()
    
    def _create_new_faiss_index(self) -> None:
        """Create a new FAISS index."""
        if faiss is None:
            return
        # IndexFlatIP for cosine similarity (after L2 normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []
        logger.info("Created new FAISS index")
    
    def _save_faiss_index(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if self.index is None or faiss is None:
            return
        
        try:
            faiss.write_index(self.index, str(self.faiss_index_path))
            with open(self.faiss_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks_metadata, f)
            logger.debug(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as exc:
            logger.error(f"Failed to save FAISS index: {exc}")

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

        # Add to FAISS index
        if self.index is not None and chunks and faiss is not None and np is not None:
            # Convert embeddings to numpy array and normalize for cosine similarity
            embeddings_np = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_np)
            
            # Remove old chunks from metadata for this repo
            self.chunks_metadata = [
                meta for meta in self.chunks_metadata
                if not (meta["repo_full_name"] == repository.full_name and meta["commit_sha"] == repository.commit_sha)
            ]
            
            # Rebuild index with remaining + new chunks
            # (FAISS doesn't support deletion, so we rebuild)
            if len(self.chunks_metadata) > 0:
                # Get embeddings for existing chunks from SQLite
                existing_embeddings = []
                with self._connect() as conn:
                    for meta in self.chunks_metadata:
                        row = conn.execute(
                            "SELECT embedding_json FROM embeddings WHERE chunk_id = ?",
                            (meta["chunk_id"],)
                        ).fetchone()
                        if row:
                            existing_embeddings.append(json.loads(row["embedding_json"]))
                
                if existing_embeddings:
                    existing_np = np.array(existing_embeddings, dtype=np.float32)
                    faiss.normalize_L2(existing_np)
                    all_embeddings = np.vstack([existing_np, embeddings_np])
                else:
                    all_embeddings = embeddings_np
            else:
                all_embeddings = embeddings_np
            
            # Recreate index with all embeddings
            self.index = faiss.IndexFlatIP(self.dimension)
            if all_embeddings.shape[0] > 0:
                self.index.add(all_embeddings)
            
            # Add new chunks to metadata
            start_idx = len(self.chunks_metadata)
            for i, chunk in enumerate(chunks):
                self.chunks_metadata.append({
                    "idx": start_idx + i,
                    "chunk_id": chunk.chunk_id,
                    "repo_full_name": chunk.repo_full_name,
                    "commit_sha": chunk.commit_sha,
                    "path": chunk.path,
                    "chunk_role": chunk.chunk_role,
                    "language": chunk.language or "",
                    "symbol": chunk.symbol or "",
                    "start_line": chunk.start_line or 0,
                    "end_line": chunk.end_line or 0,
                    "repo_score": chunk.repo_score,
                })
            
            # Persist FAISS index
            self._save_faiss_index()

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
        """Dense retrieval over stored embeddings using FAISS."""

        self.ensure_ready()
        if not query_embedding or not allowed_repos:
            return []

        allowed_names = list(allowed_repos.keys())

        # Use FAISS for fast vector search
        if self.index is not None and faiss is not None and np is not None and self.index.ntotal > 0:
            try:
                # Normalize query embedding for cosine similarity
                query_np = np.array([query_embedding], dtype=np.float32)
                faiss.normalize_L2(query_np)
                
                # Search FAISS (get more candidates for filtering)
                search_k = min(max(top_k * 4, top_k), self.index.ntotal)
                scores, indices = self.index.search(query_np, search_k)
                
                # Filter and build results
                results: list[dict[str, Any]] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.chunks_metadata):
                        continue
                    
                    metadata = self.chunks_metadata[idx]
                    
                    # Apply repo and commit filters
                    if metadata["repo_full_name"] not in allowed_repos:
                        continue
                    if metadata["commit_sha"] != allowed_repos[metadata["repo_full_name"]]:
                        continue
                    
                    # Get chunk text from SQLite
                    with self._connect() as conn:
                        row = conn.execute(
                            "SELECT text FROM chunks WHERE chunk_id = ?",
                            (metadata["chunk_id"],)
                        ).fetchone()
                        if not row:
                            continue
                        text = row["text"]
                    
                    results.append({
                        "chunk_id": metadata["chunk_id"],
                        "repo_full_name": metadata["repo_full_name"],
                        "path": metadata["path"],
                        "chunk_role": metadata["chunk_role"],
                        "language": metadata["language"] or None,
                        "start_line": metadata["start_line"] or None,
                        "end_line": metadata["end_line"] or None,
                        "repo_score": float(metadata["repo_score"]),
                        "text": text,
                        "dense_score": max(0.0, float(score)),  # Score already normalized [0,1]
                    })
                    
                    if len(results) >= top_k * 2:
                        break
                
                return results
            except Exception as exc:
                logger.warning(f"Dense search via FAISS failed, falling back to SQLite: {exc}")

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
