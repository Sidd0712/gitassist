"""Embedding + indexing worker that runs on the developer's machine.

The backend (local or on Render) never indexes repos itself. It queues work
in the shared Postgres database and this process does it:

- query lane:    embeds chat/search queries (latency-critical, own model)
- document lane: embeds short repo-metadata batches for candidate search
- repo lane:     claims repo_index_jobs and indexes whole repos — download,
                 chunk, embed in-process, write chunks progressively

While it runs it heartbeats, so the backend knows embeddings are available;
when it stops, the backend degrades instead of waiting (reports still work,
chat pauses, index jobs simply wait in the queue).

Run from backend/:
    pip install -r requirements-worker.txt
    python scripts/local_embed_worker.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastembed import TextEmbedding  # noqa: E402

from core.config import get_settings  # noqa: E402
from models.schemas import RepoSearchResult  # noqa: E402
from services.rag.embedding_service import BGE_QUERY_INSTRUCTION  # noqa: E402
from services.rag.store_service import get_rag_store  # noqa: E402

logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("embed-worker")

_IDLE_POLL_SECONDS = 0.5
_REPO_IDLE_POLL_SECONDS = 3.0
_HEARTBEAT_SECONDS = 5.0
# Measured on an i7-1355U: small batches sorted by length run 3x faster than
# unsorted batches of 64 (less padding), and 64 x 512 tokens needs an 800 MB
# attention buffer.
_EMBED_BATCH = 8


def embed_sorted(model: TextEmbedding, texts: list[str]) -> list[list[float]]:
    """Embed texts in length order to minimise padding, returned in input order."""

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = list(model.embed([texts[i] or " " for i in order], batch_size=_EMBED_BATCH))
    result: list[list[float]] = [[] for _ in texts]
    for position, index in enumerate(order):
        result[index] = vectors[position].tolist()
    return result


def _run_embedding_lane(is_query: bool, model: TextEmbedding, stop: threading.Event) -> None:
    lane = "query" if is_query else "document"
    store = get_rag_store()
    logger.info("%s lane ready", lane)

    while not stop.is_set():
        try:
            job = store.claim_embedding_job(is_query)
        except Exception:
            logger.exception("%s lane: failed to poll for jobs", lane)
            stop.wait(5.0)
            continue
        if job is None:
            stop.wait(_IDLE_POLL_SECONDS)
            continue

        texts = [t.strip() if t and t.strip() else " " for t in job["texts"]]
        if is_query:
            texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
        started = time.monotonic()
        try:
            vectors = embed_sorted(model, texts)
            store.complete_embedding_job(job["job_id"], vectors)
            logger.info("%s lane: embedded %d texts in %.1fs", lane, len(vectors), time.monotonic() - started)
        except Exception as exc:
            logger.exception("%s lane: job %s failed", lane, job["job_id"])
            try:
                store.fail_embedding_job(job["job_id"], str(exc))
            except Exception:
                logger.exception("%s lane: could not mark job %s failed", lane, job["job_id"])


def _run_repo_lane(worker_id: str, model: TextEmbedding, stop: threading.Event) -> None:
    from services.rag.corpus_service import CorpusService

    store = get_rag_store()
    corpus = CorpusService()
    # One loop for the lane's lifetime: the shared GitHub httpx client is bound
    # to the loop it was first used on.
    loop = asyncio.new_event_loop()
    logger.info("repo lane ready")

    while not stop.is_set():
        try:
            job = store.claim_index_job(worker_id, corpus.embedding_model_name, corpus.chunking_version)
        except Exception:
            logger.exception("repo lane: failed to poll for index jobs")
            stop.wait(10.0)
            continue
        if job is None:
            stop.wait(_REPO_IDLE_POLL_SECONDS)
            continue

        repository = RepoSearchResult(**{**job["repository"], "commit_sha": job["commit_sha"]})
        started = time.monotonic()
        logger.info("repo lane: indexing %s@%s", repository.full_name, repository.commit_sha[:8])
        try:
            count = loop.run_until_complete(
                corpus.index_repository(repository, job["path_terms"], lambda texts: embed_sorted(model, texts))
            )
            store.finish_index_job(job["full_name"], job["commit_sha"], corpus.embedding_model_name, corpus.chunking_version)
            logger.info(
                "repo lane: %s done, %d chunks in %.0fs", repository.full_name, count, time.monotonic() - started
            )
        except Exception as exc:
            logger.exception("repo lane: indexing %s failed", repository.full_name)
            try:
                store.finish_index_job(
                    job["full_name"], job["commit_sha"], corpus.embedding_model_name, corpus.chunking_version,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                logger.exception("repo lane: could not record failure for %s", repository.full_name)


def main() -> int:
    settings = get_settings()
    store = get_rag_store()
    store.ensure_ready()
    worker_id = socket.gethostname()
    stop = threading.Event()
    released = store.release_index_jobs(worker_id)
    if released:
        logger.info("Re-queued %d index job(s) interrupted by the last shutdown", released)

    # Separate model instances per lane. Queries are one or two texts; the
    # document lane sits on the research request's critical path (candidate
    # search embeds ~60 repo descriptions), so it gets half the cores; the
    # repo lane gets the rest of the machine.
    cpus = os.cpu_count() or 4
    query_model = TextEmbedding(model_name=settings.LOCAL_EMBEDDING_MODEL, threads=2)
    document_model = TextEmbedding(model_name=settings.LOCAL_EMBEDDING_MODEL, threads=max(2, cpus // 2))
    repo_model = TextEmbedding(model_name=settings.LOCAL_EMBEDDING_MODEL, threads=max(1, cpus - 2))

    lanes = [
        threading.Thread(target=_run_embedding_lane, args=(True, query_model, stop), daemon=True),
        threading.Thread(target=_run_embedding_lane, args=(False, document_model, stop), daemon=True),
        threading.Thread(target=_run_repo_lane, args=(worker_id, repo_model, stop), daemon=True),
    ]
    for lane in lanes:
        lane.start()

    logger.info("Worker %s started (%s); Ctrl+C to stop", worker_id, settings.LOCAL_EMBEDDING_MODEL)
    try:
        while True:
            try:
                store.record_embedding_worker_heartbeat(worker_id)
            except Exception:
                logger.exception("Heartbeat failed")
            time.sleep(_HEARTBEAT_SECONDS)
    except KeyboardInterrupt:
        logger.info("Stopping worker")
        stop.set()
        for lane in lanes:
            lane.join(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
