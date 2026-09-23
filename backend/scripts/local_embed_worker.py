"""Embedding worker that runs on the developer's machine.

The backend (local or on Render) queues texts in the shared Postgres
`embedding_jobs` table; this process claims them, embeds with fastembed, and
writes the vectors back. While it runs, it heartbeats so the backend knows
embeddings are available; when it stops, the backend degrades instead of
waiting (reports still work, indexing and chat pause).

Two lanes, each with its own model instance, so a one-line chat or search
query never waits behind a repo's multi-minute indexing job.

Run from backend/:
    pip install -r requirements-worker.txt
    python scripts/local_embed_worker.py
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastembed import TextEmbedding  # noqa: E402

from core.config import get_settings  # noqa: E402
from services.rag.embedding_service import BGE_QUERY_INSTRUCTION  # noqa: E402
from services.rag.store_service import get_rag_store  # noqa: E402

logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger("embed-worker")

_IDLE_POLL_SECONDS = 0.5
_HEARTBEAT_SECONDS = 5.0


def _run_lane(is_query: bool, model_name: str, stop: threading.Event) -> None:
    lane = "query" if is_query else "document"
    store = get_rag_store()
    model = TextEmbedding(model_name=model_name)
    logger.info("%s lane ready (%s)", lane, model_name)

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
            vectors = [v.tolist() for v in model.embed(texts, batch_size=64)]
            store.complete_embedding_job(job["job_id"], vectors)
            elapsed = time.monotonic() - started
            logger.info("%s lane: embedded %d texts in %.1fs", lane, len(vectors), elapsed)
        except Exception as exc:
            logger.exception("%s lane: job %s failed", lane, job["job_id"])
            try:
                store.fail_embedding_job(job["job_id"], str(exc))
            except Exception:
                logger.exception("%s lane: could not mark job %s failed", lane, job["job_id"])


def main() -> int:
    settings = get_settings()
    store = get_rag_store()
    store.ensure_ready()
    worker_id = socket.gethostname()
    stop = threading.Event()

    lanes = [
        threading.Thread(target=_run_lane, args=(True, settings.LOCAL_EMBEDDING_MODEL, stop), daemon=True),
        threading.Thread(target=_run_lane, args=(False, settings.LOCAL_EMBEDDING_MODEL, stop), daemon=True),
    ]
    for lane in lanes:
        lane.start()

    logger.info("Worker %s started; Ctrl+C to stop", worker_id)
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
