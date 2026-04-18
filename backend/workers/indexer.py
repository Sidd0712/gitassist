"""Background worker that drains repo indexing jobs from Postgres."""

from __future__ import annotations

import asyncio
import logging
import socket
from contextlib import suppress

from core.config import get_settings
from services.rag.corpus_service import CorpusService
from services.rag.store_service import get_rag_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_worker() -> None:
    """Poll for jobs and process them with bounded concurrency."""

    settings = get_settings()
    store = get_rag_store()
    store.ensure_ready()

    worker_id = f"{socket.gethostname()}:{id(asyncio.current_task())}"
    corpus_service = CorpusService()
    active_tasks: set[asyncio.Task] = set()

    logger.info(
        "Indexer worker started with concurrency=%d and poll_interval=%ss",
        settings.INDEXER_MAX_CONCURRENCY,
        settings.INDEXER_POLL_INTERVAL_SECONDS,
    )

    try:
        while True:
            while len(active_tasks) < settings.INDEXER_MAX_CONCURRENCY:
                claimed = corpus_service.claim_index_job(worker_id)
                if claimed is None:
                    break

                task = asyncio.create_task(corpus_service.process_index_job(claimed))

                def _finalize(completed: asyncio.Task, claimed_job=claimed) -> None:
                    active_tasks.discard(completed)
                    with suppress(asyncio.CancelledError):
                        try:
                            completed.result()
                        except Exception as exc:  # pragma: no cover - runtime guard
                            corpus_service.store.mark_index_job_failed(
                                claimed_job,
                                corpus_service.embedding_model_name,
                                corpus_service.chunking_version,
                                str(exc),
                            )
                            logger.exception(
                                "Indexer worker failed for %s",
                                claimed_job.plan.repository.full_name,
                            )

                task.add_done_callback(_finalize)
                active_tasks.add(task)

            if active_tasks:
                await asyncio.wait(active_tasks, timeout=settings.INDEXER_POLL_INTERVAL_SECONDS, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(settings.INDEXER_POLL_INTERVAL_SECONDS)
    finally:
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*active_tasks, return_exceptions=True)
        store.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
