"""
Standalone GitHub Actions embedding worker.

Deliberately independent of the main backend package (no `core.config` or
`services.*` imports) — this runs on a fresh GitHub-hosted runner via
.github/workflows/embed-worker.yml, and keeping its dependency surface to
just psycopg + fastembed keeps the per-job install/cold-start cost low and
immune to unrelated app import failures.

Flow: read job_id from argv, fetch the pending row embedding_service.py
wrote via RAGStore.create_embedding_job, embed with the same model
(BAAI/bge-small-en-v1.5) and same query/document convention the backend
expects, write the result back, mark complete (or failed).
"""

from __future__ import annotations

import json
import os
import sys
import time

import psycopg


def _log(msg: str) -> None:
    print(msg, flush=True)

# Kept in sync with BGE_QUERY_INSTRUCTION in services/rag/embedding_service.py —
# duplicated rather than imported so this script has zero dependency on the
# rest of the backend package.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main() -> int:
    job_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("JOB_ID")
    if not job_id:
        print("ERROR: no job_id provided (arg1 or JOB_ID env var)", file=sys.stderr)
        return 1

    database_url = os.environ["DATABASE_URL"]
    conn = psycopg.connect(database_url, autocommit=True)

    try:
        t_select = time.monotonic()
        row = conn.execute(
            "SELECT is_query, input_texts FROM embedding_jobs WHERE job_id = %s AND status = 'pending'",
            (job_id,),
        ).fetchone()
        _log(f"TIMING select={time.monotonic() - t_select:.2f}s")
        if row is None:
            print(f"ERROR: job {job_id} not found or not pending", file=sys.stderr)
            return 1

        is_query, input_texts = row
        texts = json.loads(input_texts) if isinstance(input_texts, str) else input_texts
        _log(f"Job has {len(texts)} texts, total chars={sum(len(t) for t in texts)}")

        try:
            t0 = time.monotonic()
            from fastembed import TextEmbedding

            model = TextEmbedding(model_name=MODEL_NAME)
            _log(f"TIMING model_load={time.monotonic() - t0:.2f}s")

            cleaned = [t.strip() if t and t.strip() else " " for t in texts]
            if is_query:
                cleaned = [BGE_QUERY_INSTRUCTION + t for t in cleaned]

            t1 = time.monotonic()
            vectors = [v.tolist() for v in model.embed(cleaned, batch_size=96)]
            _log(f"TIMING embed={time.monotonic() - t1:.2f}s for {len(vectors)} vectors")

            t2 = time.monotonic()
            payload = json.dumps(vectors)
            _log(f"TIMING json_dumps={time.monotonic() - t2:.2f}s, payload_bytes={len(payload)}")

            t3 = time.monotonic()
            conn.execute(
                """
                UPDATE embedding_jobs
                SET status = 'complete', result = %s::jsonb, completed_at = NOW()
                WHERE job_id = %s
                """,
                (payload, job_id),
            )
            _log(f"TIMING db_update={time.monotonic() - t3:.2f}s")
            print(f"OK: embedded {len(vectors)} vectors for job {job_id}")
            return 0

        except Exception as exc:  # noqa: BLE001 - report any failure back to the caller
            conn.execute(
                """
                UPDATE embedding_jobs
                SET status = 'failed', error = %s, completed_at = NOW()
                WHERE job_id = %s
                """,
                (str(exc), job_id),
            )
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
