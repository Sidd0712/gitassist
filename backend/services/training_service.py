"""Offline evaluation fixture helpers for the RAG pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from models.schemas import AnalysisResponse, RepoSearchResult

logger = logging.getLogger(__name__)

EVAL_FIXTURES_DIR = Path(__file__).parent.parent / "data" / "eval_fixtures"


def compile_training_dataset(repos: list[RepoSearchResult], idea: str) -> dict:
    """Legacy compatibility wrapper retained for offline analysis only."""

    return compile_retrieval_fixture(repos=repos, idea=idea, analysis=None)


def compile_retrieval_fixture(
    repos: list[RepoSearchResult],
    idea: str,
    analysis: AnalysisResponse | None,
) -> dict:
    """Persist a lightweight evaluation artifact outside the request path."""

    EVAL_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fixture_path = EVAL_FIXTURES_DIR / f"fixture_{timestamp}.json"

    payload = {
        "idea": idea,
        "repos": [repo.model_dump() for repo in repos],
        "analysis": analysis.model_dump() if analysis else None,
        "timestamp": timestamp,
    }
    fixture_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("Wrote RAG evaluation fixture for %d repos to %s", len(repos), fixture_path)
    return {
        "fixture_path": str(fixture_path),
        "repo_count": len(repos),
        "timestamp": timestamp,
    }
