"""GitAssist AI FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from routers.research import router as research_router
from services.rag.store_service import get_rag_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = get_rag_store()
    store.ensure_ready()
    stats = store.get_stats()
    logger.info("%s starting up", settings.APP_NAME)
    logger.info("  LLM model: %s", settings.LLM_MODEL)
    logger.info("  Embedding model: %s", settings.EMBEDDING_MODEL)
    logger.info(
        "  RAG store: %s (%d indexed repos, %d chunks)",
        store.backend_name,
        stats["completed_repos"],
        stats["chunks"],
    )
    logger.info("  Candidate repo limit: %d", settings.RAG_CANDIDATE_REPO_LIMIT)
    logger.info("  Deep index repo limit: %d", settings.RAG_DEEP_INDEX_REPO_LIMIT)
    yield
    store.close()
    logger.info("%s shutting down", settings.APP_NAME)


app = FastAPI(
    title="GitAssist AI",
    description="AI-powered GitHub repository researcher and learning path generator",
    version="0.2.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research_router)
