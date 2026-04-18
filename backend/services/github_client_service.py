"""Shared GitHub HTTP client with connection pooling."""

from __future__ import annotations

import logging

import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)

_github_client: httpx.AsyncClient | None = None


def _default_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "GitAssist-AI",
    }
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
    return headers


def get_github_client() -> httpx.AsyncClient:
    """Return a shared AsyncClient for GitHub API requests."""

    global _github_client
    if _github_client is not None:
        return _github_client

    settings = get_settings()
    limits = httpx.Limits(
        max_connections=settings.GITHUB_MAX_CONNECTIONS,
        max_keepalive_connections=settings.GITHUB_MAX_KEEPALIVE_CONNECTIONS,
    )
    _github_client = httpx.AsyncClient(
        timeout=settings.GITHUB_HTTP_TIMEOUT_SECONDS,
        limits=limits,
        headers=_default_headers(),
        follow_redirects=True,
    )
    logger.info(
        "Initialized shared GitHub client with %d max connections",
        settings.GITHUB_MAX_CONNECTIONS,
    )
    return _github_client


async def close_github_client() -> None:
    """Close the shared GitHub client during application shutdown."""

    global _github_client
    if _github_client is None:
        return
    await _github_client.aclose()
    _github_client = None
