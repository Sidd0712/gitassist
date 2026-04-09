"""Compatibility ASGI entrypoint for `uvicorn app:main --reload`."""

from main import app

# Uvicorn expects `module:attribute`, and the current user command is `app:main`.
# Expose the FastAPI app under both names so either `app:app` or `app:main` works.
main = app
