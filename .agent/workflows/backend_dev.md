---
description: Backend Development Workflow
---
1. **Framework:** Use FastAPI for all backend development.
2. **Async Everything:** Use `async def` for endpoint definitions, and use `httpx` instead of `requests` to avoid blocking the event loop when querying external APIs like GitHub and the LLM.
3. **GitHub API Handling:** 
   - Always implement caching for GitHub responses to avoid rate limits.
   - Add clear logging when interacting with the GitHub REST API.
   - Fall back gracefully if a repo is too large to scrape entirely (e.g., focus only on `README.md` and `docker-compose.yml` or `package.json`).
4. **Code Quality:** Use Pydantic models extensively for input validation and output schema definitions. Maintain modular microservice architecture (e.g. `routers/`, `services/`, `models/`).
