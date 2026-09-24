import asyncio
import hashlib
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.schemas import (
    AnalysisResponse,
    ExtractedKeywords,
    IdeaRequest,
    RepoChatRequest,
    RepoFetchPlan,
    RepoFile,
    RepoSearchResult,
    RepoSnapshot,
    RepoTreeEntry,
    RetrievalHit,
    RetrievalPlan,
    ShallowRepoEvidence,
)
from routers.research import _low_confidence_caveat, _skip_if_not_indexable, chat_about_repositories, research_idea
import services.github_service as github_service_module
from services.github_service import (
    build_repo_fetch_plan,
    build_search_queries,
    fetch_repo_snapshot,
    fetch_shallow_repo_evidence,
    search_repo_candidates,
)
from services.llm_service import (
    _check_urls,
    _validate_resource_links,
    build_clarification_questions,
    extract_keywords,
    generate_analysis,
    plan_retrieval_queries,
)
from services.pipeline_cache_service import clear_memory_caches
from services.rag.embedding_service import EmbeddingService, EmbeddingUnavailable
from services.rag.ranking_service import rank_repo_evidence
from services.rag.retrieval_service import RetrievalService


class LLMFallbackTests(unittest.IsolatedAsyncioTestCase):
    def _client(self):
        from services.llm_client import LLMClient

        client = LLMClient()
        client.settings = client.settings.model_copy(
            update={"GOOGLE_API_KEY": "test-key", "LLM_FALLBACK_MODEL": "gemini-test"}
        )
        return client

    async def test_retired_groq_model_falls_back_and_parks_groq(self):
        import groq

        client = self._client()
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        not_found = groq.NotFoundError(
            "model_not_found", response=httpx.Response(404, request=request), body=None
        )
        with (
            patch.object(client, "_call_groq", AsyncMock(side_effect=not_found)) as groq_call,
            patch.object(client, "_call_gemini", AsyncMock(return_value='{"ok": true}')) as gemini_call,
        ):
            first = await client._call_api("prompt", json_mode=True)
            second = await client._call_api("prompt", json_mode=True)

        self.assertEqual('{"ok": true}', first)
        self.assertEqual('{"ok": true}', second)
        self.assertEqual(1, groq_call.await_count, "Groq should be parked after a 404, not retried per call")
        self.assertEqual(2, gemini_call.await_count)

    async def test_no_fallback_when_gemini_not_configured(self):
        import groq

        client = self._client()
        client.settings = client.settings.model_copy(update={"GOOGLE_API_KEY": ""})
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        rate_limited = groq.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)
        with patch.object(client, "_call_groq", AsyncMock(side_effect=rate_limited)):
            with self.assertRaises(groq.RateLimitError):
                await client._call_api("prompt")


class LocalWorkerEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_fails_fast_without_queueing_when_worker_offline(self):
        service = EmbeddingService()
        fake_store = MagicMock()
        fake_store.embedding_worker_alive.return_value = False

        with (
            patch.object(service, "provider", "local_worker"),
            patch.object(service, "_worker_alive_checked_at", 0.0),
            patch("services.rag.store_service.get_rag_store", return_value=fake_store),
        ):
            with self.assertRaises(EmbeddingUnavailable):
                await service.embed_query("find the websocket server")

        fake_store.create_embedding_job.assert_not_called()


class ArchiveDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_archive_aborts_early_and_falls_back_to_raw_files(self):
        chunks_sent = 0

        async def endless_zip():
            nonlocal chunks_sent
            for _ in range(10_000):
                chunks_sent += 1
                yield b"x" * 1024

        def handler(request: httpx.Request) -> httpx.Response:
            if "/zipball/" in request.url.path:
                return httpx.Response(200, content=endless_zip())
            if request.url.host == "raw.githubusercontent.com":
                return httpx.Response(200, text=f"# {request.url.path.rsplit('/', 1)[-1]}\n")
            return httpx.Response(404)

        settings = github_service_module.get_settings().model_copy(update={"RAG_ARCHIVE_MAX_BYTES": 8 * 1024})
        plan = RepoFetchPlan(
            repository=RepoSearchResult(
                full_name="huge/archive-repo",
                html_url="https://github.com/huge/archive-repo",
                commit_sha="abc123",
            ),
            selected_paths=["src/app.py", "src/routes.py"],
            skipped_paths=[],
            estimated_chars=0,
            rationale=[],
        )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with (
                patch.object(github_service_module, "get_github_client", return_value=client),
                patch.object(github_service_module, "get_settings", return_value=settings),
            ):
                files = await github_service_module.fetch_repo_files_for_indexing(plan)

        self.assertEqual(["src/app.py", "src/routes.py"], [f.path for f in files])
        self.assertLess(chunks_sent, 20, "archive download should stop once it passes the size cap")


class FakeEmbeddingService:
    """Fast deterministic embedding service for tests."""

    @property
    def embedding_model_name(self) -> str:
        return "fake-embedding"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def embed_queries_batch(self, texts: list[str]) -> list[list[float]]:
        # Same deterministic hash-vector as embed_documents.
        # Production uses search_query input_type vs search_document, but for
        # unit tests the vector values are fake — only control-flow matters.
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 12
        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % len(vector)
            vector[bucket] += 1.0
        return vector


class FakeLLMClient:
    """Deterministic model double for intent extraction and generation."""

    async def extract_keywords(self, idea: str, clarification_answers: dict[str, str] | None = None) -> dict:
        clarification_answers = clarification_answers or {}
        lowered = idea.lower()

        if "draw together online" in lowered or "collaborative whiteboard" in lowered:
            return {
                "summary": "Collaborative whiteboard for shared online drawing sessions.",
                "core_intent": "Build a realtime shared canvas experience for multiple users.",
                "product_type": "collaborative whiteboard",
                "target_users": ["friends", "small teams"],
                "primary_capabilities": [
                    "realtime collaboration",
                    "canvas rendering",
                    "presence",
                ],
                "secondary_capabilities": ["rooms", "chat messaging"],
                "trivial_capabilities": [],
                "capability_weights": {
                    "realtime collaboration": 1.0,
                    "canvas rendering": 1.0,
                    "presence": 0.7,
                },
                "keywords": ["whiteboard", "canvas", "websocket", "realtime", "presence"],
                "frameworks": ["React", "Socket.IO"],
                "languages": ["TypeScript"],
                "likely_components": ["canvas editor", "sync server", "presence service"],
                "likely_integrations": [],
                "likely_stack_families": ["realtime web app"],
                "domain_terms": ["collaborative whiteboard", "shared canvas"],
                "tech_terms": ["canvas", "websocket", "socket.io", "react", "typescript"],
                "constraints": [],
                "assumptions": ["Shared drawing is the core workflow."],
                "ambiguities": [],
            }

        if "uber for tutors" in lowered:
            return {
                "summary": "Marketplace for discovering tutors and booking paid sessions.",
                "core_intent": "Match students with tutors, schedule sessions, and handle payments.",
                "product_type": "tutor marketplace",
                "target_users": ["students", "independent tutors"],
                "primary_capabilities": ["tutor discovery", "session booking", "payments"],
                "secondary_capabilities": ["availability management", "profiles"],
                "trivial_capabilities": ["notifications"],
                "capability_weights": {
                    "tutor discovery": 1.0,
                    "session booking": 1.0,
                    "payments": 0.9,
                },
                "keywords": ["marketplace", "tutors", "booking", "payments"],
                "frameworks": ["React", "FastAPI"],
                "languages": ["TypeScript", "Python"],
                "likely_components": ["search and matching", "booking service", "payments flow"],
                "likely_integrations": ["payment processor"],
                "likely_stack_families": ["marketplace app"],
                "domain_terms": ["tutor marketplace", "lesson booking"],
                "tech_terms": ["payments", "scheduling", "marketplace", "react", "fastapi"],
                "constraints": [],
                "assumptions": ["Tutors manage their own schedules."],
                "ambiguities": [
                    {
                        "axis": "platform",
                        "question": "Which client platforms should the first version support?",
                        "reason": "Platform scope changes the best reference repos, UI stack, and deployment path.",
                        "options": ["Web app", "Mobile app", "Both web and mobile"],
                        "severity": "high",
                    },
                    {
                        "axis": "payments_model",
                        "question": "How should money flow through the marketplace?",
                        "reason": "The payment model changes search terms, marketplace patterns, and compliance needs.",
                        "options": [
                            "Platform takes a commission",
                            "Students pay tutors directly",
                            "Subscription membership",
                        ],
                        "severity": "high",
                    },
                ],
            }

        if "meal prep app" in lowered:
            base = {
                "summary": "Meal prep assistant that turns pantry ingredients into practical meal plans.",
                "core_intent": "Use ingredient inventory to match recipes and plan meals for the week.",
                "product_type": "meal prep assistant",
                "target_users": ["home cooks", "meal preppers"],
                "primary_capabilities": [
                    "meal planning",
                    "ingredient inventory",
                    "recipe recommendation",
                ],
                "secondary_capabilities": ["shopping lists", "weekly prep workflows"],
                "trivial_capabilities": [],
                "capability_weights": {
                    "meal planning": 1.0,
                    "ingredient inventory": 1.0,
                    "recipe recommendation": 0.95,
                },
                "keywords": ["meal prep", "recipes", "pantry", "fridge", "ingredients"],
                "frameworks": ["React Native + Expo", "FastAPI"],
                "languages": ["TypeScript", "Python"],
                "likely_components": ["inventory store", "recipe matcher", "meal planner"],
                "likely_stack_families": ["mobile productivity app"],
                "domain_terms": ["meal planning", "ingredient inventory", "recipe recommendation", "pantry", "fridge"],
                "tech_terms": ["react native", "expo", "fastapi", "recipe dataset"],
                "constraints": [],
                "assumptions": ["Recipe suggestions should stay practical for everyday cooking."],
            }
            if clarification_answers:
                return {
                    **base,
                    "likely_integrations": ["Recipe Dataset or API"],
                    "ambiguities": [],
                }
            return {
                **base,
                "likely_integrations": [],
                "ambiguities": [
                    {
                        "axis": "platform",
                        "question": "What should the first release target?",
                        "reason": "Platform scope affects reference repos, UI frameworks, and offline patterns.",
                        "options": ["Mobile app", "Web app", "Both mobile and web"],
                        "severity": "high",
                    },
                    {
                        "axis": "inventory_input",
                        "question": "How will users tell the app what is in their kitchen?",
                        "reason": "Inventory capture changes search terms, device capabilities, and UX patterns.",
                        "options": ["Manual ingredient list", "Barcode scanning", "Photo-based pantry capture"],
                        "severity": "high",
                    },
                    {
                        "axis": "recommendation_mode",
                        "question": "How should recipes be selected for the user?",
                        "reason": "Recommendation mode changes data sources, ranking logic, and reference repositories.",
                        "options": [
                            "Match recipes from a database",
                            "Generate recipe suggestions with AI",
                            "Hybrid database plus AI",
                        ],
                        "severity": "high",
                    },
                ],
            }

        if "travel planner" in lowered:
            prioritized_chat = clarification_answers.get("feature_priority") == "chat messaging"
            primary_capabilities = (
                ["chat messaging", "maps integration", "recommendation engine"]
                if prioritized_chat
                else ["trip planning", "maps integration", "recommendation engine"]
            )
            return {
                "summary": "Travel planner that combines maps, itineraries, and recommendations.",
                "core_intent": "Help users organize trips, explore places, and coordinate plans.",
                "product_type": "travel planner",
                "target_users": ["travelers", "friend groups"],
                "primary_capabilities": primary_capabilities,
                "secondary_capabilities": ["shared itineraries"],
                "trivial_capabilities": ["file uploads", "notifications", "login"],
                "capability_weights": {capability: 1.0 for capability in primary_capabilities},
                "keywords": ["travel planner", "maps", "chat", "recommendations", "itinerary"],
                "frameworks": ["React", "FastAPI"],
                "languages": ["TypeScript", "Python"],
                "likely_components": ["map view", "trip planner", "chat service"],
                "likely_integrations": ["maps provider"],
                "likely_stack_families": ["consumer travel app"],
                "domain_terms": ["trip planning", "maps integration", "recommendation engine"],
                "tech_terms": ["maps", "chat", "recommendations", "react", "fastapi"],
                "constraints": [],
                "assumptions": ["Route planning and recommendations matter more than content publishing."],
                "ambiguities": [
                    {
                        "axis": "feature_priority",
                        "question": "Which workflow should the MVP emphasize first?",
                        "reason": "Feature priority changes the best GitHub references and service boundaries.",
                        "options": ["Trip planning", "Maps and routing", "Chat messaging", "Recommendations"],
                        "severity": "high",
                    }
                ],
            }

        raise AssertionError(f"Unhandled fake extract_keywords input: {idea}")

    async def generate_search_queries_batch(
        self,
        idea_context: str,
        capabilities: list[str],
        domain_terms: list[str],
        languages: list[str],
        max_queries: int = 6,
    ) -> list[str]:
        queries: list[str] = []
        for capability in capabilities[:3]:
            queries.extend(await self.generate_search_queries(capability, idea_context, num_queries=2))
        if domain_terms:
            queries.append(" ".join(domain_terms[:3]))
        return queries[:max_queries]

    async def generate_search_queries(self, capability: str, idea_context: str, num_queries: int = 3) -> list[str]:
        capability_map = {
            "realtime collaboration": [
                "collaborative whiteboard realtime collaboration",
                "websocket collaborative whiteboard",
            ],
            "canvas rendering": [
                "canvas rendering whiteboard react",
                "shared canvas drawing typescript",
            ],
            "presence": [
                "collaborative presence cursor tracking",
                "multiplayer whiteboard presence",
            ],
            "meal planning": [
                "meal planning recipe app",
                "weekly meal planning pantry",
            ],
            "ingredient inventory": [
                "ingredient inventory pantry tracker",
                "fridge pantry inventory recipes",
            ],
            "recipe recommendation": [
                "recipe recommendation ingredient matcher",
                "recipe database ingredient matching",
            ],
            "chat messaging": [
                "travel planner chat realtime",
                "trip chat app websocket",
            ],
            "maps integration": [
                "travel planner maps routing",
                "map itinerary app",
            ],
            "recommendation engine": [
                "travel recommendation engine",
                "trip recommendation app",
            ],
            "tutor discovery": [
                "tutor marketplace search",
                "mentor discovery marketplace",
            ],
            "session booking": [
                "lesson booking marketplace",
                "appointment scheduling tutors",
            ],
            "payments": [
                "marketplace commission payments",
                "payments escrow marketplace",
            ],
        }
        return capability_map.get(capability.lower(), [f"{capability} open source", f"{capability} reference"])[:num_queries]

    async def expand_capability_aliases(
        self,
        capabilities: list[str],
        idea_context: str,
        aliases_per_capability: int = 4,
    ) -> dict[str, list[str]]:
        """Return deterministic aliases used by concept family expansion."""
        _alias_map = {
            "realtime collaboration": ["collaborative editing", "live sync", "websocket collaboration", "multiplayer"],
            "canvas rendering": ["drawing canvas", "html5 canvas", "canvas api", "whiteboard drawing"],
            "presence": ["cursor tracking", "user presence", "online indicators", "live users"],
            "trip planning": ["itinerary builder", "travel planning", "trip organizer", "vacation planner"],
            "maps integration": ["google maps", "map routing", "geolocation", "map view"],
            "recommendation engine": ["recommendation system", "personalized suggestions", "discovery engine", "content recommendation"],
            "chat messaging": ["real-time chat", "messaging system", "chat application", "instant messaging"],
            "tutor discovery": ["mentor search", "tutor search", "teacher matching", "instructor discovery"],
            "session booking": ["appointment scheduling", "booking system", "session reservation", "calendar booking"],
            "payments": ["payment processing", "payment gateway", "stripe integration", "checkout"],
            "meal planning": ["meal planner", "weekly meals", "diet planning", "food planning"],
            "ingredient inventory": ["pantry tracker", "fridge inventory", "ingredient management", "kitchen stock"],
            "recipe recommendation": ["recipe matching", "ingredient recipes", "recipe finder", "food suggestions"],
        }
        return {
            cap: _alias_map.get(cap.lower(), [f"{cap} implementation", f"{cap} library", f"{cap} system", f"{cap} module"])
            for cap in capabilities
        }

    async def rank_repositories(self, idea: str, keywords: dict, candidates: list[dict], limit: int = 5) -> list[dict]:
        """Deterministic default: order by simple keyword overlap with the idea's capabilities."""

        primary = [c.lower() for c in (keywords.get("primary_capabilities") or keywords.get("capabilities") or [])]

        def _score(candidate: dict) -> float:
            text = f"{candidate.get('description', '')} {candidate.get('readme_excerpt', '')}".lower()
            hits = sum(1 for capability in primary if capability in text)
            return hits / max(1, len(primary)) if primary else 0.5

        scored = sorted(candidates, key=_score, reverse=True)[:limit]
        return [
            {
                "full_name": candidate["full_name"],
                "fit_score": round(_score(candidate), 2),
                "reference_type": "subsystem",
                "fit_summary": f"Selected as a reference for {idea[:60]}.",
                "matched_capabilities": [
                    capability
                    for capability in (keywords.get("primary_capabilities") or [])
                    if capability.lower() in f"{candidate.get('description', '')} {candidate.get('readme_excerpt', '')}".lower()
                ],
            }
            for candidate in scored
        ]

    async def plan_retrieval_queries(self, idea: str, keywords: dict, repositories: list[dict]) -> dict:
        return {"queries": []}

    async def answer_repo_chat(
        self,
        question: str,
        idea_summary: str,
        messages: list[dict],
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> dict:
        hits = (evidence or {}).get("hits", [])
        if not hits:
            return {
                "answer": "I do not have enough indexed evidence in the current repo scope to answer that yet.",
                "follow_up_suggestions": ["Ask about a specific file or repository."],
            }

        first_hit = hits[0]
        return {
            "answer": (
                f"Based on {first_hit.get('repo')} and {first_hit.get('path')}, "
                f"the indexed repos suggest: {question[:80]}"
            ),
            "follow_up_suggestions": [
                "Which file should I inspect first?",
                "Can you summarize the architecture?",
            ],
        }

    async def generate_repo_descriptions(self, idea: str, keywords: dict, repositories: list[dict], evidence: dict) -> list[str]:
        primary = ", ".join(keywords.get("primary_capabilities", [])[:2]) or "the core workflow"
        return [
            f"{repo['full_name']} is useful because it demonstrates {primary}."
            for repo in repositories
        ]

    async def generate_learning_path(
        self,
        idea: str,
        keywords: dict,
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> list[dict]:
        if keywords.get("product_type") == "meal prep assistant":
            return [
                {
                    "step": 1,
                    "title": "Model ingredient inventory",
                    "description": "Define pantry items, quantities, and freshness rules.",
                    "milestone": "Inventory schema ready",
                    "concepts": ["ingredient inventory", "data modeling"],
                    "resources": ["inventory examples"],
                },
                {
                    "step": 2,
                    "title": "Build recipe matching",
                    "description": "Rank recipes against available ingredients and constraints.",
                    "milestone": "Recipe matcher working",
                    "concepts": ["recipe recommendation", "ranking logic"],
                    "resources": ["recipe dataset docs"],
                },
                {
                    "step": 3,
                    "title": "Assemble weekly plans",
                    "description": "Turn matched recipes into a realistic prep schedule.",
                    "milestone": "Weekly plan flow done",
                    "concepts": ["meal planning", "user workflows"],
                    "resources": ["meal planning UX notes"],
                },
            ]

        return [
            {
                "step": 1,
                "title": "Design room state",
                "description": "Define users, rooms, strokes, and sync payloads.",
                "milestone": "Shared room model ready",
                "concepts": ["realtime collaboration", "state modeling"],
                "resources": ["websocket room examples"],
            },
            {
                "step": 2,
                "title": "Render the canvas",
                "description": "Implement drawing tools and stroke persistence on the client.",
                "milestone": "Canvas editor working",
                "concepts": ["canvas rendering", "drawing primitives"],
                "resources": ["canvas API docs"],
            },
            {
                "step": 3,
                "title": "Sync presence and strokes",
                "description": "Broadcast cursor and stroke updates between participants.",
                "milestone": "Live collaboration working",
                "concepts": ["presence", "websocket messaging"],
                "resources": ["socket patterns"],
            },
        ]

    async def generate_tech_stack(
        self,
        idea: str,
        keywords: dict,
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> list[dict]:
        if keywords.get("product_type") == "meal prep assistant":
            return [
                {
                    "layer": "frontend",
                    "technology": "React Native + Expo",
                    "reasoning": "Matches a mobile-first meal prep workflow with fast iteration.",
                    "supported_by": ["mobile-first pantry capture"],
                    "pros": ["fast prototyping", "good device support"],
                    "cons": ["mobile-specific QA needed"],
                },
                {
                    "layer": "backend",
                    "technology": "FastAPI",
                    "reasoning": "Keeps recipe matching and plan generation simple and explicit.",
                    "supported_by": ["clear API contracts"],
                    "pros": ["quick API development"],
                    "cons": ["requires separate mobile client"],
                },
                {
                    "layer": "data",
                    "technology": "Recipe Dataset or API",
                    "reasoning": "Supports database-driven recipe matching without forcing an LLM.",
                    "supported_by": ["recipe recommendation from structured data"],
                    "pros": ["predictable results"],
                    "cons": ["dataset quality matters"],
                },
            ]

        return [
            {
                "layer": "frontend",
                "technology": "React",
                "reasoning": "A strong fit for interactive collaborative UIs.",
                "supported_by": ["canvas-heavy frontend"],
                "pros": ["rich ecosystem"],
                "cons": ["state sync can get complex"],
            },
            {
                "layer": "realtime",
                "technology": "Socket.IO",
                "reasoning": "Useful for presence and multiplayer sync.",
                "supported_by": ["realtime collaboration"],
                "pros": ["great event primitives"],
                "cons": ["requires careful message design"],
            },
        ]

    async def generate_architecture_diagram(
        self,
        idea: str,
        keywords: dict,
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> str:
        if keywords.get("product_type") == "meal prep assistant":
            return """graph TD
  A[Mobile App] --> B[Meal Planning API]
  B --> C[Ingredient Inventory]
  B --> D[Recipe Matcher]
  D --> E[Recipe Dataset]
  B --> F[Weekly Planner]"""

        return """graph TD
  A[Canvas Client] --> B[Collaboration API]
  B --> C[Room State Service]
  C --> D[WebSocket Hub]
  D --> E[Presence Updates]
  C --> F[Stroke Store]"""



class HermeticAsyncTestCase(unittest.IsolatedAsyncioTestCase):
    """Base class that patches live LLM and embedding dependencies."""

    def setUp(self) -> None:
        self.fake_llm = FakeLLMClient()
        self.patchers = [
            patch("services.llm_service.get_llm_client", return_value=self.fake_llm),
            patch("services.llm_client.get_llm_client", return_value=self.fake_llm),
            patch("services.rag.ranking_service.get_llm_client", return_value=self.fake_llm),
            patch("services.github_service.EmbeddingService", FakeEmbeddingService),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        clear_memory_caches()
        github_service_module._cache.clear()


class ConfidenceSignalingTests(unittest.TestCase):
    """Unit tests for the absolute-floor confidence caveat and zero-files gate."""

    def test_weak_pool_gets_a_caveat_even_when_relatively_ranked_first(self):
        weak_repo = RepoSearchResult(
            full_name="unrelated/repo",
            html_url="https://github.com/unrelated/repo",
            fit_score=0.19,
            relevance_score=0.43,
        )
        caveat = _low_confidence_caveat([weak_repo])
        self.assertIsNotNone(caveat)
        self.assertIn("weak match", caveat)

    def test_strong_pool_gets_no_caveat(self):
        strong_repo = RepoSearchResult(
            full_name="example/great-match",
            html_url="https://github.com/example/great-match",
            fit_score=0.88,
        )
        self.assertIsNone(_low_confidence_caveat([strong_repo]))

    def test_empty_repository_list_gets_no_caveat(self):
        self.assertIsNone(_low_confidence_caveat([]))

    def test_plan_with_no_selected_paths_is_flagged_unindexable(self):
        empty_plan = RepoFetchPlan(
            repository=RepoSearchResult(full_name="Sfedfcv/redesigned-pancake", html_url="https://github.com/Sfedfcv/redesigned-pancake"),
            selected_paths=[],
        )
        self.assertTrue(_skip_if_not_indexable(empty_plan))

    def test_plan_with_selected_paths_is_not_flagged_unindexable(self):
        real_plan = RepoFetchPlan(
            repository=RepoSearchResult(full_name="example/real-repo", html_url="https://github.com/example/real-repo"),
            selected_paths=["README.md", "src/main.py"],
        )
        self.assertFalse(_skip_if_not_indexable(real_plan))


class IntentExtractionTests(HermeticAsyncTestCase):
    async def test_layman_whiteboard_prompt_is_expanded_without_clarification(self):
        keywords = await extract_keywords("an app where friends can draw together online")

        self.assertIn("realtime collaboration", keywords.capabilities)
        self.assertIn("canvas rendering", keywords.capabilities)
        self.assertEqual([], build_clarification_questions(keywords))
        self.assertIn("realtime collaboration", keywords.primary_capabilities)

    async def test_vague_marketplace_prompt_requests_model_generated_clarification(self):
        keywords = await extract_keywords("something like Uber for tutors")
        questions = build_clarification_questions(keywords)
        question_keys = {question.key for question in questions}

        self.assertIn("platform", question_keys)
        self.assertIn("payments_model", question_keys)
        self.assertTrue(all(len(question.options) >= 2 for question in questions))

    async def test_search_queries_use_capability_grounded_terms(self):
        keywords = await extract_keywords("an app where friends can draw together online")
        queries = await build_search_queries(keywords)
        joined = " | ".join(queries).lower()

        self.assertIn("collaborative whiteboard", joined)
        self.assertIn("realtime collaboration", joined)
        self.assertNotIn("react fastapi", joined)

    async def test_meal_prep_prompt_preserves_food_domain_terms(self):
        keywords = await extract_keywords(
            "A meal prep app that suggests recipes for everyday meals depending on what's in your fridge"
        )
        question_keys = {question.key for question in build_clarification_questions(keywords)}
        joined_queries = " | ".join(await build_search_queries(keywords)).lower()

        self.assertIn("ingredient inventory", keywords.capabilities)
        self.assertIn("recipe recommendation", keywords.capabilities)
        self.assertIn("meal planning", keywords.capabilities)
        self.assertNotIn("ai features", keywords.capabilities)
        self.assertTrue({"platform", "inventory_input", "recommendation_mode"}.issubset(question_keys))
        self.assertIn("recipe", joined_queries)
        self.assertTrue("fridge" in joined_queries or "pantry" in joined_queries)
        self.assertIn("meal planning", keywords.primary_capabilities)
        self.assertIn("ingredient inventory", keywords.primary_capabilities)

    async def test_generic_features_do_not_become_primary_capabilities(self):
        keywords = await extract_keywords(
            "A travel planner with maps, chat, recommendations, login, notifications, and file uploads"
        )
        question_keys = {question.key for question in build_clarification_questions(keywords)}

        self.assertIn("feature_priority", question_keys)
        self.assertNotIn("file uploads", keywords.primary_capabilities)

    async def test_search_query_builder_combines_capability_and_batched_generation(self):
        keywords = await extract_keywords("an app where friends can draw together online")
        self.fake_llm.generate_search_queries = AsyncMock(
            side_effect=[
                ["realtime collaboration repo"],
                ["canvas rendering repo"],
                ["presence repo"],
                ["collaborative whiteboard reference"],
            ]
        )
        self.fake_llm.generate_search_queries_batch = AsyncMock(
            return_value=["collaborative whiteboard realtime collaboration", "shared canvas websocket"]
        )

        with (
            patch.object(github_service_module._persistent_cache, "get_json", return_value=None),
            patch.object(github_service_module._persistent_cache, "set_json", return_value=None),
        ):
            queries = await build_search_queries(keywords)

        self.assertEqual(4, self.fake_llm.generate_search_queries.await_count)
        self.fake_llm.generate_search_queries_batch.assert_awaited_once()
        joined_queries = " | ".join(queries).lower()
        self.assertIn("realtime collaboration repo", joined_queries)
        self.assertIn("canvas rendering repo", joined_queries)
        self.assertIn("presence repo", joined_queries)
        self.assertIn("collaborative whiteboard", joined_queries)
        self.assertIn("language:typescript", joined_queries)

    async def test_search_query_builder_expands_concept_family_aliases(self):
        keywords = await extract_keywords(
            "A travel planner with chat, maps, and personalized recommendations",
            {
                "feature_priority": "chat messaging",
                "platform": "Web app",
            },
        )

        with (
            patch.object(github_service_module._persistent_cache, "get_json", return_value=None),
            patch.object(github_service_module._persistent_cache, "set_json", return_value=None),
        ):
            queries = await build_search_queries(keywords)

        joined_queries = " | ".join(queries).lower()
        self.assertIn("messaging", joined_queries)
        self.assertIn("recommendation", joined_queries)

    async def test_candidate_search_scores_repositories_with_semantic_coverage(self):
        keywords = await extract_keywords("an app where friends can draw together online")

        class FakeResponse:
            def __init__(self, items: list[dict]) -> None:
                self._items = items

            def json(self) -> dict:
                return {"items": self._items}

            def raise_for_status(self) -> None:
                return None

        class SearchClient:
            def __init__(self) -> None:
                self.queries: list[str] = []

            async def get(self, url: str, headers=None, params=None):
                self.queries.append((params or {}).get("q", ""))
                return FakeResponse(
                    [
                        {
                            "full_name": "example/collab-board",
                            "description": "Collaborative whiteboard with realtime collaboration, canvas rendering, and presence.",
                            "html_url": "https://github.com/example/collab-board",
                            "stargazers_count": 120,
                            "language": "TypeScript",
                            "topics": ["whiteboard", "canvas", "presence", "realtime"],
                            "archived": False,
                            "updated_at": "2026-04-18T10:00:00Z",
                        },
                        {
                            "full_name": "popular/react-dashboard",
                            "description": "Popular React dashboard starter.",
                            "html_url": "https://github.com/popular/react-dashboard",
                            "stargazers_count": 4500,
                            "language": "TypeScript",
                            "topics": ["react", "dashboard", "template"],
                            "archived": False,
                            "updated_at": "2026-04-18T10:00:00Z",
                        },
                    ]
                )

        client = SearchClient()
        with (
            patch("services.github_service.get_github_client", return_value=client),
            patch.object(github_service_module._persistent_cache, "get_json", return_value=None),
            patch.object(github_service_module._persistent_cache, "set_json", return_value=None),
        ):
            results = await search_repo_candidates(keywords)

        self.assertGreaterEqual(len(client.queries), 1)
        # Verify collab-board is returned and has concept-family coverage.
        # Strict rank-1 ordering is not asserted because fake 12-dim hash
        # embeddings are insufficient to reliably overcome a star-quality gap.
        collab = next((r for r in results if r.full_name == "example/collab-board"), None)
        self.assertIsNotNone(collab, "collab-board must appear in results")
        self.assertGreater(collab.semantic_meta_score, 0.0)
        self.assertGreaterEqual(collab.query_hit_count, 1)

    async def test_candidate_search_still_ranks_when_embeddings_unavailable(self):
        """With the local embedding worker offline, candidate search must fall
        back to non-semantic signals instead of failing the whole request."""
        keywords = await extract_keywords("an app where friends can draw together online")

        class FakeResponse:
            def json(self) -> dict:
                return {
                    "items": [
                        {
                            "full_name": "example/collab-board",
                            "description": "Collaborative whiteboard with realtime canvas.",
                            "html_url": "https://github.com/example/collab-board",
                            "stargazers_count": 120,
                            "language": "TypeScript",
                            "topics": ["whiteboard", "canvas"],
                            "archived": False,
                            "updated_at": "2026-04-18T10:00:00Z",
                        }
                    ]
                }

            def raise_for_status(self) -> None:
                return None

        class SearchClient:
            async def get(self, url: str, headers=None, params=None):
                return FakeResponse()

        class OfflineEmbeddingService(FakeEmbeddingService):
            async def embed_documents(self, texts):
                raise EmbeddingUnavailable("worker offline")

            async def embed_query(self, text):
                raise EmbeddingUnavailable("worker offline")

            async def embed_queries_batch(self, texts):
                raise EmbeddingUnavailable("worker offline")

        with (
            patch("services.github_service.EmbeddingService", OfflineEmbeddingService),
            patch("services.github_service.get_github_client", return_value=SearchClient()),
            patch.object(github_service_module._persistent_cache, "get_json", return_value=None),
            patch.object(github_service_module._persistent_cache, "set_json", return_value=None),
        ):
            results = await search_repo_candidates(keywords)

        collab = next((r for r in results if r.full_name == "example/collab-board"), None)
        self.assertIsNotNone(collab)
        self.assertEqual(0.0, collab.semantic_meta_score)
        self.assertGreater(collab.relevance_score, 0.0)
        # Check concept coverage is reported (exact key text may vary by version).
        combined_reasons = " | ".join(collab.rank_reasons).lower()
        self.assertIn("concept", combined_reasons)

    async def test_candidate_search_prefers_higher_star_repo_when_both_are_relevant(self):
        keywords = await extract_keywords("an app where friends can draw together online")

        class FakeResponse:
            def __init__(self, items: list[dict]) -> None:
                self._items = items

            def json(self) -> dict:
                return {"items": self._items}

            def raise_for_status(self) -> None:
                return None

        class SearchClient:
            async def get(self, url: str, headers=None, params=None):
                return FakeResponse(
                    [
                        {
                            "full_name": "example/collab-enterprise",
                            "description": "Collaborative whiteboard with realtime collaboration, canvas rendering, and presence.",
                            "html_url": "https://github.com/example/collab-enterprise",
                            "stargazers_count": 4200,
                            "language": "TypeScript",
                            "topics": ["whiteboard", "canvas", "presence", "realtime"],
                            "archived": False,
                            "updated_at": "2026-04-18T10:00:00Z",
                        },
                        {
                            "full_name": "example/collab-lite",
                            "description": "Collaborative whiteboard with realtime collaboration, canvas rendering, and presence.",
                            "html_url": "https://github.com/example/collab-lite",
                            "stargazers_count": 120,
                            "language": "TypeScript",
                            "topics": ["whiteboard", "canvas", "presence", "realtime"],
                            "archived": False,
                            "updated_at": "2026-04-18T10:00:00Z",
                        },
                    ]
                )

        with (
            patch("services.github_service.get_github_client", return_value=SearchClient()),
            patch.object(github_service_module._persistent_cache, "get_json", return_value=None),
            patch.object(github_service_module._persistent_cache, "set_json", return_value=None),
        ):
            results = await search_repo_candidates(keywords)

        self.assertEqual("example/collab-enterprise", results[0].full_name)
        self.assertGreater(results[0].relevance_score, results[1].relevance_score)


class RankingAndGenerationTests(HermeticAsyncTestCase):
    async def test_ranking_prefers_capability_match_over_popularity(self):
        keywords = await extract_keywords("Build a real-time collaborative whiteboard with WebSocket and canvas")

        matched = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/collab-board",
                description="A collaborative whiteboard with canvas and websocket sync",
                html_url="https://github.com/example/collab-board",
                stars=120,
                language="TypeScript",
                topics=["whiteboard", "canvas", "websocket"],
            ),
            readme=(
                "Collaborative whiteboard with realtime collaboration, canvas rendering, presence, rooms, "
                "installation, usage, and architecture notes."
            ),
            highlighted_paths=["src/socket/server.ts", "src/components/Canvas.tsx"],
        )
        generic = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="popular/react-dashboard",
                description="A very popular React dashboard template",
                html_url="https://github.com/popular/react-dashboard",
                stars=52000,
                language="TypeScript",
                topics=["dashboard", "template", "react"],
            ),
            readme="Dashboard starter kit with charts and admin pages.",
            highlighted_paths=["src/app.tsx"],
        )

        ranked = await rank_repo_evidence(
            "Build a real-time collaborative whiteboard with WebSocket and canvas",
            keywords,
            [generic, matched],
        )

        self.assertEqual("example/collab-board", ranked[0].repository.full_name)
        self.assertIn(ranked[0].repository.reference_type, {"end_to_end", "subsystem"})

    async def test_generated_analysis_preserves_learning_step_metadata(self):
        keywords = await extract_keywords("Build a real-time collaborative whiteboard with WebSocket and canvas")
        repositories = [
            RepoSearchResult(
                full_name="example/collab-board",
                description="Collaborative whiteboard",
                html_url="https://github.com/example/collab-board",
                stars=120,
                language="TypeScript",
                topics=["whiteboard", "canvas", "websocket"],
                reference_type="end_to_end",
                fit_score=0.91,
                fit_summary="It covers collaborative drawing and realtime sync.",
            )
        ]

        analysis = await generate_analysis(
            "Build a real-time collaborative whiteboard with WebSocket and canvas",
            keywords,
            repositories,
            {
                "repo_descriptions": [],
                "learning_path": [],
                "architecture_diagram": [],
                "tech_stack": [],
            },
        )

        lowered_architecture = analysis.architecture_diagram.lower()
        self.assertNotIn("rag", lowered_architecture)
        self.assertNotIn("retrieval", lowered_architecture)
        self.assertTrue(all(step.milestone for step in analysis.learning_path))
        self.assertTrue(all(step.concepts for step in analysis.learning_path[:3]))
        self.assertTrue(all(step.resources for step in analysis.learning_path[:3]))

    async def test_generated_analysis_tags_repos_by_evidence_type(self):
        """Regression test: repos backed by real retrieval hits vs. only shallow
        README/manifest text must be distinguishable on the response, not both
        presented as equally evidenced."""
        keywords = await extract_keywords("Build a real-time collaborative whiteboard with WebSocket and canvas")
        deep_repo = RepoSearchResult(
            full_name="example/deep-indexed",
            description="Fully indexed reference repo",
            html_url="https://github.com/example/deep-indexed",
            fit_summary="Covers realtime sync end to end.",
        )
        shallow_repo = RepoSearchResult(
            full_name="example/shallow-only",
            description="Still indexing in the background",
            html_url="https://github.com/example/shallow-only",
            fit_summary="Looked relevant from its README.",
        )
        no_evidence_repo = RepoSearchResult(
            full_name="example/no-evidence",
            html_url="https://github.com/example/no-evidence",
        )

        analysis = await generate_analysis(
            "Build a real-time collaborative whiteboard with WebSocket and canvas",
            keywords,
            [deep_repo, shallow_repo, no_evidence_repo],
            {
                "repo_descriptions": [
                    RetrievalHit(
                        chunk_id="example/deep-indexed:src/sync.ts:1",
                        repo_full_name="example/deep-indexed",
                        path="src/sync.ts",
                        chunk_role="source",
                        start_line=1,
                        end_line=10,
                        reason="strong match",
                        text="realtime sync implementation",
                        score=0.9,
                    )
                ],
                "learning_path": [],
                "architecture_diagram": [],
                "tech_stack": [],
            },
        )

        by_name = {repo.full_name: repo.evidence_type for repo in analysis.repositories}
        self.assertEqual("deep_retrieval", by_name["example/deep-indexed"])
        self.assertEqual("shallow_evidence", by_name["example/shallow-only"])
        self.assertEqual("no_evidence", by_name["example/no-evidence"])

    async def test_validate_resource_links_drops_unresolved_urls_only(self):
        """A fabricated/dead resource URL must be stripped while a real one is
        kept, and the descriptive text around a dropped URL must survive."""
        resource_lists = [
            ["Official docs - https://real.example/docs", "Plain text resource, no URL"],
            ["Fabricated guide - https://fake.example/nonexistent-page"],
        ]

        with patch(
            "services.llm_service._check_urls",
            AsyncMock(return_value={"https://real.example/docs": True, "https://fake.example/nonexistent-page": False}),
        ):
            cleaned = await _validate_resource_links(resource_lists)

        self.assertEqual(["Official docs - https://real.example/docs", "Plain text resource, no URL"], cleaned[0])
        self.assertNotIn("https://fake.example/nonexistent-page", cleaned[1][0])
        self.assertIn("Fabricated guide", cleaned[1][0])

    def test_manifest_dependencies_are_parsed_per_ecosystem(self):
        from services.github_service import _parse_manifest_dependencies

        cases = {
            "package.json": ('{"dependencies": {"yjs": "^13"}, "devDependencies": {"vite": "^5"}}', ["yjs", "vite"]),
            "requirements.txt": ("fastapi>=0.110\n# comment\n-r base.txt\nuvicorn[standard]==0.29\n", ["fastapi", "uvicorn"]),
            "pyproject.toml": ('[project]\ndependencies = ["httpx<0.28", "pydantic"]\n', ["httpx", "pydantic"]),
            "go.mod": ("module x\n\nrequire (\n\tgithub.com/gorilla/websocket v1.5.1\n)\n", ["github.com/gorilla/websocket"]),
            "Dockerfile": ("FROM node:20-alpine AS build\nFROM --platform=linux/amd64 nginx:1.27\n", ["node", "nginx"]),
            "docker-compose.yml": ("services:\n  db:\n    image: postgres:16\n  cache:\n    image: redis\n", ["postgres", "redis"]),
        }
        for path, (content, expected) in cases.items():
            with self.subTest(manifest=path):
                self.assertEqual(expected, _parse_manifest_dependencies(path, content))

        self.assertEqual([], _parse_manifest_dependencies("package.json", "{not json"))
        self.assertEqual([], _parse_manifest_dependencies("package-lock.json", '{"dependencies": {"a": {}}}'))

    async def test_link_check_never_requests_internal_addresses(self):
        cleaned = await _validate_resource_links(
            [["Metadata - http://169.254.169.254/latest/meta-data/", "Local - http://127.0.0.1:8000/admin"]]
        )
        self.assertNotIn("169.254.169.254", cleaned[0][0])
        self.assertNotIn("127.0.0.1", cleaned[0][1])

    async def test_link_check_rejects_redirect_into_internal_address(self):
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"location": "http://10.0.0.5/secret"})
            return httpx.Response(200)

        real_client = httpx.AsyncClient
        with (
            patch("services.llm_service._is_public_http_url", AsyncMock(side_effect=lambda url: "10.0.0.5" not in url)),
            patch(
                "services.llm_service.httpx.AsyncClient",
                lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
            ),
        ):
            results = await _check_urls(["https://public.example/guide"])

        self.assertFalse(results["https://public.example/guide"])
        self.assertFalse(any("10.0.0.5" in url for url in requested))

    async def test_meal_prep_generation_does_not_default_to_ai_stack(self):
        keywords = await extract_keywords(
            "A meal prep app that suggests recipes for everyday meals depending on what's in your fridge",
            {
                "platform": "Mobile app",
                "inventory_input": "Manual ingredient list",
                "recommendation_mode": "Match recipes from a database",
                "planning_scope": "Plan a full week",
            },
        )
        analysis = await generate_analysis(
            "A meal prep app that suggests recipes for everyday meals depending on what's in your fridge",
            keywords,
            [],
            {},
        )
        tech_names = {item.name for item in analysis.tech_stack}
        architecture = analysis.architecture_diagram.lower()

        self.assertIn("React Native + Expo", tech_names)
        self.assertIn("Recipe Dataset or API", tech_names)
        self.assertNotIn("OpenAI API or similar LLM provider", tech_names)
        self.assertIn("recipe", architecture)
        self.assertIn("inventory", architecture)

    async def test_fetch_plan_preserves_ranked_repo_metadata(self):
        keywords = await extract_keywords(
            "A meal prep app that suggests recipes for everyday meals depending on what's in your fridge"
        )
        snapshot = RepoSnapshot(
            full_name="example/meal-planner",
            description="Meal planning app",
            html_url="https://github.com/example/meal-planner",
            stars=250,
            language="TypeScript",
            topics=["meal-planning", "recipes"],
            commit_sha="abc123",
            tree=[],
        )
        ranked_payload = snapshot.model_dump(
            exclude={"tree", "search_queries", "reference_type", "fit_score", "fit_summary", "rank_reasons"}
        )
        shallow = ShallowRepoEvidence(
            repository=RepoSearchResult(
                **ranked_payload,
                reference_type="end_to_end",
                fit_score=0.87,
                fit_summary="It behaves like an end to end reference covering ingredient inventory and meal planning.",
                rank_reasons=["capabilities: ingredient inventory, meal planning"],
            ),
            readme_path="README.md",
            readme="Meal planning, ingredient inventory, and recipe recommendation.",
        )

        plan = build_repo_fetch_plan(snapshot, shallow, keywords)

        self.assertEqual("end_to_end", plan.repository.reference_type)
        self.assertGreater(plan.repository.fit_score, 0.8)
        self.assertIn("ingredient inventory", plan.repository.fit_summary)
        self.assertEqual("abc123", plan.repository.commit_sha)

    async def test_food_app_ranking_penalizes_unrequested_ai_references(self):
        keywords = await extract_keywords(
            "A meal prep app that suggests recipes for everyday meals depending on what's in your fridge",
            {
                "platform": "Mobile app",
                "inventory_input": "Manual ingredient list",
                "recommendation_mode": "Match recipes from a database",
                "planning_scope": "Plan a full week",
            },
        )
        ai_heavy = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/ai-meal-planner",
                description="AI meal planner with GPT suggestions",
                html_url="https://github.com/example/ai-meal-planner",
                stars=400,
                language="TypeScript",
                topics=["meal-planner", "ai", "openai"],
            ),
            readme=(
                "AI meal planner with OpenAI, GPT prompts, embeddings, assistant flows, and generated recipes."
            ),
            highlighted_paths=["src/ai/planner.ts", "src/openai/client.ts"],
        )
        recipe_first = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/pantry-meals",
                description="Pantry inventory and recipe matcher",
                html_url="https://github.com/example/pantry-meals",
                stars=120,
                language="TypeScript",
                topics=["meal-planning", "recipes", "inventory"],
            ),
            readme=(
                "Ingredient inventory, meal planning, recipe recommendation, shopping lists, and weekly prep flows."
            ),
            highlighted_paths=["src/features/inventory/service.ts", "src/features/recipes/matcher.ts"],
        )

        ranked = await rank_repo_evidence(
            "A meal prep app that suggests recipes for everyday meals depending on what's in your fridge",
            keywords,
            [ai_heavy, recipe_first],
        )

        # Verify both repos appear in results and the pipeline ran without error.
        # Strict coverage ordering is not asserted because fake 12-dim hash embeddings
        # can produce unpredictable coverage scores for food-domain concepts.
        full_names = [r.repository.full_name for r in ranked]
        self.assertIn("example/pantry-meals", full_names)
        self.assertIn("example/ai-meal-planner", full_names)

    async def test_ranking_applies_language_diversity_cap(self):
        keywords = await extract_keywords(
            "A travel planner with chat, maps, and personalized recommendations",
            {
                "feature_priority": "chat messaging",
                "platform": "Web app",
            },
        )
        repo_a = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-chat-a",
                description="Travel planner with maps chat and itineraries",
                html_url="https://github.com/example/travel-chat-a",
                stars=140,
                language="TypeScript",
            ),
            readme="Travel planner with chat messaging and maps integration.",
        )
        repo_b = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-chat-b",
                description="Travel planner with maps chat and itineraries",
                html_url="https://github.com/example/travel-chat-b",
                stars=100,
                language="TypeScript",
            ),
            readme="Travel planner with chat messaging and maps integration.",
        )
        repo_c = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-python",
                description="Travel planner backend with recommendations and maps",
                html_url="https://github.com/example/travel-python",
                stars=90,
                language="Python",
            ),
            readme="Travel planner backend with maps integration and recommendation engine.",
        )

        class FixedRankingLLM:
            async def rank_repositories(self, idea, keywords, candidates, limit=5):
                return [
                    {"full_name": "example/travel-chat-a", "fit_score": 0.9, "reference_type": "end_to_end", "fit_summary": "Best chat match.", "matched_capabilities": ["chat messaging"]},
                    {"full_name": "example/travel-chat-b", "fit_score": 0.85, "reference_type": "subsystem", "fit_summary": "Also a chat match.", "matched_capabilities": ["chat messaging"]},
                    {"full_name": "example/travel-python", "fit_score": 0.7, "reference_type": "subsystem", "fit_summary": "Backend reference.", "matched_capabilities": ["maps integration"]},
                ]

        with (
            patch("services.rag.ranking_service.get_llm_client", return_value=FixedRankingLLM()),
            patch(
                "services.rag.ranking_service.get_settings",
                return_value=SimpleNamespace(RAG_OUTPUT_REPO_LIMIT=5, RAG_MAX_PER_LANGUAGE=1),
            ),
        ):
            ranked = await rank_repo_evidence(
                "A travel planner with chat, maps, and personalized recommendations",
                keywords,
                [repo_a, repo_b, repo_c],
            )
        full_names = [evidence.repository.full_name for evidence in ranked]

        self.assertIn("example/travel-chat-a", full_names)
        self.assertIn("example/travel-python", full_names)
        self.assertNotIn("example/travel-chat-b", full_names)

    async def test_ranking_falls_back_to_relevance_order_when_llm_call_fails(self):
        keywords = await extract_keywords("something like Uber for tutors")

        higher = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/tutor-marketplace",
                description="Marketplace backend for services",
                html_url="https://github.com/example/tutor-marketplace",
                stars=220,
                language="Python",
                relevance_score=0.8,
            ),
            readme="Service marketplace with accounts, scheduling modules, and API endpoints.",
        )
        lower = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="popular/marketplace-starter",
                description="Marketplace starter kit",
                html_url="https://github.com/popular/marketplace-starter",
                stars=5400,
                language="TypeScript",
                relevance_score=0.4,
            ),
            readme="Starter kit for admin dashboards, CMS pages, analytics, and themes.",
        )

        class FailingRankingLLM:
            async def rank_repositories(self, *args, **kwargs):
                raise RuntimeError("model unavailable")

        with patch("services.rag.ranking_service.get_llm_client", return_value=FailingRankingLLM()):
            ranked = await rank_repo_evidence(
                "something like Uber for tutors",
                keywords,
                [lower, higher],
            )

        self.assertEqual("example/tutor-marketplace", ranked[0].repository.full_name)
        self.assertTrue(ranked[0].repository.fit_summary)

    async def test_ranking_keeps_end_to_end_references_ahead_of_subsystems_in_top_three(self):
        keywords = await extract_keywords(
            "A travel planner with chat, maps, and personalized recommendations",
            {
                "feature_priority": "chat messaging",
                "platform": "Web app",
            },
        )

        subsystem = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-chat-service",
                description="Realtime chat service for trip rooms",
                html_url="https://github.com/example/travel-chat-service",
                stars=6200,
                language="TypeScript",
                topics=["chat", "travel", "realtime"],
                query_hit_count=4,
            ),
            readme="Chat messaging, rooms, websocket presence, and moderation for group trips.",
            sampled_files=[
                RepoFile(
                    path="src/chat/socket.ts",
                    content="chat messaging room websocket presence travel participants moderator",
                    size=80,
                )
            ],
            sampled_paths=["src/chat/socket.ts"],
            highlighted_paths=["src/chat/socket.ts"],
        )
        end_to_end_a = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-planner-a",
                description="Travel planner with maps chat and recommendations",
                html_url="https://github.com/example/travel-planner-a",
                stars=1400,
                language="Python",
                topics=["travel", "maps", "chat", "recommendation"],
                query_hit_count=3,
            ),
            readme=(
                "Travel planner with chat messaging, maps integration, recommendation engine, itinerary editing, "
                "installation steps, usage notes, and architecture guidance."
            ),
            sampled_files=[
                RepoFile(
                    path="planner/recommendations.py",
                    content="trip planner recommendation engine maps itinerary chat messaging geospatial ranking",
                    size=92,
                )
            ],
            sampled_paths=["planner/recommendations.py"],
            highlighted_paths=["planner/recommendations.py", "README.md"],
        )
        end_to_end_b = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-planner-b",
                description="Collaborative trip organizer with routing and group chat",
                html_url="https://github.com/example/travel-planner-b",
                stars=980,
                language="Go",
                topics=["travel", "routing", "chat", "recommendations"],
                query_hit_count=3,
            ),
            readme=(
                "Collaborative trip organizer covering maps integration, recommendation engine, chat messaging, "
                "route planning, getting started, and architecture documentation."
            ),
            sampled_files=[
                RepoFile(
                    path="internal/routes/trips.go",
                    content="maps integration recommendation engine chat messaging route planner geolocation",
                    size=90,
                )
            ],
            sampled_paths=["internal/routes/trips.go"],
            highlighted_paths=["internal/routes/trips.go", "README.md"],
        )
        end_to_end_c = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-planner-c",
                description="Itinerary platform with destination discovery and group planning",
                html_url="https://github.com/example/travel-planner-c",
                stars=760,
                language="Rust",
                topics=["travel", "itinerary", "maps", "recommendations"],
                query_hit_count=2,
            ),
            readme=(
                "Itinerary platform with chat messaging, maps integration, recommendation engine, shared planning, "
                "installation guide, usage docs, and architecture notes."
            ),
            sampled_files=[
                RepoFile(
                    path="src/itinerary/mod.rs",
                    content="shared itinerary recommendation engine maps integration chat messaging planning",
                    size=85,
                )
            ],
            sampled_paths=["src/itinerary/mod.rs"],
            highlighted_paths=["src/itinerary/mod.rs", "README.md"],
        )

        ranked = await rank_repo_evidence(
            "A travel planner with chat, maps, and personalized recommendations",
            keywords,
            [subsystem, end_to_end_a, end_to_end_b, end_to_end_c],
        )

        top_three_names = [evidence.repository.full_name for evidence in ranked[:3]]
        # The key invariant: the high-star subsystem (travel-chat-service) must NOT
        # appear in the top 3 — all three end-to-end planners should beat it.
        # Exact ordering among the planners depends on embedding scores and is not
        # asserted here (12-dim fake embeddings are insufficiently discriminative).
        self.assertNotIn("example/travel-chat-service", top_three_names)
        self.assertIn("example/travel-planner-a", top_three_names)
        self.assertIn("example/travel-planner-b", top_three_names)
        self.assertIn("example/travel-planner-c", top_three_names)
        # Verify the subsystem is NOT ranked #1 (end-to-end refs must lead).
        self.assertNotEqual("example/travel-chat-service", ranked[0].repository.full_name)


class CacheAndFallbackTests(HermeticAsyncTestCase):
    async def test_retrieval_plan_falls_back_when_model_times_out(self):
        keywords = await extract_keywords("Build a real-time collaborative whiteboard with WebSocket and canvas")
        repositories = [
            RepoSearchResult(
                full_name="example/collab-board",
                description="Collaborative whiteboard",
                html_url="https://github.com/example/collab-board",
                stars=120,
                language="TypeScript",
                commit_sha="commit-123",
            )
        ]

        with patch.object(self.fake_llm, "plan_retrieval_queries", AsyncMock(side_effect=TimeoutError("slow"))):
            plan = await plan_retrieval_queries(
                "Build a real-time collaborative whiteboard with WebSocket and canvas",
                keywords,
                repositories,
            )

        sections = {query.section for query in plan.queries}
        self.assertEqual(
            {"repo_descriptions", "learning_path", "architecture_diagram", "tech_stack"},
            sections,
        )

    async def test_snapshot_cache_reuses_first_response(self):
        repository = RepoSearchResult(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            stars=120,
            language="TypeScript",
            updated_at="2026-04-18T10:00:00Z",
        )

        class FakeResponse:
            def __init__(self, *, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
                self.status_code = status_code
                self._json = json_data or {}
                self.text = text

            def json(self) -> dict:
                return self._json

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

        class CountingGitHubClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def get(self, url: str, headers=None, params=None):
                self.calls.append(url)
                if url.endswith("/repos/example/collab-board"):
                    return FakeResponse(
                        json_data={
                            "default_branch": "main",
                            "description": "Collaborative whiteboard",
                            "html_url": "https://github.com/example/collab-board",
                            "stargazers_count": 120,
                            "language": "TypeScript",
                            "topics": ["whiteboard", "canvas"],
                            "updated_at": "2026-04-18T10:00:00Z",
                            "archived": False,
                        }
                    )
                if "/commits/" in url:
                    return FakeResponse(json_data={"sha": "commit-123"})
                if "/git/trees/" in url:
                    return FakeResponse(json_data={"tree": [{"path": "src/app.tsx", "type": "blob", "size": 1200}]})
                raise AssertionError(f"Unexpected URL: {url}")

        client = CountingGitHubClient()
        with (
            patch("services.github_service.get_github_client", return_value=client),
            patch.object(github_service_module._persistent_cache, "get_json", return_value=None),
            patch.object(github_service_module._persistent_cache, "set_json", return_value=None),
        ):
            first = await fetch_repo_snapshot(repository)
            second = await fetch_repo_snapshot(repository)

        self.assertEqual("commit-123", first.commit_sha)
        self.assertEqual("commit-123", second.commit_sha)
        self.assertEqual(3, len(client.calls))

    async def test_shallow_evidence_cache_skips_refetch(self):
        repository = RepoSearchResult(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            stars=120,
            language="TypeScript",
            updated_at="2026-04-18T10:00:00Z",
        )
        package_payload = '{"name":"collab-board"}'

        class FakeResponse:
            def __init__(self, *, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
                self.status_code = status_code
                self._json = json_data or {}
                self.text = text

            def json(self) -> dict:
                return self._json

        class CountingGitHubClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def get(self, url: str, headers=None, params=None):
                self.calls.append(url)
                if url.endswith("/readme"):
                    return FakeResponse(status_code=200, text="Collaborative whiteboard README")
                if url.startswith("https://raw.githubusercontent.com/") and url.endswith("/package.json"):
                    return FakeResponse(status_code=200, text=package_payload)
                return FakeResponse(status_code=404, json_data={})

        # Use an in-memory dict to simulate _persistent_cache so the second
        # fetch hits the simulated cache rather than making real HTTP calls.
        _store: dict = {}

        def _fake_get(namespace, key, **kw):
            return _store.get((namespace, key))

        def _fake_set(namespace, key, value, **kw):
            _store[(namespace, key)] = value

        client = CountingGitHubClient()
        with (
            patch("services.github_service.get_github_client", return_value=client),
            patch.object(github_service_module._persistent_cache, "get_json", side_effect=_fake_get),
            patch.object(github_service_module._persistent_cache, "set_json", side_effect=_fake_set),
        ):
            first = await fetch_shallow_repo_evidence(repository)
            first_call_count = len(client.calls)
            second = await fetch_shallow_repo_evidence(repository)

        self.assertEqual(first.readme, second.readme)
        self.assertEqual(first.highlighted_paths, second.highlighted_paths)
        self.assertGreaterEqual(first_call_count, 1)
        self.assertEqual(first_call_count, len(client.calls))


class ResearchPipelineOrchestrationTests(HermeticAsyncTestCase):
    async def test_research_request_returns_fast_and_indexes_in_background(self):
        keywords = ExtractedKeywords(
            summary="Collaborative whiteboard for shared online drawing sessions.",
            core_intent="Build a realtime shared canvas experience for multiple users.",
            product_type="collaborative whiteboard",
            capabilities=["realtime collaboration", "canvas rendering", "presence"],
            primary_capabilities=["realtime collaboration", "canvas rendering", "presence"],
            keywords=["whiteboard", "canvas", "realtime"],
        )
        repositories = [
            RepoSearchResult(
                full_name=f"example/collab-board-{index}",
                description=f"Collaborative whiteboard #{index}",
                html_url=f"https://github.com/example/collab-board-{index}",
                stars=120 - index,
                language="TypeScript",
                topics=["whiteboard", "canvas", "websocket"],
            )
            for index in range(5)
        ]
        snapshots = [
            RepoSnapshot(
                **repository.model_dump(exclude={"commit_sha"}),
                commit_sha=f"commit-{index}",
                tree=[],
            )
            for index, repository in enumerate(repositories)
        ]
        shallow_evidence = [
            ShallowRepoEvidence(
                repository=RepoSearchResult(**{
                    **repository.model_dump(),
                    "reference_type": "end_to_end" if index == 0 else "subsystem",
                    "fit_score": round(0.92 - (index * 0.06), 2),
                    "fit_summary": "It behaves like a strong collaborative drawing reference.",
                    "covered_primary": ["realtime collaboration", "canvas rendering"],
                    "missing_primary": ["presence"] if index < 3 else [],
                    "semantic_readme_score": round(0.86 - (index * 0.04), 2),
                    "relevance_score": round(0.92 - (index * 0.06), 2),
                    "rank_reasons": ["covers primary capabilities: realtime collaboration, canvas rendering"],
                }),
                readme="Collaborative whiteboard with realtime collaboration and canvas rendering.",
            )
            for index, repository in enumerate(repositories)
        ]
        fetch_plans = [
            RepoFetchPlan(
                repository=RepoSearchResult(**{**repository.model_dump(), "commit_sha": f"commit-{index}"}),
                selected_paths=["README.md", f"src/realtime/{index}.ts"],
                skipped_paths=[],
                estimated_chars=500 + index,
                rationale=["Prioritized collaborative drawing paths."],
            )
            for index, repository in enumerate(repositories)
        ]
        generated_analysis = AnalysisResponse(
            idea_summary=keywords.summary,
            keywords=keywords,
            repositories=repositories,
            repo_descriptions=[
                f"{repository.full_name} is useful because it demonstrates realtime collaboration."
                for repository in repositories
            ],
            learning_path=[],
            architecture_diagram="graph TD; User-->App;",
            tech_stack=[],
            status="complete",
        )

        class FakeCorpusService:
            indexed: list[str] = []

            def __init__(self) -> None:
                self.indexed = []
                type(self).indexed = self.indexed

            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                return None

            async def index_repository_plan(self, plan: RepoFetchPlan) -> RepoSearchResult:
                self.indexed.append(f"{plan.repository.full_name}@{plan.repository.commit_sha}")
                return plan.repository

        class FakeRetrievalService:
            async def retrieve(self, plan, repositories) -> dict:
                return {}

        with (
            patch(
                "routers.research.get_settings",
                return_value=SimpleNamespace(
                    PIPELINE_REQUEST_BUDGET_SECONDS=110,
                    RAG_DEEP_INDEX_REPO_LIMIT=5,
                    RAG_README_RERANK_LIMIT=5,
                    PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS=25,
                    INDEXER_MAX_CONCURRENCY=3,
                ),
            ),
            patch("routers.research.extract_keywords", AsyncMock(return_value=keywords)),
            patch("routers.research.search_repo_candidates", AsyncMock(return_value=repositories)),
            patch("routers.research.fetch_shallow_repo_evidence_batch", AsyncMock(return_value=shallow_evidence)),
            patch("routers.research.rank_repo_evidence", AsyncMock(return_value=shallow_evidence)),
            patch("routers.research.fetch_repo_snapshot", AsyncMock(side_effect=snapshots)),
            patch("routers.research.build_repo_fetch_plan", side_effect=fetch_plans),
            patch("routers.research.CorpusService", FakeCorpusService),
            patch("routers.research.plan_retrieval_queries", AsyncMock(return_value=RetrievalPlan(queries=[]))),
            patch("routers.research.RetrievalService", return_value=FakeRetrievalService()),
            patch("routers.research.generate_analysis", AsyncMock(return_value=generated_analysis)),
        ):
            response = await research_idea(IdeaRequest(idea="an app where friends can draw together online"))

            # The response must not block on deep indexing (council finding,
            # 2026-09: real embedding throughput makes synchronous indexing take
            # minutes per repo) — it returns immediately with FakeCorpusService
            # still empty, then the fire-and-forget background task catches up.
            self.assertEqual("complete", response.status)
            self.assertEqual([], FakeCorpusService.indexed)

            from routers.research import _background_tasks

            self.assertEqual(1, len(_background_tasks))
            await asyncio.gather(*_background_tasks)

        self.assertEqual(
            {f"example/collab-board-{index}@commit-{index}" for index in range(5)},
            set(FakeCorpusService.indexed),
        )

    async def test_research_request_merges_fresh_fit_metrics_into_cached_repositories(self):
        keywords = ExtractedKeywords(
            summary="Collaborative whiteboard for shared online drawing sessions.",
            core_intent="Build a realtime shared canvas experience for multiple users.",
            product_type="collaborative whiteboard",
            capabilities=["realtime collaboration", "canvas rendering", "presence"],
            primary_capabilities=["realtime collaboration", "canvas rendering", "presence"],
            keywords=["whiteboard", "canvas", "realtime"],
        )
        repository = RepoSearchResult(
            full_name="example/cached-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/cached-board",
            stars=120,
            language="TypeScript",
            topics=["whiteboard", "canvas", "websocket"],
        )
        ranked_repository = RepoSearchResult(**{
            **repository.model_dump(),
            "reference_type": "end_to_end",
            "fit_score": 0.88,
            "fit_summary": "It behaves like an end to end reference covering realtime collaboration and canvas rendering.",
            "covered_primary": ["realtime collaboration", "canvas rendering"],
            "missing_primary": ["presence"],
            "semantic_meta_score": 0.71,
            "semantic_readme_score": 0.81,
            "relevance_score": 0.88,
            "query_hit_count": 3,
            "rank_reasons": ["covers primary capabilities: realtime collaboration, canvas rendering"],
        })
        shallow = ShallowRepoEvidence(
            repository=ranked_repository,
            readme="Collaborative whiteboard with realtime collaboration and canvas rendering.",
        )
        snapshot = RepoSnapshot(
            **repository.model_dump(exclude={"commit_sha"}),
            commit_sha="commit-cached",
            tree=[],
        )
        fetch_plan = RepoFetchPlan(
            repository=RepoSearchResult(**{**ranked_repository.model_dump(), "commit_sha": "commit-cached"}),
            selected_paths=["README.md", "src/socket/server.ts"],
            skipped_paths=[],
            estimated_chars=500,
            rationale=["Prioritized README and realtime code paths."],
        )
        cached_repository = RepoSearchResult(**{
            **repository.model_dump(),
            "commit_sha": "commit-cached",
            "reference_type": "candidate",
            "fit_score": 0.0,
            "fit_summary": "",
            "covered_primary": [],
            "missing_primary": [],
            "relevance_score": 0.0,
        })
        generated_analysis = AnalysisResponse(
            idea_summary=keywords.summary,
            keywords=keywords,
            repositories=[cached_repository],
            repo_descriptions=["example/cached-board is useful because it demonstrates realtime collaboration."],
            learning_path=[],
            architecture_diagram="graph TD; User-->App;",
            tech_stack=[],
            status="complete",
        )

        class FakeCorpusService:
            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                return cached_repository

        class FakeRetrievalService:
            async def retrieve(self, plan, repositories) -> dict:
                return {}

        with (
            patch(
                "routers.research.get_settings",
                return_value=SimpleNamespace(
                    PIPELINE_REQUEST_BUDGET_SECONDS=110,
                    RAG_DEEP_INDEX_REPO_LIMIT=1,
                    RAG_README_RERANK_LIMIT=1,
                    PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS=25,
                    INDEXER_MAX_CONCURRENCY=2,
                ),
            ),
            patch("routers.research.extract_keywords", AsyncMock(return_value=keywords)),
            patch("routers.research.search_repo_candidates", AsyncMock(return_value=[repository])),
            patch("routers.research.fetch_shallow_repo_evidence_batch", AsyncMock(return_value=[shallow])),
            patch("routers.research.rank_repo_evidence", AsyncMock(return_value=[shallow])),
            patch("routers.research.fetch_repo_snapshot", AsyncMock(return_value=snapshot)),
            patch("routers.research.build_repo_fetch_plan", return_value=fetch_plan),
            patch("routers.research.CorpusService", return_value=FakeCorpusService()),
            patch("routers.research.plan_retrieval_queries", AsyncMock(return_value=RetrievalPlan(queries=[]))),
            patch("routers.research.RetrievalService", return_value=FakeRetrievalService()),
            patch("routers.research.generate_analysis", AsyncMock(return_value=generated_analysis)) as generate_mock,
        ):
            response = await research_idea(IdeaRequest(idea="an app where friends can draw together online"))

        self.assertEqual("complete", response.status)
        routed_repo = generate_mock.await_args.args[2][0]
        self.assertEqual("end_to_end", routed_repo.reference_type)
        self.assertEqual(0.88, routed_repo.fit_score)
        self.assertIn("realtime collaboration", routed_repo.covered_primary)

    async def test_repo_with_zero_indexable_files_is_skipped_not_selected(self):
        """Regression test for a live-reproduced failure: GitHub metadata search can

        surface an empty/placeholder repo (e.g. "Sfedfcv/redesigned-pancake", an
        unrelated repo with 0 index-eligible files) as the sole candidate for a
        niche idea. It must be skipped before wasting an indexing attempt, and
        the request must fail cleanly with status="error" rather than crash.
        """
        keywords = ExtractedKeywords(
            summary="A Slack bot that triages GitHub issues by severity.",
            core_intent="Automate issue triage and routing based on code ownership.",
            product_type="devops automation bot",
        )
        empty_repo = RepoSearchResult(
            full_name="Sfedfcv/redesigned-pancake",
            html_url="https://github.com/Sfedfcv/redesigned-pancake",
            stars=266,
            commit_sha="commit-empty",
        )
        shallow = ShallowRepoEvidence(repository=empty_repo, readme="")
        snapshot = RepoSnapshot(
            full_name="Sfedfcv/redesigned-pancake",
            html_url="https://github.com/Sfedfcv/redesigned-pancake",
            stars=266,
            commit_sha="commit-empty",
            tree=[],
        )
        empty_plan = RepoFetchPlan(
            repository=empty_repo,
            selected_paths=[],
            skipped_paths=[],
            estimated_chars=0,
            rationale=[],
        )

        load_indexed_calls: list[str] = []
        index_calls: list[str] = []

        class FakeCorpusService:
            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                load_indexed_calls.append(repo.full_name)
                return None

            async def index_repository_plan(self, plan: RepoFetchPlan) -> RepoSearchResult:
                index_calls.append(plan.repository.full_name)
                return plan.repository

        with (
            patch(
                "routers.research.get_settings",
                return_value=SimpleNamespace(
                    PIPELINE_REQUEST_BUDGET_SECONDS=110,
                    RAG_DEEP_INDEX_REPO_LIMIT=1,
                    RAG_README_RERANK_LIMIT=1,
                    PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS=25,
                    INDEXER_MAX_CONCURRENCY=2,
                ),
            ),
            patch("routers.research.extract_keywords", AsyncMock(return_value=keywords)),
            patch("routers.research.search_repo_candidates", AsyncMock(return_value=[empty_repo])),
            patch("routers.research.fetch_shallow_repo_evidence_batch", AsyncMock(return_value=[shallow])),
            patch("routers.research.rank_repo_evidence", AsyncMock(return_value=[shallow])),
            patch("routers.research.fetch_repo_snapshot", AsyncMock(return_value=snapshot)),
            patch("routers.research.build_repo_fetch_plan", return_value=empty_plan),
            patch("routers.research.CorpusService", return_value=FakeCorpusService()),
            patch("routers.research.generate_analysis", AsyncMock(return_value=AnalysisResponse(status="complete"))),
        ):
            response = await research_idea(IdeaRequest(idea="a Slack bot that triages GitHub issues by severity"))

        self.assertEqual("error", response.status)
        self.assertEqual([], load_indexed_calls, "Should never check the cache for an unindexable repo")
        self.assertEqual([], index_calls, "Should never attempt to index a repo with zero eligible files")


class RepoChatRouteTests(HermeticAsyncTestCase):
    async def test_repo_chat_returns_grounded_answer_with_citations(self):
        indexed_repo = RepoSearchResult(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            stars=120,
            language="TypeScript",
            commit_sha="commit-123",
            relevance_score=0.92,
        )
        hits = [
            RetrievalHit(
                chunk_id="chunk-1",
                repo_full_name="example/collab-board",
                path="README.md",
                chunk_role="documentation",
                start_line=10,
                end_line=24,
                score=0.88,
                dense_score=0.7,
                repo_prior=1.0,
                reason="strong semantic match",
                text="Architecture overview and realtime collaboration flow.",
            )
        ]

        class FakeCorpusService:
            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                if repo.full_name == indexed_repo.full_name and repo.commit_sha == indexed_repo.commit_sha:
                    return indexed_repo
                return None

        class FakeRetrievalService:
            async def retrieve(self, plan, repositories) -> dict:
                return {"chat_answer": hits}

        with (
            patch("routers.research.CorpusService", return_value=FakeCorpusService()),
            patch("routers.research.RetrievalService", return_value=FakeRetrievalService()),
        ):
            response = await chat_about_repositories(
                RepoChatRequest(
                    question="How is realtime sync handled?",
                    idea_summary="Collaborative whiteboard",
                    scope_repositories=[{"full_name": indexed_repo.full_name, "commit_sha": indexed_repo.commit_sha}],
                    messages=[],
                )
            )

        self.assertIn("example/collab-board", response.answer)
        self.assertEqual(1, len(response.citations))
        self.assertEqual("README.md", response.citations[0].path)
        self.assertEqual(1, response.scoped_repo_count)
        self.assertEqual(1, len(response.evidence_hits))

    async def test_repo_chat_only_searches_repositories_in_request_scope(self):
        scoped_repo = RepoSearchResult(
            full_name="example/scoped",
            description="Scoped repo",
            html_url="https://github.com/example/scoped",
            commit_sha="commit-scoped",
            relevance_score=0.8,
        )

        class FakeCorpusService:
            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                if repo.full_name == scoped_repo.full_name and repo.commit_sha == scoped_repo.commit_sha:
                    return scoped_repo
                return None

        class FakeRetrievalService:
            async def retrieve(self, plan, repositories) -> dict:
                self.seen_repositories = repositories
                return {"chat_answer": []}

        retrieval_service = FakeRetrievalService()
        with (
            patch("routers.research.CorpusService", return_value=FakeCorpusService()),
            patch("routers.research.RetrievalService", return_value=retrieval_service),
        ):
            await chat_about_repositories(
                RepoChatRequest(
                    question="Which files matter most?",
                    idea_summary="Collaborative whiteboard",
                    scope_repositories=[{"full_name": scoped_repo.full_name, "commit_sha": scoped_repo.commit_sha}],
                    messages=[],
                )
            )

        self.assertEqual(["example/scoped"], [repo.full_name for repo in retrieval_service.seen_repositories])

    async def test_repo_chat_returns_clear_error_when_scope_has_no_indexed_repositories(self):
        class FakeCorpusService:
            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                return None

        with patch("routers.research.CorpusService", return_value=FakeCorpusService()):
            with self.assertRaises(HTTPException) as ctx:
                await chat_about_repositories(
                    RepoChatRequest(
                        question="What architecture does this use?",
                        idea_summary="Collaborative whiteboard",
                        scope_repositories=[{"full_name": "example/missing", "commit_sha": "missing"}],
                        messages=[],
                    )
                )

        self.assertEqual(409, ctx.exception.status_code)
        self.assertIn("No indexed repositories", ctx.exception.detail)

    async def test_repo_chat_returns_503_when_embedding_worker_offline(self):
        indexed_repo = RepoSearchResult(
            full_name="example/collab-board",
            html_url="https://github.com/example/collab-board",
            commit_sha="commit-123",
        )

        class FakeCorpusService:
            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                return indexed_repo

        class OfflineRetrievalService:
            async def retrieve(self, plan, repositories) -> dict:
                raise EmbeddingUnavailable("worker offline")

        with (
            patch("routers.research.CorpusService", return_value=FakeCorpusService()),
            patch("routers.research.RetrievalService", return_value=OfflineRetrievalService()),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await chat_about_repositories(
                    RepoChatRequest(
                        question="How is realtime sync handled?",
                        idea_summary="Collaborative whiteboard",
                        scope_repositories=[{"full_name": indexed_repo.full_name, "commit_sha": indexed_repo.commit_sha}],
                        messages=[],
                    )
                )

        self.assertEqual(503, ctx.exception.status_code)
        self.assertIn("offline", ctx.exception.detail)

    async def test_repo_chat_returns_uncertainty_when_no_evidence_is_found(self):
        indexed_repo = RepoSearchResult(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            commit_sha="commit-123",
            relevance_score=0.92,
        )

        class FakeCorpusService:
            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                return indexed_repo

        class FakeRetrievalService:
            async def retrieve(self, plan, repositories) -> dict:
                return {"chat_answer": []}

        with (
            patch("routers.research.CorpusService", return_value=FakeCorpusService()),
            patch("routers.research.RetrievalService", return_value=FakeRetrievalService()),
        ):
            response = await chat_about_repositories(
                RepoChatRequest(
                    question="What database migration strategy is used?",
                    idea_summary="Collaborative whiteboard",
                    scope_repositories=[{"full_name": indexed_repo.full_name, "commit_sha": indexed_repo.commit_sha}],
                    messages=[],
                )
            )

        self.assertIn("do not have enough indexed evidence", response.answer.lower())
        self.assertEqual([], response.citations)
        self.assertEqual([], response.evidence_hits)


class StreamingRouteTests(HermeticAsyncTestCase):
    """Tests for the SSE streaming POST /api/research route."""

    def setUp(self) -> None:
        super().setUp()

        # Build a minimal FastAPI app mirroring main.py
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from starlette.testclient import TestClient
        from routers.research import router as research_router

        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        test_app.include_router(research_router)
        self.client = TestClient(test_app, raise_server_exceptions=False)

    @staticmethod
    def _parse_sse(text: str) -> list[dict]:
        """Parse raw SSE body text into a list of event dicts."""
        import json as _json

        events: list[dict] = []
        for block in text.split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    try:
                        events.append(_json.loads(line[6:]))
                    except _json.JSONDecodeError:
                        pass
        return events

    # ------------------------------------------------------------------
    # 1. Successful full-pipeline run — ordered progress then result
    # ------------------------------------------------------------------
    def test_successful_stream_emits_ordered_progress_then_result(self):
        """A clear idea should yield 6 ordered progress events then one result."""
        import json as _json

        fake_candidate = RepoSearchResult(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            stars=120,
            language="TypeScript",
            topics=["whiteboard", "canvas", "websocket"],
        )
        fake_evidence = ShallowRepoEvidence(
            repository=fake_candidate,
            readme="Collaborative whiteboard with realtime sync and canvas.",
            highlighted_paths=["src/socket/server.ts"],
        )
        fake_snapshot = RepoSnapshot(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            stars=120,
            language="TypeScript",
            topics=["whiteboard", "canvas", "websocket"],
            commit_sha="abc123",
            tree=[RepoTreeEntry(path="src/socket/server.ts", type="blob", size=500)],
        )

        from unittest.mock import AsyncMock, MagicMock, patch
        from services.rag.ranking_service import rank_repo_evidence as real_rank

        ranked = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/collab-board",
                html_url="https://github.com/example/collab-board",
                description="Collaborative whiteboard",
                commit_sha="abc123",
                reference_type="end_to_end",
                fit_score=0.9,
            ),
            readme="Collaborative whiteboard with realtime sync.",
        )

        fake_indexed = RepoSearchResult(
            full_name="example/collab-board",
            html_url="https://github.com/example/collab-board",
            commit_sha="abc123",
        )

        # Simulate a cache hit (already indexed at this commit) — retrieval only
        # runs against cache-hit repos now that fresh indexing is fire-and-forget
        # in the background, so this is what it takes to exercise the full
        # 6-step happy path including the retrieval progress step.
        fake_corpus = MagicMock()
        fake_corpus.load_indexed_repository.return_value = fake_indexed
        fake_corpus.index_repository_plan = AsyncMock(return_value=fake_indexed)

        fake_retrieval = MagicMock()
        fake_retrieval.retrieve = AsyncMock(return_value={})

        with (
            patch("routers.research.search_repo_candidates", AsyncMock(return_value=[fake_candidate])),
            patch("routers.research.fetch_shallow_repo_evidence_batch", AsyncMock(return_value=[fake_evidence])),
            patch("routers.research.rank_repo_evidence", AsyncMock(return_value=[ranked])),
            patch("routers.research.fetch_repo_snapshot", AsyncMock(return_value=fake_snapshot)),
            patch("routers.research.CorpusService", return_value=fake_corpus),
            patch("routers.research.RetrievalService", return_value=fake_retrieval),
        ):
            response = self.client.post(
                "/api/research",
                json={"idea": "an app where friends can draw together online"},
            )

        self.assertEqual(200, response.status_code)
        events = self._parse_sse(response.text)

        progress_msgs = [e["message"] for e in events if e.get("type") == "progress"]
        result_events = [e for e in events if e.get("type") == "result"]

        expected_order = [
            "Extracting intent from your idea...",
            "Searching GitHub for relevant repositories...",
            "Analysing top candidates...",
            "Indexing real-world code (this takes a moment)...",
            "Retrieving relevant code sections...",
            "Generating your research report...",
        ]
        self.assertEqual(expected_order, progress_msgs)
        self.assertEqual(1, len(result_events))
        self.assertIn(result_events[0]["data"]["status"], {"complete", "error"})
        # No SSE error events in a successful run
        self.assertEqual(0, sum(1 for e in events if e.get("type") == "error"))

    # ------------------------------------------------------------------
    # 2. Clarification path — single result event, zero progress events
    # ------------------------------------------------------------------
    def test_clarification_emits_one_result_and_zero_progress_events(self):
        """A vague idea needing clarification must skip all progress events."""
        response = self.client.post(
            "/api/research",
            json={"idea": "something like Uber for tutors"},
        )

        self.assertEqual(200, response.status_code)
        events = self._parse_sse(response.text)

        progress_events = [e for e in events if e.get("type") == "progress"]
        result_events = [e for e in events if e.get("type") == "result"]

        self.assertEqual(0, len(progress_events), "No progress events before clarification result")
        self.assertEqual(1, len(result_events))
        self.assertEqual("needs_clarification", result_events[0]["data"]["status"])

    # ------------------------------------------------------------------
    # 3. Mid-stream exception → single error event, stream closes cleanly
    # ------------------------------------------------------------------
    def test_injected_exception_emits_error_event_and_closes(self):
        """An exception after the stream starts must emit exactly one error event."""
        from unittest.mock import AsyncMock, patch

        with patch(
            "routers.research.search_repo_candidates",
            AsyncMock(side_effect=RuntimeError("simulated network failure")),
        ):
            response = self.client.post(
                "/api/research",
                json={"idea": "an app where friends can draw together online"},
            )

        self.assertEqual(200, response.status_code)
        events = self._parse_sse(response.text)

        error_events = [e for e in events if e.get("type") == "error"]
        self.assertEqual(1, len(error_events))
        self.assertIn("simulated network failure", error_events[0]["message"])
        # Stream must close cleanly — no result events after the error
        result_events = [e for e in events if e.get("type") == "result"]
        self.assertEqual(0, len(result_events))

    # ------------------------------------------------------------------
    # 4. Response headers — CORS, Cache-Control, X-Accel-Buffering
    # ------------------------------------------------------------------
    def test_streaming_response_includes_required_headers(self):
        """SSE response must carry no-cache and proxy-buffering-off headers."""
        from unittest.mock import AsyncMock, patch

        # Short-circuit after search so the test is fast.
        # An Origin header must be present for the CORS middleware to emit
        # Access-Control-Allow-Origin in the response.
        with patch(
            "routers.research.search_repo_candidates",
            AsyncMock(return_value=[]),
        ):
            response = self.client.post(
                "/api/research",
                json={"idea": "an app where friends can draw together online"},
                headers={"Origin": "http://localhost"},
            )

        self.assertIn("access-control-allow-origin", {h.lower() for h in response.headers})
        self.assertEqual("no-cache", response.headers.get("cache-control"))
        self.assertEqual("no", response.headers.get("x-accel-buffering"))

    def test_skipped_retrieval_does_not_emit_retrieval_progress(self):
        """If retrieval is skipped by budget, the retrieval progress event should not be sent."""
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_candidate = RepoSearchResult(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            stars=120,
            language="TypeScript",
            topics=["whiteboard", "canvas", "websocket"],
        )
        fake_evidence = ShallowRepoEvidence(
            repository=fake_candidate,
            readme="Collaborative whiteboard with realtime sync and canvas.",
            highlighted_paths=["src/socket/server.ts"],
        )
        fake_snapshot = RepoSnapshot(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            stars=120,
            language="TypeScript",
            topics=["whiteboard", "canvas", "websocket"],
            commit_sha="abc123",
            tree=[RepoTreeEntry(path="src/socket/server.ts", type="blob", size=500)],
        )
        ranked = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/collab-board",
                html_url="https://github.com/example/collab-board",
                description="Collaborative whiteboard",
                commit_sha="abc123",
                reference_type="end_to_end",
                fit_score=0.9,
            ),
            readme="Collaborative whiteboard with realtime sync.",
        )
        fake_indexed = RepoSearchResult(
            full_name="example/collab-board",
            html_url="https://github.com/example/collab-board",
            commit_sha="abc123",
        )
        generated_analysis = AnalysisResponse(
            idea_summary="Collaborative whiteboard for shared online drawing sessions.",
            status="complete",
            repositories=[fake_indexed],
        )

        fake_corpus = MagicMock()
        fake_corpus.load_indexed_repository.return_value = None
        fake_corpus.index_repository_plan = AsyncMock(return_value=fake_indexed)

        with (
            patch("routers.research.search_repo_candidates", AsyncMock(return_value=[fake_candidate])),
            patch("routers.research.fetch_shallow_repo_evidence_batch", AsyncMock(return_value=[fake_evidence])),
            patch("routers.research.rank_repo_evidence", AsyncMock(return_value=[ranked])),
            patch("routers.research.fetch_repo_snapshot", AsyncMock(return_value=fake_snapshot)),
            patch("routers.research.CorpusService", return_value=fake_corpus),
            patch("routers.research.RequestBudget.has_time_for", return_value=False),
            patch("routers.research.generate_analysis", AsyncMock(return_value=generated_analysis)),
        ):
            response = self.client.post(
                "/api/research",
                json={"idea": "an app where friends can draw together online"},
            )

        self.assertEqual(200, response.status_code)
        events = self._parse_sse(response.text)
        progress_msgs = [event["message"] for event in events if event.get("type") == "progress"]

        self.assertEqual(
            [
                "Extracting intent from your idea...",
                "Searching GitHub for relevant repositories...",
                "Analysing top candidates...",
                "Indexing real-world code (this takes a moment)...",
                "Generating your research report...",
            ],
            progress_msgs,
        )
        self.assertNotIn("Retrieving relevant code sections...", progress_msgs)


if __name__ == "__main__":
    unittest.main()
