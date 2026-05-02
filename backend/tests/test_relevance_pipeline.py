import hashlib
import re
import sys
import unittest
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
    RepoSearchResult,
    RepoSnapshot,
    RetrievalHit,
    RetrievalPlan,
    ShallowRepoEvidence,
)
from routers.research import chat_about_repositories, research_idea
import services.github_service as github_service_module
from services.github_service import (
    build_repo_fetch_plan,
    build_search_queries,
    fetch_repo_snapshot,
    fetch_shallow_repo_evidence,
    search_repo_candidates,
)
from services.llm_service import build_clarification_questions, extract_keywords, generate_analysis, plan_retrieval_queries
from services.pipeline_cache_service import clear_memory_caches
from services.rag.ranking_service import rank_repo_evidence
from services.rag.retrieval_service import RetrievalService


class FakeEmbeddingService:
    """Fast deterministic embedding service for tests."""

    @property
    def embedding_model_name(self) -> str:
        return "fake-embedding"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

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

    async def calculate_ranking_weights(self, idea: str, repositories_count: int) -> dict:
        return {
            "readme_semantic": 0.45,
            "metadata_semantic": 0.2,
            "capability_coverage": 0.2,
            "doc_quality": 0.07,
            "query_diversity": 0.04,
            "star_quality": 0.04,
        }

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

    async def determine_retrieval_weights(self, query_type: str, context: str) -> dict:
        return {
            "dense_weight": 0.5,
            "lexical_weight": 0.25,
            "repo_weight": 0.15,
            "role_weight": 0.1,
        }


class HermeticAsyncTestCase(unittest.IsolatedAsyncioTestCase):
    """Base class that patches live LLM and embedding dependencies."""

    def setUp(self) -> None:
        self.fake_llm = FakeLLMClient()
        self.patchers = [
            patch("services.llm_service.get_llm_client", return_value=self.fake_llm),
            patch("services.llm_client.get_llm_client", return_value=self.fake_llm),
            patch("services.github_service.EmbeddingService", FakeEmbeddingService),
            patch("services.rag.ranking_service.EmbeddingService", FakeEmbeddingService),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        clear_memory_caches()
        github_service_module._cache.clear()


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

    async def test_generic_features_are_trivial_by_default(self):
        keywords = await extract_keywords(
            "A travel planner with maps, chat, recommendations, login, notifications, and file uploads"
        )
        question_keys = {question.key for question in build_clarification_questions(keywords)}

        self.assertIn("feature_priority", question_keys)
        self.assertIn("file uploads", keywords.trivial_capabilities)
        self.assertIn("notifications", keywords.trivial_capabilities)
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
                            "stargazers_count": 52000,
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
        self.assertEqual("example/collab-board", results[0].full_name)
        self.assertGreater(results[0].semantic_meta_score, 0.0)
        self.assertGreaterEqual(results[0].query_hit_count, 1)
        self.assertIn("concept-family coverage", " | ".join(results[0].rank_reasons))

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

        self.assertEqual("example/pantry-meals", ranked[0].repository.full_name)

    async def test_ranking_dedupes_similar_repos_and_keeps_language_diversity(self):
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
                topics=["travel", "chat", "maps"],
                query_hit_count=3,
                semantic_meta_score=0.8,
            ),
            readme=(
                "Travel planner with chat messaging, maps integration, recommendation engine, and shared itineraries."
            ),
            highlighted_paths=["README.md", "package.json"],
        )
        repo_b = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-chat-b",
                description="Travel planner with maps chat and itineraries",
                html_url="https://github.com/example/travel-chat-b",
                stars=100,
                language="TypeScript",
                topics=["travel", "chat", "maps"],
                query_hit_count=2,
                semantic_meta_score=0.79,
            ),
            readme=(
                "Travel planner with chat messaging, maps integration, recommendation engine, and shared itineraries."
            ),
            highlighted_paths=["README.md", "package.json"],
        )
        repo_c = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/travel-python",
                description="Travel planner backend with recommendations and maps",
                html_url="https://github.com/example/travel-python",
                stars=90,
                language="Python",
                topics=["travel", "recommendation", "maps"],
                query_hit_count=2,
                semantic_meta_score=0.75,
            ),
            readme=(
                "Travel planner backend with maps integration, recommendation engine, and itinerary persistence."
            ),
            highlighted_paths=["README.md", "requirements.txt"],
        )

        ranked = await rank_repo_evidence(
            "A travel planner with chat, maps, and personalized recommendations",
            keywords,
            [repo_a, repo_b, repo_c],
        )
        full_names = [evidence.repository.full_name for evidence in ranked]

        self.assertIn("example/travel-chat-a", full_names)
        self.assertIn("example/travel-python", full_names)
        self.assertNotIn("example/travel-chat-b", full_names)

    async def test_ranking_uses_sampled_code_to_rescue_vague_but_relevant_repo(self):
        keywords = await extract_keywords("something like Uber for tutors")

        vague_but_relevant = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="example/tutor-marketplace",
                description="Marketplace backend for services",
                html_url="https://github.com/example/tutor-marketplace",
                stars=220,
                language="Python",
                topics=["marketplace", "booking", "payments"],
            ),
            readme="Service marketplace with accounts, scheduling modules, and API endpoints.",
            sampled_files=[
                RepoFile(
                    path="services/booking/payments.py",
                    content=(
                        "def create_lesson_checkout(tutor_id, student_id): "
                        "schedule lesson booking, tutor discovery, payout invoice, checkout session"
                    ),
                    size=124,
                )
            ],
            sampled_paths=["services/booking/payments.py"],
            highlighted_paths=["services/booking/payments.py"],
        )
        generic_starter = ShallowRepoEvidence(
            repository=RepoSearchResult(
                full_name="popular/marketplace-starter",
                description="Marketplace starter kit",
                html_url="https://github.com/popular/marketplace-starter",
                stars=5400,
                language="TypeScript",
                topics=["starter", "dashboard", "marketplace"],
            ),
            readme="Starter kit for admin dashboards, CMS pages, analytics, and themes.",
            sampled_files=[
                RepoFile(
                    path="src/admin/dashboard.ts",
                    content="render charts reports theme admin analytics tables tenant settings",
                    size=72,
                )
            ],
            sampled_paths=["src/admin/dashboard.ts"],
            highlighted_paths=["src/admin/dashboard.ts"],
        )

        ranked = await rank_repo_evidence(
            "something like Uber for tutors",
            keywords,
            [generic_starter, vague_but_relevant],
        )

        self.assertEqual("example/tutor-marketplace", ranked[0].repository.full_name)
        self.assertGreater(ranked[0].semantic_code_score, 0.0)
        self.assertIn("sampled code semantic score", " | ".join(ranked[0].score_reasons))

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
        self.assertEqual(
            [
                "example/travel-planner-a",
                "example/travel-planner-b",
                "example/travel-planner-c",
            ],
            top_three_names,
        )
        self.assertTrue(all(evidence.repository.reference_type == "end_to_end" for evidence in ranked[:3]))


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
        with patch("services.github_service.get_github_client", return_value=client):
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
        package_payload = {
            "encoding": "base64",
            "content": b64encode(b'{"name":"collab-board"}').decode("ascii"),
        }

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
                if url.endswith("/contents/package.json"):
                    return FakeResponse(status_code=200, json_data=package_payload)
                return FakeResponse(status_code=404, json_data={})

        client = CountingGitHubClient()
        with patch("services.github_service.get_github_client", return_value=client):
            first = await fetch_shallow_repo_evidence(repository)
            first_call_count = len(client.calls)
            second = await fetch_shallow_repo_evidence(repository)

        self.assertEqual(first.readme, second.readme)
        self.assertEqual(first.highlighted_paths, second.highlighted_paths)
        self.assertGreaterEqual(first_call_count, 1)
        self.assertEqual(first_call_count, len(client.calls))


class ResearchPipelineOrchestrationTests(HermeticAsyncTestCase):
    async def test_research_request_enqueues_background_indexing_without_blocking(self):
        keywords = ExtractedKeywords(
            summary="Collaborative whiteboard for shared online drawing sessions.",
            core_intent="Build a realtime shared canvas experience for multiple users.",
            product_type="collaborative whiteboard",
            capabilities=["realtime collaboration", "canvas rendering", "presence"],
            primary_capabilities=["realtime collaboration", "canvas rendering", "presence"],
            keywords=["whiteboard", "canvas", "realtime"],
        )
        repository = RepoSearchResult(
            full_name="example/collab-board",
            description="Collaborative whiteboard",
            html_url="https://github.com/example/collab-board",
            stars=120,
            language="TypeScript",
            topics=["whiteboard", "canvas", "websocket"],
        )
        secondary_repository = RepoSearchResult(
            full_name="example/secondary-board",
            description="Secondary whiteboard reference",
            html_url="https://github.com/example/secondary-board",
            stars=80,
            language="TypeScript",
            topics=["whiteboard", "multiplayer"],
        )
        snapshot = RepoSnapshot(
            **repository.model_dump(exclude={"commit_sha"}),
            commit_sha="commit-123",
            tree=[],
        )
        secondary_snapshot = RepoSnapshot(
            **secondary_repository.model_dump(exclude={"commit_sha"}),
            commit_sha="commit-456",
            tree=[],
        )
        fetch_plan = RepoFetchPlan(
            repository=RepoSearchResult(**{**repository.model_dump(), "commit_sha": "commit-123"}),
            selected_paths=["README.md", "src/socket/server.ts"],
            skipped_paths=[],
            estimated_chars=500,
            rationale=["Prioritized README and realtime code paths."],
        )
        secondary_fetch_plan = RepoFetchPlan(
            repository=RepoSearchResult(**{**secondary_repository.model_dump(), "commit_sha": "commit-456"}),
            selected_paths=["README.md", "src/canvas/index.ts"],
            skipped_paths=[],
            estimated_chars=420,
            rationale=["Prioritized collaborative canvas paths."],
        )
        shallow = ShallowRepoEvidence(
            repository=RepoSearchResult(
                **repository.model_dump(),
                reference_type="end_to_end",
                fit_score=0.91,
                fit_summary="It behaves like an end to end reference covering realtime collaboration and canvas rendering.",
                covered_primary=["realtime collaboration", "canvas rendering"],
                missing_primary=["presence"],
                semantic_readme_score=0.82,
                relevance_score=0.91,
                rank_reasons=["covers primary capabilities: realtime collaboration, canvas rendering"],
            ),
            readme="Collaborative whiteboard with realtime collaboration and canvas rendering.",
        )
        secondary_shallow = ShallowRepoEvidence(
            repository=RepoSearchResult(
                **secondary_repository.model_dump(),
                reference_type="subsystem",
                fit_score=0.73,
                fit_summary="It behaves like a subsystem reference covering canvas rendering.",
                covered_primary=["canvas rendering"],
                missing_primary=["realtime collaboration", "presence"],
                semantic_readme_score=0.67,
                relevance_score=0.73,
                rank_reasons=["covers primary capabilities: canvas rendering"],
            ),
            readme="Collaborative canvas UI focused on drawing interactions.",
        )
        generated_analysis = AnalysisResponse(
            idea_summary=keywords.summary,
            keywords=keywords,
            repositories=[repository],
            repo_descriptions=["example/collab-board is useful because it demonstrates realtime collaboration."],
            learning_path=[],
            architecture_diagram="graph TD; User-->App;",
            tech_stack=[],
            status="complete",
        )

        class FakeCorpusService:
            queued: list[str] = []
            indexed: list[str] = []

            def __init__(self) -> None:
                self.queued = []
                type(self).queued = self.queued
                self.indexed = []
                type(self).indexed = self.indexed

            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                return None

            def enqueue_index_job_placeholder(self, plan: RepoFetchPlan) -> bool:
                self.queued.append(f"{plan.repository.full_name}@{plan.repository.commit_sha}")
                return True

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
                    RAG_DEEP_INDEX_REPO_LIMIT=2,
                    RAG_README_RERANK_LIMIT=2,
                    INDEXING_MODE="background",
                    RAG_INLINE_BOOTSTRAP_REPO_LIMIT=1,
                    PIPELINE_RETRIEVAL_MIN_BUDGET_SECONDS=25,
                    INDEXER_MAX_CONCURRENCY=2,
                ),
            ),
            patch("routers.research.extract_keywords", AsyncMock(return_value=keywords)),
            patch("routers.research.search_repo_candidates", AsyncMock(return_value=[repository, secondary_repository])),
            patch("routers.research.fetch_shallow_repo_evidence_batch", AsyncMock(return_value=[shallow, secondary_shallow])),
            patch("routers.research.rank_repo_evidence", AsyncMock(return_value=[shallow, secondary_shallow])),
            patch("routers.research.fetch_repo_snapshot", AsyncMock(side_effect=[snapshot, secondary_snapshot])),
            patch("routers.research.build_repo_fetch_plan", side_effect=[fetch_plan, secondary_fetch_plan]),
            patch("routers.research.CorpusService", FakeCorpusService),
            patch("routers.research.plan_retrieval_queries", AsyncMock(return_value=RetrievalPlan(queries=[]))) as plan_mock,
            patch("routers.research.RetrievalService", return_value=FakeRetrievalService()),
            patch("routers.research.generate_analysis", AsyncMock(return_value=generated_analysis)) as generate_mock,
        ):
            response = await research_idea(IdeaRequest(idea="an app where friends can draw together online"))

        self.assertEqual("complete", response.status)
        self.assertEqual(["example/secondary-board@commit-456"], FakeCorpusService.queued)
        self.assertEqual(["example/collab-board@commit-123"], FakeCorpusService.indexed)
        plan_mock.assert_awaited_once()
        generate_mock.assert_awaited_once()
        self.assertEqual({}, generate_mock.await_args.args[3])

    async def test_research_request_inlines_five_repositories_when_bootstrap_limit_is_five(self):
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
                repository=RepoSearchResult(
                    **repository.model_dump(),
                    reference_type="end_to_end" if index == 0 else "subsystem",
                    fit_score=round(0.92 - (index * 0.06), 2),
                    fit_summary="It behaves like a strong collaborative drawing reference.",
                    covered_primary=["realtime collaboration", "canvas rendering"],
                    missing_primary=["presence"] if index < 3 else [],
                    semantic_readme_score=round(0.86 - (index * 0.04), 2),
                    relevance_score=round(0.92 - (index * 0.06), 2),
                    rank_reasons=["covers primary capabilities: realtime collaboration, canvas rendering"],
                ),
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
            queued: list[str] = []
            indexed: list[str] = []

            def __init__(self) -> None:
                self.queued = []
                type(self).queued = self.queued
                self.indexed = []
                type(self).indexed = self.indexed

            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                return None

            def enqueue_index_job_placeholder(self, plan: RepoFetchPlan) -> bool:
                self.queued.append(f"{plan.repository.full_name}@{plan.repository.commit_sha}")
                return True

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
                    INDEXING_MODE="background",
                    RAG_INLINE_BOOTSTRAP_REPO_LIMIT=5,
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

        self.assertEqual("complete", response.status)
        self.assertEqual([], FakeCorpusService.queued)
        self.assertEqual(
            [f"example/collab-board-{index}@commit-{index}" for index in range(5)],
            FakeCorpusService.indexed,
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
        ranked_repository = RepoSearchResult(
            **repository.model_dump(),
            reference_type="end_to_end",
            fit_score=0.88,
            fit_summary="It behaves like an end to end reference covering realtime collaboration and canvas rendering.",
            covered_primary=["realtime collaboration", "canvas rendering"],
            missing_primary=["presence"],
            semantic_meta_score=0.71,
            semantic_readme_score=0.81,
            relevance_score=0.88,
            query_hit_count=3,
            rank_reasons=["covers primary capabilities: realtime collaboration, canvas rendering"],
        )
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
        cached_repository = RepoSearchResult(
            **{**repository.model_dump(), "commit_sha": "commit-cached"},
            reference_type="candidate",
            fit_score=0.0,
            fit_summary="",
            covered_primary=[],
            missing_primary=[],
            relevance_score=0.0,
        )
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
                    INDEXING_MODE="background",
                    RAG_INLINE_BOOTSTRAP_REPO_LIMIT=1,
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
                lexical_score=0.6,
                repo_prior=1.0,
                role_prior=1.0,
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

    def test_chat_answer_role_prior_favors_docs_config_and_entrypoints(self):
        retrieval_service = RetrievalService.__new__(RetrievalService)

        documentation_score = retrieval_service._role_prior("documentation", ["documentation"], "chat_answer", "README.md")
        config_score = retrieval_service._role_prior("config", ["config"], "chat_answer", "requirements.txt")
        entrypoint_score = retrieval_service._role_prior("entrypoint", ["source"], "chat_answer", "src/server/main.py")

        self.assertGreaterEqual(documentation_score, 1.0)
        self.assertGreaterEqual(config_score, 1.0)
        self.assertGreaterEqual(entrypoint_score, 0.95)


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
            tree=[],
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

        fake_corpus = MagicMock()
        fake_corpus.load_indexed_repository.return_value = None
        fake_corpus.enqueue_index_job_placeholder.return_value = False
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
            tree=[],
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
        fake_corpus.enqueue_index_job_placeholder.return_value = False
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
