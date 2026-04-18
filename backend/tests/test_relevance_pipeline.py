import hashlib
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.schemas import AnalysisResponse, ExtractedKeywords, IdeaRequest, RepoFetchPlan, RepoSearchResult, RepoSnapshot, ShallowRepoEvidence
from routers.research import research_idea
from services.github_service import build_repo_fetch_plan, build_search_queries
from services.llm_service import build_clarification_questions, extract_keywords, generate_analysis
from services.rag.ranking_service import rank_repo_evidence


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

    async def generate_repo_descriptions(self, idea: str, keywords: dict, repositories: list[dict], evidence: dict) -> list[str]:
        primary = ", ".join(keywords.get("primary_capabilities", [])[:2]) or "the core workflow"
        return [
            f"{repo['full_name']} is useful because it demonstrates {primary}."
            for repo in repositories
        ]

    async def generate_learning_path(self, idea: str, keywords: dict, repositories: list[dict]) -> list[dict]:
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

    async def generate_tech_stack(self, idea: str, keywords: dict, repositories: list[dict]) -> list[dict]:
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

    async def generate_architecture_diagram(self, idea: str, keywords: dict, repositories: list[dict]) -> str:
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

        if hasattr(rank_repo_evidence, "_ai_weights"):
            delattr(rank_repo_evidence, "_ai_weights")


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
        shallow = ShallowRepoEvidence(
            repository=repository,
            readme_path="README.md",
            readme="Collaborative whiteboard with realtime collaboration and canvas rendering.",
            highlighted_paths=["README.md", "src/socket/server.ts"],
        )
        snapshot = RepoSnapshot(
            **repository.model_dump(exclude={"commit_sha"}),
            commit_sha="commit-123",
            tree=[],
        )
        fetch_plan = RepoFetchPlan(
            repository=RepoSearchResult(**{**repository.model_dump(), "commit_sha": "commit-123"}),
            selected_paths=["README.md", "src/socket/server.ts"],
            skipped_paths=[],
            estimated_chars=500,
            rationale=["Prioritized README and realtime code paths."],
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

            def __init__(self) -> None:
                self.queued = []
                type(self).queued = self.queued

            def load_indexed_repository(self, repo: RepoSearchResult) -> RepoSearchResult | None:
                return None

            def enqueue_index_job(self, plan: RepoFetchPlan, shallow_evidence: ShallowRepoEvidence) -> bool:
                self.queued.append(f"{plan.repository.full_name}@{plan.repository.commit_sha}")
                return True

        with (
            patch("routers.research.extract_keywords", AsyncMock(return_value=keywords)),
            patch("routers.research.search_repo_candidates", AsyncMock(return_value=[repository])),
            patch("routers.research.fetch_shallow_repo_evidence_batch", AsyncMock(return_value=[shallow])),
            patch("routers.research.rank_repo_evidence", AsyncMock(return_value=[shallow])),
            patch("routers.research.fetch_repo_snapshot", AsyncMock(return_value=snapshot)),
            patch("routers.research.build_repo_fetch_plan", return_value=fetch_plan),
            patch("routers.research.CorpusService", FakeCorpusService),
            patch("routers.research.plan_retrieval_queries", AsyncMock()) as plan_mock,
            patch("routers.research.generate_analysis", AsyncMock(return_value=generated_analysis)) as generate_mock,
        ):
            response = await research_idea(IdeaRequest(idea="an app where friends can draw together online"))

        self.assertEqual("complete", response.status)
        self.assertEqual(["example/collab-board@commit-123"], FakeCorpusService.queued)
        plan_mock.assert_not_awaited()
        generate_mock.assert_awaited_once()
        self.assertEqual({}, generate_mock.await_args.args[3])


if __name__ == "__main__":
    unittest.main()
