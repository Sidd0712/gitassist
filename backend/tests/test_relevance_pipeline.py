import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.schemas import RepoSearchResult, RepoSnapshot, ShallowRepoEvidence
from services.github_service import build_repo_fetch_plan, build_search_queries
from services.llm_service import build_clarification_questions, extract_keywords, generate_analysis
from services.rag.ranking_service import rank_repo_evidence


class IntentExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_layman_whiteboard_prompt_is_expanded_without_clarification(self):
        keywords = await extract_keywords("an app where friends can draw together online")

        self.assertIn("realtime collaboration", keywords.capabilities)
        self.assertIn("canvas rendering", keywords.capabilities)
        self.assertEqual([], build_clarification_questions(keywords))

    async def test_vague_marketplace_prompt_requests_clarification(self):
        keywords = await extract_keywords("something like Uber for tutors")
        question_keys = {question.key for question in build_clarification_questions(keywords)}

        self.assertIn("platform", question_keys)
        self.assertIn("payments", question_keys)

    async def test_search_queries_use_normalized_capabilities(self):
        keywords = await extract_keywords("an app where friends can draw together online")
        queries = build_search_queries(keywords)
        joined = " | ".join(queries).lower()

        self.assertIn("collaborative whiteboard", joined)
        self.assertIn("realtime collaboration", joined)

    async def test_meal_prep_prompt_preserves_food_concepts(self):
        keywords = await extract_keywords("A meal prep app that suggests recipes for everyday meals depending on what's in your fridge")
        question_keys = {question.key for question in build_clarification_questions(keywords)}
        joined_queries = " | ".join(build_search_queries(keywords)).lower()

        self.assertIn("ingredient inventory", keywords.capabilities)
        self.assertIn("recipe recommendation", keywords.capabilities)
        self.assertIn("meal planning", keywords.capabilities)
        self.assertNotIn("ai features", keywords.capabilities)
        self.assertTrue({"platform", "inventory_input", "recommendation_mode"}.issubset(question_keys))
        self.assertIn("recipe", joined_queries)
        self.assertTrue("fridge" in joined_queries or "pantry" in joined_queries)


class RankingAndGenerationTests(unittest.IsolatedAsyncioTestCase):
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
            readme="Collaborative whiteboard with canvas rendering, realtime sync, presence, rooms, and auth.",
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

        ranked = rank_repo_evidence("Build a real-time collaborative whiteboard with WebSocket and canvas", keywords, [generic, matched])

        self.assertEqual("example/collab-board", ranked[0].repository.full_name)
        self.assertIn(ranked[0].repository.reference_type, {"end_to_end", "subsystem"})

    async def test_generated_architecture_does_not_leak_internal_rag_terms(self):
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
        ranked_payload = snapshot.model_dump(exclude={"tree", "search_queries", "reference_type", "fit_score", "fit_summary", "rank_reasons"})
        shallow = ShallowRepoEvidence(
            repository=RepoSearchResult(
                **ranked_payload,
                reference_type="end_to_end",
                fit_score=0.87,
                fit_summary="It behaves like an end to end reference covering ingredient inventory and meal planning.",
                rank_reasons=["capabilities: ingredient inventory, meal planning"],
            ),
            readme_path="README.md",
            readme="Meal planning, ingredients, recipe recommendations.",
        )

        plan = build_repo_fetch_plan(snapshot, shallow, keywords)

        self.assertEqual("end_to_end", plan.repository.reference_type)
        self.assertGreater(plan.repository.fit_score, 0.8)
        self.assertIn("ingredient inventory", plan.repository.fit_summary)

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
            readme="AI meal planner with OpenAI, GPT prompts, embeddings, assistant flows, and generated recipes.",
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
            readme="Ingredient inventory, pantry search, recipe matching, shopping lists, and weekly meal planning.",
            highlighted_paths=["src/features/inventory/service.ts", "src/features/recipes/matcher.ts"],
        )

        ranked = rank_repo_evidence(  # type: ignore[arg-type]
            "A meal prep app that suggests recipes for everyday meals depending on what's in your fridge",
            keywords,
            [ai_heavy, recipe_first],
        )

        self.assertEqual("example/pantry-meals", ranked[0].repository.full_name)


if __name__ == "__main__":
    unittest.main()
