"""Live smoke tests against the real Groq API.

Every test in test_relevance_pipeline.py mocks the LLM client, which is exactly
why a full Groq model deprecation (LLM_MODEL pointing at a retired model) went
undetected: every mock assumes the underlying API call succeeds and returns
well-shaped output, which is precisely the assumption a model retirement or a
reasoning-token budget regression breaks.

These tests make real network calls to the model configured in LLM_MODEL, one
per LLM call site in llm_client.py, so a bad model swap, a wrong
reasoning_effort setting, or a too-small max_tokens budget fails here instead
of in production. They are deliberately narrow: assert each call *succeeds*
and returns non-empty, well-shaped output within its token budget -- not that
the output is good (that's an eval-harness/human-review concern, not a smoke
test's job).

Run before shipping any change to LLM_MODEL, reasoning_effort, or max_tokens
in services/llm_client.py:

    python -m unittest tests.test_live_smoke -v

Skipped automatically if GROQ_API_KEY is not configured.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from services.llm_client import get_llm_client
from services.llm_service import build_clarification_questions, extract_keywords

_settings = get_settings()


@unittest.skipUnless(
    _settings.GROQ_API_KEY,
    "GROQ_API_KEY not configured; skipping live Groq smoke test",
)
class LiveGroqSmokeTests(unittest.IsolatedAsyncioTestCase):
    """One real call per LLM call site against the configured model."""

    async def test_extract_keywords_returns_populated_json(self):
        keywords = await extract_keywords(
            "a recipe sharing app where users can post recipes and follow other cooks"
        )
        self.assertTrue(keywords.summary.strip())
        self.assertTrue(keywords.primary_capabilities or keywords.capabilities)

    async def test_extract_keywords_feeds_clarification_gate_without_crashing(self):
        # Exercises the clarification-gate path (build_clarification_questions)
        # against real model output shape rather than a mock's. Not asserting
        # questions are non-empty -- whether this idea is ambiguous enough is
        # the model's judgment call, not this test's.
        keywords = await extract_keywords("I want to build something for freelancers")
        questions = build_clarification_questions(keywords)
        for question in questions:
            self.assertTrue(question.key)
            self.assertGreaterEqual(len(question.options), 2)

    async def test_generate_repo_descriptions_not_truncated(self):
        llm = get_llm_client()
        descriptions = await llm.generate_repo_descriptions(
            idea="a recipe sharing app",
            keywords={
                "primary_capabilities": ["post recipes", "follow cooks"],
                "product_type": "recipe sharing app",
            },
            repositories=[{"full_name": "example/recipe-app"}],
        )
        self.assertTrue(descriptions)
        self.assertTrue(all(isinstance(d, str) and d.strip() for d in descriptions))

    async def test_generate_learning_path_not_truncated(self):
        llm = get_llm_client()
        path = await llm.generate_learning_path(
            idea="a recipe sharing app",
            keywords={
                "primary_capabilities": ["post recipes"],
                "secondary_capabilities": ["meal plans"],
                "product_type": "recipe sharing app",
            },
            repositories=[{"full_name": "example/recipe-app"}],
        )
        self.assertTrue(path)
        self.assertTrue(all(isinstance(step, dict) and step.get("title") for step in path))

    async def test_generate_tech_stack_not_truncated(self):
        llm = get_llm_client()
        stack = await llm.generate_tech_stack(
            idea="a recipe sharing app",
            keywords={
                "primary_capabilities": ["post recipes"],
                "frameworks": ["React"],
                "product_type": "recipe sharing app",
            },
            repositories=[{"full_name": "example/recipe-app"}],
        )
        self.assertTrue(stack)
        self.assertTrue(all(isinstance(item, dict) and item.get("technology") for item in stack))

    async def test_generate_architecture_diagram_not_truncated(self):
        llm = get_llm_client()
        diagram = await llm.generate_architecture_diagram(
            idea="a recipe sharing app",
            keywords={
                "primary_capabilities": ["post recipes"],
                "likely_components": ["API", "database"],
                "product_type": "recipe sharing app",
            },
            repositories=[{"full_name": "example/recipe-app"}],
        )
        self.assertTrue(diagram.strip().lower().startswith("graph"))

    async def test_answer_repo_chat_returns_grounded_answer(self):
        # Exercises the /api/research/chat call site independently of the main
        # /api/research pipeline -- it uses the same LLM client but a
        # different prompt shape and was not covered by prior manual testing.
        llm = get_llm_client()
        result = await llm.answer_repo_chat(
            question="What does this repo do?",
            idea_summary="a recipe sharing app",
            messages=[],
            repositories=[{"full_name": "example/recipe-app", "description": "a recipe app"}],
            evidence={
                "repositories": [],
                "hits": [
                    {
                        "repo": "example/recipe-app",
                        "path": "README.md",
                        "lines": [1, 10],
                        "reason": "match",
                        "snippet": "This is a recipe sharing app.",
                    }
                ],
            },
        )
        self.assertTrue(str(result.get("answer", "")).strip())


if __name__ == "__main__":
    unittest.main()
