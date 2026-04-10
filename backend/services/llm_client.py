"""HuggingFace LLM client for all model-based analysis."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

from core.config import get_settings

logger = logging.getLogger(__name__)

# Thread pool for running sync model operations without blocking
_executor = ThreadPoolExecutor(max_workers=1)


class LLMClient:
    """Unified interface for HuggingFace model interactions."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def _get_client(self) -> HuggingFacePipeline:
        """Lazy-load the HuggingFace model."""
        if self._client is None:
            logger.info("Initializing HuggingFace LLM: %s", self.settings.LLM_MODEL)
            try:
                # Create the transformers pipeline first
                hf_pipeline = pipeline(
                    "text-generation",
                    model=self.settings.LLM_MODEL,
                    device_map="auto",  # Automatically use GPU if available
                    model_kwargs={
                        "temperature": 0.7,
                        "max_new_tokens": 1024,
                        "torch_dtype": "auto",
                    },
                )
                
                # Wrap in LangChain
                self._client = HuggingFacePipeline(
                    model=hf_pipeline,
                )
                logger.info("✓ HuggingFace LLM loaded successfully")
            except Exception as exc:
                logger.error("Failed to initialize HuggingFace LLM: %s", exc)
                logger.error("Make sure the model '%s' is valid", self.settings.LLM_MODEL)
                raise
        return self._client

    async def extract_keywords(self, idea: str, clarification_answers: dict[str, str] | None = None) -> dict:
        """Use model to extract technical keywords and capabilities from user idea."""
        
        clarification = ""
        if clarification_answers:
            clarification = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in clarification_answers.items())
            clarification = f"\nClarifications:\n{clarification}"

        prompt = f"""Analyze this project idea and extract technical structure in JSON format.

Project Idea:
{idea}{clarification}

Extract and return ONLY valid JSON (no markdown, no explanation):
{{
  "summary": "one-line technical summary",
  "core_intent": "core business/technical intent",
  "product_type": "category (e.g., 'social travel planner', 'collaborative whiteboard')",
  "target_users": ["user segment 1", "user segment 2"],
  "primary_capabilities": ["main capability 1", "main capability 2"],
  "secondary_capabilities": ["supporting capability 1", "supporting capability 2"],
  "keywords": ["technical keyword 1", "technical keyword 2"],
  "frameworks": ["framework 1", "framework 2"],
  "languages": ["language 1", "language 2"],
  "likely_components": ["component 1", "component 2"],
  "likely_integrations": ["integration 1", "integration 2"],
  "likely_stack_families": ["stack family 1", "stack family 2"],
  "constraints": ["constraint 1"],
  "assumptions": ["assumption 1"],
  "ambiguities": [
    {{"axis": "ambiguity name", "reason": "why it matters", "severity": "high/medium/low"}}
  ]
}}"""

        try:
            client = self._get_client()
            # Use executor to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.invoke(prompt)
            )
            
            # Parse JSON from response
            result = self._parse_json_response(response)
            logger.info("✓ Keywords extracted: %d capabilities, %d frameworks", 
                       len(result.get("primary_capabilities", [])) + len(result.get("secondary_capabilities", [])),
                       len(result.get("frameworks", [])))
            return result
        except Exception as exc:
            logger.error("Keyword extraction failed: %s", exc)
            raise

    async def build_clarification_questions(self, idea: str, extracted: dict) -> list[dict]:
        """Use model to identify key ambiguities needing clarification."""
        
        prompt = f"""Given this project idea and extracted analysis, suggest 2-3 critical clarification questions.

Idea: {idea}

Current Analysis:
- Product Type: {extracted.get('product_type', 'unknown')}
- Capabilities: {', '.join(extracted.get('primary_capabilities', []))}
- Ambiguities: {extracted.get('ambiguities', [])}

Return ONLY valid JSON array (no markdown):
[
  {{
    "key": "unique_key",
    "question": "clarification question",
    "options": ["option 1", "option 2", "option 3"],
    "reason": "why this matters"
  }}
]"""

        try:
            client = self._get_client()
            response = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.invoke(prompt)
            )
            questions = self._parse_json_response(response)
            if not isinstance(questions, list):
                questions = [questions]
            logger.info("✓ Generated %d clarification questions", len(questions))
            return questions[:4]  # Limit to 4 questions
        except Exception as exc:
            logger.error("Clarification question generation failed: %s", exc)
            return []

    async def plan_retrieval_queries(self, idea: str, keywords: dict, repositories: list[dict]) -> dict:
        """Use model to generate targeted retrieval queries for each analysis section."""
        
        repo_summary = ", ".join(r.get("full_name", "") for r in repositories[:5]) if repositories else "no repos"
        
        prompt = f"""Given a project idea and initial repositories, create retrieval queries for analysis sections.

Idea: {idea}
Key Capabilities: {', '.join(keywords.get('primary_capabilities', []))}
Reference Repos: {repo_summary}

Generate retrieval queries for: repo_descriptions, learning_path, architecture_diagram, tech_stack

Return ONLY valid JSON (no markdown):
{{
  "queries": [
    {{
      "section": "section_name",
      "query": "natural language retrieval query",
      "top_k": 8
    }}
  ]
}}"""

        try:
            client = self._get_client()
            response = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.invoke(prompt)
            )
            result = self._parse_json_response(response)
            logger.info("✓ Generated %d retrieval queries", len(result.get("queries", [])))
            return result
        except Exception as exc:
            logger.error("Retrieval planning failed: %s", exc)
            raise

    async def generate_repo_descriptions(self, idea: str, keywords: dict, repositories: list[dict], evidence: dict) -> list[str]:
        """Use model to generate natural descriptions of reference repositories."""
        
        repo_names = [r.get("full_name", "") for r in repositories[:5]]
        
        prompt = f"""Given project idea, repositories, and retrieved evidence, generate concise descriptions of why each repo matters.

Project Idea: {idea}
Primary Features: {', '.join(keywords.get('primary_capabilities', []))}
Reference Repos: {', '.join(repo_names)}

For each repository, explain:
1. End-to-end match to the project idea
2. Key architectural patterns or components relevant
3. Learning value for this specific project

Return as JSON array:
[
  "repository summary 1",
  "repository summary 2"
]

No markdown, just the JSON array."""

        try:
            client = self._get_client()
            response = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.invoke(prompt)
            )
            descriptions = self._parse_json_response(response)
            if not isinstance(descriptions, list):
                descriptions = [descriptions]
            logger.info("✓ Generated %d repository descriptions", len(descriptions))
            return descriptions
        except Exception as exc:
            logger.error("Repository description generation failed: %s", exc)
            return [f"Repository reference for {keywords.get('product_type', 'project')}"] * len(repositories)

    async def generate_learning_path(self, idea: str, keywords: dict, repositories: list[dict]) -> list[dict]:
        """Use model to generate step-by-step learning path."""
        
        primary = ", ".join(keywords.get("primary_capabilities", [])[:3])
        secondary = ", ".join(keywords.get("secondary_capabilities", [])[:3])
        
        prompt = f"""Create a practical learning path for building: {keywords.get('product_type', 'project')}

Primary Focus: {primary}
Secondary Features: {secondary}
Implementation References: {repositories[0].get('full_name', 'reference') if repositories else 'reference'} and others

Generate 6-8 concrete learning steps with milestones. Return JSON array (no markdown):
[
  {{
    "step": 1,
    "title": "step title",
    "description": "what to learn",
    "milestone": "concrete deliverable"
  }}
]"""

        try:
            client = self._get_client()
            response = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.invoke(prompt)
            )
            path = self._parse_json_response(response)
            if not isinstance(path, list):
                path = [path]
            logger.info("✓ Generated %d-step learning path", len(path))
            return path
        except Exception as exc:
            logger.error("Learning path generation failed: %s", exc)
            return []

    async def generate_tech_stack(self, idea: str, keywords: dict, repositories: list[dict]) -> list[dict]:
        """Use model to recommend technology decisions."""
        
        primary = ", ".join(keywords.get("primary_capabilities", [])[:3])
        frameworks = ", ".join(keywords.get("frameworks", [])[:5])
        
        prompt = f"""Recommend technology stack for: {keywords.get('product_type', 'project')}

Primary Capabilities: {primary}
Suggested Frameworks: {frameworks}

For each major layer (frontend, backend, database, realtime, etc.), provide:
1. Recommended technology
2. Why it matches the project needs
3. Pros for this specific use case
4. Cons/tradeoffs to consider

Return JSON array (no markdown):
[
  {{
    "layer": "layer name",
    "technology": "tech name",
    "reasoning": "why chosen",
    "pros": ["pro 1"],
    "cons": ["con 1"]
  }}
]"""

        try:
            client = self._get_client()
            response = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.invoke(prompt)
            )
            stack = self._parse_json_response(response)
            if not isinstance(stack, list):
                stack = [stack]
            logger.info("✓ Generated %d tech stack recommendations", len(stack))
            return stack
        except Exception as exc:
            logger.error("Tech stack generation failed: %s", exc)
            return []

    async def generate_architecture_diagram(self, idea: str, keywords: dict, repositories: list[dict]) -> str:
        """Use model to generate Mermaid diagram code."""
        
        primary = ", ".join(keywords.get("primary_capabilities", [])[:3])
        components = ", ".join(keywords.get("likely_components", [])[:5])
        
        prompt = f"""Generate a Mermaid diagram for: {keywords.get('product_type', 'project')}

Main Capabilities: {primary}
Key Components: {components}

Create a Mermaid flowchart showing:
- User/Client layer
- API/Backend services
- Data layer
- External integrations
- Key data flows

Return ONLY valid Mermaid syntax, no markdown:
graph TD
  ...nodes and connections..."""

        try:
            client = self._get_client()
            response = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.invoke(prompt)
            )
            # Clean response
            diagram = response.strip()
            if diagram.startswith("```"):
                diagram = "\n".join(line for line in diagram.split("\n") if not line.startswith("```"))
            logger.info("✓ Generated architecture diagram")
            return diagram
        except Exception as exc:
            logger.error("Architecture diagram generation failed: %s", exc)
            return "graph TD\n  A[User] --> B[API]\n  B --> C[Database]"

    def _parse_json_response(self, response: str) -> dict | list:
        """Extract and parse JSON from model response."""
        
        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try finding JSON in markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                return json.loads(response[start:end].strip())
        
        # Try finding JSON block with ```
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                content = response[start:end].strip()
                if content.startswith("json"):
                    content = content[4:].strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        
        # Last resort: find first { or [ and extract to last } or ]
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = response.find(start_char)
            end = response.rfind(end_char)
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end+1])
                except json.JSONDecodeError:
                    pass
        
        logger.warning("Could not parse JSON from response: %s", response[:200])
        return {} if "{" in response else []


# Singleton instance
_llm_client = None


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
