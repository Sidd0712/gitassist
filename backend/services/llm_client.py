"""Groq API LLM client for all model-based analysis."""

from __future__ import annotations

import asyncio
import json
import logging

from groq import AsyncGroq, RateLimitError, APIError

from core.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified interface for Groq API interactions."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set. Please add it to your .env file.")
            logger.warning("Get your free API key at: https://console.groq.com")
        self.client = AsyncGroq(api_key=self.settings.GROQ_API_KEY)
        self.model = self.settings.LLM_MODEL
        logger.info("✓ Groq API client initialized (model: %s)", self.model)

    async def _call_api(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Make async API call to Groq with retry logic."""
        max_retries = self.settings.LLM_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Groq API call: model={self.model}, temp={temperature}, prompt_len={len(prompt)}")
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                result = response.choices[0].message.content
                logger.debug(f"Groq API response: {len(result)} chars")
                return result
                
            except RateLimitError as exc:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Rate limit exceeded after %d retries", max_retries)
                    raise
                    
            except APIError as exc:
                logger.error(f"Groq API error: {exc}")
                raise
                
            except Exception as exc:
                logger.error(f"Unexpected error calling Groq API: {exc}")
                raise

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
            # Use low temperature for consistent structured output
            response = await self._call_api(prompt, temperature=0.3, max_tokens=1500)
            
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
            response = await self._call_api(prompt, temperature=0.5, max_tokens=800)
            questions = self._parse_json_response(response)
            if not isinstance(questions, list):
                questions = [questions]
            logger.info("✓ Generated %d clarification questions", len(questions))
            return questions
        except Exception as exc:
            logger.error("Clarification question generation failed: %s", exc)
            raise

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
            response = await self._call_api(prompt, temperature=0.4, max_tokens=1200)
            plan = self._parse_json_response(response)
            logger.info("✓ Retrieval plan generated for %d sections", len(plan.get("queries", [])))
            return plan
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
            response = await self._call_api(prompt, temperature=0.6, max_tokens=1500)
            descriptions = self._parse_json_response(response)
            if not isinstance(descriptions, list):
                descriptions = [descriptions]
            logger.info("✓ Generated %d repository descriptions", len(descriptions))
            return descriptions
        except Exception as exc:
            logger.error("Repository description generation failed: %s", exc)
            raise

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
            response = await self._call_api(prompt, temperature=0.7, max_tokens=2000)
            path = self._parse_json_response(response)
            if not isinstance(path, list):
                path = [path]
            logger.info("✓ Generated %d-step learning path", len(path))
            return path
        except Exception as exc:
            logger.error("Learning path generation failed: %s", exc)
            raise

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
            response = await self._call_api(prompt, temperature=0.6, max_tokens=1800)
            stack = self._parse_json_response(response)
            if not isinstance(stack, list):
                stack = [stack]
            logger.info("✓ Generated %d tech stack recommendations", len(stack))
            return stack
        except Exception as exc:
            logger.error("Tech stack generation failed: %s", exc)
            raise

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

IMPORTANT RULES:
1. Start with "graph TD" or "graph LR"
2. Use ONLY alphanumeric node IDs (A, B, C, etc.)
3. Use proper Mermaid arrow syntax: --> for directed edges
4. Put labels in square brackets for rectangles: A[Label]
5. Put labels in parentheses for rounded: A(Label)
6. Escape special characters in labels
7. Do NOT use semicolons at line ends
8. Keep labels short and clear

Example format:
graph TD
  A[User] --> B[API Gateway]
  B --> C[Auth Service]
  C --> D[Database]

Return ONLY valid Mermaid code, no markdown blocks, no explanations."""

        try:
            response = await self._call_api(prompt, temperature=0.3, max_tokens=1200)
            # Clean response
            diagram = response.strip()
            
            # Remove markdown code fences if present
            if diagram.startswith("```"):
                lines = diagram.split("\n")
                # Remove first and last lines if they're code fences
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                diagram = "\n".join(lines).strip()
            
            # Remove "mermaid" language identifier if present
            if diagram.lower().startswith("mermaid"):
                diagram = diagram[7:].strip()
            
            # Ensure it starts with graph declaration
            if not diagram.startswith("graph "):
                diagram = "graph TD\n" + diagram
            
            # Basic validation - check for common issues
            lines = diagram.split("\n")
            fixed_lines = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                # Remove trailing semicolons (not needed in Mermaid)
                if line.endswith(";"):
                    line = line[:-1]
                fixed_lines.append(line)
            
            diagram = "\n".join(fixed_lines)
            
            logger.info("✓ Generated architecture diagram")
            return diagram
        except Exception as exc:
            logger.error("Architecture diagram generation failed: %s", exc)
            # Return a simple fallback diagram
            return """graph TD
  A[User/Client] --> B[API Server]
  B --> C[Business Logic]
  C --> D[Database]
  B --> E[External Services]"""

    async def generate_search_queries(self, capability: str, idea_context: str, num_queries: int = 3) -> list[str]:
        """Generate GitHub search queries for a capability using AI."""
        
        prompt = f"""Generate {num_queries} diverse GitHub search queries to find repositories that implement: {capability}

Project context: {idea_context}

Return ONLY a JSON array of search query strings:
["query 1", "query 2", "query 3"]"""

        response = await self._call_api(prompt, temperature=0.4, max_tokens=400)
        return self._parse_json_response(response)

    async def evaluate_repository_quality(self, repo_name: str, description: str, readme: str, topics: list[str]) -> dict:
        """Evaluate repository quality and relevance using AI."""
        
        prompt = f"""Evaluate this GitHub repository's quality and purpose.

Repository: {repo_name}
Description: {description}
Topics: {', '.join(topics)}
README preview: {readme[:800]}

Return JSON with:
{{
  "quality_score": 0.0-1.0,
  "is_template_or_tutorial": true/false,
  "is_production_quality": true/false,
  "primary_purpose": "description",
  "reasoning": "why this score"
}}"""

        response = await self._call_api(prompt, temperature=0.3, max_tokens=600)
        return self._parse_json_response(response)

    async def prioritize_files(self, file_paths: list[str], project_intent: str, max_files: int = 30) -> list[dict]:
        """Rank files by relevance to project intent using AI."""
        
        prompt = f"""Given this project intent: {project_intent}

Rank these files by relevance (most important first):
{chr(10).join(file_paths[:100])}

Return top {max_files} as JSON array:
[{{
  "path": "file path",
  "priority_score": 0.0-1.0,
  "reason": "why important"
}}]"""

        response = await self._call_api(prompt, temperature=0.4, max_tokens=1000)
        return self._parse_json_response(response)

    async def evaluate_capability_match(self, capability: str, repo_text: str, keywords_context: dict) -> float:
        """Score how well a repository supports a capability using AI."""
        
        prompt = f"""Does this repository support the capability: {capability}?

Repository text:
{repo_text[:2000]}

Project keywords: {keywords_context.get('keywords', [])}
Domain terms: {keywords_context.get('domain_terms', [])}

Return JSON:
{{
  "match_score": 0.0-1.0,
  "confidence": "high|medium|low",
  "evidence": "brief explanation"
}}"""

        response = await self._call_api(prompt, temperature=0.3, max_tokens=400)
        result = self._parse_json_response(response)
        return result.get("match_score", 0.0)

    async def calculate_ranking_weights(self, idea: str, repositories_count: int) -> dict:
        """Determine optimal repository ranking weights using AI."""
        
        prompt = f"""For ranking {repositories_count} repositories for this idea:
{idea}

Determine importance weights for:
- readme_semantic_score
- metadata_semantic_score
- capability_coverage
- documentation_quality
- query_diversity
- star_quality

Return JSON with weights 0.0-1.0 that sum to 1.0:
{{
  "readme_semantic": 0.0-1.0,
  "metadata_semantic": 0.0-1.0,
  "capability_coverage": 0.0-1.0,
  "doc_quality": 0.0-1.0,
  "query_diversity": 0.0-1.0,
  "star_quality": 0.0-1.0
}}"""

        response = await self._call_api(prompt, temperature=0.4, max_tokens=500)
        return self._parse_json_response(response)

    async def determine_retrieval_weights(self, query_type: str, context: str) -> dict:
        """Calculate optimal retrieval scoring weights using AI."""
        
        prompt = f"""For a {query_type} retrieval query in this context:
{context}

Determine optimal weights for combining these signals:
- dense_score (semantic similarity via embeddings)
- lexical_score (keyword/BM25 matching)
- repo_prior (repository quality/relevance)
- role_prior (chunk type relevance)

Return JSON with weights that sum to 1.0:
{{
  "dense_weight": 0.0-1.0,
  "lexical_weight": 0.0-1.0,
  "repo_weight": 0.0-1.0,
  "role_weight": 0.0-1.0
}}"""

        response = await self._call_api(prompt, temperature=0.4, max_tokens=400)
        return self._parse_json_response(response)

    async def assess_document_quality(self, text: str) -> float:
        """Evaluate documentation quality using AI."""
        
        prompt = f"""Rate this documentation's quality:

{text[:1500]}

Consider:
- Completeness
- Clarity
- Usefulness for learning
- Examples and code samples

Return JSON:
{{
  "quality_score": 0.0-1.0
}}"""

        response = await self._call_api(prompt, temperature=0.3, max_tokens=300)
        result = self._parse_json_response(response)
        return result.get("quality_score", 0.0)

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
        
        # FAIL FAST: No fallback, raise error
        logger.error("Failed to parse JSON from AI response: %s", response[:200])
        raise ValueError(f"AI model did not return valid JSON. Response preview: {response[:200]}")


# Singleton instance
_llm_client = None


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
