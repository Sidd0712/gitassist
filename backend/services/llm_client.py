"""Groq API LLM client for all model-based analysis."""

from __future__ import annotations

import asyncio
import json
import logging

from groq import APIError, AsyncGroq, RateLimitError

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
        logger.info("Groq API client initialized (model: %s)", self.model)

    async def _call_api(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout_seconds: int | None = None,
    ) -> str:
        """Make async API call to Groq with retry logic."""

        max_retries = self.settings.LLM_MAX_RETRIES
        timeout_seconds = timeout_seconds or self.settings.LLM_TIMEOUT_SECONDS

        for attempt in range(max_retries):
            try:
                logger.debug(
                    "Groq API call: model=%s temp=%.2f prompt_len=%d timeout=%ss",
                    self.model,
                    temperature,
                    len(prompt),
                    timeout_seconds,
                )
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout_seconds,
                )
                result = response.choices[0].message.content
                logger.debug("Groq API response: %d chars", len(result))
                return result

            except RateLimitError:
                if attempt >= max_retries - 1:
                    logger.error("Rate limit exceeded after %d retries", max_retries)
                    raise
                wait_time = 2**attempt
                logger.warning("Rate limit hit (attempt %d/%d), waiting %ss", attempt + 1, max_retries, wait_time)
                await asyncio.sleep(wait_time)
            except TimeoutError:
                logger.error("Groq API call timed out after %ss", timeout_seconds)
                raise
            except APIError as exc:
                logger.error("Groq API error: %s", exc)
                raise
            except Exception as exc:
                logger.error("Unexpected error calling Groq API: %s", exc)
                raise

        raise RuntimeError("Groq API call failed unexpectedly.")

    async def _call_json_api(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
    ) -> dict | list:
        response = await self._call_api(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        return self._parse_json_response(response)

    async def extract_keywords(self, idea: str, clarification_answers: dict[str, str] | None = None) -> dict:
        """Use model to extract technical keywords and capabilities from user idea."""

        clarification = ""
        if clarification_answers:
            lines = [f"- {key.replace('_', ' ')}: {value}" for key, value in clarification_answers.items()]
            clarification_block = "\n".join(lines)
            clarification = f"\nClarifications:\n{clarification_block}"

        prompt = f"""Analyze this project idea and extract technical structure in JSON format.

Project Idea:
{idea}{clarification}

Important instructions:
- Treat clarification answers as authoritative. Use them to sharpen the extracted keywords, stack families, integrations, and likely components.
- Only include ambiguities that are still unresolved after considering the clarification answers.
- Ambiguities should focus on build decisions that would materially change GitHub repository search terms, architecture, data model, deployment shape, or the main frameworks/services to look for.
- Each ambiguity must include 2-4 concrete, mutually exclusive options that would help narrow repository search and keyword extraction.
- Keep ambiguity axes stable and machine-friendly using concise snake_case keys.
- Do not invent generic filler options like "simple", "standard", or "advanced" unless the project idea itself truly implies those choices.

Extract and return ONLY valid JSON (no markdown, no explanation):
{{
  "summary": "one-line technical summary",
  "core_intent": "core business/technical intent",
  "product_type": "category (e.g., social travel planner, collaborative whiteboard)",
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
    {{
      "axis": "ambiguity_name",
      "question": "single concrete question to ask the user",
      "reason": "why the answer changes repo search, architecture, or stack decisions",
      "options": ["option 1", "option 2", "option 3"],
      "severity": "high/medium/low"
    }}
  ]
}}"""

        result = await self._call_json_api(
            prompt,
            temperature=0.3,
            max_tokens=1200,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        logger.info(
            "Keywords extracted: %d capabilities, %d frameworks",
            len(result.get("primary_capabilities", [])) + len(result.get("secondary_capabilities", [])),
            len(result.get("frameworks", [])),
        )
        return result

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

        questions = await self._call_json_api(
            prompt,
            temperature=0.5,
            max_tokens=700,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        if not isinstance(questions, list):
            questions = [questions]
        logger.info("Generated %d clarification questions", len(questions))
        return questions

    async def plan_retrieval_queries(self, idea: str, keywords: dict, repositories: list[dict]) -> dict:
        """Use model to generate targeted retrieval queries and weights for each analysis section."""

        repo_summary = ", ".join(repo.get("full_name", "") for repo in repositories[:5]) if repositories else "no repos"

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
      "preferred_roles": ["documentation", "source"],
      "top_k": 8,
      "weights": {{
        "dense_weight": 0.0-1.0,
        "lexical_weight": 0.0-1.0,
        "repo_weight": 0.0-1.0,
        "role_weight": 0.0-1.0
      }}
    }}
  ]
}}"""

        plan = await self._call_json_api(
            prompt,
            temperature=0.35,
            max_tokens=900,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        logger.info("Retrieval plan generated for %d sections", len(plan.get("queries", [])))
        return plan

    async def answer_repo_chat(
        self,
        question: str,
        idea_summary: str,
        messages: list[dict],
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> dict:
        """Answer a repo-scoped chat question using only retrieved evidence."""

        history = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')[:600]}"
            for message in messages[-6:]
            if message.get("content")
        )
        repo_summary = ", ".join(repo.get("full_name", "") for repo in repositories[:6]) if repositories else "no repos"

        prompt = f"""You are answering questions about indexed GitHub repositories for a product idea.

Rules:
- Only answer using the provided evidence and repo metadata.
- If the evidence is insufficient, say that clearly and do not invent implementation details.
- Prefer concrete repo/file references when explaining where the answer comes from.
- Keep the answer concise but useful.
- Suggest 2-3 brief follow-up questions the user could ask next.

Idea Summary: {idea_summary or "Not provided"}
Scoped Repositories: {repo_summary}
Recent Conversation:
{history or "No prior chat history."}

User Question: {question}

Retrieved Evidence:
{self._format_evidence(evidence)}

Return ONLY valid JSON (no markdown):
{{
  "answer": "grounded answer",
  "follow_up_suggestions": ["follow-up 1", "follow-up 2", "follow-up 3"]
}}"""

        result = await self._call_json_api(
            prompt,
            temperature=0.2,
            max_tokens=900,
            timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
        )
        return result if isinstance(result, dict) else {}

    async def generate_repo_descriptions(
        self,
        idea: str,
        keywords: dict,
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> list[str]:
        """Use model to generate natural descriptions of reference repositories."""

        repo_names = [repo.get("full_name", "") for repo in repositories[:5]]
        prompt = f"""Given project idea, repositories, and retrieved evidence, generate concise descriptions of why each repo matters.

Project Idea: {idea}
Primary Features: {', '.join(keywords.get('primary_capabilities', []))}
Reference Repos: {', '.join(repo_names)}
Evidence:
{self._format_evidence(evidence)}

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

        descriptions = await self._call_json_api(
            prompt,
            temperature=0.55,
            max_tokens=900,
            timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
        )
        if not isinstance(descriptions, list):
            descriptions = [descriptions]
        logger.info("Generated %d repository descriptions", len(descriptions))
        return descriptions

    async def generate_learning_path(
        self,
        idea: str,
        keywords: dict,
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> list[dict]:
        """Use model to generate step-by-step learning path."""

        primary = ", ".join(keywords.get("primary_capabilities", [])[:3])
        secondary = ", ".join(keywords.get("secondary_capabilities", [])[:3])
        prompt = f"""Create a practical learning path for building: {keywords.get('product_type', 'project')}

Primary Focus: {primary}
Secondary Features: {secondary}
Implementation References: {repositories[0].get('full_name', 'reference') if repositories else 'reference'} and others
Evidence:
{self._format_evidence(evidence)}

Generate 5-6 concrete learning steps with milestones. Return JSON array (no markdown):
[
  {{
    "step": 1,
    "title": "step title",
    "description": "what to learn",
    "milestone": "concrete deliverable",
    "concepts": ["concept 1"],
    "resources": ["resource 1"]
  }}
]"""

        path = await self._call_json_api(
            prompt,
            temperature=0.65,
            max_tokens=1400,
            timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
        )
        if not isinstance(path, list):
            path = [path]
        logger.info("Generated %d-step learning path", len(path))
        return path

    async def generate_tech_stack(
        self,
        idea: str,
        keywords: dict,
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> list[dict]:
        """Use model to recommend technology decisions."""

        primary = ", ".join(keywords.get("primary_capabilities", [])[:3])
        frameworks = ", ".join(keywords.get("frameworks", [])[:5])
        prompt = f"""Recommend technology stack for: {keywords.get('product_type', 'project')}

Primary Capabilities: {primary}
Suggested Frameworks: {frameworks}
Evidence:
{self._format_evidence(evidence)}

For each major layer (frontend, backend, database, realtime, deployment), provide:
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
    "supported_by": ["evidence point 1"],
    "pros": ["pro 1"],
    "cons": ["con 1"]
  }}
]"""

        stack = await self._call_json_api(
            prompt,
            temperature=0.55,
            max_tokens=1200,
            timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
        )
        if not isinstance(stack, list):
            stack = [stack]
        logger.info("Generated %d tech stack recommendations", len(stack))
        return stack

    async def generate_architecture_diagram(
        self,
        idea: str,
        keywords: dict,
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> str:
        """Use model to generate Mermaid diagram code."""

        primary = ", ".join(keywords.get("primary_capabilities", [])[:3])
        components = ", ".join(keywords.get("likely_components", [])[:5])
        prompt = f"""Generate a Mermaid diagram for: {keywords.get('product_type', 'project')}

Main Capabilities: {primary}
Key Components: {components}
Evidence:
{self._format_evidence(evidence)}

Create a Mermaid flowchart showing:
- User or client layer
- API or backend services
- Data layer
- External integrations
- Key data flows

IMPORTANT RULES:
1. Start with "graph TD" or "graph LR"
2. Use only alphanumeric node IDs (A, B, C, etc.)
3. Use proper Mermaid arrow syntax: --> for directed edges
4. Put labels in square brackets for rectangles: A[Label]
5. Put labels in parentheses for rounded nodes: A(Label)
6. Do not use semicolons at line ends
7. Keep labels short and clear

Return ONLY valid Mermaid code, no markdown blocks, no explanations."""

        try:
            response = await self._call_api(
                prompt,
                temperature=0.2,
                max_tokens=700,
                timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.error("Architecture diagram generation failed: %s", exc)
            return """graph TD
  A[User/Client] --> B[API Server]
  B --> C[Business Logic]
  C --> D[Database]
  B --> E[External Services]"""

        diagram = response.strip()
        if diagram.startswith("```"):
            lines = diagram.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            diagram = "\n".join(lines).strip()
        if diagram.lower().startswith("mermaid"):
            diagram = diagram[7:].strip()
        if not diagram.startswith("graph "):
            diagram = f"graph TD\n{diagram}"

        cleaned_lines: list[str] = []
        for line in diagram.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if stripped.endswith(";"):
                stripped = stripped[:-1]
            cleaned_lines.append(stripped)

        logger.info("Generated architecture diagram")
        return "\n".join(cleaned_lines)

    async def generate_search_queries_batch(
        self,
        idea_context: str,
        capabilities: list[str],
        domain_terms: list[str],
        languages: list[str],
        max_queries: int = 6,
    ) -> list[str]:
        """Generate a bundled set of GitHub search queries for the current idea."""

        prompt = f"""Generate up to {max_queries} diverse GitHub search queries for this software idea.

Project context: {idea_context}
Primary capabilities: {', '.join(capabilities[:4])}
Domain terms: {', '.join(domain_terms[:6])}
Languages: {', '.join(languages[:2])}

Instructions:
- Return one combined query set for the whole idea, not separate sections.
- Favor phrases that would find end-to-end repos or strong subsystem references.
- Keep the queries concise and realistic for GitHub repository search.
- Avoid repeating the same terms with tiny wording changes.
- Each query must be 2-5 words maximum. No long sentences.
- Do not append the idea context to every query — use it only to choose the right terms.

Return ONLY a JSON array of search query strings:
["query 1", "query 2", "query 3"]"""

        queries = await self._call_json_api(
            prompt,
            temperature=0.35,
            max_tokens=700,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        if not isinstance(queries, list):
            return []
        return [query for query in queries if isinstance(query, str)]

    async def generate_search_queries(self, capability: str, idea_context: str, num_queries: int = 3) -> list[str]:
        """Generate GitHub search queries for a capability using AI."""

        prompt = f"""Generate {num_queries} diverse GitHub search queries to find repositories that implement: {capability}

Project context: {idea_context}

Return ONLY a JSON array of search query strings:
["query 1", "query 2", "query 3"]"""

        queries = await self._call_json_api(
            prompt,
            temperature=0.4,
            max_tokens=400,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        return queries if isinstance(queries, list) else []
    
    async def expand_capability_aliases(
        self,
        capabilities: list[str],
        idea_context: str,
        aliases_per_capability: int = 4,
    ) -> dict[str, list[str]]:
        """Generate synonyms and related search terms for each capability."""

        prompt = f"""You are a JSON API. Return ONLY a JSON object, no explanation, no prose, no markdown.

    Task: for each capability below, generate {aliases_per_capability} concise synonyms or related GitHub search terms.

    Idea context: {idea_context[:200]}
    Capabilities: {json.dumps(capabilities)}

    Rules:
    - Keys must exactly match the input capability strings
    - Each value is a list of {aliases_per_capability} short search-friendly synonyms
    - Synonyms should be terms a developer would use on GitHub, not marketing language
    - Return nothing except the JSON object

    {json.dumps({cap: ["synonym1", "synonym2", "synonym3", "synonym4"] for cap in capabilities})}

    Replace the placeholder values with real synonyms and return the object:"""

        result = await self._call_json_api(
            prompt,
            temperature=0.2,
            max_tokens=800,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        return result if isinstance(result, dict) else {}

    async def evaluate_repository_quality(self, repo_name: str, description: str, readme: str, topics: list[str]) -> dict:
        """Evaluate repository quality and relevance using AI."""

        prompt = f"""You are a JSON API. Return ONLY a JSON object.

Evaluate this GitHub repository:
Repository: {repo_name}
Description: {description[:300]}
Topics: {', '.join(topics)}
README: {readme[:600]}

{{"quality_score": 0.0, "is_template_or_tutorial": false, "is_production_quality": false, "primary_purpose": "one phrase", "reasoning": "one sentence max"}}

Replace values and return the object:"""

        result = await self._call_json_api(
            prompt,
            temperature=0.3,
            max_tokens=600,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        return result if isinstance(result, dict) else {}

    async def prioritize_files(self, file_paths: list[str], project_intent: str, max_files: int = 30) -> list[dict]:
        """Rank files by relevance to project intent using AI."""

        prompt = f"""Given this project intent: {project_intent}

Rank these files by relevance (most important first):
{chr(10).join(file_paths[:100])}

Return top {max_files} as JSON array:
[{{"path": "file path", "priority_score": 0.0-1.0, "reason": "why important"}}]"""

        result = await self._call_json_api(
            prompt,
            temperature=0.4,
            max_tokens=1000,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        return result if isinstance(result, list) else []

    async def evaluate_capability_match(self, capability: str, repo_text: str, keywords_context: dict) -> float:
        """Score how well a repository supports a capability using AI."""

        prompt = f"""You are a JSON API. Return ONLY a JSON object.

Does this repository support the capability: {capability}

Repository text: {repo_text[:2000]}
Project keywords: {keywords_context.get('keywords', [])}

{{"match_score": 0.0, "confidence": "high", "evidence": "one sentence max"}}

Replace values and return the object:"""

        result = await self._call_json_api(
            prompt,
            temperature=0.3,
            max_tokens=400,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        if not isinstance(result, dict):
            return 0.0
        return float(result.get("match_score", 0.0))

    async def calculate_ranking_weights(self, idea: str, repositories_count: int) -> dict:
        """Determine optimal repository ranking weights using AI."""

        prompt = f"""You are a JSON API. Return ONLY a JSON object, no explanation, no prose, no markdown.

    Task: assign ranking weights for {repositories_count} repositories matching this idea: {idea}

    Rules:
    - All 6 values must be floats between 0.0 and 1.0
    - Values must sum to exactly 1.0
    - Return nothing except the JSON object

    {{"readme_semantic": 0.0, "metadata_semantic": 0.0, "capability_coverage": 0.0, "doc_quality": 0.0, "query_diversity": 0.0, "star_quality": 0.0}}

    Replace the 0.0 values with your weights and return the object:"""

        result = await self._call_json_api(
            prompt,
            temperature=0.1,
            max_tokens=120,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        return result if isinstance(result, dict) else {}

    async def determine_retrieval_weights(self, query_type: str, context: str) -> dict:
        """Calculate optimal retrieval scoring weights using AI."""

        prompt = f"""You are a JSON API. Return ONLY a JSON object, no explanation, no prose, no markdown.

Task: assign retrieval weights for a {query_type} query in this context: {context[:200]}

Rules:
- All 4 values must be floats between 0.0 and 1.0
- Values must sum to exactly 1.0
- Return nothing except the JSON object

{{"dense_weight": 0.0, "lexical_weight": 0.0, "repo_weight": 0.0, "role_weight": 0.0}}

Replace the 0.0 values with your weights and return the object:"""

        result = await self._call_json_api(
            prompt,
            temperature=0.3,
            max_tokens=400,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        return result if isinstance(result, dict) else {}

    async def assess_document_quality(self, text: str) -> float:
        """Evaluate documentation quality using AI."""

        prompt = f"""You are a JSON API. Return ONLY a JSON object.

Rate this documentation's quality as a float from 0.0 (useless) to 1.0 (excellent):

{text[:1500]}

{{"quality_score": 0.0}}

Replace 0.0 with your score:"""

        result = await self._call_json_api(
            prompt,
            temperature=0.3,
            max_tokens=300,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
        )
        if not isinstance(result, dict):
            return 0.0
        return float(result.get("quality_score", 0.0))

    def _format_evidence(self, evidence: dict | None) -> str:
        if not evidence:
            return "No additional grounded evidence was available beyond the shortlisted repositories."
        return json.dumps(evidence, ensure_ascii=True, indent=2)[:4000]

    def _parse_json_response(self, response: str) -> dict | list:
        """Extract and parse JSON from model response."""

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                return json.loads(response[start:end].strip())

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

        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = response.find(start_char)
            end = response.rfind(end_char)
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start : end + 1])
                except json.JSONDecodeError:
                    pass

        logger.error("Failed to parse JSON from AI response: %s", response[:200])
        raise ValueError(f"AI model did not return valid JSON. Response preview: {response[:200]}")


_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client."""

    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
