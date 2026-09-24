"""Groq API LLM client for all model-based analysis."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
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
        self._groq_unavailable_until = 0.0
        self._gemini_unavailable_until: dict[str, float] = {}
        logger.info(
            "Groq API client initialized (model: %s, gemini: %s)",
            self.model,
            ", ".join(self._gemini_models) if self._fallback_enabled else "disabled",
        )

    @property
    def _gemini_models(self) -> list[str]:
        return [m for m in (self.settings.LLM_FALLBACK_MODEL, self.settings.LLM_SECONDARY_GEMINI_MODEL) if m]

    @property
    def _fallback_enabled(self) -> bool:
        return bool(self.settings.GOOGLE_API_KEY and self._gemini_models)

    async def _call_api(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout_seconds: int | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        prefer: str = "groq",
    ) -> str:
        """Call the preferred provider, falling back to the other when it fails.

        Groq's free tier allows 8K tokens/min, so big-context calls (ranking,
        report sections, chat) pass prefer="gemini" (250K TPM) and use Groq
        only as the fallback; small calls stay Groq-first. Only provider
        failures fall back — a bad response from a healthy provider is the
        caller's problem.
        """

        timeout_seconds = timeout_seconds or self.settings.LLM_TIMEOUT_SECONDS
        if prefer == "gemini" and self._fallback_enabled:
            try:
                return await self._call_gemini_chain(prompt, temperature, max_tokens, timeout_seconds, json_mode)
            except Exception as exc:
                logger.warning("Gemini failed (%s); falling back to Groq", exc)
                return await self._call_groq(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    json_mode=json_mode,
                    seed=seed,
                )

        if self._fallback_enabled and time.monotonic() < self._groq_unavailable_until:
            return await self._call_gemini_chain(prompt, temperature, max_tokens, timeout_seconds, json_mode)

        try:
            return await self._call_groq(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                json_mode=json_mode,
                seed=seed,
            )
        except (APIError, TimeoutError) as exc:
            if not self._fallback_enabled:
                raise
            status = getattr(exc, "status_code", None)
            if isinstance(exc, RateLimitError):
                cooldown = 60.0
            elif status in (401, 403, 404):
                cooldown = 600.0  # bad key or retired model — won't fix itself soon
            elif isinstance(exc, TimeoutError):
                cooldown = 0.0
            else:
                cooldown = 30.0
            self._groq_unavailable_until = max(self._groq_unavailable_until, time.monotonic() + cooldown)
            logger.warning("Groq failed (%s: %s); falling back to Gemini", type(exc).__name__, status or "-")
            return await self._call_gemini_chain(prompt, temperature, max_tokens, timeout_seconds, json_mode)

    async def _call_gemini_chain(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        json_mode: bool,
    ) -> str:
        """Try each configured Gemini model; each has its own free quota."""

        last_error: Exception = RuntimeError("No Gemini model configured")
        now = time.monotonic()
        models = [m for m in self._gemini_models if self._gemini_unavailable_until.get(m, 0.0) <= now]
        for model in models or self._gemini_models:
            try:
                return await self._call_gemini(prompt, temperature, max_tokens, timeout_seconds, json_mode, model)
            except Exception as exc:
                last_error = exc
                if "429" in str(exc):
                    self._gemini_unavailable_until[model] = time.monotonic() + 60.0
                logger.warning("Gemini model %s failed: %s", model, str(exc)[:200])
        raise last_error

    async def _call_gemini(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        json_mode: bool,
        model: str | None = None,
    ) -> str:
        model = model or self._gemini_models[0]
        generation_config: dict = {
            "temperature": temperature,
            # Thinking tokens count against maxOutputTokens; "low" keeps them
            # small, and the headroom stops them from truncating the answer.
            "maxOutputTokens": max_tokens + 1024,
            "thinkingConfig": {"thinkingLevel": "low"},
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": self.settings.GOOGLE_API_KEY},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": generation_config,
                },
            )
        if response.status_code != 200:
            raise RuntimeError(f"Gemini {model} failed: {response.status_code} {response.text[:300]}")
        candidates = response.json().get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini fallback returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        if not text:
            raise RuntimeError(f"Gemini fallback returned no text (finishReason={candidates[0].get('finishReason')})")
        logger.info("Served by Gemini (%s): %d chars", model, len(text))
        return text

    async def _call_groq(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        json_mode: bool,
        seed: int | None,
    ) -> str:
        """Make async API call to Groq with retry logic."""

        max_retries = self.settings.LLM_MAX_RETRIES

        for attempt in range(max_retries):
            try:
                logger.debug(
                    "Groq API call: model=%s temp=%.2f prompt_len=%d timeout=%ss",
                    self.model,
                    temperature,
                    len(prompt),
                    timeout_seconds,
                )
                extra_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                if seed is not None:
                    extra_kwargs["seed"] = seed
                if "gpt-oss" in self.model:
                    extra_kwargs["reasoning_effort"] = "low"
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **extra_kwargs,
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
        json_mode: bool = True,
        seed: int | None = None,
        prefer: str = "groq",
    ) -> dict | list:
        """Call the model expecting a JSON response.

        Groq's (OpenAI-compatible) json_object response_format requires the
        top-level response to be a JSON *object* — it cannot produce a bare
        JSON array. Callers whose prompt asks for a bare array (e.g. `[...]`)
        must pass json_mode=False and rely on _parse_json_response's fallback
        parsing instead.
        """

        response = await self._call_api(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            json_mode=json_mode,
            seed=seed,
            prefer=prefer,
        )
        return self._parse_json_response(response)

    @staticmethod
    def _coerce_json_list(value: object) -> list:
        """Best-effort coercion of a JSON response into a list.

        Models sometimes ignore a "return a bare array" instruction and wrap
        the array in an object instead (e.g. {"items": [...]}). If we get a
        dict with exactly one list-valued key, unwrap it rather than nesting
        the whole dict as a single malformed list entry.
        """

        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            list_values = [v for v in value.values() if isinstance(v, list)]
            if len(list_values) == 1:
                return list_values[0]
        return []

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
            temperature=0.0,
            max_tokens=1200,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
            seed=42,
        )
        logger.info(
            "Keywords extracted: %d capabilities, %d frameworks",
            len(result.get("primary_capabilities", [])) + len(result.get("secondary_capabilities", [])),
            len(result.get("frameworks", [])),
        )
        return result

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
        "repo_weight": 0.0-1.0
      }}
    }}
  ]
}}"""

        plan = await self._call_json_api(
            prompt,
            temperature=0.35,
            max_tokens=900,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
            prefer="gemini",
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
        """Answer a repo-scoped chat question from the repo map and stitched code.

        `evidence` carries "segments" (stitched code, most relevant first),
        "repo_maps" (structure of every scoped repo) and "intent". The answer
        is Markdown, not JSON: code inside JSON strings breaks and truncates.
        Gemini gets the full context; the Groq fallback (8K tokens/min) a
        trimmed one.
        """

        evidence = evidence or {}
        full_prompt = self._chat_prompt(
            question, idea_summary, messages, evidence, self.settings.CHAT_CONTEXT_CHARS, history_turns=8
        )
        options = {
            "temperature": 0.25,
            "max_tokens": self.settings.CHAT_MAX_OUTPUT_TOKENS,
            "timeout_seconds": self.settings.CHAT_TIMEOUT_SECONDS,
            "json_mode": False,
        }
        text: str | None = None
        if self._fallback_enabled:
            try:
                text = await self._call_gemini_chain(
                    full_prompt, options["temperature"], options["max_tokens"], options["timeout_seconds"], False
                )
            except Exception as exc:
                logger.warning("Chat: Gemini failed (%s); falling back to Groq with a trimmed context", str(exc)[:200])
        if text is None:
            trimmed = self._chat_prompt(
                question, idea_summary, messages, evidence, self.settings.CHAT_FALLBACK_CONTEXT_CHARS, history_turns=2
            )
            text = await self._call_groq(trimmed, seed=None, **{**options, "max_tokens": 1500})
        return self._split_follow_ups(text)

    _CHAT_INTENT_GUIDANCE = {
        "code": "Walk through the implementation step by step, quoting the key lines of each step.",
        "setup": "Give the concrete setup/run steps from the README and config as a numbered list with exact commands.",
        "stack": "List each technology with where it's declared or imported and what the repo uses it for.",
        "compare": "Compare the repositories side by side on exactly what was asked, then recommend one for this project and say why.",
        "architecture": "Describe the main modules/folders and how a request or data flows through them, citing the files involved.",
    }

    def _chat_prompt(
        self,
        question: str,
        idea_summary: str,
        messages: list[dict],
        evidence: dict,
        context_chars: int,
        history_turns: int,
    ) -> str:
        history = "\n\n".join(
            f"{'Developer' if m.get('role') == 'user' else 'You'}: {m.get('content', '')[:1500]}"
            for m in messages[-history_turns:]
            if m.get("content")
        )

        map_blocks = []
        for repo in evidence.get("repo_maps", []):
            lines = [f"### {repo['full_name']} ({repo.get('language') or 'unknown language'}, ★{repo.get('stars', 0)})"]
            if repo.get("description"):
                lines.append(repo["description"])
            lines.append(f"Indexed files: {repo.get('file_count', 0)}")
            if repo.get("top_level_dirs"):
                lines.append("Top-level folders: " + ", ".join(repo["top_level_dirs"]))
            if repo.get("root_files"):
                lines.append("Root files: " + ", ".join(repo["root_files"]))
            if repo.get("key_files"):
                lines.append("Key files: " + ", ".join(repo["key_files"]))
            if repo.get("dependencies"):
                lines.append("Declared dependencies: " + ", ".join(repo["dependencies"]))
            if repo.get("readme_opening") and context_chars > 20_000:
                lines.append("README opening:\n" + repo["readme_opening"])
            map_blocks.append("\n".join(lines))

        excerpt_blocks, used = [], 0
        for index, segment in enumerate(evidence.get("segments", []), start=1):
            language = (segment.get("language") or "").lower().replace("c#", "csharp").replace("c++", "cpp")
            block = (
                f"[{index}] {segment['repo_full_name']} · {segment['path']}:"
                f"L{segment['start_line']}-L{segment['end_line']}\n```{language}\n{segment['text']}\n```"
            )
            if used + len(block) > context_chars and excerpt_blocks:
                break
            excerpt_blocks.append(block)
            used += len(block)

        guidance = self._CHAT_INTENT_GUIDANCE.get(evidence.get("intent", ""), "")
        return f"""You are GitAssist's code guide. A developer is planning this project:
{idea_summary or "(no summary given)"}

They are studying the GitHub repositories below. Every repository has been indexed; the map lists its files, and the excerpts are the sections that best match their question. Answer using ONLY this material.

How to answer:
- Start with the direct answer in one or two sentences.
- Back it with real code: quote the relevant lines from the excerpts in fenced code blocks with a language tag. Put the source on its own line directly above each block as inline code, e.g. `owner/repo · path:L10-L42` — never inside the fenced block.
- Refer to files by their exact paths. Never invent files, functions, APIs, settings or behaviour that are not in the excerpts or the map.
- If the excerpts only partly answer, say what they do show, then name the specific files from the map that would hold the rest.
- Never claim a repository has no code or no implementation: the map lists every indexed file. If something isn't in the excerpts, say it wasn't in the retrieved sections.
- Relate it to the developer's project: how they could reuse or adapt the approach, pitfalls to avoid, and where they could improve on it. Be specific, not generic.
- Use Markdown: short paragraphs, bullet lists, fenced code. Aim for 150-450 words plus code; go longer only for a requested walkthrough.
{f"- {guidance}" if guidance else ""}

Conversation so far:
{history or "(this is the first question)"}

Repository map:
{chr(10).join(map_blocks) or "(no repositories)"}

Code excerpts (most relevant first):
{chr(10).join(excerpt_blocks) or "(no matching sections were retrieved)"}

Developer's question: {question}

After your answer, write a line containing only FOLLOW_UPS: and then 3 short follow-up questions this developer would find useful next, one per line, each starting with "- "."""

    @staticmethod
    def _split_follow_ups(text: str) -> dict:
        answer, _, tail = text.partition("FOLLOW_UPS:")
        follow_ups = [
            line.strip().lstrip("-*•0123456789. ").strip()
            for line in tail.splitlines()
            if line.strip().lstrip("-*•0123456789. ").strip()
        ]
        return {"answer": answer.strip(), "follow_up_suggestions": follow_ups[:3]}

    async def generate_repo_descriptions(
        self,
        idea: str,
        keywords: dict,
        repositories: list[dict],
        evidence: dict | None = None,
    ) -> list[str]:
        """Use model to generate natural descriptions of reference repositories."""

        repo_names = [repo.get("full_name", "") for repo in repositories[:5]]
        prompt = f"""Describe why each reference repository matters for this project, strictly from its evidence.

Project idea: {idea}
Primary features: {', '.join(keywords.get('primary_capabilities', []))}
Repositories: {', '.join(repo_names)}

Evidence (per repository: README excerpt, file paths, declared dependencies, and code hits when available):
{self._format_evidence(evidence)}

For each repository write 3-4 sentences covering: what it actually is and does; which parts are relevant to this project (name real files, folders or dependencies from its evidence); and what the developer can learn or reuse from it.

Hard rules:
- Only state what that repository's evidence shows. Do not name components, services, queues, APIs or patterns that aren't in its README, paths, dependencies or code hits.
- If a repository only matches the technique (e.g. a generic library) rather than the project's domain, say so plainly.
- If its evidence is thin (a short README and few files), say that it's a small project and keep claims modest.
- Plain text, no Markdown.

Return ONLY a JSON object mapping each repository full_name to its description:
{{"owner/repo": "description"}}"""

        result = await self._call_json_api(
            prompt,
            temperature=0.3,
            max_tokens=1800,
            timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
            json_mode=True,
            prefer="gemini",
        )
        if isinstance(result, dict) and not any(isinstance(v, list) for v in result.values()):
            by_name = {str(k).strip().lower(): v for k, v in result.items() if isinstance(v, str)}
            descriptions = [by_name.get(name.lower(), "") for name in repo_names]
        else:  # the model ignored the object format and returned an array
            descriptions = [item for item in self._coerce_json_list(result) if isinstance(item, str)]
        logger.info("Generated %d repository descriptions", sum(1 for d in descriptions if d))
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

Project idea: {idea}
Target users: {', '.join(keywords.get('target_users', [])[:3]) or 'not specified'}
Likely components: {', '.join(keywords.get('likely_components', [])[:6])}
Likely integrations / data sources: {', '.join(keywords.get('likely_integrations', [])[:6]) or 'not specified'}
Constraints: {', '.join(keywords.get('constraints', [])[:4]) or 'none stated'}

Generate 5-6 concrete learning steps with milestones, ordered so each builds on the last.

Coverage rules:
- Include a step for how the project gets its data or inputs (the actual sources, APIs, scraping or user input it needs), if it depends on external data.
- Include a step for the hardest domain-specific component of this idea (the part that makes it this product, not a generic CRUD app). Name the technique it needs.
- Where a reference repository's evidence shows a relevant file or folder, name it in that step's description ("study owner/repo's path/to/file") so the developer knows what to read.
- Keep scope realistic for the target users and constraints; no enterprise infrastructure unless the idea requires it.

For "resources": never invent a specific article, blog post, or deep sub-page
URL -- you cannot verify one exists. Only include a URL if it is either (a) a
path shown in the Evidence above, referenced as "owner/repo - path", or (b) a
top-level official documentation homepage for a named technology (e.g.
"https://react.dev", "https://fastapi.tiangolo.com", "https://developer.mozilla.org",
"https://docs.python.org", "https://redis.io") with no sub-path appended.
Otherwise give the resource as plain text with no URL at all (e.g. "Official
Express.js routing guide").

Return JSON array (no markdown):
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
            max_tokens=2200,
            timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
            json_mode=False,
            prefer="gemini",
        )
        path = [item for item in self._coerce_json_list(path) if isinstance(item, dict)]
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

Project idea: {idea}
Target users: {', '.join(keywords.get('target_users', [])[:3]) or 'not specified'}
Likely components: {', '.join(keywords.get('likely_components', [])[:6])}
Constraints: {', '.join(keywords.get('constraints', [])[:4]) or 'none stated'}

Cover only the layers this project actually needs (e.g. frontend, backend,
database, data ingestion, ML/analytics, realtime, deployment). For each give
the recommended technology, why it fits, pros, and cons.

Rules:
- Right-size for the target users and constraints. Prefer the simplest stack
  that works; don't recommend Kubernetes, Kafka, microservices or other
  large-scale infrastructure unless the idea explicitly needs that scale.
- Cover the project's domain-specific needs (e.g. the data collection or
  analysis libraries it depends on), not only generic web layers.
- Each repository in the Evidence lists "declared_dependencies" (real
  packages/images parsed from its manifests). Prefer technologies that appear
  there and name the repositories that declare them in "supported_by". If no
  repository backs a choice, put ["general recommendation"] in "supported_by".
- Keep pros/cons current and specific; no outdated claims.

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
            max_tokens=2000,
            timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
            json_mode=False,
            prefer="gemini",
        )
        stack = [item for item in self._coerce_json_list(stack) if isinstance(item, dict)]
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

Project idea: {idea}
Likely integrations / data sources: {', '.join(keywords.get('likely_integrations', [])[:6]) or 'not specified'}

Create a Mermaid flowchart of THIS project (not a generic web app):
- Use the project's own components by name (from Key Components and the idea),
  including how data enters the system and where the domain logic lives
- The client/UI, services, data stores and external sources it needs
- Label every edge with what flows along it: A -->|job listings| B
- 7-12 nodes; group related nodes with subgraph blocks if it helps

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
                max_tokens=1100,
                timeout_seconds=self.settings.LLM_GENERATION_TIMEOUT_SECONDS,
                prefer="gemini",
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
            json_mode=False,
        )
        return [query for query in self._coerce_json_list(queries) if isinstance(query, str)]

    async def rank_repositories(
        self,
        idea: str,
        keywords: dict,
        candidates: list[dict],
        limit: int = 5,
    ) -> list[dict]:
        """Judge and select the best reference repositories for this idea."""

        candidates_text = "\n\n".join(
            f"### {c.get('full_name', '')}\n"
            f"Stars: {c.get('stars', 0)} | Language: {c.get('language', '')} | Topics: {', '.join(c.get('topics', [])[:8])}\n"
            f"Description: {c.get('description', '')}\n"
            f"Manifests: {', '.join(c.get('manifest_files', []))}\n"
            f"Sampled code files: {', '.join(c.get('sampled_paths', [])[:8])}\n"
            f"README excerpt: {c.get('readme_excerpt', '')}"
            for c in candidates
        )

        prompt = f"""You are a JSON API. Return ONLY a JSON object, no explanation, no markdown.

Project idea: {idea}
Domain: {keywords.get('product_type', '')} — {', '.join(keywords.get('domain_terms', [])[:6])}
Key capabilities: {', '.join(keywords.get('primary_capabilities', []) or keywords.get('capabilities', [])[:3])}

Below are candidate GitHub repositories. Select up to {limit + 3} to use as real-world reference
implementations for this idea, ordered best first.

{candidates_text}

Judge in this order:
1. Domain match: does the repository work on the same subject as the idea (the same kind of data,
   users and problem), not just use the same technique? For example, for a job-skills trend tool a
   job-listings scraper or skills dashboard is a domain match; a generic time-series library is not.
2. Coverage of the key capabilities.
3. Real, runnable implementation (not an awesome-list, tutorial collection or empty placeholder).

Rules:
- "end_to_end" only if domain_match is true and it implements most of the idea.
- A generic library or framework that only matches the technique is "pattern" with fit_score <= 0.6.
- Prefer a smaller domain-matching project over a famous technique-only library.
- Leave out awesome-lists, course collections and placeholder repos entirely.

For each selected repository return:
- full_name: exact repo full_name as given above
- domain_match: true or false (see rule 1)
- fit_score: 0.0-1.0, how well it fits as a reference for this idea
- reference_type: "end_to_end", "subsystem" (implements one important part), or "pattern" (useful techniques only)
- fit_summary: one sentence explaining why it was selected, grounded in its README/files
- matched_capabilities: list of capability names from "Key capabilities" that this repo demonstrates

Return ONLY:
{{"repositories": [{{"full_name": "...", "domain_match": true, "fit_score": 0.0, "reference_type": "...", "fit_summary": "...", "matched_capabilities": []}}]}}"""

        result = await self._call_json_api(
            prompt,
            temperature=0.2,
            max_tokens=1500,
            timeout_seconds=self.settings.LLM_AUX_TIMEOUT_SECONDS,
            prefer="gemini",
        )
        if isinstance(result, dict):
            repositories = result.get("repositories", [])
        elif isinstance(result, list):
            repositories = result
        else:
            repositories = []
        return [item for item in repositories if isinstance(item, dict) and item.get("full_name")]

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
            json_mode=False,
        )
        return self._coerce_json_list(queries)
    
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

    def _format_evidence(self, evidence: dict | None) -> str:
        if not evidence:
            return "No additional grounded evidence was available beyond the shortlisted repositories."
        # Compact JSON: indentation alone used to eat a third of the budget, and the
        # old 4,000-char cut left later repos with no evidence at all.
        return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))[:16000]

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
