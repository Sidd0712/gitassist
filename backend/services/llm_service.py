"""Idea-intent extraction and idea-first analysis synthesis."""

from __future__ import annotations

import logging
import re
from collections import Counter

from models.schemas import (
    AmbiguityFlag,
    AnalysisResponse,
    ClarificationQuestion,
    ExtractedKeywords,
    LearningStep,
    RepoSearchResult,
    RetrievalHit,
    RetrievalPlan,
    RetrievalQuery,
    TechRecommendation,
)
from services.rag.citation_service import build_analysis_evidence

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "app",
    "application",
    "build",
    "create",
    "for",
    "from",
    "idea",
    "in",
    "into",
    "like",
    "make",
    "of",
    "platform",
    "project",
    "site",
    "something",
    "system",
    "that",
    "the",
    "tool",
    "using",
    "users",
    "web",
    "website",
    "with",
}

PRODUCT_TYPE_HINTS = [
    ("meal prep app", ("meal prep", "meal plan", "meal planning", "weekly meals", "everyday meals")),
    ("recipe recommendation app", ("recipe", "recipes", "fridge", "pantry", "ingredients")),
    ("collaborative whiteboard", ("whiteboard", "draw", "drawing", "sketch", "canvas")),
    ("chat application", ("chat", "message", "messaging", "conversation")),
    ("booking platform", ("booking", "appointment", "schedule", "reservation")),
    ("marketplace platform", ("marketplace", "buy", "sell", "hire", "vendor", "merchant", "uber for")),
    ("e-commerce storefront", ("cart", "checkout", "catalog", "store", "shop", "order")),
    ("analytics dashboard", ("analytics", "dashboard", "metrics", "reporting")),
    ("content platform", ("blog", "cms", "publish", "editor", "article")),
    ("video collaboration app", ("video", "meeting", "call", "stream", "audio")),
    ("developer CLI", ("cli", "command line", "terminal")),
    ("API service", ("api", "sdk", "service", "backend only")),
]

CAPABILITY_CATALOG: dict[str, dict[str, list[str] | str]] = {
    "ingredient inventory": {
        "aliases": ["fridge", "pantry", "ingredients", "ingredient", "available ingredients", "what's in your fridge", "what is in your fridge", "leftovers"],
        "components": ["ingredient inventory", "ingredient normalization service"],
        "integrations": [],
        "stack_families": ["inventory tracking"],
        "frameworks": [],
        "keywords": ["pantry inventory", "available ingredients", "ingredient matching"],
    },
    "recipe recommendation": {
        "aliases": ["recipe", "recipes", "meal suggestions", "meal ideas", "suggest recipe", "suggest recipes", "what can i cook", "cook with"],
        "components": ["recipe catalog", "recipe recommendation engine"],
        "integrations": ["recipe dataset or API"],
        "stack_families": ["search and recommendation"],
        "frameworks": ["PostgreSQL search"],
        "keywords": ["recipe matching", "ingredient-based recommendations", "recipe ranking"],
    },
    "meal planning": {
        "aliases": ["meal prep", "meal plan", "meal planning", "everyday meals", "daily meals", "weekly meals"],
        "components": ["meal planner workflow", "prep schedule"],
        "integrations": [],
        "stack_families": ["planning workflow"],
        "frameworks": [],
        "keywords": ["weekly meal plan", "meal prep schedule"],
    },
    "grocery planning": {
        "aliases": ["grocery", "shopping list", "shopping", "restock"],
        "components": ["grocery list generator"],
        "integrations": [],
        "stack_families": ["shopping list workflow"],
        "frameworks": [],
        "keywords": ["grocery planning", "shopping list"],
    },
    "nutrition tracking": {
        "aliases": ["calories", "macros", "nutrition", "diet", "protein", "carbs", "meal goals"],
        "components": ["nutrition scoring"],
        "integrations": ["nutrition dataset or API"],
        "stack_families": ["nutrition analysis"],
        "frameworks": [],
        "keywords": ["macro tracking", "nutrition scoring"],
    },
    "realtime collaboration": {
        "aliases": ["collaborative", "collaboration", "live", "real time", "real-time", "realtime", "share", "shared", "sync", "together", "presence", "multi-user"],
        "components": ["realtime gateway", "presence/session manager"],
        "integrations": [],
        "stack_families": ["websocket transport", "pub-sub fanout"],
        "frameworks": ["WebSocket", "Socket.IO"],
        "keywords": ["realtime sync", "presence", "shared state"],
    },
    "canvas rendering": {
        "aliases": ["whiteboard", "draw", "drawing", "sketch", "canvas", "annotate"],
        "components": ["canvas rendering layer", "drawing tool state"],
        "integrations": [],
        "stack_families": ["browser canvas", "scene graph"],
        "frameworks": ["Canvas API", "Fabric.js", "Konva"],
        "keywords": ["canvas rendering", "drawing tools", "stroke history"],
    },
    "chat messaging": {
        "aliases": ["chat", "message", "messaging", "conversation", "dm", "group chat"],
        "components": ["message service", "conversation state"],
        "integrations": ["push notifications"],
        "stack_families": ["message event stream"],
        "frameworks": ["WebSocket", "Socket.IO"],
        "keywords": ["message delivery", "conversation history"],
    },
    "booking and scheduling": {
        "aliases": ["booking", "appointment", "schedule", "reservation", "calendar", "availability"],
        "components": ["availability engine", "booking workflow"],
        "integrations": ["calendar integration", "email notifications"],
        "stack_families": ["transactional API", "calendar sync"],
        "frameworks": ["PostgreSQL", "Calendar API"],
        "keywords": ["availability slots", "booking flow", "rescheduling"],
    },
    "marketplace matching": {
        "aliases": ["marketplace", "hire", "buyer", "seller", "vendor", "uber for", "airbnb for"],
        "components": ["listing and discovery", "matching workflow", "multi-role access"],
        "integrations": [],
        "stack_families": ["search and discovery"],
        "frameworks": ["PostgreSQL"],
        "keywords": ["two-sided marketplace", "provider matching", "listing search"],
    },
    "authentication": {
        "aliases": ["auth", "authentication", "login", "signup", "account", "user account", "sign in"],
        "components": ["auth/session service", "user profile store"],
        "integrations": ["auth provider"],
        "stack_families": ["session management"],
        "frameworks": ["JWT", "Auth.js", "Clerk"],
        "keywords": ["login flow", "session auth", "role-based access"],
    },
    "roles and permissions": {
        "aliases": ["role", "roles", "permission", "admin", "moderator", "teacher", "student", "tutor", "trainer"],
        "components": ["role-based access control"],
        "integrations": [],
        "stack_families": ["RBAC"],
        "frameworks": ["RBAC"],
        "keywords": ["user roles", "access control"],
    },
    "persistence": {
        "aliases": ["save", "stored", "history", "database", "persist", "saved", "draft", "records"],
        "components": ["primary database"],
        "integrations": [],
        "stack_families": ["transactional database"],
        "frameworks": ["PostgreSQL"],
        "keywords": ["data persistence", "history", "storage"],
    },
    "payments and billing": {
        "aliases": ["payment", "payments", "billing", "checkout", "subscription", "invoice", "wallet", "commission"],
        "components": ["billing service", "checkout flow"],
        "integrations": ["payment gateway"],
        "stack_families": ["payment processing"],
        "frameworks": ["Stripe"],
        "keywords": ["payment flow", "billing", "commission"],
    },
    "ai features": {
        "aliases": ["ai", "llm", "gpt", "openai", "chatbot", "copilot", "rag", "embedding", "vector search"],
        "components": ["AI inference service", "prompt orchestration"],
        "integrations": ["LLM provider"],
        "stack_families": ["model inference API"],
        "frameworks": ["OpenAI API", "LangChain"],
        "keywords": ["AI inference", "prompting", "model integration"],
    },
    "file uploads": {
        "aliases": ["upload", "uploads", "attachment", "attachments", "file", "document", "image", "media"],
        "components": ["file upload pipeline", "asset storage"],
        "integrations": ["object storage"],
        "stack_families": ["blob storage"],
        "frameworks": ["S3-compatible storage"],
        "keywords": ["asset upload", "file storage"],
    },
    "search and filtering": {
        "aliases": ["search", "filter", "discover", "discovery", "browse", "find"],
        "components": ["search service", "filter/query layer"],
        "integrations": [],
        "stack_families": ["full-text search"],
        "frameworks": ["PostgreSQL search", "Elasticsearch"],
        "keywords": ["search index", "filtering", "ranking"],
    },
    "maps and geolocation": {
        "aliases": ["map", "maps", "location", "route", "nearby", "geo", "gps", "geolocation"],
        "components": ["location service", "map visualization"],
        "integrations": ["maps API"],
        "stack_families": ["geospatial search"],
        "frameworks": ["Mapbox", "Google Maps"],
        "keywords": ["geolocation", "map rendering", "route lookup"],
    },
    "notifications": {
        "aliases": ["notification", "notifications", "email", "sms", "push", "alert", "alerts"],
        "components": ["notification dispatcher"],
        "integrations": ["email provider", "push provider"],
        "stack_families": ["async job delivery"],
        "frameworks": ["Resend", "Firebase Cloud Messaging"],
        "keywords": ["notification delivery", "email alerts"],
    },
    "video or voice": {
        "aliases": ["video", "call", "meeting", "audio", "voice", "stream"],
        "components": ["media session service", "realtime media transport"],
        "integrations": ["TURN/STUN service"],
        "stack_families": ["WebRTC media transport"],
        "frameworks": ["WebRTC"],
        "keywords": ["video session", "media signaling"],
    },
    "analytics": {
        "aliases": ["analytics", "dashboard", "metrics", "report", "reporting", "insights"],
        "components": ["analytics pipeline", "dashboard views"],
        "integrations": [],
        "stack_families": ["analytical queries"],
        "frameworks": ["PostgreSQL", "ClickHouse"],
        "keywords": ["metrics dashboard", "reporting"],
    },
}

EXPLICIT_FRAMEWORKS = {
    "react": "React",
    "next": "Next.js",
    "next.js": "Next.js",
    "vue": "Vue",
    "angular": "Angular",
    "svelte": "Svelte",
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "express": "Express",
    "nestjs": "NestJS",
    "socket.io": "Socket.IO",
    "websocket": "WebSocket",
    "webrtc": "WebRTC",
    "canvas": "Canvas API",
    "fabric.js": "Fabric.js",
    "konva": "Konva",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "stripe": "Stripe",
    "openai": "OpenAI API",
}

EXPLICIT_LANGUAGES = {
    "python": "Python",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "go": "Go",
    "rust": "Rust",
    "java": "Java",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "php": "PHP",
    "ruby": "Ruby",
    "c#": "C#",
}

QUESTION_BANK = {
    "platform": ClarificationQuestion(
        key="platform",
        question="What should the primary product surface be?",
        options=["Web app", "Mobile app", "Web + mobile", "API/service only", "CLI tool"],
        reason="The platform choice changes the frontend stack, client architecture, and how we evaluate repo references.",
    ),
    "interaction_model": ClarificationQuestion(
        key="interaction_model",
        question="Should the core experience update live for multiple users, or can it be asynchronous?",
        options=["Realtime multi-user", "Mostly asynchronous", "Both realtime and async"],
        reason="This determines whether the design needs a realtime transport and shared state synchronization.",
    ),
    "payments": ClarificationQuestion(
        key="payments",
        question="Will money move through the product itself?",
        options=["Yes, built-in payments", "No payments needed", "Maybe later"],
        reason="Payments and payouts add major workflow, compliance, and integration requirements.",
    ),
    "auth": ClarificationQuestion(
        key="auth",
        question="Do users need accounts and sign-in from day one?",
        options=["Yes, required", "Optional", "No, public-first"],
        reason="Auth affects the data model, permissions, and repository patterns we should prioritize.",
    ),
    "inventory_input": ClarificationQuestion(
        key="inventory_input",
        question="How will users tell the app what ingredients they have?",
        options=["Manual ingredient list", "Photo or barcode scan", "Both manual and scan"],
        reason="Ingredient input changes the client UX, integrations, and whether you need image or barcode processing.",
    ),
    "recommendation_mode": ClarificationQuestion(
        key="recommendation_mode",
        question="How should meal suggestions be produced?",
        options=["Match recipes from a database", "AI-generated meal suggestions", "Both curated recipes and AI"],
        reason="This decides whether the product needs a recipe dataset, an LLM integration, or both.",
    ),
    "planning_scope": ClarificationQuestion(
        key="planning_scope",
        question="What should the first version optimize for?",
        options=["Suggest the next meal", "Plan a full day", "Plan a full week"],
        reason="A single-meal recommender is a much smaller first build than a full prep planner with schedules and grocery flows.",
    ),
}

BANNED_ARCHITECTURE_TERMS = {"rag", "vector db", "retrieval", "embedding", "chunking", "corpus"}


async def extract_keywords(idea: str, clarification_answers: dict[str, str] | None = None) -> ExtractedKeywords:
    """Extract structured technical intent from the user's idea."""

    clarification_answers = clarification_answers or {}
    normalized = _normalize_text(idea, clarification_answers)
    tokens = _tokenize(normalized)

    capabilities = _detect_capabilities(idea, clarification_answers)
    frameworks = _detect_explicit_frameworks(normalized)
    languages = _detect_explicit_languages(normalized)
    product_type = _detect_product_type(normalized, capabilities)
    platform = _detect_platform(normalized, clarification_answers)
    target_users = _detect_target_users(normalized)
    constraints = _detect_constraints(normalized, clarification_answers)
    likely_components = _derive_components(capabilities, platform, clarification_answers)
    likely_integrations = _derive_integrations(capabilities, clarification_answers)
    likely_stack_families = _derive_stack_families(platform, capabilities, constraints, clarification_answers)
    assumptions = _derive_assumptions(platform, capabilities, clarification_answers)
    ambiguities = _detect_ambiguities(
        idea=idea,
        normalized=normalized,
        clarification_answers=clarification_answers,
        platform=platform,
        capabilities=capabilities,
    )
    summary = _build_summary(product_type, platform, capabilities, target_users)
    keywords = _build_keywords(tokens, product_type, capabilities, likely_stack_families)

    for capability in capabilities:
        frameworks.extend(_catalog_list(capability, "frameworks"))
    frameworks = _dedupe_preserve(frameworks)[:8]

    result = ExtractedKeywords(
        keywords=keywords[:12],
        frameworks=frameworks,
        languages=languages[:4],
        summary=summary,
        product_type=product_type,
        target_users=target_users[:4],
        capabilities=capabilities[:12],
        constraints=constraints[:8],
        likely_components=likely_components[:10],
        likely_integrations=likely_integrations[:8],
        likely_stack_families=likely_stack_families[:10],
        ambiguities=ambiguities,
        assumptions=assumptions[:8],
    )

    logger.info(
        "Intent extraction complete: %d capabilities, %d frameworks, %d ambiguities",
        len(result.capabilities),
        len(result.frameworks),
        len([item for item in result.ambiguities if not item.resolved]),
    )
    return result


def build_clarification_questions(keywords: ExtractedKeywords) -> list[ClarificationQuestion]:
    """Return high-priority clarification questions for unresolved build choices."""

    questions: list[ClarificationQuestion] = []
    for ambiguity in keywords.ambiguities:
        if ambiguity.resolved or ambiguity.severity != "high":
            continue
        question = QUESTION_BANK.get(ambiguity.axis)
        if question is None:
            continue
        questions.append(question)
        if len(questions) >= 4:
            break
    return questions


async def plan_retrieval_queries(
    idea: str,
    keywords: ExtractedKeywords,
    repositories: list[RepoSearchResult],
) -> RetrievalPlan:
    """Create an idea-first retrieval plan for each output section."""

    repo_hint = ", ".join(repo.reference_type for repo in repositories[:4] if repo.reference_type != "candidate")
    summary = keywords.summary or idea
    capability_scope = ", ".join(keywords.capabilities[:5] or keywords.keywords[:5])
    stack_scope = ", ".join(keywords.likely_stack_families[:4] or keywords.frameworks[:4])
    architecture_scope = ", ".join(keywords.likely_components[:5] or keywords.capabilities[:4])

    return RetrievalPlan(
        queries=[
            RetrievalQuery(
                section="repo_descriptions",
                query=(
                    f"{summary}. Explain why a repository is useful as an implementation reference for "
                    f"{capability_scope}. Prioritize end-to-end matches before subsystem patterns. {repo_hint}"
                ),
                preferred_roles=["documentation", "entrypoint", "source", "config"],
                top_k=8,
            ),
            RetrievalQuery(
                section="learning_path",
                query=(
                    f"{summary}. Focus on setup docs, examples, onboarding notes, and implementation milestones "
                    f"for {capability_scope}."
                ),
                preferred_roles=["documentation", "example", "config", "source"],
                top_k=8,
            ),
            RetrievalQuery(
                section="architecture_diagram",
                query=(
                    f"{summary}. Focus on components, API boundaries, realtime flows, data persistence, auth, "
                    f"and integrations for {architecture_scope}."
                ),
                preferred_roles=["entrypoint", "source", "config", "documentation"],
                top_k=8,
            ),
            RetrievalQuery(
                section="tech_stack",
                query=(
                    f"{summary}. Focus on dependencies, manifests, deployment choices, and technology decisions "
                    f"for {stack_scope}."
                ),
                preferred_roles=["config", "documentation", "entrypoint"],
                top_k=8,
            ),
        ]
    )


async def generate_analysis(
    idea: str,
    keywords: ExtractedKeywords,
    repositories: list[RepoSearchResult],
    section_hits: dict[str, list[RetrievalHit]],
) -> AnalysisResponse:
    """Generate an idea-first response grounded by repository evidence."""

    repo_descriptions = _build_repo_descriptions(keywords, repositories, section_hits)
    learning_path = _build_learning_path(keywords, repositories)
    architecture_diagram = _build_architecture_diagram(keywords)
    tech_stack = _build_tech_stack(keywords, repositories)

    return AnalysisResponse(
        idea_summary=keywords.summary or idea,
        keywords=keywords,
        assumptions=keywords.assumptions,
        repositories=repositories,
        repo_descriptions=repo_descriptions,
        learning_path=learning_path,
        architecture_diagram=architecture_diagram,
        tech_stack=tech_stack,
        evidence=build_analysis_evidence(section_hits),
        status="complete",
    )


def _normalize_text(idea: str, clarification_answers: dict[str, str]) -> str:
    if not clarification_answers:
        return idea.strip()
    answer_suffix = " ".join(f"{key.replace('_', ' ')} {value}" for key, value in clarification_answers.items() if value)
    return f"{idea.strip()} {answer_suffix}".strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9#+./-]+", text.lower()) if len(token) > 2]


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _detect_capabilities(text: str, clarification_answers: dict[str, str]) -> list[str]:
    lowered = text.lower()
    scores: Counter[str] = Counter()

    for capability, metadata in CAPABILITY_CATALOG.items():
        aliases = metadata.get("aliases", [])
        for alias in aliases:
            if alias in lowered:
                scores[capability] += 2 if " " in alias or "-" in alias else 1
        for keyword in metadata.get("keywords", []):
            if keyword in lowered:
                scores[capability] += 2

    if "friends" in lowered or "teams" in lowered or "people" in lowered:
        scores["authentication"] += 1
        scores["roles and permissions"] += 1
    if "online" in lowered:
        scores["persistence"] += 1
    if "uber for" in lowered:
        scores["marketplace matching"] += 3
        scores["payments and billing"] += 1
    if "booking" in lowered and ("trainer" in lowered or "tutor" in lowered or "coach" in lowered):
        scores["marketplace matching"] += 2
        scores["roles and permissions"] += 1
    if "canvas" in lowered and "websocket" in lowered:
        scores["realtime collaboration"] += 2
    if any(term in lowered for term in ("meal", "recipe", "recipes", "fridge", "pantry", "ingredients", "ingredient")):
        scores["recipe recommendation"] += 2
    if any(term in lowered for term in ("fridge", "pantry", "ingredients", "ingredient", "leftovers")):
        scores["ingredient inventory"] += 3
    if any(term in lowered for term in ("meal prep", "meal plan", "daily meals", "weekly meals", "everyday meals")):
        scores["meal planning"] += 3
    if any(term in lowered for term in ("grocery", "shopping list", "restock")):
        scores["grocery planning"] += 2

    recommendation_answer = clarification_answers.get("recommendation_mode", "").lower()
    inventory_answer = clarification_answers.get("inventory_input", "").lower()
    planning_answer = clarification_answers.get("planning_scope", "").lower()
    auth_answer = clarification_answers.get("auth", "").lower()
    payments_answer = clarification_answers.get("payments", "").lower()
    interaction_answer = clarification_answers.get("interaction_model", "").lower()
    if auth_answer.startswith("yes") or auth_answer.startswith("optional"):
        scores["authentication"] += 2 if auth_answer.startswith("yes") else 1
    if payments_answer.startswith("yes"):
        scores["payments and billing"] += 3
    if interaction_answer.startswith("realtime") or interaction_answer.startswith("both realtime"):
        scores["realtime collaboration"] += 2
    if "manual" in inventory_answer or "scan" in inventory_answer or "barcode" in inventory_answer or "photo" in inventory_answer:
        scores["ingredient inventory"] += 2
    if "database" in recommendation_answer or "curated" in recommendation_answer or "recipe" in recommendation_answer:
        scores["recipe recommendation"] += 2
    if "ai" in recommendation_answer:
        scores["ai features"] += 3
    if "scan" in inventory_answer or "barcode" in inventory_answer or "photo" in inventory_answer:
        scores["file uploads"] += 2
    if "week" in planning_answer or "day" in planning_answer:
        scores["meal planning"] += 1

    if not scores:
        generic = []
        if any(token in lowered for token in ("dashboard", "portal", "admin")):
            generic.append("analytics")
        if any(token in lowered for token in ("api", "service")):
            generic.append("persistence")
        return generic

    return [name for name, _score in scores.most_common(10)]


def _detect_explicit_frameworks(text: str) -> list[str]:
    lowered = text.lower()
    frameworks = [canonical for raw, canonical in EXPLICIT_FRAMEWORKS.items() if raw in lowered]
    return _dedupe_preserve(frameworks)


def _detect_explicit_languages(text: str) -> list[str]:
    lowered = text.lower()
    languages = [canonical for raw, canonical in EXPLICIT_LANGUAGES.items() if raw in lowered]
    return _dedupe_preserve(languages)


def _detect_product_type(text: str, capabilities: list[str]) -> str:
    lowered = text.lower()
    for product_type, aliases in PRODUCT_TYPE_HINTS:
        if any(alias in lowered for alias in aliases):
            return product_type
    if "meal planning" in capabilities and "recipe recommendation" in capabilities:
        return "meal prep app"
    if "recipe recommendation" in capabilities and "ingredient inventory" in capabilities:
        return "recipe recommendation app"
    if "canvas rendering" in capabilities and "realtime collaboration" in capabilities:
        return "collaborative whiteboard"
    if "marketplace matching" in capabilities and "booking and scheduling" in capabilities:
        return "booking marketplace"
    if "marketplace matching" in capabilities:
        return "marketplace platform"
    if "booking and scheduling" in capabilities:
        return "booking platform"
    if "chat messaging" in capabilities:
        return "chat application"
    return "software product"


def _detect_platform(text: str, clarification_answers: dict[str, str]) -> str:
    answer = clarification_answers.get("platform", "").lower()
    if answer:
        if "mobile" in answer and "web" in answer:
            return "web and mobile"
        if "mobile" in answer:
            return "mobile"
        if "api" in answer:
            return "api"
        if "cli" in answer:
            return "cli"
        return "web"

    lowered = text.lower()
    if any(token in lowered for token in ("cli", "command line", "terminal")):
        return "cli"
    if any(token in lowered for token in ("ios", "android", "mobile", "react native", "flutter")):
        return "mobile"
    if any(token in lowered for token in ("api", "sdk", "backend only", "service only")):
        return "api"
    if any(token in lowered for token in ("website", "web", "browser", "dashboard", "canvas", "frontend", "react")):
        return "web"
    return ""


def _detect_target_users(text: str) -> list[str]:
    lowered = text.lower()
    targets: list[str] = []
    for term in ("friends", "teams", "students", "teachers", "tutors", "trainers", "admins", "customers", "vendors", "families", "home cooks", "parents"):
        if term in lowered:
            targets.append(term)
    match = re.search(r"\bfor ([a-z][a-z\s-]{2,40})", lowered)
    if match:
        candidate = match.group(1).strip(" .,")
        if not any(token in candidate for token in ("meal", "recipe", "recipes", "fridge", "ingredient", "what", "app", "product")):
            targets.append(candidate)
    return _dedupe_preserve(targets)[:4]


def _detect_constraints(text: str, clarification_answers: dict[str, str]) -> list[str]:
    lowered = text.lower()
    constraints: list[str] = []
    term_map = {
        "realtime": "Realtime updates",
        "real-time": "Realtime updates",
        "live": "Realtime updates",
        "offline": "Offline support",
        "multi-tenant": "Multi-tenant support",
        "encryption": "End-to-end or transport encryption",
        "scalable": "Scalability matters",
        "low latency": "Low latency interactions",
        "responsive": "Responsive UX",
    }
    for term, label in term_map.items():
        if term in lowered:
            constraints.append(label)

    if clarification_answers.get("interaction_model", "").lower().startswith("realtime"):
        constraints.append("Realtime updates")
    if clarification_answers.get("payments", "").lower().startswith("yes"):
        constraints.append("Integrated payments")
    if clarification_answers.get("auth", "").lower().startswith("yes"):
        constraints.append("Authenticated user flows")
    if "photo" in clarification_answers.get("inventory_input", "").lower() or "barcode" in clarification_answers.get("inventory_input", "").lower():
        constraints.append("Ingredient scanning support")
    if "week" in clarification_answers.get("planning_scope", "").lower():
        constraints.append("Weekly meal planning")
    return _dedupe_preserve(constraints)


def _derive_components(capabilities: list[str], platform: str, clarification_answers: dict[str, str]) -> list[str]:
    components = ["frontend client" if platform in {"web", "web and mobile", ""} else "client interface"]
    if platform == "mobile":
        components[0] = "mobile client"
    elif platform == "api":
        components[0] = "API consumers"
    elif platform == "cli":
        components[0] = "CLI interface"

    if platform != "cli":
        components.append("backend API")

    for capability in capabilities:
        components.extend(_catalog_list(capability, "components"))
    inventory_answer = clarification_answers.get("inventory_input", "").lower()
    if "scan" in inventory_answer or "barcode" in inventory_answer or "photo" in inventory_answer:
        components.append("ingredient scan pipeline")
    if capabilities and "persistence" not in capabilities:
        components.append("primary database")
    return _dedupe_preserve(components)


def _derive_integrations(capabilities: list[str], clarification_answers: dict[str, str]) -> list[str]:
    integrations: list[str] = []
    for capability in capabilities:
        integrations.extend(_catalog_list(capability, "integrations"))
    recommendation_answer = clarification_answers.get("recommendation_mode", "").lower()
    inventory_answer = clarification_answers.get("inventory_input", "").lower()
    if "ai" in recommendation_answer:
        integrations.append("LLM provider")
    if "scan" in inventory_answer or "barcode" in inventory_answer or "photo" in inventory_answer:
        integrations.append("barcode or OCR service")
    return _dedupe_preserve(integrations)


def _derive_stack_families(
    platform: str,
    capabilities: list[str],
    constraints: list[str],
    clarification_answers: dict[str, str],
) -> list[str]:
    families: list[str] = []
    if platform in {"", "web", "web and mobile"}:
        families.append("interactive web client")
    if platform in {"mobile", "web and mobile"}:
        families.append("mobile client")
    if platform != "cli":
        families.append("API backend")
    if platform == "cli":
        families.append("command-line runtime")
    for capability in capabilities:
        families.extend(_catalog_list(capability, "stack_families"))
    recommendation_answer = clarification_answers.get("recommendation_mode", "").lower()
    if "ai" in recommendation_answer:
        families.append("model inference API")
    if "scan" in clarification_answers.get("inventory_input", "").lower():
        families.append("vision-assisted input")
    if constraints:
        families.append("observability and deployment")
    return _dedupe_preserve(families)


def _derive_assumptions(platform: str, capabilities: list[str], clarification_answers: dict[str, str]) -> list[str]:
    assumptions: list[str] = []
    if not clarification_answers.get("platform"):
        if platform == "web":
            assumptions.append("Assuming a browser-first web application unless stated otherwise.")
        elif not platform:
            assumptions.append("Assuming a web-first product because no platform was specified.")

    collaborative_caps = {"realtime collaboration", "chat messaging", "marketplace matching", "booking and scheduling"}
    if collaborative_caps.intersection(capabilities) and not clarification_answers.get("auth"):
        assumptions.append("Assuming user accounts will be needed because the workflow involves shared or role-based data.")
    if any(cap in capabilities for cap in ("canvas rendering", "booking and scheduling", "chat messaging")):
        assumptions.append("Assuming the product should persist its main user data instead of being session-only.")
    if "recipe recommendation" in capabilities and not clarification_answers.get("recommendation_mode"):
        assumptions.append("Assuming the first version should match against a recipe catalog before adding AI-generated suggestions.")
    if "ingredient inventory" in capabilities and not clarification_answers.get("inventory_input"):
        assumptions.append("Assuming ingredients can be entered manually in the first version before adding barcode or photo scanning.")
    return _dedupe_preserve(assumptions)


def _detect_ambiguities(
    *,
    idea: str,
    normalized: str,
    clarification_answers: dict[str, str],
    platform: str,
    capabilities: list[str],
) -> list[AmbiguityFlag]:
    lowered = normalized.lower()
    technical_signal = len(_detect_explicit_frameworks(normalized)) + len(_detect_explicit_languages(normalized))
    specific_signal = len(capabilities) + technical_signal
    ambiguities: list[AmbiguityFlag] = []

    def add(axis: str, reason: str, severity: str = "high") -> None:
        answer = clarification_answers.get(axis)
        ambiguities.append(
            AmbiguityFlag(
                axis=axis,
                reason=reason,
                severity=severity,
                resolved=bool(answer),
                answer=answer or None,
            )
        )

    if not platform and specific_signal < 4:
        add("platform", "The idea does not clearly say whether this should be web, mobile, API-only, or CLI.", "high")

    collaboration_like = {"realtime collaboration", "chat messaging", "video or voice"}
    if collaboration_like.intersection(capabilities):
        explicit_realtime = any(
            term in lowered
            for term in ("real time", "real-time", "realtime", "live", "websocket", "socket.io", "webrtc", "together online")
        ) or ("together" in lowered and "online" in lowered)
        if not explicit_realtime and not clarification_answers.get("interaction_model"):
            add(
                "interaction_model",
                "The idea implies shared interaction, but it does not confirm whether updates must happen live.",
                "high",
            )

    explicit_payment = any(term in lowered for term in ("payment", "payments", "billing", "checkout", "subscription", "invoice", "commission"))
    commerce_like = {"marketplace matching", "payments and billing", "booking and scheduling"}
    if "marketplace matching" in capabilities and not explicit_payment and not clarification_answers.get("payments"):
        add("payments", "Marketplace-style products often need payments or payouts, but that is not confirmed here.", "high")
    elif commerce_like.intersection(capabilities) and not clarification_answers.get("payments") and "subscription" in lowered:
        add("payments", "Billing seems likely, but the exact payment scope is not confirmed.", "high")

    multi_user_like = {"marketplace matching", "booking and scheduling"}
    explicit_auth = any(term in lowered for term in ("auth", "login", "signup", "account", "sign in"))
    if multi_user_like.intersection(capabilities) and not explicit_auth and not clarification_answers.get("auth") and specific_signal < 6:
        add("auth", "Multi-user workflows usually need sign-in and permissions, but that is not explicit.", "high")

    food_like = {"ingredient inventory", "recipe recommendation", "meal planning"}
    explicit_scan = any(term in lowered for term in ("barcode", "scan", "photo of", "camera", "image upload"))
    explicit_ai = any(term in lowered for term in ("ai", "llm", "gpt", "openai"))
    explicit_planning_scope = any(term in lowered for term in ("next meal", "single meal", "daily meals", "weekly meals", "meal prep"))
    if food_like.intersection(capabilities) and not clarification_answers.get("inventory_input") and not explicit_scan:
        add("inventory_input", "The fridge or pantry flow is central, but it is unclear whether users enter ingredients manually or by scanning.", "high")
    if "recipe recommendation" in capabilities and not clarification_answers.get("recommendation_mode") and not explicit_ai:
        add("recommendation_mode", "Recipe suggestions could come from ingredient matching, AI generation, or both, and that changes the implementation.", "high")
    if "meal planning" in capabilities and not clarification_answers.get("planning_scope") and not explicit_planning_scope:
        add("planning_scope", "Meal prep products can optimize for the next meal, a day plan, or a weekly planner, which changes scope a lot.", "high")

    return ambiguities


def _build_summary(product_type: str, platform: str, capabilities: list[str], target_users: list[str]) -> str:
    audience = f" for {', '.join(target_users[:2])}" if target_users else ""
    capability_text = ", ".join(capabilities[:3]) if capabilities else "core product workflows"
    platform_text = f"{platform} " if platform else ""
    return f"A {platform_text}{product_type}{audience} centered on {capability_text}."


def _build_keywords(tokens: list[str], product_type: str, capabilities: list[str], stack_families: list[str]) -> list[str]:
    keywords: list[str] = []
    if product_type and product_type != "software product":
        keywords.append(product_type)
    keywords.extend(capabilities)
    keywords.extend(token for token in tokens if token not in STOPWORDS)
    keywords.extend(stack_families[:4])
    return _dedupe_preserve(keywords)


def _catalog_list(capability: str, field: str) -> list[str]:
    metadata = CAPABILITY_CATALOG.get(capability, {})
    values = metadata.get(field, [])
    return list(values) if isinstance(values, list) else []


def _build_repo_descriptions(
    keywords: ExtractedKeywords,
    repositories: list[RepoSearchResult],
    section_hits: dict[str, list[RetrievalHit]],
) -> list[str]:
    hit_map: dict[str, list[RetrievalHit]] = {}
    for hit in section_hits.get("repo_descriptions", []):
        hit_map.setdefault(hit.repo_full_name, []).append(hit)

    descriptions: list[str] = []
    for repo in repositories:
        capability_text = ", ".join(repo.rank_reasons[:2] or keywords.capabilities[:2])
        top_hit = hit_map.get(repo.full_name, [])
        evidence_path = f" Key evidence appears in `{top_hit[0].path}`." if top_hit else ""
        reference_type = repo.reference_type.replace("_", " ")
        descriptions.append(
            (
                f"{repo.full_name} is a {reference_type} reference for this idea. "
                f"{repo.fit_summary or 'It matches the product through capability overlap and implementation coverage.'} "
                f"{capability_text}.{evidence_path}"
            ).strip()
        )
    return descriptions


def _build_learning_path(keywords: ExtractedKeywords, repositories: list[RepoSearchResult]) -> list[LearningStep]:
    repo_resources = [repo.html_url for repo in repositories[:3]]
    if "recipe recommendation" in keywords.capabilities or "ingredient inventory" in keywords.capabilities:
        return [
            LearningStep(
                step_number=1,
                title="Model ingredients, recipes, and meal plans",
                milestone="Define the food domain and user flow",
                description=(
                    "Design the core entities first: ingredient inventory, normalized ingredient names, recipes, servings, "
                    "saved meals, and optional prep plans."
                ),
                concepts=["ingredient normalization", "recipe schema", "user flow mapping"],
                resources=repo_resources[:1],
            ),
            LearningStep(
                step_number=2,
                title="Build ingredient capture and pantry management",
                milestone="Let users record what they have at home",
                description=(
                    "Start with manual ingredient entry, quantity tracking, expiry notes, and pantry categories before "
                    "adding scan-based input."
                ),
                concepts=["forms", "inventory CRUD", "unit handling", "searchable pantry state"],
                resources=repo_resources[:2],
            ),
            LearningStep(
                step_number=3,
                title="Implement recipe matching and ranking",
                milestone="Suggest meals from available ingredients",
                description=(
                    "Ship a rule-based recipe matcher first: filter by available ingredients, rank by coverage, and show "
                    "what is missing for each recipe."
                ),
                concepts=["matching rules", "ranking heuristics", "recipe filters", "ingredient coverage"],
                resources=repo_resources[:2],
            ),
            LearningStep(
                step_number=4,
                title="Add meal prep planning",
                milestone="Turn suggestions into a repeatable plan",
                description=(
                    "Layer in saved meal plans, prep schedules, and grocery list generation so the product becomes useful "
                    "for day-to-day planning rather than one-off suggestions."
                ),
                concepts=["planner workflows", "grocery list generation", "calendar state"],
                resources=repo_resources[:3],
            ),
            LearningStep(
                step_number=5,
                title="Harden personalization and integrations",
                milestone="Improve relevance and production readiness",
                description=(
                    "Add dietary preferences, auth, notifications, and optional OCR or AI enhancements only after the "
                    "ingredient-to-recipe loop is solid."
                ),
                concepts=["preferences", "auth", "notifications", "optional scanning or AI"],
                resources=repo_resources,
            ),
        ]

    primary_cap = keywords.capabilities[0] if keywords.capabilities else "the core product workflow"
    secondary_cap = keywords.capabilities[1] if len(keywords.capabilities) > 1 else "supporting user flows"
    tertiary_cap = keywords.capabilities[2] if len(keywords.capabilities) > 2 else "production hardening"

    return [
        LearningStep(
            step_number=1,
            title="Define the core user journey",
            milestone="Map the product flow and data model",
            description=(
                f"Write down the main actors, screens, and data entities for the {keywords.product_type or 'product'} "
                f"before choosing implementation details."
            ),
            concepts=["user flows", "domain model", "success metrics"],
            resources=repo_resources[:1],
        ),
        LearningStep(
            step_number=2,
            title="Build the application shell",
            milestone="Stand up the client, API, and persistence foundation",
            description=(
                "Create the base frontend and backend structure, then model the persistence layer around the main entities "
                "you identified in step 1."
            ),
            concepts=["routing", "API contracts", "database schema"],
            resources=repo_resources[:2],
        ),
        LearningStep(
            step_number=3,
            title=f"Implement {primary_cap}",
            milestone="Ship the smallest end-to-end version of the primary workflow",
            description=(
                f"Focus on the first version of {primary_cap} with realistic input validation, persistence, and the "
                "minimum UI needed for real users."
            ),
            concepts=keywords.capabilities[:2] or ["feature slicing", "vertical implementation"],
            resources=repo_resources[:2],
        ),
        LearningStep(
            step_number=4,
            title=f"Add {secondary_cap}",
            milestone="Layer on the next capability without rewriting the foundation",
            description=(
                f"Introduce {secondary_cap} using a clean service boundary so the product can grow while staying easy to test."
            ),
            concepts=keywords.capabilities[1:4] or ["modular architecture", "integration design"],
            resources=repo_resources[:3],
        ),
        LearningStep(
            step_number=5,
            title="Harden for production",
            milestone="Prepare deployment, observability, and scale paths",
            description=(
                f"Add authentication, monitoring, deployment automation, and failure handling around {tertiary_cap} "
                "before broadening the feature set."
            ),
            concepts=["auth", "deployment", "monitoring", "testing"],
            resources=repo_resources,
        ),
    ]


def _build_architecture_diagram(keywords: ExtractedKeywords) -> str:
    platform = _detect_platform_from_summary(keywords)
    lines = ["flowchart TD"]

    if platform == "mobile":
        lines.append('Client["Mobile App"]')
    elif platform == "cli":
        lines.append('Client["CLI Client"]')
    elif platform == "api":
        lines.append('Client["API Consumer"]')
    else:
        lines.append('Client["Web Client"]')

    if platform != "cli":
        lines.append('API["Backend API"]')
        lines.append("Client --> API")

    components = set(component.lower() for component in keywords.likely_components)

    if any("auth" in component for component in components):
        lines.append('Auth["Auth / Session Service"]')
        lines.append("Client --> Auth")
        if platform != "cli":
            lines.append("API --> Auth")
    if any("realtime" in component or "presence" in component for component in components):
        lines.append('Realtime["Realtime Gateway"]')
        lines.append("Client <--> Realtime")
        if platform != "cli":
            lines.append("Realtime --> API")
    if any("ingredient inventory" in component or "ingredient normalization" in component for component in components):
        lines.append('Inventory["Ingredient Inventory Service"]')
        if platform != "cli":
            lines.append("API --> Inventory")
        else:
            lines.append("Client --> Inventory")
    if any("recipe catalog" in component or "recipe recommendation" in component for component in components):
        lines.append('Recipes["Recipe Catalog / Matching Engine"]')
        if platform != "cli":
            lines.append("API --> Recipes")
        else:
            lines.append("Client --> Recipes")
    if any("meal planner" in component or "prep schedule" in component for component in components):
        lines.append('Planner["Meal Planner Workflow"]')
        if platform != "cli":
            lines.append("API --> Planner")
    if any("grocery list" in component for component in components):
        lines.append('Groceries["Grocery List Generator"]')
        if platform != "cli":
            lines.append("API --> Groceries")
    if any("canvas" in component or "drawing" in component for component in components):
        lines.append('Canvas["Canvas / Interaction Layer"]')
        lines.append("Client --> Canvas")
    if platform != "cli":
        lines.append('DB["Primary Database"]')
        lines.append("API --> DB")
    if any("storage" in component or "asset" in component for component in components):
        lines.append('Storage["Object Storage"]')
        lines.append("API --> Storage")
    if "payment gateway" in [integration.lower() for integration in keywords.likely_integrations]:
        lines.append('Payments["Payment Gateway"]')
        if platform != "cli":
            lines.append("API --> Payments")
    if "recipe dataset or api" in [integration.lower() for integration in keywords.likely_integrations]:
        lines.append('RecipeAPI["Recipe Dataset / API"]')
        if platform != "cli":
            lines.append("Recipes --> RecipeAPI")
    if "nutrition dataset or api" in [integration.lower() for integration in keywords.likely_integrations]:
        lines.append('NutritionAPI["Nutrition Data API"]')
        if platform != "cli":
            lines.append("API --> NutritionAPI")
    if "barcode or ocr service" in [integration.lower() for integration in keywords.likely_integrations]:
        lines.append('Scanner["OCR / Barcode Service"]')
        if platform != "cli":
            lines.append("Client --> Scanner")
    if "llm provider" in [integration.lower() for integration in keywords.likely_integrations]:
        lines.append('AI["AI Inference Provider"]')
        if platform != "cli":
            lines.append("API --> AI")
    if "maps api" in [integration.lower() for integration in keywords.likely_integrations]:
        lines.append('Maps["Maps / Geolocation API"]')
        if platform != "cli":
            lines.append("API --> Maps")
    if any(integration.lower() in {"email provider", "push provider", "calendar integration"} for integration in keywords.likely_integrations):
        lines.append('Jobs["Async Jobs / Notifications"]')
        if platform != "cli":
            lines.append("API --> Jobs")

    return "\n".join(_guard_architecture_lines(lines))


def _detect_platform_from_summary(keywords: ExtractedKeywords) -> str:
    summary = keywords.summary.lower()
    if "mobile" in summary:
        return "mobile"
    if "cli" in summary:
        return "cli"
    if "api" in summary and "web" not in summary:
        return "api"
    return "web"


def _guard_architecture_lines(lines: list[str]) -> list[str]:
    guarded: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(term in lowered for term in BANNED_ARCHITECTURE_TERMS):
            continue
        guarded.append(line)
    return guarded


def _build_tech_stack(keywords: ExtractedKeywords, repositories: list[RepoSearchResult]) -> list[TechRecommendation]:
    repo_support = [repo.full_name for repo in repositories[:2]]
    stack: list[TechRecommendation] = []
    summary = keywords.summary.lower()
    food_app = "recipe recommendation" in keywords.capabilities or "ingredient inventory" in keywords.capabilities

    if food_app:
        if "mobile" in summary:
            stack.append(
                TechRecommendation(
                    name="React Native + Expo",
                    category="Frontend",
                    why_recommended="A mobile-first meal prep app benefits from native device access for pantry entry, reminders, and optional camera scanning.",
                    supported_by=repo_support,
                    pros=["Fast mobile iteration", "Good support for camera, notifications, and offline-friendly UX"],
                    cons=["You may still want a lightweight web client later for onboarding or admin tools"],
                )
            )
        else:
            stack.append(
                TechRecommendation(
                    name="React + TypeScript",
                    category="Frontend",
                    why_recommended="A typed web client is a strong starting point for pantry management, recipe browsing, and meal planning workflows.",
                    supported_by=repo_support,
                    pros=["Fast iteration on forms, filters, and planning UI", "Large ecosystem for auth and state management"],
                    cons=["You may still need a mobile client later if camera-first pantry capture becomes important"],
                )
            )

        stack.append(
            TechRecommendation(
                name="FastAPI",
                category="Backend",
                why_recommended="FastAPI is a good fit for inventory CRUD, recipe matching endpoints, scheduled planning jobs, and third-party food APIs.",
                supported_by=repo_support,
                pros=["Clear API modeling", "Works well for search, ranking, and background tasks"],
                cons=["You still need to design the matching and recommendation rules carefully"],
            )
        )
        stack.append(
            TechRecommendation(
                name="PostgreSQL",
                category="Database",
                why_recommended="The product needs structured storage for ingredients, recipes, meal plans, user preferences, and grocery lists.",
                supported_by=repo_support,
                pros=["Strong fit for relational product data", "Can handle filtering and reporting well for v1"],
                cons=["Ingredient and recipe search may need careful indexing as the catalog grows"],
            )
        )
        stack.append(
            TechRecommendation(
                name="PostgreSQL Full-Text Search or Meilisearch",
                category="Matching",
                why_recommended="Ingredient-to-recipe matching works best when you can normalize ingredient names and rank recipes by coverage, not just exact string matches.",
                supported_by=repo_support,
                pros=["Good fit for pantry search and recipe discovery", "Lets you rank by ingredient overlap and missing items"],
                cons=["Requires normalization rules for ingredient names, units, and substitutions"],
            )
        )
        stack.append(
            TechRecommendation(
                name="Recipe Dataset or API",
                category="Data",
                why_recommended="A recipe recommendation app needs a real recipe corpus before personalization or AI becomes useful.",
                supported_by=repo_support,
                pros=["Lets you ship ingredient-based suggestions quickly", "Keeps the first version grounded in structured recipes"],
                cons=["External APIs can be expensive or restrictive, so a curated seed dataset may be better for v1"],
            )
        )
        if "LLM provider" in keywords.likely_integrations or "ai features" in keywords.capabilities:
            stack.append(
                TechRecommendation(
                    name="OpenAI API or similar LLM provider",
                    category="AI",
                    why_recommended="Only add an LLM if you want generated meal plans, substitutions, or natural-language pantry suggestions on top of a real recipe catalog.",
                    supported_by=repo_support,
                    pros=["Good for flexible suggestions and natural-language UX", "Can augment rule-based recommendations"],
                    cons=["Should not replace a structured recipe and ingredient matching system for v1"],
                )
            )
        if "barcode or OCR service" in keywords.likely_integrations or "Ingredient scanning support" in keywords.constraints:
            stack.append(
                TechRecommendation(
                    name="OCR / Barcode Scanning",
                    category="Input",
                    why_recommended="Scanning helps users add pantry items faster if camera-based capture is part of the product scope.",
                    supported_by=repo_support,
                    pros=["Improves pantry input UX", "Reduces manual entry friction"],
                    cons=["Requires image handling and item normalization, which adds complexity early"],
                )
            )
        return stack[:7]

    if "cli" in summary:
        stack.append(
            TechRecommendation(
                name="Python + Typer",
                category="Client",
                why_recommended="A Python CLI stack is fast to ship and easy to extend for developer tooling.",
                supported_by=repo_support,
                pros=["Quick iteration for command workflows", "Rich ecosystem for file and API automation"],
                cons=["Not ideal if you later need a browser UI without building a second client"],
            )
        )
    else:
        frontend_name = "React + TypeScript"
        if any(cap == "canvas rendering" for cap in keywords.capabilities):
            frontend_name = "React + TypeScript + Fabric.js"
        stack.append(
            TechRecommendation(
                name=frontend_name,
                category="Frontend",
                why_recommended="A typed interactive frontend is a strong default for custom product flows and rich UI state.",
                supported_by=repo_support,
                pros=["Fast iteration on product UX", "Large ecosystem for auth, state, and component tooling"],
                cons=["Requires client-side state discipline as the app grows"],
            )
        )

    backend_name = "FastAPI"
    if any(cap in keywords.capabilities for cap in ("realtime collaboration", "chat messaging", "video or voice")):
        backend_name = "Node.js + NestJS or Express"
    stack.append(
        TechRecommendation(
            name=backend_name,
            category="Backend",
            why_recommended="The backend choice should match the workload shape and integration needs, not this analyzer's own stack.",
            supported_by=repo_support,
            pros=["Strong ecosystem for APIs and service composition", "Fits well with common deployment targets"],
            cons=["Requires deliberate service boundaries once background jobs and integrations grow"],
        )
    )

    if any(cap in keywords.capabilities for cap in ("realtime collaboration", "chat messaging", "video or voice")):
        stack.append(
            TechRecommendation(
                name="WebSocket / Socket.IO",
                category="Realtime",
                why_recommended="Live multi-user workflows need a transport optimized for events, presence, and low-latency updates.",
                supported_by=repo_support,
                pros=["Good fit for shared state and live interaction", "Broad ecosystem and deployment support"],
                cons=["Adds connection-state complexity and scaling considerations"],
            )
        )

    stack.append(
        TechRecommendation(
            name="PostgreSQL",
            category="Database",
            why_recommended="A relational database is a safe default for product workflows, permissions, and reporting.",
            supported_by=repo_support,
            pros=["Reliable transactional model", "Works well for most CRUD, workflow, and reporting use cases"],
            cons=["May need extra services for heavy search or high-volume event streaming"],
        )
    )

    if any(cap == "authentication" or cap == "roles and permissions" for cap in keywords.capabilities):
        stack.append(
            TechRecommendation(
                name="Managed Auth Provider or Auth.js",
                category="Auth",
                why_recommended="Account management is usually better delegated to a mature auth solution than built from scratch.",
                supported_by=repo_support,
                pros=["Speeds up login, session, and role management", "Reduces security footguns"],
                cons=["Adds dependency on a third-party service or auth library model"],
            )
        )

    if any(cap == "payments and billing" for cap in keywords.capabilities):
        stack.append(
            TechRecommendation(
                name="Stripe",
                category="Payments",
                why_recommended="Stripe is the fastest path to reliable checkout, subscriptions, or marketplace payouts.",
                supported_by=repo_support,
                pros=["Mature APIs and documentation", "Works well for checkout and recurring billing"],
                cons=["Marketplace payouts and compliance flows still add product complexity"],
            )
        )

    if any(cap == "file uploads" for cap in keywords.capabilities):
        stack.append(
            TechRecommendation(
                name="S3-Compatible Object Storage",
                category="Storage",
                why_recommended="Binary uploads and generated assets should live outside the primary database.",
                supported_by=repo_support,
                pros=["Scales well for media and documents", "Simple integration with signed URLs"],
                cons=["Requires lifecycle management and access control design"],
            )
        )

    return stack[:7]
