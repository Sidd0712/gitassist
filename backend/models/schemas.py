"""Pydantic schemas for request/response validation and RAG internals."""

from typing import Literal

from pydantic import BaseModel, Field


class ClarificationQuestion(BaseModel):
    """A build-critical follow-up question shown before full analysis."""

    key: str
    question: str
    options: list[str] = Field(default_factory=list)
    reason: str = ""


class AmbiguityFlag(BaseModel):
    """An unresolved or resolved ambiguity axis for a project idea."""

    axis: str
    question: str = ""
    reason: str
    options: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "high"
    resolved: bool = False
    answer: str | None = None


class IdeaRequest(BaseModel):
    """User's plain-text project idea and optional clarification answers."""

    idea: str = Field(..., min_length=10, max_length=5000, description="The project idea in plain English")
    clarification_answers: dict[str, str] = Field(default_factory=dict)


class RepoChatScopeRepository(BaseModel):
    """A repository commit that the chat request is allowed to search."""

    full_name: str
    commit_sha: str = Field(..., min_length=1, max_length=200)


class ChatMessage(BaseModel):
    """A single chat turn exchanged in the repo chat panel."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class RepoChatRequest(BaseModel):
    """A stateless repo chat request scoped to indexed repositories."""

    question: str = Field(..., min_length=2, max_length=3000)
    idea_summary: str = Field("", max_length=1000)
    scope_repositories: list[RepoChatScopeRepository] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)


class ExtractedKeywords(BaseModel):
    """Structured technical intent extracted from the user's idea."""

    keywords: list[str] = Field(default_factory=list, description="Technical keywords")
    frameworks: list[str] = Field(default_factory=list, description="Relevant frameworks/libraries")
    languages: list[str] = Field(default_factory=list, description="Programming languages")
    summary: str = Field("", description="One-line technical summary of the idea")
    core_intent: str = ""
    product_type: str = ""
    target_users: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    primary_capabilities: list[str] = Field(default_factory=list)
    secondary_capabilities: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    tech_terms: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    likely_components: list[str] = Field(default_factory=list)
    likely_integrations: list[str] = Field(default_factory=list)
    likely_stack_families: list[str] = Field(default_factory=list)
    ambiguities: list[AmbiguityFlag] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class RepoFile(BaseModel):
    """A single file fetched from a GitHub repository."""

    path: str
    content: str
    size: int = 0


class RepoTreeEntry(BaseModel):
    """A file entry from the Git tree API."""

    path: str
    type: str = "blob"
    size: int = 0


class RepoSearchResult(BaseModel):
    """A GitHub repository discovered via search."""

    full_name: str
    description: str | None = None
    html_url: str
    stars: int = 0
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    default_branch: str = "HEAD"
    updated_at: str | None = None
    archived: bool = False
    commit_sha: str = ""
    relevance_score: float = 0.0
    semantic_meta_score: float = 0.0
    semantic_readme_score: float = 0.0
    query_hit_count: int = 0
    rank_reasons: list[str] = Field(default_factory=list)
    reference_type: Literal["candidate", "end_to_end", "subsystem", "pattern"] = "candidate"
    fit_score: float = 0.0
    fit_summary: str = ""
    covered_primary: list[str] = Field(default_factory=list)
    missing_primary: list[str] = Field(default_factory=list)
    files: list[RepoFile] = Field(default_factory=list, description="Fetched source files")
    evidence_type: Literal["deep_retrieval", "shallow_evidence", "no_evidence"] = "shallow_evidence"
    dependencies: list[str] = Field(default_factory=list, description="Packages/images declared in root manifests")


class RepoSnapshot(RepoSearchResult):
    """A repository with Git tree and search metadata."""

    tree: list[RepoTreeEntry] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class ShallowRepoEvidence(BaseModel):
    """Lightweight repo evidence used for reranking before deep indexing."""

    repository: RepoSearchResult
    readme_path: str | None = None
    readme: str = ""
    manifest_files: list[RepoFile] = Field(default_factory=list)
    sampled_files: list[RepoFile] = Field(default_factory=list)
    sampled_paths: list[str] = Field(default_factory=list)
    highlighted_paths: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    matched_frameworks: list[str] = Field(default_factory=list)
    matched_capabilities: list[str] = Field(default_factory=list)
    score: float = 0.0


class RepoFetchPlan(BaseModel):
    """Selected files for deep fetching and indexing."""

    repository: RepoSearchResult
    selected_paths: list[str] = Field(default_factory=list)
    skipped_paths: list[str] = Field(default_factory=list)
    estimated_chars: int = 0
    rationale: list[str] = Field(default_factory=list)


class RetrievalWeights(BaseModel):
    """Signal weights used when scoring dense retrieval hits."""

    dense_weight: float = 0.8
    repo_weight: float = 0.2


class RetrievalQuery(BaseModel):
    """A query targeted at a specific analysis section."""

    section: Literal["repo_descriptions", "learning_path", "architecture_diagram", "tech_stack", "chat_answer"]
    query: str
    preferred_roles: list[str] = Field(default_factory=list)
    top_k: int = 8
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)


class RetrievalPlan(BaseModel):
    """The retrieval plan for a user idea."""

    queries: list[RetrievalQuery] = Field(default_factory=list)


class CorpusChunk(BaseModel):
    """A persisted chunk in the local RAG corpus."""

    chunk_id: str
    repo_full_name: str
    commit_sha: str
    path: str
    chunk_role: str
    language: str | None = None
    symbol: str | None = None
    heading: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    token_count: int = 0
    repo_score: float = 0.0
    content_hash: str = ""
    text: str


class RetrievalHit(BaseModel):
    """A retrieved chunk with ranking metadata."""

    chunk_id: str
    repo_full_name: str
    path: str
    chunk_role: str
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    score: float = 0.0
    dense_score: float = 0.0
    repo_prior: float = 0.0
    reason: str = ""
    text: str


class Citation(BaseModel):
    """A file-level citation supporting a generated claim."""

    repo_full_name: str
    path: str
    start_line: int | None = None
    end_line: int | None = None
    reason: str = ""
    commit_sha: str | None = None  # lets the UI link to the exact indexed lines on GitHub


class IndexStatusRequest(BaseModel):
    """Repos whose indexing progress the chat panel wants to show."""

    repositories: list[RepoChatScopeRepository] = Field(default_factory=list, max_length=10)


class RepoIndexStatus(BaseModel):
    full_name: str
    state: Literal["not_queued", "queued", "indexing", "partial", "completed", "failed"]
    chunk_count: int = 0
    error: str | None = None


class IndexStatusResponse(BaseModel):
    worker_online: bool
    repositories: list[RepoIndexStatus] = Field(default_factory=list)


class RepoChatEvidenceHit(BaseModel):
    """A compact retrieval hit returned for chat evidence inspection."""

    repo_full_name: str
    path: str
    start_line: int | None = None
    end_line: int | None = None
    reason: str = ""
    snippet: str = ""
    score: float = 0.0


class AnalysisEvidence(BaseModel):
    """Citations grouped by output section."""

    repo_descriptions: list[Citation] = Field(default_factory=list)
    learning_path: list[Citation] = Field(default_factory=list)
    architecture_diagram: list[Citation] = Field(default_factory=list)
    tech_stack: list[Citation] = Field(default_factory=list)


class LearningStep(BaseModel):
    """A single step in the learning/build path."""

    step_number: int
    title: str
    description: str
    milestone: str = ""
    concepts: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)


class TechRecommendation(BaseModel):
    """A technology recommendation with pros/cons."""

    name: str
    category: str
    why_recommended: str = ""
    supported_by: list[str] = Field(default_factory=list)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """The full AI analysis output returned to the frontend."""

    idea_summary: str = ""
    keywords: ExtractedKeywords = Field(default_factory=ExtractedKeywords)
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    repositories: list[RepoSearchResult] = Field(default_factory=list)
    repo_descriptions: list[str] = Field(default_factory=list, description="Simplified code explanations per repo")
    learning_path: list[LearningStep] = Field(default_factory=list)
    architecture_diagram: str = Field("", description="Mermaid.js diagram source")
    tech_stack: list[TechRecommendation] = Field(default_factory=list)
    evidence: AnalysisEvidence | None = None
    status: Literal["needs_clarification", "complete", "error"] = "complete"
    error: str | None = None


class RepoChatResponse(BaseModel):
    """The grounded chat answer returned for the repo chat panel."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence_hits: list[RepoChatEvidenceHit] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)
    scoped_repo_count: int = 0
