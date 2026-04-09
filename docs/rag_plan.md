# Smart RAG Replacement Plan for GitHub and LLM Services

## Purpose

This document replaces the current "dump and prompt" approach in `backend/services/github_service.py` and `backend/services/llm_service.py` with a project-specific Retrieval-Augmented Generation (RAG) design.

It is grounded in the current product goals from `docs/implementation_plan.md`:

- translate a plain-English idea into technical search intent
- find relevant GitHub repositories
- understand those repositories accurately
- generate repo descriptions, learning paths, architecture diagrams, and tech-stack recommendations

## Recommendation Summary

For this project, the best replacement is not request-time fine-tuning or "train on fetched repos, then answer." The more accurate and practical architecture is:

1. Use the LLM to extract keywords and retrieval intent.
2. Search GitHub broadly for candidate repositories.
3. Do a shallow ingest first (README, manifests, tree, metadata).
4. Re-rank repositories before deep fetching.
5. Build a persistent local corpus keyed by repo commit SHA.
6. Chunk code and docs with metadata.
7. Run hybrid retrieval:
   - dense similarity for semantic matching
   - lexical search for exact names, files, APIs, and symbols
   - metadata priors for file type, language, repo score, and chunk role
8. Generate each analysis section from retrieved evidence instead of dumping entire codebases into one prompt.

This keeps the current FastAPI app simple, improves groundedness, reduces token waste, and gives us a much better path to accurate answers.

## Why the Current Flow Is Inaccurate

The current implementation has four major problems:

### 1. Repository selection is shallow

`github_service.search_repos()` builds one combined search string from a few extracted terms, sorts by stars, and returns the top N repositories. That favors popularity over true relevance.

### 2. Repository ingestion is noisy

`github_service.fetch_repo_code()` fetches many files from each repository up to a character budget, but it does not understand:

- which files are central vs peripheral
- which files are architecture-defining vs implementation detail
- which files match the user idea
- which files are redundant or low-signal

### 3. Prompt building is lossy

`llm_service._build_repo_context()` concatenates all fetched files into one context string with a hard character cap. Once the budget is hit:

- later files are dropped
- important files may never be seen
- unrelated files consume context
- the LLM has no ranking signal about which chunks matter most

### 4. The current "training" step does not improve runtime accuracy

`training_service.compile_training_dataset()` creates JSONL entries, but the request still answers from a single prompt dump. That means we pay ingestion cost without gaining a true retrieval or fine-tuning benefit inside the active request path.

## Constraints and Assumptions

This plan is optimized for the current repository and architecture:

- backend is a single FastAPI app with no database yet
- current output contract is `AnalysisResponse`
- LangChain and `langchain-openai` are already installed
- the app currently works synchronously behind one `/api/research` route
- local disk persistence is acceptable
- minimizing frontend breakage is important

Because of those constraints, the first version should be local-first and operationally light:

- dense store: Chroma with on-disk persistence
- lexical store and metadata: SQLite + FTS5
- embeddings: OpenAI-compatible embedding model
- orchestration: custom service code, not a fully abstracted agent framework

If the app later becomes multi-instance or hosted at scale, the dense store can be swapped for Qdrant without changing the higher-level retrieval interfaces.

## Target Architecture

### High-Level Flow

1. User submits an idea.
2. LLM extracts keywords, frameworks, languages, and analysis intents.
3. GitHub search returns a wider candidate set.
4. Shallow ingest gathers repo metadata, tree, README, manifests, and key config files.
5. Repo ranker selects the top repositories for deep indexing.
6. Deep ingest fetches only relevant files.
7. Chunking produces code/doc/config chunks with rich metadata.
8. Embeddings and lexical index are persisted by repo SHA.
9. Retrieval runs per analysis section.
10. LLM generates grounded JSON from retrieved evidence.
11. Response returns the same high-level analysis fields, plus optional sources/citations.

### Core Design Principles

- Shallow-to-deep ingestion: do not fully fetch every candidate repo up front.
- Persistent corpora: reuse indexed repos across requests if the commit SHA is unchanged.
- Hybrid retrieval: combine semantic, lexical, and metadata signals.
- Section-specific retrieval: architecture, learning path, and tech-stack recommendations should not all use the same retrieved context.
- Grounded synthesis: every major claim should come from retrieved evidence.
- Stable API first: keep the current `/api/research` contract working while improving internals.

## Proposed Backend Structure

Recommended new layout:

```text
backend/
  services/
    github_service.py
    llm_service.py
    training_service.py
    rag/
      __init__.py
      corpus_service.py
      chunking_service.py
      embedding_service.py
      retrieval_service.py
      ranking_service.py
      citation_service.py
      query_planning_service.py
      store_service.py
  data/
    corpora/
    chroma/
    retrieval.db
```

### Responsibility Split

`github_service.py`
- GitHub-only responsibilities
- candidate search
- repo metadata fetch
- tree fetch
- shallow file fetch
- deep file fetch for approved repos
- caching and GitHub rate-limit handling

`llm_service.py`
- prompting only
- keyword extraction
- retrieval query planning
- grounded section synthesis
- final JSON normalization and repair

`services/rag/*`
- corpus creation
- chunking
- embeddings
- lexical index
- dense index
- chunk ranking
- citations

`training_service.py`
- no longer part of the request-critical path
- repurpose for offline evaluation dataset generation or remove later

## Detailed Implementation Plan

## Phase 0: Baseline Instrumentation

Before replacing logic, add visibility so we can measure improvement.

### Goals

- capture current latency and payload sizes
- capture how many files/chars are fetched today
- log which repos are chosen and why
- record prompt size before the model call

### Changes

`backend/routers/research.py`
- log end-to-end timings for keyword extraction, search, fetch, and generation

`backend/services/github_service.py`
- log candidate counts, fetch counts, and bytes/characters per repo

`backend/services/llm_service.py`
- log prompt character count and repo count currently sent to the model

### Acceptance Criteria

- we can compare old vs new latency
- we can compare total fetched chars vs final retrieved chars
- we can compare token budget waste before and after RAG

## Phase 1: Replace Full-Repo Dumping With Staged GitHub Ingestion

This is the first major accuracy improvement.

### What Changes

Replace the current "search top repos, then fetch full codebases" behavior with:

1. Search a wider candidate pool.
2. Collect only lightweight repo evidence first.
3. Re-rank candidates.
4. Deep-fetch only the top few repositories.

### GitHub Search Improvements

Current behavior:
- one search query
- stars-first ranking
- small candidate set

Proposed behavior:
- build 3 to 5 query variants from extracted keywords/frameworks/languages
- merge and deduplicate results
- fetch 15 to 25 candidates before reranking

Suggested query sources:
- keyword-heavy query
- framework-heavy query
- language-constrained query
- architecture keyword query from the idea summary
- optional domain-specific query if the idea implies a vertical (agents, ecommerce, chatbot, RAG, etc.)

### Shallow Ingest Payload

For each candidate repo, fetch:

- repo metadata: stars, language, topics, default branch, description, updated_at, archived, size
- recursive tree
- README
- key manifests and configs:
  - `package.json`
  - `pyproject.toml`
  - `requirements.txt`
  - `go.mod`
  - `Cargo.toml`
  - `Dockerfile`
  - `docker-compose.yml`
  - `README*`
  - framework config files

Do not fetch the full source code yet.

### Repo Re-Ranking

Score candidate repos using a weighted mix of:

- GitHub search score and star quality
- keyword overlap with repo name, description, topics, README
- framework/language match
- presence of architectural files relevant to the idea
- recency and maintenance
- penalty for archived or obviously template/demo-only repos

Recommended result:
- deep-index top 3 to 5 repos
- keep remaining candidates as fallback

### Changes to `backend/services/github_service.py`

Refactor into these logical functions:

- `search_repo_candidates(keywords: ExtractedKeywords) -> list[RepoCandidate]`
- `fetch_repo_snapshot(full_name: str) -> RepoSnapshot`
- `fetch_shallow_repo_evidence(snapshot: RepoSnapshot) -> ShallowRepoEvidence`
- `build_repo_fetch_plan(snapshot: RepoSnapshot, shallow: ShallowRepoEvidence, idea: str) -> RepoFetchPlan`
- `fetch_repo_files_for_indexing(plan: RepoFetchPlan) -> list[RawRepoFile]`

The file should stop returning giant `repo.files` lists for direct prompt injection.

## Phase 2: Build a Persistent Corpus and Hybrid Index

This phase turns fetched repository data into a reusable knowledge base.

### Corpus Keying Strategy

Each indexed repo should be keyed by:

- `repo_full_name`
- `default_branch`
- `commit_sha`
- `embedding_model`
- `chunking_version`

This lets us:

- reuse existing indexes if the repo has not changed
- invalidate only when the commit SHA or chunking schema changes
- avoid repeated embedding costs

### Data to Persist

Store three layers:

1. Raw repo snapshot
- repo metadata
- commit SHA
- fetched files

2. Chunk manifest
- chunk id
- repo name
- path
- language
- chunk type
- start/end line
- token count
- content hash
- symbols/imports if available

3. Retrieval indexes
- dense vectors in Chroma
- lexical/searchable metadata in SQLite FTS5

### Recommended Storage Layout

`backend/data/corpora/<repo_full_name>/<commit_sha>/`
- `repo.json`
- `files.jsonl`
- `chunks.jsonl`

`backend/data/chroma/`
- Chroma persistent vector store

`backend/data/retrieval.db`
- SQLite metadata tables
- FTS5 table for lexical search

### Why This Fits the Current Project

- no external infrastructure required
- works well for local development
- easy to inspect and debug
- accurate enough for code and documentation retrieval
- can be swapped later without changing the API contract

## Phase 3: Replace Naive Concatenation With Query-Aware Chunking

The current system works at file level. For RAG, that is too coarse.

### Chunk Types

Treat different file categories differently:

1. Documentation chunks
- README
- docs pages
- examples/tutorials

2. Config/manifests
- package manifests
- environment files
- Docker/Kubernetes configs
- build config

3. Source code chunks
- module/class/function-level chunks where possible
- fallback to token-based windows with overlap

4. Test/example chunks
- useful for "how does this work?" and learning path generation

### Recommended Chunking Rules

Documentation:
- split by headings and subheadings
- preserve heading path in metadata

Config/manifests:
- keep entire file as one chunk unless very large
- mark as `chunk_role=config`

Source code:
- prefer symbol-aware chunks
- include a small amount of surrounding context
- retain path, symbol name, start/end lines, imports, and repo metadata

Fallback source chunk sizing:
- target 300 to 500 tokens
- 50 to 80 token overlap

### Accuracy Notes

For this codebase, symbol-aware chunking is worth the effort because the output asks for:

- architecture explanations
- learning paths
- stack recommendations
- simplified repo descriptions

Those outputs depend on understanding module boundaries, not just raw text similarity.

### Implementation Recommendation

Start with custom chunking logic plus lightweight language heuristics:

- Python: `ast`
- JS/TS: regex plus export/function/class heuristics initially
- docs/config: rule-based splitting

Then add tree-sitter in a later pass if needed.

This avoids over-engineering the first implementation while still producing much better chunks than file-level dumps.

## Phase 4: Add Dense + Lexical + Metadata Retrieval

This is the core of the smart RAG system.

### Retrieval Inputs

A single user request should produce multiple retrieval queries, not just one:

- repo relevance query
- architecture query
- implementation patterns query
- learning path query
- tech stack query

These can be generated by a lightweight query planner in `llm_service.py`.

### Retrieval Strategy

Use hybrid retrieval with three layers:

1. Dense retrieval
- semantic similarity over chunk embeddings

2. Lexical retrieval
- exact token matches over file path, symbol names, package names, framework names, and text

3. Metadata priors
- boost docs for learning-path queries
- boost manifests/configs for tech-stack queries
- boost entrypoints/routers/services for architecture queries
- boost repo score from the shallow reranker

### Recommended Scoring Formula

Start with a simple weighted score:

`final_score = 0.45 * dense + 0.25 * lexical + 0.20 * repo_prior + 0.10 * chunk_role_prior`

Then apply:

- Maximal Marginal Relevance (MMR) for diversity
- per-repo caps so one repo does not dominate everything
- section-specific top-k selection

### Retrieval Output Shape

Each retrieved hit should carry:

- `chunk_id`
- `repo_full_name`
- `path`
- `chunk_role`
- `language`
- `start_line`
- `end_line`
- `score`
- `text`
- `why_selected`

This will make debugging and prompt grounding much easier.

## Phase 5: Rebuild `llm_service.py` Around Grounded Generation

`llm_service.py` should stop receiving one giant repo dump and instead become a grounded synthesis layer.

### Keep

- keyword extraction
- JSON response normalization

### Remove

- `_build_repo_context()`
- any direct concatenation of full repo files into one global prompt

### Add

- `plan_retrieval_queries(idea, keywords) -> RetrievalPlan`
- `generate_repo_descriptions(...)`
- `generate_learning_path(...)`
- `generate_architecture_diagram(...)`
- `generate_tech_stack(...)`
- `merge_grounded_sections(...) -> AnalysisResponse`

### Section-Specific Generation

This is important for accuracy because each output requires different evidence.

#### Repo Descriptions

Use:
- repo README/docs chunks
- top config/manifests
- top architecture/source chunks

Prompt rule:
- describe only what is supported by retrieved evidence

#### Learning Path

Use:
- README/tutorial/example chunks
- setup instructions
- tests/examples that demonstrate usage

Prompt rule:
- produce a build path that matches the user's idea and the retrieved repos

#### Architecture Diagram

Use:
- entrypoints
- routers/controllers
- service layer files
- config and infrastructure chunks

Prompt rule:
- synthesize only the main components and flows supported by retrieved evidence

#### Tech Stack

Use:
- manifest and lock/config files first
- README and deployment docs second

Prompt rule:
- distinguish between explicitly observed tech and inferred recommendations

### Grounding Rules For Prompts

Every synthesis prompt should include rules like:

- use only retrieved evidence
- separate explicit facts from inference
- do not invent dependencies or architecture components
- if evidence is weak, say so
- prefer repo/path citations in internal reasoning and output metadata

## Phase 6: Add Citations and Evidence Tracking

The current response has no way to show what evidence supports the answer.

### Schema Changes

Add new optional models in `backend/models/schemas.py`:

- `Citation`
- `RetrievalHit`
- `AnalysisEvidence`

Suggested fields:

```python
class Citation(BaseModel):
    repo_full_name: str
    path: str
    start_line: int | None = None
    end_line: int | None = None
    reason: str = ""
```

```python
class AnalysisEvidence(BaseModel):
    repo_descriptions: list[Citation] = Field(default_factory=list)
    learning_path: list[Citation] = Field(default_factory=list)
    architecture_diagram: list[Citation] = Field(default_factory=list)
    tech_stack: list[Citation] = Field(default_factory=list)
```

Then extend `AnalysisResponse` with an optional field:

- `evidence: AnalysisEvidence | None = None`

### Why This Matters

- better frontend transparency later
- easier debugging and eval
- reduced hallucination risk
- lets us compare generated claims against actual retrieved files

The frontend does not need to consume this immediately. It can be added as a backward-compatible field.

## Phase 7: Update the Research Router Without Breaking the API

`backend/routers/research.py` should keep the same endpoint and return shape as much as possible.

### New Request Flow

1. extract keywords
2. search candidate repos
3. shallow ingest and rerank repos
4. ensure deep index exists for top repos
5. plan retrieval queries
6. retrieve evidence by section
7. generate grounded analysis
8. return `AnalysisResponse`

### Important Routing Decision

For the first RAG version, keep the endpoint synchronous if:

- candidate set is limited
- deep indexing is cached
- only top few repos are indexed

If latency is too high, the next step is not to undo RAG. It is to:

- move deep indexing to a background task
- return partial progress or cached results
- optionally add a job-based API later

## Phase 8: Reframe `training_service.py`

Based on `docs/implementation_plan.md`, the original concept assumed dynamic retraining/fine-tuning on fetched code. For this product and codebase, that should not stay in the synchronous request path.

### Recommended Change

Short term:
- remove `compile_training_dataset()` from the critical `/api/research` flow

Medium term:
- repurpose it to generate offline evaluation fixtures
- or use it to prepare curated fine-tuning data only if we later discover a stable repeated task worth training for

### Reason

For this use case, runtime accuracy depends much more on:

- finding the right repos
- retrieving the right chunks
- grounding the generation

than on creating request-time JSONL files.

## Concrete File-Level Plan

### `backend/services/github_service.py`

Refactor from "search + full code dump" into "candidate search + staged ingestion":

- improve query generation
- fetch repo metadata and tree first
- fetch README/manifests before full source
- build deep-fetch plans using repo score and file priority
- cache snapshots by repo SHA

### `backend/services/llm_service.py`

Refactor from "one prompt with all files" into "query planner + grounded synthesizer":

- keep keyword extraction
- add retrieval query planning
- generate section-specific outputs from retrieved evidence
- remove `_build_repo_context()`

### `backend/routers/research.py`

Swap orchestration order:

- from:
  - search
  - fetch everything
  - compile dataset
  - prompt all code

- to:
  - search candidates
  - shallow ingest
  - rerank
  - ensure index
  - retrieve
  - grounded generation

### `backend/models/schemas.py`

Add:

- repo snapshot and corpus models
- retrieval hit models
- citation/evidence models

Keep:

- `AnalysisResponse` and current consumer-facing fields

### `backend/core/config.py`

Add RAG-specific config:

- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS` if needed
- `RAG_VECTOR_STORE_PATH`
- `RAG_SQLITE_PATH`
- `RAG_CHUNK_TOKENS`
- `RAG_CHUNK_OVERLAP`
- `RAG_CANDIDATE_REPO_LIMIT`
- `RAG_DEEP_INDEX_REPO_LIMIT`
- `RAG_SECTION_TOP_K`
- `RAG_MAX_CONTEXT_TOKENS`
- `RAG_INDEX_TTL_HOURS` or SHA-only invalidation policy

### New Files

Recommended:

- `backend/services/rag/corpus_service.py`
- `backend/services/rag/chunking_service.py`
- `backend/services/rag/embedding_service.py`
- `backend/services/rag/retrieval_service.py`
- `backend/services/rag/ranking_service.py`
- `backend/services/rag/store_service.py`
- `backend/services/rag/citation_service.py`
- `backend/services/rag/query_planning_service.py`

## Suggested Retrieval Policy By Output Type

This project generates four very different outputs, so retrieval should be tailored.

| Output | Primary evidence | Secondary evidence | Notes |
| --- | --- | --- | --- |
| Repo descriptions | README, manifests, top architecture chunks | examples/tests | keep per-repo filtering strong |
| Learning path | README/tutorial/docs/examples | tests and setup files | optimize for educational clarity |
| Architecture diagram | entrypoints, routers, services, configs | README architecture sections | prioritize component boundaries |
| Tech stack | manifests, lock/config files | README/deploy docs | distinguish observed vs recommended |

This is a major accuracy win compared with one shared prompt context.

## Evaluation Plan

We should not ship this blindly. Add a lightweight eval loop.

### Offline Eval Set

Create 15 to 25 test ideas that reflect the app's actual use:

- API backend
- AI agent app
- RAG assistant
- workflow automation tool
- developer productivity app
- chat application

For each test idea, save:

- expected useful repos
- expected tech-stack signals
- expected architecture themes

### Metrics

Track:

- repo precision@k
- chunk precision@k
- citation validity
- groundedness / unsupported claim rate
- average tokens sent to LLM
- average latency
- GitHub API calls per request

### Human Review Checklist

- are the chosen repos actually relevant?
- are repo descriptions supported by retrieved evidence?
- does the learning path feel actionable?
- does the architecture diagram match the repos and the user idea?
- are tech recommendations grounded rather than generic?

## Performance and Cost Controls

RAG improves accuracy, but only if we avoid replacing one expensive pipeline with another.

### Controls To Add

- persistent repo index by commit SHA
- cap deep indexing to top 3 to 5 repos
- cap chunk counts per repo by file priority
- use section-specific context budgets
- cache embeddings and retrieval results for repeated ideas
- store repo-level summary chunks for reuse

### File Priority Heuristics For Deep Fetch

Fetch first:

- README/docs
- manifests/configs
- entrypoints
- routers/controllers
- services/core domain modules
- examples/tests

De-prioritize or skip:

- generated files
- vendor/build artifacts
- lockfiles unless specifically useful
- very large minified assets
- binary files

## Risks and Mitigations

### GitHub Rate Limits

Risk:
- shallow + deep ingest can increase request count

Mitigation:
- cache by repo SHA
- use tree API aggressively
- batch and concurrency-limit file fetches
- deep fetch only after reranking

### Latency On Cold Start

Risk:
- first request for a new repo set may be slow

Mitigation:
- keep first version synchronous but heavily cached
- consider background indexing next
- return cached results when available

### Hallucinated Architecture Summaries

Risk:
- LLM may still overgeneralize from partial context

Mitigation:
- section-specific retrieval
- prompt grounding rules
- citations
- explicit/inferred distinction

### Chunk Quality

Risk:
- weak chunking reduces retrieval quality

Mitigation:
- start with chunk metadata and symbol heuristics
- add tree-sitter only after measuring actual gaps

## Rollout Order

Recommended implementation order:

1. Add instrumentation and config.
2. Refactor GitHub search into candidate search + shallow ingest.
3. Add repo reranking.
4. Add corpus persistence and chunk manifests.
5. Add dense and lexical indexes.
6. Add retrieval query planning.
7. Replace `_build_repo_context()` with section-specific retrieval.
8. Add evidence/citation models.
9. Remove `training_service.py` from the request path.
10. Run offline eval and tune weights.

## Definition of Done

This migration is complete when:

- no request path depends on dumping full repo contents into a single prompt
- `github_service.py` performs staged ingestion instead of full eager fetch
- `llm_service.py` generates analysis from retrieved evidence, not concatenated raw code
- repo selection is reranked before deep indexing
- corpus reuse works across repeated requests using repo SHA
- outputs remain compatible with the current frontend response shape
- evidence is available for debugging and future UI surfacing
- offline eval shows better repo relevance and fewer unsupported claims than the current baseline

## Final Recommendation

The best path for this project is a local-first hybrid RAG system with:

- staged GitHub ingestion
- persistent repo corpora keyed by commit SHA
- symbol-aware chunking
- Chroma for dense retrieval
- SQLite FTS5 for lexical retrieval and metadata
- section-specific retrieval and grounded generation

That will be materially more accurate than the current "fetch everything and hope the prompt fits" design, while still fitting the current FastAPI codebase and keeping the frontend contract mostly stable.
