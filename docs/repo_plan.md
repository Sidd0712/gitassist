# GitHub Repository Ranking + Retrieval Redesign

## Summary

Redesign the GitHub discovery and RAG preparation pipeline so repository selection is driven by idea understanding, semantic relevance, feature coverage, and diversity rather than stars, framework overlap, or raw keyword matches. The new pipeline uses structured intent extraction, staged GitHub ingestion, embedding-based reranking, deduplication, language diversity caps, and coverage-aware final selection. Expensive README and code retrieval happens only after aggressive filtering to reduce latency and API cost.

## Key Changes

### 1. Structured intent drives everything

- Extend idea understanding beyond flat keywords into `core_intent`, `primary_capabilities`, `secondary_capabilities`, `trivial_capabilities`, `capability_weights`, `domain_terms`, and `tech_terms`.
- Treat generic capabilities such as auth, uploads, persistence, and notifications as low-priority by default unless the user clearly makes them central.
- Add optional clarification for feature priority when an idea contains multiple equally important non-generic features and there is no clear dominant focus.
- Keep current intent fields for compatibility, but make all ranking logic consume the new weighted intent structure.

### 2. Replace keyword-and-stars candidate discovery

- Generate multiple GitHub query families from domain terms and weighted capabilities instead of frameworks.
- Build separate query groups for core intent, primary capabilities, end-to-end app matches, and subsystem matches.
- Remove star ordering as the dominant search strategy in ranking; stars become only a small secondary quality signal.
- Apply fast metadata filtering before any README or code fetch to demote infrastructure repos, starter kits, challenge repos, list repos, and generic framework-heavy candidates.

### 3. Introduce staged semantic reranking

- Stage 1: score pooled GitHub candidates using metadata only: repo name, description, topics, query-hit count, and lightweight penalties.
- Stage 2: fetch README plus a small set of root manifests for a limited shortlist and compute embedding similarity between the resolved idea and repo summaries.
- Stage 3: rerank using a composite score dominated by semantic relevance and weighted capability coverage, with small contributions from docs quality, query diversity, and stars.
- Make framework and stack overlap a tie-breaker only, not a primary ranking feature.
- Add explicit penalties when a repo emphasizes implementation styles that are outside the resolved scope, such as AI/RAG-heavy repos for non-AI ideas.

### 4. Enforce coverage, deduplication, and stack diversity

- Add repo-level similarity checks using README embeddings to remove near-duplicate repositories before final selection.
- Group candidates by language or stack family and cap representation so one ecosystem cannot dominate the result set.
- Select final repositories with a greedy coverage-aware algorithm that ensures the final set collectively covers all primary capabilities when possible.
- Prefer at least one end-to-end reference if available, then fill missing capabilities with subsystem or pattern references.
- Add transparent repo explanations that state which primary and secondary capabilities each repository covers, what it misses, and why it was still selected.

### 5. Reduce latency with hierarchical retrieval

- Return up to 10 ranked repositories, but only deep-index the top 4 for code-level RAG.
- Treat the remaining returned repositories as shallow references backed by README and root manifests only.
- Keep full code fetches limited to a smaller staged subset, with tighter caps on files and characters.
- Update deep-fetch planning so README, manifests, and architecture-defining files are fetched before broad source expansion.
- Preserve the current hybrid retrieval model for indexed repos, but seed it with better repo priors and multi-capability retrieval queries.

### 6. Render-aware persistence and configuration

- Assume deployment on Render with remote embeddings and persistent disk as the preferred production setup.
- Add explicit config for candidate pool size, README rerank limit, output repo count, deep-index repo count, dedupe threshold, and language diversity caps.
- Use persistent storage paths for corpus, SQLite, and vector store when Render disk is available; allow degraded stateless behavior if persistent disk is absent.
- Keep remote embedding support as the primary semantic path and retain the existing local fallback only as a degraded mode.

## Public Interfaces / Types

- Extend `ExtractedKeywords` with weighted intent fields: `core_intent`, `primary_capabilities`, `secondary_capabilities`, `trivial_capabilities`, `capability_weights`, `domain_terms`, and `tech_terms`.
- Extend `RepoSearchResult` with optional transparency fields such as `semantic_meta_score`, `semantic_readme_score`, `query_hit_count`, `covered_primary`, and `missing_primary`.
- Keep `/api/research` shape stable and continue returning `AnalysisResponse`, but populate the new optional fields so the frontend can surface `why selected` explanations later without a contract break.
- Keep the current clarification flow, but allow it to ask priority questions when multi-feature intent is ambiguous.

## Test Plan

- Intent decomposition tests:
  - multi-feature ideas classify core vs trivial capabilities correctly
  - generic capabilities stay low-priority unless explicitly central
  - priority clarification appears only when feature importance is ambiguous
- Query generation tests:
  - domain and primary-capability terms dominate generated GitHub queries
  - frameworks do not dominate candidate search unless explicitly requested
- Ranking tests:
  - semantic relevance outranks star count
  - relevant lower-star repos beat popular but generic framework repos
  - non-AI ideas demote AI-heavy repos even when names overlap
- Diversity tests:
  - duplicate or near-duplicate README repos collapse to one
  - final selection respects per-language caps when the pool supports it
  - final repo set covers all primary capabilities when suitable candidates exist
- Latency and staging tests:
  - README fetch happens only for the rerank shortlist
  - deep code indexing happens only for the top 4 selected repos
- End-to-end tests:
  - multi-aspect idea like `travel planner with chat, recommendations, and maps` returns a diversified repo set with coverage across all major features
  - returned repo explanations identify matched capabilities and uncovered areas

## Assumptions

- The system should return up to 10 repositories total and deep-index only 4.
- Remote embeddings are the primary production strategy.
- Render hosting should be planned around persistent disk when available.
- Backward compatibility with the current `/api/research` route and main frontend flow is important.
- The plan should be implemented incrementally inside the existing backend services rather than replacing the architecture wholesale in one step.
