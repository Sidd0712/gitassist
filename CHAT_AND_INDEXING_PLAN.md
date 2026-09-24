# Chat-first overhaul: whole-repo indexing, better chat, ranking and descriptions

Status: implemented and verified end to end (see §5) · Owner: Sidd0712 · Written 2026-09-24

Repo chat becomes the main feature. A user should be able to ask
"show me how X is implemented", "how would I adapt this to my idea" or
"where could I improve on this" and get an answer grounded in the repos'
**actual code**, with file and line references that open on GitHub.

Everything below is based on measurements from this machine and the live
database, not guesses. The "Evidence" section lists them.

---

## 1. What's wrong today (evidence)

| # | Problem | Evidence |
|---|---|---|
| E1 | Only a small set of planned files is ever indexed, never the whole repo | `build_repo_fetch_plan` caps at 150 files, ordered by name heuristics |
| E2 | The file filter drops real code | `_should_fetch` matches skip words as **substrings**: `packages` removes every monorepo's `packages/` (most of tldraw), `bin` removes `combine.py`, `dist` removes `distance.py`, `obj` removes `objects/` |
| E3 | Lockfiles are indexed as noise | `package-lock.json` = 71 of Tech-Raddar's 246 chunks |
| E4 | The picker prefers launcher scripts over source code | Time-Series-Library: dozens of `scripts/**/*.sh` indexed, only `models/__init__.py` from `models/` |
| E5 | The JS/TS chunker splits at every indented `const` | `index.js` (17 KB) became 173 chunks, many one line (`const email = ...`, 29 chars) with no context |
| E6 | Long chunks are silently cut off | Windows can reach 2,400+ chars, which is more than bge-small's 512-token limit. The tail is never embedded |
| E7 | Indexing runs on Render (512 MB) | A 604 MB zip OOM-killed the instance (fixed in `b8d8671`, but the design is still fragile) |
| E8 | Filtered vector search can starve | HNSW returns ~40 candidates and then filters to the 5 scoped repos. As the corpus grows, scoped queries return too few hits. pgvector 0.8.0 (installed) supports iterative scans, but we don't use them |
| E9 | Chat sees almost nothing | 6 hits × 600 chars, 900 output tokens, and a retrieval query diluted by the idea summary and chat history |
| E10 | Chat can't match identifiers | Dense-only retrieval. A full-text GIN index already exists (`corpus_chunks_fts_idx`) and is never queried |
| E11 | Chat says "no code available" when there is code | The Tech-Raddar `index.js` had 173 chunks, yet the answer claimed no code |
| E12 | The chat UI renders plain text | No markdown or code blocks, no sources shown, and a 340×460 panel |
| E13 | Groq free tier: **8,000 tokens/min**, 1,000 requests/day, on every model | Read from response headers. Repo ranking alone is ~7k tokens, and the 4 concurrent generation calls cause the 429s seen in the Render logs |
| E14 | Ranking prefers the technique over the domain | "Job skill trends" returned 4 generic time-series libraries. The two job-market repos that were found and indexed were dropped |
| E15 | Descriptions invent components | Tech-Raddar was described as having "message queue, GraphQL, data collectors". It is one Express `index.js` |
| E16 | The report ignores the hardest parts of the idea, and the tech stack is overkill | No data-sourcing or skill-extraction step. Kafka + Kubernetes + TimescaleDB for a student MVP |
| E17 | Dead storage | 6,858 `v1` chunks (~30 MB of the 0.5 GB Neon free tier) are unreachable |

### Measured capacity

- **Embedding speed on this laptop** (i7-1355U, bge-small): 5.1 chunks/s as the worker batches today. **15.3 chunks/s with length-sorted batches of 16**. A batch of 64 at 512 tokens needs an 805 MB attention buffer.
- **Whole-repo sizes** (eligible text files, ~1 chunk per KB):

  | Repo | ≈ chunks | Time at 15/s |
  |---|---|---|
  | Tech-Raddar-Backend | 20 | 1 s |
  | full-stack-fastapi-template | 520 | 35 s |
  | deepvats | 690 | 45 s |
  | Time-Series-Library | 1,150 | 75 s |
  | PaddleTS | 3,000 | 3.3 min |
  | excalidraw | 7,400 | 8 min |
  | tldraw | 22,000 | 25 min |

- **Storage:** ~4.6 KB per chunk including the HNSW and GIN indexes, against Neon's 0.5 GB free tier. That's about 60,000 chunks with headroom.
- **LLM limits:** Groq allows 8K TPM and 1,000 RPD. `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite` each allow 250K TPM, 15 RPM and 500 RPD, on separate quotas.

---

## 2. Target design

```
Render (512 MB)                          Your laptop (worker)
───────────────                          ────────────────────
research request                         repo lane:
  rank → report (fast)                     claim repo_index_jobs row
  enqueue repo_index_jobs  ──Postgres──►   download zip (no 25 MB cap)
                                           pick ALL eligible files, by priority
chat request                               chunk → embed (length-sorted batches)
  embed question ──query lane──►           write chunks every ~256 → 'partial'
  hybrid search (dense+FTS)                done → 'completed'
  expand neighbours + repo map           query lane: embed chat questions (own
  Gemini answer (markdown)                 thread + model, never blocked)
```

### 2.1 Whole-repo indexing moves to the laptop
- **Render stops indexing.** It enqueues `(full_name, commit_sha, repo_json)` into the existing `repo_index_jobs` table. That removes the OOM risk (E7) for good, and the queue is durable: jobs wait while the laptop is off.
- **The worker gets a repo lane.** It claims jobs with `FOR UPDATE SKIP LOCKED`, fetches the tree, selects every eligible file (§2.2), downloads one zip (cap 300 MB, raw-file fallback), chunks (§2.3) and embeds in-process.
- **Progressive availability.** Chunks are written in priority order, in batches of ~256. `repo_indexes.index_state` moves `indexing` → `partial` → `completed`, and chat can search a `partial` repo. A big repo is chat-ready within a minute (README and core source first) while the rest fills in.
- **Caps.**
  - `RAG_REPO_MAX_CHUNKS = 8000` per repo. Priority order means a capped repo keeps its most important files.
  - `RAG_CORPUS_MAX_CHUNKS = 60000` in total, with least-recently-used repos evicted before a new one is written.
- **Cleanup.** Chunks from old `chunking_version`s are purged (E17), and the chunking version goes to `v3`.
- **Retries.** A failed job is retried up to 3 times with its error stored. Jobs left `running` by a dead worker are reclaimed after 30 minutes.
- **Separate CPU.** The query lane keeps its own model instance and threads, so chat questions stay fast while a big repo indexes.

### 2.2 File selection: "whole repo", minus the junk
- **Skip directories by path segment, not substring.** `node_modules`, `vendor`, `dist`, `build`, `.next`, `target`, `coverage`, `__pycache__`, `.venv`, `site-packages`, and so on.
- **Skip files that aren't useful code or docs:**
  - lockfiles (`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `poetry.lock`, `Cargo.lock`, `go.sum`, …)
  - minified and generated files (`*.min.js`, `*.map`, `*_pb2.py`, `*.pb.go`)
  - binaries (null-byte check)
  - files over 150 KB
  - JSON/YAML data files over 20 KB
- **Notebooks become code.** `.ipynb` files are converted to their code and markdown cells, because many ML/data repos (deepvats) live in notebooks.
- **Priority order**, which is the order of indexing and what survives the cap:
  1. root README and manifests
  2. source files in source directories (`src/`, `lib/`, `app/`, `packages/*/src`, the package directory)
  3. entrypoints, routes and services
  4. docs
  5. config
  6. examples
  7. tests
  8. scripts

  Within a tier, files matching the idea's terms come first, then shallower paths.

### 2.3 Chunker v3
- **JS/TS boundaries:** only **top-level** declarations (`function`, `class`, `export …`, top-level `const x = (…) =>`, `interface`, `type`). No more splitting at indented `const` (E5).
- **Merge pass:** neighbouring small spans in a file are merged up to ~1,200 chars. That stops one-line fragments.
- **Hard ceiling:** ~1,600 chars per chunk (≈ within 512 model tokens). Longer spans are windowed with overlap (E6).
- **Contextual header for embedding:** `repo / path / symbol` is prefixed to the text that gets embedded, not to the stored text. "Where is the auth middleware" then matches `src/middleware/auth.ts` even when the code never says "auth middleware".

### 2.4 Chat retrieval
- **Hybrid search.** Dense pgvector search runs with `hnsw.iterative_scan = relaxed_order` (E8). Full-text search runs on the existing GIN index (E10) using identifiers and keywords from the question. The two are fused with Reciprocal Rank Fusion.
- **Question-first queries.** Embed the question on its own, plus the question combined with the previous user turn, so follow-ups like "and how does it store them?" still work. The idea summary is no longer mixed into the embedding (E9).
- **Intent-aware roles.**
  - "Show code / how is X implemented" → source and entrypoints.
  - "Setup / install / run" → docs and config.
  - "Stack / dependencies" → manifests and config.
  - "Compare / which repo" → balanced hits per repo.
- **Repo targeting.** If the question names a repo, search only that repo.
- **Context expansion.** The top hits pull in the neighbouring chunks of the same file, so the model sees whole functions and not fragments.
- **Repo map, always included.** Each repo's top-level structure, key files, declared dependencies and README opening go into every prompt. Overview questions work, and the model can't claim a repo "has no code" (E11).
- **Context budget.** ~60k chars for Gemini. A trimmed ~10k chars when falling back to Groq, to fit its 8K TPM.

### 2.5 Chat generation
- **Provider order:** `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` (separate quota) → Groq with trimmed context.
- **Markdown answers, not JSON.** Code inside JSON strings breaks and truncates. Answers quote real code in fenced blocks labelled `path:Lstart-Lend`, reference files as `owner/repo/path`, and tie the answer back to the user's idea: how to adapt it and what to improve.
- **Output budget:** 2,500 tokens. The last 8 turns are kept.
- **Follow-up suggestions** are parsed from a trailing `FOLLOW_UPS:` block.

### 2.6 Chat UI
- A small markdown renderer (code blocks, inline code, bold, lists, headings, links). It builds React elements, with no `innerHTML`, so there's no XSS risk.
- A **Sources** list under each answer, linking to GitHub pinned to the indexed commit and line range.
- A larger default panel (420×620). Multi-line input: Enter sends, Shift+Enter adds a new line.
- **Index status chips** per repo (ready / indexing N chunks / queued / failed), polled from a new `GET /api/research/index-status`. It also says when the laptop worker is offline.
- Chat unlocks once **any** scoped repo is searchable, not only when all are.

### 2.7 Ranking (E14)
- **Moves to Gemini first.** The ~7k-token prompt no longer competes for Groq's 8K TPM.
- **A richer digest per candidate:** topics, a longer README excerpt and its manifests.
- **Rubric: domain before technique.** The model returns `domain_match` for each candidate.
  - `end_to_end` requires a domain match.
  - A generic library that only matches the technique is capped at `pattern` with fit ≤ 0.6.
- **Enforced in code:** at least min(3, available) domain-matching repos in the top 5, and at most 2 technique-only libraries.

### 2.8 Report quality (E15, E16)
- **Descriptions** are grounded per repo in its README excerpt, manifest/code paths and declared dependencies, plus deep hits when present. They come back as `{full_name: text}` (no ordering drift). The rules: only claim what the evidence shows, name real files, and say plainly when evidence is README-level only.
- **Learning path** must cover data acquisition and the hardest domain-specific component (from `likely_components` / `likely_integrations`), and name the reference files to study in each step.
- **Tech stack** is sized to the stated users and constraints. No Kubernetes, Kafka or microservices unless the idea needs them. Each item cites `supported_by` repos, or says it's a general recommendation. No stale claims.
- **Architecture** uses the idea's own components, with labelled edges.
- **The 4 generation calls go Gemini-first**, which removes the 429 retry stalls.

### 2.9 Out of scope for now
- Streaming chat tokens. Flash-lite answers in a few seconds, so revisit if latency bothers users.
- A code-specific embedding model. It would mean a new vector dimension and a full re-index, so revisit after measuring v3 retrieval quality.
- Auto-starting the worker. A `start_worker.cmd` plus Task Scheduler instructions are included; it isn't wired to boot.

---

## 3. Implementation order

1. **Config + schema:** new settings, the `repo_index_jobs.repo_json` column, and purging old chunking versions.
2. **File selection + chunker v3 + notebooks** (pure functions, unit-tested).
3. **Worker repo lane:** claim, index progressively, retry, evict to budget. Separate query lane. Delete the document lane and the Render-side indexing.
4. **Render:** enqueue instead of fire-and-forget, the `index-status` endpoint, and chat that accepts `partial`.
5. **Hybrid retrieval + context expansion + repo map.**
6. **LLM routing** (Gemini-first for heavy calls, model rotation) + the new chat prompt.
7. **Ranking rubric + composition rules.**
8. **Report prompts** (descriptions, learning path, tech stack, architecture).
9. **Frontend:** markdown, sources, status chips, bigger panel.
10. **Verify end to end** on the job-skills idea and a whiteboard idea. Re-rate against the scorecard.

## 4. How we'll know it worked

Re-run the "job skills trends" idea and ask the same 3 chat questions.

- At least 2 of the 5 repos are job-market or skill-analysis projects.
- No description names a component that isn't in its evidence.
- "Show the code" answers contain real fenced code with correct `path:L` labels that open on GitHub.
- No answer claims a repo has no code when it has indexed chunks.
- Tech-Raddar `index.js` ≤ 25 chunks (was 173). No lockfile chunks anywhere.
- A 3,000-chunk repo is chat-ready (`partial`) within ~60 s and complete within ~4 min.
- No Groq 429 retry stalls in a report run.

## 5. Results (measured 2026-09-24, local backend + worker)

| Check | Before | After |
|---|---|---|
| Job-skills idea: domain-matching repos in top 5 | 0 (4 generic time-series libraries) | 4 (job-market forecasting, quant-jobs research, skills hub, placement prediction) + 1 labelled `pattern` |
| Descriptions naming components not in evidence | Several (e.g. "GraphQL, message queue" for a single `index.js`) | None found; each names real files |
| Tech-Raddar `index.js` chunks | 173 | 15 |
| Chat "show the code" answers | "No code snippets available" | Real fenced code with `path:L` labels, e.g. `db_service.py:L34-L68`, a notebook's skill-extraction regex |
| Chat answer time | ~4 s (tiny context) | 6–8 s (up to 60k chars of stitched code, Gemini) |
| Candidate search (document-lane embeddings) | 31 s | 12 s |
| Report end to end | 88 s | 75 s |
| Constella (1,142 chunks) searchable | — | 48 s after queueing (`partial`), complete in 225 s |
| Groq 429 stalls in a report run | Yes (Render log) | None; heavy calls go Gemini-first |

Throughput note: this laptop embeds real v3 chunks (avg ~1,200 chars with header) at ~7.5 chunks/s; with Neon writes overlapped the repo lane sustains ~5 chunks/s. A 3,000-chunk repo takes ~10 min to complete but is chat-ready (first 256 chunks, highest priority files) within a minute.
