---
name: GitHub Repository Scraper
description: Best practices for searching and downloading code from GitHub without hitting token limits.
---

## Rules for Scraping Repositories

When using the GitHub API to gather context for a project:

1. **Search Intelligently:** Use the `q=` parameter in `/search/repositories` to combine frameworks and keywords. Sort by stars or best match.
2. **Fetch ALL Code:** Unlike standard RAG approaches, you MUST crawl and fetch the ENTIRE repository codebase.
3. **Target Every File:** Download absolutely all code files associated with the repository to ensure maximal context for the re-training phase.
4. **Skip Invalid Types:** You can still ignore binary blobs (e.g., images, compiled libraries) and `.git` folders, but ensure 100% of the actual source code and documentation is scraped.
5. **Prepare for Fine-Tuning:** Collect and format these files into a structured dataset format suitable for model fine-tuning/re-training, rather than just dropping them into a context window.
