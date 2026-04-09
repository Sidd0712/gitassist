# AI-Powered GitHub Repository Researcher

This application acts as an intelligent pair-programmer and researcher. Engineers explain their ideas in plain text, and the system translates the idea into technical concepts, searches GitHub for relevant repositories, analyzes the code context, and provides learning paths, architecture diagrams, and approach recommendations.

## Goal Description
Build a full-stack web application (React frontend, FastAPI backend) that:
1. **Idea Translation:** Uses an LLM to convert plain English ideas into a list of technical keywords, jargons, and frameworks.
2. **Smart Repo Searching:** Uses precise technical keywords to search GitHub for relevant repositories and fetches their entire codebase.
3. **Model Re-training:** Compiles all the collected source code from the discovered repositories and dynamically re-trains/fine-tunes the model purely on this aggregated code data.
4. **Analysis & Output Generation:** Uses the selectively re-trained model to output:
   - Detailed repository descriptions and simplified code explanations.
   - Comprehensive learning/build paths.
   - System Architecture diagrams (using Mermaid).
   - Expected Tech Stack and best approaches (with pros/cons).

## Proposed Architecture

1. **Frontend (React)**
   - **Vite:** Build tool for fast loading and hot module replacement.
   - **React (TS/JS):** Declarative UI.
   - **Tailwind CSS + Framer Motion:** For highly aesthetic, glassmorphism UI with micro-animations.
   - **Mermaid.js:** For rendering system architecture diagrams.
   - **Zustand or Context API:** For state management.

2. **Backend (FastAPI)**
   - **FastAPI:** High-performance async Python framework.
   - **Pydantic:** Type validation.
   - **LangChain / LLM Provider:** For chunking text, querying LLMs.
   - **GitHub API Integration:** Search for repos (`/search/repositories`) and fetch code (`/repos/{owner}/{repo}/contents`).

3. **AI Pipeline Flow**
   - User Input -> LLM (Extract Keywords/Frameworks)
   - Keywords -> GitHub Search API -> Top N Repos
   - Top N Repos -> Fetch ALL codebase files (100% of source code)
   - Training Pipeline -> Re-Train/Fine-Tune Model on the aggregate GitHub codebase
   - Re-Trained Model Inference -> Output (Repo Descriptions, Learning Steps, System Architecture Diagram, Tech Stack, Approaches)

## Vibecoding Rules & Configs (Created)

We have created the necessary files to make this a productive vibecoding session:
- `.agent/workflows/backend_dev.md`
- `.agent/workflows/frontend_dev.md`
- `.agent/skills/github_scraper.md`
- `.github/workflows/ci.yml`

---
*Status: Ready to begin implementation. Please review and let me know when you'd like to start coding!*
