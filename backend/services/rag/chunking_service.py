"""Chunk repository files into retrieval-friendly units."""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass

from core.config import get_settings
from models.schemas import CorpusChunk, RepoFile, RepoSearchResult

logger = logging.getLogger(__name__)

CONFIG_FILENAMES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "pyproject.toml",
    "requirements.txt",
    "poetry.lock",
    "go.mod",
    "cargo.toml",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".env.example",
}

ENTRYPOINT_NAMES = {
    "main.py",
    "app.py",
    "server.py",
    "main.ts",
    "main.js",
    "index.ts",
    "index.js",
    "api.py",
}


@dataclass
class Span:
    """Simple 1-based line span."""

    start_line: int
    end_line: int
    symbol: str | None = None
    heading: str | None = None


class ChunkingService:
    """Create role-aware chunks for repository files."""

    def __init__(self) -> None:
        settings = get_settings()
        self.chunk_tokens = settings.RAG_CHUNK_TOKENS
        self.chunk_overlap = settings.RAG_CHUNK_OVERLAP
        self.chunking_version = settings.RAG_CHUNKING_VERSION

    def chunk_repository(self, repository: RepoSearchResult, files: list[RepoFile]) -> list[CorpusChunk]:
        """Chunk every fetched repo file."""

        chunks: list[CorpusChunk] = []
        for repo_file in files:
            file_chunks = self.chunk_file(repository, repo_file)
            chunks.extend(file_chunks)
        return chunks

    def chunk_file(self, repository: RepoSearchResult, repo_file: RepoFile) -> list[CorpusChunk]:
        """Chunk a single file based on its type."""

        text = repo_file.content.replace("\r\n", "\n").strip()
        if not text:
            return []

        role = self._classify_path(repo_file.path)
        spans: list[Span]
        if role == "documentation":
            spans = self._split_documentation(text)
        elif role == "config":
            spans = self._split_config(text)
        elif repo_file.path.lower().endswith(".py"):
            spans = self._split_python(text)
        elif repo_file.path.lower().endswith((".js", ".jsx", ".ts", ".tsx")):
            spans = self._split_javascript(text)
        else:
            spans = self._split_fallback(text)

        lines = text.splitlines()
        chunks: list[CorpusChunk] = []
        for span in spans:
            chunk_text = "\n".join(lines[span.start_line - 1 : span.end_line]).strip()
            if not chunk_text:
                continue
            content_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha1(
                f"{repository.full_name}:{repository.commit_sha}:{repo_file.path}:{span.start_line}:{span.end_line}:{content_hash}".encode(
                    "utf-8"
                )
            ).hexdigest()
            chunks.append(
                CorpusChunk(
                    chunk_id=chunk_id,
                    repo_full_name=repository.full_name,
                    commit_sha=repository.commit_sha,
                    path=repo_file.path,
                    chunk_role=role,
                    language=self._infer_language(repo_file.path, repository.language),
                    symbol=span.symbol,
                    heading=span.heading,
                    start_line=span.start_line,
                    end_line=span.end_line,
                    token_count=self._estimate_tokens(chunk_text),
                    repo_score=repository.relevance_score,
                    content_hash=content_hash,
                    text=chunk_text,
                )
            )
        return chunks

    def _classify_path(self, path: str) -> str:
        lowered = path.lower()
        filename = lowered.rsplit("/", 1)[-1]

        if filename.startswith("readme") or "/docs/" in lowered or lowered.endswith((".md", ".rst", ".txt")):
            return "documentation"
        if any(part in lowered for part in ("/test", "/tests", "__tests__", "/spec", "/specs")):
            return "test"
        if any(part in lowered for part in ("/example", "/examples", "/demo", "/samples")):
            return "example"
        if filename in CONFIG_FILENAMES or lowered.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg")):
            return "config"
        if filename in ENTRYPOINT_NAMES or any(part in lowered for part in ("/router", "/routes", "/api", "/main", "/app")):
            return "entrypoint"
        return "source"

    def _split_documentation(self, text: str) -> list[Span]:
        lines = text.splitlines()
        headings: list[tuple[int, str]] = []
        for index, line in enumerate(lines, start=1):
            if re.match(r"^\s{0,3}#{1,6}\s+\S+", line):
                headings.append((index, line.strip().lstrip("#").strip()))

        if not headings:
            return self._split_fallback(text)

        spans: list[Span] = []
        for idx, (start_line, heading) in enumerate(headings):
            end_line = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else len(lines)
            spans.extend(self._window_span(lines, start_line, end_line, heading=heading))
        return spans

    def _split_config(self, text: str) -> list[Span]:
        if self._estimate_tokens(text) <= self.chunk_tokens * 2:
            line_count = len(text.splitlines())
            return [Span(1, max(line_count, 1))]
        return self._split_fallback(text)

    def _split_python(self, text: str) -> list[Span]:
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self._split_fallback(text)

        symbols = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not symbols:
            return self._split_fallback(text)

        spans: list[Span] = []
        first_symbol_line = getattr(symbols[0], "lineno", 1)
        if first_symbol_line > 1:
            spans.extend(self._window_span(lines, 1, first_symbol_line - 1, symbol="module_preamble"))

        for node in symbols:
            start_line = getattr(node, "lineno", 1)
            end_line = getattr(node, "end_lineno", start_line)
            spans.extend(self._window_span(lines, start_line, end_line, symbol=getattr(node, "name", None)))
        return spans

    def _split_javascript(self, text: str) -> list[Span]:
        lines = text.splitlines()
        markers: list[tuple[int, str]] = []
        patterns = [
            r"^\s*export\s+(?:default\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)",
            r"^\s*(?:async\s+)?function\s+([A-Za-z0-9_]+)",
            r"^\s*export\s+class\s+([A-Za-z0-9_]+)",
            r"^\s*class\s+([A-Za-z0-9_]+)",
            r"^\s*export\s+const\s+([A-Za-z0-9_]+)\s*=",
            r"^\s*const\s+([A-Za-z0-9_]+)\s*=",
        ]

        for index, line in enumerate(lines, start=1):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    markers.append((index, match.group(1)))
                    break

        if not markers:
            return self._split_fallback(text)

        spans: list[Span] = []
        if markers[0][0] > 1:
            spans.extend(self._window_span(lines, 1, markers[0][0] - 1, symbol="module_preamble"))
        for idx, (start_line, symbol) in enumerate(markers):
            end_line = markers[idx + 1][0] - 1 if idx + 1 < len(markers) else len(lines)
            spans.extend(self._window_span(lines, start_line, end_line, symbol=symbol))
        return spans

    def _split_fallback(self, text: str) -> list[Span]:
        lines = text.splitlines()
        return self._window_span(lines, 1, len(lines))

    def _window_span(
        self,
        lines: list[str],
        start_line: int,
        end_line: int,
        *,
        symbol: str | None = None,
        heading: str | None = None,
    ) -> list[Span]:
        span_lines = lines[start_line - 1 : end_line]
        if not span_lines:
            return []

        if self._estimate_tokens("\n".join(span_lines)) <= self.chunk_tokens * 2:
            return [Span(start_line, end_line, symbol=symbol, heading=heading)]

        spans: list[Span] = []
        current_start = start_line
        current_tokens = 0
        overlap_lines = 0

        for index in range(start_line, end_line + 1):
            line = lines[index - 1]
            current_tokens += self._estimate_tokens(line)
            if current_tokens >= self.chunk_tokens:
                spans.append(Span(current_start, index, symbol=symbol, heading=heading))

                overlap_tokens = 0
                overlap_lines = 0
                back_index = index
                while back_index >= current_start:
                    overlap_tokens += self._estimate_tokens(lines[back_index - 1])
                    overlap_lines += 1
                    if overlap_tokens >= self.chunk_overlap:
                        break
                    back_index -= 1

                current_start = max(current_start + 1, index - overlap_lines + 1)
                current_tokens = self._estimate_tokens("\n".join(lines[current_start - 1 : index + 1]))

        if not spans or spans[-1].end_line < end_line:
            spans.append(Span(current_start, end_line, symbol=symbol, heading=heading))

        deduped: list[Span] = []
        seen: set[tuple[int, int]] = set()
        for span in spans:
            key = (span.start_line, span.end_line)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(span)
        return deduped

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _infer_language(self, path: str, fallback_language: str | None) -> str | None:
        lowered = path.lower()
        if lowered.endswith(".py"):
            return "Python"
        if lowered.endswith((".ts", ".tsx")):
            return "TypeScript"
        if lowered.endswith((".js", ".jsx")):
            return "JavaScript"
        if lowered.endswith(".go"):
            return "Go"
        if lowered.endswith(".rs"):
            return "Rust"
        if lowered.endswith(".java"):
            return "Java"
        if lowered.endswith(".md"):
            return "Markdown"
        return fallback_language
