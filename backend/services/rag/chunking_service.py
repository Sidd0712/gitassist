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

_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "e2e", "testing"}
_EXAMPLE_DIRS = {"example", "examples", "demo", "demos", "sample", "samples", "playground"}
_DOC_DIRS = {"docs", "doc", "documentation", "website"}
_CONFIG_FILENAMES = {
    "package.json", "cargo.toml", "go.mod", "pyproject.toml", "requirements.txt", "setup.cfg",
    "setup.py", "tsconfig.json", "gemfile", "composer.json", "pom.xml", "build.gradle",
}
_ENTRY_FILENAMES = {"main.py", "app.py", "server.py", "index.js", "index.ts", "main.go", "main.rs", "manage.py"}
_ENTRY_PATH_TOKENS = ("/router", "/routes", "/api/", "/handler", "/controller", "/views", "/endpoints")

# Top-level declarations only (no leading whitespace). Splitting at indented
# `const x =` lines inside functions used to shred a 17 KB file into 173
# fragments, many a single line with no context.
_JS_BOUNDARIES = re.compile(
    r"^(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:async\s+)?"
    r"(?:function\*?\s+(?P<fn>[\w$]+)|(?:abstract\s+)?class\s+(?P<cls>[\w$]+)"
    r"|(?:const|let|var)\s+(?P<var>[\w$]+)\s*[=:]|(?:interface|type|enum)\s+(?P<type>[\w$]+))"
    r"|^export\s+default\b"
)
# Best-effort label for a window inside a larger span: the first definition,
# method or HTTP route it contains.
_WINDOW_SYMBOL = re.compile(
    r"^\s*(?:(?:export\s+)?(?:async\s+)?(?:def|function|func|fn|class|interface|struct|impl)\s+([\w$]+)"
    r"|(?:app|router|server|api)\.(get|post|put|patch|delete|use)\(\s*['\"`]([^'\"`]+)"
    r"|(?:public|private|protected|static|async)[\w\s<>\[\],]*?\s([\w$]+)\s*\([^;]*$)",
    re.M,
)


@dataclass
class Span:
    """Simple 1-based line span."""

    start_line: int
    end_line: int
    symbol: str | None = None
    heading: str | None = None


class ChunkingService:
    """Create role-aware, size-bounded chunks for repository files."""

    def __init__(self) -> None:
        settings = get_settings()
        self.target_chars = settings.RAG_CHUNK_TARGET_CHARS
        self.max_chars = settings.RAG_CHUNK_MAX_CHARS
        self.overlap_lines = settings.RAG_CHUNK_OVERLAP_LINES
        self.chunking_version = settings.RAG_CHUNKING_VERSION

    def chunk_repository(self, repository: RepoSearchResult, files: list[RepoFile]) -> list[CorpusChunk]:
        """Chunk every fetched repo file, preserving file order."""

        chunks: list[CorpusChunk] = []
        for repo_file in files:
            chunks.extend(self.chunk_file(repository, repo_file))
        return chunks

    @staticmethod
    def embedding_text(chunk: CorpusChunk) -> str:
        """Text that gets embedded: a location header plus the chunk.

        The header lets "where is the auth middleware" match
        src/middleware/auth.ts even when the code never says so. It is not
        stored, so snippets shown to users stay exactly as written.
        """

        label = chunk.symbol or chunk.heading
        header = f"{chunk.repo_full_name} · {chunk.path}" + (f" · {label}" if label else "")
        return f"{header}\n{chunk.text}"

    def chunk_file(self, repository: RepoSearchResult, repo_file: RepoFile) -> list[CorpusChunk]:
        """Chunk a single file based on its type."""

        text = repo_file.content.replace("\r\n", "\n").strip("\n")
        if not text.strip():
            return []
        lines = text.split("\n")

        role = self.classify_path(repo_file.path)
        lowered = repo_file.path.lower()
        if role == "documentation":
            spans = self._split_documentation(lines)
        elif lowered.endswith((".py", ".ipynb")):
            spans = self._split_python(text, lines)
        elif lowered.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte")):
            spans = self._split_javascript(lines)
        else:
            spans = [Span(1, len(lines))]

        spans = self._bound(self._merge_small(spans, lines), lines)

        chunks: list[CorpusChunk] = []
        for span in spans:
            # Trim blank edge lines from the span itself, so start_line/end_line
            # map exactly onto the stored text (chat stitches neighbours by line).
            while span.start_line < span.end_line and not lines[span.start_line - 1].strip():
                span.start_line += 1
            while span.end_line > span.start_line and not lines[span.end_line - 1].strip():
                span.end_line -= 1
            chunk_text = "\n".join(lines[span.start_line - 1 : span.end_line])
            if not chunk_text.strip():
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
                    token_count=max(1, len(chunk_text) // 4),
                    repo_score=repository.relevance_score,
                    content_hash=content_hash,
                    text=chunk_text,
                )
            )
        return chunks

    @staticmethod
    def classify_path(path: str) -> str:
        """Classify a file's role from its path segments and extension."""

        parts = path.lower().split("/")
        filename, dirs = parts[-1], set(parts[:-1])

        if filename in _CONFIG_FILENAMES:
            return "config"
        # HTML is almost always content (docs sites, blog pages, templates), not logic.
        if filename.startswith("readme") or filename.endswith((".md", ".rst", ".txt", ".html", ".htm")) or dirs & _DOC_DIRS:
            return "documentation"
        if dirs & _TEST_DIRS or ".test." in filename or ".spec." in filename or filename.startswith("test_"):
            return "test"
        if dirs & _EXAMPLE_DIRS:
            return "example"
        if (
            filename.startswith(("dockerfile", "makefile", ".env", ".git"))
            or filename.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml"))
            or "config" in filename
        ):
            return "config"
        lowered = "/" + path.lower()
        if filename in _ENTRY_FILENAMES or any(token in lowered for token in _ENTRY_PATH_TOKENS):
            return "entrypoint"
        return "source"

    # ── Splitters: find natural boundaries; sizing happens afterwards ─────────

    def _split_documentation(self, lines: list[str]) -> list[Span]:
        headings = [
            (index, line.strip().lstrip("#").strip())
            for index, line in enumerate(lines, start=1)
            if re.match(r"^\s{0,3}#{1,6}\s+\S+", line)
        ]
        if not headings:
            return [Span(1, len(lines))]

        spans: list[Span] = []
        if headings[0][0] > 1:
            spans.append(Span(1, headings[0][0] - 1))
        for idx, (start_line, heading) in enumerate(headings):
            end_line = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else len(lines)
            spans.append(Span(start_line, end_line, heading=heading))
        return spans

    def _split_python(self, text: str, lines: list[str]) -> list[Span]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [Span(1, len(lines))]

        spans: list[Span] = []
        cursor = 1
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            end = node.end_lineno or node.lineno
            if start > cursor:
                spans.append(Span(cursor, start - 1, symbol="module"))
            spans.extend(self._python_node_spans(node, start, end, lines))
            cursor = end + 1
        if cursor <= len(lines):
            spans.append(Span(cursor, len(lines), symbol="module" if spans else None))
        return spans or [Span(1, len(lines))]

    def _python_node_spans(self, node, start: int, end: int, lines: list[str]) -> list[Span]:
        """Large classes split per method so each chunk is one coherent unit."""

        if not isinstance(node, ast.ClassDef) or self._span_chars(lines, start, end) <= self.max_chars:
            return [Span(start, end, symbol=node.name)]

        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not methods:
            return [Span(start, end, symbol=node.name)]

        spans: list[Span] = []
        cursor = start
        for method in methods:
            m_start = min([method.lineno, *(d.lineno for d in method.decorator_list)])
            m_end = method.end_lineno or method.lineno
            if m_start > cursor:
                spans.append(Span(cursor, m_start - 1, symbol=node.name))
            spans.append(Span(m_start, m_end, symbol=f"{node.name}.{method.name}"))
            cursor = m_end + 1
        if cursor <= end:
            spans.append(Span(cursor, end, symbol=node.name))
        return spans

    def _split_javascript(self, lines: list[str]) -> list[Span]:
        markers: list[tuple[int, str | None]] = []
        for index, line in enumerate(lines, start=1):
            match = _JS_BOUNDARIES.match(line)
            if match:
                name = next((g for g in match.group("fn", "cls", "var", "type") if g), None)
                markers.append((index, name or "default export"))

        if not markers:
            return [Span(1, len(lines))]

        spans: list[Span] = []
        if markers[0][0] > 1:
            spans.append(Span(1, markers[0][0] - 1, symbol="imports"))
        for idx, (start_line, symbol) in enumerate(markers):
            end_line = markers[idx + 1][0] - 1 if idx + 1 < len(markers) else len(lines)
            spans.append(Span(start_line, end_line, symbol=symbol))
        return spans

    # ── Sizing ────────────────────────────────────────────────────────────────

    def _span_chars(self, lines: list[str], start: int, end: int) -> int:
        return sum(len(line) + 1 for line in lines[start - 1 : end])

    def _merge_small(self, spans: list[Span], lines: list[str]) -> list[Span]:
        """Fold neighbouring small spans together up to the target size."""

        merged: list[Span] = []
        for span in spans:
            if merged:
                previous = merged[-1]
                combined = self._span_chars(lines, previous.start_line, span.end_line)
                tiny = self._span_chars(lines, span.start_line, span.end_line) < self.target_chars // 5
                if combined <= self.target_chars or (tiny and combined <= self.max_chars):
                    names = [
                        name
                        for symbol in (previous.symbol, span.symbol)
                        for name in (symbol or "").split(", ")
                        if name and name not in {"module", "imports"}
                    ]
                    merged[-1] = Span(
                        previous.start_line,
                        span.end_line,
                        symbol=", ".join(dict.fromkeys(names))[:120] or previous.symbol,
                        heading=previous.heading or span.heading,
                    )
                    continue
            merged.append(span)
        return merged

    def _bound(self, spans: list[Span], lines: list[str]) -> list[Span]:
        """Window any span over max_chars, preferring blank-line breaks."""

        bounded: list[Span] = []
        for span in spans:
            if self._span_chars(lines, span.start_line, span.end_line) <= self.max_chars:
                bounded.append(span)
                continue

            start = span.start_line
            while True:
                end = start
                size = len(lines[start - 1]) + 1
                while end < span.end_line:
                    if size >= self.target_chars and not lines[end - 1].strip():
                        break  # past the target and at a blank line: natural break
                    next_size = len(lines[end]) + 1  # lines[end] is 1-based line end+1
                    if size + next_size > self.max_chars:
                        break
                    size += next_size
                    end += 1
                window_symbol = self._window_symbol("\n".join(lines[start - 1 : end]))
                if span.symbol and window_symbol and window_symbol not in span.symbol:
                    symbol = f"{span.symbol} › {window_symbol}"
                else:
                    symbol = span.symbol or window_symbol
                bounded.append(Span(start, end, symbol=symbol, heading=span.heading))
                if end >= span.end_line:
                    break
                start = max(start + 1, end + 1 - self.overlap_lines)
        return bounded

    @staticmethod
    def _window_symbol(text: str) -> str | None:
        match = _WINDOW_SYMBOL.search(text)
        if not match:
            return None
        if match.group(2):
            return f"{match.group(2).upper()} {match.group(3)}"
        return match.group(1) or match.group(4)

    def _infer_language(self, path: str, fallback_language: str | None) -> str | None:
        lowered = path.lower()
        for suffixes, language in (
            ((".py", ".ipynb"), "Python"),
            ((".ts", ".tsx"), "TypeScript"),
            ((".js", ".jsx", ".mjs", ".cjs"), "JavaScript"),
            ((".go",), "Go"),
            ((".rs",), "Rust"),
            ((".java",), "Java"),
            ((".kt",), "Kotlin"),
            ((".rb",), "Ruby"),
            ((".php",), "PHP"),
            ((".cs",), "C#"),
            ((".c", ".h"), "C"),
            ((".cpp", ".hpp"), "C++"),
            ((".swift",), "Swift"),
            ((".vue",), "Vue"),
            ((".svelte",), "Svelte"),
            ((".sql",), "SQL"),
            ((".md", ".rst"), "Markdown"),
        ):
            if lowered.endswith(suffixes):
                return language
        return fallback_language
