"""Real, no-mocks smoke test: submit an idea, wait for indexing, chat about it,
and verify every file path the app claims actually exists on GitHub.

This is the manual verification process from the 2026-09-25 end-to-end test,
turned into a repeatable script instead of a one-off. It hits a live backend
(default http://127.0.0.1:8000) and the real GitHub API — it needs a running
worker to see indexing/chat actually work, and GITHUB_TOKEN in .env to avoid
GitHub's unauthenticated rate limit.

Usage (from backend/):
    python scripts/e2e_smoke_test.py "a personal finance tracker app" \
        --answers '{"frontend_framework": "React Native"}' \
        --questions '["How does X work?", "Where is Y implemented?"]'

If the idea needs clarification and --answers isn't given, the questions are
printed and the script exits — rerun with --answers filled in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from core.config import get_settings  # noqa: E402

_DEFAULT_QUESTIONS = [
    "How does this app's core feature actually work, with real code?",
    "Which repo should I base my project on, and what should I watch out for?",
]


def run_report(base: str, idea: str, answers: dict[str, str]) -> dict:
    """Submit an idea via SSE, return the final result payload."""

    result: dict | None = None
    with httpx.Client(timeout=180.0) as client:
        with client.stream(
            "POST", f"{base}/api/research", json={"idea": idea, "clarification_answers": answers}
        ) as resp:
            resp.raise_for_status()
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    for line in block.splitlines():
                        if not line.startswith("data: "):
                            continue
                        payload = json.loads(line[6:])
                        if payload.get("type") == "progress":
                            print(f"  ... {payload.get('message')}", flush=True)
                        elif payload.get("type") == "result":
                            result = payload.get("data")
                        elif payload.get("type") == "error":
                            raise RuntimeError(payload.get("message"))
    if result is None:
        raise RuntimeError("Stream ended without a result event.")
    return result


def wait_for_searchable(base: str, scope: list[dict], timeout_s: float = 240.0) -> None:
    started = time.monotonic()
    with httpx.Client(timeout=30.0) as client:
        while time.monotonic() - started < timeout_s:
            status = client.post(f"{base}/api/research/index-status", json={"repositories": scope}).json()
            if any(r["state"] in ("partial", "completed") for r in status["repositories"]):
                return
            time.sleep(3)
    raise TimeoutError(f"No repo became searchable within {timeout_s}s")


def verify_paths(token: str | None, claims: set[tuple[str, str]]) -> list[str]:
    """Check (full_name, path) pairs against real GitHub trees. Returns failures."""

    headers = {"Authorization": f"token {token}"} if token else {}
    failures: list[str] = []
    trees: dict[str, set[str]] = {}
    with httpx.Client(timeout=20.0) as client:
        for full_name, path in sorted(claims):
            if full_name not in trees:
                r = client.get(f"https://api.github.com/repos/{full_name}/git/trees/HEAD?recursive=1", headers=headers)
                trees[full_name] = {t["path"] for t in r.json().get("tree", [])} if r.status_code == 200 else set()
            if path not in trees[full_name]:
                failures.append(f"{full_name}: {path}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("idea")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--answers", default="{}", help="JSON clarification answers")
    parser.add_argument("--questions", default=json.dumps(_DEFAULT_QUESTIONS), help="JSON list of chat questions")
    args = parser.parse_args()

    base = args.base
    answers = json.loads(args.answers)
    questions = json.loads(args.questions)
    settings = get_settings()

    print(f"Submitting idea: {args.idea!r}")
    result = run_report(base, args.idea, answers)

    if result["status"] == "needs_clarification":
        print("\nNeeds clarification — rerun with --answers:")
        print(json.dumps(result["clarification_questions"], indent=2))
        return 1

    repos = result["repositories"]
    print(f"\n{len(repos)} repos returned:")
    for r in repos:
        print(f"  - {r['full_name']} (fit={r.get('fit_score')})")

    # Report descriptions are free text (no structured path field to check
    # without re-parsing prose), so this verifies the other half of the
    # grounding claim: every path chat actually cites as evidence.
    scope = [{"full_name": r["full_name"], "commit_sha": r["commit_sha"]} for r in repos if r.get("commit_sha")]
    print("\nWaiting for at least one repo to become searchable...")
    wait_for_searchable(base, scope)

    print("\nAsking chat questions:")
    citation_claims: set[tuple[str, str]] = set()
    messages: list[dict] = []
    with httpx.Client(timeout=90.0) as client:
        for q in questions:
            resp = client.post(
                f"{base}/api/research/chat",
                json={"question": q, "idea_summary": result["idea_summary"], "scope_repositories": scope, "messages": messages},
            )
            if resp.status_code != 200:
                print(f"  FAIL [{resp.status_code}] {q}: {resp.text[:300]}")
                continue
            data = resp.json()
            print(f"  OK ({data['scoped_repo_count']}/{len(scope)} repos) {q}")
            for c in data["citations"]:
                citation_claims.add((c["repo_full_name"], c["path"]))
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": data["answer"]})

    print(f"\nVerifying {len(citation_claims)} unique chat citations against GitHub...")
    failures = verify_paths(settings.GITHUB_TOKEN, citation_claims)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"FAILED: {len(failures)} citation(s) don't exist on GitHub:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: all {len(citation_claims)} chat citations verified real. Zero fabrications.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
