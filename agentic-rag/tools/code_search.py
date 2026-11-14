
import json
import os
import re
from pathlib import Path


ALLOWED_EXTENSIONS = {
    ".py", ".java", ".cs", ".js", ".ts",
    ".json", ".yaml", ".yml", ".md",
    ".xml", ".sql", ".properties",
}

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", ".pytest_cache", ".idea",
    "dist", "build", "target", "bin", "obj",
    ".next",
}

EXCLUDED_FILES = {
    ".env", ".env.local", ".env.production",
}

STOP_WORDS = {
    "where", "what", "which", "who", "how",
    "is", "are", "the", "a", "an", "in",
    "of", "for", "to", "do", "does",
    "find", "show", "me", "code", "file",
    "defined", "definition", "implemented",
    "implementation", "function", "class",
}

MAX_FILES = 2000
MAX_FILE_BYTES = 128 * 1024
MAX_RESULTS = 8


def extract_terms(query):
    terms = re.findall(
        r"[A-Za-z_][A-Za-z_0-9]*",
        query
    )

    return [
        term.lower()
        for term in terms
        if term.lower() not in STOP_WORDS
        and len(term) >= 2
    ]


def search_code(root, query):
    """
    Read-only local repository search.

    Returns matching file paths, line numbers,
    and source code snippets.
    """

    root = Path(root).resolve()

    if not root.is_dir():
        raise ValueError("Repository directory not found")

    terms = extract_terms(query)

    if not terms:
        return []

    matches = []
    scanned_files = 0

    for current_dir, dirs, files in os.walk(
        root, followlinks=False
    ):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in EXCLUDED_DIRS
            and not (Path(current_dir) / directory).is_symlink()
        )

        for filename in sorted(files):
            path = Path(current_dir) / filename

            if (
                filename in EXCLUDED_FILES
                or filename.startswith(".env.")
                or path.suffix.lower() not in ALLOWED_EXTENSIONS
                or path.is_symlink()
            ):
                continue

            if scanned_files >= MAX_FILES:
                break

            scanned_files += 1

            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue

                lines = path.read_text(
                    encoding="utf-8"
                ).splitlines()

            except (OSError, UnicodeError):
                continue

            for line_number, line in enumerate(
                lines, start=1
            ):
                lower = line.lower()

                score = sum(
                    term in lower for term in set(terms)
                )

                if score == 0:
                    continue

                matches.append({
                    "file": path.relative_to(root).as_posix(),
                    "line": line_number,
                    "snippet": line.strip()[:300],
                    "score": score,
                })

        if scanned_files >= MAX_FILES:
            break

    matches.sort(
        key=lambda item: (
            -item["score"],
            item["file"],
            item["line"],
        )
    )

    return matches[:MAX_RESULTS]


def build_code_search_tool(repository_root):
    """
    Executor-compatible adapter.

    The repository path is fixed by the application,
    not selected by the LLM.
    """

    repository_root = Path(repository_root).resolve()

    if not repository_root.is_dir():
        raise ValueError("Invalid repository root")

    def code_search_tool(instruction, context):
        query = (instruction or "").strip()

        if not query:
            raise ValueError("Code search query is empty")

        if len(query) > 500:
            raise ValueError("Code search query too long")

        results = search_code(
            repository_root,
            query,
        )

        return json.dumps({
            "query": query,
            "found": bool(results),
            "count": len(results),
            "results": results,
        })

    return code_search_tool
