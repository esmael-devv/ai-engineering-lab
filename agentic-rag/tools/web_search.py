
import importlib.util
import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]

LEGACY_SEARCH_FILE = (
    REPO_ROOT / "ai-agent-lab" / "research_tools.py"
)


@lru_cache(maxsize=1)
def get_existing_search_tool():
    """
    Load the existing LangChain web_search tool
    from Project 3.

    No duplicate DuckDuckGo implementation.
    """

    if not LEGACY_SEARCH_FILE.is_file():
        raise FileNotFoundError(
            f"Existing search tool not found: "
            f"{LEGACY_SEARCH_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "legacy_research_tools",
        LEGACY_SEARCH_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Cannot load research_tools.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.web_search


def run_existing_search(query):
    """
    Adapter between Project 3's LangChain tool
    and Project 5's executor.
    """

    search_tool = get_existing_search_tool()

    return search_tool.invoke({
        "query": query
    })


def parse_search_results(raw):
    """
    Parse the existing search tool's formatted text.

    Returns structured search-result dictionaries.
    """

    if not isinstance(raw, str):
        raise TypeError("Search backend must return text")

    if raw.startswith("SEARCH_ERROR:"):
        raise RuntimeError(raw)

    if raw.strip() == "No search results found.":
        return []

    results = []

    # Project 3 separates results with blank lines.
    blocks = re.split(r"\n\s*\n", raw.strip())

    for block in blocks:
        lines = block.splitlines()

        if not lines:
            continue

        title_match = re.match(
            r"^\d+\.\s+(.+)$",
            lines[0].strip()
        )

        if not title_match:
            continue

        title = title_match.group(1).strip()

        url = ""
        snippet_lines = []
        reading_snippet = False

        for line in lines[1:]:
            stripped = line.strip()

            if stripped.startswith("URL:"):
                url = stripped[4:].strip()

            elif stripped.startswith("Snippet:"):
                reading_snippet = True
                snippet_lines.append(
                    stripped[len("Snippet:"):].strip()
                )

            elif reading_snippet:
                snippet_lines.append(stripped)

        parsed_url = urlparse(url)

        if (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.netloc
        ):
            continue

        results.append({
            "title": title,
            "url": url,
            "snippet": " ".join(snippet_lines),
        })

    return results


def build_web_search_tool(search_fn=None):
    """
    Executor-compatible web search tool.

    search_fn allows mocked searches in unit tests.
    """

    search_fn = (
        run_existing_search
        if search_fn is None
        else search_fn
    )

    def web_search_tool(instruction, context):
        query = (instruction or "").strip()

        if not query:
            raise ValueError("Search query cannot be empty")

        if len(query) > 500:
            raise ValueError("Search query is too long")

        raw = search_fn(query)

        results = parse_search_results(raw)

        if raw.strip() and not results:
            if raw.strip() != "No search results found.":
                raise ValueError(
                    "Search response could not be parsed"
                )

        return json.dumps({
            "query": query,
            "found": bool(results),
            "count": len(results),
            "results": results,
        }, ensure_ascii=False)

    return web_search_tool
