
import importlib.util
import json
import re
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

MEMORY_FILE = (
    REPO_ROOT / "memory-chat" / "memory_vault.py"
)

STOP_WORDS = {
    "what", "where", "when", "which", "who",
    "how", "did", "do", "does", "is", "are",
    "the", "my", "me", "i", "a", "an", "to",
    "about", "tell", "show", "remember",
}

MAX_RESULTS = 10


@lru_cache(maxsize=1)
def get_existing_memory_module():
    """
    Reuse the persistent SQLite memory vault
    implemented in Project 2.
    """

    if not MEMORY_FILE.is_file():
        raise FileNotFoundError(
            f"Memory vault not found: {MEMORY_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "existing_memory_vault",
        MEMORY_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Cannot load memory vault")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def retrieve_session_memories(session_id):
    module = get_existing_memory_module()

    return module.get_memories(session_id)


def build_memory_search_tool(
    session_id,
    memory_fn=None,
):
    """
    Search saved facts belonging to one session.

    The session_id is controlled by the application,
    never by the LLM.
    """

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A session ID is required")

    memory_fn = (
        retrieve_session_memories
        if memory_fn is None
        else memory_fn
    )

    def memory_search_tool(instruction, context):
        query = (instruction or "").strip()

        if not query:
            raise ValueError("Memory query cannot be empty")

        if len(query) > 500:
            raise ValueError("Memory query is too long")

        terms = {
            word.lower()
            for word in re.findall(r"[A-Za-z_0-9]+", query)
            if word.lower() not in STOP_WORDS
        }

        if not terms:
            return json.dumps({
                "query": query,
                "found": False,
                "count": 0,
                "results": [],
            })

        memories = memory_fn(session_id)
        matches = []

        for key, value in memories:
            key = str(key)
            value = str(value)

            searchable = (
                key.replace("_", " ") + " " + value
            ).lower()

            score = sum(
                term in searchable
                for term in terms
            )

            if score:
                matches.append({
                    "key": key,
                    "value": value,
                    "score": score,
                    "source": "memory_vault",
                })

        matches.sort(
            key=lambda result: (
                -result["score"],
                result["key"],
            )
        )

        matches = matches[:MAX_RESULTS]

        return json.dumps({
            "query": query,
            "found": bool(matches),
            "count": len(matches),
            "results": matches,
        })

    return memory_search_tool
