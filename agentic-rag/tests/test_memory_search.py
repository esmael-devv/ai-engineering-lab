
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from executor import ToolRegistry, execute_plan
from schemas import AgentPlan
from tools.memory_search import build_memory_search_tool


TEST_MEMORIES = [
    ("favorite_language", "Java"),
    ("learning_langchain", "LangChain"),
    ("favorite_database", "PostgreSQL"),
]


def fake_memory_fn(session_id):
    assert session_id == "test-session"
    return TEST_MEMORIES


def test_find_saved_language():
    tool = build_memory_search_tool(
        "test-session",
        memory_fn=fake_memory_fn,
    )

    result = json.loads(
        tool("What is my favorite language?", None)
    )

    assert result["found"] is True
    assert any(
        item["value"] == "Java"
        for item in result["results"]
    )


def test_find_learning_topic():
    tool = build_memory_search_tool(
        "test-session",
        memory_fn=fake_memory_fn,
    )

    result = json.loads(
        tool("LangChain", None)
    )

    assert result["found"] is True
    assert result["results"][0]["value"] == "LangChain"


def test_unknown_memory_returns_empty():
    tool = build_memory_search_tool(
        "test-session",
        memory_fn=fake_memory_fn,
    )

    result = json.loads(
        tool("Kubernetes", None)
    )

    assert result["found"] is False
    assert result["count"] == 0


def test_missing_session_rejected():
    with pytest.raises(ValueError):
        build_memory_search_tool(
            "",
            memory_fn=fake_memory_fn,
        )


def test_empty_query_rejected():
    tool = build_memory_search_tool(
        "test-session",
        memory_fn=fake_memory_fn,
    )

    with pytest.raises(ValueError):
        tool(" ", None)


def test_memory_search_with_executor():
    registry = ToolRegistry()

    registry.register(
        "memory_search",
        build_memory_search_tool(
            "test-session",
            memory_fn=fake_memory_fn,
        ),
    )

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "memory_search",
                "instruction": "favorite language",
            }
        ],
    })

    execution = execute_plan(
        plan,
        "What is my favorite language?",
        registry,
    )

    assert execution["completed"] is True

    result = json.loads(
        execution["results"][0]["result"]
    )

    assert result["found"] is True
    assert result["results"][0]["value"] == "Java"
