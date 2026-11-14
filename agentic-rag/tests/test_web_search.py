
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
from tools.web_search import (
    build_web_search_tool,
    parse_search_results,
)


FAKE_SEARCH_RESPONSE = """
1. Apache Kafka Documentation
   URL: https://kafka.apache.org/documentation/
   Snippet: Official documentation for Apache Kafka.

2. Apache Kafka Overview
   URL: https://kafka.apache.org/
   Snippet: Apache Kafka is an event streaming platform.
"""


def test_parse_existing_search_format():
    results = parse_search_results(
        FAKE_SEARCH_RESPONSE
    )

    assert len(results) == 2
    assert results[0]["title"] == (
        "Apache Kafka Documentation"
    )
    assert results[0]["url"] == (
        "https://kafka.apache.org/documentation/"
    )


def test_web_search_returns_structured_json():
    tool = build_web_search_tool(
        search_fn=lambda query: FAKE_SEARCH_RESPONSE
    )

    result = json.loads(
        tool("Apache Kafka documentation", None)
    )

    assert result["found"] is True
    assert result["count"] == 2
    assert len(result["results"]) == 2


def test_empty_search_results():
    tool = build_web_search_tool(
        search_fn=lambda query: (
            "No search results found."
        )
    )

    result = json.loads(
        tool("Unknown topic", None)
    )

    assert result["found"] is False
    assert result["results"] == []


def test_search_error_propagates():
    tool = build_web_search_tool(
        search_fn=lambda query: (
            "SEARCH_ERROR: RuntimeError: Rate limited"
        )
    )

    with pytest.raises(RuntimeError):
        tool("Apache Kafka", None)


def test_invalid_query_rejected():
    tool = build_web_search_tool(
        search_fn=lambda query: FAKE_SEARCH_RESPONSE
    )

    with pytest.raises(ValueError):
        tool(" ", None)


def test_web_search_with_executor():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        build_web_search_tool(
            search_fn=lambda query: FAKE_SEARCH_RESPONSE
        )
    )

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "web_search",
                "instruction": (
                    "Find official Apache Kafka documentation"
                )
            }
        ]
    })

    result = execute_plan(
        plan,
        "Find Kafka documentation",
        registry
    )

    assert result["completed"] is True

    output = json.loads(
        result["results"][0]["result"]
    )

    assert output["count"] == 2
    assert output["results"][0]["url"].startswith(
        "https://"
    )
