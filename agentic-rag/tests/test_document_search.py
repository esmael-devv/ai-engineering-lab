
import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from tools.document_search import build_document_search_tool
from executor import ToolRegistry, execute_plan
from schemas import AgentPlan


def fake_search(collection_name, question):
    assert collection_name == "test_collection"

    documents = [
        Document(
            page_content="Revenue increased from 100 to 120.",
            metadata={
                "page": 2,
                "source": "annual_report.pdf"
            }
        )
    ]

    metrics = {
        "candidate_count": 4,
        "selected_count": 1
    }

    return documents, metrics


def test_document_search_returns_json():
    tool = build_document_search_tool(
        "test_collection",
        search_fn=fake_search
    )

    result = json.loads(
        tool("Find revenue growth", None)
    )

    assert result["found"] is True
    assert result["selected_count"] == 1
    assert result["matches"][0]["page"] == 3
    assert result["matches"][0]["source"] == (
        "annual_report.pdf"
    )


def test_empty_document_results():
    def empty_search(collection_name, question):
        return [], {"candidate_count": 0}

    tool = build_document_search_tool(
        "test_collection",
        search_fn=empty_search
    )

    result = json.loads(
        tool("Find revenue", None)
    )

    assert result["found"] is False
    assert result["matches"] == []


def test_document_search_with_executor():
    registry = ToolRegistry()

    registry.register(
        "document_search",
        build_document_search_tool(
            "test_collection",
            search_fn=fake_search
        )
    )

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "document_search",
                "instruction": "Find revenue growth"
            }
        ]
    })

    result = execute_plan(
        plan,
        "Find revenue growth",
        registry
    )

    assert result["completed"] is True

    output = json.loads(
        result["results"][0]["result"]
    )

    assert output["found"] is True
    assert output["matches"][0]["page"] == 3


def test_empty_collection_name_rejected():
    with pytest.raises(ValueError):
        build_document_search_tool("")
