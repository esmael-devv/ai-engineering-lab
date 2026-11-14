
import json

from observation import evaluate_observation


def test_successful_memory_search():
    record = {
        "tool": "memory_search",
        "success": True,
        "result": json.dumps({
            "found": True,
            "results": [
                {"key": "favorite_language", "value": "Java"}
            ]
        }),
    }

    observation = evaluate_observation(record)

    assert observation["status"] == "evidence_available"
    assert observation["has_evidence"] is True


def test_search_without_results():
    record = {
        "tool": "web_search",
        "success": True,
        "result": json.dumps({
            "found": False,
            "results": [],
        }),
    }

    observation = evaluate_observation(record)

    assert observation["status"] == "not_found"
    assert observation["has_evidence"] is False


def test_failed_tool_execution():
    record = {
        "tool": "code_search",
        "success": False,
        "error": "Repository unavailable",
        "result": None,
    }

    observation = evaluate_observation(record)

    assert observation["status"] == "error"


def test_invalid_search_output():
    record = {
        "tool": "document_search",
        "success": True,
        "result": "This is not JSON",
    }

    observation = evaluate_observation(record)

    assert observation["status"] == "invalid"
