
import json


SEARCH_TOOLS = {
    "document_search",
    "web_search",
    "code_search",
    "memory_search",
}


def evaluate_observation(record):
    """
    Classify one completed tool execution.

    This checks structural evidence availability,
    not whether evidence answers the user's question.
    """

    tool = record.get("tool")

    if not record.get("success"):
        return {
            "tool": tool,
            "status": "error",
            "reason": record.get("error", "Tool execution failed"),
            "has_evidence": False,
        }

    result = record.get("result")

    if tool == "calculator":
        if not isinstance(result, str) or not result.strip():
            return {
                "tool": tool,
                "status": "invalid",
                "reason": "Empty calculation result",
                "has_evidence": False,
            }

        return {
            "tool": tool,
            "status": "completed",
            "reason": "Calculation completed",
            "has_evidence": True,
        }

    if tool not in SEARCH_TOOLS:
        return {
            "tool": tool,
            "status": "unsupported",
            "reason": "Unknown tool type",
            "has_evidence": False,
        }

    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return {
            "tool": tool,
            "status": "invalid",
            "reason": "Invalid JSON tool output",
            "has_evidence": False,
        }

    if not isinstance(data, dict):
        return {
            "tool": tool,
            "status": "invalid",
            "reason": "Expected a JSON object",
            "has_evidence": False,
        }

    expected_field = (
        "matches"
        if tool == "document_search"
        else "results"
    )

    items = data.get(expected_field)

    if not isinstance(items, list):
        return {
            "tool": tool,
            "status": "invalid",
            "reason": f"Missing {expected_field} list",
            "has_evidence": False,
        }

    if not items:
        return {
            "tool": tool,
            "status": "not_found",
            "reason": "No matching evidence returned",
            "has_evidence": False,
        }

    return {
        "tool": tool,
        "status": "evidence_available",
        "reason": f"{len(items)} candidate results retrieved",
        "has_evidence": True,
        "candidate_count": len(items),
    }
