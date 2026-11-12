
from unittest.mock import patch

import agent_ui
from citation_validator import validate_citations


def test_citation_cleaning():
    answer = (
        "Population: 905748"
        "\u30103\u2020URL\u3011"
    )

    result = validate_citations(answer)

    assert result["valid"] is False
    assert result["cleaned_answer"] == "Population: 905748"

    print("PASS: Unsupported citations removed")


def test_dashboard_execution():
    mock_result = {
        "answer": "10% is 90,574.8.",
        "execution_log": [
            {
                "step": 1,
                "tool": "web_search",
                "arguments": {
                    "query": "Columbus population"
                },
                "result": (
                    "URL: https://www.census.gov/"
                    "quickfacts/columbuscityohio"
                ),
                "success": True
            },
            {
                "step": 2,
                "tool": "calculate_percentage",
                "arguments": {
                    "value": 905748,
                    "percentage": 10
                },
                "result": "90574.8",
                "success": True
            }
        ],
        "total_tool_calls": 2,
        "tool_errors": 0
    }

    with patch.object(
        agent_ui,
        "run_agent",
        return_value=mock_result
    ):
        answer, trace, metrics, sources = (
            agent_ui.execute_agent(
                "Find Columbus population and calculate 10%"
            )
        )

    assert "90,574.8" in answer
    assert len(trace) == 2
    assert "Total tool calls: 2" in metrics
    assert "Successful calls: 2" in metrics
    assert "census.gov" in sources

    print("PASS: Dashboard renders agent results")
    print("PASS: Tool metrics calculated correctly")
    print("PASS: Source URLs extracted successfully")


if __name__ == "__main__":
    test_citation_cleaning()
    test_dashboard_execution()

    print("\nALL AGENT UI TESTS PASSED")

