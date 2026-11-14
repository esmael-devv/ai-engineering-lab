
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from executor import ExecutionContext
from tools.fact_extractor import (
    extract_revenue_facts,
    fact_extractor_tool,
)


def make_search_result(content):
    return {
        "found": True,
        "matches": [
            {
                "source": "annual_report.pdf",
                "page": 3,
                "content": content,
            }
        ],
    }


def test_extract_two_revenue_facts():
    data = make_search_result(
        "Revenue in 2024 was $100 million. "
        "Revenue in 2025 was $120 million."
    )

    result = extract_revenue_facts(data)

    assert result["count"] == 2
    assert result["facts"][0]["year"] == "2024"
    assert result["facts"][0]["amount_usd"] == "100000000"
    assert result["facts"][1]["amount_usd"] == "120000000"
    assert result["facts"][0]["page"] == 3


def test_unrelated_document_returns_no_facts():
    data = make_search_result(
        "PCA reduces the dimensionality of face images."
    )

    result = extract_revenue_facts(data)

    assert result["count"] == 0
    assert result["facts"] == []


def test_duplicate_revenue_is_not_counted_twice():
    data = make_search_result(
        "Revenue in 2024 was $100 million. "
        "Revenue in 2024 was $100 million."
    )

    result = extract_revenue_facts(data)

    assert result["count"] == 1


def test_conflicting_revenue_is_rejected():
    data = make_search_result(
        "Revenue in 2024 was $100 million. "
        "Revenue in 2024 was $150 million."
    )

    with pytest.raises(ValueError):
        extract_revenue_facts(data)


def test_missing_source_is_rejected():
    data = make_search_result(
        "Revenue in 2024 was $100 million."
    )
    data["matches"][0]["source"] = None

    with pytest.raises(ValueError):
        extract_revenue_facts(data)


def test_tool_reads_previous_document_result():
    context = ExecutionContext(
        original_request="Find revenue"
    )

    search_result = make_search_result(
        "Revenue in 2024 was $100 million."
    )

    context.previous_results.append({
        "step": 1,
        "tool": "document_search",
        "success": True,
        "result": json.dumps(search_result),
    })

    output = json.loads(
        fact_extractor_tool(
            instruction="Extract revenue",
            context=context,
        )
    )

    assert output["count"] == 1
    assert output["facts"][0]["year"] == "2024"
