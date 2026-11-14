
from dashboard_format import (
    format_answer,
    format_execution_summary,
    format_evidence_status,
)


def test_web_citation_becomes_link():
    sources = [{
        "id": "E1",
        "type": "web",
        "reference": "https://kafka.apache.org/",
    }]

    result = format_answer(
        "Kafka documentation [E1].",
        sources,
    )

    assert "[[E1]](https://kafka.apache.org/)" in result


def test_code_citation_remains_plain():
    sources = [{
        "id": "E2",
        "type": "code",
        "reference": "src/example.py:L10",
    }]

    result = format_answer(
        "Found a reference [E2].",
        sources,
    )

    assert result == "Found a reference [E2]."


def test_unknown_citation_is_not_linked():
    assert format_answer(
        "Unsupported citation [E99].",
        [],
    ) == "Unsupported citation [E99]."


def test_execution_summary():
    metrics = {
        "tool_calls": 3,
        "successful_calls": 2,
        "failed_calls": 1,
        "total_seconds": 4.25,
        "evidence_count": 8,
    }

    summary = format_execution_summary(metrics)

    assert "Tool calls:** 3" in summary
    assert "Failed:** 1" in summary
    assert "4.25s" in summary


def test_insufficient_evidence_shows_warning():
    metrics = {
        "evidence_sufficient": False,
        "verification_warnings": [
            "No implementation candidates found."
        ],
    }

    status = format_evidence_status(metrics)

    assert "needs review" in status
    assert "No implementation candidates found." in status
