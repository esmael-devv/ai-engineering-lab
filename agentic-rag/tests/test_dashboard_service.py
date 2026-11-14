
import json

import pytest

from dashboard_service import run_dashboard_request
from executor import ToolRegistry


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages):
        return FakeResponse(next(self.responses))


def test_dashboard_returns_answer_and_metrics():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        lambda instruction, context: json.dumps({
            "found": True,
            "results": [{
                "title": "Apache Kafka",
                "url": "https://kafka.apache.org/",
                "snippet": "Official Apache Kafka website",
            }],
        }),
    )

    model = FakeModel([
        json.dumps({
            "action": "tool",
            "step": {
                "tool": "web_search",
                "instruction": "Apache Kafka",
                "arguments": None,
            },
        }),
        json.dumps({
            "action": "finish",
            "step": None,
        }),
        json.dumps({
            "answer": "See Apache Kafka [E1].",
            "citations": ["E1"],
            "evidence_sufficient": True,
        }),
    ])

    result = run_dashboard_request(
        "Find Apache Kafka documentation",
        model=model,
        registry=registry,
    )

    assert "[E1]" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["cited"] is True
    assert result["metrics"]["tool_calls"] == 1
    assert result["metrics"]["failed_calls"] == 0
    assert result["metrics"]["total_seconds"] >= 0


def test_dashboard_rejects_empty_question():
    with pytest.raises(ValueError):
        run_dashboard_request(" ")
