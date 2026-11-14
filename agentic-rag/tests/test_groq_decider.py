
import json

import pytest

from groq_decider import (
    create_groq_decider,
    build_decision_context,
)
from executor import ExecutionContext


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        return FakeResponse(
            next(self.responses)
        )


def make_context():
    return ExecutionContext(
        original_request="Find information"
    )


def test_groq_selects_web_search():
    model = FakeModel([
        json.dumps({
            "action": "tool",
            "step": {
                "tool": "web_search",
                "instruction": "Apache Kafka documentation",
            },
        })
    ])

    decide = create_groq_decider(
        model,
        {"web_search", "calculator"},
    )

    action = decide(
        "Find Kafka documentation",
        [],
        make_context(),
    )

    assert action.tool == "web_search"
    assert action.instruction == (
        "Apache Kafka documentation"
    )


def test_groq_can_finish():
    model = FakeModel([
        '{"action":"finish","step":null}'
    ])

    decide = create_groq_decider(
        model,
        {"web_search"},
    )

    assert decide(
        "Find information",
        [],
        make_context(),
    ) is None


def test_unavailable_tool_rejected():
    model = FakeModel([
        json.dumps({
            "action": "tool",
            "step": {
                "tool": "code_search",
                "instruction": "Find class",
            },
        })
    ])

    decide = create_groq_decider(
        model,
        {"web_search"},
    )

    with pytest.raises(ValueError):
        decide(
            "Find class",
            [],
            make_context(),
        )


def test_invalid_json_rejected():
    model = FakeModel([
        "I would search the web."
    ])

    decide = create_groq_decider(
        model,
        {"web_search"},
    )

    with pytest.raises(ValueError):
        decide(
            "Find information",
            [],
            make_context(),
        )


def test_previous_observation_reaches_model():
    model = FakeModel([
        '{"action":"finish","step":null}'
    ])

    decide = create_groq_decider(
        model,
        {"web_search"},
    )

    history = [{
        "record": {
            "tool": "web_search",
            "instruction": "Original search",
            "success": True,
            "result": '{"found":false,"results":[]}',
        },
        "observation": {
            "status": "not_found",
            "has_evidence": False,
        },
    }]

    decide(
        "Find information",
        history,
        make_context(),
    )

    message_content = model.calls[0][1].content

    assert "not_found" in message_content
    assert "Original search" in message_content
    assert "web_search" in message_content


def test_search_tool_rejects_string_arguments():
    model = FakeModel([
        json.dumps({
            "action": "tool",
            "step": {
                "tool": "web_search",
                "instruction": "Apache Kafka documentation",
                "arguments": "Apache Kafka documentation",
            },
        })
    ])

    decide = create_groq_decider(
        model,
        {"web_search"},
    )

    with pytest.raises(ValueError):
        decide(
            "Find Kafka documentation",
            [],
            make_context(),
        )
