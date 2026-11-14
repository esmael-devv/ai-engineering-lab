
import json

from adaptive_main import run_with_model
from executor import ToolRegistry


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages):
        return FakeResponse(next(self.responses))


def test_adaptive_runner_uses_registry():
    registry = ToolRegistry()

    registry.register(
        "memory_search",
        lambda instruction, context: json.dumps({
            "found": True,
            "results": [
                {
                    "key": "favorite_language",
                    "value": "Java",
                }
            ],
        }),
    )

    model = FakeModel([
        json.dumps({
            "action": "tool",
            "step": {
                "tool": "memory_search",
                "instruction": "favorite language",
            },
        }),
        json.dumps({
            "action": "finish",
            "step": None,
        }),
    ])

    result = run_with_model(
        "What is my favorite language?",
        model,
        registry,
    )

    assert result["completed"] is True
    assert result["tool_calls"] == 1
    assert result["history"][0]["record"]["tool"] == (
        "memory_search"
    )


def test_adaptive_runner_blocks_unavailable_tool():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        lambda instruction, context: "{}",
    )

    model = FakeModel([
        json.dumps({
            "action": "tool",
            "step": {
                "tool": "code_search",
                "instruction": "Find a class",
            },
        })
    ])

    try:
        run_with_model(
            "Find a class",
            model,
            registry,
        )
        assert False, "Expected unavailable tool error"

    except ValueError as error:
        assert "Unavailable tool" in str(error)
