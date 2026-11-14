
import json

from adaptive_agent import run_adaptive_agent
from executor import ToolRegistry
from schemas import ToolStep


def test_two_step_adaptive_execution():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        lambda instruction, context: json.dumps({
            "found": True,
            "results": [
                {"title": "Example", "url": "https://example.com"}
            ],
        }),
    )

    registry.register(
        "calculator",
        lambda instruction, context: "20",
    )

    def decide(request, history, context):
        if not history:
            return ToolStep(
                tool="web_search",
                instruction="Find revenue data",
            )

        if len(history) == 1:
            assert history[0]["observation"]["has_evidence"]
            assert len(context.previous_results) == 1

            return ToolStep(
                tool="calculator",
                instruction="Calculate growth",
            )

        return None

    output = run_adaptive_agent(
        "Find revenue and calculate growth",
        registry,
        decide,
    )

    assert output["completed"] is True
    assert output["tool_calls"] == 2
    assert output["results"][-1]["result"] == "20"


def test_switch_tool_after_empty_search():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        lambda instruction, context: json.dumps({
            "found": False,
            "results": [],
        }),
    )

    registry.register(
        "memory_search",
        lambda instruction, context: json.dumps({
            "found": True,
            "results": [{"key": "language", "value": "Java"}],
        }),
    )

    def decide(request, history, context):
        if not history:
            return ToolStep(
                tool="web_search",
                instruction="Look up language",
            )

        if len(history) == 1:
            assert (
                history[0]["observation"]["status"]
                == "not_found"
            )

            return ToolStep(
                tool="memory_search",
                instruction="Find saved language",
            )

        return None

    output = run_adaptive_agent(
        "What language do I prefer?",
        registry,
        decide,
    )

    assert output["tool_calls"] == 2
    assert output["completed"] is True


def test_repeated_action_is_blocked():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        lambda instruction, context: json.dumps({
            "results": [],
        }),
    )

    def decide(request, history, context):
        return ToolStep(
            tool="web_search",
            instruction="Same search",
        )

    output = run_adaptive_agent(
        "Find information",
        registry,
        decide,
    )

    assert output["tool_calls"] == 1
    assert output["stop_reason"] == (
        "repeated_action_blocked"
    )


def test_tool_failure_is_observable():
    registry = ToolRegistry()

    def failing_tool(instruction, context):
        raise RuntimeError("Search unavailable")

    registry.register("web_search", failing_tool)

    def decide(request, history, context):
        if not history:
            return ToolStep(
                tool="web_search",
                instruction="Find data",
            )

        assert history[0]["observation"]["status"] == "error"
        return None

    output = run_adaptive_agent(
        "Find data",
        registry,
        decide,
    )

    assert output["tool_calls"] == 1
    assert output["completed"] is True
    assert output["history"][0]["record"]["success"] is False


def test_tool_call_limit():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        lambda instruction, context: json.dumps({
            "results": [],
        }),
    )

    def decide(request, history, context):
        return ToolStep(
            tool="web_search",
            instruction=f"Search attempt {len(history) + 1}",
        )

    output = run_adaptive_agent(
        "Find information",
        registry,
        decide,
        max_tool_calls=3,
    )

    assert output["tool_calls"] == 3
    assert output["completed"] is False
    assert output["stop_reason"] == "tool_limit_reached"
