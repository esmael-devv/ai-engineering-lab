
import sys
from pathlib import Path

import pytest
from tools.calculator import calculator_tool

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from schemas import AgentPlan
from executor import ToolRegistry, execute_plan


def make_plan(steps):
    return AgentPlan(
        requires_tools=bool(steps),
        steps=steps
    )


def test_single_tool_execution():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        lambda instruction, context: "Revenue: 120"
    )

    plan = make_plan([
        {
            "tool": "web_search",
            "instruction": "Find revenue"
        }
    ])

    result = execute_plan(
        plan, "Find revenue", registry
    )

    assert result["completed"] is True
    assert len(result["execution_log"]) == 1


def test_multi_tool_dependency():
    registry = ToolRegistry()

    def search(instruction, context):
        return "Revenue: current=120, previous=100"

    def calculate(instruction, context):
        previous = context.previous_results[0]["result"]

        assert "current=120" in previous

        return "Revenue growth: 20%"

    registry.register("web_search", search)
    registry.register("calculator", calculate)

    plan = make_plan([
        {
            "tool": "web_search",
            "instruction": "Find revenue figures"
        },
        {
            "tool": "calculator",
            "instruction": "Calculate growth"
        }
    ])

    result = execute_plan(
        plan, "Calculate revenue growth", registry
    )

    assert result["completed"] is True
    assert len(result["results"]) == 2
    assert result["results"][1]["result"] == (
        "Revenue growth: 20%"
    )


def test_unregistered_tool_fails():
    registry = ToolRegistry()

    plan = make_plan([
        {
            "tool": "web_search",
            "instruction": "Find information"
        }
    ])

    result = execute_plan(
        plan, "Search", registry
    )

    assert result["completed"] is False
    assert result["execution_log"][0]["success"] is False


def test_failure_stops_execution():
    registry = ToolRegistry()

    def broken_search(instruction, context):
        raise RuntimeError("Search unavailable")

    def calculator(instruction, context):
        pytest.fail("Calculator should not run")

    registry.register("web_search", broken_search)
    registry.register("calculator", calculator)

    plan = make_plan([
        {
            "tool": "web_search",
            "instruction": "Search"
        },
        {
            "tool": "calculator",
            "instruction": "Calculate"
        }
    ])

    result = execute_plan(
        plan, "Research and calculate", registry
    )

    assert result["completed"] is False
    assert len(result["execution_log"]) == 1


def test_duplicate_registration_rejected():
    registry = ToolRegistry()

    registry.register(
        "web_search",
        lambda instruction, context: "Result"
    )

    with pytest.raises(ValueError):
        registry.register(
            "web_search",
            lambda instruction, context: "Another result"
        )


def test_empty_plan():
    registry = ToolRegistry()

    result = execute_plan(
        make_plan([]),
        "Hello",
        registry
    )

    assert result["completed"] is True
    assert result["execution_log"] == []



def test_two_tool_revenue_growth():
    registry = ToolRegistry()

    # Simulates a successful web search.
    # These are illustrative figures, not real revenue data.
    def fake_web_search(instruction, context):
        return (
            '{"revenue_2024": 100, '
            '"revenue_2025": 120}'
        )

    registry.register("web_search", fake_web_search)
    registry.register("calculator", calculator_tool)

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "web_search",
                "instruction": "Find revenue for 2024 and 2025"
            },
            {
                "tool": "calculator",
                "instruction": "Calculate percentage growth",
                "arguments": {
                    "operation": "percent_change",
                    "a": "${step_1.revenue_2024}",
                    "b": "${step_1.revenue_2025}"
                }
            }
        ]
    })

    result = execute_plan(
        plan,
        "Find revenue growth",
        registry
    )

    assert result["completed"] is True
    assert len(result["results"]) == 2
    assert result["results"][1]["result"] == "20.0"


def test_missing_reference_stops_execution():
    registry = ToolRegistry()

    def fake_web_search(instruction, context):
        return '{"revenue_2024": 100}'

    registry.register("web_search", fake_web_search)
    registry.register("calculator", calculator_tool)

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "web_search",
                "instruction": "Find revenue"
            },
            {
                "tool": "calculator",
                "instruction": "Calculate growth",
                "arguments": {
                    "operation": "percent_change",
                    "a": "${step_1.revenue_2024}",
                    "b": "${step_1.revenue_2025}"
                }
            }
        ]
    })

    result = execute_plan(
        plan,
        "Calculate growth",
        registry
    )

    assert result["completed"] is False
    assert len(result["execution_log"]) == 2
    assert result["execution_log"][1]["success"] is False
    assert "revenue_2025" in (
        result["execution_log"][1]["error"]
    )


def test_reference_and_literal_combination():
    registry = ToolRegistry()

    def fake_web_search(instruction, context):
        return '{"price": 250}'

    registry.register("web_search", fake_web_search)
    registry.register("calculator", calculator_tool)

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "web_search",
                "instruction": "Find price"
            },
            {
                "tool": "calculator",
                "instruction": "Calculate 10% of the price",
                "arguments": {
                    "operation": "percentage_of",
                    "a": 10,
                    "b": "${step_1.price}"
                }
            }
        ]
    })

    result = execute_plan(
        plan,
        "What is 10% of the price?",
        registry
    )

    assert result["completed"] is True
    assert result["results"][1]["result"] == "25.0"
