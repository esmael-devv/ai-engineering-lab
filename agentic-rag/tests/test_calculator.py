
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from tools.calculator import calculate, calculator_tool
from executor import ToolRegistry, execute_plan
from schemas import AgentPlan


def test_addition():
    assert calculate("add", 10, 20) == 30


def test_percentage():
    assert calculate(
        "percentage_of", 15, 850
    ) == 127.5


def test_revenue_growth():
    assert calculate(
        "percent_change", 100, 120
    ) == 20


def test_division_by_zero():
    with pytest.raises(ValueError):
        calculate("divide", 10, 0)


def test_invalid_operation():
    with pytest.raises(ValueError):
        calculate("execute_code", 10, 20)


def test_invalid_json_instruction():
    with pytest.raises(ValueError):
        calculator_tool(
            "Calculate 15% of 850",
            context=None
        )


def test_calculator_with_executor():
    registry = ToolRegistry()

    registry.register(
        "calculator",
        calculator_tool
    )

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "calculator",
                "instruction": (
                    '{"operation":"percentage_of",'
                    '"a":15,"b":850}'
                )
            }
        ]
    })

    result = execute_plan(
        plan,
        "What is 15% of 850?",
        registry
    )

    assert result["completed"] is True
    assert result["results"][0]["result"] == "127.50"


def test_structured_calculator_execution():
    registry = ToolRegistry()
    registry.register("calculator", calculator_tool)

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "calculator",
                "instruction": "Calculate 15% of 850",
                "arguments": {
                    "operation": "percentage_of",
                    "a": 15,
                    "b": 850
                }
            }
        ]
    })

    result = execute_plan(
        plan,
        "What is 15% of 850?",
        registry
    )

    assert result["completed"] is True
    assert result["results"][0]["result"] == "127.50"


def test_structured_revenue_growth():
    registry = ToolRegistry()
    registry.register("calculator", calculator_tool)

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "calculator",
                "instruction": "Calculate revenue growth",
                "arguments": {
                    "operation": "percent_change",
                    "a": 100,
                    "b": 120
                }
            }
        ]
    })

    result = execute_plan(
        plan,
        "Revenue increased from 100 to 120",
        registry
    )

    assert result["completed"] is True
    assert result["results"][0]["result"] == "20.0"
