
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from schemas import AgentPlan


def test_direct_plan():
    plan = AgentPlan(
        requires_tools=False,
        steps=[]
    )

    assert plan.steps == []


def test_multi_tool_plan():
    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "document_search",
                "instruction": "Find revenue"
            },
            {
                "tool": "calculator",
                "instruction": "Calculate growth"
            }
        ]
    })

    assert len(plan.steps) == 2


def test_unknown_tool_rejected():
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({
            "requires_tools": True,
            "steps": [
                {
                    "tool": "delete_database",
                    "instruction": "Delete everything"
                }
            ]
        })


def test_inconsistent_plan_rejected():
    with pytest.raises(ValidationError):
        AgentPlan(
            requires_tools=False,
            steps=[
                {
                    "tool": "web_search",
                    "instruction": "Find information"
                }
            ]
        )


def test_too_many_steps_rejected():
    with pytest.raises(ValidationError):
        AgentPlan(
            requires_tools=True,
            steps=[
                {
                    "tool": "web_search",
                    "instruction": "Search"
                }
                for _ in range(6)
            ]
        )


def test_structured_calculator_arguments():
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

    args = plan.steps[0].arguments

    assert args.operation == "percentage_of"
    assert args.a == 15
    assert args.b == 850


def test_invalid_calculator_operation():
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({
            "requires_tools": True,
            "steps": [
                {
                    "tool": "calculator",
                    "instruction": "Run arbitrary code",
                    "arguments": {
                        "operation": "execute_python",
                        "a": 10,
                        "b": 20
                    }
                }
            ]
        })


def test_calculator_arguments_on_web_tool_rejected():
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({
            "requires_tools": True,
            "steps": [
                {
                    "tool": "web_search",
                    "instruction": "Find current data",
                    "arguments": {
                        "operation": "add",
                        "a": 10,
                        "b": 20
                    }
                }
            ]
        })
