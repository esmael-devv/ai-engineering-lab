
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from planner import create_plan


def fake_model(response_text):
    model = MagicMock()
    model.invoke.return_value.content = response_text
    return model


def test_direct_plan():
    model = fake_model("""
    {
        "requires_tools": false,
        "steps": []
    }
    """)

    plan = create_plan("Hello!", model)

    assert plan.requires_tools is False
    assert len(plan.steps) == 0


def test_multi_tool_plan():
    model = fake_model("""
    {
        "requires_tools": true,
        "steps": [
            {
                "tool": "web_search",
                "instruction": "Find revenue figures"
            },
            {
                "tool": "calculator",
                "instruction": "Calculate growth"
            }
        ]
    }
    """)

    plan = create_plan(
        "Find revenue and calculate growth",
        model
    )

    assert plan.requires_tools is True
    assert len(plan.steps) == 2
    assert plan.steps[0].tool == "web_search"
    assert plan.steps[1].tool == "calculator"


def test_invalid_json():
    model = fake_model("This is not JSON")

    with pytest.raises(ValueError):
        create_plan("Research something", model)


def test_unauthorized_tool_rejected():
    model = fake_model("""
    {
        "requires_tools": true,
        "steps": [
            {
                "tool": "delete_database",
                "instruction": "Delete data"
            }
        ]
    }
    """)

    with pytest.raises(ValueError):
        create_plan("Delete data", model)


def test_empty_request():
    model = MagicMock()

    with pytest.raises(ValueError):
        create_plan("", model)

    model.invoke.assert_not_called()


def test_planner_returns_structured_calculator_args():
    model = fake_model("""
    {
        "requires_tools": true,
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
    }
    """)

    plan = create_plan(
        "What is 15% of 850?",
        model
    )

    assert plan.steps[0].tool == "calculator"
    assert plan.steps[0].arguments.operation == "percentage_of"
    assert plan.steps[0].arguments.a == 15
    assert plan.steps[0].arguments.b == 850
