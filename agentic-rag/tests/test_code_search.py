
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from executor import ToolRegistry, execute_plan
from schemas import AgentPlan
from tools.code_search import (
    search_code,
    build_code_search_tool,
)


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "executor.py").write_text(
        "class ToolRegistry:\n"
        "    pass\n"
        "\n"
        "def execute_plan():\n"
        "    return True\n",
        encoding="utf-8",
    )

    (tmp_path / ".env").write_text(
        "SECRET_KEY=private",
        encoding="utf-8",
    )

    (tmp_path / ".venv").mkdir()

    (tmp_path / ".venv" / "hidden.py").write_text(
        "def secret_function(): pass",
        encoding="utf-8",
    )

    return tmp_path


def test_find_class_definition(sample_repo):
    results = search_code(
        sample_repo,
        "ToolRegistry",
    )

    assert results
    assert results[0]["file"] == "executor.py"
    assert results[0]["line"] == 1


def test_find_function_definition(sample_repo):
    results = search_code(
        sample_repo,
        "execute_plan",
    )

    assert any(
        item["line"] == 4
        for item in results
    )


def test_excludes_environment_files(sample_repo):
    results = search_code(
        sample_repo,
        "SECRET_KEY",
    )

    assert results == []


def test_excludes_virtual_environment(sample_repo):
    results = search_code(
        sample_repo,
        "secret_function",
    )

    assert results == []


def test_code_search_returns_json(sample_repo):
    tool = build_code_search_tool(sample_repo)

    output = json.loads(
        tool("Where is ToolRegistry defined?", None)
    )

    assert output["found"] is True
    assert output["results"][0]["file"] == (
        "executor.py"
    )


def test_code_search_with_executor(sample_repo):
    registry = ToolRegistry()

    registry.register(
        "code_search",
        build_code_search_tool(sample_repo),
    )

    plan = AgentPlan.model_validate({
        "requires_tools": True,
        "steps": [
            {
                "tool": "code_search",
                "instruction": "execute_plan",
            }
        ],
    })

    execution = execute_plan(
        plan,
        "Where is execute_plan defined?",
        registry,
    )

    assert execution["completed"] is True

    output = json.loads(
        execution["results"][0]["result"]
    )

    assert output["found"] is True
    assert output["results"][0]["line"] == 4
