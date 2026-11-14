
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from planner import create_plan


def test_repository_question_uses_code_search():
    class FakeResponse:
        content = """
        {
          "requires_tools": true,
          "steps": [
            {
              "tool": "code_search",
              "instruction": "ToolRegistry"
            }
          ]
        }
        """

    class FakeModel:
        def invoke(self, messages):
            assert any(
                "code_search" in str(message.content)
                for message in messages
            )
            return FakeResponse()

    plan = create_plan(
        "Where is ToolRegistry class defined "
        "in my repository?",
        FakeModel(),
    )

    assert plan.steps[0].tool == "code_search"
