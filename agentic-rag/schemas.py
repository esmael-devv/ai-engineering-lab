
from typing import Literal

from pydantic import BaseModel, Field, model_validator


ToolName = Literal[
    "document_search",
    "web_search",
    "code_search",
    "memory_search",
    "calculator",
]


from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
        "percentage_of",
        "percent_change",
    ]

    a: int | float | str
    b: int | float | str




class ToolStep(BaseModel):
    tool: ToolName
    instruction: str = Field(min_length=1)

    # Optional during migration from the old planner.
    arguments: CalculatorArguments | None = None

    @model_validator(mode="after")
    def validate_arguments(self):
        if self.arguments is not None and self.tool != "calculator":
            raise ValueError(
                "Calculator arguments cannot be used "
                "with another tool"
            )

        return self



class AgentPlan(BaseModel):
    requires_tools: bool
    steps: list[ToolStep] = Field(
        default_factory=list,
        max_length=5,
    )

    @model_validator(mode="after")
    def validate_plan(self):
        if self.requires_tools != bool(self.steps):
            raise ValueError(
                "requires_tools must match whether steps exist"
            )

        return self
