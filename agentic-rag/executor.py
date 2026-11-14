
import time
import json

from reference_resolver import resolve_value
from dataclasses import dataclass, field
from typing import Callable

from schemas import AgentPlan


@dataclass
class ExecutionContext:
    original_request: str
    previous_results: list[dict] = field(
        default_factory=list
    )


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    
    def names(self):
        return frozenset(self._tools.keys())


    def register(self, name, handler):
        if name in self._tools:
            raise ValueError(
                f"Tool already registered: {name}"
            )

        if not callable(handler):
            raise TypeError("Tool handler must be callable")

        self._tools[name] = handler

    def get(self, name):
        if name not in self._tools:
            raise ValueError(
                f"Tool not registered: {name}"
            )

        return self._tools[name]


def execute_plan(plan, request, registry, context=None):
    """
    Execute a validated plan sequentially.

    Stops on the first failed tool.
    Does not make any LLM calls.
    """

    if not isinstance(plan, AgentPlan):
        raise TypeError("Expected a validated AgentPlan")

    if context is None:
        context = ExecutionContext(original_request=request)
    elif context.original_request != request:
        raise ValueError("Context belongs to another request")

    execution_log = []

    for index, step in enumerate(
        plan.steps,
        start=len(context.previous_results) + 1
    ):
        start = time.perf_counter()
        try:
            handler = registry.get(step.tool)

            if step.tool == "calculator" and step.arguments is not None:

                # Obtain validated calculator arguments
                arguments = step.arguments.model_dump()

                # Resolve references against completed tool results
                arguments["a"] = resolve_value(
                    arguments["a"],
                    context
                )

                arguments["b"] = resolve_value(
                    arguments["b"],
                    context
                )

                # Convert resolved arguments into the format
                # expected by calculator_tool()
                tool_instruction = json.dumps(arguments)

            else:
                # Preserve compatibility with existing tools
                tool_instruction = step.instruction

            result = handler(
                instruction=tool_instruction,
                context=context
            )

            if not isinstance(result, str):
                raise TypeError(
                    "Tool must return a string"
                )

            elapsed = time.perf_counter() - start

            record = {
                "step": index,
                "tool": step.tool,
                "instruction": step.instruction,
                "result": result,
                "success": True,
                "elapsed_seconds": round(elapsed, 4),
            }

            execution_log.append(record)
            context.previous_results.append(record)

        except Exception as error:
            elapsed = time.perf_counter() - start

            execution_log.append({
                "step": index,
                "tool": step.tool,
                "instruction": step.instruction,
                "result": None,
                "success": False,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(elapsed, 4),
            })

            # Do not run dependent steps after failure.
            break

    return {
        "completed": (
            len(execution_log) == len(plan.steps)
            and all(item["success"] for item in execution_log)
        ),
        "execution_log": execution_log,
        "results": context.previous_results,
    }



def names(self):
    """Return the registered tool names."""
    return frozenset(self._tools)
