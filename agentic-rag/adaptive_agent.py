
import json

from executor import (
    ExecutionContext,
    execute_plan,
)
from observation import evaluate_observation
from schemas import AgentPlan, ToolStep


MAX_TOOL_CALLS = 5


def run_adaptive_agent(
    request,
    registry,
    decide_next,
    max_tool_calls=MAX_TOOL_CALLS,
):
    """
    Bounded adaptive execution loop.

    decide_next(request, history, context)
    returns:
      - ToolStep: execute the next tool
      - None: finish

    The decision provider is injectable so
    execution can be tested without Groq.
    """

    if not isinstance(request, str) or not request.strip():
        raise ValueError("Request cannot be empty")

    if not 1 <= max_tool_calls <= 10:
        raise ValueError("Invalid tool-call limit")

    context = ExecutionContext(
        original_request=request
    )

    history = []
    attempted_actions = set()

    stop_reason = "tool_limit_reached"

    for _ in range(max_tool_calls):
        action = decide_next(
            request,
            history,
            context,
        )

        if action is None:
            stop_reason = "decision_finished"
            break

        if not isinstance(action, ToolStep):
            raise TypeError(
                "Decision must return ToolStep or None"
            )

        fingerprint = json.dumps(
            action.model_dump(),
            sort_keys=True,
        )

        if fingerprint in attempted_actions:
            stop_reason = "repeated_action_blocked"
            break

        attempted_actions.add(fingerprint)

        plan = AgentPlan(
            requires_tools=True,
            steps=[action],
        )

        execution = execute_plan(
            plan,
            request,
            registry,
            context=context,
        )

        record = execution["execution_log"][0]

        observation = evaluate_observation(record)

        history.append({
            "record": record,
            "observation": observation,
        })

    return {
        "request": request,
        "completed": stop_reason == "decision_finished",
        "stop_reason": stop_reason,
        "tool_calls": len(history),
        "history": history,
        "results": context.previous_results,
    }
