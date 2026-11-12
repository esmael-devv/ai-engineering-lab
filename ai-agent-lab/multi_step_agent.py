
from langchain_core.messages import HumanMessage, ToolMessage

from multi_tool_agent import (
    model_with_tools,
    tool_registry
)


MAX_STEPS = 5



def run_agent(question: str):
    messages = [HumanMessage(content=question)]

    print(f"\nUSER: {question}")

    for step in range(1, MAX_STEPS + 2):

        # After the final tool round, require a final answer.
        active_model = (
            model_with_tools
            if step <= MAX_STEPS
            else model_with_tools.bind(
                tool_choice="none"
            )
        )

        response = active_model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\nFINAL ANSWER: {response.content}")
            return response.content

        if step > MAX_STEPS:
            raise RuntimeError(
                "Model requested tools after the step limit."
            )

        print(f"\nSTEP {step}")

        for call in response.tool_calls:
            name = call["name"]
            arguments = call["args"]

            try:
                if name not in tool_registry:
                    raise ValueError(f"Unknown tool: {name}")

                result = tool_registry[name].invoke(arguments)

            except Exception as error:
                result = (
                    f"TOOL_ERROR: {type(error).__name__}: {error}"
                )

            print(f"Tool: {name}")
            print(f"Arguments: {arguments}")
            print(f"Result: {result}")

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    raise RuntimeError("Agent could not complete the task.")

if __name__ == "__main__":
    run_agent(
        "Use the divide tool to calculate 100 divided by 0. "
        "If the tool fails, explain why."
    )
