
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv(
    Path(__file__).resolve().parents[2] / ".env"
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    reasoning_effort="low"
)

tools = {
    multiply.name: multiply
}

model_with_tools = model.bind_tools([multiply])

messages = [
    HumanMessage(
        content="What is 25 multiplied by 48? Use the multiplication tool."
    )
]

# STEP 1: Ask Groq to select a tool
response = model_with_tools.invoke(messages)
messages.append(response)

print("\nRequested tool calls:")
print(response.tool_calls)

# STEP 2: Execute requested tools
for tool_call in response.tool_calls:
    tool_name = tool_call["name"]

    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")

    selected_tool = tools[tool_name]

    result = selected_tool.invoke(tool_call["args"])

    print(f"\nExecuted: {tool_name}")
    print(f"Arguments: {tool_call['args']}")
    print(f"Tool result: {result}")

    # STEP 3: Return the tool result to Groq
    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    )

# STEP 4: Ask Groq for its final answer
if response.tool_calls:
    final_response = model_with_tools.invoke(messages)
    print("\nFinal AI answer:")
    print(final_response.content)
else:
    print("\nAI answered without a tool:")
    print(response.content)
