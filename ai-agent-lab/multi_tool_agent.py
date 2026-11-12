
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv(
    Path(__file__).resolve().parents[2] / ".env"
)


# -----------------------------------------
# DEFINE TOOLS
# -----------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate a percentage of a given value."""
    return value * percentage / 100


@tool
def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. The divisor b must not be zero."""
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b


# -----------------------------------------
# CONFIGURE MODEL
# -----------------------------------------

tools = [add, multiply, calculate_percentage, word_count, divide]

tool_registry = {
    t.name: t for t in tools
}

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    reasoning_effort="low"
)

model_with_tools = model.bind_tools(tools)


# -----------------------------------------
# AGENT EXECUTION
# -----------------------------------------

def run_agent(question: str):
    messages = [HumanMessage(content=question)]

    response = model_with_tools.invoke(messages)
    messages.append(response)

    print(f"\nUSER: {question}")

    if not response.tool_calls:
        print("AI:", response.content)
        return

    for tool_call in response.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        if name not in tool_registry:
            raise ValueError(f"Unknown tool: {name}")

        result = tool_registry[name].invoke(args)

        print(f"TOOL: {name}")
        print(f"ARGS: {args}")
        print(f"RESULT: {result}")

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    final_response = model_with_tools.invoke(messages)

    print("AI:", final_response.content)


# -----------------------------------------
# TEST QUESTIONS
# -----------------------------------------

if __name__ == "__main__":
    questions = [
        "What is 15% of 240?",
        "How many words are in 'AI agents can execute Python tools'?",
        "What is 125 multiplied by 16?",
        "What is 450 plus 275?"
    ]

    for question in questions:
        run_agent(question)
