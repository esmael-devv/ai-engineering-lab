
from pathlib import Path
from planner import create_planner_model, create_plan
from executor import ToolRegistry, execute_plan
from tools.calculator import calculator_tool
from tools.web_search import build_web_search_tool
from tools.code_search import build_code_search_tool
from tools.memory_search import build_memory_search_tool

def build_registry():
    registry = ToolRegistry()

    registry.register(
        "calculator",
        calculator_tool
    )
    registry.register(
        "web_search",
        build_web_search_tool()
    )
    registry.register(
        "code_search",
        build_code_search_tool(
            Path(__file__).resolve().parent.parent
        )
    )
    registry.register(
        "memory_search",
        build_memory_search_tool(
            session_id="chat-1"
        )
    )

    return registry


def run_agent(request, model, registry):

    print("\n" + "=" * 60)
    print("USER REQUEST:", request)

    # Step 1: Ask Groq to generate a plan
    plan = create_plan(request, model)

    print("\n=== GENERATED PLAN ===")
    print(plan.model_dump_json(indent=2))

    # Step 2: Refuse to execute unregistered tools
    available_tools = {
        "calculator",
        "web_search",
        "code_search",
        "memory_search",
    }

    for step in plan.steps:
        if step.tool not in available_tools:
            raise ValueError(
                f"Tool not implemented yet: {step.tool}"
            )

        if step.tool == "calculator" and step.arguments is None:
            raise ValueError(
                "Calculator requires structured arguments. "
                "No operands were provided."
            )

    # Step 3: Execute the validated plan
    result = execute_plan(
        plan=plan,
        request=request,
        registry=registry
    )

    print("\n=== EXECUTION TRACE ===")

    for entry in result["execution_log"]:
        print(
            f"Step {entry['step']} | "
            f"Tool: {entry['tool']} | "
            f"Success: {entry['success']} | "
            f"Result: {entry['result']}"
        )

        if not entry["success"]:
            print("Error:", entry.get("error"))

    print("\n=== FINAL RESULT ===")

    if result["completed"]:
        if result["results"]:
            print(result["results"][-1]["result"])
        else:
            print("No tool execution required.")
    else:
        print("Execution failed.")

    return result


def main():
    model = create_planner_model()
    registry = build_registry()

    print("\nUNIVERSAL KNOWLEDGE AGENT")
    print("Step 5E - End-to-End Calculator")
    print("Type 'exit' to quit.")

    while True:
        try:
            request = input("\nRequest: ").strip()

        except (KeyboardInterrupt, EOFError):
            break

        if request.lower() in {"exit", "quit"}:
            break

        if not request:
            continue

        try:
            run_agent(request, model, registry)
        except Exception as error:
            print(
                f"AGENT ERROR: "
                f"{type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    main()
