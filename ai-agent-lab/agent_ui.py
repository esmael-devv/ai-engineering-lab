
import gradio as gr
import pandas as pd

from research_calculator_agent import run_agent
import re
from urllib.parse import urlparse
from citation_validator import validate_citations


def extract_source_urls(execution_log):
    """Collect unique source URLs from successful web searches."""
    urls = []
    seen = set()

    for entry in execution_log:
        if entry.get("tool") != "web_search":
            continue

        if not entry.get("success"):
            continue

        text = str(entry.get("result", ""))

        matches = re.findall(
            r"https?://[^\s<>]+",
            text
        )

        for match in matches:
            url = match.rstrip(".,;)]")

            parsed = urlparse(url)

            if parsed.scheme not in ("http", "https"):
                continue

            if not parsed.netloc or url in seen:
                continue

            seen.add(url)
            urls.append(url)

    return urls


def execute_agent(question):
    question = (question or "").strip()

    if not question:
        raise gr.Error("Enter a research question.")

    try:
        result = run_agent(question)
    except Exception as error:
        raise gr.Error(
            f"Agent execution failed: {type(error).__name__}: {error}"
        )

    if not isinstance(result, dict):
        raise gr.Error(
            "Agent returned an unexpected result format."
        )

    execution_log = result.get("execution_log", [])

    columns = [
        "Step",
        "Tool",
        "Arguments",
        "Result",
        "Success"
    ]

    rows = [
        {
            "Step": entry["step"],
            "Tool": entry["tool"],
            "Arguments": str(entry["arguments"]),
            "Result": entry["result"],
            "Success": entry["success"]
        }
        for entry in execution_log
    ]

    trace = pd.DataFrame(rows, columns=columns)

    total = len(execution_log)
    successful = sum(
        bool(entry["success"]) for entry in execution_log
    )
    errors = total - successful

    metrics = (
        f"Total tool calls: {total}\n"
        f"Successful calls: {successful}\n"
        f"Tool errors: {errors}"
    )

    validation = validate_citations(
        str(result["answer"])
    )

    cleaned_answer = validation["cleaned_answer"]

    source_urls = extract_source_urls(execution_log)

    if source_urls:
        sources = "\n".join(
            f"- {url}" for url in source_urls
        )
    else:
        sources = "No source URLs were retrieved."

    return cleaned_answer, trace, metrics, sources


with gr.Blocks(title="AI Agent Lab") as demo:

    gr.Markdown("# 🤖 AI Agent Lab")
    gr.Markdown(
        "Research Assistant · Tool Calling · "
        "Multi-Step Execution · Groq"
    )

    question = gr.Textbox(
        label="Research Question",
        placeholder=(
            "Find Columbus's 2020 population and "
            "calculate 10% of it."
        ),
        lines=3
    )

    run_button = gr.Button(
        "🚀 Run Agent",
        variant="primary"
    )

    answer = gr.Markdown(
        label="Research Answer"
    )

    gr.Markdown("### 🔍 Agent Execution Trace")

    execution_table = gr.Dataframe(
        headers=[
            "Step",
            "Tool",
            "Arguments",
            "Result",
            "Success"
        ],
        label="Tool Execution History",
        interactive=False
    )

    metrics_box = gr.Textbox(
        label="Agent Metrics",
        lines=3,
        interactive=False
    )

    sources_box = gr.Markdown(
        value="Source URLs will appear here."
    )

    run_button.click(
        fn=execute_agent,
        inputs=question,
        outputs=[
            answer,
            execution_table,
            metrics_box,
            sources_box
        ]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True
    )
