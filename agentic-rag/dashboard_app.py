
import gradio as gr

from dashboard_service import run_dashboard_request

from dashboard_format import (
    format_answer,
    format_execution_summary,
    format_evidence_status,
)



EXAMPLES = [
    "What is my favorite programming language?",
    "Where is the ToolRegistry class defined?",
    "What is 15% of 850?",
    (
        "Find the official Apache Kafka documentation "
        "and locate Kafka-related code in my local repository. "
        "Use both web_search and code_search before finishing."
    ),
]



def process_question(question):
    try:
        result = run_dashboard_request(question)

        metrics = result["metrics"]

        formatted_answer = format_answer(
            result["answer"],
            result["sources"],
        )

        summary = format_execution_summary(metrics)
        evidence_status = format_evidence_status(metrics)

        source_rows = [
            [
                item["id"],
                item["type"],
                item["source"],
                item["reference"],
                "Yes" if item["cited"] else "No",
            ]
            for item in result["sources"]
        ]

        trace_rows = [
            [
                item["attempt"],
                item["tool"],
                item["observation"],
                "Yes" if item["success"] else "No",
                item["seconds"],
                item["error"] or "",
            ]
            for item in result["trace"]
        ]

        return (
            summary,
            evidence_status,
            formatted_answer,
            source_rows,
            trace_rows,
            metrics,
        )

    except Exception as error:
        return (
            "Agent execution failed.",
            "### ❌ Execution error",
            f"**Error:** {type(error).__name__}: {error}",
            [],
            [],
            {"status": "failed"},
        )


with gr.Blocks(title="Universal Knowledge Agent") as demo:

    gr.Markdown(
        "# 🧠 Universal Knowledge Agent\n"
        "Ask questions across documents, websites, "
        "source code, saved memory, and calculations.\n\n"
        "*Powered by Groq and reusable local tools.*"
    )

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="What would you like to investigate?",
            lines=3,
            scale=5,
        )

        run_button = gr.Button(
            "Run Agent",
            variant="primary",
            scale=1,
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=question,
    )


    gr.Markdown("### Execution overview")

    execution_summary = gr.Markdown(
        value="Run a question to see execution metrics."
    )

    evidence_status = gr.Markdown(
        value="Evidence status will appear here."
    )


    with gr.Tabs():

        with gr.Tab("Answer"):
            answer_output = gr.Markdown(
                value="Your answer will appear here."
            )

        with gr.Tab("Sources"):
            sources_output = gr.Dataframe(
                headers=[
                    "Evidence ID",
                    "Type",
                    "Source",
                    "Reference",
                    "Cited",
                ],
                datatype=[
                    "str", "str", "str", "str", "str"
                ],
                interactive=False,
                label="Retrieved evidence",
            )

        with gr.Tab("Execution Trace"):
            trace_output = gr.Dataframe(
                headers=[
                    "Attempt",
                    "Tool",
                    "Observation",
                    "Success",
                    "Seconds",
                    "Error",
                ],
                datatype=[
                    "number", "str", "str",
                    "str", "number", "str"
                ],
                interactive=False,
                label="Agent tool execution",
            )

        with gr.Tab("Metrics"):
            metrics_output = gr.JSON(
                label="Execution metrics"
            )

    dashboard_outputs = [
        execution_summary,
        evidence_status,
        answer_output,
        sources_output,
        trace_output,
        metrics_output,
    ]

    run_button.click(
        fn=process_question,
        inputs=question,
        outputs=dashboard_outputs,
    )

    question.submit(
        fn=process_question,
        inputs=question,
        outputs=dashboard_outputs,
    )


if __name__ == "__main__":
    demo.queue().launch()
