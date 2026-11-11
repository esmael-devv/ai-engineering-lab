
import gradio as gr
import pandas as pd

from memory_battle import compare_strategies
from benchmark_explorer import available_runs, explore_run


def run_battle(session_id, question, expected_answer):
    if not session_id.strip() or not question.strip():
        raise gr.Error("Session ID and question are required.")

    results = compare_strategies(
        session_id.strip(),
        question.strip()
    )

    rows = []

    for result in results:
        answer = str(result["answer"])

        # Simple baseline: exact expected answer appears in response.
        # This is not a comprehensive factual accuracy evaluator.
        correct = (
            expected_answer.strip().casefold() in answer.casefold()
            if expected_answer.strip()
            else None
        )

        rows.append({
            "Strategy": result["strategy"],
            "Answer": answer,
            "Correct": correct,
            "Input Tokens": result["input_tokens"],
            "Output Tokens": result["output_tokens"],
            "Latency (ms)": result["latency_ms"]
        })

    df = pd.DataFrame(rows)

    return df, df, df



with gr.Blocks(title="AI Memory Battle") as demo:

    gr.Markdown("# 🧠 AI Memory Battle Arena")
    gr.Markdown(
        "Compare memory strategies and explore "
        "historical benchmark results."
    )

    with gr.Tabs():

        # TAB 1 — LIVE BATTLE
        with gr.Tab("⚔️ Live Battle"):

            session_id = gr.Textbox(
                label="Session ID",
                value="chat-7435783e"
            )

            question = gr.Textbox(
                label="Evaluation Question",
                value="What is my secret project code?"
            )

            expected = gr.Textbox(
                label="Expected Answer",
                value="NEBULA-7294"
            )

            run_button = gr.Button(
                "Run Memory Battle",
                variant="primary"
            )

            results_table = gr.Dataframe(
                label="Strategy Comparison"
            )

            token_chart = gr.BarPlot(
                x="Strategy",
                y="Input Tokens",
                title="Input Token Usage"
            )

            latency_chart = gr.BarPlot(
                x="Strategy",
                y="Latency (ms)",
                title="Response Latency"
            )

            run_button.click(
                fn=run_battle,
                inputs=[session_id, question, expected],
                outputs=[
                    results_table,
                    token_chart,
                    latency_chart
                ]
            )

        # TAB 2 — HISTORICAL BENCHMARKS
        with gr.Tab("📊 Benchmark Explorer"):

            gr.Markdown(
                "Analyze saved benchmark runs "
                "without making new Groq requests."
            )

            run_selector = gr.Dropdown(
                choices=available_runs(),
                label="Select Benchmark Run",
                value=(
                    available_runs()[0]
                    if available_runs()
                    else None
                )
            )

            explore_button = gr.Button(
                "Load Benchmark"
            )

            leaderboard_table = gr.Dataframe(
                label="Benchmark Summary"
            )

            detailed_table = gr.Dataframe(
                label="Individual Test Results"
            )

            explore_button.click(
                fn=explore_run,
                inputs=[run_selector],
                outputs=[
                    leaderboard_table,
                    detailed_table
                ]
            )


if __name__ == "__main__":
    demo.launch(show_error=True)



