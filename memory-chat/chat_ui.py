
import uuid

import gradio as gr
import pandas as pd

from memory_chat import chatbot, memory_optimizer
from analytics import invoke_with_metrics
from sqlite_memory import SQLiteChatHistory, list_sessions
from memory_vault import get_memories, save_memory, delete_memory
from memory_validator import validate_memory
from memory_extractor import process_memory

from memory_battle import compare_strategies
from benchmark_explorer import available_runs, explore_run
from usage_tracker import get_preprocessing_usage


# ==================================================
# SESSION MANAGEMENT
# ==================================================

def new_session():
    return f"chat-{uuid.uuid4().hex[:8]}"


def load_history(session_id):
    if not session_id:
        return []

    messages = SQLiteChatHistory(session_id).messages

    return [
        {
            "role": "user" if msg.type == "human" else "assistant",
            "content": str(msg.content)
        }
        for msg in messages
        if msg.type in ("human", "ai")
    ]


def start_new_chat():
    session_id = new_session()

    return (
        session_id,
        session_id,
        [],
        "New conversation started.",
        "No memories saved for this conversation.",
        "",
        "",
        ""
    )


def switch_session(session_id):
    if not session_id:
        raise gr.Error("Select a conversation.")

    saved_sessions = {sid for sid, _ in list_sessions()}

    if session_id not in saved_sessions:
        raise gr.Error("Conversation not found.")

    return (
        session_id,
        session_id,
        load_history(session_id),
        f"Loaded {session_id}",
        display_memories(session_id),
        "",
        "",
        ""
    )


def refresh_sessions():
    sessions = [sid for sid, _ in list_sessions()]

    return gr.update(choices=sessions)


# ==================================================
# CHAT
# ==================================================

def send_message(question, session_id):
    question = (question or "").strip()

    if not question:
        return (
            load_history(session_id),
            "",
            "No message sent.",
            display_memories(session_id)
        )

    saved_memories = get_memories(session_id)

    memory_context = "\n".join(
        f"{key}: {value}"
        for key, value in saved_memories
    )

    try:
        response, metrics = invoke_with_metrics(
            chatbot,
            question,
            session_id,
            memory_context
        )
    except Exception as error:
        raise gr.Error(
            f"Chat request failed: {error}"
        )

    try:
        process_memory(session_id, question)
    except Exception as error:
        print(f"Memory extraction failed: {error}")

    metrics_text = (
        f"Latency: {metrics.latency_ms:.0f} ms | "
        f"Input: {metrics.input_tokens} | "
        f"Output: {metrics.output_tokens} | "
        f"Total: {metrics.total_tokens}"
    )

    return (
        load_history(session_id),
        "",
        metrics_text,
        display_memories(session_id)
    )


# ==================================================
# MEMORY VAULT
# ==================================================

def display_memories(session_id):
    if not session_id:
        return "No conversation selected."

    memories = get_memories(session_id)

    if not memories:
        return "No memories saved for this conversation."

    return "\n".join(
        f"{key}: {value}"
        for key, value in memories
    )


def remember_fact(session_id, key, value):
    validated = validate_memory(key, value)

    if validated is None:
        return (
            "Invalid memory key or value.",
            display_memories(session_id)
        )

    normalized_key, normalized_value = validated

    save_memory(
        session_id,
        normalized_key,
        normalized_value
    )

    return (
        f"Saved: {normalized_key} = {normalized_value}",
        display_memories(session_id)
    )


def forget_fact(session_id, key):
    validated = validate_memory(key, "placeholder")

    if validated is None:
        return (
            "Invalid memory key.",
            display_memories(session_id)
        )

    normalized_key, _ = validated

    deleted = delete_memory(session_id, normalized_key)

    status = (
        f"Deleted: {normalized_key}"
        if deleted
        else f"Memory not found: {normalized_key}"
    )

    return status, display_memories(session_id)


# ==================================================
# SESSION DELETION
# ==================================================

def delete_chat_history(session_id):
    if not session_id:
        return [], "No active conversation.", ""

    SQLiteChatHistory(session_id).clear()

    return (
        [],
        f"Deleted conversation history for {session_id}",
        ""
    )


# ==================================================
# MEMORY DEBUG
# ==================================================

def debug_memory(session_id):
    if not session_id:
        return "No active conversation."

    history = SQLiteChatHistory(session_id).messages

    optimized = memory_optimizer.invoke({
        "history": history
    })

    retained = optimized["history"]

    report = [
        f"Session: {session_id}",
        f"Stored messages: {len(history)}",
        f"Retained messages: {len(retained)}",
        "",
        "RETAINED MESSAGES:"
    ]

    if not retained:
        report.append("(No messages retained)")

    for message in retained:
        report.append(
            f"[{message.type}] {str(message.content)[:200]}"
        )

    return "\n".join(report)


# ==================================================
# PREPROCESSING ANALYTICS
# ==================================================

def show_preprocessing_usage(session_id):
    if not session_id:
        return pd.DataFrame()

    records = get_preprocessing_usage(session_id)

    columns = [
        "Operation",
        "Model",
        "Calls",
        "Input Tokens",
        "Output Tokens",
        "Total Tokens"
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    rows = []

    for record in records:
        input_tokens = record["input_tokens"]
        output_tokens = record["output_tokens"]

        rows.append({
            "Operation": record["operation"],
            "Model": record["model"],
            "Calls": record["calls"],
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens,
            "Total Tokens": input_tokens + output_tokens
        })

    return pd.DataFrame(rows, columns=columns)


# ==================================================
# MEMORY BATTLE ARENA
# ==================================================

def run_battle(session_id, question, expected_answer):
    session_id = (session_id or "").strip()
    question = (question or "").strip()
    expected_answer = (expected_answer or "").strip()

    if not session_id or not question:
        raise gr.Error(
            "Session ID and evaluation question are required."
        )

    results = compare_strategies(
        session_id,
        question
    )

    rows = []

    for result in results:
        answer = str(result.get("answer", ""))
        status = result.get("status", "unknown")

        correct = None

        if status == "completed" and expected_answer:
            correct = (
                expected_answer.casefold() in answer.casefold()
            )

        rows.append({
            "Strategy": result["strategy"],
            "Answer": answer,
            "Status": status,
            "Correct": correct,
            "Input Tokens": result.get("input_tokens"),
            "Output Tokens": result.get("output_tokens"),
            "Latency (ms)": result.get("latency_ms")
        })

    df = pd.DataFrame(rows)

    token_df = df[
        ["Strategy", "Input Tokens"]
    ].dropna(subset=["Input Tokens"]).copy()

    latency_df = df[
        ["Strategy", "Latency (ms)"]
    ].dropna(subset=["Latency (ms)"]).copy()

    return df, token_df, latency_df


def use_active_session(session_id):
    return session_id


# ==================================================
# BENCHMARK EXPLORER
# ==================================================

def refresh_benchmark_runs():
    runs = available_runs()

    return gr.update(
        choices=runs,
        value=runs[0] if runs else None
    )


def load_benchmark(filename):
    if not filename:
        raise gr.Error("No benchmark run selected.")

    return explore_run(filename)


# ==================================================
# GRADIO APPLICATION
# ==================================================

with gr.Blocks(title="AI Memory Lab") as demo:

    gr.Markdown("# 🧠 AI Memory Lab")

    gr.Markdown(
        "Groq · LangChain · SQLite · Persistent Memory · "
        "Memory Strategy Evaluation"
    )

    session_state = gr.State(value=new_session)

    with gr.Tabs():

        # ==========================================
        # TAB 1 - CHAT
        # ==========================================

        with gr.Tab("💬 AI Chat"):

            gr.Markdown(
                "### Persistent Memory Chatbot"
            )

            with gr.Row():

                session_label = gr.Textbox(
                    label="Active Session",
                    interactive=False
                )

                new_button = gr.Button(
                    "➕ New Chat"
                )

            with gr.Row():

                session_dropdown = gr.Dropdown(
                    choices=[
                        sid for sid, _ in list_sessions()
                    ],
                    label="Saved Conversations"
                )

                refresh_button = gr.Button(
                    "🔄 Refresh Sessions"
                )

            chat = gr.Chatbot(
                label="Conversation",
                height=450
            )

            message = gr.Textbox(
                label="Message",
                placeholder="Ask me anything...",
                lines=2
            )

            send_button = gr.Button(
                "Send",
                variant="primary"
            )

            metrics_box = gr.Textbox(
                label="Latest Response Metrics",
                interactive=False
            )

            # --------------------------------------
            # MEMORY VAULT
            # --------------------------------------

            with gr.Accordion(
                "🧠 Memory Vault",
                open=False
            ):

                memory_display = gr.Textbox(
                    label="Saved Memories",
                    lines=8,
                    interactive=False
                )

                view_memories_button = gr.Button(
                    "🔄 Refresh Memories"
                )

                with gr.Row():

                    memory_key = gr.Textbox(
                        label="Memory Key",
                        placeholder="favorite_language"
                    )

                    memory_value = gr.Textbox(
                        label="Memory Value",
                        placeholder="Java"
                    )

                with gr.Row():

                    remember_button = gr.Button(
                        "💾 Save Memory"
                    )

                    forget_button = gr.Button(
                        "🗑️ Forget Memory"
                    )

                memory_status = gr.Textbox(
                    label="Memory Status",
                    interactive=False
                )

            # --------------------------------------
            # MEMORY DEBUG
            # --------------------------------------

            with gr.Accordion(
                "🔍 Session Tools & Memory Debug",
                open=False
            ):

                gr.Markdown(
                    "Inspect how much chat history the "
                    "memory optimizer retains."
                )

                debug_button = gr.Button(
                    "Inspect Memory Window"
                )

                debug_output = gr.Textbox(
                    label="Memory Debug",
                    lines=12,
                    interactive=False
                )

                gr.Markdown(
                    "**Delete Conversation History** "
                    "removes chat messages from the active "
                    "session. Memory Vault facts remain saved."
                )

                delete_history_button = gr.Button(
                    "🗑️ Delete Conversation History",
                    variant="stop"
                )

                delete_status = gr.Textbox(
                    label="Session Status",
                    interactive=False
                )

            # --------------------------------------
            # PREPROCESSING USAGE
            # --------------------------------------

            with gr.Accordion(
                "📊 Preprocessing Token Analytics",
                open=False
            ):

                gr.Markdown(
                    "Previously recorded token consumption "
                    "for memory extraction and summarization."
                )

                usage_button = gr.Button(
                    "Refresh Usage"
                )

                usage_table = gr.Dataframe(
                    label="Preprocessing Usage",
                    interactive=False
                )

            # --------------------------------------
            # CHAT EVENTS
            # --------------------------------------

            new_button.click(
                fn=start_new_chat,
                outputs=[
                    session_state,
                    session_label,
                    chat,
                    metrics_box,
                    memory_display,
                    debug_output,
                    delete_status,
                    memory_status
                ]
            )

            refresh_button.click(
                fn=refresh_sessions,
                outputs=session_dropdown
            )

            session_dropdown.change(
                fn=switch_session,
                inputs=session_dropdown,
                outputs=[
                    session_state,
                    session_label,
                    chat,
                    metrics_box,
                    memory_display,
                    debug_output,
                    delete_status,
                    memory_status
                ]
            )

            send_button.click(
                fn=send_message,
                inputs=[message, session_state],
                outputs=[
                    chat,
                    message,
                    metrics_box,
                    memory_display
                ]
            )

            message.submit(
                fn=send_message,
                inputs=[message, session_state],
                outputs=[
                    chat,
                    message,
                    metrics_box,
                    memory_display
                ]
            )

            view_memories_button.click(
                fn=display_memories,
                inputs=session_state,
                outputs=memory_display
            )

            remember_button.click(
                fn=remember_fact,
                inputs=[
                    session_state,
                    memory_key,
                    memory_value
                ],
                outputs=[
                    memory_status,
                    memory_display
                ]
            )

            forget_button.click(
                fn=forget_fact,
                inputs=[
                    session_state,
                    memory_key
                ],
                outputs=[
                    memory_status,
                    memory_display
                ]
            )

            debug_button.click(
                fn=debug_memory,
                inputs=session_state,
                outputs=debug_output
            )

            delete_history_button.click(
                fn=delete_chat_history,
                inputs=session_state,
                outputs=[
                    chat,
                    delete_status,
                    debug_output
                ]
            )

            usage_button.click(
                fn=show_preprocessing_usage,
                inputs=session_state,
                outputs=usage_table
            )

        # ==========================================
        # TAB 2 - MEMORY BATTLE
        # ==========================================

        with gr.Tab("⚔️ Memory Battle"):

            gr.Markdown(
                "### Five-Strategy Memory Comparison"
            )

            gr.Markdown(
                "Compare Full History, Rolling Window, "
                "Smart Memory Vault, Hybrid Summary + Vault, "
                "and Summary Only."
            )

            battle_session = gr.Textbox(
                label="Session ID",
                placeholder="Enter an existing session ID"
            )

            use_session_button = gr.Button(
                "Use Active Chat Session"
            )

            battle_question = gr.Textbox(
                label="Evaluation Question",
                placeholder="What is my favorite programming language?"
            )

            battle_expected = gr.Textbox(
                label="Expected Answer (Optional)",
                placeholder="Java"
            )

            run_battle_button = gr.Button(
                "⚔️ Run Memory Battle",
                variant="primary"
            )

            gr.Markdown(
                "Running a live battle makes Groq API calls. "
                "Strategies exceeding the configured token "
                "budget may be skipped."
            )

            battle_results = gr.Dataframe(
                label="Strategy Comparison",
                interactive=False
            )

            with gr.Row():

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

            use_session_button.click(
                fn=use_active_session,
                inputs=session_state,
                outputs=battle_session
            )

            run_battle_button.click(
                fn=run_battle,
                inputs=[
                    battle_session,
                    battle_question,
                    battle_expected
                ],
                outputs=[
                    battle_results,
                    token_chart,
                    latency_chart
                ]
            )

        # ==========================================
        # TAB 3 - BENCHMARK EXPLORER
        # ==========================================

        with gr.Tab("📊 Benchmark Explorer"):

            gr.Markdown(
                "### Historical Benchmark Results"
            )

            gr.Markdown(
                "Explore saved benchmark CSV files. "
                "This tab does not make Groq API calls."
            )

            saved_runs = available_runs()

            run_selector = gr.Dropdown(
                choices=saved_runs,
                value=(
                    saved_runs[0] if saved_runs else None
                ),
                label="Select Benchmark Run"
            )

            with gr.Row():

                refresh_runs_button = gr.Button(
                    "🔄 Refresh Runs"
                )

                explore_button = gr.Button(
                    "Load Benchmark",
                    variant="primary"
                )

            leaderboard_table = gr.Dataframe(
                label="Benchmark Summary",
                interactive=False
            )

            detailed_table = gr.Dataframe(
                label="Individual Test Results",
                interactive=False
            )

            refresh_runs_button.click(
                fn=refresh_benchmark_runs,
                outputs=run_selector
            )

            explore_button.click(
                fn=load_benchmark,
                inputs=run_selector,
                outputs=[
                    leaderboard_table,
                    detailed_table
                ]
            )

    # ==============================================
    # INITIAL PAGE LOAD
    # ==============================================

    demo.load(
        fn=lambda sid: (
            sid,
            display_memories(sid)
        ),
        inputs=session_state,
        outputs=[
            session_label,
            memory_display
        ]
    )


# ==================================================
# START APPLICATION
# ==================================================

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
