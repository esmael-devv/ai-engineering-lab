
import time


from pathlib import Path
from langchain_core.messages.utils import count_tokens_approximately
from fact_preserving_summary import build_fact_preserving_summary
from summary_memory import get_summary
from dotenv import load_dotenv


load_dotenv(
    Path(__file__).resolve().parents[2] / ".env"
)

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from sqlite_memory import SQLiteChatHistory
from memory_optimizer import optimize_history
from memory_vault import get_memories




model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the user's question using the available "
        "conversation history and saved memories. "
        "If the information is missing, say you don't know.\n"
        "Saved memories:\n{memories}"
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | model


def run_strategy(session_id, question, strategy):
    full_history = SQLiteChatHistory(session_id).messages

    if strategy == "Full History":
        history = full_history
        memories = ""

    elif strategy == "Summary Only":
        history = []

        saved_summary = get_summary(session_id)

        if not saved_summary or not saved_summary[0].strip():
            return {
                "strategy": strategy,
                "answer": "SKIPPED: No summary available.",
                "latency_ms": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "status": "skipped_missing_summary"
            }

        memories = saved_summary[0]

    elif strategy == "Rolling Window":
        history = optimize_history(
            {"history": full_history},
            max_tokens=1500
        )["history"]
        memories = ""

    elif strategy == "Smart Memory Vault":
        history = optimize_history(
            {"history": full_history},
            max_tokens=1500
        )["history"]

        saved = get_memories(session_id)
        memories = "\n".join(
            f"{key}: {value}" for key, value in saved
        )
    elif strategy == "Hybrid Summary + Vault":
        # Use saved facts and compressed conversation context
        history = []
        memories = build_fact_preserving_summary(session_id)    

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

        # Check estimated prompt size before sending to Groq
    formatted_prompt = prompt.invoke({
        "history": history,
        "memories": memories,
        "question": question
    })

    estimated_tokens = count_tokens_approximately(
        formatted_prompt.to_messages()
    )

    MAX_PROMPT_TOKENS = 6000

    if estimated_tokens > MAX_PROMPT_TOKENS:
        return {
            "strategy": strategy,
            "answer": (
                f"SKIPPED: Estimated {estimated_tokens} "
                f"prompt tokens exceed the {MAX_PROMPT_TOKENS} limit."
            ),
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "status": "skipped_token_budget"
        }

    start = time.perf_counter()

    response = chain.invoke({
        "history": history,
        "memories": memories,
        "question": question
    })

    latency_ms = round(
        (time.perf_counter() - start) * 1000, 2
    )

    usage = response.usage_metadata or {}

    return {
        "strategy": strategy,
        "answer": response.content,
        "latency_ms": latency_ms,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "status": "completed"
    }


def compare_strategies(session_id, question):
    strategies = [
        "Full History",
        "Rolling Window",
        "Smart Memory Vault",
        "Hybrid Summary + Vault",
        "Summary Only"
    ]

    return [
        run_strategy(session_id, question, strategy)
        for strategy in strategies
    ]
