
import time
from dataclasses import dataclass


@dataclass
class RequestMetrics:
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


def invoke_with_metrics(chatbot, question, session_id, memories=""):
    start = time.perf_counter()

    response = chatbot.invoke(
        {
            "question": question,
            "memories": memories
        },
        config={"configurable": {"session_id": session_id}}
    )

    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.usage_metadata or {}

    metrics = RequestMetrics(
        latency_ms=round(latency_ms, 2),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens")
    )

    return response, metrics
