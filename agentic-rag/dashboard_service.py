
import time
from functools import lru_cache

from adaptive_main import run_with_model
from main import build_registry
from planner import create_planner_model
from evidence_aggregator import aggregate_evidence
from answer_generator import generate_grounded_answer
from evidence_guard import apply_evidence_guard


@lru_cache(maxsize=1)
def get_model():
    return create_planner_model().bind(
        response_format={"type": "json_object"}
    )


@lru_cache(maxsize=1)
def get_registry():
    return build_registry()


def run_dashboard_request(
    question,
    model=None,
    registry=None,
):
    """
    Reuse the existing adaptive agent.

    Return UI-ready answer, sources,
    execution history and metrics.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")

    model = model if model is not None else get_model()
    registry = registry if registry is not None else get_registry()

    start = time.perf_counter()

    execution = run_with_model(
        request=question.strip(),
        model=model,
        registry=registry,
        max_tool_calls=5,
    )

    evidence_package = aggregate_evidence(execution)

    answer = generate_grounded_answer(
        question=question.strip(),
        evidence_package=evidence_package,
        model=model,
    )

    answer = apply_evidence_guard(
        question=question.strip(),
        grounded_answer=answer,
        evidence_package=evidence_package,
    )

    elapsed = time.perf_counter() - start

    sources = [
        {
            "id": item["id"],
            "type": item["type"],
            "source": item["source"],
            "reference": item["reference"],
            "content": item["content"],
            "cited": item["id"] in answer["citations"],
        }
        for item in evidence_package["evidence"]
    ]

    trace = [
        {
            "attempt": index,
            "tool": entry["record"]["tool"],
            "instruction": entry["record"]["instruction"],
            "success": entry["record"]["success"],
            "observation": entry["observation"]["status"],
            "error": entry["record"].get("error", ""),
            "seconds": entry["record"].get(
                "elapsed_seconds"
            ),
        }
        for index, entry in enumerate(
            execution["history"], start=1
        )
    ]

    metrics = {
        "total_seconds": round(elapsed, 3),
        "tool_calls": execution["tool_calls"],
        "successful_calls": sum(
            item["success"] for item in trace
        ),
        "failed_calls": sum(
            not item["success"] for item in trace
        ),
        "evidence_count": len(sources),
        "cited_sources": len(answer["citations"]),
        "evidence_sufficient": answer["evidence_sufficient"],
        "verification_status": answer["verification_status"],
        "verification_warnings": answer[
            "verification_warnings"
        ],
        "stop_reason": execution["stop_reason"],
        "token_usage": "Not yet instrumented",
    }

    return {
        "answer": answer["answer"],
        "sources": sources,
        "trace": trace,
        "metrics": metrics,
    }
