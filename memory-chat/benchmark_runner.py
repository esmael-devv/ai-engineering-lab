
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from memory_battle import compare_strategies
from datetime import datetime, timezone
import hashlib


BASE_DIR = Path(__file__).resolve().parent


def normalize_answer(value):
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"[\W_]+", "", text.casefold())


def benchmark_metadata():
    dataset_path = BASE_DIR / "benchmark_cases.json"

    dataset_hash = hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()[:12]

    return {
        "model": "openai/gpt-oss-20b",
        "window_budget": 1500,
        "dataset_hash": dataset_hash,
        "benchmark_version": "3.0",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }


def run_benchmark(repeats=1):
    cases = json.loads(
        (BASE_DIR / "benchmark_cases.json").read_text(
            encoding="utf-8"
        )
    )

    results = []
    metadata = benchmark_metadata()

    for case_number, case in enumerate(cases, start=1):
        for trial in range(1, repeats + 1):
            print(
                f"Test {case_number}/{len(cases)} "
                f"| Trial {trial}/{repeats}"
            )

            battle_results = compare_strategies(
                case["session_id"],
                case["question"]
            )

            for result in battle_results:
                expected = normalize_answer(
                    case["expected_answer"]
                )
                actual = normalize_answer(
                    result["answer"]
                )

                results.append({
                    "test_case": case_number,
                    "trial": trial,
                    "strategy": result["strategy"],
                    "correct": (
                            bool(expected) and expected in actual
                            if result.get("status", "completed") == "completed"
                            else None
                        ),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency_ms": result["latency_ms"],
                    "status": result.get("status", "completed"),
                    **metadata
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_benchmark()

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df.to_csv(
        BASE_DIR / f"benchmark_results_{timestamp}.csv",
        index=False
    )

    print("\nIndividual Results:")
    print(df.to_string(index=False))

    
    summary = df.groupby("strategy").agg(
        total_attempts=("status", "count"),
        completed=("status", lambda x: (x == "completed").sum()),
        skipped=("status", lambda x: (x != "completed").sum()),
        recall_accuracy=("correct", "mean"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median")
    )

    summary["recall_accuracy"] *= 100
    summary["completion_rate"] = (
        summary["completed"] / summary["total_attempts"] * 100
    )

    print("\nBenchmark Summary:")
    print(summary.round(2).to_string())
