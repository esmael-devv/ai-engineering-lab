
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def available_runs():
    return [
        file.name
        for file in sorted(
            BASE_DIR.glob("benchmark_results_*.csv"),
            reverse=True
        )
    ]


def explore_run(filename):
    if not filename:
        return pd.DataFrame(), pd.DataFrame()

    # Restrict selection to known benchmark files
    if filename not in available_runs():
        raise ValueError("Unknown benchmark file.")

    df = pd.read_csv(BASE_DIR / filename)

    df["correct"] = (
        df["correct"]
        .astype("string")
        .str.lower()
        .map({"true": 1.0, "false": 0.0})
    )

    df["completed"] = df["status"] == "completed"

    summary = df.groupby("strategy").agg(
        attempts=("status", "count"),
        completed=("completed", "sum"),
        accuracy=("correct", "mean"),
        avg_input_tokens=("input_tokens", "mean"),
        median_latency_ms=("latency_ms", "median")
    ).reset_index()

    summary["accuracy"] *= 100
    summary["completion_rate"] = (
        summary["completed"] / summary["attempts"] * 100
    )

    summary.loc[
        summary["completed"] == 0, "accuracy"
    ] = float("nan")

    return summary.round(2), df
