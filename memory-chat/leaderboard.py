
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def load_benchmark_results():
    files = sorted(BASE_DIR.glob("benchmark_results_*.csv"))

    if not files:
        return pd.DataFrame()

    frames = []

    for file in files:
        df = pd.read_csv(file)

        required = {
            "strategy", "correct", "input_tokens",
            "latency_ms", "status"
        }

        if not required.issubset(df.columns):
            continue

        df["source_file"] = file.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_leaderboard():
    df = load_benchmark_results()

    if df.empty:
        return pd.DataFrame()

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

    summary["completion_rate"] = (
        summary["completed"] / summary["attempts"] * 100
    )

    summary["accuracy"] = summary["accuracy"] * 100
    # Accuracy is undefined when no requests completed.
    summary.loc[
        summary["completed"] == 0,
        "accuracy"
    ] = float("nan")

    return summary.sort_values(
    ["completion_rate", "accuracy"],
    ascending=[False, False],
    na_position="last"
    )


if __name__ == "__main__":
    leaderboard = build_leaderboard()

    if leaderboard.empty:
        print("No compatible benchmark results found.")
    else:
        print(leaderboard.round(2).to_string(index=False))
