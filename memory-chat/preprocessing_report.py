
import sys
import pandas as pd

from usage_tracker import get_preprocessing_usage


def build_preprocessing_report(session_id):
    records = get_preprocessing_usage(session_id)

    columns = [
        "operation",
        "model",
        "calls",
        "input_tokens",
        "output_tokens",
        "total_tokens"
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)

    df["total_tokens"] = (
        df["input_tokens"] + df["output_tokens"]
    )

    return df[columns]


def print_preprocessing_report(session_id):
    df = build_preprocessing_report(session_id)

    print(f"\nSession: {session_id}")

    if df.empty:
        print("No preprocessing usage recorded.")
        return

    print("\nPreprocessing Operations:")
    print(df.to_string(index=False))

    print("\nTotals:")
    print("API calls:", int(df["calls"].sum()))
    print("Input tokens:", int(df["input_tokens"].sum()))
    print("Output tokens:", int(df["output_tokens"].sum()))
    print("Total tokens:", int(df["total_tokens"].sum()))


if __name__ == "__main__":
    session_id = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "usage-test"
    )

    print_preprocessing_report(session_id)
