
from unittest.mock import patch

from langchain_core.messages import AIMessage

import lightweight_summary
from usage_tracker import get_preprocessing_usage

import tempfile
from pathlib import Path
from unittest.mock import patch

import usage_tracker


def test_summary_usage_tracking():
    session_id = "usage-integration-test"

    fake_response = AIMessage(
        content="Favorite language: Java",
        usage_metadata={
            "input_tokens": 200,
            "output_tokens": 50,
            "total_tokens": 250
        }
    )

    with patch.object(
        lightweight_summary,
        "chain"
    ) as mock_chain:
        mock_chain.invoke.return_value = fake_response

        result = lightweight_summary.generate_lightweight_summary(
            session_id
        )

    assert "Java" in result

    records = get_preprocessing_usage(session_id)

    assert any(
        record["operation"] == "lightweight_summary"
        and record["input_tokens"] >= 200
        and record["output_tokens"] >= 50
        for record in records
    )

    print("PASS: Summary token usage persisted correctly")


def test_empty_summary_records_usage():
    session_id = "usage-empty-test"

    fake_response = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 150,
            "output_tokens": 300,
            "total_tokens": 450
        }
    )

    before = get_preprocessing_usage(session_id)
    before_calls = sum(r["calls"] for r in before)
    before_tokens = sum(
        r["input_tokens"] + r["output_tokens"]
        for r in before
    )

    with patch.object(lightweight_summary, "chain") as mock_chain:
        mock_chain.invoke.return_value = fake_response

        try:
            lightweight_summary.generate_lightweight_summary(
                session_id
            )
        except ValueError as error:
            assert "empty summary" in str(error).lower()
        else:
            raise AssertionError(
                "Expected ValueError for empty summary"
            )

    after = get_preprocessing_usage(session_id)

    after_calls = sum(r["calls"] for r in after)
    after_tokens = sum(
        r["input_tokens"] + r["output_tokens"]
        for r in after
    )

    assert after_calls - before_calls == 1
    assert after_tokens - before_tokens == 450

    print("PASS: Empty summary rejected; 450 tokens recorded")



if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = Path(temp_dir) / "test_usage.db"

        with patch.object(usage_tracker, "DB_PATH", test_db):
            test_summary_usage_tracking()
            test_empty_summary_records_usage()

        print("PASS: All usage tests used temporary SQLite storage")
