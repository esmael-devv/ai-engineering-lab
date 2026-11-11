
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

import memory_extractor
import usage_tracker


def fake_response(content, input_tokens=100, output_tokens=30):
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    )


def test_valid_extraction():
    session_id = "extract-valid"

    response = fake_response(json.dumps({
        "memories": [
            {"key": "favorite_language", "value": "Java"}
        ]
    }))

    with patch.object(memory_extractor, "chain") as mock_chain:
        mock_chain.invoke.return_value = response

        memories = memory_extractor.extract_memories(
            session_id,
            "My favorite programming language is Java."
        )

    assert memories == [
        {"key": "favorite_language", "value": "Java"}
    ]

    records = usage_tracker.get_preprocessing_usage(session_id)

    assert len(records) == 1
    assert records[0]["input_tokens"] == 100
    assert records[0]["output_tokens"] == 30

    print("PASS: Valid JSON extraction and token tracking")


def test_invalid_json():
    session_id = "extract-invalid"

    response = fake_response(
        "This is not JSON",
        input_tokens=80,
        output_tokens=20
    )

    with patch.object(memory_extractor, "chain") as mock_chain:
        mock_chain.invoke.return_value = response

        memories = memory_extractor.extract_memories(
            session_id,
            "Hello!"
        )

    assert memories == []

    records = usage_tracker.get_preprocessing_usage(session_id)

    assert records[0]["input_tokens"] == 80
    assert records[0]["output_tokens"] == 20

    print("PASS: Invalid JSON rejected; usage recorded")


def test_malformed_memories():
    session_id = "extract-malformed"

    response = fake_response(json.dumps({
        "memories": [
            None,
            42,
            {"key": "project", "value": "Orion"},
            {"key": "", "value": "ignored"},
            {"key": "language", "value": 123}
        ]
    }))

    saved = []

    with patch.object(memory_extractor, "chain") as mock_chain:
        mock_chain.invoke.return_value = response

        with patch.object(
            memory_extractor,
            "save_memory",
            side_effect=lambda session, key, value:
                saved.append((session, key, value))
        ):
            memory_extractor.process_memory(
                session_id,
                "My project is Orion."
            )

    assert saved == [
        (session_id, "project", "Orion")
    ]

    print("PASS: Malformed memory entries safely ignored")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = Path(temp_dir) / "test_usage.db"

        with patch.object(usage_tracker, "DB_PATH", test_db):
            test_valid_extraction()
            test_invalid_json()
            test_malformed_memories()

    print("\nALL MEMORY EXTRACTION TESTS PASSED")
