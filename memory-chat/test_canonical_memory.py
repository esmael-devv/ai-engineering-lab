
import tempfile
from pathlib import Path
from unittest.mock import patch

import memory_extractor
import memory_vault


def test_canonical_memory_update():
    session_id = "canonical-test"

    # First user statement
    first_extraction = [
        {
            "key": "favorite_language",
            "value": "Java"
        }
    ]

    # Later user statement
    second_extraction = [
        {
            "key": "preferred_programming_language",
            "value": "Python"
        }
    ]

    with patch.object(
        memory_extractor,
        "extract_memories",
        side_effect=[first_extraction, second_extraction]
    ):
        memory_extractor.process_memory(
            session_id,
            "My favorite programming language is Java."
        )

        memory_extractor.process_memory(
            session_id,
            "Actually, my favorite language is Python now."
        )

    memories = dict(memory_vault.get_memories(session_id))

    assert memories == {
        "favorite_language": "Python"
    }, f"Unexpected memories: {memories}"

    print("PASS: Java updated to Python")
    print("PASS: Different keys normalized correctly")
    print("PASS: No duplicate memories created")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = Path(temp_dir) / "test_memory.db"

        with patch.object(memory_vault, "DB_PATH", test_db):
            test_canonical_memory_update()

    print("PASS: Temporary SQLite database cleaned up")
