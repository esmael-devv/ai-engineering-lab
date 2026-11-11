
import tempfile
from pathlib import Path
from unittest.mock import patch

import memory_vault


def test_exact_key_update():
    session_id = "conflict-test"

    memory_vault.save_memory(
        session_id,
        "favorite_language",
        "Java"
    )

    initial = dict(memory_vault.get_memories(session_id))

    assert initial["favorite_language"] == "Java"

    memory_vault.save_memory(
        session_id,
        "favorite_language",
        "Python"
    )

    updated = dict(memory_vault.get_memories(session_id))

    assert updated["favorite_language"] == "Python"

    assert len(updated) == 1

    print("PASS: Java correctly updated to Python")
    print("PASS: No duplicate memory created")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:

        test_db = Path(temp_dir) / "test_memory.db"

        with patch.object(memory_vault, "DB_PATH", test_db):
            test_exact_key_update()

    print("PASS: Test database cleaned up successfully")
