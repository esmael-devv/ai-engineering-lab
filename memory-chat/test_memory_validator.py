
from unittest.mock import patch

import memory_extractor


def test_process_memory_validation():
    session_id = "validation-test"

    extracted_memories = [
        {"key": "favorite_language", "value": "Java"},
        {"key": "api_key", "value": "fake-secret"},
        {"key": "project_name", "value": "Orion"},
        {"key": "password", "value": "fake-password"},
        {"key": "preferred_cloud", "value": "AWS"},
        {"key": "invalid-key", "value": "Rejected"},
        {"key": "", "value": "Rejected"},
        None,
        42
    ]

    with patch.object(
        memory_extractor,
        "extract_memories",
        return_value=extracted_memories
    ):
        with patch.object(
            memory_extractor,
            "save_memory"
        ) as mock_save:

            memory_extractor.process_memory(
                session_id,
                "Synthetic test message"
            )

    actual = [
        call.args
        for call in mock_save.call_args_list
    ]

    expected = [
        ("validation-test", "favorite_language", "Java"),
        ("validation-test", "project_name", "Orion"),
        ("validation-test", "preferred_cloud", "AWS")
    ]

    assert actual == expected, (
        f"Unexpected saved memories: {actual}"
    )

    print("PASS: Only three valid memories were saved")
    print("PASS: Sensitive keys and invalid entries rejected")
    print("PASS: No Groq API calls required")


if __name__ == "__main__":
    test_process_memory_validation()
