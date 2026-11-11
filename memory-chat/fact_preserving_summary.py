
from memory_vault import get_memories
from summary_memory import get_summary


def build_fact_preserving_summary(session_id):
    saved_summary = get_summary(session_id)

    conversation_summary = (
        saved_summary[0]
        if saved_summary
        else "No conversation summary available."
    )

    memories = get_memories(session_id)

    if memories:
        facts = "\n".join(
            f"- {key}: {value}"
            for key, value in memories
        )
    else:
        facts = "No verified memories available."

    return f"""
IMPORTANT SAVED FACTS:
{facts}

CONVERSATION SUMMARY:
{conversation_summary}
""".strip()
