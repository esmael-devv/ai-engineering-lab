
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "memory.db"


def initialize_summary_storage():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                summarized_message_count INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_summary(session_id, summary, message_count):
    initialize_summary_storage()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO conversation_summaries
                (session_id, summary, summarized_message_count)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET
                summary = excluded.summary,
                summarized_message_count =
                    excluded.summarized_message_count,
                updated_at = CURRENT_TIMESTAMP
        """, (session_id, summary, message_count))


def get_summary(session_id):
    initialize_summary_storage()

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT summary, summarized_message_count
            FROM conversation_summaries
            WHERE session_id = ?
        """, (session_id,)).fetchone()

    return row if row else None
