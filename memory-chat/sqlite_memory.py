
import sqlite3
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_from_dict, messages_to_dict

import json


DB_PATH = Path(__file__).parent / "memory.db"


class SQLiteChatHistory(BaseChatMessageHistory):

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(DB_PATH)

    def _initialize_database(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            """)

    @property
    def messages(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT message FROM messages
                WHERE session_id = ?
                ORDER BY id
                """,
                (self.session_id,)
            ).fetchall()

        return messages_from_dict(
            [json.loads(row[0]) for row in rows]
        )

    def add_messages(self, messages):
        serialized = messages_to_dict(messages)

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO messages (session_id, message)
                VALUES (?, ?)
                """,
                [
                    (self.session_id, json.dumps(message))
                    for message in serialized
                ]
            )

    def clear(self):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (self.session_id,)
            )


def list_sessions():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT session_id, COUNT(*)
            FROM messages
            GROUP BY session_id
            ORDER BY session_id
            """
        ).fetchall()

    return rows
