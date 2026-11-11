
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / "memory.db"


@contextmanager
def db_connection():
    conn = sqlite3.connect(DB_PATH)

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_memory_vault():
    with db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, memory_key)
            )
        """)


def save_memory(session_id, key, value):
    initialize_memory_vault()

    with db_connection() as conn:
        conn.execute("""
            INSERT INTO memories (
                session_id,
                memory_key,
                memory_value
            )
            VALUES (?, ?, ?)
            ON CONFLICT(session_id, memory_key)
            DO UPDATE SET
                memory_value = excluded.memory_value,
                updated_at = CURRENT_TIMESTAMP
        """, (session_id, key, value))


def get_memories(session_id):
    initialize_memory_vault()

    with db_connection() as conn:
        return conn.execute("""
            SELECT memory_key, memory_value
            FROM memories
            WHERE session_id = ?
            ORDER BY memory_key
        """, (session_id,)).fetchall()


def delete_memory(session_id, key):
    initialize_memory_vault()

    with db_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM memories
            WHERE session_id = ?
              AND memory_key = ?
        """, (session_id, key))

        return cursor.rowcount > 0
