
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / "memory.db"

@contextmanager
def db_connection():
    connection = sqlite3.connect(DB_PATH)

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_usage_table():
    with db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS preprocessing_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def record_preprocessing_usage(
    session_id,
    operation,
    model,
    input_tokens,
    output_tokens
):
    initialize_usage_table()

    with db_connection() as connection:
        connection.execute("""
            INSERT INTO preprocessing_usage (
                session_id,
                operation,
                model,
                input_tokens,
                output_tokens,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            operation,
            model,
            input_tokens,
            output_tokens,
            datetime.now(timezone.utc).isoformat()
        ))


def get_preprocessing_usage(session_id):
    initialize_usage_table()

    with db_connection() as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT
                operation,
                model,
                COUNT(*) AS calls,
                SUM(input_tokens) AS input_tokens,
                SUM(output_tokens) AS output_tokens
            FROM preprocessing_usage
            WHERE session_id = ?
            GROUP BY operation, model
            ORDER BY operation
        """, (session_id,)).fetchall()

    return [dict(row) for row in rows]
