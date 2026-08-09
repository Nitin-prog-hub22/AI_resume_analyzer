"""
database.py
-------------

Stores a lightweight HISTORY of past analyses (filename, target role,
match score, matched/missing skill lists, timestamp) — never the raw
resume text or file itself
Uses SQLite via Python's built-in sqlite3 module, so no extra
dependency or server is needed for local use or grading. 
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_filename TEXT NOT NULL,
    target_role TEXT NOT NULL,
    match_score REAL NOT NULL,
    matched_skills TEXT NOT NULL,
    missing_skills TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the analysis_history table if it doesn't already exist."""
    with _connect() as conn:
        conn.execute(_SCHEMA)


def save_analysis(
    resume_filename: str,
    target_role: str,
    match_score: float,
    matched_skills: list,
    missing_skills: list,
) -> int:
    """
    Save one analysis result to the history table.

    Returns:
        The new row's id.
    """
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analysis_history
                (resume_filename, target_role, match_score, matched_skills, missing_skills, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                resume_filename,
                target_role,
                match_score,
                json.dumps(matched_skills),
                json.dumps(missing_skills),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return cursor.lastrowid


def get_history(limit: int = 20) -> list:
    """Return the most recent `limit` analysis records, newest first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    history = []
    for row in rows:
        record = dict(row)
        record["matched_skills"] = json.loads(record["matched_skills"])
        record["missing_skills"] = json.loads(record["missing_skills"])
        history.append(record)
    return history


def clear_history() -> None:
    """Delete all saved history records (used by a 'Clear history' button)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM analysis_history")


if __name__ == "__main__":
    save_analysis("resume_a_data_analyst.docx", "Data Analyst", 36.0, ["python", "sql"], ["pandas", "tableau"])
    for record in get_history(5):
        print(record)
