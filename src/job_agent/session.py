"""SQLite-backed session, trace, cache, and feedback storage."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from job_agent.schemas import FeedbackRequest, SessionRecord, ToolTrace


class SessionStore:
    """Small SQLite store for portfolio-grade state recovery."""

    def __init__(self, db_path: Path | str = "reports/job_agent.sqlite") -> None:
        """Initialize and migrate the database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    company TEXT NOT NULL,
                    role TEXT NOT NULL,
                    current_step TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_cache (
                    args_hash TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    feedback_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_session(self, record: SessionRecord) -> None:
        """Upsert a session state record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, request_id, status, company, role, current_step, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status=excluded.status,
                    current_step=excluded.current_step,
                    payload_json=excluded.payload_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    record.session_id,
                    record.request_id,
                    record.status,
                    record.company,
                    record.role,
                    record.current_step,
                    json.dumps(record.payload, ensure_ascii=False),
                ),
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Load a session record."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, request_id, status, company, role, current_step, payload_json
                FROM sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            session_id=row[0],
            request_id=row[1],
            status=row[2],
            company=row[3],
            role=row[4],
            current_step=row[5],
            payload=json.loads(row[6]),
        )

    def append_trace(self, session_id: str, trace: ToolTrace) -> None:
        """Append a tool trace."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO traces (session_id, trace_json) VALUES (?, ?)",
                (session_id, trace.model_dump_json()),
            )
        trace_dir = self.db_path.parent / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        with (trace_dir / f"{session_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(trace.model_dump_json() + "\n")

    def get_traces(self, session_id: str) -> list[ToolTrace]:
        """Load traces for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT trace_json FROM traces WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [ToolTrace.model_validate_json(row[0]) for row in rows]

    def cache_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Return a stable cache key for a tool invocation."""
        payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()

    def get_cached(self, args_hash: str) -> dict[str, Any] | None:
        """Load cached tool result."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM tool_cache WHERE args_hash = ?",
                (args_hash,),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def set_cached(self, args_hash: str, result: dict[str, Any]) -> None:
        """Store cached tool result."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tool_cache (args_hash, result_json)
                VALUES (?, ?)
                """,
                (args_hash, json.dumps(result, ensure_ascii=False)),
            )

    def save_feedback(self, session_id: str, feedback: FeedbackRequest) -> None:
        """Save user feedback."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (session_id, feedback_json) VALUES (?, ?)",
                (session_id, feedback.model_dump_json()),
            )
        feedback_dir = self.db_path.parent / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback_path = feedback_dir / "user_feedback.csv"
        exists = feedback_path.exists()
        with feedback_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["session_id", "rating", "comment", "failure_tags"],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "session_id": session_id,
                    "rating": feedback.rating,
                    "comment": feedback.comment,
                    "failure_tags": "|".join(feedback.failure_tags),
                }
            )
