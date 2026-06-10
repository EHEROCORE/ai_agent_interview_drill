"""Integration tests for failure recovery, resume, and new API routes."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_agent import nodes
from job_agent.entrypoint import app
from job_agent.schemas import PrepareRequest
from job_agent.session import SessionStore
from job_agent.workflow import prepare_application, resume_application


def _request() -> PrepareRequest:
    return PrepareRequest(
        company="Tencent Yuanbao",
        role="AI Application Developer",
        job_description=(
            "Develop Agent systems with RAG, FastAPI, evaluation harness and state recovery."
        ),
        cv_text="Skills: Python, RAG, FastAPI, evaluation harness. Project: Job Application Agent.",
        target_interview_type="AI Agent",
    )


def test_new_routes_are_registered() -> None:
    """FastAPI app should expose resume, traces, and eval routes."""
    routes = {route.path for route in app.routes}
    assert "/sessions/{session_id}/resume" in routes
    assert "/sessions/{session_id}/traces" in routes
    assert "/eval/run" in routes


def test_failed_node_then_resume_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A node failure should be recoverable: resume continues from the failed node."""
    store = SessionStore(tmp_path / "s.sqlite")
    calls = {"n": 0}
    real = nodes.interview_question_tool

    def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient interview tool failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(nodes, "interview_question_tool", flaky)

    with pytest.raises(RuntimeError):
        prepare_application(_request(), store=store, session_id="sess-1")

    record = store.get_session("sess-1")
    assert record is not None
    assert record.status == "failed"
    assert record.current_step == "generate_questions"
    # generate_questions should NOT be among completed checkpoints yet.
    assert "generate_questions" not in store.get_completed_nodes("sess-1")
    assert "retrieve" in store.get_completed_nodes("sess-1")

    # Resume: the flaky tool now succeeds and the run completes.
    resumed = resume_application("sess-1", store=store)
    assert resumed.report
    assert Path(resumed.report_path).exists()
    assert store.get_session("sess-1").status == "completed"
