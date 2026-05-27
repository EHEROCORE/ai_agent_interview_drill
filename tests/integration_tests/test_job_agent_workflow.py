"""Integration tests for the Job Application Research Agent."""

from pathlib import Path

from job_agent.entrypoint import app
from job_agent.schemas import PrepareRequest
from job_agent.session import SessionStore
from job_agent.workflow import prepare_application


def _request() -> PrepareRequest:
    return PrepareRequest(
        company="Tencent Yuanbao",
        role="AI Application Developer",
        job_description=(
            "Develop Agent systems with RAG, Prompt injection protection, "
            "FastAPI, evaluation harness, TTFT optimization and state recovery."
        ),
        cv_text=(
            "Skills: Python, RAG, FastAPI, prompt injection guardrails, "
            "evaluation harness, SQLite state recovery. Project: Job Application Agent."
        ),
        target_interview_type="AI Agent",
    )


def test_prepare_application_completes_full_workflow(tmp_path: Path) -> None:
    """The workflow should generate a report, citations, traces, and persisted state."""
    store = SessionStore(tmp_path / "job_agent.sqlite")
    response = prepare_application(_request(), store=store)
    assert response.report_path
    assert Path(response.report_path).exists()
    assert response.citations
    assert response.matches
    assert response.interview_questions
    assert response.metrics.tool_call_count >= 5
    assert store.get_session(response.session_id) is not None
    assert store.get_traces(response.session_id)


def test_interrupted_session_can_be_inspected_for_recovery(tmp_path: Path) -> None:
    """Completed session state should expose the latest checkpoint and traces."""
    store = SessionStore(tmp_path / "job_agent.sqlite")
    response = prepare_application(_request(), store=store)
    record = store.get_session(response.session_id)
    assert record is not None
    assert record.current_step == "completed"
    assert record.payload["report_path"] == response.report_path


def test_fastapi_routes_are_registered() -> None:
    """FastAPI app should expose production-style entrypoints."""
    routes = {route.path for route in app.routes}
    assert "/healthz" in routes
    assert "/prepare" in routes
    assert "/sessions/{session_id}" in routes
    assert "/sessions/{session_id}/feedback" in routes
