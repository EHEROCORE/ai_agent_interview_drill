"""FastAPI entrypoint for the job application research agent."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from job_agent.cache import get_cache
from job_agent.evaluation import run_eval
from job_agent.guardrails import check_prompt_injection
from job_agent.ingestion import IngestManifest, ingest
from job_agent.schemas import FeedbackRequest, PrepareRequest, PrepareResponse
from job_agent.session import SessionStore
from job_agent.workflow import prepare_application, resume_application

RATE_WINDOW_SECONDS = 60
RATE_LIMIT = 20

app = FastAPI(title="Job Application Research Agent", version="0.2.0")


def _store() -> SessionStore:
    db_path = os.environ.get("JOB_AGENT_DB_PATH")
    return SessionStore(Path(db_path) if db_path else Path("reports") / "job_agent.sqlite")


def _check_auth(x_api_key: str | None = Header(default=None)) -> str:
    expected = os.environ.get("JOB_AGENT_API_KEY")
    identity = x_api_key or "anonymous"
    if expected and x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    hits = get_cache().hits_in_window(f"ratelimit:{identity}", RATE_WINDOW_SECONDS)
    if hits > RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
        )
    return identity


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/prepare", response_model=PrepareResponse)
def prepare(
    request: PrepareRequest,
    _identity: str = Depends(_check_auth),
) -> PrepareResponse:
    """Prepare a cited job application/interview report."""
    guardrail = check_prompt_injection(
        request.company,
        request.role,
        request.job_description,
        request.cv_text,
    )
    if guardrail.blocked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Request blocked by guardrail.", "guardrail": guardrail.model_dump()},
        )
    return prepare_application(request, store=_store())


@app.post("/prepare/stream")
def prepare_stream(
    request: PrepareRequest, _identity: str = Depends(_check_auth)
) -> StreamingResponse:
    """Stream a report skeleton first, then the full cited result as NDJSON."""

    def _events() -> Iterator[str]:
        skeleton = {
            "event": "skeleton",
            "company": request.company,
            "role": request.role,
            "sections": ["Role Requirements", "JD-CV Match Table", "Interview Questions", "Citations"],
        }
        yield json.dumps(skeleton, ensure_ascii=False) + "\n"
        guardrail = check_prompt_injection(
            request.company, request.role, request.job_description, request.cv_text
        )
        if guardrail.classification == "block":
            yield json.dumps({"event": "blocked", "guardrail": guardrail.model_dump()}, ensure_ascii=False) + "\n"
            return
        response = prepare_application(request, store=_store())
        yield json.dumps({"event": "result", "response": response.model_dump()}, ensure_ascii=False) + "\n"

    return StreamingResponse(_events(), media_type="application/x-ndjson")


@app.get("/sessions/{session_id}")
def get_session(session_id: str, _identity: str = Depends(_check_auth)) -> dict[str, object]:
    """Inspect a session and its tool traces."""
    store = _store()
    record = store.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session": record.model_dump(),
        "traces": [trace.model_dump() for trace in store.get_traces(session_id)],
    }


@app.get("/sessions/{session_id}/traces")
def get_session_traces(
    session_id: str, _identity: str = Depends(_check_auth)
) -> dict[str, object]:
    """Return tool traces and node checkpoints for observability."""
    store = _store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "tool_traces": [trace.model_dump() for trace in store.get_traces(session_id)],
        "completed_nodes": store.get_completed_nodes(session_id),
    }


@app.post("/sessions/{session_id}/resume", response_model=PrepareResponse)
def resume_session(
    session_id: str, _identity: str = Depends(_check_auth)
) -> PrepareResponse:
    """Resume an interrupted session from its last successful checkpoint."""
    store = _store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    try:
        return resume_application(session_id, store=store)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/sessions/{session_id}/feedback")
def save_feedback(
    session_id: str,
    feedback: FeedbackRequest,
    _identity: str = Depends(_check_auth),
) -> dict[str, str]:
    """Store user feedback for the data flywheel."""
    store = _store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    store.save_feedback(session_id, feedback)
    return {"status": "saved"}


@app.post("/knowledge/ingest")
def ingest_knowledge(
    manifest: IngestManifest, _identity: str = Depends(_check_auth)
) -> dict[str, object]:
    """Ingest documents into the persistent Qdrant knowledge collection."""
    location = os.environ.get("QDRANT_PATH", str(Path("reports") / "qdrant"))
    try:
        return ingest(manifest, location=location, root=Path.cwd())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/eval/run")
def trigger_eval(_identity: str = Depends(_check_auth)) -> dict[str, object]:
    """Run the offline evaluation suite and return summary metrics."""
    eval_dir = Path("data") / "eval"
    eval_file = eval_dir / "job_agent_eval_v2.jsonl"
    if not eval_file.exists():
        eval_file = eval_dir / "job_agent_eval.jsonl"
    if not eval_file.exists():
        raise HTTPException(status_code=404, detail="Eval dataset not found.")
    summary = run_eval(eval_file, Path("reports") / "eval")
    return {"status": "ok", "eval_file": str(eval_file), "summary": summary}
