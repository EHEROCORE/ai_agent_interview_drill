"""FastAPI entrypoint for the job application research agent."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status

from job_agent.guardrails import check_prompt_injection
from job_agent.schemas import FeedbackRequest, PrepareRequest, PrepareResponse
from job_agent.session import SessionStore
from job_agent.workflow import prepare_application

RATE_WINDOW_SECONDS = 60
RATE_LIMIT = 20
REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(title="Job Application Research Agent", version="0.1.0")


def _store() -> SessionStore:
    return SessionStore(Path("reports") / "job_agent.sqlite")


def _check_auth(x_api_key: str | None = Header(default=None)) -> str:
    expected = os.environ.get("JOB_AGENT_API_KEY")
    identity = x_api_key or "anonymous"
    if expected and x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    now = time.time()
    bucket = REQUEST_LOG[identity]
    while bucket and now - bucket[0] > RATE_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
        )
    bucket.append(now)
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
