"""Public workflow entrypoint for the job application research agent.

Orchestration now lives in :mod:`job_agent.nodes` (explicit graph nodes) and
:mod:`job_agent.runner` (sequential executor with tracing, checkpointing, resume).
This module keeps the stable ``prepare_application`` contract and the CV-source
loading / path-sandbox validation used by the API and tests.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from job_agent.runner import run_pipeline
from job_agent.schemas import PrepareRequest, PrepareResponse
from job_agent.session import SessionStore


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_cv_text(request: PrepareRequest, project_root: Path) -> str:
    if request.cv_text:
        return request.cv_text
    if not request.cv_file_path:
        raise ValueError("No CV source supplied.")
    path = Path(request.cv_file_path)
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    allowed_roots = [project_root.resolve(), project_root.parent.resolve()]
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        raise ValueError("CV file path is outside the allowed workspace.")
    if resolved.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Only .txt and .md CV files are supported in the demo API.")
    if resolved.stat().st_size > 2_000_000:
        raise ValueError("CV file is too large for the demo API.")
    return resolved.read_text(encoding="utf-8", errors="ignore")


def prepare_application(
    request: PrepareRequest,
    store: SessionStore | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
) -> PrepareResponse:
    """Run the end-to-end job application preparation pipeline."""
    project_root = _project_root()
    store = store or SessionStore(project_root / "reports" / "job_agent.sqlite")
    session_id = session_id or str(uuid.uuid4())
    request_id = request_id or str(uuid.uuid4())
    cv_text = _load_cv_text(request, project_root)
    return run_pipeline(
        request, store=store, session_id=session_id, request_id=request_id, cv_text=cv_text
    )


def resume_application(session_id: str, store: SessionStore) -> PrepareResponse:
    """Resume an interrupted session from its last successful checkpoint."""
    return run_pipeline(None, store=store, session_id=session_id, resume=True)
