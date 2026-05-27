"""Deterministic workflow powering the job application research agent."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from job_agent.guardrails import check_output_grounding, check_prompt_injection
from job_agent.metrics import Stopwatch
from job_agent.schemas import (
    GuardrailResult,
    Metrics,
    PrepareRequest,
    PrepareResponse,
    SessionRecord,
    ToolTrace,
)
from job_agent.session import SessionStore
from job_agent.tools import (
    build_default_index,
    cv_match_tool,
    cv_parser_tool,
    interview_question_tool,
    jd_parser_tool,
    rag_retrieval_tool,
    report_writer_tool,
)


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


def _trace(
    store: SessionStore,
    session_id: str,
    tool_name: str,
    args: dict[str, Any],
    fn: Callable[[], Any],
) -> tuple[Any, ToolTrace]:
    start = time.perf_counter()
    args_hash = store.cache_key(tool_name, args)
    try:
        result = fn()
        latency = round((time.perf_counter() - start) * 1000, 3)
        observation = (
            result
            if isinstance(result, dict)
            else {"result": json.loads(json.dumps(result, default=str))}
        )
        trace = ToolTrace(
            tool_name=tool_name,
            args_hash=args_hash,
            status="ok",
            latency_ms=latency,
            observation=observation,
        )
    except Exception as exc:  # pragma: no cover - exercised by integration failures
        latency = round((time.perf_counter() - start) * 1000, 3)
        trace = ToolTrace(
            tool_name=tool_name,
            args_hash=args_hash,
            status="error",
            latency_ms=latency,
            observation={"error": str(exc)},
        )
        store.append_trace(session_id, trace)
        raise
    store.append_trace(session_id, trace)
    return result, trace


def prepare_application(
    request: PrepareRequest,
    store: SessionStore | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
) -> PrepareResponse:
    """Run the end-to-end job application preparation workflow."""
    project_root = _project_root()
    store = store or SessionStore(project_root / "reports" / "job_agent.sqlite")
    session_id = session_id or str(uuid.uuid4())
    request_id = request_id or str(uuid.uuid4())
    timer = Stopwatch()
    traces: list[ToolTrace] = []

    cv_text = _load_cv_text(request, project_root)
    guardrail = check_prompt_injection(
        request.company,
        request.role,
        request.job_description,
        cv_text,
    )
    initial_status = "blocked" if guardrail.blocked else "running"
    store.save_session(
        SessionRecord(
            session_id=session_id,
            request_id=request_id,
            status=initial_status,
            company=request.company,
            role=request.role,
            current_step="input_guardrail",
            payload={"guardrail": guardrail.model_dump()},
        )
    )
    if guardrail.blocked:
        metrics = Metrics(
            ttft_ms=timer.ms(),
            retrieval_latency_ms=0,
            total_latency_ms=timer.ms(),
            tool_call_count=0,
            guardrail_pass=False,
            citation_count=0,
        )
        return PrepareResponse(
            session_id=session_id,
            request_id=request_id,
            company=request.company,
            role=request.role,
            report="Request blocked by input guardrail.",
            report_path="",
            citations=[],
            matches=[],
            interview_questions=[],
            guardrail=guardrail,
            metrics=metrics,
            tool_trace=[],
        )

    requirements, trace = _trace(
        store,
        session_id,
        "jd_parser_tool",
        {"job_description": request.job_description},
        lambda: jd_parser_tool(request.job_description),
    )
    traces.append(trace)

    cv_profile, trace = _trace(
        store,
        session_id,
        "cv_parser_tool",
        {"cv_text_hash": store.cache_key("cv_text", {"cv_text": cv_text})},
        lambda: cv_parser_tool(cv_text),
    )
    traces.append(trace)

    retrieval_timer = Stopwatch()
    index = build_default_index(
        project_root=project_root,
        cv_text=cv_text,
        job_description=request.job_description,
        company=request.company,
        role=request.role,
    )
    retrieval_query = " ".join(
        [request.company, request.role, request.target_interview_type]
        + [item.name for item in requirements]
    )
    citations, trace = _trace(
        store,
        session_id,
        "rag_retrieval_tool",
        {"query": retrieval_query, "top_k": 10},
        lambda: rag_retrieval_tool(index, retrieval_query, top_k=10),
    )
    traces.append(trace)
    retrieval_latency_ms = retrieval_timer.ms()
    ttft_ms = timer.ms()

    matches, trace = _trace(
        store,
        session_id,
        "cv_match_tool",
        {"requirements": [item.name for item in requirements]},
        lambda: cv_match_tool(requirements, citations),
    )
    traces.append(trace)

    questions, trace = _trace(
        store,
        session_id,
        "interview_question_tool",
        {"role": request.role, "target_interview_type": request.target_interview_type},
        lambda: interview_question_tool(
            request.role,
            requirements,
            matches,
            request.target_interview_type,
        ),
    )
    traces.append(trace)

    report, trace = _trace(
        store,
        session_id,
        "report_writer_tool",
        {"company": request.company, "role": request.role},
        lambda: report_writer_tool(
            request.company,
            request.role,
            [item.name for item in requirements],
            matches,
            questions,
            citations,
        ),
    )
    traces.append(trace)

    output_guardrail: GuardrailResult = check_output_grounding(report, citations)
    reports_dir = project_root / "reports" / "job_agent"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{session_id}.md"
    report_path.write_text(report, encoding="utf-8")

    metrics = Metrics(
        ttft_ms=ttft_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        total_latency_ms=timer.ms(),
        tool_call_count=len(traces),
        guardrail_pass=not output_guardrail.blocked and not guardrail.blocked,
        citation_count=len(citations),
    )

    payload = {
        "guardrail": guardrail.model_dump(),
        "output_guardrail": output_guardrail.model_dump(),
        "cv_profile": cv_profile.model_dump(),
        "metrics": metrics.model_dump(),
        "report_path": str(report_path),
    }
    store.save_session(
        SessionRecord(
            session_id=session_id,
            request_id=request_id,
            status="completed",
            company=request.company,
            role=request.role,
            current_step="completed",
            payload=payload,
        )
    )

    return PrepareResponse(
        session_id=session_id,
        request_id=request_id,
        company=request.company,
        role=request.role,
        report=report,
        report_path=str(report_path),
        citations=citations,
        matches=matches,
        interview_questions=questions,
        guardrail=guardrail,
        metrics=metrics,
        tool_trace=traces,
    )
