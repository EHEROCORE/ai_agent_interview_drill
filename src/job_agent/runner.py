"""Sequential orchestrator with per-node tracing, checkpointing and resume.

The runner executes :data:`job_agent.nodes.PIPELINE` in order. After every node it
writes a :class:`NodeTrace` and a checkpoint snapshot. If a node raises, the failure
is recorded and the session is marked ``failed`` so it can later be resumed from the
first not-yet-completed node via :func:`run_pipeline` with ``resume=True``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from job_agent.metrics import Stopwatch
from job_agent.nodes import PIPELINE, GraphState
from job_agent.schemas import (
    CVProfile,
    GuardrailResult,
    MatchItem,
    Metrics,
    NodeTrace,
    PrepareRequest,
    PrepareResponse,
    Requirement,
    RetrievedChunk,
    SessionRecord,
    ToolTrace,
)
from job_agent.session import SessionStore

# State keys that are JSON-serializable (the index/project_root are rebuilt instead).
_MODEL_LIST_KEYS = {
    "requirements": Requirement,
    "citations": RetrievedChunk,
    "matches": MatchItem,
    "tool_trace": ToolTrace,
    "node_trace": NodeTrace,
}
_MODEL_KEYS = {
    "request": PrepareRequest,
    "guardrail": GuardrailResult,
    "cv_profile": CVProfile,
    "output_guardrail": GuardrailResult,
}
_SCALAR_KEYS = {
    "cv_text", "session_id", "request_id", "interview_questions", "report",
    "report_path", "retrieval_rounds", "cache_hits", "retrieval_latency_ms",
    "ttft_ms", "status", "failed_node", "intent", "selected_retrievers",
}


def _serialize_state(state: GraphState) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, model in _MODEL_LIST_KEYS.items():
        if key in state:
            out[key] = [item.model_dump() for item in state[key]]
    for key in _MODEL_KEYS:
        if key in state and state[key] is not None:
            out[key] = state[key].model_dump()
    for key in _SCALAR_KEYS:
        if key in state:
            out[key] = state[key]
    return out


def _deserialize_state(data: dict[str, Any]) -> GraphState:
    state: GraphState = {}
    for key, model in _MODEL_LIST_KEYS.items():
        if key in data:
            state[key] = [model.model_validate(item) for item in data[key]]
    for key, model in _MODEL_KEYS.items():
        if key in data and data[key] is not None:
            state[key] = model.model_validate(data[key])
    for key in _SCALAR_KEYS:
        if key in data:
            state[key] = data[key]
    return state


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assemble_response(state: GraphState, metrics: Metrics) -> PrepareResponse:
    request = state["request"]
    guardrail = state.get("guardrail") or GuardrailResult(blocked=False, risk_score=0.0)
    return PrepareResponse(
        session_id=state["session_id"],
        request_id=state["request_id"],
        company=request.company,
        role=request.role,
        intent=state.get("intent", ""),
        selected_retrievers=state.get("selected_retrievers", []),
        report=state.get("report", ""),
        report_path=state.get("report_path", ""),
        citations=state.get("citations", []),
        matches=state.get("matches", []),
        interview_questions=state.get("interview_questions", []),
        guardrail=guardrail,
        output_guardrail=state.get("output_guardrail"),
        metrics=metrics,
        tool_trace=state.get("tool_trace", []),
        node_trace=state.get("node_trace", []),
    )


def _blocked_response(state: GraphState, timer: Stopwatch) -> PrepareResponse:
    metrics = Metrics(
        ttft_ms=timer.ms(), retrieval_latency_ms=0.0, total_latency_ms=timer.ms(),
        tool_call_count=0, guardrail_pass=False, citation_count=0,
        retrieval_rounds=0, cache_hits=0,
    )
    state["report"] = "Request blocked by input guardrail."
    return _assemble_response(state, metrics)


def run_pipeline(
    request: PrepareRequest | None,
    store: SessionStore,
    session_id: str,
    request_id: str | None = None,
    *,
    cv_text: str | None = None,
    resume: bool = False,
) -> PrepareResponse:
    """Execute the node pipeline end-to-end (or resume an interrupted run)."""
    timer = Stopwatch()
    project_root = _project_root()

    completed: list[str] = []
    if resume:
        checkpoint = store.get_latest_checkpoint(session_id)
        if checkpoint is None:
            raise ValueError("No checkpoint found for session; cannot resume.")
        state = _deserialize_state(checkpoint[2])
        state["project_root"] = project_root
        completed = store.get_completed_nodes(session_id)
        request = state["request"]
        request_id = state.get("request_id", request_id)
    else:
        if request is None:
            raise ValueError("request is required for a fresh run.")
        state = GraphState(
            request=request,
            project_root=project_root,
            session_id=session_id,
            request_id=request_id or session_id,
            cv_text=cv_text or request.cv_text or "",
            tool_trace=[],
            node_trace=[],
            cache_hits=0,
        )

    assert request is not None
    state["request_id"] = request_id or state.get("request_id", session_id)

    store.save_session(
        SessionRecord(
            session_id=session_id, request_id=state["request_id"], status="running",
            company=request.company, role=request.role,
            current_step="resume" if resume else "start", payload={},
        )
    )

    for index, (node_name, node_fn) in enumerate(PIPELINE):
        if node_name in completed:
            continue
        start = time.perf_counter()
        try:
            update = node_fn(state, store)
            state.update(update)
            latency = round((time.perf_counter() - start) * 1000, 3)

            # entry_guard may short-circuit the whole run.
            if node_name == "entry_guard" and state["guardrail"].classification == "block":
                state.setdefault("node_trace", []).append(
                    NodeTrace(node=node_name, status="blocked", latency_ms=latency, checkpoint=True)
                )
                store.save_checkpoint(session_id, node_name, "blocked", _serialize_state(state))
                store.save_session(
                    SessionRecord(
                        session_id=session_id, request_id=state["request_id"], status="blocked",
                        company=request.company, role=request.role,
                        current_step="input_guardrail",
                        payload={"guardrail": state["guardrail"].model_dump()},
                    )
                )
                return _blocked_response(state, timer)

            node_trace = NodeTrace(node=node_name, status="ok", latency_ms=latency, checkpoint=True)
            state.setdefault("node_trace", []).append(node_trace)
            store.save_checkpoint(session_id, node_name, "ok", _serialize_state(state))
            if "ttft_ms" not in state and node_name == "retrieve":
                state["ttft_ms"] = timer.ms()
        except Exception as exc:
            latency = round((time.perf_counter() - start) * 1000, 3)
            state.setdefault("node_trace", []).append(
                NodeTrace(node=node_name, status="error", latency_ms=latency,
                          detail={"error": str(exc)})
            )
            state["failed_node"] = node_name
            store.save_checkpoint(session_id, node_name, "error", _serialize_state(state))
            store.save_session(
                SessionRecord(
                    session_id=session_id, request_id=state["request_id"], status="failed",
                    company=request.company, role=request.role, current_step=node_name,
                    payload={"failed_node": node_name, "error": str(exc)},
                )
            )
            raise

    output_guardrail = state.get("output_guardrail")
    metrics = Metrics(
        ttft_ms=state.get("ttft_ms", timer.ms()),
        retrieval_latency_ms=state.get("retrieval_latency_ms", 0.0),
        total_latency_ms=timer.ms(),
        tool_call_count=len(state.get("tool_trace", [])),
        guardrail_pass=not state["guardrail"].blocked
        and not (output_guardrail.blocked if output_guardrail else False),
        citation_count=len(state.get("citations", [])),
        retrieval_rounds=state.get("retrieval_rounds", 1),
        cache_hits=state.get("cache_hits", 0),
    )

    payload = {
        "guardrail": state["guardrail"].model_dump(),
        "output_guardrail": output_guardrail.model_dump() if output_guardrail else None,
        "cv_profile": state["cv_profile"].model_dump() if "cv_profile" in state else None,
        "metrics": metrics.model_dump(),
        "report_path": state.get("report_path", ""),
    }
    store.save_session(
        SessionRecord(
            session_id=session_id, request_id=state["request_id"], status="completed",
            company=request.company, role=request.role, current_step="completed",
            payload=payload,
        )
    )
    return _assemble_response(state, metrics)
