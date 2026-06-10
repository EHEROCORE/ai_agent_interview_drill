"""Explicit graph nodes and orchestrator for the job application research agent.

This module replaces the previous single-step workflow with an explicit, traceable
pipeline. Each node:

* reads/writes a structured :class:`GraphState`,
* records a :class:`NodeTrace` (status + latency + checkpoint flag),
* persists a checkpoint so an interrupted run can resume from the last good node.

The pipeline is deterministic by default (no LLM key required). It also implements a
bounded *agentic retrieval* loop: after the first match it checks evidence
sufficiency and, while requirement gaps remain, rewrites the query and retrieves
again up to ``MAX_RETRIEVAL_ROUNDS``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, TypedDict

from job_agent.guardrails import check_claim_grounding, classify_input, redact_sensitive
from job_agent.hybrid import HybridRetriever
from job_agent.llm import classify_intent, rewrite_query, select_retrievers
from job_agent.rag import HybridRAGIndex
from job_agent.rerank import rerank
from job_agent.schemas import (
    CVProfile,
    GuardrailResult,
    MatchItem,
    NodeTrace,
    PrepareRequest,
    Requirement,
    RetrievedChunk,
    ToolTrace,
)
from job_agent.session import SessionStore
from job_agent.tool_executor import execute
from job_agent.tools import (
    build_default_index,
    cv_match_tool,
    cv_parser_tool,
    interview_question_tool,
    jd_parser_tool,
    rag_retrieval_tool,
    report_writer_tool,
)

MAX_RETRIEVAL_ROUNDS = 2
RETRIEVAL_TOP_K = 10


def _jsonable(value: Any) -> Any:
    """JSON-serialize a tool result, expanding pydantic models to dicts."""
    from pydantic import BaseModel

    def _default(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return str(obj)

    return json.loads(json.dumps(value, default=_default))


class GraphState(TypedDict, total=False):
    """Mutable state carried through the job-agent graph."""

    request: PrepareRequest
    project_root: Path
    session_id: str
    request_id: str
    cv_text: str

    guardrail: GuardrailResult
    requirements: list[Requirement]
    cv_profile: CVProfile
    index: HybridRAGIndex
    retriever: Any
    citations: list[RetrievedChunk]
    matches: list[MatchItem]
    interview_questions: list[str]
    report: str
    report_path: str
    output_guardrail: GuardrailResult

    intent: str
    selected_retrievers: list[str]
    tool_trace: list[ToolTrace]
    node_trace: list[NodeTrace]
    retrieval_rounds: int
    cache_hits: int
    retrieval_latency_ms: float
    ttft_ms: float
    status: str
    failed_node: str


# --------------------------------------------------------------------------- #
# Tool tracing + caching helper
# --------------------------------------------------------------------------- #


def _trace_tool(
    state: GraphState,
    store: SessionStore | None,
    tool_name: str,
    args: dict[str, Any],
    fn: Callable[[], Any],
    *,
    use_cache: bool = False,
    timeout: float | None = None,
    retries: int = 0,
    fallback: Callable[[], Any] | None = None,
) -> Any:
    """Run a tool (with timeout/retry/fallback), trace it, and optionally cache it."""
    args_hash = (
        store.cache_key(tool_name, args) if store else json.dumps(args, sort_keys=True, default=str)
    )
    start = time.perf_counter()

    cached = store.get_cached(args_hash) if (use_cache and store) else None
    if cached is not None:
        latency = round((time.perf_counter() - start) * 1000, 3)
        trace = ToolTrace(
            tool_name=tool_name,
            args_hash=args_hash,
            status="ok",
            latency_ms=latency,
            observation={"cache": "hit", **cached},
        )
        state.setdefault("tool_trace", []).append(trace)
        state["cache_hits"] = state.get("cache_hits", 0) + 1
        if store:
            store.append_trace(state["session_id"], trace)
        return cached.get("__result__", cached)

    try:
        result = execute(fn, timeout=timeout, retries=retries, fallback=fallback)
    except Exception as exc:  # pragma: no cover - exercised via failure tests
        latency = round((time.perf_counter() - start) * 1000, 3)
        trace = ToolTrace(
            tool_name=tool_name,
            args_hash=args_hash,
            status="error",
            latency_ms=latency,
            observation={"error": str(exc)},
        )
        state.setdefault("tool_trace", []).append(trace)
        if store:
            store.append_trace(state["session_id"], trace)
        raise

    latency = round((time.perf_counter() - start) * 1000, 3)
    observation = result if isinstance(result, dict) else {"result": _jsonable(result)}
    trace = ToolTrace(
        tool_name=tool_name,
        args_hash=args_hash,
        status="ok",
        latency_ms=latency,
        observation=observation,
    )
    state.setdefault("tool_trace", []).append(trace)
    if store:
        store.append_trace(state["session_id"], trace)
        if use_cache:
            store.set_cached(args_hash, {"__result__": observation.get("result", observation)})
    return result


# --------------------------------------------------------------------------- #
# Node implementations (each returns a partial GraphState update)
# --------------------------------------------------------------------------- #


def entry_guard(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Classify the input into pass/sanitize/clarify/block."""
    request = state["request"]
    guardrail = classify_input(
        request.company, request.role, request.job_description, state.get("cv_text")
    )
    return {"guardrail": guardrail}


def parse_jd(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Extract structured requirements from the job description."""
    request = state["request"]
    requirements = _trace_tool(
        state, store, "jd_parser_tool",
        {"job_description": request.job_description},
        lambda: jd_parser_tool(request.job_description),
    )
    return {"requirements": requirements}


def parse_cv(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Extract a structured candidate profile from the CV text."""
    cv_text = state["cv_text"]
    cv_profile = _trace_tool(
        state, store, "cv_parser_tool",
        {"cv_text_hash": store.cache_key("cv_text", {"cv_text": cv_text}) if store else "n/a"},
        lambda: cv_parser_tool(cv_text),
    )
    return {"cv_profile": cv_profile}


def intent_route(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Classify intent and select logical retrievers (Agentic RAG routing)."""
    request = state["request"]
    intent = classify_intent(request, state.get("requirements", []))
    return {"intent": intent, "selected_retrievers": select_retrievers(intent)}


def _build_query(request: PrepareRequest, requirements: list[Requirement], extra: list[str] | None = None) -> str:
    return rewrite_query(request, requirements, extra)


def _ensure_index(state: GraphState) -> HybridRAGIndex:
    request = state["request"]
    index = state.get("index")
    if index is None:
        index = build_default_index(
            project_root=state["project_root"], cv_text=state["cv_text"],
            job_description=request.job_description, company=request.company,
            role=request.role, tenant_id=request.tenant_id,
        )
        state["index"] = index
    return index


def _run_retrieval(
    state: GraphState, store: SessionStore | None, query: str, extra_args: dict[str, Any] | None = None
) -> list[RetrievedChunk]:
    """Dispatch retrieval by ``retrieval_mode`` (hybrid vs deterministic), then rerank.

    ``hybrid`` uses the BM25+dense RRF retriever (built once and cached on state);
    ``deterministic`` uses the lexical index. Retrieval is cached via the tool cache.
    """
    request = state["request"]
    index = _ensure_index(state)
    user_roles = list(request.user_roles) + [f"tenant:{request.tenant_id}"]
    args = {"query": query, "top_k": RETRIEVAL_TOP_K, "tenant_id": request.tenant_id,
            "mode": request.retrieval_mode, **(extra_args or {})}

    if request.retrieval_mode in ("hybrid", "dense"):
        retriever = state.get("retriever")
        if retriever is None:
            retriever = HybridRetriever(index)
            state["retriever"] = retriever
        if request.retrieval_mode == "dense":
            tool_name = "dense_retrieval_tool"
            query_fn = retriever.query_dense
        else:
            tool_name = "hybrid_retrieval_tool"
            query_fn = retriever.query
        raw = _trace_tool(
            state, store, tool_name, args,
            lambda: query_fn(query, top_k=RETRIEVAL_TOP_K,
                             tenant_id=request.tenant_id, user_roles=user_roles),
            use_cache=True, retries=1, fallback=lambda: [],
        )
    else:
        raw = _trace_tool(
            state, store, "rag_retrieval_tool", args,
            lambda: rag_retrieval_tool(index, query, top_k=RETRIEVAL_TOP_K,
                                       tenant_id=request.tenant_id, user_roles=user_roles),
            use_cache=True, retries=1, fallback=lambda: [],
        )
    if isinstance(raw, list) and raw and not isinstance(raw[0], RetrievedChunk):
        raw = [RetrievedChunk.model_validate(item) for item in raw]
    return rerank(query, raw, top_k=RETRIEVAL_TOP_K)


def retrieve(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Build the tenant-scoped index and run the first retrieval round."""
    query = _build_query(state["request"], state.get("requirements", []))
    start = time.perf_counter()
    citations = _run_retrieval(state, store, query)
    return {
        "index": state.get("index"),
        "citations": citations,
        "retrieval_rounds": 1,
        "retrieval_latency_ms": round((time.perf_counter() - start) * 1000, 3),
    }


def match(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Match requirements against retrieved evidence."""
    requirements = state.get("requirements", [])
    citations = state.get("citations", [])
    matches = _trace_tool(
        state, store, "cv_match_tool",
        {"requirements": [item.name for item in requirements]},
        lambda: cv_match_tool(requirements, citations),
    )
    return {"matches": matches}


def evidence_check(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Agentic retrieval loop: while gaps remain, rewrite query and retrieve again."""
    request = state["request"]
    _ensure_index(state)  # rebuilt lazily when resuming mid-pipeline
    rounds = state.get("retrieval_rounds", 1)
    citations = list(state.get("citations", []))
    matches = state.get("matches", [])

    while rounds < MAX_RETRIEVAL_ROUNDS:
        gaps = [m.requirement for m in matches if m.match_level == "gap"]
        if not gaps:
            break
        rounds += 1
        query = _build_query(request, state.get("requirements", []), extra=gaps + ["project", "experience"])
        supplementary = _run_retrieval(state, store, query, extra_args={"round": rounds})
        seen = {c.citation_id for c in citations}
        citations.extend(c for c in supplementary if c.citation_id not in seen)
        matches = _trace_tool(
            state, store, "cv_match_tool",
            {"requirements": [item.name for item in state.get("requirements", [])], "round": rounds},
            lambda: cv_match_tool(state.get("requirements", []), citations),
        )

    return {"citations": citations, "matches": matches, "retrieval_rounds": rounds}


def generate_questions(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Generate targeted interview questions."""
    request = state["request"]
    questions = _trace_tool(
        state, store, "interview_question_tool",
        {"role": request.role, "target_interview_type": request.target_interview_type},
        lambda: interview_question_tool(
            request.role, state.get("requirements", []), state.get("matches", []),
            request.target_interview_type,
        ),
    )
    return {"interview_questions": questions}


def write_report(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Write the cited markdown preparation report."""
    request = state["request"]
    report = _trace_tool(
        state, store, "report_writer_tool",
        {"company": request.company, "role": request.role},
        lambda: report_writer_tool(
            request.company, request.role,
            [item.name for item in state.get("requirements", [])],
            state.get("matches", []), state.get("interview_questions", []),
            state.get("citations", []),
        ),
    )
    return {"report": redact_sensitive(report)}


def grounding_check(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Run claim-level grounding on the final report."""
    output_guardrail = check_claim_grounding(state.get("report", ""), state.get("citations", []))
    return {"output_guardrail": output_guardrail}


def persist(state: GraphState, store: SessionStore | None = None) -> GraphState:
    """Write the report file (final persistence handled by the runner)."""
    project_root = state["project_root"]
    reports_dir = project_root / "reports" / "job_agent"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{state['session_id']}.md"
    report_path.write_text(state.get("report", ""), encoding="utf-8")
    return {"report_path": str(report_path)}


# Ordered pipeline. ``entry_guard`` is handled specially (it can short-circuit).
PIPELINE: list[tuple[str, Callable[[GraphState, SessionStore | None], GraphState]]] = [
    ("entry_guard", entry_guard),
    ("parse_jd", parse_jd),
    ("parse_cv", parse_cv),
    ("intent_route", intent_route),
    ("retrieve", retrieve),
    ("match", match),
    ("evidence_check", evidence_check),
    ("generate_questions", generate_questions),
    ("write_report", write_report),
    ("grounding_check", grounding_check),
    ("persist", persist),
]

NODE_NAMES = [name for name, _ in PIPELINE]
