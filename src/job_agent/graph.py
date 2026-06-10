"""Explicit multi-node LangGraph for the job application research agent.

The same node functions that power the deterministic runner (:mod:`job_agent.nodes`)
are wired here into a real ``StateGraph`` so the pipeline is inspectable in LangGraph
Studio:

    entry_guard -> parse_jd -> parse_cv -> retrieve -> match -> evidence_check
                -> generate_questions -> write_report -> grounding_check -> persist

``entry_guard`` conditionally routes to ``__end__`` when the input is classified as a
hard block.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph

from job_agent.nodes import (
    PIPELINE,
    entry_guard,
    evidence_check,
    generate_questions,
    grounding_check,
    intent_route,
    match,
    parse_cv,
    parse_jd,
    persist,
    retrieve,
    write_report,
)
from job_agent.schemas import PrepareRequest


class JobGraphState(TypedDict, total=False):
    """LangGraph channel schema (trace channels use additive reducers)."""

    request: PrepareRequest
    project_root: Path
    session_id: str
    request_id: str
    cv_text: str
    guardrail: Any
    requirements: list[Any]
    cv_profile: Any
    index: Any
    citations: list[Any]
    matches: list[Any]
    interview_questions: list[str]
    report: str
    report_path: str
    output_guardrail: Any
    retrieval_rounds: int
    cache_hits: int
    retrieval_latency_ms: float
    tool_trace: Annotated[list[Any], operator.add]
    node_trace: Annotated[list[Any], operator.add]


def _wrap(fn: Any) -> Any:
    """Adapt a runner node (state, store) -> update for LangGraph (store omitted)."""

    def _node(state: JobGraphState) -> dict[str, Any]:
        return fn(dict(state), None)

    _node.__name__ = fn.__name__
    return _node


def _route_after_guard(state: JobGraphState) -> str:
    guardrail = state.get("guardrail")
    if guardrail is not None and getattr(guardrail, "classification", "pass") == "block":
        return "blocked"
    return "continue"


_NODE_FNS = {
    "entry_guard": entry_guard,
    "parse_jd": parse_jd,
    "parse_cv": parse_cv,
    "intent_route": intent_route,
    "retrieve": retrieve,
    "match": match,
    "evidence_check": evidence_check,
    "generate_questions": generate_questions,
    "write_report": write_report,
    "grounding_check": grounding_check,
    "persist": persist,
}

builder = StateGraph(JobGraphState)
for name, fn in _NODE_FNS.items():
    builder.add_node(name, _wrap(fn))

order = [name for name, _ in PIPELINE]
builder.add_edge("__start__", "entry_guard")
builder.add_conditional_edges(
    "entry_guard", _route_after_guard, {"continue": "parse_jd", "blocked": "__end__"}
)
for prev, nxt in zip(order[1:], order[2:]):
    builder.add_edge(prev, nxt)
builder.add_edge("persist", "__end__")

graph = builder.compile(name="Job Application Research Agent")
