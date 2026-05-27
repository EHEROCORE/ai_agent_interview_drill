"""LangGraph state for the job application research agent."""

from __future__ import annotations

from typing import Any, TypedDict

from job_agent.schemas import PrepareRequest, PrepareResponse


class JobAgentState(TypedDict, total=False):
    """State carried through the deterministic job-agent graph."""

    request: PrepareRequest
    response: PrepareResponse
    error: str
    artifacts: dict[str, Any]
