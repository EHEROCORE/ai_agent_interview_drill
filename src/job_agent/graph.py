"""LangGraph workflow for the job application research agent."""

from __future__ import annotations

from langgraph.graph import StateGraph

from job_agent.state import JobAgentState
from job_agent.workflow import prepare_application


def run_prepare(state: JobAgentState) -> JobAgentState:
    """Run the deterministic preparation workflow inside LangGraph."""
    request = state["request"]
    response = prepare_application(request)
    return {"response": response, "artifacts": {"report_path": response.report_path}}


builder = StateGraph(JobAgentState)
builder.add_node("prepare_application", run_prepare)
builder.add_edge("__start__", "prepare_application")
builder.add_edge("prepare_application", "__end__")

graph = builder.compile(name="Job Application Research Agent")
