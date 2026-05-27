"""RAG retrieval tools for the job application agent."""

from __future__ import annotations

from pathlib import Path

from job_agent.rag import HybridRAGIndex
from job_agent.schemas import RetrievedChunk


def build_default_index(
    project_root: Path,
    cv_text: str,
    job_description: str,
    company: str,
    role: str,
) -> HybridRAGIndex:
    """Build a local RAG index from workspace files and request documents."""
    index = HybridRAGIndex()
    index.add_document(cv_text, "request:cv", "cv", {"company": company, "role": role})
    index.add_document(
        job_description,
        "request:job_description",
        "job_description",
        {"company": company, "role": role},
    )

    workspace_root = project_root.parent
    for pattern, source_type in [
        ("interview_questions/*.md", "interview_question_bank"),
        ("resume_drafts/*.md", "project_note"),
        ("plans/*.md", "planning_note"),
    ]:
        index.ingest_paths(list(workspace_root.glob(pattern)), source_type)

    data_root = project_root / "data"
    for pattern, source_type in [
        ("company_notes/*.md", "company_note"),
        ("jd/*.md", "job_description"),
        ("cv/*.md", "cv"),
    ]:
        index.ingest_paths(list(data_root.glob(pattern)), source_type)

    return index


def rag_retrieval_tool(
    index: HybridRAGIndex,
    query: str,
    filters: dict[str, str] | None = None,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Retrieve cited evidence from the local RAG index."""
    return index.query(query=query, filters=filters, top_k=top_k)
