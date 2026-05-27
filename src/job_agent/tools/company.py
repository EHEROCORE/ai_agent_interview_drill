"""Company research tool."""

from __future__ import annotations

from job_agent.schemas import RetrievedChunk


def company_research_tool(company: str, role: str, evidence: list[RetrievedChunk]) -> list[str]:
    """Summarize company/role research from retrieved evidence."""
    facts: list[str] = []
    for chunk in evidence:
        if company.lower() in chunk.text.lower() or role.lower() in chunk.text.lower():
            facts.append(f"{chunk.text[:180]} [{chunk.citation_id}]")
    if not facts:
        facts.append(
            f"No grounded company-specific fact found for {company}; use web search or add company notes before claiming details."
        )
    return facts[:5]
