"""JD-CV matching tool."""

from __future__ import annotations

from job_agent.rag import tokenize
from job_agent.schemas import MatchItem, Requirement, RetrievedChunk


def cv_match_tool(
    requirements: list[Requirement],
    evidence_chunks: list[RetrievedChunk],
) -> list[MatchItem]:
    """Match each requirement against retrieved CV/project evidence."""
    matches: list[MatchItem] = []
    for requirement in requirements:
        req_tokens = set(tokenize(requirement.name))
        relevant = [
            chunk
            for chunk in evidence_chunks
            if req_tokens & set(tokenize(chunk.text))
            or requirement.name.lower() in chunk.text.lower()
        ][:3]
        if relevant:
            top_score = max(chunk.score for chunk in relevant)
            level = "strong" if top_score >= 0.35 else "partial"
            gap = "" if level == "strong" else f"Add stronger evidence for {requirement.name}."
        else:
            level = "gap"
            gap = f"No direct CV/project evidence found for {requirement.name}."

        suggestion = (
            f"Prepare a concise project example for {requirement.name}."
            if level != "strong"
            else f"Use existing evidence to explain {requirement.name} with metrics."
        )
        matches.append(
            MatchItem(
                requirement=requirement.name,
                match_level=level,
                evidence=relevant,
                gap=gap,
                preparation_suggestion=suggestion,
            )
        )
    return matches
