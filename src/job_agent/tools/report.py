"""Report writer tool."""

from __future__ import annotations

from job_agent.schemas import MatchItem, RetrievedChunk


def _citation_text(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No direct citation"
    return ", ".join(f"[{chunk.citation_id}]" for chunk in chunks[:3])


def report_writer_tool(
    company: str,
    role: str,
    jd_summary: list[str],
    matches: list[MatchItem],
    interview_questions: list[str],
    citations: list[RetrievedChunk],
) -> str:
    """Generate a cited markdown preparation report."""
    lines = [
        f"# {company} {role} Interview Preparation Report",
        "",
        "## Role Requirements",
    ]
    for item in jd_summary:
        lines.append(f"- {item}")

    lines.extend(["", "## JD-CV Match Table", ""])
    lines.append("| Requirement | Match | Evidence | Preparation |")
    lines.append("|---|---|---|---|")
    for match in matches:
        lines.append(
            f"| {match.requirement} | {match.match_level} | {_citation_text(match.evidence)} | {match.preparation_suggestion} |"
        )

    lines.extend(["", "## Interview Questions", ""])
    for question in interview_questions:
        lines.append(f"- {question}")

    lines.extend(["", "## Citations", ""])
    for chunk in citations:
        source = chunk.source_path.replace("|", "/")
        excerpt = " ".join(chunk.text.split())[:180]
        lines.append(f"- [{chunk.citation_id}] {source}: {excerpt}")

    return "\n".join(lines)
