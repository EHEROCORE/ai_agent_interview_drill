"""Interview question generation tool."""

from __future__ import annotations

from job_agent.schemas import MatchItem, Requirement


def interview_question_tool(
    role: str,
    requirements: list[Requirement],
    matches: list[MatchItem],
    target_interview_type: str,
) -> list[str]:
    """Generate targeted interview questions from JD requirements and gaps."""
    questions: list[str] = [
        f"Give a 90-second introduction for a {role} interview, emphasizing the most relevant project evidence.",
        f"How would you design an end-to-end {target_interview_type} system for this role?",
    ]

    for requirement in requirements[:8]:
        questions.append(f"Explain your practical experience with {requirement.name}.")

    for match in matches:
        if match.match_level == "gap":
            questions.append(
                f"If asked about {match.requirement}, how would you explain your learning plan and current gap?"
            )

    return list(dict.fromkeys(questions))[:12]
