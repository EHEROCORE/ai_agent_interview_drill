"""CV parser tool."""

from __future__ import annotations

import re

from job_agent.schemas import CVProfile


def _split_items(text: str) -> list[str]:
    return [
        item.strip(" \t-•")
        for item in re.split(r"[,;\n|]", text)
        if item.strip(" \t-•")
    ]


def cv_parser_tool(cv_text: str) -> CVProfile:
    """Extract a lightweight structured profile from CV text."""
    skills: list[str] = []
    projects: list[str] = []
    experience: list[str] = []
    education: list[str] = []

    for line in cv_text.splitlines():
        clean = line.strip()
        lower = clean.lower()
        if not clean:
            continue
        if "skill" in lower or "programming" in lower or "machine learning" in lower:
            skills.extend(_split_items(clean.split(":", 1)[-1]))
        elif "project" in lower or "system" in lower or "agent" in lower:
            projects.append(clean[:240])
        elif "intern" in lower or "experience" in lower or "worked" in lower:
            experience.append(clean[:240])
        elif "university" in lower or "msc" in lower or "beng" in lower:
            education.append(clean[:240])

    if not skills:
        skills = _split_items(cv_text)[:30]

    return CVProfile(
        skills=list(dict.fromkeys(skills))[:40],
        projects=list(dict.fromkeys(projects))[:20],
        experience=list(dict.fromkeys(experience))[:20],
        education=list(dict.fromkeys(education))[:10],
    )
