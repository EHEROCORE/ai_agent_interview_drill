"""Job description parser tool."""

from __future__ import annotations

import re

from job_agent.schemas import Requirement

KEYWORDS: dict[str, tuple[str, str]] = {
    "rag": ("RAG", "rag"),
    "retrieval": ("retrieval", "rag"),
    "agent": ("AI Agent", "agent"),
    "react": ("ReAct", "agent"),
    "function call": ("Function Calling", "agent"),
    "tool": ("Tool Use", "agent"),
    "langgraph": ("LangGraph", "agent"),
    "prompt": ("Prompt Engineering", "agent"),
    "fastapi": ("FastAPI", "backend"),
    "api": ("API integration", "backend"),
    "mysql": ("MySQL", "backend"),
    "python": ("Python", "technical"),
    "pytorch": ("PyTorch", "technical"),
    "transformers": ("Hugging Face Transformers", "technical"),
    "evaluation": ("Evaluation", "technical"),
    "safety": ("Safety / Guardrails", "agent"),
    "communication": ("Communication", "soft_skill"),
}


def jd_parser_tool(job_description: str) -> list[Requirement]:
    """Extract role requirements from a job description."""
    normalized = job_description.lower()
    lines = [line.strip() for line in re.split(r"[\n.;]", job_description) if line.strip()]
    requirements: list[Requirement] = []
    seen: set[str] = set()

    for keyword, (name, category) in KEYWORDS.items():
        if keyword in normalized and name.lower() not in seen:
            evidence = next((line for line in lines if keyword in line.lower()), keyword)
            requirements.append(
                Requirement(name=name, category=category, evidence=evidence[:240])
            )
            seen.add(name.lower())

    if not requirements:
        for line in lines[:8]:
            if len(line) > 12:
                requirements.append(
                    Requirement(name=line[:80], category="technical", evidence=line[:240])
                )

    return requirements[:12]
