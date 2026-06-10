"""Intent routing, retriever selection, and query rewrite.

Deterministic by default so the agent runs and is testable without any API key. When
``llm_mode="auto"`` and a chat model is configured, the same functions can defer to an
LLM; any failure degrades back to the rule-based path.
"""

from __future__ import annotations

import os

from job_agent.rag import tokenize
from job_agent.schemas import PrepareRequest, Requirement

# Logical retrievers available to the agent.
RETRIEVERS = ["interview_question", "technical_note", "resume_project", "plan", "company", "memory"]

INTENTS = [
    "interview_prep",
    "project_deepdive",
    "concept_qa",
    "jd_cv_match",
    "evidence_lookup",
    "company_prep",
    "multi_source",
    "no_answer",
]

# Intent -> the retrievers that intent should consult.
INTENT_RETRIEVERS: dict[str, list[str]] = {
    "interview_prep": ["interview_question", "technical_note", "resume_project"],
    "project_deepdive": ["resume_project", "technical_note"],
    "concept_qa": ["technical_note", "interview_question"],
    "jd_cv_match": ["resume_project", "technical_note", "company"],
    "evidence_lookup": ["resume_project", "memory"],
    "company_prep": ["company", "plan"],
    "multi_source": ["interview_question", "technical_note", "resume_project", "company"],
    "no_answer": ["memory"],
}

_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("company_prep", ("company", "culture", "team", "mission", "about the company")),
    ("concept_qa", ("what is", "explain", "difference between", "how does", "define")),
    ("project_deepdive", ("project", "deep dive", "your experience building", "walk me through")),
    ("evidence_lookup", ("evidence", "where in your cv", "prove", "show me")),
    ("jd_cv_match", ("match", "fit", "requirement", "required", "must have", "gap analysis")),
    ("interview_prep", ("interview", "prepare", "questions", "mock")),
]


def classify_intent(request: PrepareRequest, requirements: list[Requirement] | None = None) -> str:
    """Classify the request intent (deterministic; LLM-assisted when enabled)."""
    if _llm_enabled(request):  # pragma: no cover - requires configured model
        intent = _llm_classify_intent(request)
        if intent in INTENTS:
            return intent

    text = f"{request.role} {request.target_interview_type} {request.job_description}".lower()
    for intent, keywords in _INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return intent
    # Default for the end-to-end preparation task.
    return "interview_prep"


def select_retrievers(intent: str) -> list[str]:
    """Return the logical retrievers for an intent."""
    return INTENT_RETRIEVERS.get(intent, ["technical_note", "resume_project"])


def rewrite_query(request: PrepareRequest, requirements: list[Requirement], extra: list[str] | None = None) -> str:
    """Produce a retrieval query (deterministic expansion; LLM-assisted when enabled)."""
    base_terms = [request.company, request.role, request.target_interview_type]
    base_terms += [item.name for item in requirements]
    base_terms += extra or []
    # De-duplicate while preserving order and dropping trivial tokens.
    seen: set[str] = set()
    terms: list[str] = []
    for term in base_terms:
        key = term.lower().strip()
        if key and key not in seen:
            seen.add(key)
            terms.append(term)
    return " ".join(terms)


def _llm_enabled(request: PrepareRequest) -> bool:
    if request.llm_mode != "auto":
        return False
    return bool(
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
    )


def _llm_classify_intent(request: PrepareRequest) -> str:  # pragma: no cover - optional
    """Classify intent via the configured chat model; fall back on any error."""
    try:
        from common.utils import load_chat_model

        model = load_chat_model(os.environ.get("MODEL", "qwen:qwen-flash"))
        prompt = (
            "Classify the intent into one of: " + ", ".join(INTENTS) + ".\n"
            f"Role: {request.role}\nJD: {request.job_description[:500]}\n"
            "Answer with only the intent label."
        )
        response = model.invoke(prompt)
        label = str(getattr(response, "content", response)).strip().lower()
        for intent in INTENTS:
            if intent in label:
                return intent
    except Exception:
        return "interview_prep"
    return "interview_prep"


def keyword_overlap(query: str, text: str) -> float:
    """Return the token-overlap ratio (used by evaluation faithfulness checks)."""
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q:
        return 0.0
    return len(q & t) / len(q)
