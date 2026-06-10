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

# Logical retriever -> the chunk ``source_type`` values it draws from (used by
# source-level routing to filter the candidate pool). Kept deliberately broad so a
# single intent still reaches the CV when relevant — over-narrow filters hurt recall.
RETRIEVER_SOURCE_TYPES: dict[str, set[str]] = {
    "interview_question": {"interview_question_bank", "interview_question"},
    "technical_note": {"project_note", "planning_note", "technical_note", "note"},
    "resume_project": {"cv", "project_note", "resume_project"},
    "plan": {"planning_note", "plan"},
    "company": {"company_note", "company", "job_description"},
    "memory": {"memory"},
}

# Low-confidence routing should broaden rather than narrow.
LOW_CONFIDENCE_THRESHOLD = 0.5


def source_types_for(retrievers: list[str]) -> set[str]:
    """Union of chunk ``source_type`` values for the selected logical retrievers."""
    types: set[str] = set()
    for retriever in retrievers:
        types |= RETRIEVER_SOURCE_TYPES.get(retriever, set())
    return types


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
    """Classify the request intent — LLM-assisted when ``llm_mode=auto``, else rule-based.

    The rule classifier collapses on colloquial / abbreviated inputs (v3 intent ~0.1);
    routing this single decision through an LLM is the highest-leverage use of a model.
    Any LLM failure / invalid label degrades to the deterministic classifier.
    """
    if _llm_enabled(request):
        intent = _llm_classify_intent(request)
        if intent in INTENTS:
            return intent
    return _rule_intent(request)


def _rule_intent(request: PrepareRequest) -> str:
    text = f"{request.role} {request.target_interview_type} {request.job_description}".lower()
    for intent, keywords in _INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return intent
    return "interview_prep"  # default for the end-to-end preparation task


def select_retrievers(intent: str) -> list[str]:
    """Return the logical retrievers for an intent."""
    return INTENT_RETRIEVERS.get(intent, ["technical_note", "resume_project"])


def route_intent(request: PrepareRequest, requirements: list[Requirement] | None = None) -> dict:
    """Full routing decision (Intent Router v2).

    Returns ``{intent, retrievers, confidence, rewrite_needed, source}``. LLM mode asks for
    a strict JSON object (primary/secondary intent, confidence, retrievers, rewrite_needed);
    invalid JSON / labels or low confidence degrade to the rule classifier, and low
    confidence broadens to ``multi_source`` rather than narrowing (protects recall).
    """
    if _llm_enabled(request):
        decision = _llm_route(request)
        if decision is not None:
            intent = decision["intent"]
            confidence = decision["confidence"]
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                intent = "multi_source"  # broaden on uncertainty
            retrievers = decision.get("retrievers") or select_retrievers(intent)
            retrievers = [r for r in retrievers if r in RETRIEVERS] or select_retrievers(intent)
            return {
                "intent": intent, "retrievers": retrievers, "confidence": confidence,
                "rewrite_needed": bool(decision.get("rewrite_needed", True)), "source": "llm",
            }
    intent = _rule_intent(request)
    return {
        "intent": intent, "retrievers": select_retrievers(intent),
        "confidence": 1.0, "rewrite_needed": True, "source": "rule",
    }


def rewrite_query(request: PrepareRequest, requirements: list[Requirement], extra: list[str] | None = None) -> str:
    """Deterministic retrieval-query expansion (company/role/target + requirements + extra)."""
    base_terms = [request.company, request.role, request.target_interview_type]
    base_terms += [item.name for item in requirements]
    base_terms += extra or []
    return _dedup_terms(base_terms)


def _dedup_terms(terms: list[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = term.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(term)
    return " ".join(out)


def llm_rewrite_query(request: PrepareRequest) -> str | None:
    """LLM rewrite of the request into a focused English retrieval query.

    Expands abbreviations (TTFT → time to first token) and normalizes colloquial Chinese
    to professional English terms — directly attacking the query↔evidence vocabulary gap.
    Returns ``None`` when disabled or on any failure (caller keeps the deterministic query).
    """
    if not _llm_enabled(request):
        return None
    if os.environ.get("JOB_AGENT_LLM_REWRITE", "").lower() in ("0", "off", "false"):
        return None  # ablation switch: keep LLM routing but disable query rewrite
    prompt = (
        "Rewrite the following job-interview context into a concise English search query "
        "of professional technical keywords. Expand abbreviations (e.g. TTFT -> time to "
        "first token) and translate colloquial phrasing into standard terminology. "
        "Return ONLY the query, no explanation.\n"
        f"Company: {request.company}\nRole: {request.role}\n"
        f"Context: {request.job_description[:600]}"
    )
    try:
        out = _chat_completion(prompt).strip()
        return out or None
    except Exception:
        return None


_ROUTE_FEWSHOT = (
    "Examples (note the boundaries):\n"
    '- "Explain what RAG is and how retrieval grounds generation" -> concept_qa '
    "(asks to explain a concept, NOT general prep).\n"
    '- "Walk me through the RAG project on your CV" -> project_deepdive '
    "(dig into a specific project, NOT concept_qa).\n"
    '- "Which of my CV projects best matches this JD?" -> jd_cv_match '
    "(align JD to CV).\n"
    '- "Where in my CV is the evidence for evaluation experience?" -> evidence_lookup '
    "(locate evidence, NOT jd_cv_match).\n"
    '- "Combine interview bank, project notes and company info" -> multi_source.\n"'
)


def _llm_route(request: PrepareRequest) -> dict | None:
    """Structured routing via JSON. Returns a validated dict or ``None`` on any failure."""
    import json

    prompt = (
        "You are an intent router for a job-interview RAG agent. Classify the request and "
        "return ONLY a JSON object (no markdown) with keys: primary_intent (one of: "
        + ", ".join(INTENTS) + "), secondary_intents (list, may be empty), confidence "
        "(0.0-1.0), retrievers (subset of: " + ", ".join(RETRIEVERS) + "), rewrite_needed "
        "(boolean), reason (short string).\n" + _ROUTE_FEWSHOT
        + f"\nRole: {request.role}\nTarget: {request.target_interview_type}\n"
        f"Context: {request.job_description[:600]}\nJSON:"
    )
    try:
        raw = _chat_completion(prompt).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < 0:
            return None
        data = json.loads(raw[start : end + 1])
    except Exception:
        return None
    intent = str(data.get("primary_intent", "")).strip().lower()
    if intent not in INTENTS:
        return None
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    retrievers = [str(r).strip() for r in data.get("retrievers", []) if isinstance(r, str)]
    return {
        "intent": intent,
        "confidence": confidence,
        "retrievers": retrievers,
        "rewrite_needed": bool(data.get("rewrite_needed", True)),
    }


def _llm_enabled(request: PrepareRequest) -> bool:
    if request.llm_mode != "auto":
        return False
    return bool(
        os.environ.get("JOB_AGENT_LLM_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
    )


def _llm_classify_intent(request: PrepareRequest) -> str:
    """Classify intent via the chat model; return ``""`` on failure (caller falls back)."""
    prompt = (
        "Classify the user's intent into exactly one of these labels:\n"
        + ", ".join(INTENTS) + ".\n"
        "interview_prep=general prep; project_deepdive=dig into a project; "
        "concept_qa=explain a concept; jd_cv_match=match JD to CV; "
        "evidence_lookup=find CV evidence; company_prep=about the company; "
        "multi_source=combine many sources; no_answer=insufficient/out-of-scope.\n"
        f"Role: {request.role}\nContext: {request.job_description[:500]}\n"
        "Answer with ONLY the label."
    )
    try:
        label = _chat_completion(prompt).strip().lower()
    except Exception:
        return ""
    for intent in INTENTS:
        if intent in label:
            return intent
    return ""


def _chat_completion(prompt: str) -> str:  # pragma: no cover - needs network + key
    """Single LLM seam (OpenAI-compatible chat; SiliconFlow by default). Mocked in tests."""
    from openai import OpenAI

    base = os.environ.get("JOB_AGENT_LLM_API_BASE", "https://api.siliconflow.com/v1")
    api_key = (
        os.environ.get("JOB_AGENT_LLM_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    model = os.environ.get("JOB_AGENT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    client = OpenAI(base_url=base, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


def keyword_overlap(query: str, text: str) -> float:
    """Return the token-overlap ratio (used by evaluation faithfulness checks)."""
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q:
        return 0.0
    return len(q & t) / len(q)
