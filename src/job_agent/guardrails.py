"""Input, tool, and output guardrails for the job agent."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from job_agent.schemas import GuardrailResult, RetrievedChunk

INJECTION_PATTERNS = {
    "ignore_instructions": re.compile(
        r"ignore (all )?(previous|prior|above|system) instructions", re.I
    ),
    "system_prompt_leak": re.compile(
        r"(reveal|show|print|dump).{0,30}(system prompt|developer message|hidden prompt)",
        re.I,
    ),
    "tool_bypass": re.compile(r"(bypass|disable).{0,30}(tool|guardrail|safety)", re.I),
    "credential_leak": re.compile(r"(api key|secret|password|token|credential)", re.I),
    "fake_experience": re.compile(
        r"(fabricate|make up|invent).{0,40}(experience|internship|project|cv)",
        re.I,
    ),
}

ALLOWED_TOOLS = {
    "rag_retrieval_tool",
    "jd_parser_tool",
    "cv_parser_tool",
    "cv_match_tool",
    "company_research_tool",
    "interview_question_tool",
    "report_writer_tool",
}


# Patterns whose presence should hard-block the request outright.
HARD_BLOCK_REASONS = {"ignore_instructions", "system_prompt_leak", "tool_bypass"}
# Patterns that should trigger a sanitize/redaction path rather than a block.
SANITIZE_REASONS = {"credential_leak"}
# Patterns that indicate the user is asking us to fabricate experience.
CLARIFY_REASONS = {"fake_experience"}

REDACTION_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)(system prompt|developer message)\s*[:=].*"),
    re.compile(r"[A-Za-z]:\\\\[^\s]+|/(?:home|Users)/[^\s]+"),
]


def check_prompt_injection(*texts: str | None) -> GuardrailResult:
    """Detect common prompt-injection and unsafe CV-claim patterns."""
    reasons: list[str] = []
    combined = "\n".join(text for text in texts if text)
    for name, pattern in INJECTION_PATTERNS.items():
        if pattern.search(combined):
            reasons.append(name)

    risk = min(1.0, 0.25 * len(reasons))
    return GuardrailResult(
        blocked=risk >= 0.75,
        risk_score=risk,
        reasons=reasons,
        classification=_classify(reasons, risk),
    )


def _classify(reasons: list[str], risk: float) -> str:
    """Map detected reasons to an action category."""
    reason_set = set(reasons)
    if reason_set & HARD_BLOCK_REASONS and risk >= 0.5:
        return "block"
    if reason_set & CLARIFY_REASONS:
        return "clarify"
    if reason_set & SANITIZE_REASONS:
        return "sanitize"
    if reasons:
        return "sanitize"
    return "pass"


def classify_input(*texts: str | None) -> GuardrailResult:
    """Classify input into pass/sanitize/clarify/block (alias with semantics)."""
    return check_prompt_injection(*texts)


def redact_sensitive(text: str) -> str:
    """Redact API keys, tokens, system prompts, and internal paths from text."""
    redacted = text
    for pattern in REDACTION_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def validate_tool_call(tool_name: str, args: Mapping[str, Any]) -> GuardrailResult:
    """Validate tool allowlist and basic argument shape."""
    reasons: list[str] = []
    if tool_name not in ALLOWED_TOOLS:
        reasons.append("tool_not_allowlisted")
    if not isinstance(args, Mapping):
        reasons.append("tool_args_not_mapping")
    if any(str(key).startswith("_") for key in args):
        reasons.append("private_arg_name")
    return GuardrailResult(
        blocked=bool(reasons),
        risk_score=1.0 if reasons else 0.0,
        reasons=reasons,
    )


CLAIM_KEYWORDS = re.compile(
    r"\b(strong|expert|proficient|mastery|built|implemented|developed|delivered|led|"
    r"experience|hands-on|production)\b",
    re.I,
)
STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "have", "from", "your", "into",
    "using", "use", "a", "an", "of", "to", "in", "on", "by", "is", "are", "as",
    "requirement", "match", "evidence", "preparation", "no", "direct", "citation",
}


def _claim_tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z0-9+#.]+", text.lower()) if tok not in STOPWORDS and len(tok) > 2}


def check_claim_grounding(report: str, citations: list[RetrievedChunk]) -> GuardrailResult:
    """Claim-level grounding: verify each capability claim against retrieved evidence.

    A "claim" is a report line that asserts a candidate capability. It is supported
    when it carries an inline ``[C...]`` citation marker or its salient tokens overlap
    with at least one retrieved chunk. Unsupported capability claims are surfaced so
    the workflow can downgrade them to gaps/assumptions instead of fabricating skills.
    """
    citation_ids = {chunk.citation_id for chunk in citations}
    citation_token_sets = [_claim_tokens(chunk.text) for chunk in citations]

    unsupported: list[str] = []
    reasons: list[str] = []
    in_match_section = False

    # Only the JD-CV match table makes first-person capability claims about the
    # candidate. Interview-question prompts and citations are not claims, so scanning
    # them would produce false positives.
    for raw in report.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            in_match_section = "match" in line.lower()
            continue
        if not in_match_section or not line:
            continue
        is_claim_line = line.startswith("-") or (line.startswith("|") and "---" not in line)
        if not is_claim_line or not CLAIM_KEYWORDS.search(line):
            continue

        inline_ids = set(re.findall(r"\[(C\d+)\]", line))
        if inline_ids & citation_ids:
            continue
        line_tokens = _claim_tokens(line)
        supported = any(len(line_tokens & chunk_tokens) >= 2 for chunk_tokens in citation_token_sets)
        if not supported:
            unsupported.append(line[:160])

    if not citations:
        reasons.append("no_citations")
    if "[C" not in report:
        reasons.append("missing_inline_citations")
    if re.search(r"\bguarantee(d)?\b|\b100%\b", report, flags=re.I):
        reasons.append("overconfident_claim")
    if unsupported:
        reasons.append("unsupported_claims")

    risk = min(1.0, 0.2 * len(reasons) + 0.1 * len(unsupported))
    # Many unsupported capability claims is a fabrication risk worth blocking.
    blocked = len(unsupported) >= 3
    classification = "block" if blocked else ("sanitize" if unsupported else "pass")
    return GuardrailResult(
        blocked=blocked,
        risk_score=round(risk, 4),
        reasons=reasons,
        classification=classification,
        unsupported_claims=unsupported,
    )


def check_output_grounding(report: str, citations: list[RetrievedChunk]) -> GuardrailResult:
    """Backward-compatible grounding check delegating to claim-level grounding."""
    return check_claim_grounding(report, citations)
