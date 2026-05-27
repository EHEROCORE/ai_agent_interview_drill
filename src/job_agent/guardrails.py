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


def check_prompt_injection(*texts: str | None) -> GuardrailResult:
    """Detect common prompt-injection and unsafe CV-claim patterns."""
    reasons: list[str] = []
    combined = "\n".join(text for text in texts if text)
    for name, pattern in INJECTION_PATTERNS.items():
        if pattern.search(combined):
            reasons.append(name)

    risk = min(1.0, 0.25 * len(reasons))
    return GuardrailResult(blocked=risk >= 0.75, risk_score=risk, reasons=reasons)


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


def check_output_grounding(report: str, citations: list[RetrievedChunk]) -> GuardrailResult:
    """Check that final report uses citations for evidence-heavy claims."""
    reasons: list[str] = []
    if not citations:
        reasons.append("no_citations")
    if "[C" not in report:
        reasons.append("missing_inline_citations")
    if re.search(r"\bguarantee(d)?\b|\b100%\b", report, flags=re.I):
        reasons.append("overconfident_claim")
    risk = min(1.0, 0.35 * len(reasons))
    return GuardrailResult(blocked=False, risk_score=risk, reasons=reasons)
