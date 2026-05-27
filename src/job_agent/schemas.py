"""Structured schemas for the job application research agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from common.basemodel import AgentBaseModel


class PrepareRequest(AgentBaseModel):
    """Input payload for preparing an application/interview report."""

    company: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=160)
    job_description: str = Field(min_length=20, max_length=30000)
    cv_text: str | None = Field(default=None, max_length=30000)
    cv_file_path: str | None = Field(default=None, max_length=500)
    target_interview_type: str = Field(default="AI Agent / LLM application")

    @field_validator("company", "role", "target_interview_type")
    @classmethod
    def strip_short_text(cls, value: str) -> str:
        """Normalize short text fields."""
        return value.strip()

    @model_validator(mode="after")
    def require_cv_source(self) -> PrepareRequest:
        """Require either inline CV text or a CV file path."""
        if not self.cv_text and not self.cv_file_path:
            raise ValueError("Either cv_text or cv_file_path is required.")
        return self


class FeedbackRequest(AgentBaseModel):
    """User feedback for data flywheel collection."""

    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=2000)
    failure_tags: list[str] = Field(default_factory=list)


class GuardrailResult(AgentBaseModel):
    """Guardrail decision and reasons."""

    blocked: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class Requirement(AgentBaseModel):
    """A requirement extracted from a job description."""

    name: str
    category: Literal["technical", "agent", "rag", "backend", "soft_skill", "domain"]
    evidence: str


class CVProfile(AgentBaseModel):
    """Structured candidate profile extracted from a CV."""

    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)


class RetrievedChunk(AgentBaseModel):
    """RAG retrieval result."""

    citation_id: str
    text: str
    source_path: str
    metadata: dict[str, str] = Field(default_factory=dict)
    score: float


class MatchItem(AgentBaseModel):
    """A JD requirement matched against CV/RAG evidence."""

    requirement: str
    match_level: Literal["strong", "partial", "gap"]
    evidence: list[RetrievedChunk] = Field(default_factory=list)
    gap: str = ""
    preparation_suggestion: str = ""


class ToolTrace(AgentBaseModel):
    """Trace entry for a tool call."""

    tool_name: str
    args_hash: str
    status: Literal["ok", "error", "blocked"]
    latency_ms: float
    observation: dict[str, Any] = Field(default_factory=dict)


class Metrics(AgentBaseModel):
    """Request-level observability metrics."""

    ttft_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float
    tool_call_count: int
    guardrail_pass: bool
    citation_count: int


class PrepareResponse(AgentBaseModel):
    """Response returned by the prepare workflow and API."""

    session_id: str
    request_id: str
    company: str
    role: str
    report: str
    report_path: str
    citations: list[RetrievedChunk]
    matches: list[MatchItem]
    interview_questions: list[str]
    guardrail: GuardrailResult
    metrics: Metrics
    tool_trace: list[ToolTrace]


class SessionRecord(AgentBaseModel):
    """Persisted session state for recovery and inspection."""

    session_id: str
    request_id: str
    status: Literal["running", "completed", "failed", "blocked"]
    company: str
    role: str
    current_step: str
    payload: dict[str, Any] = Field(default_factory=dict)
