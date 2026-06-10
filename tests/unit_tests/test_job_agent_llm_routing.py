"""Unit tests for LLM intent routing + query rewrite (mocked LLM seam)."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_agent import llm
from job_agent.schemas import PrepareRequest
from job_agent.tuning import INTENT_REPORT_METRICS, run_intent_compare


def _req(llm_mode: str = "auto", jd: str = "Walk me through your RAG project.") -> PrepareRequest:
    return PrepareRequest(
        company="Tencent", role="AI Agent Engineer", job_description=jd,
        cv_text="Python, RAG, FastAPI.", llm_mode=llm_mode,
    )


def test_llm_intent_used_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_chat_completion", lambda prompt: "concept_qa")
    # Rule routing on this JD would say project_deepdive; the LLM overrides it.
    assert llm.classify_intent(_req(jd="Walk me through your RAG project.")) == "concept_qa"


def test_llm_intent_invalid_label_falls_back_to_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_chat_completion", lambda prompt: "not-a-real-label")
    # Falls back to the rule classifier (JD mentions a project).
    assert llm.classify_intent(_req(jd="Walk me through your project.")) == "project_deepdive"


def test_llm_intent_exception_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    def boom(prompt: str) -> str:
        raise RuntimeError("api down")

    monkeypatch.setattr(llm, "_chat_completion", boom)
    assert llm.classify_intent(_req(jd="Explain what RAG is.")) == "concept_qa"  # rule path


def test_llm_disabled_without_key_or_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JOB_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    # llm_mode=auto but no key -> disabled; llm_mode=off -> disabled.
    assert llm._llm_enabled(_req(llm_mode="auto")) is False
    assert llm.llm_rewrite_query(_req(llm_mode="auto")) is None
    monkeypatch.setenv("SILICONFLOW_API_KEY", "k")
    assert llm._llm_enabled(_req(llm_mode="off")) is False


def test_llm_query_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_chat_completion", lambda prompt: "loop detection termination agent")
    assert llm.llm_rewrite_query(_req()) == "loop detection termination agent"


def test_intent_compare_harness_offline(tmp_path: Path) -> None:
    cases = [
        {"id": "i1", "task_type": "concept_qa", "expected_intent": "concept_qa",
         "gold_evidence": ["RAG"], "must_answer_points": ["RAG"],
         "request": {"company": "X", "role": "AI Agent Engineer",
                     "job_description": "Explain what RAG is and how retrieval works.",
                     "cv_text": "Python RAG FastAPI."}},
    ]
    result = run_intent_compare(cases, tmp_path, modes=["off", "auto"], metric="intent_accuracy")
    assert {r["llm_mode"] for r in result["results"]} == {"off", "auto"}
    report = (tmp_path / "intent_before_after.md").read_text(encoding="utf-8")
    assert "Intent Routing Before/After" in report
    for metric in INTENT_REPORT_METRICS:
        assert metric in report
