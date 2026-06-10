"""Unit tests for Phase-5.3: Intent Router v2, source-level routing, confusion matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_agent import llm
from job_agent.evaluation import intent_confusion
from job_agent.hybrid import HybridRetriever
from job_agent.llm import RETRIEVER_SOURCE_TYPES, route_intent, source_types_for
from job_agent.rag import HybridRAGIndex
from job_agent.schemas import PrepareRequest
from job_agent.session import SessionStore
from job_agent.tuning import ROUTING_REPORT_METRICS, run_routing_compare
from job_agent.workflow import prepare_application


def _req(**kw) -> PrepareRequest:
    base = dict(company="Tencent", role="AI Agent Engineer",
               job_description="Explain what RAG is and how retrieval works. Python FastAPI.",
               cv_text="Skills: Python, RAG, FastAPI. Project: Job Agent.")
    base.update(kw)
    return PrepareRequest(**base)


# --------------------------------------------------------------------------- #
# Intent Router v2
# --------------------------------------------------------------------------- #


def test_route_intent_rule_fallback() -> None:
    decision = route_intent(_req(llm_mode="off"))
    assert decision["intent"] == "concept_qa"
    assert decision["source"] == "rule"
    assert decision["confidence"] == 1.0


def test_route_intent_llm_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "k")
    monkeypatch.setattr(llm, "_chat_completion", lambda p: (
        '{"primary_intent":"project_deepdive","secondary_intents":[],'
        '"confidence":0.9,"retrievers":["resume_project","technical_note"],'
        '"rewrite_needed":true,"reason":"dig into project"}'
    ))
    decision = route_intent(_req(llm_mode="auto"))
    assert decision["intent"] == "project_deepdive"
    assert decision["source"] == "llm"
    assert "resume_project" in decision["retrievers"]


def test_route_intent_low_confidence_broadens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "k")
    monkeypatch.setattr(llm, "_chat_completion", lambda p: (
        '{"primary_intent":"concept_qa","confidence":0.2,"retrievers":["technical_note"],'
        '"rewrite_needed":false}'
    ))
    assert route_intent(_req(llm_mode="auto"))["intent"] == "multi_source"


def test_route_intent_invalid_json_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "k")
    monkeypatch.setattr(llm, "_chat_completion", lambda p: "not json at all")
    decision = route_intent(_req(llm_mode="auto", job_description="Walk me through your project."))
    assert decision["source"] == "rule"  # invalid JSON -> rule path
    assert decision["intent"] == "project_deepdive"


# --------------------------------------------------------------------------- #
# Source-level routing
# --------------------------------------------------------------------------- #


def test_source_types_for_union() -> None:
    types = source_types_for(["resume_project", "company"])
    assert "cv" in types
    assert RETRIEVER_SOURCE_TYPES["company"] <= types


def test_hybrid_source_filter_restricts_pool() -> None:
    index = HybridRAGIndex()
    index.add_document("python rag fastapi cv project", "cv1", "cv",
                       tenant_id="t", acl_roles=["tenant:t"])
    index.add_document("interview question about agents", "iq1", "interview_question_bank",
                       tenant_id="t", acl_roles=["tenant:t"])
    retriever = HybridRetriever(index)
    only_cv = retriever.query("python rag agents", top_k=10, tenant_id="t",
                              user_roles=["tenant:t"], source_types={"cv"})
    assert only_cv and all(c.metadata.get("source_type") == "cv" for c in only_cv)


def test_source_routing_broaden_fallback_protects_recall(tmp_path: Path) -> None:
    """concept_qa excludes CV; with little non-CV evidence the agent must broaden back."""
    store = SessionStore(tmp_path / "s.sqlite")
    resp = prepare_application(_req(retrieval_mode="hybrid", source_routing=True), store=store)
    # CV-centric corpus -> source filter is too narrow -> broaden -> still gets citations.
    assert resp.citations
    record = store.get_session(resp.session_id)
    assert record is not None  # ran to completion despite narrow source filter


# --------------------------------------------------------------------------- #
# Confusion matrix + routing ablation
# --------------------------------------------------------------------------- #


def test_intent_confusion_matrix() -> None:
    rows = [
        {"id": "a", "expected_intent": "concept_qa", "predicted_intent": "concept_qa"},
        {"id": "b", "expected_intent": "concept_qa", "predicted_intent": "interview_prep"},
        {"id": "c", "expected_intent": "jd_cv_match", "predicted_intent": "evidence_lookup"},
    ]
    data = intent_confusion(rows)
    assert data["matrix"]["concept_qa"]["interview_prep"] == 1
    pairs = {(t["expected"], t["predicted"]) for t in data["top_confused"]}
    assert ("concept_qa", "interview_prep") in pairs


def test_routing_ablation_harness_offline(tmp_path: Path) -> None:
    cases = [{
        "id": "r1", "task_type": "concept_qa", "expected_intent": "concept_qa",
        "gold_evidence": ["RAG"], "must_answer_points": ["RAG"],
        "request": {"company": "X", "role": "AI Agent Engineer",
                    "job_description": "Explain what RAG is.", "cv_text": "Python RAG FastAPI.",
                    "retrieval_mode": "hybrid"},
    }]
    result = run_routing_compare(cases, tmp_path, metric="recall_at_k")
    assert {r["arm"] for r in result["results"]} == {
        "off", "routing_only", "routing+source", "routing+source+rewrite"}
    report = (tmp_path / "routing_ablation.md").read_text(encoding="utf-8")
    for metric in ROUTING_REPORT_METRICS:
        assert metric in report
    # The rewrite ablation env must not leak.
    import os
    assert "JOB_AGENT_LLM_REWRITE" not in os.environ
