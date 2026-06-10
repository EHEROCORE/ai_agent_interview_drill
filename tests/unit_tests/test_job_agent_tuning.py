"""Unit test for the hybrid-fusion weight sweep (offline, tiny grid)."""

from __future__ import annotations

import os
from pathlib import Path

from job_agent.rerank import rerank
from job_agent.schemas import RetrievedChunk
from job_agent.tuning import (
    REPORT_METRICS,
    RERANK_REPORT_METRICS,
    run_fusion_sweep,
    run_rerank_compare,
)


def _cases() -> list[dict]:
    base_request = {
        "company": "Tencent", "role": "AI Agent Engineer",
        "job_description": "Python RAG AI Agent FastAPI evaluation guardrails.",
        "cv_text": "Skills: Python, RAG, FastAPI, LangGraph. Project: Job Agent.",
        "retrieval_mode": "hybrid", "tenant_id": "sweep",
    }
    return [
        {"id": "s1", "task_type": "interview_answer", "expected_intent": "interview_prep",
         "gold_evidence": ["RAG", "FastAPI"], "must_answer_points": ["RAG"], "request": base_request},
        {"id": "s2", "task_type": "jd_cv_match", "expected_intent": "jd_cv_match",
         "gold_evidence": ["Python"], "must_answer_points": ["Python"],
         "request": {**base_request, "job_description": "Python and RAG required."}},
    ]


def test_fusion_sweep_runs_and_ranks(tmp_path: Path) -> None:
    grid = [
        {"bm25_weight": 1.0, "dense_weight": 1.0},
        {"bm25_weight": 0.5, "dense_weight": 2.0, "rrf_k": 30},
    ]
    result = run_fusion_sweep(_cases(), tmp_path, grid=grid, metric="recall_at_k")

    assert len(result["results"]) == 2
    # Ranked descending by the target metric.
    metrics = [r["summary"]["recall_at_k"] for r in result["results"]]
    assert metrics == sorted(metrics, reverse=True)
    assert result["best"]["combo"] in grid

    report = (tmp_path / "fusion_sweep.md").read_text(encoding="utf-8")
    assert "Hybrid Fusion Weight Sweep" in report
    for metric in REPORT_METRICS:
        assert metric in report
    assert (tmp_path / "fusion_sweep.json").exists()
    # Sweeping must not leak fusion env vars after completion.
    assert "JOB_AGENT_FUSION_DENSE_WEIGHT" not in os.environ


def test_rerank_none_is_passthrough() -> None:
    """The 'none' reranker keeps fusion order (only truncating to top_k)."""
    chunks = [
        RetrievedChunk(citation_id="C1", text="kubernetes", source_path="x", metadata={}, score=0.9),
        RetrievedChunk(citation_id="C2", text="python rag fastapi", source_path="y", metadata={}, score=0.1),
    ]
    prior = os.environ.get("JOB_AGENT_RERANKER")
    os.environ["JOB_AGENT_RERANKER"] = "none"
    try:
        out = rerank("python rag", chunks, top_k=2)
    finally:
        if prior is None:
            os.environ.pop("JOB_AGENT_RERANKER", None)
        else:
            os.environ["JOB_AGENT_RERANKER"] = prior
    assert [c.citation_id for c in out] == ["C1", "C2"]  # order preserved, not re-sorted


def test_rerank_compare_runs_and_restores_env(tmp_path: Path) -> None:
    result = run_rerank_compare(_cases(), tmp_path, strategies=["none", "heuristic"], metric="mrr")
    assert len(result["results"]) == 2
    report = (tmp_path / "rerank_before_after.md").read_text(encoding="utf-8")
    assert "Rerank Before/After" in report
    for metric in RERANK_REPORT_METRICS:
        assert metric in report
    assert (tmp_path / "rerank_before_after.json").exists()
    # Pinned fusion env + reranker env must be restored after the comparison.
    assert "JOB_AGENT_FUSION_DENSE_WEIGHT" not in os.environ
    assert "JOB_AGENT_RERANKER" not in os.environ
