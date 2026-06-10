"""Integration tests for Phase-2 API routes: streaming and knowledge ingestion."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from job_agent import entrypoint
from job_agent.cache import reset_cache
from job_agent.entrypoint import app


def _client() -> TestClient:
    reset_cache()
    return TestClient(app)


def test_phase2_routes_registered() -> None:
    routes = {route.path for route in app.routes}
    assert "/prepare/stream" in routes
    assert "/knowledge/ingest" in routes


def test_prepare_stream_emits_skeleton_then_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JOB_AGENT_DB_PATH", str(tmp_path / "job_agent.sqlite"))
    client = _client()
    payload = {
        "company": "Tencent", "role": "AI Agent Engineer",
        "job_description": "Python RAG AI Agent FastAPI evaluation guardrails.",
        "cv_text": "Skills: Python, RAG, FastAPI. Project: Job Agent.",
        "retrieval_mode": "hybrid",
    }
    with client.stream("POST", "/prepare/stream", json=payload) as response:
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]
    assert events[0]["event"] == "skeleton"
    assert events[-1]["event"] == "result"
    assert events[-1]["response"]["report"]


def test_knowledge_ingest_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qd"))
    client = _client()
    manifest = {
        "collection": "job_agent_knowledge",
        "items": [
            {"text": "RAG with FastAPI and LangGraph notes.", "doc_type": "note",
             "tenant_id": "acme", "acl_roles": ["tenant:acme"]},
        ],
    }
    response = client.post("/knowledge/ingest", json=manifest)
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_upserted"] >= 1
    assert body["collection"] == "job_agent_knowledge"


def test_eval_run_prefers_v2_dataset(monkeypatch) -> None:
    seen = {}

    def fake_run_eval(eval_file, output_dir):
        seen["eval_file"] = eval_file
        seen["output_dir"] = output_dir
        return {"recall_at_k": 1.0}

    monkeypatch.setattr(entrypoint, "run_eval", fake_run_eval)
    client = _client()
    response = client.post("/eval/run")
    assert response.status_code == 200
    body = response.json()
    assert body["eval_file"].endswith("job_agent_eval_v2.jsonl")
    assert seen["eval_file"].name == "job_agent_eval_v2.jsonl"
