"""End-to-end local demo for the Job Application Research Agent.

Runs fully offline (deterministic embedder, local Qdrant, no API key):
  1. ingest a small knowledge manifest into Qdrant,
  2. prepare a cited report (hybrid retrieval),
  3. resume the same session from its checkpoints,
  4. run the 150-case evaluation and print the summary.

Usage:  uv run python scripts/demo.py
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

from job_agent.ingestion import IngestManifest, ManifestItem, ingest
from job_agent.schemas import PrepareRequest
from job_agent.session import SessionStore
from job_agent.workflow import prepare_application, resume_application


def _section(title: str) -> None:
    print("\n" + "=" * 70)  # noqa: T201
    print(title)  # noqa: T201
    print("=" * 70)  # noqa: T201


def main() -> None:
    """Run the four-step demo."""
    workdir = Path(tempfile.mkdtemp(prefix="job_agent_demo_"))
    store = SessionStore(workdir / "demo.sqlite")

    _section("1) Ingest knowledge -> Qdrant (local, persistent)")
    manifest = IngestManifest(items=[
        ManifestItem(
            text="# Agent Notes\n\nRAG grounds generation with retrieved evidence. "
                 "ReAct loops reason and act; loop detection prevents repeats.",
            doc_type="technical_note", tenant_id="demo", topic="agent",
            acl_roles=["tenant:demo"],
        ),
    ])
    ingest_result = ingest(manifest, location=str(workdir / "qdrant"), root=Path.cwd())
    print(json.dumps(ingest_result, indent=2, ensure_ascii=False))  # noqa: T201

    _section("2) Prepare a cited report (hybrid retrieval)")
    request = PrepareRequest(
        company="Tencent", role="AI Agent Engineer",
        job_description="We need Python, RAG, AI Agent, Function Calling, FastAPI, "
                        "evaluation and safety guardrails.",
        cv_text="Skills: Python, RAG, FastAPI, LangGraph, evaluation harness. "
                "Project: Job Application Agent with hybrid RAG and citations.",
        retrieval_mode="hybrid", tenant_id="demo",
    )
    session_id = str(uuid.uuid4())
    response = prepare_application(request, store=store, session_id=session_id)
    print(f"intent           : {response.intent}")  # noqa: T201
    print(f"retrievers       : {response.selected_retrievers}")  # noqa: T201
    print(f"citations        : {len(response.citations)}")  # noqa: T201
    print(f"nodes executed   : {[t.node for t in response.node_trace]}")  # noqa: T201
    print(f"retrieval rounds : {response.metrics.retrieval_rounds}")  # noqa: T201
    print(f"guardrail        : {response.guardrail.classification} | "  # noqa: T201
          f"grounding unsupported: {len(response.output_guardrail.unsupported_claims) if response.output_guardrail else 0}")
    print(f"report saved to  : {response.report_path}")  # noqa: T201

    _section("3) Resume the session from checkpoints")
    resumed = resume_application(session_id, store=store)
    print(f"resumed session  : {resumed.session_id} (report len={len(resumed.report)})")  # noqa: T201
    print(f"completed nodes  : {store.get_completed_nodes(session_id)}")  # noqa: T201

    _section("4) Run the 150-case evaluation")
    from job_agent.evaluation import run_eval
    eval_file = Path("data") / "eval" / "job_agent_eval_v2.jsonl"
    if eval_file.exists():
        summary = run_eval(eval_file, workdir / "eval")
        print(json.dumps(summary, indent=2, ensure_ascii=False))  # noqa: T201
    else:
        print("eval dataset missing; run `uv run python -m job_agent.eval_dataset`")  # noqa: T201

    _section("Done")
    print(f"artifacts in: {workdir}")  # noqa: T201


if __name__ == "__main__":
    main()
