"""Unit tests for Phase-2 retrieval stack, infra, and evaluation."""

from __future__ import annotations

from pathlib import Path

from job_agent.bm25 import BM25Index
from job_agent.cache import InMemoryCache, get_cache, reset_cache
from job_agent.embeddings import HashingEmbedder, get_embedder
from job_agent.evaluation import ndcg_at_k, recall_at_k, run_eval
from job_agent.hybrid import HybridRetriever, reciprocal_rank_fusion
from job_agent.ingestion import IngestManifest, ManifestItem, build_chunks, ingest
from job_agent.llm import classify_intent, select_retrievers
from job_agent.rag import HybridRAGIndex
from job_agent.rerank import heuristic_rerank
from job_agent.schemas import PrepareRequest, RetrievedChunk
from job_agent.tool_executor import ToolExecutionError, execute
from job_agent.vectorstore import QdrantVectorStore


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=64)
    v1 = emb.embed(["python rag fastapi"])[0]
    v2 = emb.embed(["python rag fastapi"])[0]
    assert v1 == v2
    assert abs(sum(x * x for x in v1) ** 0.5 - 1.0) < 1e-6
    assert isinstance(get_embedder(), HashingEmbedder)


def test_bm25_ranks_relevant_documents_higher() -> None:
    idx = BM25Index()
    idx.add("python rag retrieval augmented generation")
    idx.add("kubernetes docker deployment pipelines")
    top = idx.top_k("rag retrieval", top_k=2)
    assert top
    assert top[0][0] == 0


def test_qdrant_vectorstore_tenant_and_acl_filter() -> None:
    store = QdrantVectorStore(location=":memory:", embedder=HashingEmbedder(dim=64))
    store.upsert([
        {"citation_id": "A", "text": "python rag", "source_path": "a",
         "metadata": {"tenant_id": "t1", "acl_roles": "tenant:t1"}},
        {"citation_id": "B", "text": "python rag", "source_path": "b",
         "metadata": {"tenant_id": "t2", "acl_roles": "tenant:t2"}},
    ])
    hits = store.search("python rag", tenant_id="t1", user_roles=["tenant:t1"])
    assert {h.citation_id for h in hits} == {"A"}
    store.close()


def test_rrf_fuses_rankings() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]])
    assert fused["b"] > fused["c"]


def test_hybrid_retriever_enforces_acl() -> None:
    index = HybridRAGIndex()
    index.add_document("python rag fastapi project", "p1", "cv", tenant_id="t1", acl_roles=["tenant:t1"])
    index.add_document("python rag fastapi secret", "p2", "cv", tenant_id="t2", acl_roles=["tenant:t2"])
    retriever = HybridRetriever(index, embedder=HashingEmbedder(dim=64))
    results = retriever.query("python rag", top_k=5, tenant_id=None, user_roles=["tenant:t1"])
    assert results
    assert all(r.source_path != "p2" for r in results)


def test_heuristic_rerank_orders_by_relevance() -> None:
    chunks = [
        RetrievedChunk(citation_id="C1", text="kubernetes docker", source_path="x", metadata={}, score=0.4),
        RetrievedChunk(citation_id="C2", text="python rag fastapi", source_path="y", metadata={}, score=0.3),
    ]
    ranked = heuristic_rerank("python rag", chunks)
    assert ranked[0].citation_id == "C2"


def test_ingestion_chunks_have_unified_payload(tmp_path: Path) -> None:
    manifest = IngestManifest(items=[
        ManifestItem(text="# Notes\n\nRAG with FastAPI.", doc_type="note", tenant_id="acme",
                     topic="agent", acl_roles=["tenant:acme"]),
    ])
    chunks = build_chunks(manifest, tmp_path)
    assert chunks
    meta = chunks[0]["metadata"]
    for key in ("tenant_id", "doc_type", "section", "topic", "chunk_id", "updated_at", "acl_roles"):
        assert key in meta


def test_ingestion_persists_and_reloads(tmp_path: Path) -> None:
    manifest = IngestManifest(items=[
        ManifestItem(text="RAG FastAPI LangGraph notes.", tenant_id="acme", acl_roles=["tenant:acme"]),
    ])
    loc = str(tmp_path / "qd")
    result = ingest(manifest, location=loc, root=tmp_path)
    assert result["chunks_upserted"] >= 1
    reopened = QdrantVectorStore(collection=manifest.collection, location=loc)
    assert reopened.count() == result["total_points"]
    reopened.close()


def test_tool_executor_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("boom")
        return "ok"

    assert execute(flaky, retries=2) == "ok"


def test_tool_executor_fallback_and_error() -> None:
    assert execute(lambda: (_ for _ in ()).throw(RuntimeError("x")), fallback=lambda: "fb") == "fb"
    try:
        execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        raise AssertionError("expected ToolExecutionError")
    except ToolExecutionError:
        pass


def test_cache_rate_limit_window() -> None:
    cache = InMemoryCache()
    for i in range(3):
        assert cache.hits_in_window("k", 60) == i + 1
    cache.set("v", {"a": 1}, ttl=60)
    assert cache.get("v") == {"a": 1}
    reset_cache()
    assert get_cache() is not None


def test_intent_routing_and_retriever_selection() -> None:
    req = PrepareRequest(company="X", role="AI Agent Engineer",
                         job_description="Explain what RAG is and how retrieval works in agents.",
                         cv_text="python rag")
    assert classify_intent(req) == "concept_qa"
    assert "technical_note" in select_retrievers("concept_qa")


def test_eval_metric_primitives() -> None:
    chunks = [
        RetrievedChunk(citation_id="C1", text="python rag", source_path="x", metadata={}, score=1.0),
        RetrievedChunk(citation_id="C2", text="kubernetes", source_path="y", metadata={}, score=0.5),
    ]
    assert recall_at_k(chunks, ["rag"]) == 1.0
    assert recall_at_k(chunks, ["spark"]) == 0.0
    assert 0.0 < ndcg_at_k(chunks, ["rag"]) <= 1.0


def test_run_eval_v2_produces_real_metrics(tmp_path: Path) -> None:
    eval_file = Path("data/eval/job_agent_eval_v2.jsonl")
    summary = run_eval(eval_file, tmp_path)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert summary["recall_at_k"] >= 0.8
    assert summary["faithfulness"] >= 0.9
    assert 0.0 <= summary["intent_accuracy"] <= 1.0
    assert summary["task_success_rate"] >= 0.8
