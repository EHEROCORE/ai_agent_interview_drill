"""Unit tests for Phase-4: embedding cache, tunable fusion, post-retrieval ACL audit."""

from __future__ import annotations

from pathlib import Path

from job_agent.acl import audit
from job_agent.embeddings import CachingEmbedder, EmbeddingCache, HashingEmbedder
from job_agent.hybrid import FusionConfig, reciprocal_rank_fusion
from job_agent.schemas import RetrievedChunk


class _CountingEmbedder(HashingEmbedder):
    """Hashing embedder that counts how many texts it actually embeds."""

    def __init__(self) -> None:
        super().__init__(dim=8)
        self.name = "counting"
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return super().embed(texts)


def test_embedding_cache_serves_hits_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "ec.sqlite"
    inner = _CountingEmbedder()
    cached = CachingEmbedder(inner, EmbeddingCache(path))

    cached.embed(["a", "b"])
    cached.embed(["a", "b", "c"])  # a,b are hits; only c is new
    assert inner.calls == 3
    assert cached.inner_calls == 3

    # A fresh instance on the same path reuses everything (cross-process persistence).
    inner2 = _CountingEmbedder()
    cached2 = CachingEmbedder(inner2, EmbeddingCache(path))
    vecs = cached2.embed(["a", "b", "c"])
    assert inner2.calls == 0
    assert len(vecs) == 3 and len(vecs[0]) == 8


def test_fusion_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("JOB_AGENT_FUSION_DENSE_WEIGHT", "2.0")
    monkeypatch.setenv("JOB_AGENT_FUSION_RRF_K", "10")
    monkeypatch.setenv("JOB_AGENT_FUSION_FINAL_TOPK", "5")
    cfg = FusionConfig.from_env()
    assert cfg.dense_weight == 2.0
    assert cfg.rrf_k == 10
    assert cfg.final_top_k == 5
    assert cfg.bm25_weight == 1.0  # untouched default


def test_weighted_rrf_favours_higher_weighted_arm() -> None:
    bm25 = ["x", "a"]      # x ranked first by sparse
    dense = ["y", "a"]     # y ranked first by dense
    # Weight dense 3x: y should outrank x despite identical ranks.
    fused = reciprocal_rank_fusion([bm25, dense], k=60, weights=[1.0, 3.0])
    assert fused["y"] > fused["x"]


def _chunk(cid: str, tenant: str, roles: str) -> RetrievedChunk:
    return RetrievedChunk(citation_id=cid, text="t", source_path=cid,
                          metadata={"tenant_id": tenant, "acl_roles": roles}, score=1.0)


def test_post_retrieval_acl_audit_catches_leaks() -> None:
    chunks = [
        _chunk("ok_pub", "acme", "public"),                 # public role within tenant
        _chunk("ok_tenant", "acme", "tenant:acme"),         # matching tenant role
        _chunk("leak_tenant", "other", "tenant:other"),     # wrong tenant (hard partition)
        _chunk("leak_role", "acme", "tenant:secret"),       # right tenant, wrong role
    ]
    allowed, denied = audit(chunks, tenant_id="acme", user_roles=["tenant:acme"])
    allowed_ids = {c.citation_id for c in allowed}
    assert allowed_ids == {"ok_pub", "ok_tenant"}
    reasons = {d["citation_id"]: d["reason"] for d in denied}
    assert reasons["leak_tenant"] == "tenant_mismatch"
    assert reasons["leak_role"] == "role_denied"
