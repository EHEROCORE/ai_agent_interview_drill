"""Hybrid retriever: BM25 (sparse) + Qdrant (dense) fused with RRF + metadata boost.

Built on top of an existing :class:`HybridRAGIndex` so it shares the same tenant/ACL
metadata and chunk objects. Acts as a drop-in for the deterministic lexical retriever:
``query()`` returns ``list[RetrievedChunk]`` with tenant isolation and ACL enforced on
both arms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from job_agent.bm25 import BM25Index
from job_agent.embeddings import Embedder
from job_agent.rag import HybridRAGIndex, tokenize
from job_agent.schemas import RetrievedChunk
from job_agent.vectorstore import QdrantVectorStore

RRF_K = 60


@dataclass(frozen=True)
class FusionConfig:
    """Tunable hybrid-fusion knobs.

    These are the levers for the Phase-4 finding that an evenly-weighted fusion can
    *underperform* a strong dense arm. They are overridable via ``JOB_AGENT_FUSION_*``
    env vars so weights / RRF k / per-arm top-k can be tuned against an eval set
    without code changes. Defaults preserve the original balanced behaviour.
    """

    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    rrf_k: int = RRF_K
    metadata_boost_weight: float = 0.02
    sparse_top_k: int = 30
    dense_top_k: int = 30
    final_top_k: int = 10

    @classmethod
    def from_env(cls) -> FusionConfig:
        """Build a config from ``JOB_AGENT_FUSION_*`` env vars (falling back to defaults)."""
        def _f(name: str, default: float) -> float:
            try:
                return float(os.environ[name])
            except (KeyError, ValueError):
                return default

        def _i(name: str, default: int) -> int:
            try:
                return int(os.environ[name])
            except (KeyError, ValueError):
                return default

        return cls(
            bm25_weight=_f("JOB_AGENT_FUSION_BM25_WEIGHT", 1.0),
            dense_weight=_f("JOB_AGENT_FUSION_DENSE_WEIGHT", 1.0),
            rrf_k=_i("JOB_AGENT_FUSION_RRF_K", RRF_K),
            metadata_boost_weight=_f("JOB_AGENT_FUSION_META_BOOST", 0.02),
            sparse_top_k=_i("JOB_AGENT_FUSION_SPARSE_TOPK", 30),
            dense_top_k=_i("JOB_AGENT_FUSION_DENSE_TOPK", 30),
            final_top_k=_i("JOB_AGENT_FUSION_FINAL_TOPK", 10),
        )


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K, weights: list[float] | None = None
) -> dict[str, float]:
    """Fuse several ranked id-lists into a single score map via (weighted) RRF."""
    fused: dict[str, float] = {}
    for i, ranked in enumerate(ranked_lists):
        weight = weights[i] if weights is not None else 1.0
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)
    return fused


class HybridRetriever:
    """Sparse+dense hybrid retriever with tunable, weighted reciprocal-rank fusion."""

    def __init__(
        self,
        index: HybridRAGIndex,
        embedder: Embedder | None = None,
        fusion: FusionConfig | None = None,
    ) -> None:
        """Build BM25 and Qdrant arms from an indexed :class:`HybridRAGIndex`."""
        self._index = index
        self.fusion = fusion or FusionConfig.from_env()
        self._by_id = {chunk.citation_id: chunk for chunk in index.chunks}
        self._bm25 = BM25Index()
        self._bm25_ids: list[str] = []
        for chunk in index.chunks:
            self._bm25.add(chunk.text)
            self._bm25_ids.append(chunk.citation_id)
        self._dense = QdrantVectorStore(location=":memory:", embedder=embedder)
        self._dense.upsert(
            [
                {
                    "citation_id": chunk.citation_id,
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "metadata": chunk.metadata,
                }
                for chunk in index.chunks
            ]
        )

    def _acl_allows(
        self, citation_id: str, tenant_id: str | None, user_roles: list[str] | None,
        source_types: set[str] | None = None,
    ) -> bool:
        chunk = self._by_id.get(citation_id)
        if chunk is None:
            return False
        if tenant_id is not None and chunk.metadata.get("tenant_id", "default") != tenant_id:
            return False
        if source_types is not None and chunk.metadata.get("source_type") not in source_types:
            return False
        return HybridRAGIndex._acl_ok(chunk, user_roles)

    def _metadata_boost(self, citation_id: str, query_tokens: set[str]) -> float:
        chunk = self._by_id.get(citation_id)
        if chunk is None:
            return 0.0
        values = " ".join(str(value) for value in chunk.metadata.values()).lower()
        meta_tokens = set(tokenize(values))
        return self.fusion.metadata_boost_weight * len(query_tokens & meta_tokens)

    def query_dense(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
        user_roles: list[str] | None = None,
        source_types: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Dense-only retrieval (the Qdrant arm), for lexical/dense/hybrid A/B."""
        pool = top_k * 3 if source_types is not None else top_k
        hits = self._dense.search(query, top_k=pool, tenant_id=tenant_id, user_roles=user_roles)
        results: list[RetrievedChunk] = []
        for hit in hits:
            chunk = self._by_id.get(hit.citation_id)
            if chunk is None:
                continue
            if source_types is not None and chunk.metadata.get("source_type") not in source_types:
                continue
            results.append(
                RetrievedChunk(
                    citation_id=chunk.citation_id, text=chunk.text,
                    source_path=chunk.source_path, metadata=chunk.metadata,
                    score=round(float(hit.score), 6),
                )
            )
        return results[:top_k]

    def query(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
        user_roles: list[str] | None = None,
        source_types: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve: BM25 + dense, fused with weighted RRF + metadata boost.

        Per-arm pool sizes, RRF k, arm weights and the metadata-boost weight come from
        :class:`FusionConfig`; ``top_k`` overrides the final cut. ``source_types`` applies
        source-level routing to both arms.
        """
        cfg = self.fusion

        bm25_ranked = [
            self._bm25_ids[idx]
            for idx, _ in self._bm25.top_k(query, cfg.sparse_top_k)
            if self._acl_allows(self._bm25_ids[idx], tenant_id, user_roles, source_types)
        ]
        dense_ranked = [
            hit.citation_id
            for hit in self._dense.search(
                query, top_k=cfg.dense_top_k, tenant_id=tenant_id, user_roles=user_roles
            )
            if self._acl_allows(hit.citation_id, tenant_id, user_roles, source_types)
        ]

        fused = reciprocal_rank_fusion(
            [bm25_ranked, dense_ranked], k=cfg.rrf_k,
            weights=[cfg.bm25_weight, cfg.dense_weight],
        )
        query_tokens = set(tokenize(query))
        for doc_id in list(fused):
            fused[doc_id] += self._metadata_boost(doc_id, query_tokens)

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        results: list[RetrievedChunk] = []
        for doc_id, score in ordered[:top_k]:
            chunk = self._by_id.get(doc_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    citation_id=chunk.citation_id,
                    text=chunk.text,
                    source_path=chunk.source_path,
                    metadata=chunk.metadata,
                    score=round(float(score), 6),
                )
            )
        return results
