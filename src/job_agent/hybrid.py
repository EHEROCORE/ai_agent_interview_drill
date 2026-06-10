"""Hybrid retriever: BM25 (sparse) + Qdrant (dense) fused with RRF + metadata boost.

Built on top of an existing :class:`HybridRAGIndex` so it shares the same tenant/ACL
metadata and chunk objects. Acts as a drop-in for the deterministic lexical retriever:
``query()`` returns ``list[RetrievedChunk]`` with tenant isolation and ACL enforced on
both arms.
"""

from __future__ import annotations

from job_agent.bm25 import BM25Index
from job_agent.embeddings import Embedder
from job_agent.rag import HybridRAGIndex, tokenize
from job_agent.schemas import RetrievedChunk
from job_agent.vectorstore import QdrantVectorStore

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K
) -> dict[str, float]:
    """Fuse several ranked id-lists into a single score map via RRF."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    """Sparse+dense hybrid retriever with reciprocal-rank fusion."""

    def __init__(self, index: HybridRAGIndex, embedder: Embedder | None = None) -> None:
        """Build BM25 and Qdrant arms from an indexed :class:`HybridRAGIndex`."""
        self._index = index
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

    def _acl_allows(self, citation_id: str, tenant_id: str | None, user_roles: list[str] | None) -> bool:
        chunk = self._by_id.get(citation_id)
        if chunk is None:
            return False
        if tenant_id is not None and chunk.metadata.get("tenant_id", "default") != tenant_id:
            return False
        return HybridRAGIndex._acl_ok(chunk, user_roles)

    def _metadata_boost(self, citation_id: str, query_tokens: set[str]) -> float:
        chunk = self._by_id.get(citation_id)
        if chunk is None:
            return 0.0
        values = " ".join(str(value) for value in chunk.metadata.values()).lower()
        meta_tokens = set(tokenize(values))
        return 0.02 * len(query_tokens & meta_tokens)

    def query_dense(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
        user_roles: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Dense-only retrieval (the Qdrant arm), for lexical/dense/hybrid A/B."""
        hits = self._dense.search(query, top_k=top_k, tenant_id=tenant_id, user_roles=user_roles)
        results: list[RetrievedChunk] = []
        for hit in hits:
            chunk = self._by_id.get(hit.citation_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    citation_id=chunk.citation_id, text=chunk.text,
                    source_path=chunk.source_path, metadata=chunk.metadata,
                    score=round(float(hit.score), 6),
                )
            )
        return results

    def query(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
        user_roles: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve: BM25 + dense, fused with RRF + metadata boost."""
        pool = max(top_k * 3, top_k)

        bm25_ranked = [
            self._bm25_ids[idx]
            for idx, _ in self._bm25.top_k(query, pool)
            if self._acl_allows(self._bm25_ids[idx], tenant_id, user_roles)
        ]
        dense_ranked = [
            hit.citation_id
            for hit in self._dense.search(query, top_k=pool, tenant_id=tenant_id, user_roles=user_roles)
        ]

        fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked])
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
