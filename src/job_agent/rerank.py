"""Two-layer reranking: deterministic heuristic default + optional cross-encoder.

The heuristic reranker keeps the pipeline offline and reproducible. When
``JOB_AGENT_RERANKER=cross-encoder`` is set and ``sentence-transformers`` is available,
a real cross-encoder is used; any failure degrades back to the heuristic.
"""

from __future__ import annotations

import os
from collections import Counter

from job_agent.rag import tokenize
from job_agent.schemas import RetrievedChunk


def _heuristic_score(query: str, chunk: RetrievedChunk) -> float:
    query_tokens = Counter(tokenize(query))
    chunk_tokens = Counter(tokenize(chunk.text))
    overlap = sum(min(query_tokens[t], chunk_tokens[t]) for t in query_tokens if t in chunk_tokens)
    coverage = len(set(query_tokens) & set(chunk_tokens)) / (len(query_tokens) or 1)
    # Blend retrieval score, lexical overlap and query coverage.
    return 0.5 * chunk.score + 0.3 * coverage + 0.2 * min(overlap / 10.0, 1.0)


def heuristic_rerank(
    query: str, chunks: list[RetrievedChunk], top_k: int | None = None
) -> list[RetrievedChunk]:
    """Rerank by a blend of retrieval score, lexical overlap and coverage."""
    scored = sorted(chunks, key=lambda chunk: _heuristic_score(query, chunk), reverse=True)
    return scored[: top_k or len(scored)]


def rerank(
    query: str, chunks: list[RetrievedChunk], top_k: int | None = None
) -> list[RetrievedChunk]:
    """Rerank using the configured backend, defaulting to the heuristic reranker."""
    backend = os.environ.get("JOB_AGENT_RERANKER", "heuristic").lower()
    if backend == "cross-encoder":  # pragma: no cover - optional heavy backend
        try:
            return _cross_encoder_rerank(query, chunks, top_k)
        except Exception:
            return heuristic_rerank(query, chunks, top_k)
    return heuristic_rerank(query, chunks, top_k)


def _cross_encoder_rerank(  # pragma: no cover - optional heavy backend
    query: str, chunks: list[RetrievedChunk], top_k: int | None
) -> list[RetrievedChunk]:
    from sentence_transformers import CrossEncoder

    model_name = os.environ.get("JOB_AGENT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = CrossEncoder(model_name)
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
    return [chunk for chunk, _ in ranked][: top_k or len(ranked)]
