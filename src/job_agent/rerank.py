"""Reranking backends: deterministic heuristic default + optional real rerankers.

``JOB_AGENT_RERANKER`` selects the backend; all degrade to the heuristic on failure:

* ``heuristic`` (default) — offline blend of retrieval score / overlap / coverage.
* ``none`` — passthrough (keep fusion order); the rerank A/B baseline.
* ``api`` — hosted cross-encoder via an OpenAI-style ``/rerank`` endpoint
  (e.g. SiliconFlow ``Qwen/Qwen3-Reranker-8B``) — runs on an API key, no torch.
* ``cross-encoder`` — local ``sentence-transformers`` CrossEncoder (needs torch).
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
    if backend == "none":
        return chunks[: top_k or len(chunks)]
    if backend == "api":  # pragma: no cover - needs network + key
        try:
            return _api_rerank(query, chunks, top_k)
        except Exception:
            return heuristic_rerank(query, chunks, top_k)
    if backend == "cross-encoder":  # pragma: no cover - optional heavy backend
        try:
            return _cross_encoder_rerank(query, chunks, top_k)
        except Exception:
            return heuristic_rerank(query, chunks, top_k)
    return heuristic_rerank(query, chunks, top_k)


def _rerank_api_key() -> str:
    return (
        os.environ.get("JOB_AGENT_RERANK_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )


def _api_rerank(  # pragma: no cover - needs network + key
    query: str, chunks: list[RetrievedChunk], top_k: int | None
) -> list[RetrievedChunk]:
    """Rerank via a hosted cross-encoder ``/rerank`` endpoint (SiliconFlow-compatible)."""
    import httpx

    api_key = _rerank_api_key()
    if not api_key:
        raise RuntimeError("API reranker requires a rerank API key.")
    base = os.environ.get("JOB_AGENT_RERANK_API_BASE", "https://api.siliconflow.com/v1")
    model = os.environ.get("JOB_AGENT_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
    documents = [chunk.text for chunk in chunks]
    response = httpx.post(
        f"{base}/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "query": query, "documents": documents,
              "top_n": top_k or len(documents)},
        timeout=30.0,
    )
    response.raise_for_status()
    results = response.json()["results"]
    ranked = sorted(results, key=lambda r: float(r["relevance_score"]), reverse=True)
    return [chunks[r["index"]] for r in ranked][: top_k or len(chunks)]


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
