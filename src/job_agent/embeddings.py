"""Pluggable embedding backends with a deterministic offline default.

The default :class:`HashingEmbedder` maps text to a fixed-dimension L2-normalized
vector via token feature hashing. It needs no model download or API key, so dense
retrieval (Qdrant) is fully testable offline and reproducible. When an API/model
embedder is configured it can be swapped behind the same :class:`Embedder` protocol
without touching the vector store or retriever.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol, runtime_checkable

from job_agent.rag import tokenize

DEFAULT_DIM = 256


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (no external dependencies)."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        """Create an embedder producing ``dim``-dimensional unit vectors."""
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self._embed_one(text) for text in texts]


def get_embedder(dim: int = DEFAULT_DIM) -> Embedder:
    """Return the configured embedder.

    Defaults to the deterministic :class:`HashingEmbedder` (offline, no model). When
    ``JOB_AGENT_EMBEDDER`` selects a real backend and its dependency/key is available,
    that is used instead. Any failure degrades gracefully to the hashing embedder.

    Backends:
      * ``hashing`` (default) — deterministic feature hashing.
      * ``fastembed`` — local ONNX model (needs the MSVC runtime on Windows).
      * ``api`` — OpenAI-compatible ``/embeddings`` endpoint (e.g. SiliconFlow bge-m3).
      * ``sentence-transformers`` — local torch model.
    """
    backend = os.environ.get("JOB_AGENT_EMBEDDER", "hashing").lower()
    if backend == "hashing":
        return HashingEmbedder(dim=dim)
    try:  # pragma: no cover - exercised only when a real backend is configured
        if backend == "fastembed":
            return _build_fastembed()
        if backend == "api":
            return _build_api_embedder()
        if backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            model_name = os.environ.get(
                "JOB_AGENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            return _SentenceTransformerEmbedder(SentenceTransformer(model_name))
    except Exception:
        return HashingEmbedder(dim=dim)
    return HashingEmbedder(dim=dim)


def _build_fastembed() -> Embedder:  # pragma: no cover - optional backend
    from fastembed import TextEmbedding

    model_name = os.environ.get("JOB_AGENT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    return _FastEmbedEmbedder(TextEmbedding(model_name=model_name))


def _build_api_embedder() -> Embedder:  # pragma: no cover - needs network + key
    base_url = os.environ.get("JOB_AGENT_EMBED_API_BASE", "https://api.siliconflow.cn/v1")
    api_key = (
        os.environ.get("JOB_AGENT_EMBED_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    model = os.environ.get("JOB_AGENT_EMBED_MODEL", "BAAI/bge-m3")
    if not api_key:
        raise RuntimeError("API embedder requires an embedding API key.")
    return APIEmbedder(base_url=base_url, api_key=api_key, model=model)


class _SentenceTransformerEmbedder:  # pragma: no cover - optional heavy backend
    """Adapter for a sentence-transformers model."""

    def __init__(self, model: object) -> None:
        self._model = model
        self.dim = int(model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)  # type: ignore[attr-defined]
        return [list(map(float, vector)) for vector in vectors]


class _FastEmbedEmbedder:  # pragma: no cover - optional ONNX backend
    """Adapter for a fastembed ONNX model."""

    def __init__(self, model: object) -> None:
        self._model = model
        # Probe the dimension once.
        self.dim = len(next(iter(model.embed(["dimension probe"]))))  # type: ignore[attr-defined]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in self._model.embed(texts)]  # type: ignore[attr-defined]


class APIEmbedder:  # pragma: no cover - needs network + key
    """OpenAI-compatible ``/embeddings`` client (SiliconFlow / OpenAI / compatible)."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        """Configure the embedding endpoint."""
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self.dim = len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch via the OpenAI-compatible endpoint."""
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [list(item.embedding) for item in response.data]
