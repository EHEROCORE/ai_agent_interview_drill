"""Pluggable embedding backends with a deterministic offline default.

The default :class:`HashingEmbedder` maps text to a fixed-dimension L2-normalized
vector via token feature hashing. It needs no model download or API key, so dense
retrieval (Qdrant) is fully testable offline and reproducible. When an API/model
embedder is configured it can be swapped behind the same :class:`Embedder` protocol
without touching the vector store or retriever.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

from job_agent.rag import tokenize

DEFAULT_DIM = 256


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface."""

    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (no external dependencies)."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        """Create an embedder producing ``dim``-dimensional unit vectors."""
        self.dim = dim
        self.name = f"hashing-{dim}"

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
        inner: Embedder = HashingEmbedder(dim=dim)
        return _maybe_cache(inner, default_on=False)
    try:  # pragma: no cover - exercised only when a real backend is configured
        if backend == "fastembed":
            return _maybe_cache(_build_fastembed())
        if backend == "api":
            return _maybe_cache(_build_api_embedder())
        if backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            model_name = os.environ.get(
                "JOB_AGENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            return _maybe_cache(_SentenceTransformerEmbedder(SentenceTransformer(model_name)))
    except Exception:
        return HashingEmbedder(dim=dim)
    return HashingEmbedder(dim=dim)


def _maybe_cache(inner: Embedder, default_on: bool = True) -> Embedder:
    """Wrap ``inner`` in a persistent vector cache unless explicitly disabled.

    Real (paid/slow) backends are cached by default so repeated/rebuilt retrievers reuse
    embeddings instead of re-calling the API; the cheap hashing embedder is not cached
    unless ``JOB_AGENT_EMBED_CACHE=on``.
    """
    setting = os.environ.get("JOB_AGENT_EMBED_CACHE", "").lower()
    enabled = setting in ("1", "on", "true") if setting else default_on
    if not enabled:
        return inner
    path = os.environ.get("JOB_AGENT_EMBED_CACHE_PATH", str(Path("reports") / "embedding_cache.sqlite"))
    return CachingEmbedder(inner, EmbeddingCache(path))


def _build_fastembed() -> Embedder:  # pragma: no cover - optional backend
    from fastembed import TextEmbedding

    model_name = os.environ.get("JOB_AGENT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    return _FastEmbedEmbedder(TextEmbedding(model_name=model_name))


def _build_api_embedder() -> Embedder:  # pragma: no cover - needs network + key
    base_url = os.environ.get("JOB_AGENT_EMBED_API_BASE", "https://api.siliconflow.com/v1")
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
        self.name = f"st:{getattr(model, 'model_card_data', type(model).__name__)}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)  # type: ignore[attr-defined]
        return [list(map(float, vector)) for vector in vectors]


class _FastEmbedEmbedder:  # pragma: no cover - optional ONNX backend
    """Adapter for a fastembed ONNX model."""

    def __init__(self, model: object, name: str = "fastembed") -> None:
        self._model = model
        self.name = name
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
        self.name = f"api:{model}"
        self.dim = len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch via the OpenAI-compatible endpoint."""
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [list(item.embedding) for item in response.data]


class EmbeddingCache:
    """SQLite-backed cache mapping ``(model_name, text)`` to an embedding vector."""

    def __init__(self, path: str | Path) -> None:
        """Open (and migrate) the cache database."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
            )

    @staticmethod
    def _key(model: str, text: str) -> str:
        return hashlib.sha1(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> dict[str, list[float]]:
        """Return cached vectors for the subset of ``texts`` that are present."""
        keys = {self._key(model, t): t for t in texts}
        out: dict[str, list[float]] = {}
        if not keys:
            return out
        placeholders = ",".join("?" * len(keys))
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                list(keys),
            ).fetchall()
        for key, vector_json in rows:
            out[keys[key]] = json.loads(vector_json)
        return out

    def set_many(self, model: str, mapping: dict[str, list[float]]) -> None:
        """Persist ``{text: vector}`` for ``model``."""
        if not mapping:
            return
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [(self._key(model, text), json.dumps(vector)) for text, vector in mapping.items()],
            )


class CachingEmbedder:
    """Wrap any :class:`Embedder` with a persistent per-text vector cache.

    Only cache misses hit the inner embedder, so rebuilding the retriever (or repeated
    runs / eval sweeps) reuses embeddings instead of re-calling a slow/paid backend —
    the fix for "retrieval cache hit but embedding still recomputed".
    """

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        """Wrap ``inner`` behind ``cache``."""
        self._inner = inner
        self._cache = cache
        self.dim = inner.dim
        self.name = inner.name
        self.inner_calls = 0  # number of texts actually sent to the inner embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, serving hits from cache and persisting misses."""
        cached = self._cache.get_many(self.name, texts)
        misses = [t for t in texts if t not in cached]
        if misses:
            self.inner_calls += len(misses)
            fresh = dict(zip(misses, self._inner.embed(misses)))
            self._cache.set_many(self.name, fresh)
            cached.update(fresh)
        return [cached[t] for t in texts]
