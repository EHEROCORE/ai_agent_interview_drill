"""Qdrant-backed dense vector store with tenant + ACL payload filtering.

Defaults to Qdrant local mode (``:memory:`` for tests, a local path for persistence),
so no Docker or server is required. Tenant isolation and ACL are enforced as native
Qdrant payload filters (``must`` tenant match + ``should`` ACL role match), mirroring a
production multi-tenant deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from job_agent.embeddings import Embedder, get_embedder

DEFAULT_COLLECTION = "job_agent_knowledge"


@dataclass
class VectorHit:
    """A single dense-retrieval hit."""

    citation_id: str
    text: str
    source_path: str
    metadata: dict[str, str]
    score: float


class QdrantVectorStore:
    """Thin wrapper over a Qdrant collection for the job agent."""

    def __init__(
        self,
        collection: str = DEFAULT_COLLECTION,
        location: str = ":memory:",
        embedder: Embedder | None = None,
        recreate: bool = False,
    ) -> None:
        """Open (or create) a Qdrant collection sized for the embedder."""
        self.embedder = embedder or get_embedder()
        self.collection = collection
        if location == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(path=location)
        self._next_id = 0
        self._ensure_collection(recreate)

    def _ensure_collection(self, recreate: bool) -> None:
        exists = self._client.collection_exists(self.collection)
        if exists and recreate:
            self._client.delete_collection(self.collection)
            exists = False
        if not exists:
            self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=self.embedder.dim, distance=models.Distance.COSINE
                ),
            )
        else:
            self._next_id = self._client.count(self.collection).count

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        """Embed and upsert chunk dicts.

        Each chunk must carry ``citation_id``, ``text``, ``source_path`` and a
        ``metadata`` dict (with ``tenant_id`` / ``acl_roles``). Returns the count
        upserted.
        """
        if not chunks:
            return 0
        vectors = self.embedder.embed([chunk["text"] for chunk in chunks])
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            metadata = chunk.get("metadata", {})
            acl_roles = metadata.get("acl_roles", "public")
            acl_list = acl_roles.split(",") if isinstance(acl_roles, str) else list(acl_roles)
            points.append(
                models.PointStruct(
                    id=self._next_id,
                    vector=vector,
                    payload={
                        "citation_id": chunk["citation_id"],
                        "text": chunk["text"],
                        "source_path": chunk["source_path"],
                        "metadata": metadata,
                        "tenant_id": metadata.get("tenant_id", "default"),
                        "acl_roles": acl_list,
                    },
                )
            )
            self._next_id += 1
        self._client.upsert(self.collection, points=points)
        return len(points)

    def _build_filter(
        self, tenant_id: str | None, user_roles: list[str] | None
    ) -> models.Filter | None:
        must: list[models.Condition] = []
        if tenant_id is not None:
            must.append(
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
            )
        roles = list(user_roles or ["public"])
        if "public" not in roles:
            roles.append("public")
        should: list[models.Condition] = [
            models.FieldCondition(key="acl_roles", match=models.MatchAny(any=roles))
        ]
        if not must and not should:
            return None
        return models.Filter(must=must or None, should=should)

    def search(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
        user_roles: list[str] | None = None,
    ) -> list[VectorHit]:
        """Dense search with tenant/ACL filtering."""
        vector = self.embedder.embed([query])[0]
        result = self._client.query_points(
            self.collection,
            query=vector,
            limit=top_k,
            query_filter=self._build_filter(tenant_id, user_roles),
            with_payload=True,
        )
        hits: list[VectorHit] = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                VectorHit(
                    citation_id=payload.get("citation_id", str(point.id)),
                    text=payload.get("text", ""),
                    source_path=payload.get("source_path", ""),
                    metadata=payload.get("metadata", {}),
                    score=float(point.score),
                )
            )
        return hits

    def count(self) -> int:
        """Return the number of points in the collection."""
        return self._client.count(self.collection).count

    def close(self) -> None:
        """Release the underlying client (frees the local-mode storage lock)."""
        try:
            self._client.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
