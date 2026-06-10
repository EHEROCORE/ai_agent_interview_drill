"""Knowledge ingestion pipeline into a persistent Qdrant collection.

Flow: manifest -> parse/clean -> structure-aware chunking -> metadata enrichment ->
embedding -> Qdrant upsert. Each chunk gets a unified payload (tenant_id, doc_type,
source_path, section, topic, company, role, chunk_id, updated_at, acl_roles), matching
the retrieval/filter contract.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.basemodel import AgentBaseModel
from job_agent.embeddings import get_embedder
from job_agent.encoding import clean_text, decode_bytes, is_mojibake
from job_agent.rag import chunk_text_by_section
from job_agent.vectorstore import DEFAULT_COLLECTION, QdrantVectorStore

logger = logging.getLogger("job_agent.ingestion")


class ManifestItem(AgentBaseModel):
    """A single document to ingest."""

    source_path: str | None = None
    text: str | None = None
    doc_type: str = "note"
    tenant_id: str = "default"
    company: str = ""
    role: str = ""
    topic: str = ""
    acl_roles: list[str] | None = None


class IngestManifest(AgentBaseModel):
    """A batch of documents to ingest into a collection."""

    collection: str = DEFAULT_COLLECTION
    items: list[ManifestItem]


def _section_of(chunk: str) -> str:
    match = re.search(r"#{1,6}\s*(.+)", chunk)
    if match:
        return match.group(1).strip()[:80]
    return chunk.split("\n", 1)[0][:80]


def _load_text(item: ManifestItem, root: Path) -> tuple[str, str]:
    """Return ``(decoded_text, detected_encoding)`` for a manifest item."""
    if item.text:
        return item.text, "inline"
    if not item.source_path:
        raise ValueError("Manifest item needs text or source_path.")
    path = Path(item.source_path)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    allowed = [root.resolve(), root.parent.resolve()]
    if not any(str(resolved).startswith(str(a)) for a in allowed):
        raise ValueError("source_path is outside the allowed workspace.")
    if resolved.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Only .txt/.md sources are supported.")
    return decode_bytes(resolved.read_bytes())


def build_chunks(manifest: IngestManifest, root: Path) -> list[dict[str, Any]]:
    """Parse, clean, chunk and enrich manifest items into upsert-ready chunk dicts.

    Source bytes are decoded robustly (UTF-8 BOM / UTF-8 / GB18030 / latin-1) and each
    chunk records its ``encoding`` and a ``bad_encoding`` flag (mojibake detection) so
    low-quality sources are visible downstream rather than silently indexed.
    """
    now = datetime.now(UTC).isoformat()
    chunks: list[dict[str, Any]] = []
    counter = 0
    for item in manifest.items:
        raw, encoding = _load_text(item, root)
        bad = is_mojibake(raw)
        if bad:
            logger.warning("Possible mojibake in source %s (encoding=%s)",
                           item.source_path or item.topic, encoding)
        text = clean_text(raw)
        acl_roles = item.acl_roles or [f"tenant:{item.tenant_id}"]
        for piece in chunk_text_by_section(text):
            counter += 1
            chunk_id = f"K{counter}"
            chunks.append(
                {
                    "citation_id": chunk_id,
                    "text": piece,
                    "source_path": item.source_path or f"inline:{item.topic or item.doc_type}",
                    "metadata": {
                        "tenant_id": item.tenant_id,
                        "doc_type": item.doc_type,
                        "source_path": item.source_path or "inline",
                        "section": _section_of(piece),
                        "topic": item.topic,
                        "company": item.company,
                        "role": item.role,
                        "chunk_id": chunk_id,
                        "updated_at": now,
                        "encoding": encoding,
                        "bad_encoding": "true" if bad else "false",
                        "acl_roles": ",".join(acl_roles),
                    },
                }
            )
    return chunks


def ingest(
    manifest: IngestManifest,
    location: str,
    root: Path,
    recreate: bool = False,
) -> dict[str, Any]:
    """Run the full ingestion pipeline and upsert into a persistent Qdrant collection."""
    chunks = build_chunks(manifest, root)
    store = QdrantVectorStore(
        collection=manifest.collection, location=location,
        embedder=get_embedder(), recreate=recreate,
    )
    upserted = store.upsert(chunks)
    result = {
        "collection": manifest.collection,
        "documents": len(manifest.items),
        "chunks_upserted": upserted,
        "total_points": store.count(),
    }
    store.close()
    return result
