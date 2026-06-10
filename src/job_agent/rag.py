"""Local hybrid RAG index used by the job application agent."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from job_agent.schemas import RetrievedChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight lexical retrieval."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass
class DocumentChunk:
    """Internal indexed chunk."""

    text: str
    source_path: str
    metadata: dict[str, str]
    citation_id: str
    tokens: Counter[str] = field(default_factory=Counter)


class HybridRAGIndex:
    """Small local hybrid index with Chroma-compatible output fields.

    The implementation is deliberately dependency-light for local tests. If ChromaDB
    and sentence-transformers are installed later, this class can be swapped behind
    the same tool interface without changing the Agent contract.
    """

    def __init__(self) -> None:
        """Create an empty index."""
        self._chunks: list[DocumentChunk] = []

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Return indexed chunks."""
        return self._chunks

    def add_document(
        self,
        text: str,
        source_path: str,
        source_type: str,
        metadata: dict[str, str] | None = None,
        tenant_id: str = "default",
        acl_roles: list[str] | None = None,
    ) -> None:
        """Chunk and index a document with tenant isolation and ACL payload."""
        base_metadata = {
            "source_type": source_type,
            "tenant_id": tenant_id,
            "acl_roles": ",".join(acl_roles or ["public"]),
            **(metadata or {}),
        }
        for chunk_text in chunk_text_by_section(text):
            citation_id = f"C{len(self._chunks) + 1}"
            self._chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source_path=source_path,
                    metadata=base_metadata,
                    citation_id=citation_id,
                    tokens=Counter(tokenize(chunk_text)),
                )
            )

    def ingest_paths(
        self,
        paths: list[Path],
        source_type: str,
        tenant_id: str = "default",
        acl_roles: list[str] | None = None,
    ) -> None:
        """Index a list of text/markdown files."""
        for path in paths:
            if path.exists() and path.is_file():
                self.add_document(
                    path.read_text(encoding="utf-8", errors="ignore"),
                    source_path=str(path),
                    source_type=source_type,
                    tenant_id=tenant_id,
                    acl_roles=acl_roles,
                )

    @staticmethod
    def _acl_ok(chunk: DocumentChunk, user_roles: list[str] | None) -> bool:
        """Post-retrieval ACL check: caller role must intersect chunk ACL (or public)."""
        chunk_roles = set((chunk.metadata.get("acl_roles") or "public").split(","))
        if "public" in chunk_roles:
            return True
        return bool(set(user_roles or ["public"]) & chunk_roles)

    def query(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int = 5,
        tenant_id: str | None = None,
        user_roles: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top chunks using lexical cosine plus keyword overlap.

        Enforces tenant isolation (``tenant_id``) and ACL (``user_roles``) as hard
        payload filters before scoring, mirroring a production vector-store filter.
        """
        filters = filters or {}
        query_tokens = Counter(tokenize(query))
        scored: list[tuple[float, DocumentChunk]] = []

        for chunk in self._chunks:
            if any(chunk.metadata.get(key) != value for key, value in filters.items()):
                continue
            if tenant_id is not None and chunk.metadata.get("tenant_id", "default") != tenant_id:
                continue
            if not self._acl_ok(chunk, user_roles):
                continue
            lexical = _cosine(query_tokens, chunk.tokens)
            overlap = len(set(query_tokens) & set(chunk.tokens))
            score = lexical + 0.05 * overlap
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                citation_id=chunk.citation_id,
                text=chunk.text,
                source_path=chunk.source_path,
                metadata=chunk.metadata,
                score=round(score, 4),
            )
            for score, chunk in scored[:top_k]
        ]


def chunk_text_by_section(text: str, max_chars: int = 900) -> list[str]:
    """Chunk text by headings/paragraphs while keeping chunks compact."""
    raw_parts = re.split(r"\n(?=#{1,6}\s)|\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for raw in raw_parts:
        part = raw.strip()
        if not part:
            continue
        if len(current) + len(part) + 2 <= max_chars:
            current = f"{current}\n\n{part}".strip()
        else:
            if current:
                chunks.append(current)
            if len(part) <= max_chars:
                current = part
            else:
                chunks.extend(part[i : i + max_chars] for i in range(0, len(part), max_chars))
                current = ""
    if current:
        chunks.append(current)
    return chunks
