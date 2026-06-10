"""Independent post-retrieval ACL audit (defense-in-depth + observability).

Tenant/ACL are enforced as a *pre-scoring* hard filter in both retrieval arms. This
module adds a *second*, independent check on the final result set: it re-verifies every
returned chunk against the caller's tenant and roles, drops anything that should not be
visible, and returns structured records of *why* — so a filter bug can never silently
leak a cross-tenant document, and the reason is visible in the trace.
"""

from __future__ import annotations

from job_agent.schemas import RetrievedChunk


def _chunk_roles(chunk: RetrievedChunk) -> set[str]:
    return set((chunk.metadata.get("acl_roles") or "public").split(","))


def audit(
    chunks: list[RetrievedChunk],
    tenant_id: str | None,
    user_roles: list[str] | None,
) -> tuple[list[RetrievedChunk], list[dict[str, str]]]:
    """Re-check ``chunks`` against tenant/ACL; return ``(allowed, denied_records)``.

    ``denied_records`` entries carry ``citation_id`` and a ``reason`` (``tenant_mismatch``
    or ``role_denied``) for tracing.
    """
    roles = set(user_roles or ["public"])
    allowed: list[RetrievedChunk] = []
    denied: list[dict[str, str]] = []
    for chunk in chunks:
        chunk_tenant = chunk.metadata.get("tenant_id", "default")
        if tenant_id is not None and chunk_tenant != tenant_id:
            denied.append({"citation_id": chunk.citation_id, "reason": "tenant_mismatch",
                           "chunk_tenant": chunk_tenant})
            continue
        chunk_roles = _chunk_roles(chunk)
        if "public" not in chunk_roles and not (roles & chunk_roles):
            denied.append({"citation_id": chunk.citation_id, "reason": "role_denied",
                           "chunk_roles": ",".join(sorted(chunk_roles))})
            continue
        allowed.append(chunk)
    return allowed, denied
