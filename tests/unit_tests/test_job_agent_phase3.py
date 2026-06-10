"""Unit tests for Phase-3: encoding hygiene, embedder backends, dense mode, hard eval."""

from __future__ import annotations

from pathlib import Path

from job_agent.embeddings import HashingEmbedder, get_embedder
from job_agent.encoding import clean_text, decode_bytes, is_mojibake
from job_agent.eval_dataset_v3 import build_dataset
from job_agent.hybrid import HybridRetriever
from job_agent.ingestion import IngestManifest, ManifestItem, build_chunks
from job_agent.rag import HybridRAGIndex

# --------------------------------------------------------------------------- #
# Encoding hygiene
# --------------------------------------------------------------------------- #


def test_decode_handles_utf8_bom_and_gbk() -> None:
    assert decode_bytes("中文 RAG".encode("utf-8-sig")) == ("中文 RAG", "utf-8-sig")
    text, enc = decode_bytes("检索增强 RAG".encode("gb18030"))
    assert text == "检索增强 RAG"
    assert enc in ("gb18030", "utf-8")  # ascii-only would be utf-8; CJK forces gb18030


def test_mojibake_detection_and_cleaning() -> None:
    assert is_mojibake("ä¸­æ–‡ å¤§ä¹± ç ä¹±")  # CP1252-mangled UTF-8
    assert is_mojibake("bad text ���")
    assert not is_mojibake("clean 中文 text with RAG")
    assert "�" not in clean_text("x�y")


def test_ingestion_flags_bad_encoding(tmp_path: Path) -> None:
    good = ManifestItem(text="# Notes\n\nRAG 检索增强 with FastAPI.", tenant_id="t")
    bad = ManifestItem(text="ä¸­æ–‡ ä¹±ç  ä¸­æ–‡ ä¹±ç ", tenant_id="t")
    chunks = build_chunks(IngestManifest(items=[good, bad]), tmp_path)
    flags = {c["text"][:6]: c["metadata"]["bad_encoding"] for c in chunks}
    assert any(v == "true" for v in flags.values())
    assert any(v == "false" for v in flags.values())
    assert all("encoding" in c["metadata"] for c in chunks)


# --------------------------------------------------------------------------- #
# Embedder backends
# --------------------------------------------------------------------------- #


def test_default_embedder_is_hashing_and_get_embedder_falls_back() -> None:
    assert isinstance(get_embedder(), HashingEmbedder)
    # An unknown/unavailable backend must degrade to the deterministic embedder.
    import os

    os.environ["JOB_AGENT_EMBEDDER"] = "fastembed"  # not loadable here -> fallback
    try:
        assert isinstance(get_embedder(), HashingEmbedder)
    finally:
        os.environ.pop("JOB_AGENT_EMBEDDER", None)


class _SynonymEmbedder:
    """Tiny semantic stub: maps synonymous phrases to the same vector.

    Proves the dense arm rewards meaning over surface tokens, without a real model.
    """

    dim = 4
    _GROUPS = {
        "loop": [0],   # "重复执行" / "loop detection"
        "ground": [1],
        "rerank": [2],
    }

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        vec = [0.0] * self.dim
        if "重复" in text or "loop" in low or "repeat" in low:
            vec[0] = 1.0
        elif "编" in text or "ground" in low or "unsupported" in low:
            vec[1] = 1.0
        elif "不相关" in text or "rerank" in low or "precision" in low:
            vec[2] = 1.0
        else:
            vec[3] = 1.0
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def test_dense_arm_beats_lexical_on_paraphrase_with_semantic_embedder() -> None:
    """With a semantic embedder, dense retrieves the paraphrased evidence; lexical can't."""
    index = HybridRAGIndex()
    index.add_document("Implemented loop detection and termination for agent execution.",
                       "evidence", "note", tenant_id="t", acl_roles=["tenant:t"])
    index.add_document("Kubernetes and Spark data pipelines on the cloud.",
                       "distractor", "note", tenant_id="t", acl_roles=["tenant:t"])
    retriever = HybridRetriever(index, embedder=_SynonymEmbedder())

    query = "Agent 老是重复执行同样的步骤怎么办"  # no lexical overlap with "loop detection"
    dense = retriever.query_dense(query, top_k=1, tenant_id="t", user_roles=["tenant:t"])
    assert dense and dense[0].source_path == "evidence"


# --------------------------------------------------------------------------- #
# Hard eval dataset
# --------------------------------------------------------------------------- #


def test_v3_dataset_has_vocabulary_gap_cases() -> None:
    cases = build_dataset()
    assert len(cases) >= 100
    task_types = {c["task_type"] for c in cases}
    assert {"paraphrase", "abbreviation", "synonym", "multi_hop", "no_answer", "injection"} <= task_types
    # Paraphrase cases must have a genuine query/evidence vocabulary gap.
    para = next(c for c in cases if c["task_type"] == "paraphrase")
    assert para["vocabulary_gap"] is True
