"""Dependency-free BM25 (Okapi) index for lexical retrieval.

Used as the sparse arm of hybrid retrieval. Operates over the same chunk objects as
the dense (Qdrant) arm so the two rankings can be fused with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import math
from collections import Counter

from job_agent.rag import tokenize


class BM25Index:
    """Minimal BM25 Okapi ranking over a fixed corpus of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Create an empty BM25 index."""
        self.k1 = k1
        self.b = b
        self._docs: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def add(self, text: str) -> int:
        """Add a document and return its index."""
        tokens = tokenize(text)
        counts = Counter(tokens)
        self._docs.append(counts)
        self._doc_len.append(len(tokens))
        for term in counts:
            self._df[term] += 1
        self._avgdl = sum(self._doc_len) / len(self._doc_len)
        return len(self._docs) - 1

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # BM25 idf with +1 smoothing to keep it non-negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        """Return a BM25 score for every document against ``query``."""
        if not self._docs:
            return []
        query_terms = tokenize(query)
        results = [0.0] * len(self._docs)
        for idx, counts in enumerate(self._docs):
            dl = self._doc_len[idx] or 1
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            results[idx] = score
        return results

    def top_k(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` for the top ``top_k`` documents."""
        scored = [(idx, score) for idx, score in enumerate(self.scores(query)) if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
