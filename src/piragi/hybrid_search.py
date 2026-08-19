"""Hybrid search combining vector similarity with BM25 keyword matching.

BM25 runs over the FULL indexed corpus and can surface documents the dense
retriever missed. Vector and BM25 candidate sets are fused as a UNION via
Reciprocal Rank Fusion (or weighted scoring), not intersected.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from rank_bm25 import BM25Okapi

from .types import Citation

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b")


def _record_from(item: Union[str, Dict[str, Any], Citation]) -> Dict[str, Any]:
    """Normalize an index item to {text, source, metadata}."""
    if isinstance(item, str):
        return {"text": item, "source": None, "metadata": {}}
    if isinstance(item, dict):
        return {
            "text": item.get("text", ""),
            "source": item.get("source"),
            "metadata": item.get("metadata") or {},
        }
    # Citation-like
    return {
        "text": getattr(item, "chunk", ""),
        "source": getattr(item, "source", None),
        "metadata": getattr(item, "metadata", None) or {},
    }


class HybridSearcher:
    """Hybrid search combining dense vector retrieval with sparse BM25 via RRF."""

    def __init__(
        self,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        use_rrf: bool = True,
        rrf_k: int = 60,
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k

        self._bm25: Optional[BM25Okapi] = None
        self._records: List[Dict[str, Any]] = []
        self._chunk_texts: List[str] = []
        self._keys: List[str] = []
        self._key_to_idx: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1]

    @staticmethod
    def _key(source: Optional[str], text: str) -> str:
        """Stable key per chunk = full chunk text. Avoids the chunk[:200] prefix
        collisions of the old implementation; a vector hit and its own corpus
        record share text and therefore merge (no duplicate entries)."""
        return text or ""

    def index_chunks(self, chunks: List[Union[str, Dict[str, Any], Citation]]) -> None:
        """Index chunks for BM25 search.

        Accepts plain strings (back-compat), dicts with text/source/metadata,
        or Citation objects. Records are retained so BM25-only hits can be
        materialized into citations during fusion.
        """
        self._records = [_record_from(c) for c in chunks]
        self._chunk_texts = [r["text"] for r in self._records]
        self._keys = [self._key(r["source"], r["text"]) for r in self._records]
        # last-writer-wins on exact dup key; fine for fusion lookups
        self._key_to_idx = {k: i for i, k in enumerate(self._keys)}

        tokenized_corpus = [self._tokenize(t) for t in self._chunk_texts]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        logger.info("Indexed %d chunks for BM25 search", len(self._chunk_texts))

    def search(
        self,
        query: str,
        vector_citations: List[Citation],
        top_k: int = 10,
        candidate_pool: Optional[int] = None,
    ) -> List[Citation]:
        """Fuse vector results with an independent BM25 search over the full corpus.

        Args:
            query: Search query
            vector_citations: Candidates from the dense retriever
            top_k: Number of fused results to return
            candidate_pool: How many BM25 top hits to consider (default max(top_k*4, 50))
        """
        if not vector_citations:
            return []

        if self._bm25 is None:
            return vector_citations[:top_k]

        pool = candidate_pool or max(top_k * 4, 50)

        bm25_scores = self._bm25.get_scores(self._tokenize(query))
        top_bm25 = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:pool]

        # Build the candidate union, keyed by stable chunk key.
        citation_map: Dict[str, Citation] = {}
        vec_rank: Dict[str, int] = {}
        for rank, c in enumerate(vector_citations, 1):
            k = self._key(c.source, c.chunk)
            if k not in vec_rank:
                vec_rank[k] = rank
                citation_map[k] = c

        bm_rank: Dict[str, int] = {}
        bm_norm: Dict[str, float] = {}
        max_bm25 = max((bm25_scores[i] for i in top_bm25), default=0.0) or 1.0
        for rank, idx in enumerate(top_bm25, 1):
            if bm25_scores[idx] <= 0:
                continue
            k = self._keys[idx]
            if k not in bm_rank:
                bm_rank[k] = rank
                bm_norm[k] = bm25_scores[idx] / max_bm25
            if k not in citation_map:
                rec = self._records[idx]
                citation_map[k] = Citation(
                    source=rec["source"],
                    chunk=rec["text"],
                    score=0.0,
                    metadata=rec["metadata"],
                )

        if self.use_rrf:
            scores = self._rrf_scores(vec_rank, bm_rank)
        else:
            scores = self._weighted_scores(citation_map, vec_rank, bm_norm)

        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
        out: List[Citation] = []
        for k in ranked:
            c = citation_map[k]
            out.append(
                Citation(source=c.source, chunk=c.chunk, score=scores[k], metadata=c.metadata)
            )
        return out

    def _rrf_scores(
        self, vec_rank: Dict[str, int], bm_rank: Dict[str, int]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for k, rank in vec_rank.items():
            scores[k] += self.vector_weight / (self.rrf_k + rank)
        for k, rank in bm_rank.items():
            scores[k] += self.bm25_weight / (self.rrf_k + rank)
        return dict(scores)

    def _weighted_scores(
        self,
        citation_map: Dict[str, Citation],
        vec_rank: Dict[str, int],
        bm_norm: Dict[str, float],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for k, c in citation_map.items():
            v = (c.score if k in vec_rank else 0.0) * self.vector_weight
            b = bm_norm.get(k, 0.0) * self.bm25_weight
            scores[k] = v + b
        return scores


def create_hybrid_searcher(
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    use_rrf: bool = True,
) -> HybridSearcher:
    """Factory function to create a hybrid searcher."""
    return HybridSearcher(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        use_rrf=use_rrf,
    )
