"""Hybrid search combining vector similarity with BM25 keyword matching."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from .types import Citation

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid search combining dense vector retrieval with sparse BM25.

    This implements the fusion approach where results from both retrievers
    are combined using Reciprocal Rank Fusion (RRF) or weighted scoring.
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        use_rrf: bool = True,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_weight: Weight for vector similarity scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            use_rrf: Use Reciprocal Rank Fusion instead of weighted scoring
            rrf_k: RRF constant (typically 60)
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k

        self._bm25: Optional[BM25Okapi] = None
        self._chunk_texts: List[str] = []
        self._chunk_to_idx: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with lowercasing and basic cleanup."""
        import re
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]

    def index_chunks(self, chunks: List[str]) -> None:
        """
        Index chunks for BM25 search.

        Args:
            chunks: List of chunk text strings
        """
        self._chunk_texts = chunks
        self._chunk_to_idx = {text[:200]: i for i, text in enumerate(chunks)}
        
        tokenized_corpus = [self._tokenize(chunk) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Indexed {len(chunks)} chunks for BM25 search")

    def search(
        self,
        query: str,
        vector_citations: List[Citation],
        top_k: int = 10,
    ) -> List[Citation]:
        """
        Perform hybrid search combining vector results with BM25.

        Args:
            query: Search query
            vector_citations: Citations from vector search (already retrieved)
            top_k: Number of results to return

        Returns:
            Combined and reranked citations
        """
        if not vector_citations:
            return []

        if self._bm25 is None:
            # BM25 not indexed, fall back to vector only
            logger.warning("BM25 not indexed, using vector search only")
            return vector_citations[:top_k]

        # Get BM25 scores for all indexed chunks
        query_tokens = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)

        # Create mapping of chunk text to vector citation
        vector_scores: Dict[str, float] = {}
        citation_map: Dict[str, Citation] = {}

        for citation in vector_citations:
            key = citation.chunk[:200]
            vector_scores[key] = citation.score
            citation_map[key] = citation

        if self.use_rrf:
            # Reciprocal Rank Fusion
            combined_scores = self._rrf_fusion(
                vector_citations,
                bm25_scores,
                citation_map,
            )
        else:
            # Weighted score combination
            combined_scores = self._weighted_fusion(
                vector_citations,
                bm25_scores,
                citation_map,
            )

        # Sort by combined score and return top_k
        sorted_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for key, score in sorted_items[:top_k]:
            if key in citation_map:
                citation = citation_map[key]
                results.append(
                    Citation(
                        source=citation.source,
                        chunk=citation.chunk,
                        score=score,
                        metadata=citation.metadata,
                    )
                )

        return results

    def _rrf_fusion(
        self,
        vector_citations: List[Citation],
        bm25_scores: List[float],
        citation_map: Dict[str, Citation],
    ) -> Dict[str, float]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) for each retriever i

        Args:
            vector_citations: Vector search results
            bm25_scores: BM25 scores for indexed chunks
            citation_map: Mapping of chunk key to citation

        Returns:
            Dict of chunk key to RRF score
        """
        rrf_scores: Dict[str, float] = defaultdict(float)

        # Vector ranks
        for rank, citation in enumerate(vector_citations, 1):
            key = citation.chunk[:200]
            rrf_scores[key] += self.vector_weight / (self.rrf_k + rank)

        # BM25 ranks
        indexed_bm25 = [(i, s) for i, s in enumerate(bm25_scores)]
        indexed_bm25.sort(key=lambda x: x[1], reverse=True)

        for rank, (idx, score) in enumerate(indexed_bm25, 1):
            if idx < len(self._chunk_texts):
                key = self._chunk_texts[idx][:200]
                if key in citation_map:  # Only include if in vector results
                    rrf_scores[key] += self.bm25_weight / (self.rrf_k + rank)

        return dict(rrf_scores)

    def _weighted_fusion(
        self,
        vector_citations: List[Citation],
        bm25_scores: List[float],
        citation_map: Dict[str, Citation],
    ) -> Dict[str, float]:
        """
        Combine results using weighted score fusion.

        Args:
            vector_citations: Vector search results
            bm25_scores: BM25 scores for indexed chunks
            citation_map: Mapping of chunk key to citation

        Returns:
            Dict of chunk key to combined score
        """
        combined: Dict[str, float] = {}

        # Normalize BM25 scores to 0-1
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            normalized_bm25 = [s / max_bm25 for s in bm25_scores]
        else:
            normalized_bm25 = []

        # Combine scores
        for citation in vector_citations:
            key = citation.chunk[:200]
            vector_score = citation.score * self.vector_weight

            # Find BM25 score for this chunk
            bm25_score = 0.0
            if key in self._chunk_to_idx:
                idx = self._chunk_to_idx[key]
                if idx < len(normalized_bm25):
                    bm25_score = normalized_bm25[idx] * self.bm25_weight

            combined[key] = vector_score + bm25_score

        return combined


def create_hybrid_searcher(
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    use_rrf: bool = True,
) -> HybridSearcher:
    """
    Factory function to create a hybrid searcher.

    Args:
        vector_weight: Weight for vector similarity (0-1)
        bm25_weight: Weight for BM25 (0-1)
        use_rrf: Use Reciprocal Rank Fusion

    Returns:
        Configured HybridSearcher
    """
    return HybridSearcher(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        use_rrf=use_rrf,
    )
