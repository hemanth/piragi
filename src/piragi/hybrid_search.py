"""Hybrid search combining vector similarity with BM25 keyword matching."""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .types import Citation

logger = logging.getLogger(__name__)


class BM25:
    """
    BM25 (Best Matching 25) implementation for keyword-based retrieval.

    BM25 is a bag-of-words retrieval function that ranks documents based on
    query term frequencies, with saturation and length normalization.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Length normalization parameter (0.75 typical)
            epsilon: Floor for IDF to prevent negative values
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Corpus statistics
        self._corpus_size = 0
        self._avgdl = 0.0
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._idf: Dict[str, float] = {}
        self._doc_lens: List[int] = []
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with lowercasing and basic cleanup."""
        import re
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]

    def fit(self, corpus: List[str]) -> "BM25":
        """
        Fit BM25 on a corpus of documents.

        Args:
            corpus: List of document strings

        Returns:
            Self for chaining
        """
        self._tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self._corpus_size = len(self._tokenized_corpus)

        if self._corpus_size == 0:
            return self

        # Calculate document lengths
        self._doc_lens = [len(doc) for doc in self._tokenized_corpus]
        self._avgdl = sum(self._doc_lens) / self._corpus_size

        # Calculate document frequencies
        self._doc_freqs = defaultdict(int)
        for doc in self._tokenized_corpus:
            seen = set()
            for token in doc:
                if token not in seen:
                    self._doc_freqs[token] += 1
                    seen.add(token)

        # Calculate IDF
        self._idf = {}
        for token, freq in self._doc_freqs.items():
            # IDF with floor to avoid negative values
            idf = math.log((self._corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            self._idf[token] = max(idf, self.epsilon)

        return self

    def score(self, query: str) -> List[float]:
        """
        Score all documents against a query.

        Args:
            query: Query string

        Returns:
            List of BM25 scores for each document
        """
        if self._corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc_tokens in enumerate(self._tokenized_corpus):
            score = 0.0
            doc_len = self._doc_lens[idx]

            # Count term frequencies in document
            doc_tf = defaultdict(int)
            for token in doc_tokens:
                doc_tf[token] += 1

            # BM25 scoring
            for token in query_tokens:
                if token not in self._idf:
                    continue

                tf = doc_tf.get(token, 0)
                idf = self._idf[token]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                score += idf * numerator / denominator

            scores.append(score)

        return scores

    def get_top_k(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Get top-k document indices and scores.

        Args:
            query: Query string
            k: Number of results

        Returns:
            List of (doc_index, score) tuples
        """
        scores = self.score(query)
        if not scores:
            return []

        # Get top-k indices
        indexed_scores = [(i, s) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:k]


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

        self._bm25: Optional[BM25] = None
        self._chunk_texts: List[str] = []
        self._chunk_to_idx: Dict[str, int] = {}

    def index_chunks(self, chunks: List[str]) -> None:
        """
        Index chunks for BM25 search.

        Args:
            chunks: List of chunk text strings
        """
        self._chunk_texts = chunks
        self._chunk_to_idx = {text[:200]: i for i, text in enumerate(chunks)}
        self._bm25 = BM25()
        self._bm25.fit(chunks)
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
        bm25_scores = self._bm25.score(query)

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
        if bm25_scores:
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
