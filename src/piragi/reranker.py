"""Cross-encoder reranking for improved retrieval accuracy."""

import logging
from typing import List, Optional, Tuple

from .types import Citation

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers cross-encoder models.

    Cross-encoders jointly encode (query, passage) pairs and provide much more
    accurate relevance scores than bi-encoder similarities, at the cost of
    being slower (can't pre-compute passage embeddings).

    Best used as a second-stage reranker: retrieve top-N with bi-encoder,
    then rerank to top-K with cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Cross-encoder model to use. Good options:
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, good quality)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (slower, better quality)
                - "BAAI/bge-reranker-base" (good multilingual support)
                - "BAAI/bge-reranker-large" (best quality, slowest)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    device=self._device,
                )
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        citations: List[Citation],
        top_k: Optional[int] = None,
    ) -> List[Citation]:
        """
        Rerank citations using cross-encoder scoring.

        Args:
            query: The search query
            citations: List of citations to rerank
            top_k: Number of top results to return (None = return all)

        Returns:
            Reranked citations with updated scores
        """
        if not citations:
            return citations

        if len(citations) == 1:
            return citations

        model = self._load_model()

        # Prepare query-passage pairs
        pairs = [(query, citation.chunk) for citation in citations]

        # Score all pairs
        scores = model.predict(pairs, batch_size=self.batch_size)

        # Normalize scores to 0-1 range using sigmoid
        import numpy as np
        normalized_scores = 1 / (1 + np.exp(-scores))

        # Create new citations with updated scores
        scored_citations = []
        for citation, score in zip(citations, normalized_scores):
            new_citation = Citation(
                source=citation.source,
                chunk=citation.chunk,
                score=float(score),
                metadata=citation.metadata,
            )
            scored_citations.append(new_citation)

        # Sort by score descending
        scored_citations.sort(key=lambda c: c.score, reverse=True)

        # Return top_k if specified
        if top_k is not None:
            return scored_citations[:top_k]

        return scored_citations

    def score_pair(self, query: str, passage: str) -> float:
        """
        Score a single query-passage pair.

        Args:
            query: The search query
            passage: The passage to score

        Returns:
            Relevance score (higher = more relevant)
        """
        model = self._load_model()
        score = model.predict([(query, passage)])[0]

        # Normalize to 0-1
        import numpy as np
        return float(1 / (1 + np.exp(-score)))


class TFIDFReranker:
    """
    TF-IDF based reranker for fast keyword-aware reranking.

    Useful as a lightweight alternative to cross-encoder when speed is critical.
    Combines vector similarity with TF-IDF keyword matching.
    """

    def __init__(self, vector_weight: float = 0.6, tfidf_weight: float = 0.4):
        """
        Initialize TF-IDF reranker.

        Args:
            vector_weight: Weight for original vector similarity score (0-1)
            tfidf_weight: Weight for TF-IDF score (0-1)
        """
        if not (0 <= vector_weight <= 1) or not (0 <= tfidf_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        self.vector_weight = vector_weight
        self.tfidf_weight = tfidf_weight
        self._vectorizer = None
        self._fitted = False

    def _get_vectorizer(self):
        """Get or create TF-IDF vectorizer."""
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),  # Unigrams and bigrams
                max_features=10000,
            )
        return self._vectorizer

    def rerank(
        self,
        query: str,
        citations: List[Citation],
        top_k: Optional[int] = None,
    ) -> List[Citation]:
        """
        Rerank citations using TF-IDF scoring combined with vector scores.

        Args:
            query: The search query
            citations: List of citations to rerank
            top_k: Number of top results to return (None = return all)

        Returns:
            Reranked citations with combined scores
        """
        if not citations:
            return citations

        if len(citations) == 1:
            return citations

        vectorizer = self._get_vectorizer()

        # Fit vectorizer on all chunks
        chunks = [c.chunk for c in citations]
        try:
            tfidf_matrix = vectorizer.fit_transform(chunks)
            query_vector = vectorizer.transform([query])

            # Compute cosine similarity between query and each chunk
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

        except Exception as e:
            logger.warning(f"TF-IDF scoring failed: {e}, using vector scores only")
            tfidf_scores = [0.0] * len(citations)

        # Combine scores
        scored_citations = []
        for citation, tfidf_score in zip(citations, tfidf_scores):
            combined_score = (
                self.vector_weight * citation.score +
                self.tfidf_weight * float(tfidf_score)
            )
            new_citation = Citation(
                source=citation.source,
                chunk=citation.chunk,
                score=combined_score,
                metadata=citation.metadata,
            )
            scored_citations.append(new_citation)

        # Sort by combined score
        scored_citations.sort(key=lambda c: c.score, reverse=True)

        if top_k is not None:
            return scored_citations[:top_k]

        return scored_citations


class HybridReranker:
    """
    Hybrid reranker that combines multiple reranking strategies.

    Uses TF-IDF for initial reranking, then optionally applies
    cross-encoder for final reranking of top candidates.
    """

    def __init__(
        self,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_top_n: int = 20,
        tfidf_vector_weight: float = 0.6,
        device: Optional[str] = None,
    ):
        """
        Initialize hybrid reranker.

        Args:
            use_cross_encoder: Whether to use cross-encoder as final stage
            cross_encoder_model: Cross-encoder model name
            cross_encoder_top_n: Number of candidates to pass to cross-encoder
            tfidf_vector_weight: Weight for vector vs TF-IDF in first stage
            device: Device for cross-encoder
        """
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_top_n = cross_encoder_top_n

        # Initialize TF-IDF reranker
        self.tfidf_reranker = TFIDFReranker(
            vector_weight=tfidf_vector_weight,
            tfidf_weight=1.0 - tfidf_vector_weight,
        )

        # Initialize cross-encoder (lazy loaded)
        self._cross_encoder = None
        if use_cross_encoder:
            self._cross_encoder = CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=device,
            )

    def rerank(
        self,
        query: str,
        citations: List[Citation],
        top_k: Optional[int] = None,
    ) -> List[Citation]:
        """
        Rerank using hybrid approach.

        Args:
            query: The search query
            citations: List of citations to rerank
            top_k: Number of top results to return

        Returns:
            Reranked citations
        """
        if not citations:
            return citations

        # Stage 1: TF-IDF + vector reranking
        stage1_results = self.tfidf_reranker.rerank(
            query,
            citations,
            top_k=self.cross_encoder_top_n if self.use_cross_encoder else top_k,
        )

        # Stage 2: Cross-encoder reranking (if enabled)
        if self.use_cross_encoder and self._cross_encoder:
            final_results = self._cross_encoder.rerank(
                query,
                stage1_results,
                top_k=top_k,
            )
            return final_results

        return stage1_results[:top_k] if top_k else stage1_results
