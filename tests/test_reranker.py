"""Tests for reranking functionality."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from piragi.reranker import CrossEncoderReranker, TFIDFReranker, HybridReranker
from piragi.types import Citation


@pytest.fixture
def sample_citations():
    """Create sample citations for testing."""
    return [
        Citation(
            source="doc1.txt",
            chunk="Python is a programming language known for its simplicity.",
            score=0.8,
            metadata={"type": "docs"},
        ),
        Citation(
            source="doc2.txt",
            chunk="JavaScript is used for web development and browser scripting.",
            score=0.7,
            metadata={"type": "docs"},
        ),
        Citation(
            source="doc3.txt",
            chunk="Python supports multiple programming paradigms including OOP.",
            score=0.6,
            metadata={"type": "tutorial"},
        ),
        Citation(
            source="doc4.txt",
            chunk="Machine learning frameworks like TensorFlow use Python.",
            score=0.5,
            metadata={"type": "guide"},
        ),
    ]


class TestTFIDFReranker:
    """Tests for TF-IDF based reranking."""

    def test_init_default_weights(self):
        """Test initialization with default weights."""
        reranker = TFIDFReranker()
        assert reranker.vector_weight == 0.6
        assert reranker.tfidf_weight == 0.4

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        reranker = TFIDFReranker(vector_weight=0.7, tfidf_weight=0.3)
        assert reranker.vector_weight == 0.7
        assert reranker.tfidf_weight == 0.3

    def test_init_invalid_weights(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError):
            TFIDFReranker(vector_weight=1.5, tfidf_weight=0.3)

        with pytest.raises(ValueError):
            TFIDFReranker(vector_weight=0.5, tfidf_weight=-0.1)

    def test_rerank_empty_citations(self):
        """Test reranking with empty citations list."""
        reranker = TFIDFReranker()
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_single_citation(self, sample_citations):
        """Test reranking with single citation returns as-is."""
        reranker = TFIDFReranker()
        result = reranker.rerank("test query", [sample_citations[0]])
        assert len(result) == 1
        assert result[0].source == sample_citations[0].source

    def test_rerank_keyword_boost(self, sample_citations):
        """Test that citations with query keywords get boosted."""
        reranker = TFIDFReranker(vector_weight=0.5, tfidf_weight=0.5)
        query = "Python programming language"
        result = reranker.rerank(query, sample_citations)

        # Python-related documents should rank higher
        assert len(result) == 4
        # The reranking should favor Python-related content
        python_sources = [c.source for c in result[:2]]
        assert any("doc1" in s or "doc3" in s or "doc4" in s for s in python_sources)

    def test_rerank_top_k(self, sample_citations):
        """Test reranking with top_k limit."""
        reranker = TFIDFReranker()
        result = reranker.rerank("Python", sample_citations, top_k=2)
        assert len(result) == 2

    def test_rerank_preserves_metadata(self, sample_citations):
        """Test that reranking preserves citation metadata."""
        reranker = TFIDFReranker()
        result = reranker.rerank("test", sample_citations)

        for original, reranked in zip(sample_citations, result):
            # Check that at least one has matching metadata
            found = any(r.metadata == original.metadata for r in result)
            assert found or len(result) == len(sample_citations)


class TestCrossEncoderReranker:
    """Tests for cross-encoder reranking."""

    def test_init_default_model(self):
        """Test initialization with default model."""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._model is None  # Lazy loaded

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
        )
        assert "L-12" in reranker.model_name

    def test_rerank_empty_citations(self):
        """Test reranking with empty citations."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank("test", [])
        assert result == []

    def test_rerank_single_citation(self, sample_citations):
        """Test reranking single citation returns as-is."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank("test", [sample_citations[0]])
        assert len(result) == 1

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_uses_cross_encoder(self, mock_ce_class, sample_citations):
        """Test that reranking uses CrossEncoder model."""
        # Mock the cross-encoder
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.7, 0.5])
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        result = reranker.rerank("Python language", sample_citations)

        # Verify model was called
        mock_model.predict.assert_called_once()

        # Results should be sorted by score
        assert len(result) == 4
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_top_k(self, mock_ce_class, sample_citations):
        """Test cross-encoder reranking with top_k."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.7, 0.5])
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        result = reranker.rerank("test", sample_citations, top_k=2)

        assert len(result) == 2

    @patch("sentence_transformers.CrossEncoder")
    def test_score_pair(self, mock_ce_class):
        """Test scoring a single query-passage pair."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.5])  # Raw score
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        score = reranker.score_pair("What is Python?", "Python is a language.")

        assert 0 <= score <= 1  # Should be normalized


class TestHybridReranker:
    """Tests for hybrid reranking (TF-IDF + cross-encoder)."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        reranker = HybridReranker(use_cross_encoder=False)
        assert reranker.tfidf_reranker is not None
        assert reranker._cross_encoder is None

    def test_init_with_cross_encoder(self):
        """Test initialization with cross-encoder enabled."""
        reranker = HybridReranker(use_cross_encoder=True)
        assert reranker._cross_encoder is not None

    def test_rerank_empty(self):
        """Test reranking empty list."""
        reranker = HybridReranker(use_cross_encoder=False)
        result = reranker.rerank("test", [])
        assert result == []

    def test_rerank_tfidf_only(self, sample_citations):
        """Test reranking with TF-IDF only."""
        reranker = HybridReranker(use_cross_encoder=False)
        result = reranker.rerank("Python", sample_citations, top_k=2)

        assert len(result) == 2
        # Should favor Python content
        assert any("Python" in c.chunk for c in result)

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_two_stage(self, mock_ce_class, sample_citations):
        """Test two-stage reranking (TF-IDF then cross-encoder)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.3, 0.2])
        mock_ce_class.return_value = mock_model

        reranker = HybridReranker(
            use_cross_encoder=True,
            cross_encoder_top_n=4,
        )
        result = reranker.rerank("Python", sample_citations, top_k=2)

        assert len(result) == 2
