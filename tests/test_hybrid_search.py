"""Tests for hybrid search (BM25 + vector) functionality."""

import pytest
from piragi.hybrid_search import BM25, HybridSearcher, create_hybrid_searcher
from piragi.types import Citation


@pytest.fixture
def sample_corpus():
    """Sample document corpus for testing."""
    return [
        "Python is a high-level programming language known for readability.",
        "JavaScript enables interactive web pages and runs in browsers.",
        "Machine learning uses algorithms to learn patterns from data.",
        "Python libraries like NumPy enable scientific computing.",
        "React is a JavaScript library for building user interfaces.",
        "Deep learning is a subset of machine learning using neural networks.",
    ]


@pytest.fixture
def sample_citations(sample_corpus):
    """Create citations from sample corpus."""
    return [
        Citation(
            source=f"doc{i}.txt",
            chunk=text,
            score=0.9 - (i * 0.1),  # Decreasing scores
            metadata={"index": i},
        )
        for i, text in enumerate(sample_corpus)
    ]


class TestBM25:
    """Tests for BM25 implementation."""

    def test_init_defaults(self):
        """Test BM25 initialization with defaults."""
        bm25 = BM25()
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75

    def test_init_custom_params(self):
        """Test BM25 with custom parameters."""
        bm25 = BM25(k1=2.0, b=0.5)
        assert bm25.k1 == 2.0
        assert bm25.b == 0.5

    def test_fit_empty_corpus(self):
        """Test fitting on empty corpus."""
        bm25 = BM25()
        bm25.fit([])
        assert bm25._corpus_size == 0

    def test_fit_corpus(self, sample_corpus):
        """Test fitting on sample corpus."""
        bm25 = BM25()
        bm25.fit(sample_corpus)

        assert bm25._corpus_size == len(sample_corpus)
        assert bm25._avgdl > 0
        assert len(bm25._tokenized_corpus) == len(sample_corpus)

    def test_score_empty_corpus(self):
        """Test scoring with empty corpus."""
        bm25 = BM25()
        bm25.fit([])
        scores = bm25.score("test query")
        assert scores == []

    def test_score_single_doc(self):
        """Test scoring with single document."""
        bm25 = BM25()
        bm25.fit(["Python is a programming language"])
        scores = bm25.score("Python")

        assert len(scores) == 1
        assert scores[0] > 0  # Should have positive score

    def test_score_multiple_docs(self, sample_corpus):
        """Test scoring multiple documents."""
        bm25 = BM25()
        bm25.fit(sample_corpus)
        scores = bm25.score("Python programming")

        assert len(scores) == len(sample_corpus)
        # Python docs should score higher
        python_indices = [0, 3]  # Indices of Python docs
        max_score_idx = scores.index(max(scores))
        assert max_score_idx in python_indices

    def test_score_no_match(self, sample_corpus):
        """Test scoring when query has no matches."""
        bm25 = BM25()
        bm25.fit(sample_corpus)
        scores = bm25.score("xyznonexistent")

        # All scores should be 0 or very low
        assert all(s == 0 for s in scores)

    def test_get_top_k(self, sample_corpus):
        """Test getting top-k results."""
        bm25 = BM25()
        bm25.fit(sample_corpus)
        results = bm25.get_top_k("Python language", k=2)

        assert len(results) == 2
        # Results should be (index, score) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Scores should be descending
        assert results[0][1] >= results[1][1]

    def test_get_top_k_more_than_corpus(self, sample_corpus):
        """Test requesting more results than corpus size."""
        bm25 = BM25()
        bm25.fit(sample_corpus)
        results = bm25.get_top_k("Python", k=100)

        assert len(results) == len(sample_corpus)

    def test_tokenization(self):
        """Test internal tokenization."""
        bm25 = BM25()
        tokens = bm25._tokenize("Hello World! This is a TEST.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Single char tokens should be filtered
        assert "a" not in tokens


class TestHybridSearcher:
    """Tests for hybrid search combining vector and BM25."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        searcher = HybridSearcher()
        assert searcher.vector_weight == 0.5
        assert searcher.bm25_weight == 0.5
        assert searcher.use_rrf is True

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        searcher = HybridSearcher(vector_weight=0.7, bm25_weight=0.3)
        assert searcher.vector_weight == 0.7
        assert searcher.bm25_weight == 0.3

    def test_index_chunks(self, sample_corpus):
        """Test indexing chunks for BM25."""
        searcher = HybridSearcher()
        searcher.index_chunks(sample_corpus)

        assert searcher._bm25 is not None
        assert len(searcher._chunk_texts) == len(sample_corpus)

    def test_search_without_index(self, sample_citations):
        """Test search before indexing falls back to vector only."""
        searcher = HybridSearcher()
        # Don't call index_chunks

        result = searcher.search("Python", sample_citations, top_k=2)

        # Should return vector citations as-is (truncated to top_k)
        assert len(result) <= 2

    def test_search_with_index(self, sample_corpus, sample_citations):
        """Test search with BM25 index."""
        searcher = HybridSearcher()
        searcher.index_chunks(sample_corpus)

        result = searcher.search("Python programming", sample_citations, top_k=3)

        assert len(result) <= 3
        # Results should have combined scores
        for citation in result:
            assert 0 <= citation.score <= 2  # Combined scores can exceed 1

    def test_search_empty_citations(self, sample_corpus):
        """Test search with empty citations."""
        searcher = HybridSearcher()
        searcher.index_chunks(sample_corpus)

        result = searcher.search("Python", [], top_k=3)
        assert result == []

    def test_rrf_fusion(self, sample_corpus, sample_citations):
        """Test Reciprocal Rank Fusion mode."""
        searcher = HybridSearcher(use_rrf=True)
        searcher.index_chunks(sample_corpus)

        result = searcher.search("Python", sample_citations, top_k=2)

        assert len(result) <= 2

    def test_weighted_fusion(self, sample_corpus, sample_citations):
        """Test weighted score fusion mode."""
        searcher = HybridSearcher(use_rrf=False, vector_weight=0.7, bm25_weight=0.3)
        searcher.index_chunks(sample_corpus)

        result = searcher.search("Python", sample_citations, top_k=2)

        assert len(result) <= 2

    def test_search_preserves_metadata(self, sample_corpus, sample_citations):
        """Test that search preserves citation metadata."""
        searcher = HybridSearcher()
        searcher.index_chunks(sample_corpus)

        result = searcher.search("Python", sample_citations, top_k=3)

        for citation in result:
            assert "index" in citation.metadata


class TestCreateHybridSearcher:
    """Tests for the factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        searcher = create_hybrid_searcher()
        assert isinstance(searcher, HybridSearcher)
        assert searcher.vector_weight == 0.5

    def test_create_custom(self):
        """Test creating with custom parameters."""
        searcher = create_hybrid_searcher(
            vector_weight=0.8,
            bm25_weight=0.2,
            use_rrf=False,
        )
        assert searcher.vector_weight == 0.8
        assert searcher.bm25_weight == 0.2
        assert searcher.use_rrf is False
