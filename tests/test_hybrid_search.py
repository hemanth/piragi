"""Tests for hybrid search (BM25 + vector) functionality."""

import pytest
from piragi.hybrid_search import HybridSearcher, BM25Index, create_hybrid_searcher
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


class TestBM25Index:
    """Tests for the standalone BM25-only index (no vector store involved)."""

    def _citations(self, sample_corpus):
        return [
            Citation(source=f"doc{i}.txt", chunk=text, score=0.0, metadata={"index": i})
            for i, text in enumerate(sample_corpus)
        ]

    def test_search_before_indexing_returns_empty(self):
        """Test searching an empty index returns no results instead of erroring."""
        index = BM25Index()
        assert index.search("Python") == []
        assert index.count() == 0

    def test_index_and_search_returns_relevant_chunks(self, sample_corpus):
        """Test that indexing then searching surfaces the matching chunk(s) first."""
        index = BM25Index()
        index.index_chunks(self._citations(sample_corpus))

        results = index.search("Python programming", top_k=2)

        assert len(results) <= 2
        assert results[0].source == "doc0.txt"

    def test_search_scores_normalized_to_max_one(self, sample_corpus):
        """Test scores are normalized against the top score (article's BM25 recipe assumes 0-1 scores)."""
        index = BM25Index()
        index.index_chunks(self._citations(sample_corpus))

        results = index.search("Python", top_k=6)

        assert results[0].score == pytest.approx(1.0)
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_index_chunks_is_additive(self, sample_corpus):
        """Test repeated index_chunks() calls accumulate rather than overwrite."""
        index = BM25Index()
        index.index_chunks(self._citations(sample_corpus[:2]))
        index.index_chunks(self._citations(sample_corpus[2:]))

        assert index.count() == len(sample_corpus)

    def test_delete_by_source_removes_matching_chunks(self, sample_corpus):
        """Test delete_by_source() drops only chunks from that source, matching store's contract."""
        index = BM25Index()
        index.index_chunks(self._citations(sample_corpus))

        deleted = index.delete_by_source("doc0.txt")

        assert deleted == 1
        assert index.count() == len(sample_corpus) - 1
        assert all(c.source != "doc0.txt" for c in index._chunks)

    def test_delete_by_source_missing_source_is_noop(self, sample_corpus):
        """Test delete_by_source() for an unknown source deletes nothing."""
        index = BM25Index()
        index.index_chunks(self._citations(sample_corpus))

        deleted = index.delete_by_source("nonexistent.txt")

        assert deleted == 0
        assert index.count() == len(sample_corpus)
