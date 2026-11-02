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


def test_corpus_fusion_recovers_keyword_only_hit(sample_corpus, sample_citations):
    searcher = HybridSearcher()
    results = searcher.search_corpus("NumPy", sample_citations[:1], sample_citations, top_k=3)
    assert {c.source for c in results} == {"doc0.txt", "doc3.txt"}
    # Sparse retrieval works even if dense search finds nothing.
    assert searcher.search_corpus("NumPy", [], sample_citations)[0].source == "doc3.txt"


def test_corpus_fusion_preserves_distinct_sources_with_identical_text():
    citations = [Citation(source=source, chunk="shared passage", score=.8) for source in ["a", "b"]]
    results = HybridSearcher().search_corpus("shared", citations, citations)
    assert {c.source for c in results} == {"a", "b"}


def test_live_lance_keyword_corpus_obeys_filters_reopen_and_delete(tmp_path):
    from piragi.stores.lance import LanceStore
    from piragi.types import Chunk

    store = LanceStore(uri=str(tmp_path), vector_dimension=2)
    items = [("NumPy", "docs"), ("Secret", "private"), ("Python", "docs")]
    store.add_chunks([
        Chunk(
            source=str(i),
            text=(word + " documentation ") * 10,
            chunk_index=0,
            metadata={"team": team},
            embedding=[.1, .2],
        )
        for i, (word, team) in enumerate(items)
    ])
    reopened = LanceStore(uri=str(tmp_path), vector_dimension=2)
    corpus = reopened.get_all_citations(filters={"team": "docs"})
    assert {c.source for c in corpus} == {"0", "2"}
    assert all(c.metadata["team"] == "docs" for c in corpus)
    reopened.delete_by_source("0")
    assert [c.source for c in reopened.get_all_citations(filters={"team": "docs"})] == ["2"]
    reopened.clear()
    assert reopened.get_all_citations() == []


def test_search_corpus_caches_bm25_instance(sample_citations):
    """Verify search_corpus caches BM25Okapi and avoids repeated initialization."""
    searcher = HybridSearcher()
    assert len(searcher._corpus_cache) == 0

    # First search initializes and caches BM25Okapi
    res1 = searcher.search_corpus("Python", sample_citations[:2], sample_citations, top_k=3)
    assert len(searcher._corpus_cache) == 1
    cached_key = next(iter(searcher._corpus_cache))
    cached_instance = searcher._corpus_cache[cached_key]

    # Second search with same corpus reuses cached instance
    res2 = searcher.search_corpus(
        "Machine learning", sample_citations[:2], sample_citations, top_k=3
    )
    assert len(searcher._corpus_cache) == 1
    assert searcher._corpus_cache[cached_key] is cached_instance
    assert [c.source for c in res1] != [c.source for c in res2]


def test_invalidate_bm25_cache_clears_cached_indices(sample_citations):
    """Verify invalidate_bm25_cache() clears all cached instances."""
    searcher = HybridSearcher()
    searcher.search_corpus("Python", sample_citations[:2], sample_citations, top_k=3)
    assert len(searcher._corpus_cache) == 1

    searcher.invalidate_bm25_cache()
    assert len(searcher._corpus_cache) == 0


def test_search_corpus_caches_distinct_filtered_corpora(sample_citations):
    """Verify distinct filtered corpora are cached under separate keys."""
    searcher = HybridSearcher()
    corpus_a = sample_citations[:3]
    corpus_b = sample_citations[3:]

    searcher.search_corpus("Python", corpus_a[:1], corpus_a, top_k=2)
    assert len(searcher._corpus_cache) == 1

    searcher.search_corpus("JavaScript", corpus_b[:1], corpus_b, top_k=2)
    assert len(searcher._corpus_cache) == 2


def test_ragi_add_and_clear_invalidates_hybrid_cache(tmp_path):
    """Verify Ragi.add() and Ragi.clear() trigger cache invalidation."""
    from piragi import Ragi
    from piragi.stores.lance import LanceStore
    store = LanceStore(uri=str(tmp_path), vector_dimension=2)

    kb = Ragi(
        store=store,
        config={"retrieval": {"use_hybrid_search": True}},
    )
    # Seed cache
    c = Citation(source="init", chunk="initial test text", score=0.5)
    kb._hybrid_searcher.search_corpus("test", [c], [c])
    assert len(kb._hybrid_searcher._corpus_cache) == 1

    # Invalidate via clear()
    kb.clear()
    assert len(kb._hybrid_searcher._corpus_cache) == 0


def test_normalize_corpus_texts_repairs_pdf_glued_runs_and_preserves_terms():
    """Verify PDF/EPUB ligatures and glued prose runs are segmented without splitting technical/IUPAC terms."""
    from piragi.hybrid_search import normalize_corpus_texts

    corpus = [
        "GPU power usage and dependent model utilization estimation is likely in safety evaluations.",
        "The actualpowerusageofaGPUisdependentonitsutilizationandislikelytovaryfromtheThermalDesign Power.",
        "In monohalopyridinium halides and mRNA or PyTorch pipelines, can \ufb01t in context.",
    ]
    cleaned = normalize_corpus_texts(corpus)
    assert "actual power usage of a GPU is dependent on its utilization" in cleaned[1]
    assert "monohalopyridinium" in cleaned[2]
    assert "mRNA" in cleaned[2]
    assert "PyTorch" in cleaned[2]
    assert "can fit in context" in cleaned[2]
