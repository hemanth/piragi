"""Tests for on-the-fly embedding (query-time embedding, no persisted vector store)."""

import pytest
from unittest.mock import MagicMock

from piragi.on_the_fly import OnTheFlyIndex
from piragi.types import Chunk


@pytest.fixture
def sample_chunks():
    return [
        Chunk(text="Python is great for scripting.", source="doc0.txt", chunk_index=0),
        Chunk(text="JavaScript runs in the browser.", source="doc1.txt", chunk_index=0),
    ]


def _mock_embedder(vectors_by_text, query_vector):
    """Build a mock embedder matching EmbeddingGenerator's interface used by OnTheFlyIndex."""
    embedder = MagicMock()
    embedder.batch_size = 32
    embedder._generate_embeddings.side_effect = (
        lambda texts, batch_size: [vectors_by_text[t] for t in texts]
    )
    embedder.embed_query.return_value = query_vector
    return embedder


class TestOnTheFlyIndex:
    """Tests for query-time embedding of an in-memory chunk index."""

    def test_add_chunks_and_count(self, sample_chunks):
        index = OnTheFlyIndex()
        index.add_chunks(sample_chunks)

        assert index.count() == 2

    def test_search_before_adding_returns_empty(self):
        index = OnTheFlyIndex()
        embedder = _mock_embedder({}, [1.0, 0.0])

        assert index.search(embedder, "anything") == []
        embedder._generate_embeddings.assert_not_called()

    def test_search_ranks_by_cosine_similarity(self, sample_chunks):
        index = OnTheFlyIndex()
        index.add_chunks(sample_chunks)

        vectors = {
            "Python is great for scripting.": [1.0, 0.0],
            "JavaScript runs in the browser.": [0.0, 1.0],
        }
        embedder = _mock_embedder(vectors, query_vector=[1.0, 0.0])

        results = index.search(embedder, "Python scripting", top_k=2)

        assert results[0].source == "doc0.txt"
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.0)

    def test_search_caches_embeddings_across_calls(self, sample_chunks):
        """Test that a chunk is only embedded once even across repeated queries -
        this is the whole point of on-the-fly mode's cache (avoid re-paying embedding
        cost for unchanged chunks on every query)."""
        index = OnTheFlyIndex()
        index.add_chunks(sample_chunks)

        vectors = {
            "Python is great for scripting.": [1.0, 0.0],
            "JavaScript runs in the browser.": [0.0, 1.0],
        }
        embedder = _mock_embedder(vectors, query_vector=[1.0, 0.0])

        index.search(embedder, "first query")
        index.search(embedder, "second query")

        assert embedder._generate_embeddings.call_count == 1

    def test_search_only_embeds_new_chunks(self, sample_chunks):
        """Test that adding a new chunk only pays the embedding cost for that chunk."""
        index = OnTheFlyIndex()
        index.add_chunks(sample_chunks[:1])

        vectors = {
            "Python is great for scripting.": [1.0, 0.0],
            "JavaScript runs in the browser.": [0.0, 1.0],
        }
        embedder = _mock_embedder(vectors, query_vector=[1.0, 0.0])
        index.search(embedder, "query")

        index.add_chunks(sample_chunks[1:])
        index.search(embedder, "query")

        embedded_texts = [call.args[0] for call in embedder._generate_embeddings.call_args_list]
        assert embedded_texts == [
            ["Python is great for scripting."],
            ["JavaScript runs in the browser."],
        ]

    def test_clear_resets_chunks_and_cache(self, sample_chunks):
        index = OnTheFlyIndex()
        index.add_chunks(sample_chunks)

        index.clear()

        assert index.count() == 0
        assert index._embedding_cache == {}

    def test_delete_by_source_removes_matching_chunks(self, sample_chunks):
        index = OnTheFlyIndex()
        index.add_chunks(sample_chunks)

        deleted = index.delete_by_source("doc0.txt")

        assert deleted == 1
        assert index.count() == 1
        assert index._chunks[0].source == "doc1.txt"

    def test_delete_by_source_missing_source_is_noop(self, sample_chunks):
        index = OnTheFlyIndex()
        index.add_chunks(sample_chunks)

        deleted = index.delete_by_source("nonexistent.txt")

        assert deleted == 0
        assert index.count() == 2
