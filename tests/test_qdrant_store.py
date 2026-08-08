"""Tests for Qdrant vector store implementation."""

import pytest
from unittest.mock import patch

from piragi.types import Chunk, Citation
from piragi.stores.qdrant import QdrantStore

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None


@pytest.fixture
def store():
    """Return an empty in-memory QdrantStore."""
    if QdrantClient is None:
        pytest.skip("qdrant-client not installed")
    store = QdrantStore(url=":memory:", vector_dimension=4)
    yield store
    store.clear()

def test_init_default(store):
    assert store.url == ":memory:"
    assert store.collection_name == "chunks"
    assert store.vector_dimension == 4
    assert store.count() == 0

def test_add_and_count(store):
    chunks = [
        Chunk(text="Chunk 1", source="source1", chunk_index=0, embedding=[0.1, 0.2, 0.3, 0.4]),
        Chunk(text="Chunk 2", source="source1", chunk_index=1, embedding=[0.2, 0.3, 0.4, 0.5]),
    ]
    store.add_chunks(chunks)
    assert store.count() == 2

def test_search_returns_citations(store):
    chunks = [
        Chunk(text="This is a test chunk that is definitely long enough to pass the min length check.", source="source1", chunk_index=0, embedding=[1.0, 0.0, 0.0, 0.0]),
        Chunk(text="Another test chunk with enough length to be considered.", source="source1", chunk_index=1, embedding=[0.0, 1.0, 0.0, 0.0]),
    ]
    store.add_chunks(chunks)
    
    citations = store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=1, min_chunk_length=10)
    assert len(citations) == 1
    assert citations[0].source == "source1"
    assert "definitely long enough" in citations[0].chunk
    assert citations[0].score > 0.9

def test_search_with_filters(store):
    chunks = [
        Chunk(text="Chunk with metadata that is long enough.", source="source1", chunk_index=0, metadata={"category": "A"}, embedding=[1.0, 0.0, 0.0, 0.0]),
        Chunk(text="Chunk with metadata that is long enough.", source="source2", chunk_index=1, metadata={"category": "B"}, embedding=[1.0, 0.0, 0.0, 0.0]),
    ]
    store.add_chunks(chunks)
    
    citations = store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=10, filters={"category": "A"}, min_chunk_length=10)
    assert len(citations) == 1
    assert citations[0].source == "source1"

def test_delete_by_source(store):
    chunks = [
        Chunk(text="Chunk 1", source="source1", chunk_index=0, embedding=[0.1, 0.2, 0.3, 0.4]),
        Chunk(text="Chunk 2", source="source2", chunk_index=0, embedding=[0.2, 0.3, 0.4, 0.5]),
    ]
    store.add_chunks(chunks)
    assert store.count() == 2
    
    store.delete_by_source("source1")
    assert store.count() == 1
    
    citations = store.search(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=10, min_chunk_length=1)
    assert citations[0].source == "source2"

def test_clear(store):
    chunks = [
        Chunk(text="Chunk 1", source="source1", chunk_index=0, embedding=[0.1, 0.2, 0.3, 0.4]),
    ]
    store.add_chunks(chunks)
    assert store.count() == 1
    
    store.clear()
    assert store.count() == 0

def test_get_all_chunk_texts(store):
    chunks = [
        Chunk(text="Text 1", source="source1", chunk_index=0, embedding=[0.1, 0.2, 0.3, 0.4]),
        Chunk(text="Text 2", source="source2", chunk_index=0, embedding=[0.2, 0.3, 0.4, 0.5]),
    ]
    store.add_chunks(chunks)
    
    texts = store.get_all_chunk_texts()
    assert len(texts) == 2
    assert "Text 1" in texts
    assert "Text 2" in texts

def test_quantization_scalar_config():
    if QdrantClient is None:
        pytest.skip("qdrant-client not installed")
        
    store = QdrantStore(url=":memory:", quantization="scalar", vector_dimension=4)
    config = store._get_quantization_config()
    assert isinstance(config, models.ScalarQuantization)
    assert config.scalar.type == models.ScalarType.INT8

def test_requires_dependencies():
    with patch("piragi.stores.qdrant.QdrantClient", None):
        with pytest.raises(ImportError, match="pip install piragi\\[qdrant\\]"):
            QdrantStore()
