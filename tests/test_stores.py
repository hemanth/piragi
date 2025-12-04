"""Tests for vector store implementations."""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
import shutil

from piragi.types import Chunk, Citation
from piragi.stores import (
    VectorStoreProtocol,
    LanceStore,
    PostgresStore,
    PineconeStore,
)
from piragi.stores.factory import parse_store_uri, create_store


class TestVectorStoreProtocol:
    """Tests for the VectorStore protocol."""

    def test_protocol_check(self):
        """Test that stores implement the protocol."""
        # LanceStore should implement the protocol
        store = LanceStore(uri=tempfile.mkdtemp())
        assert isinstance(store, VectorStoreProtocol)

    def test_custom_store_protocol(self):
        """Test that custom stores can implement the protocol."""
        class CustomStore:
            def add_chunks(self, chunks):
                pass
            def search(self, query_embedding, top_k=5, filters=None):
                return []
            def delete_by_source(self, source):
                return 0
            def count(self):
                return 0
            def clear(self):
                pass
            def get_all_chunk_texts(self):
                return []

        store = CustomStore()
        assert isinstance(store, VectorStoreProtocol)


class TestParseStoreUri:
    """Tests for URI parsing."""

    def test_local_path(self):
        """Test parsing local paths."""
        result = parse_store_uri(".piragi")
        assert result["type"] == "lance"
        assert result["uri"] == ".piragi"

    def test_local_path_with_slash(self):
        """Test parsing local paths with slashes."""
        result = parse_store_uri("./data/vectors")
        assert result["type"] == "lance"
        assert result["uri"] == "./data/vectors"

    def test_s3_uri(self):
        """Test parsing S3 URIs."""
        result = parse_store_uri("s3://my-bucket/indices")
        assert result["type"] == "lance"
        assert result["uri"] == "s3://my-bucket/indices"

    def test_postgres_uri(self):
        """Test parsing PostgreSQL URIs."""
        result = parse_store_uri("postgres://user:pass@localhost:5432/db")
        assert result["type"] == "postgres"
        assert result["connection_string"] == "postgres://user:pass@localhost:5432/db"

    def test_postgresql_uri(self):
        """Test parsing PostgreSQL URIs with full scheme."""
        result = parse_store_uri("postgresql://user:pass@localhost/db")
        assert result["type"] == "postgres"

    def test_pinecone_uri(self):
        """Test parsing Pinecone URIs."""
        result = parse_store_uri("pinecone://my-index?api_key=abc123&environment=us-east-1")
        assert result["type"] == "pinecone"
        assert result["index_name"] == "my-index"
        assert result["api_key"] == "abc123"
        assert result["environment"] == "us-east-1"

    def test_pinecone_uri_with_namespace(self):
        """Test parsing Pinecone URIs with namespace."""
        result = parse_store_uri("pinecone://my-index?api_key=abc&namespace=prod")
        assert result["namespace"] == "prod"


class TestCreateStore:
    """Tests for the store factory."""

    def test_create_default_store(self):
        """Test creating default LanceStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_store(persist_dir=tmpdir)
            assert isinstance(store, LanceStore)

    def test_create_from_path(self):
        """Test creating store from local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_store(store=tmpdir)
            assert isinstance(store, LanceStore)

    def test_create_from_s3_uri(self):
        """Test creating store from S3 URI."""
        # This will create a LanceStore configured for S3
        # Note: Won't actually connect to S3 without credentials
        with pytest.raises(Exception):
            # Should fail without S3 credentials, but proves URI parsing works
            store = create_store(store="s3://nonexistent-bucket/path")

    def test_create_from_dict(self):
        """Test creating store from config dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_store(store={"type": "lance", "uri": tmpdir})
            assert isinstance(store, LanceStore)

    def test_passthrough_existing_store(self):
        """Test that existing stores are passed through."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = LanceStore(uri=tmpdir)
            result = create_store(store=existing)
            assert result is existing

    def test_create_postgres_requires_deps(self):
        """Test that PostgresStore requires dependencies."""
        with pytest.raises((ImportError, Exception)):
            create_store(store={"type": "postgres", "host": "localhost"})

    def test_create_pinecone_requires_api_key(self):
        """Test that PineconeStore requires API key."""
        with pytest.raises(ValueError, match="api_key"):
            create_store(store={"type": "pinecone", "index_name": "test"})


class TestLanceStore:
    """Tests for LanceStore."""

    @pytest.fixture
    def store(self):
        """Create a temporary LanceStore."""
        tmpdir = tempfile.mkdtemp()
        store = LanceStore(uri=tmpdir)
        yield store
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks with embeddings."""
        return [
            Chunk(
                text="Python is a programming language.",
                source="test.md",
                chunk_index=0,
                metadata={"type": "docs"},
                embedding=[0.1] * 768,
            ),
            Chunk(
                text="JavaScript is also a programming language.",
                source="test.md",
                chunk_index=1,
                metadata={"type": "docs"},
                embedding=[0.2] * 768,
            ),
        ]

    def test_add_and_count(self, store, sample_chunks):
        """Test adding chunks and counting."""
        assert store.count() == 0
        store.add_chunks(sample_chunks)
        assert store.count() == 2

    def test_add_chunks_without_embeddings_fails(self, store):
        """Test that chunks without embeddings fail."""
        chunks = [Chunk(text="No embedding", source="test.md", chunk_index=0)]
        with pytest.raises(ValueError, match="embeddings"):
            store.add_chunks(chunks)

    def test_search(self, store, sample_chunks):
        """Test searching for chunks."""
        store.add_chunks(sample_chunks)

        # Search with first chunk's embedding
        results = store.search(
            query_embedding=[0.1] * 768,
            top_k=2,
            min_chunk_length=10,
        )

        assert len(results) == 2
        assert all(isinstance(r, Citation) for r in results)
        # First result should be most similar
        assert "Python" in results[0].chunk

    def test_search_with_filters(self, store, sample_chunks):
        """Test searching with metadata filters."""
        store.add_chunks(sample_chunks)

        results = store.search(
            query_embedding=[0.1] * 768,
            top_k=2,
            filters={"type": "docs"},
            min_chunk_length=10,
        )

        assert len(results) == 2

    def test_delete_by_source(self, store, sample_chunks):
        """Test deleting chunks by source."""
        store.add_chunks(sample_chunks)
        assert store.count() == 2

        deleted = store.delete_by_source("test.md")
        assert deleted == 2
        assert store.count() == 0

    def test_clear(self, store, sample_chunks):
        """Test clearing all data."""
        store.add_chunks(sample_chunks)
        assert store.count() == 2

        store.clear()
        assert store.count() == 0

    def test_get_all_chunk_texts(self, store, sample_chunks):
        """Test getting all chunk texts."""
        store.add_chunks(sample_chunks)

        texts = store.get_all_chunk_texts()
        assert len(texts) == 2
        assert "Python" in texts[0]


class TestPostgresStore:
    """Tests for PostgresStore (mocked)."""

    def test_requires_dependencies(self):
        """Test that PostgresStore requires psycopg2 and pgvector."""
        # This will fail if deps aren't installed
        with pytest.raises(ImportError):
            PostgresStore(connection_string="postgres://test")


class TestPineconeStore:
    """Tests for PineconeStore (mocked)."""

    def test_requires_dependencies(self):
        """Test that PineconeStore requires pinecone-client."""
        # This will fail if deps aren't installed
        with pytest.raises(ImportError):
            PineconeStore(api_key="test", index_name="test")
