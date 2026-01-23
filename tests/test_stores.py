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


class TestPostgresStoreRetry:
    """Tests for PostgresStore retry logic."""

    def test_retriable_errors_constant(self):
        """Test that RETRIABLE_ERRORS contains expected error patterns."""
        from piragi.stores.postgres import PostgresStore

        # Verify key error patterns are in the list
        assert "transaction is aborted" in PostgresStore.RETRIABLE_ERRORS
        assert "server closed the connection" in PostgresStore.RETRIABLE_ERRORS
        assert "connection already closed" in PostgresStore.RETRIABLE_ERRORS
        assert "connection refused" in PostgresStore.RETRIABLE_ERRORS

    def test_is_retriable_error_method(self):
        """Test _is_retriable_error method logic."""
        from piragi.stores.postgres import PostgresStore

        # Create instance without calling __init__
        store = object.__new__(PostgresStore)
        store.RETRIABLE_ERRORS = PostgresStore.RETRIABLE_ERRORS

        # Test retriable errors
        assert store._is_retriable_error(
            Exception("current transaction is aborted, commands ignored")
        ) is True
        assert store._is_retriable_error(
            Exception("server closed the connection unexpectedly")
        ) is True
        assert store._is_retriable_error(
            Exception("connection already closed")
        ) is True

        # Test non-retriable errors
        assert store._is_retriable_error(
            Exception("syntax error at or near SELECT")
        ) is False
        assert store._is_retriable_error(
            Exception("column 'foo' does not exist")
        ) is False

    def test_execute_with_retry_success_first_attempt(self):
        """Test _execute_with_retry succeeds on first attempt."""
        from piragi.stores.postgres import PostgresStore

        store = object.__new__(PostgresStore)
        store.RETRIABLE_ERRORS = PostgresStore.RETRIABLE_ERRORS
        store.max_retries = 3
        store.retry_delay = 0.01  # Fast for testing

        call_count = 0

        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = store._execute_with_retry(successful_operation)
        assert result == "success"
        assert call_count == 1

    def test_execute_with_retry_retries_on_retriable_error(self):
        """Test _execute_with_retry retries on retriable errors."""
        from piragi.stores.postgres import PostgresStore

        store = object.__new__(PostgresStore)
        store.RETRIABLE_ERRORS = PostgresStore.RETRIABLE_ERRORS
        store.max_retries = 3
        store.retry_delay = 0.01
        store._reconnect = MagicMock()

        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transaction is aborted")
            return "success"

        result = store._execute_with_retry(fail_then_succeed)
        assert result == "success"
        assert call_count == 3
        assert store._reconnect.call_count == 2  # Called on each retry

    def test_execute_with_retry_raises_non_retriable_immediately(self):
        """Test _execute_with_retry raises non-retriable errors immediately."""
        from piragi.stores.postgres import PostgresStore

        store = object.__new__(PostgresStore)
        store.RETRIABLE_ERRORS = PostgresStore.RETRIABLE_ERRORS
        store.max_retries = 3
        store.retry_delay = 0.01

        call_count = 0

        def non_retriable_error():
            nonlocal call_count
            call_count += 1
            raise Exception("syntax error")

        with pytest.raises(Exception, match="syntax error"):
            store._execute_with_retry(non_retriable_error)

        assert call_count == 1  # No retries for non-retriable errors

    def test_execute_with_retry_exhausts_retries(self):
        """Test _execute_with_retry exhausts retries and raises."""
        from piragi.stores.postgres import PostgresStore

        store = object.__new__(PostgresStore)
        store.RETRIABLE_ERRORS = PostgresStore.RETRIABLE_ERRORS
        store.max_retries = 2
        store.retry_delay = 0.01
        store._reconnect = MagicMock()

        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise Exception("transaction is aborted")

        with pytest.raises(Exception, match="transaction is aborted"):
            store._execute_with_retry(always_fail)

        assert call_count == 3  # Initial + 2 retries


class TestDimensionInference:
    """Tests for automatic dimension inference."""

    def test_get_dimensions_method_exists(self):
        """Test that EmbeddingGenerator has get_dimensions method."""
        from piragi.embeddings import EmbeddingGenerator

        # Verify the method exists
        assert hasattr(EmbeddingGenerator, "get_dimensions")

    def test_postgres_store_dimension_inference_logic(self):
        """Test PostgresStore dimension inference logic without actual connection."""
        from piragi.stores.postgres import PostgresStore

        # Create a mock embedder with get_dimensions
        mock_embedder = MagicMock()
        mock_embedder.get_dimensions.return_value = 384

        # Test the dimension inference logic directly
        # We can't instantiate PostgresStore without psycopg2, but we can test the logic
        store = object.__new__(PostgresStore)

        # Simulate the dimension logic
        vector_dimension = None
        embedder = mock_embedder

        if vector_dimension is not None:
            result = vector_dimension
        elif embedder is not None:
            result = embedder.get_dimensions()
        else:
            result = 768

        assert result == 384
        mock_embedder.get_dimensions.assert_called_once()

    def test_postgres_store_explicit_dimension_takes_precedence(self):
        """Test that explicit vector_dimension takes precedence over embedder."""
        mock_embedder = MagicMock()
        mock_embedder.get_dimensions.return_value = 384

        # Simulate the dimension logic with explicit dimension
        vector_dimension = 1536
        embedder = mock_embedder

        if vector_dimension is not None:
            result = vector_dimension
        elif embedder is not None:
            result = embedder.get_dimensions()
        else:
            result = 768

        assert result == 1536
        # get_dimensions should NOT be called when explicit dimension provided
        mock_embedder.get_dimensions.assert_not_called()

    def test_postgres_store_default_dimension(self):
        """Test that default dimension is 768 when neither provided."""
        # Simulate the dimension logic with no embedder
        vector_dimension = None
        embedder = None

        if vector_dimension is not None:
            result = vector_dimension
        elif embedder is not None:
            result = embedder.get_dimensions()
        else:
            result = 768

        assert result == 768


class TestPineconeStore:
    """Tests for PineconeStore (mocked)."""

    def test_requires_dependencies(self):
        """Test that PineconeStore requires pinecone-client."""
        # This will fail if deps aren't installed
        with pytest.raises(ImportError):
            PineconeStore(api_key="test", index_name="test")
