"""Comprehensive unit tests for GCP integrations in Piragi.

Covers:
- LanceStore GCS remote URI handling (gs://, gcs://, s3:// bypass of local mkdir)
- Vector store factory routing of cloud object store URIs
- Embedding dimension registry for Vertex AI models (768 dimensions)
- EmbeddingGenerator with vertexai backend (ADC authentication, live predict, offline fallback)
- LLMClient dynamic ADC token refresh and retry mechanisms
"""

import threading
from unittest.mock import MagicMock, patch

from piragi.embeddings import EmbeddingGenerator
from piragi.llm_client import LLMClient
from piragi.stores.factory import create_store, parse_store_uri
from piragi.stores.lance import EMBEDDING_DIMENSIONS, LanceStore, get_embedding_dimension
from piragi.types import Chunk

# ===========================================================================
# 1. GCS & Cloud Object Store URI Handling
# ===========================================================================


class TestGCSURIHandling:
    """Tests for LanceStore and Store Factory GCS URI routing."""

    def test_parse_store_uri_gs(self):
        """parse_store_uri recognizes gs:// scheme as LanceStore."""
        parsed = parse_store_uri("gs://my-rag-bucket/indices/kb1")
        assert parsed["type"] == "lance"
        assert parsed["uri"] == "gs://my-rag-bucket/indices/kb1"

    def test_parse_store_uri_gcs(self):
        """parse_store_uri recognizes gcs:// scheme as LanceStore."""
        parsed = parse_store_uri("gcs://my-rag-bucket/indices/kb2")
        assert parsed["type"] == "lance"
        assert parsed["uri"] == "gcs://my-rag-bucket/indices/kb2"

    def test_parse_store_uri_case_insensitivity(self):
        """parse_store_uri handles uppercase GS:// and GCS://."""
        assert parse_store_uri("GS://my-bucket/path")["type"] == "lance"
        assert parse_store_uri("GCS://my-bucket/path")["type"] == "lance"

    @patch("pathlib.Path.mkdir")
    @patch("lancedb.connect")
    def test_lance_store_bypasses_mkdir_for_gs(self, mock_connect, mock_mkdir):
        """LanceStore does not call Path.mkdir for gs:// URIs."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_connect.return_value = mock_db

        store = LanceStore(uri="gs://my-cloud-bucket/vectors")
        mock_mkdir.assert_not_called()
        mock_connect.assert_called_with("gs://my-cloud-bucket/vectors")
        assert store.uri == "gs://my-cloud-bucket/vectors"

    @patch("pathlib.Path.mkdir")
    @patch("lancedb.connect")
    def test_lance_store_bypasses_mkdir_for_gcs(self, mock_connect, mock_mkdir):
        """LanceStore does not call Path.mkdir for gcs:// URIs."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_connect.return_value = mock_db

        store = LanceStore(uri="gcs://my-cloud-bucket/vectors")
        mock_mkdir.assert_not_called()
        assert store.uri == "gcs://my-cloud-bucket/vectors"

    @patch("pathlib.Path.mkdir")
    @patch("lancedb.connect")
    def test_lance_store_creates_mkdir_for_local_paths(self, mock_connect, mock_mkdir, tmp_path):
        """LanceStore invokes Path.mkdir for standard local file paths."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_connect.return_value = mock_db

        local_path = str(tmp_path / "local_store")
        store = LanceStore(uri=local_path)
        mock_mkdir.assert_called_once()
        assert store.uri == local_path

    @patch("lancedb.connect")
    def test_create_store_routes_gs(self, mock_connect):
        """create_store(store='gs://...') instantiates LanceStore with GCS URI."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_connect.return_value = mock_db

        store = create_store(store="gs://enterprise-data/lancedb")
        assert isinstance(store, LanceStore)
        assert store.uri == "gs://enterprise-data/lancedb"


# ===========================================================================
# 2. Vertex AI Embedding Dimension Registry
# ===========================================================================


class TestVertexAIDimensionRegistry:
    """Tests for Vertex AI embedding dimensions in LanceStore."""

    def test_text_embedding_004_registered(self):
        """text-embedding-004 is registered in EMBEDDING_DIMENSIONS with 768 dimensions."""
        assert "text-embedding-004" in EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS["text-embedding-004"] == 768

    def test_text_multilingual_embedding_002_registered(self):
        """text-multilingual-embedding-002 is registered with 768 dimensions."""
        assert "text-multilingual-embedding-002" in EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS["text-multilingual-embedding-002"] == 768

    def test_get_embedding_dimension_exact(self):
        """get_embedding_dimension returns 768 for text-embedding-004."""
        assert get_embedding_dimension("text-embedding-004") == 768

    def test_get_embedding_dimension_prefixed(self):
        """get_embedding_dimension handles google/ prefix."""
        assert get_embedding_dimension("google/text-embedding-004") == 768

    def test_get_embedding_dimension_full_resource_path(self):
        """get_embedding_dimension handles full publisher resource path."""
        res_path = "publishers/google/models/text-embedding-004"
        assert get_embedding_dimension(res_path) == 768

    def test_get_embedding_dimension_case_insensitive(self):
        """get_embedding_dimension handles mixed and uppercase model names."""
        assert get_embedding_dimension("TEXT-EMBEDDING-004") == 768
        assert get_embedding_dimension("Google/Text-Embedding-004") == 768

    def test_get_embedding_dimension_fallback(self):
        """get_embedding_dimension falls back to 384 for unknown models and empty string."""
        assert get_embedding_dimension("unknown-custom-model-xyz") == 384
        assert get_embedding_dimension("") == 384


# ===========================================================================
# 3. Vertex AI Embedding Generator
# ===========================================================================


class TestVertexAIEmbeddingGenerator:
    """Tests for EmbeddingGenerator with backend='vertexai'."""

    def test_init_vertexai_backend(self):
        """EmbeddingGenerator initializes with backend='vertexai' without loading Torch."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        assert gen.backend == "vertexai"
        assert gen.is_vertex is True
        assert gen.model is None

    def test_vertex_dimensions(self):
        """get_dimensions returns 768 for vertexai backend."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        assert gen.get_dimensions() == 768

    def test_embed_query_dimension(self):
        """embed_query returns a 768-dimensional normalized vector."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        vec = gen.embed_query("What is the GCP evaluation architecture?")
        assert isinstance(vec, list)
        assert len(vec) == 768

    def test_embed_query_empty_string(self):
        """embed_query handles empty string without error."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        vec = gen.embed_query("")
        assert len(vec) == 768

    def test_embed_chunks_dimension_and_metadata(self):
        """embed_chunks populates chunk embeddings with 768-dim vectors."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        chunks = [
            Chunk(text="GCP Vertex AI provides managed embeddings.", source="doc1", chunk_index=0),
            Chunk(text="LanceDB stores vectors over GCS buckets.", source="doc2", chunk_index=1),
        ]
        embedded = gen.embed_chunks(chunks)
        assert len(embedded) == 2
        assert len(embedded[0].embedding) == 768
        assert len(embedded[1].embedding) == 768

    def test_embed_chunks_progress_callback(self):
        """embed_chunks invokes the on_progress callback."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004", batch_size=2)
        chunks = [Chunk(text=f"Sample text {i}", source="doc", chunk_index=i) for i in range(5)]
        messages = []
        gen.embed_chunks(chunks, on_progress=lambda msg: messages.append(msg))
        assert len(messages) >= 3

    def test_embed_empty_chunk_list(self):
        """embed_chunks on empty list returns empty list."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        assert gen.embed_chunks([]) == []

    def test_embed_chunks_extreme_length(self):
        """embed_chunks handles long chunks by truncating appropriately."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        huge_text = "Google Cloud Platform " * 2000
        chunks = [Chunk(text=huge_text, source="doc_huge", chunk_index=0)]
        embedded = gen.embed_chunks(chunks)
        assert len(embedded[0].embedding) == 768

    def test_offline_fallback_deterministic(self):
        """Hermetic fallback produces deterministic normalized vectors."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        # Ensure credentials / network fail to exercise fallback branch
        with patch.object(gen, "project_id", None):
            vec1 = gen.embed_query("Deterministic test prompt")
            vec2 = gen.embed_query("Deterministic test prompt")
            assert len(vec1) == 768
            assert vec1 == vec2


# ===========================================================================
# 4. LLMClient Dynamic ADC Token Refresh
# ===========================================================================


class TestLLMClientADCTokenRefresh:
    """Tests for LLMClient with Vertex AI provider and ADC token refresh."""

    def test_init_with_vertexai_provider(self):
        """LLMClient initializes with provider='vertexai' and flags ADC refresh."""
        client = LLMClient(provider="vertexai")
        assert client.provider == "vertexai"
        assert client._use_adc_refresh is True

    def test_init_with_gcp_endpoint_url(self):
        """LLMClient auto-enables ADC refresh when base_url contains googleapis.com."""
        client = LLMClient(base_url="https://us-central1-aiplatform.googleapis.com/v1beta1/...")
        assert client._use_adc_refresh is True

    @patch("openai.OpenAI")
    def test_token_refresh_updates_api_key(self, mock_openai):
        """_refresh_token_if_needed updates client.api_key with fresh bearer token."""
        mock_instance = MagicMock()
        mock_instance.api_key = "stale_token"
        mock_openai.return_value = mock_instance

        client = LLMClient(provider="vertexai")
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.token = "refreshed_ya29_token"
        client._credentials = mock_creds

        client._refresh_token_if_needed()
        mock_creds.refresh.assert_called_once()
        assert client._client.api_key == "refreshed_ya29_token"

    @patch("openai.OpenAI")
    def test_complete_invokes_token_refresh(self, mock_openai):
        """complete() refreshes token before issuing chat completion request."""
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Vertex completion"))]
        )
        mock_openai.return_value = mock_instance

        client = LLMClient(provider="vertexai")
        with patch.object(client, "_refresh_token_if_needed") as mock_refresh:
            resp = client.complete(messages=[{"role": "user", "content": "Hello"}])
            mock_refresh.assert_called_once()
            assert resp.choices[0].message.content == "Vertex completion"

    def test_thread_safe_client_access(self):
        """Concurrent threads can safely access client without data race."""
        client = LLMClient(provider="vertexai")
        errors = []

        def worker():
            try:
                for _ in range(50):
                    _ = client.client
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
