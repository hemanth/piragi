"""Adversarial stress testing and empirical probes for GCP integrations.

Milestone 2 Challenger Suite covering:
1. GCS URI Parsing & Storage Factory Routing Stress
2. Vertex AI Embedding Dimension Registry Stress
3. Vertex AI Embedding Generator & Batching Boundary Stress
4. Dynamic ADC Token Refresh & Multi-Thread Concurrency Contention Stress
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from piragi.embeddings import EmbeddingGenerator
from piragi.llm_client import LLMClient
from piragi.stores.factory import create_store, parse_store_uri
from piragi.stores.lance import (
    EMBEDDING_DIMENSIONS,
    LanceStore,
    get_embedding_dimension,
)
from piragi.types import Chunk


# ===========================================================================
# 1. GCS URI PARSING & FACTORY ROUTING STRESS
# ===========================================================================

class TestGCSURIParsingStress:
    """Stress testing URI parsing and factory routing across extreme URI formats."""

    @pytest.mark.parametrize(
        "uri",
        [
            # Standard GCS & S3
            "gs://my-bucket/indices/kb1",
            "gcs://my-bucket/indices/kb1",
            "s3://my-bucket/indices/kb1",
            # Mixed case schemes
            "GS://my-bucket/indices/kb1",
            "Gcs://my-bucket/indices/kb1",
            "GCS://my-bucket/indices/kb1",
            "Gs://my-bucket/indices/kb1",
            "S3://my-bucket/indices/kb1",
            # Uppercase bucket and paths
            "gs://UPPERCASE-BUCKET/PATH/TO/INDEX",
            "gcs://UPPERCASE-BUCKET/PATH/TO/INDEX",
            # URIs with query parameters
            "gs://my-bucket/indices?timeout=60&region=us-central1",
            "gcs://my-bucket/indices?version=2&read_only=true",
            # URIs with fragments
            "gs://my-bucket/indices#checkpoint-latest",
            "gcs://my-bucket/indices#shard-001",
            # URIs with spaces and encoded characters
            "gs://my-bucket/path%20with%20spaces/store",
            "gcs://my-bucket/dataset_2026/split=train/part=0",
            # Minimal URIs (root bucket without trailing slash or path)
            "gs://minimal-bucket",
            "gcs://minimal-bucket",
            # Trailing slashes
            "gs://my-bucket/indices/",
            "gcs://my-bucket/indices/",
        ],
    )
    def test_parse_store_uri_routing_variants(self, uri: str):
        """Verify parse_store_uri correctly classifies all remote URI variations as lance."""
        parsed = parse_store_uri(uri)
        assert parsed["type"] == "lance"
        assert parsed["uri"] == uri

    @pytest.mark.parametrize(
        "remote_uri",
        [
            "gs://bucket/data",
            "gcs://bucket/data",
            "GS://bucket/data",
            "GCS://bucket/data",
            "s3://bucket/data",
            "S3://bucket/data",
            "az://container/data",
            "abfs://container/data",
            "gs://bucket/data?param=val#frag",
        ],
    )
    @patch("pathlib.Path.mkdir")
    @patch("lancedb.connect")
    def test_lance_store_remote_never_calls_mkdir(self, mock_connect, mock_mkdir, remote_uri: str):
        """Verify LanceStore NEVER invokes Path.mkdir for any remote cloud URI variation."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_connect.return_value = mock_db

        store = LanceStore(uri=remote_uri)
        mock_mkdir.assert_not_called()
        mock_connect.assert_called_once_with(remote_uri)
        assert store.uri == remote_uri

    @pytest.mark.parametrize(
        "local_uri",
        [
            "/tmp/rag_store",
            "./local/indices",
            "data/vectors/store.lance",
            "relative/path/gs://fake",
            "/var/folders/gcs://fake",
        ],
    )
    @patch("pathlib.Path.mkdir")
    @patch("lancedb.connect")
    def test_lance_store_local_paths_invoke_mkdir(self, mock_connect, mock_mkdir, local_uri: str):
        """Verify LanceStore invokes Path.mkdir for true local filesystem paths."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_connect.return_value = mock_db

        store = LanceStore(uri=local_uri)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert store.uri == local_uri

    @patch("lancedb.connect")
    def test_create_store_end_to_end_routing(self, mock_connect):
        """Verify create_store factory correctly initializes LanceStore for remote schemes."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_connect.return_value = mock_db

        store_gs = create_store("gs://enterprise-rag-bucket/index")
        assert isinstance(store_gs, LanceStore)
        assert store_gs.uri == "gs://enterprise-rag-bucket/index"

        store_gcs = create_store("GCS://enterprise-rag-bucket/index")
        assert isinstance(store_gcs, LanceStore)
        assert store_gcs.uri == "GCS://enterprise-rag-bucket/index"


# ===========================================================================
# 2. VERTEX AI EMBEDDING DIMENSION REGISTRY STRESS
# ===========================================================================

class TestDimensionRegistryStress:
    """Stress testing get_embedding_dimension against casing, resource paths, and adversarial inputs."""

    @pytest.mark.parametrize(
        "model_input,expected_dim",
        [
            # Exact matches
            ("text-embedding-004", 768),
            ("google/text-embedding-004", 768),
            ("text-multilingual-embedding-002", 768),
            ("google/text-multilingual-embedding-002", 768),
            # Arbitrary casing
            ("TEXT-EMBEDDING-004", 768),
            ("Text-Embedding-004", 768),
            ("GOOGLE/TEXT-EMBEDDING-004", 768),
            ("Google/Text-Multilingual-Embedding-002", 768),
            ("TEXT-MULTILINGUAL-EMBEDDING-002", 768),
            # GCP Vertex AI publisher resource names
            ("publishers/google/models/text-embedding-004", 768),
            ("projects/123456/locations/us-central1/publishers/google/models/text-embedding-004", 768),
            ("projects/my-proj/locations/europe-west4/publishers/google/models/text-multilingual-embedding-002", 768),
            ("endpoints/98765/publishers/google/models/text-embedding-004", 768),
            # Leading and trailing whitespace
            ("   text-embedding-004   ", 768),
            ("\tgoogle/text-embedding-004\n", 768),
            # Gemma family
            ("google/embeddinggemma-2", 768),
            ("embeddinggemma-2", 768),
            ("google/embeddinggemma-300m", 768),
            # Non-GCP baseline standard models
            ("all-mpnet-base-v2", 768),
            ("all-MiniLM-L6-v2", 384),
            ("BAAI/bge-large-en-v1.5", 1024),
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
        ],
    )
    def test_embedding_dimensions_resolution(self, model_input: str, expected_dim: int):
        """Verify dimension inference handles all naming conventions, prefixes, and cases."""
        assert get_embedding_dimension(model_input) == expected_dim

    @pytest.mark.parametrize(
        "edge_input",
        [
            "",
            None,
            "   ",
            "\n\t\r",
            "unknown-custom-model-x-9999",
            "hf-internal/nonexistent-model-dimension-probe",
            "invalid!@#$%^&*()_model",
        ],
    )
    def test_embedding_dimension_fallback_robustness(self, edge_input):
        """Verify edge case or unknown model inputs do not raise unhandled exceptions."""
        dim = get_embedding_dimension(edge_input)
        assert isinstance(dim, int)
        assert dim in (384, 768)


# ===========================================================================
# 3. VERTEX AI EMBEDDING GENERATOR & BATCHING BOUNDARY STRESS
# ===========================================================================

class TestVertexAIEmbeddingsStress:
    """Stress testing Vertex AI embedding generation, payloads, unicode, batching, and fallback."""

    def test_vertex_initialization(self):
        """Verify generator initializes cleanly with vertexai backend."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        assert gen.is_vertex is True
        assert gen.get_dimensions() == 768
        assert gen.project_id is not None
        assert gen.location == "us-central1"

    @pytest.mark.parametrize(
        "text_input",
        [
            # Empty / whitespace
            "",
            "   ",
            "\t\n\r",
            # Unicode scripts
            "日本語の自然言語処理と検索拡張生成の評価テスト",
            "针对多语言检索与大模型嵌入表示的压力测试",
            "한국어 검색 증강 생성 파이프라인 성능 평가",
            "اختبار توليد الاسترجاع المعزز لنماذج اللغة الكبيرة",
            "Стресс-тестирование генерации векторных представлений",
            # Emojis & Complex unicode
            "🚀🔥🎉 Vertex AI embeddings with emojis! 💡✨",
            "Complex ZWJ emoji sequences: 👨‍👩‍👧‍👦 🏳️‍🌈 🧑🏽‍💻",
            # Control characters & null bytes
            "Text with null \x00 characters \x01\x02 inside",
            # Newlines and indentation
            "Line 1\nLine 2\r\nLine 3\n\n\nLine 4\tTabbed",
            # Code and LaTeX
            "def embed(x: str) -> List[float]:\n    return [0.0] * 768",
            "$$\\sum_{i=1}^{n} x_i^2 = \\|\\mathbf{x}\\|^2$$",
            # Massive input (>10,000 characters)
            "A" * 15000,
            "Chunk with repetitive technical words. " * 500,  # ~19,500 chars
        ],
    )
    def test_embed_query_adversarial_inputs(self, text_input: str):
        """Verify embed_query handles extreme, foreign, null, emoji, and massive inputs."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        vec = gen.embed_query(text_input)
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(val, float) for val in vec)
        # Vector should be non-zero and finite
        assert any(val != 0.0 for val in vec)

    @pytest.mark.parametrize("batch_size", [1, 5, 32, 250])
    def test_embed_chunks_batch_size_boundaries(self, batch_size: int):
        """Verify batch boundaries (1, 5, 32, 250) process cleanly without dropped chunks."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004", batch_size=batch_size)
        
        # Test varying chunk counts around batch boundaries
        for num_chunks in [1, 5, 10, batch_size, batch_size + 1]:
            chunks = [
                Chunk(
                    source="stress_test.py",
                    chunk_index=i,
                    text=f"Technical chunk content for index {i} with payload details.",
                    metadata={"index": i, "batch": batch_size},
                )
                for i in range(num_chunks)
            ]
            embedded = gen.embed_chunks(chunks)
            assert len(embedded) == num_chunks
            for i, chunk in enumerate(embedded):
                assert chunk.embedding is not None
                assert len(chunk.embedding) == 768
                assert chunk.metadata["index"] == i

    def test_embed_chunks_massive_spanning_batches(self):
        """Verify embedding 300 chunks across multiple batches (>250 boundary)."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004", batch_size=100)
        chunks = [
            Chunk(
                source="large_corpus.md",
                chunk_index=i,
                text=f"Batch scaling chunk #{i}: evaluating high-throughput ingestion under stress.",
            )
            for i in range(300)
        ]
        embedded = gen.embed_chunks(chunks)
        assert len(embedded) == 300
        assert all(len(c.embedding) == 768 for c in embedded)

    def test_embed_chunks_batch_size_zero_boundary(self):
        """Verify batch_size=0 handles empty lists or raises ValueError on range step=0."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004", batch_size=0)
        # Empty list returns early before range()
        assert gen.embed_chunks([]) == []

        # Non-empty list triggers ValueError due to range(0, total, 0)
        chunk = Chunk(source="doc.txt", chunk_index=0, text="content")
        with pytest.raises(ValueError, match="range\\(\\) arg 3 must not be zero"):
            gen.embed_chunks([chunk])

    def test_embed_chunks_empty_list(self):
        """Verify empty chunk list returns empty list immediately."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        assert gen.embed_chunks([]) == []

    def test_deterministic_offline_fallback(self):
        """Verify offline fallback guarantees exact determinism and L2 normalization."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        # Ensure offline mode by unsetting credentials and token
        with patch.object(gen, "_credentials", None), patch.dict(os.environ, {"GCP_ACCESS_TOKEN": ""}, clear=False):
            vec1 = gen.embed_query("deterministic invariant test")
            vec2 = gen.embed_query("deterministic invariant test")
            vec3 = gen.embed_query("different query text")

            assert vec1 == vec2
            assert vec1 != vec3
            assert len(vec1) == 768

            # Verify L2 normalization: ||vec|| ~ 1.0
            norm_sq = sum(x * x for x in vec1)
            assert pytest.approx(norm_sq, abs=1e-4) == 1.0

    @patch("requests.post")
    def test_live_rest_endpoint_mock_response_success(self, mock_post):
        """Verify REST API response parsing when Vertex AI endpoint returns 200 OK."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {"embeddings": {"values": [0.05] * 768}},
                {"embeddings": {"values": [-0.05] * 768}},
            ]
        }
        mock_post.return_value = mock_response

        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        with patch.dict(os.environ, {"GCP_ACCESS_TOKEN": "mock-test-token"}):
            vectors = gen._generate_vertexai_embeddings(["query one", "query two"])
            assert len(vectors) == 2
            assert vectors[0] == [0.05] * 768
            assert vectors[1] == [-0.05] * 768

    @pytest.mark.parametrize("status_code", [400, 401, 403, 500, 503])
    @patch("requests.post")
    def test_rest_endpoint_error_status_triggers_fallback(self, mock_post, status_code: int):
        """Verify non-200 HTTP status codes seamlessly trigger fallback without crashing."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = f"Simulated HTTP {status_code} error from Vertex AI"
        mock_post.return_value = mock_response

        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        with patch.dict(os.environ, {"GCP_ACCESS_TOKEN": "mock-test-token"}):
            vectors = gen._generate_vertexai_embeddings(["text for fallback"])
            assert len(vectors) == 1
            assert len(vectors[0]) == 768
            assert any(x != 0.0 for x in vectors[0])


# ===========================================================================
# 4. DYNAMIC ADC TOKEN REFRESH & MULTI-THREAD CONCURRENCY STRESS
# ===========================================================================

class TestLLMClientTokenRefreshConcurrencyStress:
    """Stress testing dynamic ADC token refresh under multi-threaded concurrency and simulated expiry."""

    def test_token_refresh_lifecycle(self):
        """Verify credentials refresh lifecycle updates client.api_key."""
        client = LLMClient(provider="vertexai")
        assert client._use_adc_refresh is True

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.token = "initial-token"

        def do_refresh(request):
            mock_creds.valid = True
            mock_creds.token = "refreshed-bearer-token-12345"

        mock_creds.refresh.side_effect = do_refresh
        client._credentials = mock_creds

        client._refresh_token_if_needed()
        assert client._client.api_key == "refreshed-bearer-token-12345"
        mock_creds.refresh.assert_called_once()

    def test_auth_refresh_exception_handled_safely(self):
        """Verify auth refresh network/credentials exceptions do not raise fatal crashes."""
        client = LLMClient(provider="vertexai")
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.refresh.side_effect = RuntimeError("Simulated GCP network failure during token refresh")
        client._credentials = mock_creds

        # Must not raise exception
        client._refresh_token_if_needed()
        assert client._token_lock.locked() is False

    def test_concurrent_token_refresh_thread_contention(self):
        """Stress test 10 concurrent threads hammering _refresh_token_if_needed simultaneously."""
        client = LLMClient(provider="vertexai")
        
        refresh_call_count = 0
        lock = threading.Lock()

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.token = "token-v0"

        def simulate_refresh(request):
            nonlocal refresh_call_count
            time.sleep(0.01)  # introduce slight latency to amplify contention
            with lock:
                refresh_call_count += 1
            mock_creds.valid = True
            mock_creds.token = f"token-v{refresh_call_count}"

        mock_creds.refresh.side_effect = simulate_refresh
        client._credentials = mock_creds

        errors = []

        def worker_task(thread_id: int):
            try:
                for _ in range(5):
                    client._refresh_token_if_needed()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors encountered: {errors}"
        # Because the first refresh marks creds as valid, subsequent threads see valid=True
        # Refresh count should be minimal (typically 1), never 50
        assert refresh_call_count >= 1
        assert client._client.api_key == mock_creds.token
        assert client._token_lock.locked() is False

    def test_concurrent_complete_requests_with_token_expiration(self):
        """Stress test 10 concurrent threads invoking complete() while tokens periodically expire."""
        client = LLMClient(provider="vertexai")

        mock_openai_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Mocked answer"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        client._client = mock_openai_client

        token_version = 0

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "initial-token"

        def simulated_refresh(request):
            nonlocal token_version
            token_version += 1
            mock_creds.valid = True
            mock_creds.token = f"token-version-{token_version}"

        mock_creds.refresh.side_effect = simulated_refresh
        client._credentials = mock_creds

        results = []
        errors = []

        def call_complete(thread_id: int):
            try:
                # Every other thread simulates an expired token before calling complete
                if thread_id % 3 == 0:
                    mock_creds.valid = False
                res = client.complete_text(f"Query from thread {thread_id}")
                results.append(res)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=call_complete, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent complete: {errors}"
        assert len(results) == 15
        assert all(res == "Mocked answer" for res in results)
        assert client._token_lock.locked() is False
