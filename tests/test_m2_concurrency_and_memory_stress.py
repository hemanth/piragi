"""Adversarial Concurrency, Thread Safety, and Memory Leak Stress Tests for Milestone 2.

Target Areas:
1. End-to-End Pipeline Concurrency: Multi-threaded queries through Ragi with Vertex AI embeddings and store.
2. EmbeddingGenerator and LLMClient thread safety under high contention with ADC token refresh.
3. Continuous embedding generation memory leak benchmarking (150+ requests).
4. Zero-regression verification for Milestone 1 features under concurrent load.
"""

import gc
import os
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from piragi.config import RagiConfig
from piragi.core import Ragi
from piragi.embeddings import EmbeddingGenerator
from piragi.llm_client import LLMClient
from piragi.stores.protocol import VectorStoreProtocol
from piragi.types import Answer, Chunk, Citation


class MockVectorStore(VectorStoreProtocol):
    """Thread-safe in-memory vector store for concurrency stress testing."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._lock = threading.Lock()
        self._chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        with self._lock:
            self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Citation]:
        with self._lock:
            # Deterministic retrieval: return top_k citations based on stored chunks
            results = []
            for i, chunk in enumerate(self._chunks[:top_k]):
                score = 0.95 - (i * 0.05)
                results.append(
                    Citation(
                        source=chunk.source,
                        chunk=chunk.text,
                        score=score,
                        metadata=dict(chunk.metadata),
                    )
                )
            return results

    def get_all_citations(self, filters: Optional[Dict[str, Any]] = None) -> List[Citation]:
        with self._lock:
            return [
                Citation(source=c.source, chunk=c.text, score=0.5, metadata=dict(c.metadata))
                for c in self._chunks
            ]

    def delete(self, source: str) -> None:
        with self._lock:
            self._chunks = [c for c in self._chunks if c.source != source]

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()


def make_mock_llm():
    mock_llm = MagicMock()
    mock_llm.model = "gemini-1.5-pro"
    mock_llm.temperature = 0.1

    def mock_complete(messages, **kwargs):
        content = "Consensus is achieved via Paxos or Raft as discussed in [1]."
        for m in messages:
            msg_text = str(m.get("content", ""))
            if "alternative phrasings" in msg_text:
                content = "Alternative query 1\nAlternative query 2"
                break
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_llm.complete.side_effect = mock_complete
    mock_llm.complete_text.side_effect = lambda prompt, **kwargs: (
        "What are the fundamental principles of distributed consensus?"
        if ("high-level" in prompt.lower() or "step-back" in prompt.lower() or "abstract" in prompt.lower())
        else "Consensus requires majority quorum [1]."
    )
    return mock_llm


def mock_vertex_post(*args, **kwargs):
    payload = kwargs.get("json", {})
    instances = payload.get("instances", [])
    preds = [
        {"embeddings": {"values": [0.0123] * 768}}
        for _ in instances
    ]
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"predictions": preds}
    return resp


class TestRagiPipelineConcurrency:
    """Stress-test concurrent multi-threaded queries through Ragi with Vertex AI backend."""

    @pytest.fixture
    def populated_store(self):
        store = MockVectorStore(dimension=768)
        chunks = [
            Chunk(
                text=f"Paragraph {i}: Distributed systems require consensus algorithms like Paxos and Raft.",
                source="consensus_guide.md",
                chunk_index=i,
                metadata={"chunk_index": i, "parent_id": "consensus_p1", "parent_text": "Full Guide on Consensus: Raft and Paxos ensure consistency."},
            )
            for i in range(15)
        ]
        store.add(chunks)
        return store

    def test_concurrent_ragi_queries_vertex_embeddings(self, populated_store):
        """Stress-test 50 concurrent threads executing Ragi.ask() with Vertex AI backend."""
        embedder = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        assert embedder.is_vertex is True
        assert embedder.get_dimensions() == 768

        config = {
            "retrieval": {
                "context_ordering": "u_shaped",
                "auto_merge_parents": True,
                "prune_unreferenced_citations": True,
            }
        }

        mock_llm = make_mock_llm()

        with patch("piragi.llm_client.LLMClient", return_value=mock_llm), \
             patch("requests.post", side_effect=mock_vertex_post):
            ragi = Ragi(
                sources=None,
                store=populated_store,
                embedder=embedder,
                config=config,
            )
            ragi.llm_client = mock_llm
            ragi.retriever.llm_client = mock_llm
            ragi.retriever.client = mock_llm

            num_threads = 50
            results = []
            errors = []

            def worker_query(query_id: int):
                try:
                    q = f"How does consensus algorithm #{query_id} work under network partitions?"
                    ans = ragi.ask(q, top_k=3)
                    assert isinstance(ans, Answer)
                    assert ans.text is not None
                    assert len(ans.text) > 0
                    assert len(ans.citations) > 0
                    assert ans.query == q
                    results.append((query_id, ans))
                except Exception as exc:
                    errors.append((query_id, exc))

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_query, i) for i in range(num_threads)]
                for f in as_completed(futures):
                    f.result()

            assert len(errors) == 0, f"Encountered {len(errors)} errors during concurrent queries: {errors}"
            assert len(results) == num_threads, f"Expected {num_threads} completed answers, got {len(results)}"

    def test_high_concurrency_query_variation_and_stepback(self, populated_store):
        """Stress-test 30 concurrent threads executing StepBackPrompting + Hybrid variations."""
        embedder = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        config = {
            "retrieval": {
                "use_step_back": True,
                "context_ordering": "u_shaped",
                "auto_merge_parents": True,
                "prune_unreferenced_citations": True,
            }
        }

        mock_llm = make_mock_llm()

        with patch("piragi.llm_client.LLMClient", return_value=mock_llm), \
             patch("requests.post", side_effect=mock_vertex_post):
            ragi = Ragi(
                sources=None,
                store=populated_store,
                embedder=embedder,
                config=config,
            )
            ragi.llm_client = mock_llm
            ragi.retriever.llm_client = mock_llm
            ragi.retriever.client = mock_llm
            if ragi._step_back:
                ragi._step_back.llm_client = mock_llm

            num_threads = 30
            errors = []
            results = []

            def worker_stepback(idx: int):
                try:
                    q = f"Question {idx}: Why do Raft leader elections avoid split brain?"
                    ans = ragi.ask(q, top_k=4)
                    assert isinstance(ans, Answer)
                    assert len(ans.citations) > 0
                    results.append(ans)
                except Exception as e:
                    errors.append((idx, e))

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_stepback, i) for i in range(num_threads)]
                for f in as_completed(futures):
                    f.result()

            assert len(errors) == 0, f"Errors in concurrent stepback: {errors}"
            assert len(results) == num_threads


class TestEmbeddingAndLLMClientThreadSafety:
    """Stress-test thread safety of EmbeddingGenerator and LLMClient under token refresh races."""

    def test_concurrent_vertex_embed_query_and_chunks(self):
        """Stress-test simultaneous embed_query and embed_chunks across 50 threads."""
        embedder = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        num_threads = 50
        errors = []

        def worker_embedding(i: int):
            try:
                # Concurrent embed_query
                query_vec = embedder.embed_query(f"Multi-thread concurrent query {i} for stress test")
                assert len(query_vec) == 768
                assert isinstance(query_vec[0], float)

                # Concurrent embed_chunks
                chunks = [
                    Chunk(text=f"Worker {i} chunk {j} text content", source=f"doc_{i}.txt", chunk_index=j)
                    for j in range(5)
                ]
                embedded_chunks = embedder.embed_chunks(chunks)
                assert len(embedded_chunks) == 5
                for c in embedded_chunks:
                    assert len(c.embedding) == 768
            except Exception as e:
                errors.append((i, e))

        with patch("requests.post", side_effect=mock_vertex_post):
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_embedding, i) for i in range(num_threads)]
                for f in as_completed(futures):
                    f.result()

        assert len(errors) == 0, f"Concurrent embedding errors: {errors}"

    def test_concurrent_adc_token_refresh_race_condition(self):
        """Simulate expired ADC token and fire 50 threads simultaneously on LLMClient."""
        refresh_call_count = 0
        refresh_lock = threading.Lock()

        class MockADCAuth:
            def __init__(self):
                self.valid = False
                self.expired = True
                self.token = "initial-expired-token"

            def refresh(self, request):
                nonlocal refresh_call_count
                # Simulate small I/O delay during token refresh
                time.sleep(0.01)
                with refresh_lock:
                    refresh_call_count += 1
                self.valid = True
                self.expired = False
                self.token = f"refreshed-adc-token-{refresh_call_count}"

        mock_creds = MockADCAuth()

        with patch("google.auth.default", return_value=(mock_creds, "hemanth-hm")), \
             patch("openai.OpenAI") as mock_openai_cls:

            mock_openai_instance = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "ADC token verified response"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_openai_instance.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_openai_instance

            client = LLMClient(
                provider="vertexai",
                project_id="hemanth-hm",
                location="us-central1",
            )
            client._credentials = mock_creds

            # Run 50 concurrent requests
            num_threads = 50
            responses = []
            errors = []

            def worker_llm(idx: int):
                try:
                    res = client.complete_text(f"Query {idx}")
                    responses.append(res)
                except Exception as e:
                    errors.append((idx, e))

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_llm, i) for i in range(num_threads)]
                for f in as_completed(futures):
                    f.result()

            assert len(errors) == 0, f"ADC refresh race errors: {errors}"
            assert len(responses) == num_threads
            # All calls succeeded and token was properly updated
            assert mock_openai_instance.api_key.startswith("refreshed-adc-token")

    def test_llm_client_concurrent_rate_limit_backoff_stability(self):
        """Verify thread-safe retry backoff when 20 threads encounter 429 RateLimit."""
        import openai

        attempts = 0
        attempts_lock = threading.Lock()

        def flaky_create(*args, **kwargs):
            nonlocal attempts
            with attempts_lock:
                attempts += 1
                current_attempt = attempts

            # Fail the first 10 requests across all threads with 429
            if current_attempt <= 10:
                raise openai.RateLimitError(
                    message="Resource exhausted: 429",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            mock_choice = MagicMock()
            mock_choice.message.content = "Success after backoff"
            resp = MagicMock()
            resp.choices = [mock_choice]
            return resp

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create.side_effect = flaky_create
            mock_openai_cls.return_value = mock_instance

            client = LLMClient(provider="vertexai")
            client._client = mock_instance

            num_threads = 20
            results = []
            errors = []

            def worker_retry(i: int):
                try:
                    res = client.complete_text(f"Retry prompt {i}")
                    results.append(res)
                except Exception as e:
                    errors.append((i, e))

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_retry, i) for i in range(num_threads)]
                for f in as_completed(futures):
                    f.result()

            assert len(errors) == 0, f"Errors during rate limit backoff: {errors}"
            assert len(results) == num_threads
            assert all(r == "Success after backoff" for r in results)


class TestContinuousEmbeddingMemoryBenchmark:
    """Benchmark memory usage during continuous embedding generation (150+ requests)."""

    def test_continuous_embedding_generation_no_memory_leak(self):
        """Execute 150 consecutive embedding operations and verify zero memory accumulation."""
        gc.collect()
        tracemalloc.start()

        embedder = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")

        # Warm-up pass
        for _ in range(10):
            _ = embedder.embed_query("warm up query")

        gc.collect()
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_current, baseline_peak = tracemalloc.get_traced_memory()

        def mock_vertex_post(*args, **kwargs):
            payload = kwargs.get("json", {})
            instances = payload.get("instances", [])
            preds = [
                {"embeddings": {"values": [0.0123] * 768}}
                for _ in instances
            ]
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"predictions": preds}
            return resp

        with patch("requests.post", side_effect=mock_vertex_post):
            checkpoints = []
            num_iterations = 150

            for i in range(1, num_iterations + 1):
                # Mix single query and batch chunks
                query = f"Query {i}: Benchmark memory stability for continuous vector generation {i * 7}"
                _ = embedder.embed_query(query)

                if i % 3 == 0:
                    chunks = [
                        Chunk(
                            text=f"Continuous chunk {i}_{j}: Long technical description of cloud computing and storage.",
                            source=f"doc_{i}.txt",
                            chunk_index=j,
                        )
                        for j in range(10)
                    ]
                    _ = embedder.embed_chunks(chunks)

                # Record memory checkpoints every 30 iterations
                if i % 30 == 0:
                    gc.collect()
                    current_mem, peak_mem = tracemalloc.get_traced_memory()
                    current_mb = current_mem / (1024 * 1024)
                    peak_mb = peak_mem / (1024 * 1024)
                    checkpoints.append((i, current_mb, peak_mb))

            gc.collect()
            final_snapshot = tracemalloc.take_snapshot()
            final_current, final_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            final_current_mb = final_current / (1024 * 1024)
            baseline_current_mb = baseline_current / (1024 * 1024)
            net_growth_mb = final_current_mb - baseline_current_mb

            # Calculate top allocations between baseline and final
            top_stats = final_snapshot.compare_to(baseline_snapshot, "lineno")
            total_diff_bytes = sum(stat.size_diff for stat in top_stats)
            total_diff_kb = total_diff_bytes / 1024

            print("\n--- Memory Leak Benchmark Results ---")
            print(f"Baseline current: {baseline_current_mb:.3f} MB")
            for step, cur_mb, pk_mb in checkpoints:
                print(f"Iteration {step}: Current={cur_mb:.3f} MB, Peak={pk_mb:.3f} MB")
            print(f"Final current: {final_current_mb:.3f} MB")
            print(f"Net memory growth: {net_growth_mb:.3f} MB ({total_diff_kb:.1f} KB diff)")

            # Verify memory stability: net growth must be under 1.5 MB across 150 heavy iterations
            assert net_growth_mb < 1.5, (
                f"Potential memory leak detected: heap grew by {net_growth_mb:.2f} MB "
                f"over {num_iterations} iterations"
            )

    def test_continuous_offline_fallback_embedding_memory_benchmark(self):
        """Execute 200 consecutive offline fallback embeddings to test memory footprint of numpy generation."""
        gc.collect()
        tracemalloc.start()

        embedder = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        # Ensure offline mode by forcing credentials and token to None
        embedder._credentials = None
        embedder.project_id = None

        gc.collect()
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_current, _ = tracemalloc.get_traced_memory()

        num_iterations = 200
        for i in range(1, num_iterations + 1):
            _ = embedder.embed_query(f"Offline memory test query {i}")
            if i % 2 == 0:
                chunks = [
                    Chunk(
                        text=f"Offline chunk {i}_{j} description for memory stress",
                        source=f"doc_{i}.txt",
                        chunk_index=j,
                    )
                    for j in range(15)
                ]
                _ = embedder.embed_chunks(chunks)

        gc.collect()
        final_current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        net_growth_mb = (final_current - baseline_current) / (1024 * 1024)
        print(f"\nOffline net memory growth over 200 iterations: {net_growth_mb:.3f} MB")
        assert net_growth_mb < 1.0, f"Offline embedding memory growth too high: {net_growth_mb:.2f} MB"



class TestMilestone1RegressionUnderConcurrency:
    """Stress-test that Milestone 1 enhancements maintain invariance under concurrent load."""

    def test_u_shaped_ordering_and_automerging_concurrent_invariance(self):
        """Concurrently verify U-shaped attention and auto-merging across 30 parallel workers."""
        from piragi.pipelines.retrieval import RetrievalPipeline
        from piragi.retrieval import Retriever

        mock_llm_client = make_mock_llm()

        retriever = Retriever(
            llm_client=mock_llm_client,
            context_ordering="u_shaped",
            prune_unreferenced_citations=True,
        )

        embedder = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        store = MockVectorStore(dimension=768)

        pipeline = RetrievalPipeline(
            embedder=embedder,
            store=store,
            retriever=retriever,
            use_hierarchical=True,
            auto_merge_parents=True,
            context_ordering="u_shaped",
            prune_unreferenced_citations=True,
        )

        num_workers = 30
        errors = []

        def worker_pipeline(worker_id: int):
            try:
                # Construct child citations belonging to two distinct parents
                citations = [
                    Citation(
                        source=f"doc_{worker_id}.txt",
                        chunk=f"Child {i} of parent A",
                        score=0.90 - (i * 0.05),
                        metadata={
                            "parent_id": f"parent_A_{worker_id}",
                            "parent_text": f"Parent A full unified text for worker {worker_id}",
                            "chunk_index": i,
                        },
                    )
                    for i in range(3)
                ] + [
                    Citation(
                        source=f"doc_{worker_id}.txt",
                        chunk=f"Child {j} of parent B",
                        score=0.70 - (j * 0.05),
                        metadata={
                            "parent_id": f"parent_B_{worker_id}",
                            "parent_text": f"Parent B full unified text for worker {worker_id}",
                            "chunk_index": j,
                        },
                    )
                    for j in range(2)
                ]

                # 1. Test auto-merging deduplication
                merged = pipeline._expand_to_parent_context(citations)
                assert len(merged) == 2, f"Expected 2 merged parents, got {len(merged)}"
                assert merged[0].metadata["parent_id"] == f"parent_A_{worker_id}"
                assert merged[0].metadata["merged_children_count"] == 3
                assert merged[1].metadata["parent_id"] == f"parent_B_{worker_id}"
                assert merged[1].metadata["merged_children_count"] == 2

                # 2. Test U-shaped ordering
                from piragi.retrieval import reorder_u_shaped
                ordered = reorder_u_shaped(citations)
                assert len(ordered) == len(citations)
                # Best citation (score 0.90) at index 0, second best (score 0.85) at index -1
                assert ordered[0].score == 0.90
                assert ordered[-1].score == 0.85

                # 3. Test generate_answer with reorder=True
                ans = retriever.generate_answer(
                    query="What is consensus?",
                    citations=citations,
                    reorder=True,
                )
                assert isinstance(ans, Answer)
                assert len(ans.citations) > 0

            except Exception as e:
                errors.append((worker_id, e))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_pipeline, i) for i in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Milestone 1 concurrent invariant errors: {errors}"
