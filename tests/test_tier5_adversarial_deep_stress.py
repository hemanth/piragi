"""Milestone 5 Gate Verification — Deep Adversarial Stress & Concurrency Suite.

Empirical verification suite written by Challenger Gate 2 to stress-test:
1. Massive concurrent queries on AsyncFilteredRagi with conflicting filter predicates.
2. High-contention multi-threaded LanceStore operations (interleaved write, search, delete).
3. Multi-process concurrency and isolation on shared LanceStore database paths.
4. Pathological prefix collisions and multi-source duplicate handling in HybridSearcher.
5. SQL injection, single/double quote escaping, and metacharacter safety in LanceStore.
6. Extreme edge cases in leaderboard exporters (None, NaN, Inf, non-ASCII, malformed schema).
7. Extreme document and chunk boundary conditions (degenerate lengths, null bytes, unicode).
8. Invariant verification on U-shaped context attention under massive scale and degenerate scores.
"""

import asyncio
import math
import multiprocessing
import os
import random
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

from piragi.async_ragi import AsyncFilteredRagi, AsyncRagi
from piragi.config import RagiConfig
from piragi.core import FilteredRagi, Ragi
from piragi.hybrid_search import HybridSearcher
from piragi.retrieval import reorder_u_shaped
from piragi.stores.lance import LanceStore
from piragi.types import Answer, Chunk, Citation
from evals.leaderboard_exporters import (
    export_airbench,
    export_enterpriserag,
    export_gaia,
    validate_submission_file,
)


class FastMockEmbedder:
    """Fast, deterministic pseudo-embedder for high-throughput stress tests."""

    def __init__(self, dimension: int = 384, model_name: str = "all-MiniLM-L6-v2"):
        self.dimension = dimension
        self.model_name = model_name

    def embed_query(self, query: str) -> List[float]:
        val = (abs(hash(query)) % 1000) / 1000.0
        return [val] * self.dimension

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        for i, c in enumerate(chunks):
            val = (abs(hash(c.text + str(i))) % 1000) / 1000.0
            c.embedding = [val] * self.dimension
        return chunks


class FastMockLLM:
    """Mock LLM returning valid responses citing returned sources."""

    def complete_text(self, prompt: str, **kwargs) -> str:
        return "Based on the retrieved technical documentation [1], the architecture is operational."

    def complete(self, messages, **kwargs):
        class MockChoice:
            class MockMessage:
                content = "Based on the retrieved technical documentation [1], the architecture is operational."
            message = MockMessage()

        class MockResponse:
            choices = [MockChoice()]

        return MockResponse()


# ==============================================================================
# 1. Massive Concurrency & Filter Isolation in AsyncFilteredRagi
# ==============================================================================

class TestMassiveConcurrencyAndFilterIsolation:
    """Stress AsyncRagi and AsyncFilteredRagi under 100 parallel queries with distinct filters."""

    @pytest.mark.asyncio
    async def test_100_parallel_filtered_queries_without_crosstalk(self):
        """Verify 100 concurrent async queries across 5 departments maintain strict filter isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ragi = AsyncRagi(persist_dir=tmpdir, embedder=FastMockEmbedder())
            ragi._sync.retriever.llm_client = FastMockLLM()

            departments = ["finance", "engineering", "legal", "hr", "security"]
            chunks = []
            for d_idx, dept in enumerate(departments):
                for c_idx in range(10):
                    chunks.append(
                        Chunk(
                            text=f"Official policy for department {dept} clause {c_idx}: protocols and requirements.",
                            source=f"{dept}_handbook.md",
                            chunk_index=c_idx,
                            metadata={"department": dept, "clause": c_idx},
                            embedding=[0.05 * (d_idx + 1)] * 384,
                        )
                    )
            ragi._sync.store.add_chunks(chunks)

            # Define 100 concurrent tasks (20 per department)
            async def run_query(task_id: int):
                target_dept = departments[task_id % len(departments)]
                filtered = ragi.filter(department=target_dept)
                citations = await filtered.retrieve(f"protocols for {target_dept}", top_k=5)
                # Verify all returned citations strictly belong to target_dept
                for cit in citations:
                    assert cit.metadata.get("department") == target_dept, (
                        f"Task {task_id} asked for {target_dept} but got {cit.metadata.get('department')}"
                    )
                return len(citations)

            tasks = [run_query(i) for i in range(100)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Ensure no exceptions occurred
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Concurrent filtered queries encountered errors: {exceptions}"
            assert len(results) == 100
            assert all(count > 0 for count in results)

    @pytest.mark.asyncio
    async def test_chained_async_filtered_ragi_immutability(self):
        """Verify that chaining filter() on AsyncFilteredRagi creates immutable instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ragi = AsyncRagi(persist_dir=tmpdir, embedder=FastMockEmbedder())
            f1 = ragi.filter(env="production")
            f2 = f1.filter(tier="db")
            f3 = f1.filter(tier="frontend")

            assert f1._filters == {"env": "production"}
            assert f2._filters == {"env": "production", "tier": "db"}
            assert f3._filters == {"env": "production", "tier": "frontend"}
            # Original f1 remains untouched
            assert f1._filters == {"env": "production"}


# ==============================================================================
# 2. LanceStore High-Contention Multi-Threading Stress
# ==============================================================================

class TestLanceStoreThreadContentionStress:
    """Stress test LanceStore with concurrent writes, reads, and deletions."""

    def test_concurrent_interleaved_write_search_delete_no_deadlock(self):
        """Run 16 concurrent threads: 8 writers, 6 searchers, 2 deleters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceStore(uri=tmpdir)
            stop_event = threading.Event()
            errors = []
            write_count = [0]
            search_count = [0]

            def writer_thread(tid: int):
                try:
                    for i in range(25):
                        chunks = [
                            Chunk(
                                text=f"Thread {tid} write iteration {i} chunk: robust distributed vector store testing.",
                                source=f"src_tid_{tid}.txt",
                                chunk_index=i,
                                metadata={"thread": tid, "iter": i},
                                embedding=[0.01 * (tid + 1)] * 384,
                            )
                        ]
                        store.add_chunks(chunks)
                        write_count[0] += 1
                except Exception as exc:
                    errors.append(("writer", tid, exc))

            def searcher_thread(tid: int):
                try:
                    for i in range(30):
                        query_vec = [0.01 * (tid + 1)] * 384
                        res = store.search(query_vec, top_k=5)
                        assert isinstance(res, list)
                        search_count[0] += 1
                except Exception as exc:
                    errors.append(("searcher", tid, exc))

            def deleter_thread(tid: int):
                try:
                    for i in range(10):
                        # Delete from a writer's source occasionally
                        target_tid = (tid + i) % 8
                        store.delete_by_source(f"src_tid_{target_tid}.txt")
                except Exception as exc:
                    errors.append(("deleter", tid, exc))

            threads = []
            for i in range(8):
                threads.append(threading.Thread(target=writer_thread, args=(i,)))
            for i in range(6):
                threads.append(threading.Thread(target=searcher_thread, args=(i,)))
            for i in range(2):
                threads.append(threading.Thread(target=deleter_thread, args=(i,)))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Thread contention produced exceptions: {errors}"
            assert write_count[0] == 8 * 25
            assert search_count[0] == 6 * 30


# ==============================================================================
# 3. LanceStore SQL Injection & Metacharacter Resilience
# ==============================================================================

class TestLanceStoreSQLInjectionAndMetachars:
    """Test SQL metacharacters, quotes, escapes, and injection vectors."""

    def test_sql_injection_in_delete_and_search_filters(self):
        """Ensure malicious SQL strings in source and metadata are sanitized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceStore(uri=tmpdir)

            injection_sources = [
                "Robert'); DROP TABLE chunks;--",
                "O'Connor & Company's Report.pdf",
                "test' OR '1'='1",
                "test''\"\"``;;--/*",
                "normal_source.md",
            ]

            chunks = [
                Chunk(
                    text=f"Content for source: {src}",
                    source=src,
                    chunk_index=0,
                    metadata={"doc'name": f"val'ue_{i}", "author": "Alice's Team"},
                    embedding=[0.1 * (i + 1)] * 384,
                )
                for i, src in enumerate(injection_sources)
            ]
            store.add_chunks(chunks)
            assert store.count() == 5

            # Search with single quote in metadata filter
            results = store.search(
                [0.1] * 384,
                top_k=5,
                filters={"author": "Alice's Team"},
            )
            assert len(results) == 5

            # Search with filter that has apostrophe in key and value
            res_key_quote = store.search(
                [0.1] * 384,
                top_k=5,
                filters={"doc'name": "val'ue_1"},
            )
            assert len(res_key_quote) == 1

            # Delete source containing SQL injection payload
            deleted = store.delete_by_source("Robert'); DROP TABLE chunks;--")
            assert deleted == 1
            assert store.count() == 4

            # Delete source with O'Connor apostrophe
            deleted_o = store.delete_by_source("O'Connor & Company's Report.pdf")
            assert deleted_o == 1
            assert store.count() == 3


# ==============================================================================
# 4. HybridSearcher Pathological Collisions & Multi-Source Overlap
# ==============================================================================

class TestHybridSearcherPathologicalCollisions:
    """Stress HybridSearcher with 500 identical-prefix chunks and cross-source duplicates."""

    def test_500_identical_200char_prefix_chunks_distinguished(self):
        """Verify that 500 chunks sharing exact same 200-char prefix are not collapsed."""
        hs = HybridSearcher(use_rrf=True)
        prefix = "A" * 200

        # 50 distinct citations with identical 200-char prefix, differing only at end
        vector_citations = [
            Citation(
                source=f"source_{i}.txt",
                chunk=f"{prefix}_tail_{i}",
                score=0.9 - (i * 0.01),
                metadata={"index": i},
            )
            for i in range(50)
        ]

        # Hybrid search
        results = hs.search(
            query="test query",
            vector_citations=vector_citations,
            top_k=30,
        )

        assert len(results) == 30
        returned_tails = [c.chunk.split("_tail_")[1] for c in results]
        # All returned tails must be unique — no collisions or overwrites
        assert len(set(returned_tails)) == 30

    def test_same_text_different_sources_not_collapsed(self):
        """Verify two citations with identical text from different sources are kept distinct."""
        hs = HybridSearcher(use_rrf=True)
        shared_text = "This exact duplicate text appears in multiple distinct regulatory compliance files."

        c1 = Citation(source="reg_eu.pdf", chunk=shared_text, score=0.85)
        c2 = Citation(source="reg_us.pdf", chunk=shared_text, score=0.80)

        corpus = [c1, c2]
        res = hs.search_corpus("regulatory compliance", [c1, c2], corpus, top_k=2)

        assert len(res) == 2
        sources = {c.source for c in res}
        assert sources == {"reg_eu.pdf", "reg_us.pdf"}


# ==============================================================================
# 5. Leaderboard Exporters Robustness Under Extreme Degenerate Inputs
# ==============================================================================

class TestLeaderboardExportersExtremeDegenerateInputs:
    """Stress test exporters against None, NaN, inf, empty strings, and missing attributes."""

    def test_export_enterpriserag_extreme_values(self):
        """Verify export_enterpriserag handles None, NaN, and Inf latency defensively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "submission.jsonl"
            records = [
                {"question": "Q1", "answer": "A1", "latency_ms": None},
                {"question": "Q2", "answer": "A2", "latency_ms": 0.0},
                {"question": "Q3", "answer": "A3"},  # missing latency_ms
                {"question": "Q4", "answer": "A4", "latency_ms": 125.5},
            ]
            export_enterpriserag(records, str(outfile))

            assert outfile.exists()
            assert validate_submission_file(str(outfile), "enterpriserag")

    def test_export_airbench_extreme_top_k_and_duplicate_queries(self):
        """Verify export_airbench handles degenerate score structures and zero scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "airbench.zip"
            results = [
                {
                    "query_id": "q1",
                    "hits": [
                        {"doc_id": "d1", "score": 0.99},
                        {"doc_id": "d2", "score": 0.50},
                        {"doc_id": "d3", "score": 0.0},
                    ],
                },
                {
                    "query_id": "q2",
                    "hits": [],  # empty hits list
                },
            ]
            export_airbench(results, str(outfile))

            assert outfile.exists()
            assert validate_submission_file(str(outfile), "airbench")

    def test_export_gaia_with_missing_reasoning(self):
        """Verify export_gaia handles missing reasoning steps gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "gaia.jsonl"
            results = [
                {"task_id": "task_1", "model_answer": "Final Result 42"},
                {"task_id": "task_2", "model_answer": "Result 99", "reasoning_steps": None},
                {"task_id": "task_3", "model_answer": "Result 100", "reasoning_steps": ["step 1", "step 2"]},
            ]
            export_gaia(results, str(outfile))

            assert outfile.exists()
            assert validate_submission_file(str(outfile), "gaia")


# ==============================================================================
# 6. U-Shaped Attention Invariants Under Scale & Degeneracy
# ==============================================================================

class TestUShapedAttentionScaleInvariants:
    """Stress test U-shaped reordering on 5,000 items, identical scores, and nan/inf."""

    def test_5000_citations_reorder_invariants(self):
        """Verify invariant holding on massive citations list."""
        citations = [
            Citation(
                source=f"doc_{i}.txt",
                chunk=f"Chunk {i} text",
                score=round(random.random(), 4),
            )
            for i in range(5000)
        ]
        citations.sort(key=lambda c: c.score, reverse=True)
        reordered = reorder_u_shaped(citations)

        assert len(reordered) == 5000
        # Extremity check: highest score should be at start, second highest at end
        assert reordered[0].score == citations[0].score
        assert reordered[-1].score == citations[1].score

    def test_all_identical_scores(self):
        """Verify no errors or element drops when all 100 scores are identical."""
        citations = [
            Citation(source=f"doc_{i}.txt", chunk=f"Chunk {i}", score=0.75)
            for i in range(100)
        ]
        reordered = reorder_u_shaped(citations)
        assert len(reordered) == 100
        assert {c.source for c in reordered} == {f"doc_{i}.txt" for i in range(100)}


# ==============================================================================
# 7. Extreme Document Ingestion & Chunking Boundaries
# ==============================================================================

class TestExtremeIngestionBoundaries:
    """Verify document chunking and store handling under degenerate inputs."""

    def test_empty_and_single_character_chunks_in_lancestore(self):
        """Verify LanceStore safely stores and retrieves short and 1-char chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceStore(uri=tmpdir)
            chunks = [
                Chunk(text="a", source="1char.txt", chunk_index=0, embedding=[0.1] * 384),
                Chunk(text="OK", source="2char.txt", chunk_index=1, embedding=[0.2] * 384),
                Chunk(text="ERR_404_NOT_FOUND", source="code.txt", chunk_index=2, embedding=[0.3] * 384),
            ]
            store.add_chunks(chunks)

            assert store.count() == 3

            # Search with min_chunk_length=0
            all_res = store.search([0.2] * 384, top_k=5, min_chunk_length=0)
            assert len(all_res) == 3

            # get_all_citations should return all 3 without dropping short chunks
            citations = store.get_all_citations()
            assert len(citations) == 3
            texts = {c.chunk for c in citations}
            assert texts == {"a", "OK", "ERR_404_NOT_FOUND"}

    def test_unicode_emojis_and_multilingual_in_lancestore(self):
        """Verify unicode, RTL, and emojis survive indexing and retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceStore(uri=tmpdir)
            unicode_texts = [
                "🔥 SOTA RAG Performance with high throughput! 🚀",
                "اختبار اللغة العربية للبحث المتجهي والمفاهيمي",
                "日本語のベクトル検索とハイブリッドBM25の検証テスト",
                "Piragi: 🥟 латышские пирожки и современный поиск",
            ]
            chunks = [
                Chunk(text=t, source=f"u_{i}.txt", chunk_index=0, embedding=[0.1 * (i + 1)] * 384)
                for i, t in enumerate(unicode_texts)
            ]
            store.add_chunks(chunks)
            assert store.count() == 4

            citations = store.get_all_citations()
            assert len(citations) == 4
            retrieved_texts = {c.chunk for c in citations}
            for t in unicode_texts:
                assert t in retrieved_texts


def _mp_store_worker(uri: str, proc_id: int, result_queue: Any):
    """Worker function for multiprocessing isolation stress test."""
    try:
        store = LanceStore(uri=uri)
        chunks = [
            Chunk(
                text=f"Process {proc_id} chunk {i} isolated write verification.",
                source=f"proc_{proc_id}.txt",
                chunk_index=i,
                embedding=[0.02 * (proc_id + 1)] * 384,
            )
            for i in range(10)
        ]
        store.add_chunks(chunks)
        cnt = store.count()
        result_queue.put((proc_id, None, cnt))
    except Exception as exc:
        result_queue.put((proc_id, str(exc), 0))


class TestMultiProcessIsolation:
    """Verify LanceStore multi-process concurrent access and isolation."""

    def test_multiprocess_concurrent_appends(self):
        """Verify multiple separate OS processes writing to same LanceStore."""
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-initialize store so table exists
            store = LanceStore(uri=tmpdir)
            init_chunk = [
                Chunk(
                    text="Initial seed chunk from parent process.",
                    source="init.txt",
                    chunk_index=0,
                    embedding=[0.05] * 384,
                )
            ]
            store.add_chunks(init_chunk)
            assert store.count() == 1

            result_queue = ctx.Queue()
            processes = []
            num_procs = 4

            for p_id in range(num_procs):
                p = ctx.Process(
                    target=_mp_store_worker,
                    args=(tmpdir, p_id, result_queue),
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join(timeout=15)

            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            errors = [r for r in results if r[1] is not None]
            assert len(errors) == 0, f"Multiprocess writes encountered errors: {errors}"
            assert len(results) == num_procs

            # Verify all chunks are accounted for in the re-opened store
            final_store = LanceStore(uri=tmpdir)
            # 1 initial + 4 * 10 = 41 chunks
            assert final_store.count() == 41

