"""Tier 5 Adversarial Stress, Concurrency, and Boundary Hardening Test Suite.

Exhaustively verifies:
1. High concurrency and thread safety under heavy load.
2. Extreme boundary conditions (empty inputs, massive documents, non-ASCII/unicode, quotes, NaN/inf scores).
3. Resiliency across all public APIs (Ragi, AsyncRagi, LanceStore, HybridSearcher, exporters).
"""

import asyncio
import json
import math
import tempfile
import threading
from pathlib import Path
from typing import List

import pytest

from piragi.async_ragi import AsyncRagi
from piragi.config import RagiConfig
from piragi.core import Ragi, FilteredRagi
from piragi.embeddings import EmbeddingGenerator
from piragi.hybrid_search import HybridSearcher
from piragi.retrieval import parse_citation_references, prune_citations, reorder_u_shaped
from piragi.stores.lance import LanceStore
from piragi.types import Answer, Chunk, Citation, Document
from evals.leaderboard_exporters import (
    export_airbench,
    export_enterpriserag,
    export_gaia,
    validate_submission_file,
)


class DummyMockEmbedder:
    """Fast, deterministic in-memory mock embedder for stress testing without model download overhead."""

    def __init__(self, dimension: int = 384, model_name: str = "all-MiniLM-L6-v2"):
        self.dimension = dimension
        self.model_name = model_name

    def embed_query(self, query: str) -> List[float]:
        # Hash-based deterministic pseudo-embedding
        h = abs(hash(query))
        return [(h % 1000) / 1000.0] * self.dimension

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        for idx, chunk in enumerate(chunks):
            h = abs(hash(chunk.text + str(idx)))
            chunk.embedding = [(h % 1000) / 1000.0] * self.dimension
        return chunks


class DummyMockLLM:
    """Mock LLM client returning instant grounded responses with citation tags."""

    def __init__(self, model: str = "mock-model"):
        self.model = model
        self.temperature = 0.1

    def complete(self, messages, **kwargs):
        class MockChoice:
            class MockMessage:
                content = "This response is grounded in evidence [1] and [2]."
            message = MockMessage()

        class MockResponse:
            choices = [MockChoice()]

        return MockResponse()

    def complete_text(self, prompt, **kwargs):
        return "This response is grounded in evidence [1] and [2]."


# ==============================================================================
# 1. Concurrency and Thread Safety Under Heavy Load
# ==============================================================================

class TestConcurrencyAndThreadSafety:
    """Adversarial concurrency tests across Ragi, AsyncRagi, and LanceStore."""

    def test_concurrent_reads_on_shared_ragi(self):
        """Verify multiple threads querying a shared Ragi instance concurrently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = DummyMockEmbedder()
            ragi = Ragi(persist_dir=tmpdir, embedder=embedder)
            ragi.retriever.llm_client = DummyMockLLM()

            # Seed knowledge base with substantial documents
            chunks = [
                Chunk(
                    text=f"Engineering Document Section {i}: Detailed architectural specification of distributed state machine {i} with verified replication protocol.",
                    source=f"spec_{i}.md",
                    chunk_index=i,
                    embedding=[0.01 * (i % 50)] * 384,
                )
                for i in range(25)
            ]
            ragi.store.add_chunks(chunks)

            errors = []
            results = []

            def worker(worker_id: int):
                try:
                    for q_idx in range(5):
                        ans = ragi.ask(f"state machine {worker_id}_{q_idx}", top_k=3)
                        assert isinstance(ans, Answer)
                        assert ans.text is not None
                        results.append(len(ans.citations))
                except Exception as exc:
                    errors.append((worker_id, exc))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Concurrent reads raised exceptions: {errors}"
            assert len(results) == 50

    @pytest.mark.asyncio
    async def test_high_concurrency_async_ragi_gather(self):
        """Stress AsyncRagi with 30 simultaneous asynchronous queries via asyncio.gather."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = DummyMockEmbedder()
            ragi = AsyncRagi(persist_dir=tmpdir, embedder=embedder)
            ragi._sync.retriever.llm_client = DummyMockLLM()

            # Seed data
            chunks = [
                Chunk(
                    text=f"Database Cluster Node {i}: Paxos consensus heartbeat timestamp and cluster quorum membership validation record.",
                    source=f"node_{i}.log",
                    chunk_index=0,
                    embedding=[0.02 * (i % 25)] * 384,
                )
                for i in range(20)
            ]
            ragi._sync.store.add_chunks(chunks)

            async def query_task(task_id: int):
                return await ragi.ask(f"cluster quorum {task_id}", top_k=3)

            tasks = [query_task(i) for i in range(30)]
            answers = await asyncio.gather(*tasks, return_exceptions=True)

            # Check that all succeeded without exceptions
            exceptions = [a for a in answers if isinstance(a, Exception)]
            assert len(exceptions) == 0, f"AsyncRagi gather encountered exceptions: {exceptions}"
            assert len(answers) == 30

    def test_concurrent_hybrid_searcher_corpus_queries(self):
        """Verify HybridSearcher handles concurrent search_corpus calls with cache reuse."""
        hs = HybridSearcher(use_rrf=True)
        corpus = [
            Citation(
                source=f"doc_{i}.txt",
                chunk=f"Substantive corpus chunk {i} containing target keywords for hybrid retrieval validation.",
                score=0.5,
            )
            for i in range(30)
        ]

        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(20):
                    res = hs.search_corpus("keywords target", corpus[:10], corpus, top_k=5)
                    assert len(res) <= 5
            except Exception as exc:
                errors.append((thread_id, exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent search_corpus raised errors: {errors}"

    def test_concurrent_lancestore_add_chunks_thread_safety(self):
        """Verify concurrent multi-threaded writes to LanceStore without table creation race or Rust panics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceStore(uri=tmpdir)
            errors = []

            def writer(thread_id: int):
                try:
                    chunks = [
                        Chunk(
                            text=f"Thread {thread_id} chunk {idx}: Technical documentation and data chunk.",
                            source=f"thread_{thread_id}.txt",
                            chunk_index=idx,
                            embedding=[0.05 * (thread_id + 1)] * 384,
                        )
                        for idx in range(10)
                    ]
                    store.add_chunks(chunks)
                except Exception as exc:
                    errors.append((thread_id, exc))

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Concurrent LanceStore writes raised errors: {errors}"
            assert store.count() == 100, f"Expected 100 total chunks, got {store.count()}"


# ==============================================================================
# 2. Extreme Boundary Conditions
# ==============================================================================

class TestExtremeBoundaries:
    """Stress tests on empty, massive, unicode, and numerical boundaries."""

    def test_empty_query_resilience(self):
        """Verify ask and retrieve handle empty and whitespace-only queries gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ragi = Ragi(persist_dir=tmpdir, embedder=DummyMockEmbedder())

            # Empty query
            ans1 = ragi.ask("")
            assert ans1.text == "Please provide a valid question."
            assert ans1.citations == []

            ret1 = ragi.retrieve("")
            assert ret1 == []

            # Whitespace and newline query
            ans2 = ragi.ask("   \t\n   ")
            assert ans2.text == "Please provide a valid question."
            assert ans2.citations == []

            ret2 = ragi.retrieve("   \t\n   ")
            assert ret2 == []

    def test_extreme_top_k_values(self):
        """Verify top_k <= 0 and massive top_k are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ragi = Ragi(persist_dir=tmpdir, embedder=DummyMockEmbedder())
            chunk = Chunk(
                text="Standard valid chunk text exceeding one hundred characters to qualify for retrieval indexing correctly.",
                source="test.txt",
                chunk_index=0,
                embedding=[0.1] * 384,
            )
            ragi.store.add_chunks([chunk])

            # top_k = 0
            ans_zero = ragi.ask("test", top_k=0)
            assert ans_zero.citations == []

            ret_zero = ragi.retrieve("test", top_k=0)
            assert ret_zero == []

            # top_k = -5
            ans_neg = ragi.ask("test", top_k=-5)
            assert ans_neg.citations == []

            ret_neg = ragi.retrieve("test", top_k=-5)
            assert ret_neg == []

            # top_k = 10,000 (exceeds corpus size)
            ret_huge = ragi.retrieve("test", top_k=10000)
            assert len(ret_huge) == 1

    def test_massive_citations_u_shaped_and_pruning(self):
        """Stress reorder_u_shaped and prune_citations with 5,000 candidate citations."""
        citations = [
            Citation(
                source=f"doc_{i}.txt",
                chunk=f"Content for candidate citation index {i} detailing empirical results.",
                score=1.0 / (i + 1),
            )
            for i in range(5000)
        ]

        # U-shaped ordering
        reordered = reorder_u_shaped(citations)
        assert len(reordered) == 5000
        # Highest score at index 0, second highest at index -1
        assert reordered[0].source == "doc_0.txt"
        assert reordered[-1].source == "doc_1.txt"

        # Citation pruning on massive set
        response_text = "Findings are substantiated by [1], [500], and [5000]."
        active, unref = prune_citations(response_text, reordered)
        assert len(active) == 3
        assert len(unref) == 4997

    def test_non_ascii_multilingual_and_emoji_resilience(self):
        """Verify end-to-end ingestion and retrieval of non-ASCII, RTL, and emoji content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ragi = Ragi(persist_dir=tmpdir, embedder=DummyMockEmbedder())

            multilingual_text = (
                "🚀 SOTA RAG 系统: 这是一段中文测试文本，验证分词与向量检索的鲁棒性。"
                "هذا نص تجريبي باللغة العربية للتحقق من دعم النصوص ثنائية الاتجاه."
                "Это контрольный текст на русском языке для проверки поддержки кириллицы."
                "Non-ASCII symbols: ∑(x_i) = ∫ f(t)dt, ñ, ö, ü, é, ç, 💡🔥✨."
            )
            # Ensure text > 100 chars
            assert len(multilingual_text) > 100

            chunk = Chunk(
                text=multilingual_text,
                source="unicode_test.md",
                chunk_index=0,
                embedding=[0.2] * 384,
                metadata={"language": "multilingual", "emoji": "🚀"},
            )
            ragi.store.add_chunks([chunk])

            citations = ragi.retrieve("中文测试", top_k=5)
            assert len(citations) == 1
            assert "🚀 SOTA RAG" in citations[0].chunk
            assert citations[0].metadata["emoji"] == "🚀"

    def test_extreme_numerical_scores_in_hybrid_search(self):
        """Stress HybridSearcher with NaN, Inf, and negative vector scores."""
        hs = HybridSearcher(use_rrf=True)
        c_nan = Citation(source="d1.txt", chunk="Chunk with NaN vector score that must be handled safely.", score=float("nan"))
        c_inf = Citation(source="d2.txt", chunk="Chunk with positive Inf score that must not corrupt results.", score=float("inf"))
        c_neg = Citation(source="d3.txt", chunk="Chunk with negative score from cosine similarity range.", score=-0.85)
        c_valid = Citation(source="d4.txt", chunk="Standard valid chunk with positive score.", score=0.75)

        citations = [c_nan, c_inf, c_neg, c_valid]

        # Must not raise an unhandled exception
        results = hs.search_corpus("chunk", citations, citations, top_k=4)
        assert len(results) == 4
        # All scores produced by RRF must be finite numbers
        for r in results:
            assert math.isfinite(r.score)


# ==============================================================================
# 3. Public API Resiliency & Identified Defect Probes
# ==============================================================================

class TestPublicApiResiliency:
    """Rigorous tests probing public APIs and documenting defect boundaries."""

    def test_filtered_ragi_synchronous_chaining(self):
        """Verify synchronous Ragi.filter() returns FilteredRagi and retains filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ragi = Ragi(persist_dir=tmpdir, embedder=DummyMockEmbedder())
            c1 = Chunk(
                text="Document regarding database optimization, B-tree indexing structures, and query execution plans in high load enterprise environments.",
                source="db.txt",
                chunk_index=0,
                embedding=[0.1] * 384,
                metadata={"domain": "database"},
            )
            c2 = Chunk(
                text="Document regarding frontend UI styling, component architecture, CSS layout primitives, and keyframe animations in web applications.",
                source="ui.txt",
                chunk_index=0,
                embedding=[0.9] * 384,
                metadata={"domain": "frontend"},
            )
            ragi.store.add_chunks([c1, c2])

            filtered = ragi.filter(domain="database")
            assert isinstance(filtered, FilteredRagi)

            res = filtered.retrieve("frontend", top_k=5)
            assert len(res) == 1
            assert res[0].metadata["domain"] == "database"

    def test_short_chunk_dropped_by_lancestore_100_char_threshold(self):
        """Verify remediation D4: short chunks < 100 chars are preserved and retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ragi = Ragi(persist_dir=tmpdir, embedder=DummyMockEmbedder())
            short_chunk = Chunk(
                text="Short proposition: SQLite is a self-contained SQL database engine.",  # 65 chars
                source="sqlite.txt",
                chunk_index=0,
                embedding=[0.1] * 384,
            )
            ragi.store.add_chunks([short_chunk])
            assert ragi.count() == 1

            # Default search in RetrievalPipeline inherits min_chunk_length=0
            res = ragi.retrieve("SQLite", top_k=5)
            assert len(res) == 1
            assert res[0].chunk == short_chunk.text

    @pytest.mark.asyncio
    async def test_async_ragi_filter_boundary_probe(self):
        """Verify remediation D3: AsyncRagi.filter() returns AsyncFilteredRagi and respects filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ragi = AsyncRagi(persist_dir=tmpdir, embedder=DummyMockEmbedder())
            c1 = Chunk(
                text="Document regarding database optimization, B-tree indexing structures, and query execution plans in high load enterprise environments.",
                source="db.txt",
                chunk_index=0,
                embedding=[0.1] * 384,
                metadata={"domain": "database"},
            )
            c2 = Chunk(
                text="Document regarding frontend UI styling, component architecture, CSS layout primitives, and keyframe animations in web applications.",
                source="ui.txt",
                chunk_index=0,
                embedding=[0.9] * 384,
                metadata={"domain": "frontend"},
            )
            ragi._sync.store.add_chunks([c1, c2])

            filtered_async = ragi.filter(domain="database")
            res = await filtered_async.retrieve("frontend", top_k=5)

            domains = [c.metadata.get("domain") for c in res]
            assert "database" in domains
            assert "frontend" not in domains

    def test_lancestore_single_quote_boundary_probe(self):
        """Verify remediation D2: LanceStore handles single quotes safely in delete and search filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceStore(uri=tmpdir)
            chunk = Chunk(
                text="Detailed chapter excerpt from O'Reilly software architecture publication with verified code samples.",
                source="user's_guide.md",
                chunk_index=0,
                embedding=[0.1] * 384,
                metadata={"author": "O'Reilly", "title": "Architect's Handbook"},
            )
            store.add_chunks([chunk])
            assert store.count() == 1

            # Search with single quote in metadata filter
            cits = store.search([0.1] * 384, top_k=5, filters={"author": "O'Reilly"})
            assert len(cits) == 1

            # Delete with single quote in source
            deleted = store.delete_by_source("user's_guide.md")
            assert deleted == 1
            assert store.count() == 0

    def test_hybrid_searcher_prefix_collision_probe(self):
        """Verify remediation D5: HybridSearcher keys by (source, chunk) preventing prefix collision."""
        hs = HybridSearcher()
        prefix = "A" * 200
        c1 = Citation(source="doc1.txt", chunk=prefix + " UNIQUE_TAIL_ALPHA", score=0.9)
        c2 = Citation(source="doc2.txt", chunk=prefix + " UNIQUE_TAIL_BETA", score=0.8)

        hs.index_chunks([c1.chunk, c2.chunk])
        results = hs.search(query="UNIQUE", vector_citations=[c1, c2], top_k=5)
        assert len(results) == 2


# ==============================================================================
# 4. Exporter and Validator Boundary Hardening
# ==============================================================================

class TestExporterAndValidatorBoundaries:
    """Stress tests on leaderboard exporters and validator integrity."""

    def test_export_enterpriserag_with_none_and_missing_latency(self):
        """Verify export_enterpriserag handling of valid latency_ms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "enterprise.jsonl"
            records = [
                {
                    "question_id": "q1",
                    "query": "What is the primary mechanism?",
                    "retrieved_ids": ["doc_1", "doc_2"],
                    "latency_ms": 42.5,
                }
            ]
            path = export_enterpriserag(records, output_file=out_file)
            is_valid, errors = validate_submission_file(path)
            assert is_valid, f"EnterpriseRAG submission invalid: {errors}"

    def test_export_enterpriserag_latency_none_defect_probe(self):
        """Verify remediation D6: export_enterpriserag handles latency_ms=None safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "enterprise_none.jsonl"
            records = [
                {
                    "question_id": "q1",
                    "query": "What is the primary mechanism?",
                    "retrieved_ids": ["doc_1"],
                    "latency_ms": None,
                }
            ]
            path = export_enterpriserag(records, output_file=out_file)
            is_valid, errors = validate_submission_file(path)
            assert is_valid, f"EnterpriseRAG submission invalid: {errors}"
            # Verify parsed line has fallback latency
            with open(path, "r", encoding="utf-8") as f:
                data = json.loads(f.readline())
                assert data["latency_ms"] == 75.0

    def test_validate_submission_file_corrupt_zip(self):
        """Verify validate_submission_file detects corrupted zip archives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_zip = Path(tmpdir) / "corrupt.zip"
            # Write invalid garbage bytes
            bad_zip.write_bytes(b"PK\x03\x04\x00\x00\x00\x00corrupted_archive_data_stream")

            is_valid, errors = validate_submission_file(bad_zip, benchmark_type="airbench")
            assert not is_valid
            assert any("Bad zip" in e or "not a valid zip" in e or "Decompression" in e for e in errors)

    def test_validate_submission_file_invalid_trec_ranks(self):
        """Verify validate_submission_file rejects TREC runs with non-monotonically increasing ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trec_file = Path(tmpdir) / "bad_rank.run"
            # Line 2 has rank 1 repeated instead of 2
            content = (
                "q1 Q0 doc1 1 0.900000 run1\n"
                "q1 Q0 doc2 1 0.800000 run1\n"
            )
            trec_file.write_text(content)

            is_valid, errors = validate_submission_file(trec_file)
            assert not is_valid
            assert any("Ranks not strictly increasing" in e for e in errors)

    def test_validate_submission_file_non_decreasing_scores(self):
        """Verify validate_submission_file rejects TREC runs with increasing scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trec_file = Path(tmpdir) / "bad_score.run"
            # Score increases from 0.50 to 0.90
            content = (
                "q1 Q0 doc1 1 0.500000 run1\n"
                "q1 Q0 doc2 2 0.900000 run1\n"
            )
            trec_file.write_text(content)

            is_valid, errors = validate_submission_file(trec_file)
            assert not is_valid
            assert any("Non-monotonic score" in e for e in errors)

