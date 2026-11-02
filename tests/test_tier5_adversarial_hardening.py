"""Tier 5: White-Box Adversarial Coverage Hardening Suite for Piragi.

Thoroughly stress-tests edge cases, boundary conditions, error handling,
and untested branches across:
1. U-shaped context attention ordering & Lost-in-the-Middle mitigation
2. Inline citation attribution parsing and candidate pruning
3. Hierarchical chunking & auto-merging context deduplication
4. Hybrid search (BM25 sparse + dense vector), LRU cache eviction, and RRF fusion
5. Remote GCS URIs in LanceStore and Vertex AI text embeddings
6. Benchmark leaderboard exporters, schema validators, and preset optimizers
"""

from __future__ import annotations

import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from piragi.config import RagiConfig, RetrievalConfig
from piragi.embeddings import EmbeddingGenerator
from piragi.hybrid_search import HybridSearcher
from piragi.retrieval import parse_citation_references, prune_citations, reorder_u_shaped
from piragi.semantic_chunking import HierarchicalChunker
from piragi.stores.lance import LanceStore, get_embedding_dimension
from piragi.types import Citation, Document
from evals.leaderboard_exporters import (
    export_airbench,
    export_enterpriserag,
    export_gaia,
    optimize_for_leaderboard,
    validate_submission_file,
)


# ===========================================================================
# 1. U-SHAPED CONTEXT ATTENTION ORDERING ADVERSARIAL STRESS
# ===========================================================================

class TestUShapedAttentionAdversarial:
    """Stress-test U-shaped context attention layout for mathematical invariants."""

    def test_empty_and_single_and_pair(self):
        """0, 1, and 2 element lists must return equivalent copies without reordering."""
        assert reorder_u_shaped([]) == []

        c1 = Citation(source="doc1", chunk="chunk1", score=0.9)
        assert reorder_u_shaped([c1]) == [c1]

        c2 = Citation(source="doc2", chunk="chunk2", score=0.8)
        assert reorder_u_shaped([c1, c2]) == [c1, c2]

    @pytest.mark.parametrize("n", [3, 4, 5, 7, 10, 15, 20, 31, 64, 101])
    def test_u_shaped_trough_and_extremities_invariants(self, n: int):
        """For any n >= 3, top-1 is at index 0, top-2 is at index -1, and least relevant is in the trough."""
        citations = [
            Citation(source=f"doc_{i}", chunk=f"chunk_{i}", score=1.0 - (i / n))
            for i in range(n)
        ]
        reordered = reorder_u_shaped(citations)

        # Invariant 1: Length preserved
        assert len(reordered) == n

        # Invariant 2: Set bijection
        assert {c.source for c in reordered} == {c.source for c in citations}

        # Invariant 3: Highest relevance at position 0
        assert reordered[0].source == citations[0].source

        # Invariant 4: Second highest relevance at position -1
        assert reordered[-1].source == citations[1].source

        # Invariant 5: Lowest relevance citation (citations[n-1]) is in the interior
        lowest = citations[n - 1].source
        trough_index = (n - 1) // 2 if n % 2 == 1 else n // 2
        assert reordered[trough_index].source == lowest

    def test_identical_scores_stability(self):
        """Reordering citations with identical scores must not crash or drop items."""
        citations = [Citation(source=f"doc_{i}", chunk=f"text_{i}", score=0.75) for i in range(10)]
        reordered = reorder_u_shaped(citations)
        assert len(reordered) == 10
        assert {c.source for c in reordered} == {f"doc_{i}" for i in range(10)}

    def test_extreme_and_special_scores(self):
        """Handles negative scores, zero scores, and large lists cleanly."""
        citations = [
            Citation(source="d0", chunk="c0", score=100.0),
            Citation(source="d1", chunk="c1", score=0.0),
            Citation(source="d2", chunk="c2", score=-5.0),
            Citation(source="d3", chunk="c3", score=-10.0),
        ]
        reordered = reorder_u_shaped(citations)
        assert [c.source for c in reordered] == ["d0", "d2", "d3", "d1"]


# ===========================================================================
# 2. INLINE CITATION PARSING & ATTRIBUTION PRUNING ADVERSARIAL STRESS
# ===========================================================================

class TestCitationAttributionPruningAdversarial:
    """Adversarial testing of LLM citation reference parsing and pruning."""

    def test_parse_references_empty_or_zero_max(self):
        """Empty text or max_citations <= 0 must return empty list."""
        assert parse_citation_references("", max_citations=5) == []
        assert parse_citation_references("Text with [1]", max_citations=0) == []
        assert parse_citation_references("Text with [1]", max_citations=-5) == []

    def test_parse_bracket_and_text_styles(self):
        """Parses bracketed, textual, and range formats."""
        text = (
            "According to [1] and [Source 2], the effect is verified. "
            "Further results in [Sources 3, 4] and [Ref 5] corroborate this. "
            "Also see [6-8] and Source 9."
        )
        refs = parse_citation_references(text, max_citations=10)
        assert refs == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_parse_en_dash_em_dash_and_spaces(self):
        """Handles Unicode en-dash, em-dash, and arbitrary spacing in ranges."""
        text = "Evidence in [1 – 3] and [4—5]."
        refs = parse_citation_references(text, max_citations=10)
        assert refs == [1, 2, 3, 4, 5]

    def test_parse_conjunction_and_word_forms(self):
        """Handles 'and' conjunctions in citation brackets."""
        text = "Refer to [1 and 2] and [Sources 3 and 4]."
        refs = parse_citation_references(text, max_citations=10)
        assert refs == [1, 2, 3, 4]

    def test_parse_markdown_footnote_style(self):
        """Parses Markdown footnote references like [^1], [^2]."""
        text = "This claim is supported [^1] and extended in [^3]."
        refs = parse_citation_references(text, max_citations=5)
        assert refs == [1, 3]

    def test_out_of_bounds_and_invalid_indices_filtered(self):
        """Indices out of range [1, max_citations] must be cleanly ignored."""
        text = "Checked [0], [1], [5], and [999]."
        refs = parse_citation_references(text, max_citations=5)
        assert refs == [1, 5]

    def test_non_citation_brackets_ignored(self):
        """Code brackets, complex regexes, or acronyms inside brackets must not crash."""
        text = (
            "The complexity is [O(n^2)] and takes [50ms]. "
            "See [TODO: verify] and [NOTE]. Relevant document: [2]."
        )
        refs = parse_citation_references(text, max_citations=5)
        assert refs == [2]

    def test_large_range_span_protection(self):
        """Range spans exceeding 20 must be ignored to prevent denial-of-service."""
        text = "See massive span [1-500]."
        refs = parse_citation_references(text, max_citations=1000)
        assert refs == []

    def test_prune_citations_no_references_fallback(self):
        """When no citations are referenced in the answer text, fallback returns all citations."""
        citations = [
            Citation(source="doc1", chunk="chunk1", score=0.9),
            Citation(source="doc2", chunk="chunk2", score=0.8),
        ]
        active, unreferenced = prune_citations("Answer without any citations.", citations)
        assert len(active) == 2
        assert len(unreferenced) == 0

    def test_prune_citations_partial_references(self):
        """Only referenced citations are returned as active; unreferenced are placed in secondary list."""
        c1 = Citation(source="doc1", chunk="chunk1", score=0.9)
        c2 = Citation(source="doc2", chunk="chunk2", score=0.8)
        c3 = Citation(source="doc3", chunk="chunk3", score=0.7)

        active, unref = prune_citations("Answer based on [2] and [1].", [c1, c2, c3])
        # Preserves active citations
        assert [c.source for c in active] == ["doc1", "doc2"]
        # Inactive citation doc3 moved to unreferenced
        assert [c.source for c in unref] == ["doc3"]
        # citation_index metadata recorded
        assert active[0].metadata["citation_index"] == 1
        assert active[1].metadata["citation_index"] == 2
        assert unref[0].metadata["citation_index"] == 3


# ===========================================================================
# 3. HIERARCHICAL CHUNKING & AUTO-MERGING ADVERSARIAL STRESS
# ===========================================================================

class TestHierarchicalChunkingAndAutoMergeAdversarial:
    """Stress-test hierarchical chunking and parent-child auto-merging deduplication."""

    def test_empty_and_whitespace_document(self):
        """Chunking empty or whitespace document returns empty lists cleanly."""
        chunker = HierarchicalChunker(parent_chunk_size=1000, child_chunk_size=200)
        parents, children = chunker.chunk_document(Document(content="", source="empty.txt"))
        assert parents == []
        assert children == []

        parents, children = chunker.chunk_document(Document(content="   \n\t  ", source="space.txt"))
        assert parents == []
        assert children == []

    def test_single_character_and_short_document(self):
        """Single character or short document produces 1 parent and 1 child."""
        chunker = HierarchicalChunker(parent_chunk_size=500, child_chunk_size=100)
        parents, children = chunker.chunk_document(Document(content="A", source="short.txt"))
        assert len(parents) == 1
        assert len(children) == 1
        assert parents[0].text == "A"
        assert children[0].text == "A"
        assert children[0].metadata["parent_index"] == 0

    def test_giant_unbroken_text_forward_progress(self):
        """Continuous string without whitespace or sentence boundaries must make forward progress."""
        unbroken = "X" * 3500
        chunker = HierarchicalChunker(parent_chunk_size=1000, child_chunk_size=300, overlap=50)
        parents, children = chunker.chunk_document(Document(content=unbroken, source="unbroken.txt"))
        assert len(parents) >= 3
        assert len(children) >= 10
        # Check all child lengths are bounded
        assert all(len(c.text) <= 300 for c in children)

    def test_degenerate_overlap_equal_or_greater_than_child_size(self):
        """Overlap >= child_chunk_size must not cause an infinite loop."""
        chunker = HierarchicalChunker(parent_chunk_size=500, child_chunk_size=100, overlap=100)
        parents, children = chunker.chunk_document(Document(content="The quick brown fox jumps over the lazy dog. " * 10, source="overlap.txt"))
        assert len(parents) > 0
        assert len(children) > 0

    def test_expand_to_parent_context_merging_logic(self):
        """Verifies multi-sibling auto-merging deduplication, composite scoring, and metadata aggregation."""
        from piragi.pipelines.retrieval import RetrievalPipeline

        # Mock minimal pipeline
        pipeline = RetrievalPipeline(
            embedder=MagicMock(),
            store=MagicMock(),
            retriever=MagicMock(),
            auto_merge_parents=True,
        )

        c1 = Citation(
            source="doc1",
            chunk="Child chunk 1",
            score=0.80,
            metadata={"parent_text": "Full parent text 1", "parent_index": 0, "chunk_index": 1, "k1": "v1"},
        )
        c2 = Citation(
            source="doc1",
            chunk="Child chunk 2",
            score=0.90,
            metadata={"parent_text": "Full parent text 1", "parent_index": 0, "chunk_index": 2, "k2": "v2"},
        )
        c3 = Citation(
            source="doc2",
            chunk="Child chunk from doc 2",
            score=0.70,
            metadata={"parent_text": "Full parent text 2", "parent_index": 1, "chunk_index": 0},
        )
        non_hierarchical = Citation(
            source="doc3",
            chunk="Independent chunk",
            score=0.60,
            metadata={"custom": "plain"},
        )

        merged = pipeline._expand_to_parent_context([c1, c2, c3, non_hierarchical])

        # Sibling c1 and c2 should merge into single parent citation for doc1
        assert len(merged) == 3

        # Merged parent 1
        p1 = next(c for c in merged if c.source == "doc1")
        assert p1.chunk == "Full parent text 1"
        assert p1.score == 0.90  # Max score of children
        assert p1.metadata["merged_children_count"] == 2
        assert p1.metadata["child_indices"] == [1, 2]
        assert p1.metadata["k1"] == "v1"
        assert p1.metadata["k2"] == "v2"
        # Composite score reward: 0.90 + 0.05 * (2 - 1) = 0.95
        assert p1.metadata["composite_score"] == pytest.approx(0.95)

        # Independent citation untouched
        p_plain = next(c for c in merged if c.source == "doc3")
        assert p_plain.chunk == "Independent chunk"
        assert p_plain.score == 0.60

    def test_expand_to_parent_context_legacy_disabled_fallback(self):
        """When auto_merge_parents is False, uses legacy fallback expanding without deduplication."""
        from piragi.pipelines.retrieval import RetrievalPipeline

        pipeline = RetrievalPipeline(
            embedder=MagicMock(),
            store=MagicMock(),
            retriever=MagicMock(),
            auto_merge_parents=False,
        )

        c1 = Citation(
            source="doc1",
            chunk="Child 1",
            score=0.80,
            metadata={"parent_text": "Parent 1"},
        )
        c2 = Citation(
            source="doc1",
            chunk="Child 2",
            score=0.90,
            metadata={"parent_text": "Parent 1"},
        )
        expanded = pipeline._expand_to_parent_context([c1, c2])
        # Legacy expands both without merging
        assert len(expanded) == 2
        assert expanded[0].chunk == "Parent 1"
        assert expanded[1].chunk == "Parent 1"


# ===========================================================================
# 4. HYBRID SEARCH & BM25 CACHING ADVERSARIAL STRESS
# ===========================================================================

class TestHybridSearchAndBM25CachingAdversarial:
    """Stress-test BM25 sparse retrieval, tokenization, LRU cache eviction, and RRF fusion."""

    def test_empty_corpus_and_invalid_top_k(self):
        """search_corpus with top_k <= 0 or empty corpus handles edge cases gracefully."""
        searcher = HybridSearcher()
        vector_res = [Citation(source="doc1", chunk="Hello world text", score=0.9)]

        # top_k <= 0 returns empty list
        assert searcher.search_corpus("query", vector_res, corpus=[], top_k=0) == []
        assert searcher.search_corpus("query", vector_res, corpus=[], top_k=-1) == []

        # Empty corpus returns vector results
        assert searcher.search_corpus("query", vector_res, corpus=[], top_k=5) == vector_res

    def test_non_tokenizable_and_punctuation_corpus(self):
        """Corpus containing only punctuation or whitespace must not raise ZeroDivisionError."""
        searcher = HybridSearcher()
        vector_res = [Citation(source="doc1", chunk="sample text passage", score=0.85)]
        symbol_corpus = [
            Citation(source="sym1", chunk="!@#$%^&*()_+", score=0.0),
            Citation(source="sym2", chunk="   \t\n   ", score=0.0),
        ]

        # Must fall back cleanly to vector results without error
        res = searcher.search_corpus("query", vector_res, corpus=symbol_corpus, top_k=5)
        assert res == vector_res

    def test_bm25_lru_cache_eviction_under_pressure(self):
        """Corpus BM25 cache must evict oldest entries when exceeding max_cache_size=16."""
        searcher = HybridSearcher()
        assert len(searcher._corpus_cache) == 0

        # Generate 20 distinct corpora
        for i in range(20):
            corpus = [Citation(source=f"doc_{i}", chunk=f"unique distinctive text tokens {i} {i*10}", score=0.0)]
            searcher._get_or_create_bm25(corpus)

        # Cache size must be capped at 16
        assert len(searcher._corpus_cache) == 16

    def test_invalidate_bm25_cache_does_not_mutate_store_reference(self):
        """Invalidating BM25 cache clears searcher state without mutating the store's list (Defect E-01 check)."""
        searcher = HybridSearcher()
        store_chunk_texts = ["chunk 1 text", "chunk 2 text", "chunk 3 text"]

        searcher.index_chunks(store_chunk_texts)
        assert len(searcher._chunk_texts) == 3

        searcher.invalidate_bm25_cache()
        assert len(searcher._chunk_texts) == 0
        assert searcher._bm25 is None

        # Verify store's original list is completely intact
        assert len(store_chunk_texts) == 3
        assert store_chunk_texts == ["chunk 1 text", "chunk 2 text", "chunk 3 text"]

    def test_search_corpus_duplicate_citations_in_lists(self):
        """Duplicates within vector or keyword lists must be deduplicated via seen set."""
        searcher = HybridSearcher(use_rrf=True)
        dup_vector = [
            Citation(source="d1", chunk="identical chunk", score=0.9),
            Citation(source="d1", chunk="identical chunk", score=0.8),
        ]
        corpus = [Citation(source="d1", chunk="identical chunk", score=0.0)]

        results = searcher.search_corpus("identical", dup_vector, corpus, top_k=5)
        # Should contain only 1 deduplicated entry
        assert len(results) == 1
        assert results[0].source == "d1"

    def test_weighted_fusion_empty_and_zero_bm25_scores(self):
        """_weighted_fusion handles empty or zero BM25 scores without division by zero."""
        searcher = HybridSearcher(use_rrf=False)
        vector_res = [Citation(source="d1", chunk="text passage", score=0.9)]
        citation_map = {"text passage"[:200]: vector_res[0]}

        # Empty bm25 scores
        scores = searcher._weighted_fusion(vector_res, bm25_scores=[], citation_map=citation_map)
        assert "text passage"[:200] in scores

        # All zero bm25 scores
        scores_zero = searcher._weighted_fusion(vector_res, bm25_scores=[0.0, 0.0], citation_map=citation_map)
        assert "text passage"[:200] in scores_zero


# ===========================================================================
# 5. GCS REMOTE URIS & VERTEX AI EMBEDDINGS ADVERSARIAL STRESS
# ===========================================================================

class TestGCSAndVertexEmbeddingsAdversarial:
    """Stress-test LanceStore cloud URI handling and Vertex AI embeddings generation."""

    def test_gcs_uri_schemes_bypass_local_mkdir(self):
        """gs:// and gcs:// URI schemes must never call Path().mkdir()."""
        with patch("pathlib.Path.mkdir") as mock_mkdir, patch("lancedb.connect") as mock_connect:
            mock_connect.return_value = MagicMock(table_names=lambda: [])

            # gs:// scheme
            store_gs = LanceStore(uri="gs://my-bucket/indices/kb")
            mock_mkdir.assert_not_called()

            # gcs:// scheme
            store_gcs = LanceStore(uri="gcs://my-bucket/indices/kb")
            mock_mkdir.assert_not_called()

            # Mixed case GS://
            store_upper = LanceStore(uri="GS://MY-BUCKET/PATH")
            mock_mkdir.assert_not_called()

    def test_vertex_ai_dimension_registry_lookups(self):
        """Embedding dimension registry correctly resolves Vertex AI and standard models."""
        # Vertex AI models
        assert get_embedding_dimension("text-embedding-004") == 768
        assert get_embedding_dimension("google/text-embedding-004") == 768
        assert get_embedding_dimension("projects/p/locations/l/publishers/google/models/text-embedding-004") == 768
        assert get_embedding_dimension("text-multilingual-embedding-002") == 768
        assert get_embedding_dimension("google/text-multilingual-embedding-002") == 768

        # Standard models
        assert get_embedding_dimension("all-mpnet-base-v2") == 768
        assert get_embedding_dimension("all-MiniLM-L6-v2") == 384
        assert get_embedding_dimension("nvidia/llama-embed-nemotron-8b") == 4096

        # Empty model name fallback
        assert get_embedding_dimension("") == 384

    def test_uninitialized_lance_store_table_safety(self):
        """Operations on a LanceStore before adding any chunks must return safe empty defaults."""
        with patch("pathlib.Path.mkdir"), patch("lancedb.connect") as mock_connect:
            mock_connect.return_value = MagicMock(table_names=lambda: [])
            store = LanceStore(uri="gs://empty-test-store/idx")
            assert store.table is None

            # Operations on None table
            assert store.search(query_embedding=[0.1] * 768, top_k=5) == []
            assert store.count() == 0
            assert store.delete_by_source("nonexistent.txt") == 0
            assert store.get_all_citations() == []

    def test_vertex_embeddings_offline_fallback_determinism_and_norm(self):
        """Offline fallback produces deterministic, unit-normalized 768-dim embeddings."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        assert gen.is_vertex is True

        # Generate embeddings for sample texts
        texts = [
            "What is Reciprocal Rank Fusion?",
            "How does hierarchical auto-merging work?",
            "",  # empty string
            "🚀🔥 Specialized text with emojis and symbols! ✨💡",
            "Very long text " * 1000,  # > 8000 chars
        ]
        embeddings = gen._generate_vertexai_embeddings(texts)

        assert len(embeddings) == len(texts)

        for emb in embeddings:
            assert len(emb) == 768
            # Unit norm check (float32 precision)
            norm = np.linalg.norm(emb)
            assert norm == pytest.approx(1.0, abs=1e-3)

        # Determinism check: identical text gives identical embedding
        emb1 = gen._generate_vertexai_embeddings(["Deterministic check text"])
        emb2 = gen._generate_vertexai_embeddings(["Deterministic check text"])
        assert emb1[0] == emb2[0]

        # Discriminative check: different text gives different embedding
        emb3 = gen._generate_vertexai_embeddings(["Completely different semantic passage"])
        assert emb1[0] != emb3[0]

    def test_vertex_embeddings_empty_input(self):
        """Empty texts list returns empty embeddings list."""
        gen = EmbeddingGenerator(backend="vertexai", model="text-embedding-004")
        assert gen._generate_vertexai_embeddings([]) == []


# ===========================================================================
# 6. LEADERBOARD EXPORTERS & VALIDATORS ADVERSARIAL STRESS
# ===========================================================================

class TestLeaderboardExportersAdversarial:
    """Stress-test leaderboard hyperparameter presets, format converters, and schema validators."""

    @pytest.mark.parametrize("benchmark", [
        "enterpriserag", "EnterpriseRAG", "enterprise-rag", "enterprise_rag",
        "airbench", "AIR-Bench", "air_bench",
        "gaia", "GAIA", "gaia-benchmark",
    ])
    def test_optimize_for_leaderboard_aliases(self, benchmark: str):
        """All valid aliases and case variations resolve to correct benchmark presets."""
        preset = optimize_for_leaderboard(benchmark)
        assert "benchmark" in preset
        assert "top_k" in preset
        assert "retrieval" in preset
        assert "chunk" in preset
        assert "rationale" in preset

    def test_optimize_for_leaderboard_unknown_raises(self):
        """Unknown benchmark names raise ValueError with supported choices listed."""
        with pytest.raises(ValueError, match="Unknown benchmark: 'nonexistent'"):
            optimize_for_leaderboard("nonexistent")

    def test_validate_submission_nonexistent_and_empty_file(self):
        """Missing files and 0-byte files fail validation cleanly."""
        is_valid, errs = validate_submission_file("does_not_exist_file.jsonl")
        assert is_valid is False
        assert any("File not found" in e for e in errs)

        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            is_valid, errs = validate_submission_file(f.name)
            assert is_valid is False
            assert any("0 bytes" in e for e in errs)

    def test_validate_enterpriserag_schema_violations(self):
        """EnterpriseRAG validator detects missing keys, non-finite latency, and malformed citations."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            # Bad record: missing 'answer' and non-finite latency
            bad_rec = {
                "question_id": "q1",
                "query": "query",
                "citations": ["not-a-dict"],
                "doc_ids": [],
                "latency_ms": -1.0,
                "metadata": {},
            }
            f.write(json.dumps(bad_rec) + "\n")
            f.flush()

            is_valid, errs = validate_submission_file(f.name, benchmark_type="enterpriserag")
            assert is_valid is False
            assert any("Missing required key 'answer'" in e for e in errs)
            assert any("citation #0 must be a dict" in e for e in errs)
            assert any("doc_ids' must be a non-empty list" in e for e in errs)
            assert any("latency_ms' must be a non-negative number" in e for e in errs)

    def test_validate_gaia_schema_violations(self):
        """GAIA validator detects missing task_id, model_answer, and empty reasoning traces."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            bad_rec = {
                "task_id": "",
                "model_answer": "answer",
                "reasoning_trace": [],
                "sources_used": [],
            }
            f.write(json.dumps(bad_rec) + "\n")
            f.flush()

            is_valid, errs = validate_submission_file(f.name, benchmark_type="gaia")
            assert is_valid is False
            assert any("task_id' must be a non-empty string" in e for e in errs)
            assert any("reasoning_trace' must be a non-empty list" in e for e in errs)
            assert any("sources_used' must be a non-empty list" in e for e in errs)

    def test_validate_airbench_zip_structural_and_trec_violations(self):
        """AIR-Bench validator detects missing run files, rank gaps, and non-monotonic scores."""
        # 1. Non-zip file with .zip extension
        with tempfile.NamedTemporaryFile(suffix=".zip", mode="wb", delete=False) as f:
            f.write(b"NOT A REAL ZIP FILE")
            f.flush()
            is_valid, errs = validate_submission_file(f.name)
            assert is_valid is False
            assert any("not a valid zip archive" in e for e in errs)

        # 2. ZIP with non-monotonic TREC scores
        with tempfile.NamedTemporaryFile(suffix=".zip", mode="wb", delete=False) as f:
            with zipfile.ZipFile(f, "w") as zf:
                zf.writestr("manifest.json", json.dumps({"benchmark": "airbench"}))
                zf.writestr("metadata.json", json.dumps({"framework": "piragi"}))
                # Non-monotonic scores: rank 2 has higher score than rank 1
                bad_trec = (
                    "q1 Q0 doc1 1 0.500000 run\n"
                    "q1 Q0 doc2 2 0.800000 run\n"
                )
                zf.writestr("default.run", bad_trec)

            is_valid, errs = validate_submission_file(f.name)
            assert is_valid is False
            assert any("Non-monotonic score" in e for e in errs)

    def test_exporters_resilience_with_empty_or_minimal_data(self):
        """Exporters handle minimal input dictionaries and auto-synthesize missing fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            minimal_data = {
                "per_query_results": [
                    {
                        "query_id": "q101",
                        "query": "What is U-shaped context attention?",
                        "retrieved_ids": ["doc_1", "doc_2"],
                    }
                ]
            }

            f_enterprise = out_dir / "enterpriserag.jsonl"
            export_enterpriserag(minimal_data, output_file=f_enterprise)
            is_valid, errs = validate_submission_file(f_enterprise, benchmark_type="enterpriserag")
            assert is_valid is True, f"EnterpriseRAG export failed validation: {errs}"

            f_gaia = out_dir / "gaia.jsonl"
            export_gaia(minimal_data, output_file=f_gaia)
            is_valid, errs = validate_submission_file(f_gaia, benchmark_type="gaia")
            assert is_valid is True, f"GAIA export failed validation: {errs}"

            f_air = out_dir / "airbench.zip"
            export_airbench(minimal_data, output_file=f_air)
            is_valid, errs = validate_submission_file(f_air, benchmark_type="airbench")
            assert is_valid is True, f"AIR-Bench export failed validation: {errs}"
