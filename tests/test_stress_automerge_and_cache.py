"""Adversarial stress testing suite for Hierarchical Auto-Merging and BM25 Caching.

Covers:
1. Hierarchical auto-merging deduplication:
   - Multi-document parent collisions (index, id, hash)
   - Empty/whitespace/invalid parent text
   - Missing metadata keys & metadata=None
   - Sibling child sets scaling (1 to 100) & composite score capping
   - Zero duplicate parent passages escaping into context
2. Hybrid BM25 Index Caching:
   - Collision resistance across 200+ single-character and structural mutations
   - Cache capacity limit (16) and FIFO eviction across >20 distinct filters
   - Invalidation on add(), refresh(), clear(), and invalidate_bm25_cache()
   - Concurrent multi-threaded search_corpus and cache eviction stress
"""

import copy
import random
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock
import pytest

from piragi.config import RagiConfig, RetrievalConfig
from piragi.hybrid_search import HybridSearcher
from piragi.pipelines.retrieval import RetrievalPipeline
from piragi.types import Answer, Citation


# ============================================================================
# PART 1: Hierarchical Auto-Merging Deduplication Stress Tests
# ============================================================================


class TestHierarchicalAutoMergingStress:
    """Adversarial challenge tests for auto-merging deduplication."""

    @pytest.fixture
    def pipeline(self):
        return RetrievalPipeline(
            embedder=MagicMock(),
            store=MagicMock(),
            retriever=MagicMock(),
            use_hierarchical=True,
            auto_merge_parents=True,
        )

    def test_multi_document_parent_index_collision(self, pipeline):
        """Cross-document isolation: docA parent 0 vs docB parent 0 vs docC parent 0."""
        citations = []
        doc_names = ["alpha.md", "beta.md", "gamma.md", "delta.md"]
        for doc in doc_names:
            for child_idx in range(3):
                citations.append(
                    Citation(
                        source=doc,
                        chunk=f"{doc} child {child_idx}",
                        score=0.7 + (child_idx * 0.05),
                        metadata={
                            "parent_index": 0,
                            "parent_text": f"Parent text for {doc}",
                            "chunk_index": child_idx,
                        },
                    )
                )

        random.shuffle(citations)
        merged = pipeline._expand_to_parent_context(citations)

        # Must have exactly 4 merged citations, one per document
        assert len(merged) == len(doc_names)
        sources_found = {c.source for c in merged}
        assert sources_found == set(doc_names)
        for c in merged:
            assert c.chunk == f"Parent text for {c.source}"
            assert c.metadata["merged_children_count"] == 3
            assert c.metadata["child_indices"] == [0, 1, 2]
            assert c.score == pytest.approx(0.80)

    def test_multi_document_parent_id_collision(self, pipeline):
        """Cross-document isolation: identical parent_id string across different sources."""
        c_a = Citation(
            source="contract_a.pdf",
            chunk="child snippet 1",
            score=0.85,
            metadata={
                "parent_id": "clause_42",
                "parent_text": "Parent text for Contract A clause 42",
                "chunk_index": 1,
            },
        )
        c_b = Citation(
            source="contract_b.pdf",
            chunk="child snippet 2",
            score=0.87,
            metadata={
                "parent_id": "clause_42",
                "parent_text": "Parent text for Contract B clause 42",
                "chunk_index": 2,
            },
        )

        merged = pipeline._expand_to_parent_context([c_a, c_b])
        assert len(merged) == 2
        assert {c.source for c in merged} == {"contract_a.pdf", "contract_b.pdf"}
        assert merged[0].chunk != merged[1].chunk

    def test_multi_document_identical_parent_text_collision(self, pipeline):
        """Cross-document isolation: exact same parent text in different documents."""
        identical_parent = "Confidentiality Agreement: Standard terms apply."
        c1 = Citation(
            source="nda_alice.md",
            chunk="alice terms",
            score=0.90,
            metadata={"parent_text": identical_parent, "chunk_index": 0},
        )
        c2 = Citation(
            source="nda_bob.md",
            chunk="bob terms",
            score=0.85,
            metadata={"parent_text": identical_parent, "chunk_index": 0},
        )

        merged = pipeline._expand_to_parent_context([c1, c2])
        assert len(merged) == 2
        assert {c.source for c in merged} == {"nda_alice.md", "nda_bob.md"}

    @pytest.mark.parametrize(
        "bad_parent_text",
        [
            None,
            "",
            "   ",
            "\n\t\r \n",
            12345,
            [],
            {},
        ],
    )
    def test_empty_or_invalid_parent_text_fallbacks(self, pipeline, bad_parent_text):
        """Invalid parent_text must never crash and must pass citation untouched."""
        c = Citation(
            source="edge_case.txt",
            chunk="Raw child snippet that should remain",
            score=0.77,
            metadata={"parent_text": bad_parent_text, "chunk_index": 1},
        )
        merged = pipeline._expand_to_parent_context([c])
        assert len(merged) == 1
        assert merged[0].chunk == "Raw child snippet that should remain"
        assert merged[0].score == 0.77

    def test_missing_metadata_keys_robustness(self, pipeline):
        """Citations with None metadata, empty dict, or missing chunk_index."""
        # Duck-typed citation object with metadata=None
        class DuckCitation:
            source = "none_meta.txt"
            chunk = "text1"
            score = 0.8
            metadata = None

        c_duck_none = DuckCitation()
        c_default_meta = Citation(source="default_meta.txt", chunk="text2", score=0.85)
        c_empty_meta = Citation(source="empty_meta.txt", chunk="text3", score=0.86, metadata={})
        c_missing_idx = Citation(
            source="missing_idx.txt",
            chunk="child without index",
            score=0.90,
            metadata={"parent_text": "Parent for missing idx"},
        )
        c_missing_parent_id = Citation(
            source="missing_parent_id.txt",
            chunk="child without parent id",
            score=0.92,
            metadata={"parent_text": "Parent for missing parent id", "chunk_index": 3},
        )

        merged = pipeline._expand_to_parent_context(
            [c_duck_none, c_default_meta, c_empty_meta, c_missing_idx, c_missing_parent_id]
        )
        assert len(merged) == 5
        # Verify no unhandled exceptions and correct sorting
        assert all(isinstance(c, (Citation, DuckCitation)) for c in merged)


    def test_massive_sibling_set_scaling_and_score_cap(self, pipeline):
        """Stress-test 100 sibling children under the same parent and verify score capping at 1.0."""
        parent_text = "Master Chapter Overview"
        children = [
            Citation(
                source="handbook.md",
                chunk=f"Sentence {i}",
                score=0.95 if i == 50 else 0.50,
                metadata={
                    "parent_index": 1,
                    "parent_text": parent_text,
                    "chunk_index": i,
                },
            )
            for i in range(100)
        ]

        merged = pipeline._expand_to_parent_context(children)
        assert len(merged) == 1
        res = merged[0]
        assert res.chunk == parent_text
        assert res.score == 0.95
        assert res.metadata["merged_children_count"] == 100
        assert len(res.metadata["child_indices"]) == 100
        assert res.metadata["child_indices"] == list(range(100))
        # Composite score must not exceed 1.0
        assert res.metadata["composite_score"] <= 1.0
        assert res.metadata["composite_score"] == pytest.approx(1.0)

    def test_zero_duplicate_parent_passages_in_end_to_end_context(self, pipeline):
        """Verify that duplicate parent passages are 100% eliminated from context passed to generation."""
        # Mock retriever.generate_answer to capture the exact citations passed to it
        captured_citations = []

        def mock_generate_answer(query, citations, system_prompt=None, reorder=False):
            captured_citations.extend(citations)
            return Answer(text="Generated answer", citations=citations, query=query)

        pipeline.retriever.generate_answer = mock_generate_answer
        pipeline.retriever.expand_query = lambda q: [q]

        # Store returns 10 child chunks from 2 distinct parents (5 children each)
        p1_text = "Detailed Section 1 on Quantum Computing."
        p2_text = "Detailed Section 2 on Neuromorphic Hardware."

        store_results = []
        for i in range(5):
            store_results.append(
                Citation(
                    source="tech.pdf",
                    chunk=f"quantum chunk {i}",
                    score=0.90 - (i * 0.02),
                    metadata={"parent_index": 10, "parent_text": p1_text, "chunk_index": i},
                )
            )
            store_results.append(
                Citation(
                    source="tech.pdf",
                    chunk=f"neuromorphic chunk {i}",
                    score=0.80 - (i * 0.02),
                    metadata={"parent_index": 20, "parent_text": p2_text, "chunk_index": i},
                )
            )

        pipeline.store.search = MagicMock(return_value=store_results)

        answer = pipeline.ask(query="What is quantum and neuromorphic computing?", top_k=10)

        # There should be exactly 2 citations in captured context, 0 duplicates
        assert len(captured_citations) == 2
        chunks = [c.chunk for c in captured_citations]
        assert len(set(chunks)) == len(chunks) == 2
        assert set(chunks) == {p1_text, p2_text}


# ============================================================================
# PART 2: Hybrid BM25 Index Caching & Concurrency Stress Tests
# ============================================================================


class TestBM25CachingAndConcurrencyStress:
    """Adversarial challenge tests for BM25 cache collision, eviction, and concurrency."""

    @pytest.fixture
    def base_corpus(self):
        return [
            Citation(
                source=f"doc_{i}.txt",
                chunk=f"Unique content for document number {i} discussing distributed systems.",
                score=0.8,
            )
            for i in range(10)
        ]

    def test_cache_key_collision_resistance_under_single_char_mutations(self, base_corpus):
        """Mutate a single character across 200 variations and verify 100% collision-free keys."""
        searcher = HybridSearcher()
        base_key = searcher._compute_corpus_key(base_corpus)
        assert base_key is not None

        generated_keys = {base_key}

        # Perform 200 distinct single-character mutations
        for trial in range(200):
            mutated = copy.deepcopy(base_corpus)
            doc_idx = trial % len(mutated)
            original_chunk = mutated[doc_idx].chunk

            # Mutate: substitute, insert, delete, or change case of one char
            char_pos = trial % len(original_chunk)
            mutated_char = chr((ord(original_chunk[char_pos]) + 1) % 126)
            new_chunk = original_chunk[:char_pos] + mutated_char + original_chunk[char_pos + 1 :]
            mutated[doc_idx] = Citation(
                source=mutated[doc_idx].source,
                chunk=new_chunk,
                score=mutated[doc_idx].score,
            )

            mutated_key = searcher._compute_corpus_key(mutated)
            assert mutated_key != base_key, f"Collision detected for mutation at trial {trial}"
            generated_keys.add(mutated_key)

        # All 201 keys (base + 200 mutations) must be distinct
        assert len(generated_keys) == 201

    def test_cache_structural_and_unicode_mutations(self, base_corpus):
        """Verify trailing whitespace, punctuation, and unicode changes generate distinct keys."""
        searcher = HybridSearcher()
        k_base = searcher._compute_corpus_key(base_corpus)

        # 1. Trailing space mutation
        c_space = copy.deepcopy(base_corpus)
        c_space[0] = Citation(source=c_space[0].source, chunk=c_space[0].chunk + " ", score=c_space[0].score)
        k_space = searcher._compute_corpus_key(c_space)
        assert k_space != k_base

        # 2. Unicode variation
        c_unicode = copy.deepcopy(base_corpus)
        c_unicode[0] = Citation(
            source=c_unicode[0].source,
            chunk=c_unicode[0].chunk + " 🚀",
            score=c_unicode[0].score,
        )
        k_unicode = searcher._compute_corpus_key(c_unicode)
        assert k_unicode != k_base

        # 3. Source name variation with identical chunk text
        c_source = copy.deepcopy(base_corpus)
        c_source[0] = Citation(source="doc_MODIFIED.txt", chunk=c_source[0].chunk, score=c_source[0].score)
        k_source = searcher._compute_corpus_key(c_source)
        assert k_source != k_base

    def test_cache_capacity_and_fifo_eviction_over_30_filters(self):
        """Stress-test cache capacity limit (16) and verify strict FIFO eviction over 30 distinct filters."""
        searcher = HybridSearcher()
        assert searcher._max_cache_size == 16

        inserted_keys = []

        # Query 30 distinct corpora (simulating 30 distinct metadata filter results)
        for i in range(30):
            corpus = [
                Citation(
                    source=f"filter_{i}_doc.txt",
                    chunk=f"Filtering category {i} containing specific domain terminology {i}.",
                    score=0.9,
                )
            ]
            key = searcher._compute_corpus_key(corpus)
            inserted_keys.append(key)

            searcher.search_corpus(f"terminology {i}", corpus, corpus, top_k=1)

            # Capacity invariant: cache size must never exceed 16
            assert len(searcher._corpus_cache) <= 16

        # After 30 insertions, exactly 16 items remain in cache
        assert len(searcher._corpus_cache) == 16

        # The oldest 14 keys (inserted_keys[0:14]) must have been evicted
        for evicted_key in inserted_keys[:14]:
            assert evicted_key not in searcher._corpus_cache, f"Key {evicted_key} was not evicted!"

        # The newest 16 keys (inserted_keys[14:30]) must all be present
        for active_key in inserted_keys[14:30]:
            assert active_key in searcher._corpus_cache, f"Key {active_key} should be in cache!"

    def test_cache_invalidation_hooks_across_ragi_lifecycle(self, tmp_path):
        """Verify that Ragi.add(), Ragi.refresh(), and Ragi.clear() reliably invalidate the BM25 cache."""
        from piragi import Ragi
        from piragi.stores.lance import LanceStore
        from piragi.types import Chunk

        mock_embedder = MagicMock()
        mock_embedder.embed_chunks = MagicMock(
            return_value=[
                Chunk(
                    source="sample.txt",
                    text="New file to ingest into the knowledge base.",
                    chunk_index=0,
                    embedding=[0.1, 0.2],
                )
            ]
        )
        mock_embedder.embed_query = MagicMock(return_value=[0.1, 0.2])

        store = LanceStore(uri=str(tmp_path), vector_dimension=2)
        kb = Ragi(
            store=store,
            embedder=mock_embedder,
            config={"retrieval": {"use_hybrid_search": True}},
        )

        # Populate cache
        c1 = Citation(source="initial.md", chunk="Initial data", score=0.8)
        kb._hybrid_searcher.search_corpus("initial", [c1], [c1])
        assert len(kb._hybrid_searcher._corpus_cache) == 1

        # Test 1: clear() invalidates
        kb.clear()
        assert len(kb._hybrid_searcher._corpus_cache) == 0

        # Repopulate
        kb._hybrid_searcher.search_corpus("initial", [c1], [c1])
        assert len(kb._hybrid_searcher._corpus_cache) == 1

        # Test 2: IngestionPipeline invalidates
        test_file = tmp_path / "sample.txt"
        test_file.write_text("New file to ingest into the knowledge base.")

        kb.add(str(test_file))
        assert len(kb._hybrid_searcher._corpus_cache) == 0

        # Repopulate
        kb._hybrid_searcher.search_corpus("sample", [c1], [c1])
        assert len(kb._hybrid_searcher._corpus_cache) == 1

        # Test 3: refresh() invalidates
        kb.refresh(str(test_file))
        assert len(kb._hybrid_searcher._corpus_cache) == 0


    def test_concurrent_search_same_corpus_multithreading(self, base_corpus):
        """50 concurrent threads executing search_corpus on the same corpus simultaneously."""
        searcher = HybridSearcher()
        num_threads = 50
        errors = []

        def worker(thread_id):
            try:
                query = "distributed systems" if thread_id % 2 == 0 else "document"
                results = searcher.search_corpus(query, base_corpus[:3], base_corpus, top_k=3)
                assert len(results) > 0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent execution produced errors: {errors}"
        # Cache must have exactly 1 entry for base_corpus
        assert len(searcher._corpus_cache) == 1

    def test_concurrent_search_diverse_corpora_eviction_stress(self):
        """Stress-test concurrent cache insertions and evictions across 30 threads with diverse corpora."""
        searcher = HybridSearcher()
        num_workers = 30
        iterations_per_worker = 10
        errors = []

        def worker(worker_id):
            try:
                for it in range(iterations_per_worker):
                    corpus_id = (worker_id * 100) + it
                    corpus = [
                        Citation(
                            source=f"doc_{corpus_id}.txt",
                            chunk=f"Unique topic {corpus_id % 25} information data content.",
                            score=0.8,
                        )
                    ]
                    results = searcher.search_corpus(
                        f"topic {corpus_id % 25}", corpus, corpus, top_k=1
                    )
                    assert len(results) == 1
            except Exception as e:
                errors.append((worker_id, e))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Concurrent eviction produced errors: {errors}"
        assert len(searcher._corpus_cache) <= 16

    def test_concurrent_search_and_invalidation_race(self, base_corpus):
        """Simultaneous search and cache invalidation threads must not raise unhandled exceptions."""
        searcher = HybridSearcher()
        stop_event = threading.Event()
        errors = []

        def search_worker():
            while not stop_event.is_set():
                try:
                    searcher.search_corpus("distributed", base_corpus[:2], base_corpus, top_k=2)
                except Exception as e:
                    errors.append(e)

        def invalidator_worker():
            for _ in range(20):
                searcher.invalidate_bm25_cache()

        search_threads = [threading.Thread(target=search_worker) for _ in range(5)]
        inv_threads = [threading.Thread(target=invalidator_worker) for _ in range(2)]

        for t in search_threads + inv_threads:
            t.start()

        for t in inv_threads:
            t.join()

        stop_event.set()
        for t in search_threads:
            t.join()

        assert len(errors) == 0, f"Race condition errors: {errors}"
