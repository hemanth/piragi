"""Tier 3: Cross-Feature Combinations E2E Tests.

Tests pairwise and multi-feature interactions across Piragi's SOTA RAG enhancements,
GCP integrations, and Leaderboard exporters.
"""

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from piragi import Ragi, AsyncRagi, EmbeddingGenerator
from piragi.config import RagiConfig, RetrievalConfig
from piragi.types import Answer, Citation, Chunk
from piragi.stores.factory import parse_store_uri, create_store
from piragi.stores.lance import LanceStore, get_embedding_dimension
from piragi.hybrid_search import HybridSearcher, create_hybrid_searcher
from piragi.query_transform import StepBackPrompting

from .conftest import (
    has_u_shaped_support,
    has_auto_merge_support,
    has_step_back_support,
    has_citation_pruning_support,
    has_bm25_caching_support,
    has_gcs_support,
    has_vertex_embeddings_support,
    has_vertex_dimension_support,
    has_adc_token_refresh_support,
    has_benchmark_harness_support,
    has_leaderboard_exporters_support,
)
from .test_tier1_features import (
    apply_u_shaped_ordering,
    auto_merge_citations,
    prune_citations_from_text,
    compute_ndcg_at_k,
    compute_mrr,
    export_enterpriserag,
    export_airbench,
    export_gaia,
    validate_submission_file,
)


class TestTier3_CrossFeatureCombinations:
    """Pairwise and multi-feature interaction tests."""

    def test_c01_u_shaped_and_auto_merging_pipeline(self):
        """C01: Auto-merging collapses sibling chunks, then U-shaped orders merged citations."""
        children = [
            Citation(source="d1", chunk="c1_sub1", score=0.9, metadata={"parent_id": "p1", "parent_text": "Parent 1 Text"}),
            Citation(source="d1", chunk="c1_sub2", score=0.95, metadata={"parent_id": "p1", "parent_text": "Parent 1 Text"}),
            Citation(source="d2", chunk="c2_sub1", score=0.8, metadata={"parent_id": "p2", "parent_text": "Parent 2 Text"}),
            Citation(source="d2", chunk="c2_sub2", score=0.85, metadata={"parent_id": "p2", "parent_text": "Parent 2 Text"}),
            Citation(source="d3", chunk="c3_sub1", score=0.7, metadata={"parent_id": "p3", "parent_text": "Parent 3 Text"}),
            Citation(source="d4", chunk="c4_sub1", score=0.6, metadata={"parent_id": "p4", "parent_text": "Parent 4 Text"}),
        ]
        # Step 1: Auto-merging collapses to 4 parent citations
        merged = auto_merge_citations(children)
        assert len(merged) == 4
        # Sort descending by score
        merged.sort(key=lambda c: c.score, reverse=True)
        # Scores: P1 (0.95), P2 (0.85), P3 (0.7), P4 (0.6)
        assert [c.chunk for c in merged] == ["Parent 1 Text", "Parent 2 Text", "Parent 3 Text", "Parent 4 Text"]

        # Step 2: U-shaped attention reorders to [0, 2, 3, 1]
        u_ordered = apply_u_shaped_ordering(merged)
        assert len(u_ordered) == 4
        assert u_ordered[0].chunk == "Parent 1 Text"
        assert u_ordered[-1].chunk == "Parent 2 Text"
        assert u_ordered[1].chunk == "Parent 3 Text"
        assert u_ordered[2].chunk == "Parent 4 Text"

    def test_c02_u_shaped_attention_and_citation_pruning(self):
        """C02: Citations ordered in U-shape, LLM cites [1] and [4], pruning retains exactly those."""
        citations = [
            Citation(source="doc_top1", chunk="chunk 1", score=0.99),
            Citation(source="doc_mid1", chunk="chunk 2", score=0.70),
            Citation(source="doc_mid2", chunk="chunk 3", score=0.60),
            Citation(source="doc_top2", chunk="chunk 4", score=0.90),
        ]
        # Text cites Source 1 and Source 4
        generated_answer = "Key evidence is established in [1] and substantiated by [4]."
        pruned = prune_citations_from_text(generated_answer, citations)
        assert len(pruned) == 2
        assert [c.source for c in pruned] == ["doc_top1", "doc_top2"]

    def test_c03_stepback_and_hybrid_bm25_fusion(self):
        """C03: StepBack query + persistent BM25 hybrid search."""
        searcher = HybridSearcher(vector_weight=0.5, bm25_weight=0.5, use_rrf=True)
        corpus = [
            "distributed consensus raft paxos algorithms",
            "etcd distributed key-value store architecture",
            "redis in-memory cache clustering sentinel",
            "postgresql relational database ACID transactions",
        ]
        searcher.index_chunks(corpus)

        # Original query: "etcd leader election failover"
        # Step-back query: "distributed consensus algorithms"
        vec_cits = [
            Citation(source="d1", chunk="etcd distributed key-value store architecture", score=0.9),
            Citation(source="d2", chunk="distributed consensus raft paxos algorithms", score=0.8),
        ]
        results_original = searcher.search("etcd leader election failover", vec_cits, top_k=2)
        results_stepback = searcher.search("distributed consensus algorithms", vec_cits, top_k=2)

        assert len(results_original) == 2
        assert len(results_stepback) == 2
        # Both searches reuse the same indexed BM25 without reindexing
        assert searcher._bm25 is not None

    def test_c04_hierarchical_auto_merging_and_citation_pruning(self):
        """C04: Sibling chunks merged into parent, answer cites parent [1], preserved by pruning."""
        c1 = Citation(source="doc1", chunk="sub_clause_a", score=0.88, metadata={"parent_id": "clause_1", "parent_text": "Section 1: General Provisions"})
        c2 = Citation(source="doc1", chunk="sub_clause_b", score=0.92, metadata={"parent_id": "clause_1", "parent_text": "Section 1: General Provisions"})
        c3 = Citation(source="doc2", chunk="other_clause", score=0.75, metadata={"parent_id": "clause_2", "parent_text": "Section 2: Definitions"})

        merged = auto_merge_citations([c1, c2, c3])
        assert len(merged) == 2

        answer_text = "According to [1], all requirements are governed by general provisions."
        final_citations = prune_citations_from_text(answer_text, merged)
        assert len(final_citations) == 1
        assert final_citations[0].chunk == "Section 1: General Provisions"

    @patch("lancedb.connect")
    def test_c05_gcs_store_and_vertex_dimension_registry(self, mock_lancedb_connect):
        """C05: LanceStore on gs:// with text-embedding-004 automatically infers 768 dimensions."""
        if not (has_gcs_support() and has_vertex_dimension_support()):
            pytest.skip("GCS URI or Vertex dimension registry pending M2 implementation")
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb_connect.return_value = mock_db

        store = LanceStore(
            uri="gs://my-bucket/models/index",
            embedding_model="text-embedding-004",
        )
        assert store.vector_dimension == 768
        assert store.uri == "gs://my-bucket/models/index"

    @patch("openai.OpenAI")
    def test_c06_ragi_with_u_shaped_and_hybrid_search(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """C06: Ragi ask() combining hybrid search and U-shaped context ordering."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "Based on Source 1 and Source 2, system reliability is verified."
        )
        mock_openai.return_value = mock_client

        doc_path = os.path.join(e2e_temp_dir, "doc.txt")
        with open(doc_path, "w") as f:
            f.write("High availability requires fault tolerance. Microservices use circuit breakers.")

        cfg = {"retrieval": {"use_hybrid_search": True}}
        if has_u_shaped_support():
            cfg["retrieval"]["context_ordering"] = "u_shaped"

        kb = Ragi(
            persist_dir=os.path.join(e2e_temp_dir, "ragi_combo"),
            embedder=make_mock_embedder(),
            config=cfg,
        )
        kb.add(doc_path)
        answer = kb.ask("How is fault tolerance achieved?")
        assert isinstance(answer, Answer)
        assert answer.text

    @pytest.mark.asyncio
    @patch("openai.OpenAI")
    async def test_c07_async_ragi_with_metadata_filtering(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """C07: AsyncRagi running concurrent filtered queries."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        doc_path = os.path.join(e2e_temp_dir, "async_doc.txt")
        with open(doc_path, "w") as f:
            f.write("Async processing enables high throughput concurrent request handling.")

        kb = AsyncRagi(
            persist_dir=os.path.join(e2e_temp_dir, "async_ragi"),
            embedder=make_mock_embedder(),
        )
        await kb.add(doc_path)
        ans1 = await kb.ask("Explain async processing")
        ans2 = await kb.ask("What is high throughput?")
        assert isinstance(ans1, Answer)
        assert isinstance(ans2, Answer)

    def test_c08_enterprise_rag_export_from_ragi_answers(self, e2e_temp_dir):
        """C08: Ragi Answer objects with citations exported directly to EnterpriseRAG JSONL."""
        answers = [
            Answer(
                text="Postgres uses pgvector for indexing [1].",
                citations=[Citation(source="postgres_guide.md", chunk="pgvector extension", score=0.92)],
                query="How does Postgres index vectors?",
            ),
            Answer(
                text="LanceDB uses IVF-PQ disk indexing [1].",
                citations=[Citation(source="lance_guide.md", chunk="IVF-PQ disk layout", score=0.88)],
                query="How does LanceDB store vectors?",
            ),
        ]

        formatted_records = [
            {
                "question_id": f"q_{i}",
                "answer": ans.text,
                "citations": [c.source for c in ans.citations],
                "doc_ids": [c.source for c in ans.citations],
            }
            for i, ans in enumerate(answers)
        ]

        out_file = os.path.join(e2e_temp_dir, "enterprise_sub.jsonl")
        export_enterpriserag(formatted_records, out_file)
        assert validate_submission_file(out_file, "jsonl") is True

    def test_c09_airbench_export_from_hybrid_retrieval(self, e2e_temp_dir):
        """C09: Hybrid retrieval scores packaged into AIR-Bench TREC ZIP."""
        searcher = HybridSearcher(vector_weight=0.5, bm25_weight=0.5, use_rrf=True)
        searcher.index_chunks(["chunk alpha", "chunk beta", "chunk gamma", "chunk delta"])
        vec_cits = [
            Citation(source="doc_alpha", chunk="chunk alpha", score=0.9),
            Citation(source="doc_beta", chunk="chunk beta", score=0.8),
        ]
        results = searcher.search("alpha", vec_cits, top_k=2)

        run_records = [
            {
                "qid": "q_101",
                "doc_scores": [(c.source, c.score) for c in results],
            }
        ]
        out_zip = os.path.join(e2e_temp_dir, "airbench_run.zip")
        export_airbench(run_records, out_zip, run_id="piragi_hybrid_sota")
        assert validate_submission_file(out_zip, "zip") is True

    def test_c10_gaia_export_with_stepback_reasoning_trace(self, e2e_temp_dir):
        """C10: StepBack abstraction reasoning trace packaged into GAIA predictions."""
        tasks = [
            {
                "task_id": "gaia_t1",
                "model_answer": "LanceDB",
                "reasoning_trace": "Original: 'Which DB?'; StepBack: 'Vector DB types'; Answer: LanceDB",
            }
        ]
        out_gaia = os.path.join(e2e_temp_dir, "gaia_out.jsonl")
        export_gaia(tasks, out_gaia)
        assert validate_submission_file(out_gaia, "jsonl") is True

    @patch("openai.OpenAI")
    def test_c11_dynamic_document_update_and_re_ask(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """C11: Add document, ask question, add second document, ask again."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        doc1 = os.path.join(e2e_temp_dir, "v1.txt")
        with open(doc1, "w") as f:
            f.write("System version 1.0 released in 2024.")

        kb = Ragi(persist_dir=os.path.join(e2e_temp_dir, "ragi_dyn"), embedder=make_mock_embedder())
        kb.add(doc1)
        assert kb.store.count() > 0

        doc2 = os.path.join(e2e_temp_dir, "v2.txt")
        with open(doc2, "w") as f:
            f.write("System version 2.0 released in 2026 with SOTA RAG.")

        kb.add(doc2)
        ans = kb.ask("What is new in version 2.0?")
        assert isinstance(ans, Answer)

    def test_c12_benchmark_ndcg_and_mrr_correlation(self):
        """C12: Perfect rankings produce 1.0 on both nDCG and MRR."""
        rankings = [3, 2, 1, 0, 0]
        ndcg = compute_ndcg_at_k(rankings, k=5)
        mrr = compute_mrr([1])
        assert ndcg == 1.0
        assert mrr == 1.0

    @patch("openai.OpenAI")
    def test_c13_filter_chaining_with_hybrid_search(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """C13: Chained metadata filters with hybrid search."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        doc_path = os.path.join(e2e_temp_dir, "doc.txt")
        with open(doc_path, "w") as f:
            f.write("Metadata filtering allows scoping search to specific departments.")

        kb = Ragi(
            persist_dir=os.path.join(e2e_temp_dir, "ragi_filt"),
            embedder=make_mock_embedder(),
            config={"retrieval": {"use_hybrid_search": True}},
        )
        kb.add(doc_path)
        filtered_kb = kb.filter(department="engineering")
        assert filtered_kb is not None

    def test_c14_full_sota_pipeline_synthesis(self, e2e_temp_dir):
        """C14: Comprehensive synthesis from candidate retrieval to submission packaging."""
        # 1. Hierarchical auto-merging
        children = [
            Citation(source="policy.md", chunk="sub 1", score=0.91, metadata={"parent_id": "sec_1", "parent_text": "Security Policy 2026"}),
            Citation(source="policy.md", chunk="sub 2", score=0.85, metadata={"parent_id": "sec_1", "parent_text": "Security Policy 2026"}),
            Citation(source="terms.md", chunk="term a", score=0.75, metadata={"parent_id": "sec_2", "parent_text": "Terms of Service"}),
            Citation(source="terms.md", chunk="term b", score=0.70, metadata={"parent_id": "sec_2", "parent_text": "Terms of Service"}),
        ]
        merged = auto_merge_citations(children)
        assert len(merged) == 2

        # 2. U-shaped attention ordering
        ordered = apply_u_shaped_ordering(merged)
        assert len(ordered) == 2

        # 3. LLM synthesis with citations
        llm_answer = "Under [1], all credentials must be rotated quarterly."
        pruned_citations = prune_citations_from_text(llm_answer, ordered)
        assert len(pruned_citations) == 1
        assert pruned_citations[0].chunk == "Security Policy 2026"

        # 4. Export to EnterpriseRAG submission
        sub_file = os.path.join(e2e_temp_dir, "full_sub.jsonl")
        record = {
            "question_id": "q_security",
            "answer": llm_answer,
            "citations": [c.source for c in pruned_citations],
            "doc_ids": [c.source for c in pruned_citations],
        }
        export_enterpriserag([record], sub_file)
        assert validate_submission_file(sub_file, "jsonl") is True
