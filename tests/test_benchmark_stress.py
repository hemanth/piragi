"""Adversarial stress test suite for Milestone 3 Benchmark Evaluation Framework.

Covers:
1. Metric calculations under boundary conditions (k=0, negative k, large k, empty lists,
   duplicate IDs, inverted rankings, graded vs binary relevance, float relevance).
2. HierarchicalChunker boundary conditions (empty text, single char, exact parent/child sizes,
   no-space strings, overlap edge cases, runaway chunk prevention).
3. Benchmark Runner CLI arguments, execution flow, output schema validation, and summary formatting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evals.eval_benchmark import (
    BenchmarkDataset,
    BenchmarkDocument,
    BenchmarkQuery,
    BenchmarkRunner,
    ConditionResult,
    EvaluationMetrics,
    PerformanceMetrics,
    compute_mrr,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    format_cli_summary,
    load_beir_dataset,
    load_cranfield_dataset,
    load_dataset,
    main,
    parse_args,
    run_benchmark,
)
from piragi.semantic_chunking import Chunk, Document, HierarchicalChunker


# ===========================================================================
# 1. Metric Calculations Stress & Boundary Tests
# ===========================================================================

class TestMetricBoundaryStress:
    """Empirical boundary condition stress testing for evaluation metrics."""

    # --- nDCG@k Stress ---

    def test_ndcg_k_zero(self):
        """k=0 must return 0.0 without division by zero."""
        retrieved = ["doc1", "doc2"]
        qrels = {"doc1": 3, "doc2": 2}
        assert compute_ndcg_at_k(retrieved, qrels, k=0) == 0.0

    def test_ndcg_k_negative(self):
        """Negative k must return 0.0 safely."""
        retrieved = ["doc1", "doc2"]
        qrels = {"doc1": 3}
        assert compute_ndcg_at_k(retrieved, qrels, k=-1) == 0.0
        assert compute_ndcg_at_k(retrieved, qrels, k=-100) == 0.0

    def test_ndcg_k_one_hit(self):
        """k=1 with top document relevant must return 1.0."""
        retrieved = ["doc1", "doc2"]
        qrels = {"doc1": 3, "doc2": 2}
        assert compute_ndcg_at_k(retrieved, qrels, k=1) == 1.0

    def test_ndcg_k_one_miss(self):
        """k=1 with top document non-relevant must return 0.0."""
        retrieved = ["doc_irrelevant", "doc1"]
        qrels = {"doc1": 3}
        assert compute_ndcg_at_k(retrieved, qrels, k=1) == 0.0

    def test_ndcg_k_large_sparse(self):
        """k=1000 when retrieved has only 3 items must handle gracefully."""
        retrieved = ["doc1", "doc2", "doc3"]
        qrels = {"doc1": 3, "doc2": 2, "doc3": 1}
        # Since retrieved are in ideal order, nDCG must be 1.0
        assert compute_ndcg_at_k(retrieved, qrels, k=1000) == 1.0

    def test_ndcg_empty_retrieved_list(self):
        """Empty retrieved list must return 0.0."""
        assert compute_ndcg_at_k([], {"doc1": 3}, k=10) == 0.0

    def test_ndcg_empty_ground_truth(self):
        """Empty ground truth must return 0.0."""
        assert compute_ndcg_at_k(["doc1", "doc2"], {}, k=10) == 0.0
        assert compute_ndcg_at_k(["doc1", "doc2"], set(), k=10) == 0.0

    def test_ndcg_both_empty(self):
        """Both empty must return 0.0."""
        assert compute_ndcg_at_k([], {}, k=10) == 0.0

    def test_ndcg_duplicate_retrieved_ids_does_not_inflate(self):
        """Multiple occurrences of the same relevant doc ID must not inflate nDCG beyond 1.0."""
        retrieved = ["doc1", "doc1", "doc1", "doc1"]
        qrels = {"doc1": 3}
        score = compute_ndcg_at_k(retrieved, qrels, k=4)
        assert score == 1.0

        # When second relevant doc exists, duplicate first doc must not replace second doc gain
        qrels_multi = {"doc1": 3, "doc2": 2}
        score_dup = compute_ndcg_at_k(retrieved, qrels_multi, k=4)
        # Ideal has [3, 2], but retrieved only got doc1 once, so score must be strictly < 1.0
        assert 0.0 < score_dup < 1.0
        # Specifically: dcg = 7.0, idcg = 7.0 + 3.0 / log2(3) = 8.8928
        expected_score = 7.0 / (7.0 + 3.0 / math.log2(3))
        assert pytest.approx(score_dup, 1e-4) == expected_score

    def test_ndcg_inverted_rankings(self):
        """Inverted rankings (least relevant at rank 1, most relevant at rank 3) gives proper penalty."""
        # Ideal: doc3 (3), doc2 (2), doc1 (1)
        # Retrieved: doc1 (1), doc2 (2), doc3 (3)
        retrieved = ["doc1", "doc2", "doc3"]
        qrels = {"doc1": 1, "doc2": 2, "doc3": 3}
        score = compute_ndcg_at_k(retrieved, qrels, k=3)
        # dcg = (2^1 - 1)/1 + (2^2 - 1)/log2(3) + (2^3 - 1)/2 = 1 + 3/1.58496 + 3.5 = 6.392789
        # idcg = (2^3 - 1)/1 + (2^2 - 1)/log2(3) + (2^1 - 1)/2 = 7 + 3/1.58496 + 0.5 = 9.392789
        expected = 6.392789 / 9.392789
        assert pytest.approx(score, 1e-4) == expected
        assert 0.0 < score < 1.0

    def test_ndcg_graded_vs_binary_relevance(self):
        """Graded relevance distinguishes highly relevant from partially relevant."""
        qrels_graded = {"doc_high": 3, "doc_med": 1}
        # Ranking A: doc_high at rank 1, doc_med at rank 2
        score_a = compute_ndcg_at_k(["doc_high", "doc_med"], qrels_graded, k=2)
        # Ranking B: doc_med at rank 1, doc_high at rank 2
        score_b = compute_ndcg_at_k(["doc_med", "doc_high"], qrels_graded, k=2)
        assert score_a == 1.0
        assert score_b < score_a
        assert score_b > 0.0

    def test_ndcg_float_relevance_scores(self):
        """Float relevance scores in qrels dictionary are supported without error."""
        retrieved = ["doc1", "doc2"]
        qrels = {"doc1": 2.5, "doc2": 1.25}
        score = compute_ndcg_at_k(retrieved, qrels, k=2)
        assert score == 1.0

    def test_ndcg_all_zero_relevance(self):
        """All-zero relevance in ground truth returns 0.0."""
        retrieved = ["doc1", "doc2"]
        qrels = {"doc1": 0, "doc2": 0}
        assert compute_ndcg_at_k(retrieved, qrels, k=2) == 0.0

    def test_ndcg_disjoint_retrieved_and_qrels(self):
        """Zero overlap between retrieved and ground truth returns 0.0."""
        retrieved = ["doc1", "doc2"]
        qrels = {"doc3": 3, "doc4": 2}
        assert compute_ndcg_at_k(retrieved, qrels, k=2) == 0.0

    def test_ndcg_monotonicity_and_bounds(self):
        """nDCG must strictly stay bounded in [0.0, 1.0] for all combinations."""
        retrieved = [f"d_{i}" for i in range(20)]
        qrels = {f"d_{i}": (i % 4) for i in range(15)}
        for k in range(1, 25):
            score = compute_ndcg_at_k(retrieved, qrels, k=k)
            assert 0.0 <= score <= 1.0

    def test_ndcg_signature_1_float_and_int(self):
        """Signature 1 (relevance_scores: list) handles ints and floats."""
        assert compute_ndcg_at_k([3, 2, 1], k=3) == 1.0
        assert compute_ndcg_at_k([1, 2, 3], k=3) < 1.0
        assert compute_ndcg_at_k([2.5, 1.5, 0.5], k=3) == 1.0
        assert compute_ndcg_at_k([0.0, 0.0], k=2) == 0.0

    def test_ndcg_signature_1_non_numeric(self):
        """Signature 1 with non-numeric items returns 0.0 gracefully."""
        assert compute_ndcg_at_k(["str1", "str2"], k=2) == 0.0


    # --- Recall@k Stress ---

    def test_recall_k_zero_and_negative(self):
        """k <= 0 returns 0.0."""
        assert compute_recall_at_k(["d1", "d2"], {"d1"}, k=0) == 0.0
        assert compute_recall_at_k(["d1", "d2"], {"d1"}, k=-5) == 0.0

    def test_recall_k_one(self):
        """k=1 returns correct ratio."""
        assert compute_recall_at_k(["d1", "d2"], {"d1", "d2"}, k=1) == 0.5
        assert compute_recall_at_k(["d3", "d2"], {"d1", "d2"}, k=1) == 0.0

    def test_recall_k_large(self):
        """k=1000 handles gracefully without out of bounds."""
        assert compute_recall_at_k(["d1", "d2"], {"d1", "d2"}, k=1000) == 1.0

    def test_recall_empty_inputs(self):
        """Empty retrieved or ground truth returns 0.0."""
        assert compute_recall_at_k([], {"d1"}, k=5) == 0.0
        assert compute_recall_at_k(["d1"], set(), k=5) == 0.0
        assert compute_recall_at_k([], {}, k=5) == 0.0

    def test_recall_duplicate_retrieved_ids(self):
        """Duplicate retrieved doc IDs do not count as multiple hits."""
        retrieved = ["d1", "d1", "d1", "d2"]
        qrels = {"d1", "d2"}
        # At k=2, only "d1" is retrieved -> recall = 1/2 = 0.5
        assert compute_recall_at_k(retrieved, qrels, k=2) == 0.5
        # At k=4, "d1" and "d2" retrieved -> recall = 2/2 = 1.0
        assert compute_recall_at_k(retrieved, qrels, k=4) == 1.0

    def test_recall_monotonicity_with_increasing_k(self):
        """Recall@k must be monotonically non-decreasing as k increases."""
        retrieved = [f"d_{i}" for i in range(20)]
        qrels = {"d_2", "d_7", "d_12", "d_18"}
        recalls = [compute_recall_at_k(retrieved, qrels, k=k) for k in range(1, 25)]
        for i in range(len(recalls) - 1):
            assert recalls[i] <= recalls[i + 1]

    def test_recall_ground_truth_types(self):
        """Ground truth accepted as set, dict, list, or tuple."""
        retrieved = ["d1", "d2"]
        assert compute_recall_at_k(retrieved, {"d1": 3, "d2": 1}, k=2) == 1.0
        assert compute_recall_at_k(retrieved, {"d1", "d2"}, k=2) == 1.0
        assert compute_recall_at_k(retrieved, ["d1", "d2"], k=2) == 1.0
        assert compute_recall_at_k(retrieved, ("d1", "d2"), k=2) == 1.0


    # --- Precision@k Stress ---

    def test_precision_k_zero_and_negative(self):
        """k <= 0 returns 0.0."""
        assert compute_precision_at_k(["d1"], {"d1"}, k=0) == 0.0
        assert compute_precision_at_k(["d1"], {"d1"}, k=-1) == 0.0

    def test_precision_k_one(self):
        """k=1 returns 1.0 or 0.0."""
        assert compute_precision_at_k(["d1", "d2"], {"d1"}, k=1) == 1.0
        assert compute_precision_at_k(["d2", "d1"], {"d1"}, k=1) == 0.0

    def test_precision_k_large(self):
        """k=1000 with 2 hits returns 2 / 1000 = 0.002."""
        assert compute_precision_at_k(["d1", "d2"], {"d1", "d2"}, k=1000) == 0.002

    def test_precision_duplicate_retrieved(self):
        """Deduplicates retrieved IDs so duplicate hits do not inflate precision."""
        retrieved = ["d1", "d1"]
        # In top 2, unique hit is only d1, divided by k=2 -> 1/2 = 0.5
        assert compute_precision_at_k(retrieved, {"d1"}, k=2) == 0.5

    def test_precision_bounds(self):
        """Precision must always stay within [0.0, 1.0]."""
        retrieved = [f"d_{i}" for i in range(20)]
        qrels = {f"d_{i}" for i in range(5)}
        for k in range(1, 25):
            prec = compute_precision_at_k(retrieved, qrels, k=k)
            assert 0.0 <= prec <= 1.0


    # --- MRR@k Stress ---

    def test_mrr_k_zero_and_negative(self):
        """k <= 0 returns 0.0."""
        assert compute_mrr_at_k(["d1"], {"d1"}, k=0) == 0.0
        assert compute_mrr_at_k(["d1"], {"d1"}, k=-2) == 0.0

    def test_mrr_first_rank_hit(self):
        """Hit at rank 1 gives reciprocal rank 1.0."""
        assert compute_mrr_at_k(["d1", "d2"], {"d1"}, k=5) == 1.0

    def test_mrr_last_rank_hit(self):
        """Hit at rank 10 gives reciprocal rank 0.1."""
        retrieved = [f"dist_{i}" for i in range(9)] + ["target"]
        assert pytest.approx(compute_mrr_at_k(retrieved, {"target"}, k=10), 1e-4) == 0.1

    def test_mrr_beyond_k_returns_zero(self):
        """Hit at rank 11 when k=10 returns 0.0."""
        retrieved = [f"dist_{i}" for i in range(10)] + ["target"]
        assert compute_mrr_at_k(retrieved, {"target"}, k=10) == 0.0

    def test_mrr_duplicates_before_hit(self):
        """Duplicate distractor docs correctly shift hit rank."""
        retrieved = ["dist", "dist", "dist", "target"]
        # rank 1: dist, rank 2: dist, rank 3: dist, rank 4: target -> 1/4 = 0.25
        assert compute_mrr_at_k(retrieved, {"target"}, k=5) == 0.25

    def test_mrr_empty_inputs(self):
        """Empty retrieved or ground truth returns 0.0."""
        assert compute_mrr_at_k([], {"d1"}, k=5) == 0.0
        assert compute_mrr_at_k(["d1"], set(), k=5) == 0.0
        assert compute_mrr_at_k([], None, k=5) == 0.0

    def test_mrr_ranks_list_signature(self):
        """List of ranks signature computes mean of valid reciprocals."""
        # ranks: [1, 2, 4, 15] with k=10 -> [1.0, 0.5, 0.25, 0.0] -> mean = 1.75 / 4 = 0.4375
        ranks = [1, 2, 4, 15]
        score = compute_mrr_at_k(ranks, k=10)
        assert pytest.approx(score, 1e-4) == 0.4375

    def test_mrr_ranks_list_all_out_of_k(self):
        """All ranks > k returns 0.0."""
        ranks = [11, 12, 13]
        assert compute_mrr_at_k(ranks, k=10) == 0.0

    def test_mrr_ranks_list_zero_and_negative_ranks(self):
        """Non-positive ranks evaluate as 0.0 reciprocals."""
        ranks = [0, -1, 1]
        # [0.0, 0.0, 1.0] -> mean = 1/3 = 0.3333
        assert pytest.approx(compute_mrr_at_k(ranks, k=10), 1e-4) == 1.0 / 3.0

    def test_mrr_legacy_function(self):
        """compute_mrr computes mean reciprocal rank across query rank integers."""
        assert compute_mrr([]) == 0.0
        assert compute_mrr([1, 1, 1]) == 1.0
        assert pytest.approx(compute_mrr([1, 2]), 1e-4) == 0.75
        assert compute_mrr([0, 0]) == 0.0


# ===========================================================================
# 2. Hierarchical Chunker Boundary Stress Tests
# ===========================================================================

class TestHierarchicalChunkerBoundaryStress:
    """Stress tests for HierarchicalChunker under edge and adversarial inputs."""

    def test_chunker_empty_string(self):
        """Empty string must produce 0 parent chunks and 0 child chunks."""
        chunker = HierarchicalChunker(parent_chunk_size=2000, child_chunk_size=400, overlap=50)
        parents, children = chunker.chunk_document(Document(content="", source="test_empty"))
        assert len(parents) == 0
        assert len(children) == 0

    def test_chunker_single_character(self):
        """Single character produces exactly 1 parent and 1 child chunk without loop."""
        chunker = HierarchicalChunker(parent_chunk_size=2000, child_chunk_size=400, overlap=50)
        parents, children = chunker.chunk_document(Document(content="A", source="test_single"))
        assert len(parents) == 1
        assert len(children) == 1
        assert parents[0].text == "A"
        assert children[0].text == "A"
        assert children[0].metadata["chunk_type"] == "child"
        assert children[0].metadata["parent_index"] == 0
        assert children[0].metadata["parent_text"] == "A"

    def test_chunker_whitespace_only(self):
        """Whitespace-only string produces 0 chunks after stripping."""
        chunker = HierarchicalChunker(parent_chunk_size=2000, child_chunk_size=400, overlap=50)
        parents, children = chunker.chunk_document(Document(content="   \n\t  \r\n  ", source="test_ws"))
        assert len(parents) == 0
        assert len(children) == 0

    def test_chunker_exact_child_size(self):
        """Content exactly equal to child_chunk_size produces 1 parent and 1 child chunk."""
        chunk_size = 400
        chunker = HierarchicalChunker(parent_chunk_size=2000, child_chunk_size=chunk_size, overlap=50)
        content = "X" * chunk_size
        parents, children = chunker.chunk_document(Document(content=content, source="test_exact_child"))
        assert len(parents) == 1
        assert len(children) == 1
        assert len(children[0].text) == chunk_size

    def test_chunker_child_size_plus_one(self):
        """Content equal to child_chunk_size + 1 produces 2 child chunks with proper overlap."""
        chunk_size = 400
        overlap = 50
        chunker = HierarchicalChunker(parent_chunk_size=2000, child_chunk_size=chunk_size, overlap=overlap)
        content = "Y" * (chunk_size + 1)
        parents, children = chunker.chunk_document(Document(content=content, source="test_plus_one"))
        assert len(parents) == 1
        assert len(children) == 2
        assert len(children[0].text) == chunk_size
        # Second chunk starts at (400 - 50 = 350) and goes to 401 -> length 51
        assert len(children[1].text) == 51

    def test_chunker_child_size_minus_one(self):
        """Content equal to child_chunk_size - 1 produces 1 child chunk."""
        chunk_size = 400
        chunker = HierarchicalChunker(parent_chunk_size=2000, child_chunk_size=chunk_size, overlap=50)
        content = "Z" * (chunk_size - 1)
        parents, children = chunker.chunk_document(Document(content=content, source="test_minus_one"))
        assert len(parents) == 1
        assert len(children) == 1
        assert len(children[0].text) == chunk_size - 1

    def test_chunker_exact_parent_size(self):
        """Content exactly equal to parent_chunk_size produces 1 parent chunk."""
        parent_size = 2000
        chunker = HierarchicalChunker(parent_chunk_size=parent_size, child_chunk_size=400, overlap=50)
        content = "W" * parent_size
        parents, children = chunker.chunk_document(Document(content=content, source="test_exact_parent"))
        assert len(parents) == 1
        assert len(parents[0].text) == parent_size
        # 2000 chars divided by (400 - 50 = 350) stride = ceil(2000 / 350) = 6 children
        assert len(children) == 6

    def test_chunker_parent_size_plus_one(self):
        """Content equal to parent_chunk_size + 1 produces 2 parent chunks."""
        parent_size = 2000
        chunker = HierarchicalChunker(parent_chunk_size=parent_size, child_chunk_size=400, overlap=50)
        content = "W" * (parent_size + 1)
        parents, children = chunker.chunk_document(Document(content=content, source="test_parent_plus_one"))
        assert len(parents) == 2
        assert len(parents[0].text) == parent_size
        assert len(parents[1].text) == 1

    def test_chunker_massive_string_without_delimiters(self):
        """5,000 characters without a single space or punctuation chunks safely without hanging."""
        chunker = HierarchicalChunker(parent_chunk_size=2000, child_chunk_size=400, overlap=50)
        content = "A" * 5000
        t0 = time.perf_counter()
        parents, children = chunker.chunk_document(Document(content=content, source="test_dense"))
        duration = time.perf_counter() - t0
        assert duration < 0.2  # Must execute in < 200ms
        assert len(parents) == 3
        assert len(children) == 15
        # Verify all children have valid non-empty text
        assert all(len(c.text) > 0 for c in children)

    def test_chunker_repeated_sentence_delimiters(self):
        """String of repeating sentence boundaries ('. . . ! ? ') does not cause loop or crash."""
        chunker = HierarchicalChunker(parent_chunk_size=50, child_chunk_size=20, overlap=5)
        content = ". . . ! ? " * 20
        parents, children = chunker.chunk_document(Document(content=content, source="test_delims"))
        assert len(parents) > 0
        assert len(children) > 0

    def test_chunker_overlap_greater_than_or_equal_to_child_size(self):
        """Adversarial configuration where overlap >= child_size advances safely via start+1 guarantee."""
        # In lines 578-580: new_start = end - overlap; start = max(new_start, start + 1)
        chunker = HierarchicalChunker(parent_chunk_size=200, child_chunk_size=40, overlap=50)
        content = "Word " * 30
        t0 = time.perf_counter()
        parents, children = chunker.chunk_document(Document(content=content, source="test_huge_overlap"))
        duration = time.perf_counter() - t0
        assert duration < 0.2
        assert len(parents) > 0
        assert len(children) > 0

    def test_chunker_runaway_regression_check(self):
        """Regression verification: 24 documents of 300 chars must produce ~24 children, NOT 1231."""
        chunker = HierarchicalChunker(parent_chunk_size=2000, child_chunk_size=400, overlap=50)
        total_children = 0
        for i in range(24):
            doc = Document(content="Authentic sentence of aeronautical engineering test text. " * 5, source=f"doc_{i}")
            p, c = chunker.chunk_document(doc)
            total_children += len(c)
        # Each 300-char doc fits within one 400-char child chunk -> total children should be 24
        assert total_children == 24


# ===========================================================================
# 3. Benchmark Runner CLI & Output Schema Stress Tests
# ===========================================================================

class TestBenchmarkRunnerCLIAndSchemaStress:
    """Stress testing benchmark CLI combinations and output JSON schemas."""

    def test_parse_args_all_valid_permutations(self):
        """All valid permutations of dataset and condition are parsed correctly."""
        datasets = ["cranfield", "beir", "all"]
        conditions = ["baseline", "enhanced", "both"]
        for ds in datasets:
            for cond in conditions:
                argv = ["--dataset", ds, "--condition", cond, "--k", "5"]
                args = parse_args(argv)
                assert args.dataset == ds
                assert args.condition == cond
                assert args.k == 5

    def test_parse_args_invalid_dataset_fails(self):
        """Invalid dataset name causes argument parsing exit."""
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "invalid_dataset_name"])

    def test_parse_args_invalid_condition_fails(self):
        """Invalid condition name causes argument parsing exit."""
        with pytest.raises(SystemExit):
            parse_args(["--condition", "super_enhanced"])

    @patch("evals.eval_benchmark.BenchmarkRunner")
    def test_run_benchmark_baseline_only_schema(self, mock_runner_cls, tmp_path):
        """Benchmark run with condition='baseline' generates valid JSON schema without 'enhanced'."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        mock_runner.evaluate_condition.return_value = ConditionResult(
            condition="baseline",
            dataset_name="cranfield",
            metrics=EvaluationMetrics(0.80, 0.70, 0.70, 0.85, 0.14),
            performance=PerformanceMetrics(50.0, 45.0, 60.0, 20.0, 5, 0.25),
            per_query_results=[{"query_id": "q1", "ndcg@10": 0.80}],
        )

        out_path = tmp_path / "deep" / "nested" / "results_baseline.json"
        res = run_benchmark(dataset_name="cranfield", condition="baseline", output_path=str(out_path), k=5)

        assert out_path.is_file()
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["timestamp"]
        assert data["dataset_selection"] == "cranfield"
        assert data["condition_selection"] == "baseline"
        assert data["k"] == 5
        assert "cranfield" in data["datasets"]
        assert "baseline" in data["datasets"]["cranfield"]
        assert "enhanced" not in data["datasets"]["cranfield"]
        assert "overall_summary" not in data

        # Formatting must not crash when enhanced is absent
        summary = format_cli_summary(data)
        assert "PIRAGI SOTA RAG BENCHMARK EVALUATION RESULTS" in summary
        assert "CRANFIELD" in summary

    @patch("evals.eval_benchmark.BenchmarkRunner")
    def test_run_benchmark_enhanced_only_schema(self, mock_runner_cls, tmp_path):
        """Benchmark run with condition='enhanced' generates valid JSON schema without 'baseline'."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        mock_runner.evaluate_condition.return_value = ConditionResult(
            condition="enhanced",
            dataset_name="beir",
            metrics=EvaluationMetrics(0.95, 0.90, 0.90, 1.0, 0.18),
            performance=PerformanceMetrics(75.0, 70.0, 85.0, 13.0, 5, 0.38),
            per_query_results=[{"query_id": "q1", "ndcg@10": 0.95}],
        )

        out_path = tmp_path / "results_enhanced.json"
        res = run_benchmark(dataset_name="beir", condition="enhanced", output_path=str(out_path), k=10)

        assert out_path.is_file()
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["dataset_selection"] == "beir"
        assert data["condition_selection"] == "enhanced"
        assert "beir" in data["datasets"]
        assert "enhanced" in data["datasets"]["beir"]
        assert "baseline" not in data["datasets"]["beir"]

        summary = format_cli_summary(data)
        assert "BEIR" in summary

    @patch("evals.eval_benchmark.BenchmarkRunner")
    def test_run_benchmark_both_conditions_full_schema(self, mock_runner_cls, tmp_path):
        """Benchmark run with condition='both' and dataset='all' produces complete schema with overall_summary."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        def paired_side_effect(dataset, k=10):
            base_ndcg = 0.80 if dataset.name == "cranfield" else 0.88
            enh_ndcg = 0.95 if dataset.name == "cranfield" else 0.98
            return {
                "dataset": dataset.name,
                "baseline": {
                    "condition": "baseline",
                    "dataset": dataset.name,
                    "metrics": {"ndcg@10": base_ndcg, "recall@5": 0.65, "recall@10": 0.85, "mrr@10": 0.80, "precision@10": 0.15},
                    "performance": {"mean_latency_ms": 45.0, "p50_latency_ms": 40.0, "p95_latency_ms": 55.0, "throughput_qps": 22.0, "total_queries": 5, "total_duration_sec": 0.22},
                    "per_query_results": [],
                },
                "enhanced": {
                    "condition": "enhanced",
                    "dataset": dataset.name,
                    "metrics": {"ndcg@10": enh_ndcg, "recall@5": 0.85, "recall@10": 0.95, "mrr@10": 1.0, "precision@10": 0.19},
                    "performance": {"mean_latency_ms": 65.0, "p50_latency_ms": 60.0, "p95_latency_ms": 80.0, "throughput_qps": 15.0, "total_queries": 5, "total_duration_sec": 0.33},
                    "per_query_results": [],
                },
                "comparisons": {
                    "gain_ndcg_pct": round((enh_ndcg - base_ndcg) / base_ndcg * 100, 2),
                    "gain_recall_10_pct": 11.76,
                    "gain_mrr_pct": 25.0,
                    "verified": True,
                },
            }

        mock_runner.run_paired_evaluation.side_effect = paired_side_effect

        out_path = tmp_path / "results_all.json"
        res = run_benchmark(dataset_name="all", condition="both", output_path=str(out_path), k=10)

        assert out_path.is_file()
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "cranfield" in data["datasets"]
        assert "beir" in data["datasets"]
        assert "overall_summary" in data

        ov = data["overall_summary"]
        assert "baseline" in ov
        assert "enhanced" in ov
        assert "gain_ndcg_pct" in ov
        assert "gain_recall_10_pct" in ov
        assert "gain_mrr_pct" in ov
        assert "verified" in ov
        assert ov["verified"] is True
        assert ov["gain_ndcg_pct"] >= 10.0

        summary = format_cli_summary(data)
        assert "OVERALL AGGREGATE SUMMARY (Cranfield + BEIR)" in summary
        assert "PASSED (Enhanced strictly outperforms Baseline)" in summary
