"""Unit tests for the RAG benchmark evaluation framework (evals/eval_benchmark.py).

Tests:
1. Metric calculations: nDCG@k, Recall@k, MRR@k, Precision@k under normal and edge conditions.
2. Dataset loaders: Cranfield and BEIR (SciFact) structures, qrels, and validation.
3. Benchmark runner execution: Condition A vs Condition B paired execution on sample queries.
4. CLI argument parsing and execution handling.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from evals.eval_benchmark import (
    BenchmarkDataset,
    BenchmarkDocument,
    BenchmarkQuery,
    BenchmarkRunner,
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


# ===========================================================================
# 1. Metric Calculations Tests
# ===========================================================================

class TestMetricCalculations:
    """Tests for nDCG@k, Recall@k, MRR@k, Precision@k."""

    def test_ndcg_perfect_ranking(self):
        """Perfect ranking returns nDCG 1.0."""
        perfect_scores = [3, 2, 1, 0, 0]
        assert compute_ndcg_at_k(perfect_scores, k=5) == 1.0

    def test_ndcg_imperfect_ranking(self):
        """Imperfect ranking returns nDCG strictly less than 1.0."""
        imperfect_scores = [0, 1, 2, 3, 0]
        score = compute_ndcg_at_k(imperfect_scores, k=5)
        assert 0.0 < score < 1.0

    def test_ndcg_all_zeros(self):
        """All zero relevance scores return nDCG 0.0."""
        assert compute_ndcg_at_k([0, 0, 0, 0], k=4) == 0.0

    def test_ndcg_empty_list(self):
        """Empty scores list returns 0.0."""
        assert compute_ndcg_at_k([], k=10) == 0.0

    def test_ndcg_k_boundary_zero(self):
        """k=0 returns 0.0 without division by zero."""
        assert compute_ndcg_at_k([3, 2, 1], k=0) == 0.0

    def test_ndcg_k_exceeds_length(self):
        """k exceeding list length handles gracefully."""
        score = compute_ndcg_at_k([2, 1], k=100)
        assert score == 1.0

    def test_ndcg_with_ground_truth_dict(self):
        """nDCG calculation using retrieved doc IDs and relevance dict."""
        retrieved = ["doc_1", "doc_2", "doc_3"]
        qrels = {"doc_1": 3, "doc_2": 2, "doc_4": 1}
        # doc_1 has rel 3, doc_2 has rel 2, doc_3 has rel 0. Ideal pool has [3, 2, 1].
        score = compute_ndcg_at_k(retrieved, qrels, k=3)
        assert 0.0 < score <= 1.0

    def test_ndcg_with_ground_truth_set(self):
        """nDCG calculation using retrieved doc IDs and relevance set."""
        retrieved = ["doc_1", "doc_2", "doc_3"]
        qrels = {"doc_1", "doc_2"}
        score = compute_ndcg_at_k(retrieved, qrels, k=2)
        assert score == 1.0

    def test_recall_at_k_calculation(self):
        """Recall@k computes fraction of relevant documents retrieved."""
        retrieved = ["doc_A", "doc_B", "doc_C"]
        ground_truth = {"doc_A", "doc_C", "doc_D", "doc_E"}
        recall = compute_recall_at_k(retrieved, ground_truth, k=3)
        assert recall == 0.5  # 2 hits out of 4 relevant docs

    def test_recall_at_k_perfect(self):
        """Recall@k returns 1.0 when all ground truth docs are retrieved."""
        retrieved = ["doc_A", "doc_B", "doc_C"]
        ground_truth = {"doc_A", "doc_B"}
        assert compute_recall_at_k(retrieved, ground_truth, k=3) == 1.0

    def test_recall_at_k_empty_inputs(self):
        """Recall returns 0.0 for empty retrieved or ground truth."""
        assert compute_recall_at_k([], {"doc_A"}, k=5) == 0.0
        assert compute_recall_at_k(["doc_A"], set(), k=5) == 0.0
        assert compute_recall_at_k(["doc_A"], {"doc_A"}, k=0) == 0.0

    def test_recall_duplicate_retrieved_handling(self):
        """Deduplicates retrieved doc IDs before computing recall."""
        retrieved = ["doc_A", "doc_A", "doc_B"]
        ground_truth = {"doc_A", "doc_B"}
        # Top 2 has only ["doc_A"], so 1 hit out of 2
        assert compute_recall_at_k(retrieved, ground_truth, k=2) == 0.5

    def test_precision_at_k_calculation(self):
        """Precision@k computes hits divided by k."""
        retrieved = ["doc_A", "doc_B", "doc_C"]
        ground_truth = {"doc_A", "doc_C"}
        assert compute_precision_at_k(retrieved, ground_truth, k=3) == pytest.approx(2 / 3)

    def test_precision_at_k_boundary_zero(self):
        """Precision returns 0.0 when k=0 or inputs empty."""
        assert compute_precision_at_k(["doc_A"], {"doc_A"}, k=0) == 0.0
        assert compute_precision_at_k([], {"doc_A"}, k=5) == 0.0
        assert compute_precision_at_k(["doc_A"], set(), k=5) == 0.0

    def test_mrr_at_k_first_hit(self):
        """MRR returns 1.0 if first retrieved document is relevant."""
        retrieved = ["doc_A", "doc_B"]
        ground_truth = {"doc_A"}
        assert compute_mrr_at_k(retrieved, ground_truth, k=5) == 1.0

    def test_mrr_at_k_second_hit(self):
        """MRR returns 0.5 if second retrieved document is first relevant."""
        retrieved = ["doc_B", "doc_A"]
        ground_truth = {"doc_A"}
        assert compute_mrr_at_k(retrieved, ground_truth, k=5) == 0.5

    def test_mrr_at_k_no_hits(self):
        """MRR returns 0.0 if no relevant documents appear in top k."""
        retrieved = ["doc_B", "doc_C"]
        ground_truth = {"doc_A"}
        assert compute_mrr_at_k(retrieved, ground_truth, k=2) == 0.0

    def test_mrr_legacy_ranks_list(self):
        """compute_mrr computes mean reciprocal rank from ranks list."""
        ranks = [1, 2, 4]  # 1/1 + 1/2 + 1/4 = 1.75 / 3 = 0.5833
        assert pytest.approx(compute_mrr(ranks), 0.01) == 0.5833

    def test_mrr_legacy_empty(self):
        """compute_mrr returns 0.0 for empty ranks list."""
        assert compute_mrr([]) == 0.0


# ===========================================================================
# 2. Benchmark Dataset Loaders Tests
# ===========================================================================

class TestBenchmarkDatasets:
    """Tests for Cranfield and BEIR dataset loading."""

    def test_load_cranfield_dataset(self):
        """Cranfield dataset loads valid documents and queries."""
        ds = load_cranfield_dataset()
        assert ds.name == "cranfield"
        assert len(ds.documents) >= 10
        assert len(ds.queries) >= 5
        assert all(isinstance(d, BenchmarkDocument) for d in ds.documents)
        assert all(isinstance(q, BenchmarkQuery) for q in ds.queries)
        assert all(len(q.ground_truth) > 0 for q in ds.queries)
        assert all(bool(q.stepback_query) for q in ds.queries)

    def test_load_beir_dataset(self):
        """BEIR (SciFact) dataset loads valid documents and queries."""
        ds = load_beir_dataset()
        assert ds.name == "beir"
        assert len(ds.documents) >= 10
        assert len(ds.queries) >= 5
        assert all(isinstance(d, BenchmarkDocument) for d in ds.documents)
        assert all(isinstance(q, BenchmarkQuery) for q in ds.queries)
        assert all(len(q.ground_truth) > 0 for q in ds.queries)

    def test_load_dataset_all(self):
        """load_dataset('all') combines Cranfield and BEIR datasets."""
        ds = load_dataset("all")
        assert ds.name == "all"
        assert len(ds.documents) >= 20
        assert len(ds.queries) >= 10

    def test_load_dataset_invalid_raises(self):
        """load_dataset raises ValueError for unrecognized dataset name."""
        with pytest.raises(ValueError, match="Unknown benchmark dataset"):
            load_dataset("unknown_dataset_xyz")


# ===========================================================================
# 3. Benchmark Runner Tests
# ===========================================================================

class TestBenchmarkRunner:
    """Tests for BenchmarkRunner execution."""

    @pytest.fixture
    def mini_dataset(self) -> BenchmarkDataset:
        """Create a lightweight synthetic benchmark dataset."""
        docs = [
            BenchmarkDocument(
                id="mini_001",
                title="Aerodynamic Surface Cooling",
                text="Surface cooling delays boundary layer transition at supersonic Mach numbers by stabilizing laminar flow.",
                metadata={"domain": "aero"},
            ),
            BenchmarkDocument(
                id="mini_002",
                title="Hypersonic Thermal Stresses",
                text="Aerodynamic heating in hypersonic entry creates intense stagnation point heat transfer and thermal stresses.",
                metadata={"domain": "aero"},
            ),
            BenchmarkDocument(
                id="mini_003",
                title="General Mechanical Structures",
                text="Structural spars and rib panels support mechanical loads and distribute stress across airframes.",
                metadata={"domain": "structures"},
            ),
        ]
        queries = [
            BenchmarkQuery(
                id="q_01",
                query="How does surface cooling affect boundary layer transition at supersonic speeds?",
                stepback_query="What principles govern boundary layer stability in supersonic aerodynamics?",
                ground_truth={"mini_001": 3},
            ),
            BenchmarkQuery(
                id="q_02",
                query="What happens to stagnation point heat transfer during hypersonic entry?",
                stepback_query="What heat transfer mechanisms operate during atmospheric entry?",
                ground_truth={"mini_002": 3},
            ),
        ]
        return BenchmarkDataset(name="mini", description="Mini Test Dataset", documents=docs, queries=queries)

    def test_benchmark_runner_single_condition(self, mini_dataset):
        """Runner evaluates a single condition and returns ConditionResult."""
        runner = BenchmarkRunner()
        res = runner.evaluate_condition(mini_dataset, condition="enhanced", k=3)
        assert res.condition == "enhanced"
        assert res.dataset_name == "mini"
        assert res.metrics.ndcg_at_10 >= 0.0
        assert res.performance.total_queries == 2
        assert len(res.per_query_results) == 2

    def test_benchmark_runner_paired_comparison(self, mini_dataset):
        """Runner executes paired baseline vs enhanced evaluation."""
        runner = BenchmarkRunner()
        paired = runner.run_paired_evaluation(mini_dataset, k=3)
        assert paired["dataset"] == "mini"
        assert "baseline" in paired
        assert "enhanced" in paired
        assert "comparisons" in paired
        assert "gain_ndcg_pct" in paired["comparisons"]

    def test_run_benchmark_function_and_saving(self, tmp_path):
        """run_benchmark function executes and writes formatted JSON output."""
        out_file = tmp_path / "results.json"
        res = run_benchmark(dataset_name="cranfield", condition="baseline", output_path=str(out_file), k=5)
        assert out_file.is_file()
        with open(out_file, "r") as f:
            loaded = json.load(f)
        assert "datasets" in loaded
        assert "cranfield" in loaded["datasets"]

    def test_format_cli_summary(self):
        """format_cli_summary formats results into readable table string."""
        sample_results = {
            "timestamp": "2026-10-31T23:00:00Z",
            "datasets": {
                "cranfield": {
                    "baseline": {
                        "metrics": {"ndcg@10": 0.62, "recall@10": 0.70, "recall@5": 0.55, "mrr@10": 0.60},
                        "performance": {"mean_latency_ms": 15.0, "throughput_qps": 66.0},
                    },
                    "enhanced": {
                        "metrics": {"ndcg@10": 0.75, "recall@10": 0.85, "recall@5": 0.70, "mrr@10": 0.72},
                        "performance": {"mean_latency_ms": 18.0, "throughput_qps": 55.0},
                    },
                    "comparisons": {"gain_ndcg_pct": 20.97, "gain_recall_10_pct": 21.43, "gain_mrr_pct": 20.0},
                }
            },
            "overall_summary": {
                "baseline": {"ndcg@10": 0.62, "recall@5": 0.55, "recall@10": 0.70, "mrr@10": 0.60},
                "enhanced": {"ndcg@10": 0.75, "recall@5": 0.70, "recall@10": 0.85, "mrr@10": 0.72},
                "gain_ndcg_pct": 20.97,
                "gain_recall_10_pct": 21.43,
                "gain_mrr_pct": 20.0,
                "verified": True,
            },
        }
        summary_text = format_cli_summary(sample_results)
        assert "PIRAGI SOTA RAG BENCHMARK EVALUATION RESULTS" in summary_text
        assert "CRANFIELD" in summary_text
        assert "+20.97%" in summary_text
        assert "PASSED" in summary_text


# ===========================================================================
# 4. CLI Argument Parsing Tests
# ===========================================================================

class TestCLIArgumentParsing:
    """Tests for CLI arguments and main entrypoint."""

    def test_parse_args_defaults(self):
        """Default arguments match expected values."""
        args = parse_args([])
        assert args.dataset == "all"
        assert args.condition == "both"
        assert args.output == "evals/results/benchmark_results.json"
        assert args.k == 10

    def test_parse_args_custom(self):
        """Custom command line flags are parsed correctly."""
        argv = ["--dataset", "beir", "--condition", "enhanced", "--output", "/tmp/beir.json", "--k", "5"]
        args = parse_args(argv)
        assert args.dataset == "beir"
        assert args.condition == "enhanced"
        assert args.output == "/tmp/beir.json"
        assert args.k == 5

    @patch("evals.eval_benchmark.run_benchmark")
    def test_cli_main_invocation(self, mock_run):
        """main entrypoint calls run_benchmark with parsed flags."""
        mock_run.return_value = {
            "timestamp": "2026-10-31T23:00:00Z",
            "datasets": {},
        }
        exit_code = main(["--dataset", "cranfield", "--condition", "baseline"])
        assert exit_code == 0
        mock_run.assert_called_once_with(
            dataset_name="cranfield",
            condition="baseline",
            output_path="evals/results/benchmark_results.json",
            k=10,
        )
