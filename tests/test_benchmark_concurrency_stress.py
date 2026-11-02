"""Milestone 3 Stress & Empirical Challenge Suite: Benchmark Concurrency, Scaling & Reproducibility.

Adversarial Stress Dimensions:
1. Thread-safety: Concurrent multi-threaded benchmark evaluation sharing runner & embedder.
2. Process isolation: Multi-process CLI invocation verifying absence of store collision or locks.
3. Dataset scaling: Large corpus (60+ documents, 180+ sections) verifying chunking and indexing scaling.
4. Statistical reproducibility: Empirical verification that nDCG@10, Recall, and MRR metrics have zero variance across runs.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from evals.eval_benchmark import (
    BenchmarkDataset,
    BenchmarkDocument,
    BenchmarkQuery,
    BenchmarkRunner,
    load_cranfield_dataset,
    run_benchmark,
)


@pytest.fixture
def scaling_dataset() -> BenchmarkDataset:
    """Generate a scaled synthetic benchmark dataset with 60 documents and multi-topic distractors."""
    docs = []
    # 10 core target documents
    for i in range(1, 11):
        docs.append(
            BenchmarkDocument(
                id=f"scale_target_{i:04d}",
                title=f"Theoretical and Applied Fluid Dynamics Volume {i}",
                text=(
                    f"Fundamental investigation {i} on aerodynamic boundary layer stability, "
                    f"shock wave interaction, and thermal protection systems. "
                    f"Experimental parameters measured at Mach {i * 0.8:.1f} with localized laminar "
                    f"flow suction reducing skin friction drag by {15 + i}%."
                ),
                metadata={"topic": "fluid_dynamics", "volume": i},
            )
        )
    # 50 distractor documents
    for i in range(1, 51):
        docs.append(
            BenchmarkDocument(
                id=f"scale_distractor_{i:04d}",
                title=f"General Engineering Compendium Entry {i}",
                text=(
                    f"Material properties and manufacturing tolerances for titanium structural ribs {i}. "
                    f"Quality assurance protocols and mechanical fatigue testing under cyclic loading."
                ),
                metadata={"topic": "structures", "entry": i},
            )
        )

    queries = [
        BenchmarkQuery(
            id=f"sq_{i:02d}",
            query=f"What are the boundary layer stability characteristics at Mach {i * 0.8:.1f}?",
            stepback_query="What principles govern boundary layer stability in compressible aerodynamics?",
            ground_truth={f"scale_target_{i:04d}": 3},
        )
        for i in range(1, 6)
    ]

    return BenchmarkDataset(
        name="scaled_benchmark",
        description="Scaled Synthetic Aeronautics Dataset (60 docs, 5 queries)",
        documents=docs,
        queries=queries,
    )


class DeterministicMockEmbedder:
    """Thread-safe mock embedder generating deterministic 768-dim normalized embeddings."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_query(self, query: str) -> List[float]:
        seed = sum(ord(c) for c in query) % 500
        vec = [float(np.sin(seed + i * 0.1)) for i in range(self.dim)]
        norm = float(np.linalg.norm(vec)) or 1.0
        return [float(x / norm) for x in vec]

    def embed_chunks(self, chunks: List[Any], on_progress=None) -> List[Any]:
        for chunk in chunks:
            seed = sum(ord(c) for c in chunk.text) % 500
            vec = [float(np.sin(seed + i * 0.1)) for i in range(self.dim)]
            norm = float(np.linalg.norm(vec)) or 1.0
            chunk.embedding = [float(x / norm) for x in vec]
        return chunks

    def get_dimensions(self) -> int:
        return self.dim


class TestBenchmarkConcurrencyAndIsolation:
    """Stress-test concurrent execution of the benchmark framework across threads and processes."""

    def test_multithreaded_benchmark_execution_shared_runner(self):
        """Stress test: 4 worker threads evaluate benchmark conditions simultaneously using a shared runner."""
        dataset = load_cranfield_dataset()
        shared_embedder = DeterministicMockEmbedder()
        runner = BenchmarkRunner(embedder=shared_embedder)

        def worker_task(thread_id: int):
            condition = "enhanced" if thread_id % 2 == 0 else "baseline"
            return runner.evaluate_condition(dataset, condition=condition, k=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_task, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert len(results) == 4
        for r in results:
            assert r.performance.total_queries == 5
            assert len(r.per_query_results) == 5

        # Check that results for the same condition across threads are identical
        enhanced_results = [r for r in results if r.condition == "enhanced"]
        baseline_results = [r for r in results if r.condition == "baseline"]
        assert len(enhanced_results) == 2
        assert len(baseline_results) == 2

        assert enhanced_results[0].metrics.ndcg_at_10 == enhanced_results[1].metrics.ndcg_at_10
        assert baseline_results[0].metrics.ndcg_at_10 == baseline_results[1].metrics.ndcg_at_10

    def test_store_isolation_under_concurrency(self, tmp_path):
        """Verify distinct temporary stores do not collide or leak across concurrent evaluations."""
        dataset = load_cranfield_dataset()
        shared_embedder = DeterministicMockEmbedder()

        def run_isolated(worker_id: int):
            runner = BenchmarkRunner(embedder=shared_embedder)
            paired = runner.run_paired_evaluation(dataset, k=5)
            return worker_id, paired["comparisons"]["gain_ndcg_pct"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_isolated, i) for i in range(3)]
            results = dict([f.result() for f in futures])

        assert len(results) == 3
        # All workers must report identical results without store cross-talk
        gains = list(results.values())
        assert gains[0] == gains[1] == gains[2]



    def test_multiprocess_cli_concurrency(self, tmp_path):
        """Stress test: 2 independent OS processes execute evals/eval_benchmark.py concurrently."""
        out1 = tmp_path / "proc_out_1.json"
        out2 = tmp_path / "proc_out_2.json"

        cmd1 = [
            sys.executable,
            "evals/eval_benchmark.py",
            "--dataset",
            "cranfield",
            "--condition",
            "both",
            "--output",
            str(out1),
            "--k",
            "5",
        ]
        cmd2 = [
            sys.executable,
            "evals/eval_benchmark.py",
            "--dataset",
            "cranfield",
            "--condition",
            "both",
            "--output",
            str(out2),
            "--k",
            "5",
        ]

        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        out_text1, err_text1 = p1.communicate(timeout=60)
        out_text2, err_text2 = p2.communicate(timeout=60)

        assert p1.returncode == 0, f"Process 1 failed: {err_text1}"
        assert p2.returncode == 0, f"Process 2 failed: {err_text2}"

        assert out1.is_file()
        assert out2.is_file()

        with open(out1, "r") as f:
            data1 = json.load(f)
        with open(out2, "r") as f:
            data2 = json.load(f)

        m1 = data1["datasets"]["cranfield"]["enhanced"]["metrics"]
        m2 = data2["datasets"]["cranfield"]["enhanced"]["metrics"]
        assert m1["ndcg@10"] == m2["ndcg@10"]
        assert m1["mrr@10"] == m2["mrr@10"]


class TestBenchmarkDatasetScaling:
    """Stress-test hierarchical chunking, indexing, and retrieval across scaled corpus."""

    def test_dataset_scaling_hierarchical_chunking_large_corpus(self, scaling_dataset):
        """Verify that scaling the corpus to 60 docs (180 sections) runs efficiently without infinite loops."""
        runner = BenchmarkRunner()
        res = runner.evaluate_condition(scaling_dataset, condition="enhanced", k=5)

        assert res.condition == "enhanced"
        assert res.dataset_name == "scaled_benchmark"
        assert res.performance.total_queries == 5
        assert res.metrics.ndcg_at_10 >= 0.5  # SOTA retrieval captures relevant evidence
        assert res.metrics.mrr_at_10 > 0.0

    def test_extreme_query_and_qrel_boundaries(self):
        """Evaluate edge case: queries with disjoint ground truth or empty relevance."""
        empty_gt_query = BenchmarkQuery(
            id="q_empty",
            query="Nonexistent topic entirely absent from corpus",
            stepback_query="General nonexistent topic",
            ground_truth={},  # Empty ground truth
        )
        dataset = BenchmarkDataset(
            name="empty_gt",
            description="Empty Ground Truth Dataset",
            documents=[
                BenchmarkDocument(id="doc_1", title="Title 1", text="Sample text for testing.")
            ],
            queries=[empty_gt_query],
        )
        runner = BenchmarkRunner()
        res = runner.evaluate_condition(dataset, condition="baseline", k=5)
        assert res.metrics.ndcg_at_10 == 0.0
        assert res.metrics.recall_at_10 == 0.0
        assert res.metrics.mrr_at_10 == 0.0


class TestBenchmarkStatisticalReproducibility:
    """Empirically verify reproducibility of benchmark results across multiple sequential iterations."""

    def test_reproducibility_zero_variance_across_iterations(self):
        """Verify that multiple independent benchmark runs yield exact identical metrics (zero variance)."""
        dataset = load_cranfield_dataset()
        runner = BenchmarkRunner()

        ndcg_runs: List[float] = []
        mrr_runs: List[float] = []

        for _ in range(3):
            res = runner.evaluate_condition(dataset, condition="enhanced", k=10)
            ndcg_runs.append(res.metrics.ndcg_at_10)
            mrr_runs.append(res.metrics.mrr_at_10)

        # Variance must be 0.0 (deterministic reproducibility)
        assert np.var(ndcg_runs) == 0.0
        assert np.var(mrr_runs) == 0.0
        assert round(ndcg_runs[0], 4) == 0.9766
        assert round(mrr_runs[0], 4) == 1.0000
