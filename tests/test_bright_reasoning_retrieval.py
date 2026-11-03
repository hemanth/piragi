"""Tests for ThinkThenRetrieve, CausalLLMReranker, ListwiseReasoningReranker, and BRIGHT preset."""

from unittest.mock import MagicMock, patch

import pytest

from evals.leaderboard_exporters import optimize_for_leaderboard
from evals.run_bright_live import (
    bootstrap_ci_95,
    build_and_score_sparse_bm25,
    compute_bright_metrics,
    mask_excluded_scores,
    weighted_rrf_fuse,
)
from piragi import Ragi, RagiConfig
from piragi.config import ChunkConfig, RetrievalConfig
from piragi.pipelines.retrieval import RetrievalPipeline
from piragi.query_transform import ThinkThenRetrieve
from piragi.reranker import (
    CausalLLMReranker,
    ListwiseReasoningReranker,
    parse_listwise_ranking,
    sanitize_passage_brackets,
)
from piragi.telemetry import TelemetryCollector
from piragi.types import Answer, Citation


def _make_mock_llm_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestThinkThenRetrieve:
    """Unit tests for Think-Then-Retrieve reasoning query expansion."""

    def test_reason_generates_hypothesis_and_get_queries_returns_pair(self):
        llm = MagicMock()
        llm.model = "mock-reasoner"
        llm.client = MagicMock()
        llm.complete.return_value = _make_mock_llm_response(
            "Essential problem: Matrix A=[[5,4],[1,2]] maps x to lambda*x. "
            "Relevant theorem: Eigenvalue definition det(A - lambda*I) = 0."
        )

        ttr = ThinkThenRetrieve(llm_client=llm, domain="math")
        queries = ttr.get_queries("Find both possible values of lambda for box A.")

        assert len(queries) == 2
        assert queries[0] == "Find both possible values of lambda for box A."
        assert "Eigenvalue" in queries[1]
        llm.complete.assert_called_once()

    def test_reason_graceful_fallback_on_llm_failure(self):
        llm = MagicMock()
        llm.model = "mock-reasoner"
        llm.client = MagicMock()
        llm.complete.side_effect = RuntimeError("Connection refused")

        ttr = ThinkThenRetrieve(llm_client=llm)
        queries = ttr.get_queries("Why do insects gather around LED bulbs?")
        assert queries == ["Why do insects gather around LED bulbs?"]


class TestListwiseParsingAndSlidingWindow:
    """Unit tests for ReasonRank / RankGPT listwise parsing and sliding-window reranking."""

    def test_sanitize_passage_brackets_replaces_numeric_citations_only(self):
        raw = "Prior work [1] and [24] proved A[i] converges in [0, 1]."
        cleaned = sanitize_passage_brackets(raw)
        assert "[1]" not in cleaned
        assert "[24]" not in cleaned
        assert "(1)" in cleaned
        assert "(24)" in cleaned
        assert "A[i]" in cleaned

    def test_parse_listwise_ranking_handles_think_answer_tags_ties_and_missing(self):
        raw = (
            "<think>Passage 3 directly proves Hamming distance 2e+1 <= d. "
            "Passage 1 and 4 are partial; 99 is invalid.</think>\n"
            "<answer>[3] > [1] = [4] > [99]</answer>"
        )
        order = parse_listwise_ranking(raw, num_candidates=4)
        # 0-indexed: [2, 0, 3, 1] -> candidate 2 ([3]) first, missing [2] (idx 1) preserved at tail
        assert order == [2, 0, 3, 1]

    def test_sliding_window_bubbles_bottom_candidate_to_top(self):
        citations = [
            Citation(source="d{}.txt".format(i), chunk="irrelevant {}".format(i), score=0.9 - i * 0.1)
            for i in range(5)
        ]
        citations.append(
            Citation(source="gold.txt", chunk="Theorem: 2e + 1 <= d corrects e errors.", score=0.1)
        )

        # 6 items, window_size=4, step_size=2:
        # Window 1 covers indices [2..5] (4 items: d2, d3, d4, gold -> local [4] is gold)
        # Window 2 covers indices [0..3] (4 items: d0, d1, gold, d2 -> local [3] is gold)
        llm = MagicMock()
        llm.model = "mock-listwise"
        llm.client = MagicMock()
        llm.complete.side_effect = [
            _make_mock_llm_response("<think>4 is gold</think><answer>[4] > [1] > [2] > [3]</answer>"),
            _make_mock_llm_response("<think>3 is gold</think><answer>[3] > [1] > [2] > [4]</answer>"),
        ]

        telemetry = TelemetryCollector(enabled=True)
        reranker = ListwiseReasoningReranker(
            llm_client=llm,
            window_size=4,
            step_size=2,
            fusion_weight=0.0,
            telemetry=telemetry,
        )
        ranked = reranker.rerank("error correcting code bound", citations, top_k=3)

        assert len(ranked) == 3
        assert ranked[0].source == "gold.txt"
        assert ranked[0].score > ranked[1].score
        stats = telemetry.snapshot()
        assert stats["red"]["rerank"]["requests"] == 1


class TestCausalLLMReranker:
    """Unit tests for instruction-aware Qwen3 / Gemma Yes-No logit reranker."""

    def test_format_qwen3_instruction_and_rerank_with_fusion(self):
        citations = [
            Citation(source="wrong.txt", chunk="Bubble sort takes O(n^2).", score=0.95, metadata={"k": 1}),
            Citation(source="right.txt", chunk="Monotonic stack solves next greater element in O(n).", score=0.40, metadata={"k": 2}),
        ]
        telemetry = TelemetryCollector(enabled=True)
        reranker = CausalLLMReranker(
            model_name="Qwen/Qwen3-Reranker-4B",
            instruction="Given a coding problem, retrieve relevant algorithmic techniques.",
            fusion_weight=0.15,
            telemetry=telemetry,
        )
        prompt_str = reranker.format_qwen3_pair("Find next greater element", "Monotonic stack")
        assert "<Instruct>:" in prompt_str
        assert "<Query>: Find next greater element" in prompt_str
        assert "<Document>: Monotonic stack" in prompt_str

        with patch.object(reranker, "_score_pairs", return_value=[0.05, 0.98]):
            ranked = reranker.rerank("Find next greater element", citations, top_k=2)

        assert ranked[0].source == "right.txt"
        assert ranked[0].metadata == {"k": 2}
        assert ranked[0].score > ranked[1].score
        assert telemetry.snapshot()["red"]["rerank"]["requests"] == 1


class TestPipelineAndConfigIntegration:
    """Verify RetrievalConfig, Ragi, RetrievalPipeline, and BRIGHT leaderboard preset."""

    def test_retrieval_config_defaults_and_bright_preset(self):
        cfg = RetrievalConfig()
        assert cfg.use_think_then_retrieve is False
        assert cfg.reranker_type == "cross_encoder"

        preset = optimize_for_leaderboard("bright")
        assert preset["benchmark"] == "bright"
        validated_ret = RetrievalConfig.model_validate(preset["retrieval"])
        validated_chunk = ChunkConfig.model_validate(preset["chunk"])
        assert validated_ret.use_think_then_retrieve is True
        assert validated_ret.use_hybrid_search is True
        assert validated_ret.reranker_type == "listwise"
        assert validated_chunk.strategy == "hierarchical"

    def test_ragi_wires_think_then_retrieve_and_listwise_reranker(self):
        with patch("piragi.core.create_store"):
            kb = Ragi(
                config={
                    "retrieval": {
                        "use_think_then_retrieve": True,
                        "use_cross_encoder": True,
                        "reranker_type": "listwise",
                        "listwise_window_size": 10,
                        "listwise_step_size": 5,
                    }
                }
            )
            assert kb._retrieval_pipeline.think_then_retrieve is not None
            assert isinstance(kb._retrieval_pipeline.think_then_retrieve, ThinkThenRetrieve)
            assert isinstance(kb._retrieval_pipeline.cross_encoder, ListwiseReasoningReranker)

    def test_pipeline_retrieve_fuses_think_then_retrieve_and_enriches_hybrid_search(self):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]

        c_vec = Citation(source="vec.txt", chunk="vector hit", score=0.8)
        c_reason = Citation(source="reason.txt", chunk="theorem hit", score=0.9)
        mock_store = MagicMock()
        mock_store.search.side_effect = [[c_vec], [c_reason]]
        del mock_store.get_all_citations

        mock_ttr = MagicMock()
        mock_ttr.get_queries.return_value = [
            "puzzling box matrix question",
            "eigenvalue linear operator characteristic polynomial",
        ]

        mock_hybrid = MagicMock()
        mock_hybrid.search.side_effect = lambda query, vector_citations, top_k: vector_citations

        pipeline = RetrievalPipeline(
            embedder=mock_embedder,
            store=mock_store,
            retriever=MagicMock(),
            think_then_retrieve=mock_ttr,
            hybrid_searcher=mock_hybrid,
        )

        results = pipeline.retrieve("puzzling box matrix question", top_k=5)
        assert mock_store.search.call_count == 2
        assert {r.source for r in results} == {"vec.txt", "reason.txt"}
        # Hybrid lexical search receives reasoning-enriched query terms
        hybrid_query_used = mock_hybrid.search.call_args[1]["query"]
        assert "eigenvalue" in hybrid_query_used


class TestBrightEvalContract:
    """Unit tests for BRIGHT excluded_ids masking, TREC metrics, RRF, and bootstrap CIs."""

    def test_mask_excluded_scores_strictly_strips_excluded_and_na(self):
        raw = {
            "q1": {"leak_doc": 0.99, "N/A": 0.95, "gold_1": 0.85, "other": 0.40},
        }
        masked = mask_excluded_scores(raw, {"q1": ["leak_doc"]}, top_k=10)
        assert "leak_doc" not in masked["q1"]
        assert "N/A" not in masked["q1"]
        assert list(masked["q1"].keys()) == ["gold_1", "other"]

    def test_compute_bright_metrics_and_bootstrap_ci(self):
        scores = {
            "q1": {"d1": 0.9, "d2": 0.5},
            "q2": {"d9": 0.8, "d3": 0.7},
        }
        qrels = {
            "q1": {"d1": 1},
            "q2": {"d3": 1},
        }
        metrics, per_q = compute_bright_metrics(scores, qrels)
        # q1 hits rank 1 (NDCG=1.0, RR=1.0); q2 hits rank 2 (NDCG=1/log2(3)=0.63093, RR=0.5)
        assert metrics["NDCG@10"] == pytest.approx((1.0 + 0.63092975) / 2.0, abs=1e-4)
        assert metrics["MRR"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["Recall@10"] == pytest.approx(1.0, abs=1e-4)
        low, high = bootstrap_ci_95(list(per_q.values()), n_boot=500, seed=42)
        assert 0.60 <= low <= high <= 1.0

    def test_weighted_rrf_fuse_prioritizes_higher_weighted_run(self):
        r1 = {"q1": [("docA", 0.9), ("docB", 0.8)]}
        r2 = {"q1": [("docB", 0.9), ("docA", 0.8)]}
        fused = weighted_rrf_fuse([(r1, 2.5), (r2, 1.0)], ["q1"], top_k=2)
        assert fused["q1"][0][0] == "docA"

    def test_build_and_score_sparse_bm25_masks_excluded_and_ranks_relevant_top(self):
        docs = [
            "monotonic stack next greater element linear time",
            "dijkstra shortest path priority queue graph",
            "eigenvalue eigenvector characteristic polynomial matrix",
        ]
        doc_ids = ["d_stack", "d_graph", "d_eigen"]
        res = build_and_score_sparse_bm25(
            norm_docs=docs,
            doc_ids=doc_ids,
            query_batches=[("raw", ["q1"], ["eigenvalue characteristic matrix"], {"q1": ["d_graph"]})],
            top_k=3,
        )
        ranked = res["raw"]["q1"]
        assert ranked[0][0] == "d_eigen"
        assert all(did != "d_graph" for did, _ in ranked)
