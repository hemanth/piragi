"""Check evaluation arithmetic independently of model predictions."""

import importlib.util
import json
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "evals"
spec = importlib.util.spec_from_file_location("eval_openjev", EVAL_DIR / "eval_openjev.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_recall_uses_unique_relevant_sources():
    metrics = evaluation.retrieval_metrics(["wrong", "a", "a"], ["a", "b"])
    assert metrics == {"recall": 0.5, "reciprocal_rank": 0.5}
    assert evaluation.retrieval_metrics(["wrong"], ["a"])["recall"] == 0


def test_summary_pairs_conditions_and_counts_actual_calls():
    rows = []
    for query_id, times in [("q1", (4, 1)), ("q2", (2, 3))]:
        for condition, seconds in zip(("without_openjev", "with_openjev"), times):
            rows.append(
                {
                    "query_id": query_id,
                    "repeat": 0,
                    "condition": condition,
                    "seconds": seconds,
                    "recall": 0.5,
                    "reciprocal_rank": 1,
                    "expansion_calls": int(condition == "without_openjev"),
                    "answer_calls": 1,
                    "answer_truncated": False,
                    "decision_seconds": 0.1,
                }
            )
    summary = evaluation.summarize(rows)
    assert summary["without_openjev"]["p50_seconds"] == 3
    assert summary["with_openjev"]["p50_seconds"] == 2
    assert summary["paired_mean_latency_delta_seconds"] == -1
    assert summary["with_openjev"]["expansion_calls"] == 0
    assert summary["without_openjev"]["expansion_calls"] == 2


def test_fixture_has_valid_independent_source_labels():
    fixture = json.loads((EVAL_DIR / "openjev_cases.json").read_text())
    ids = [query["id"] for query in fixture["queries"]]
    assert len(ids) == len(set(ids))
    for query in fixture["queries"]:
        assert query["relevant_sources"] or query["kind"] == "unanswerable"
        assert set(query["relevant_sources"]) <= fixture["documents"].keys()


def test_engine_failure_is_recorded_even_if_piragi_catches_it():
    class BrokenEngine:
        def decide(self, **kwargs):
            raise RuntimeError("No model")

    trace = evaluation.TracedEngine(BrokenEngine())
    with pytest.raises(RuntimeError):
        trace.decide(state="question")
    assert trace.errors == ["RuntimeError"]
    assert trace.results == []


def test_unanswerable_recall_is_not_divided_by_zero():
    assert evaluation.retrieval_metrics(["adjacent"], []) == {"recall": None, "reciprocal_rank": 0}


def test_incomplete_pairs_do_not_pollute_summary():
    assert evaluation.summarize(
        [{"query_id": "q1", "repeat": 0, "condition": "without_openjev"}]
    ) == {"completed_pairs": 0, "unpaired_rows": 1}
