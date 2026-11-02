"""Paired real-model evaluation: python evals/eval_openjev.py --output /tmp/jev-eval.

Requires SemIf, its checkpoint, and a running local Ollama model. Never substitutes
mock model outputs. Raw answers are retained for human correctness assessment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import statistics
import time
from pathlib import Path


def retrieval_metrics(sources, relevant):
    relevant = set(relevant)
    found = set(sources) & relevant
    reciprocal_rank = next(
        (1 / rank for rank, source in enumerate(sources, 1) if source in relevant), 0.0
    )
    return {
        "recall": len(found) / len(relevant) if relevant else None,
        "reciprocal_rank": reciprocal_rank,
    }


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault((row["query_id"], row["repeat"]), {})[row["condition"]] = row
    complete = [pair for pair in pairs.values() if len(pair) == 2]
    paired_rows = [row for pair in complete for row in pair.values()]
    result = {"completed_pairs": len(complete), "unpaired_rows": len(rows) - len(paired_rows)}
    if not complete:
        return result
    rows = paired_rows
    for condition in ("without_openjev", "with_openjev"):
        group = [row for row in rows if row["condition"] == condition]
        latencies = [row["seconds"] for row in group]
        result[condition] = {
            "runs": len(group),
            "mean_precision_at_k": statistics.mean(row.get("precision", 0) for row in group),
            "abstentions": sum(row.get("abstained", False) for row in group),
            "false_abstentions": sum(
                row.get("abstained", False) and row.get("answerable", True) for row in group
            ),
            "unanswerable_generations": sum(
                row.get("answer_calls", 0) > 0 and not row.get("answerable", True) for row in group
            ),
            "mean_context_chars": statistics.mean(row.get("context_chars", 0) for row in group),
            "mean_recall_at_k": statistics.mean(
                row["recall"] for row in group if row["recall"] is not None
            )
            if any(row["recall"] is not None for row in group)
            else None,
            "mean_reciprocal_rank": statistics.mean(row["reciprocal_rank"] for row in group),
            "mean_seconds": statistics.mean(latencies),
            "p50_seconds": percentile(latencies, 0.5),
            "p95_seconds": percentile(latencies, 0.95),
            "expansion_calls": sum(row["expansion_calls"] for row in group),
            "answer_calls": sum(row["answer_calls"] for row in group),
            "truncated_answers": sum(row["answer_truncated"] for row in group),
            "mean_decision_seconds": statistics.mean(row["decision_seconds"] for row in group),
        }
    pairs = {}
    for row in rows:
        pairs.setdefault((row["query_id"], row["repeat"]), {})[row["condition"]] = row
    deltas = [
        pair["with_openjev"]["seconds"] - pair["without_openjev"]["seconds"]
        for pair in pairs.values()
    ]
    result["paired_mean_latency_delta_seconds"] = statistics.mean(deltas)
    result["scope"] = (
        "Authored development fixture; warm serial calls; mean recall uses document labels. "
        "Answer correctness is not automatically scored. Repeats are not independent queries. "
        "Both conditions use the same frozen candidate pool. No automatic answer-quality grade. "
        "Do not interpret this small fixture as a production accuracy benchmark."
    )
    return result


class TracedClient:
    def __init__(self, client, seed):
        self.delegate = client
        self.seed = seed
        self.phase = "answer"
        self.calls = []
        self.errors = []

    def complete(self, *args, **kwargs):
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("max_tokens", 240)
        started = time.perf_counter()
        try:
            response = self.delegate.complete(*args, **kwargs)
        except Exception as exc:
            self.errors.append(type(exc).__name__)
            raise
        self.calls.append(
            {
                "phase": self.phase,
                "seconds": time.perf_counter() - started,
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
        return response


class TracedEngine:
    def __init__(self, engine):
        self.delegate = engine
        self.results = []
        self.errors = []

    def _call(self, method, kwargs):
        started = time.perf_counter()
        try:
            result = getattr(self.delegate, method)(**kwargs)
        except Exception as exc:
            self.errors.append(type(exc).__name__)
            raise
        values = result if method == "decide_many" else {"answerability": result}
        self.results.append(
            {
                "method": method,
                "elapsed_seconds": time.perf_counter() - started,
                "judgments": {key: value.model_dump() for key, value in values.items()},
            }
        )
        return result

    def decide(self, **kwargs):
        return self._call("decide", kwargs)

    def decide_many(self, **kwargs):
        return self._call("decide_many", kwargs)


def run(args):
    if importlib.util.find_spec("semif_phase1") is None:
        raise RuntimeError("SemIf is not installed. A real with/without evaluation cannot run.")

    from piragi import Ragi
    from piragi.evidence import NO_EVIDENCE, EvidenceJudge
    from piragi.types import Answer

    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    corpus = output / "corpus"
    corpus.mkdir()
    for name, text in fixture["documents"].items():
        if Path(name).name != name:
            raise ValueError("Fixture document names must be basenames")
        (corpus / name).write_text(text)

    decision_config = {"engine": "openjev", "candidate_k": args.candidate_k}
    if args.model_path:
        decision_config["model_path"] = str(args.model_path)
    if args.revision:
        decision_config["revision"] = args.revision
    if args.backend:
        decision_config["backend"] = args.backend
    config = {
        "decision": decision_config,
        "llm": {
            "model": args.answer_model,
            "base_url": args.base_url,
            "enable_query_expansion": False,
            "enable_reranking": False,
        },
        "retrieval": {"use_hybrid_search": True},
        "embedding": {"model": "all-MiniLM-L6-v2", "device": "cpu"},
        "auto_update": {"enabled": False},
    }
    started = time.perf_counter()
    kb = Ragi(str(corpus), persist_dir=str(output / "index"), config=config)
    setup_seconds = time.perf_counter() - started
    client = TracedClient(kb.llm_client, args.seed)
    engine = TracedEngine(kb.decision_engine)
    kb.retriever.llm_client = client
    judge = EvidenceJudge(engine, kb.config.decision.min_score, kb.config.decision.min_margin)
    frozen = {}

    def measure(query, condition):
        # Retrieve exactly once per unique query. Store the evidence pool for replay.
        if query["id"] not in frozen:
            start = time.perf_counter()
            candidates = kb.retrieve(query["query"], top_k=max(args.candidate_k, args.top_k))
            frozen[query["id"]] = (candidates, time.perf_counter() - start)
        candidates, retrieval_seconds = frozen[query["id"]]
        client.calls.clear()
        client.errors.clear()
        engine.results.clear()
        engine.errors.clear()
        start = time.perf_counter()
        report = None
        selected = candidates[: args.top_k]
        abstained = not selected
        if condition == "with_openjev":
            selection = judge.select(query["query"], candidates, args.top_k)
            report, selected, abstained = selection.report, selection.citations, selection.abstained
            if report["status"] == "unavailable":
                raise RuntimeError("Judge failed; refusing to score fallback as OpenJev")
            if candidates and not any(r["method"] == "decide_many" for r in engine.results):
                raise RuntimeError("Expected a real batch of passage judgments")
        if abstained:
            answer = Answer(text=NO_EVIDENCE, query=query["query"])
        elif args.skip_generation:
            answer = Answer(text="", query=query["query"], citations=selected)
        else:
            answer = kb.retriever.generate_answer(query["query"], selected)
        stage_seconds = time.perf_counter() - start
        if engine.errors or client.errors:
            raise RuntimeError("Inference failed; refusing to score fallback as a valid eval run")
        sources = [Path(c.source).name for c in answer.citations]
        relevant = set(query["relevant_sources"])
        return {
            "query_id": query["id"],
            "query": query["query"],
            "kind": query["kind"],
            "condition": condition,
            "seconds": retrieval_seconds + stage_seconds,
            "retrieval_seconds": retrieval_seconds,
            "stage_seconds": stage_seconds,
            **retrieval_metrics(sources, relevant),
            "precision": sum(source in relevant for source in sources) / len(sources)
            if sources
            else 0,
            "candidate_recall": retrieval_metrics(
                [Path(c.source).name for c in candidates], relevant
            )["recall"],
            "answerable": bool(relevant),
            "abstained": abstained,
            "expected_sources": sorted(relevant),
            "sources": sources,
            "answer": answer.text,
            "citations": [c.model_dump() for c in answer.citations],
            "candidates": [c.model_dump() for c in candidates],
            "context_chars": sum(len(c.chunk) for c in answer.citations),
            "expansion_calls": 0,
            "answer_calls": len(client.calls),
            "answer_truncated": any(c["finish_reason"] == "length" for c in client.calls),
            "decision_seconds": sum(d["elapsed_seconds"] for d in engine.results),
            "decisions": list(engine.results),
            "decision_report": report,
            "llm_calls": list(client.calls),
        }

    # A separate warm-up question; neither warm-up observation enters the summary.
    warmup = {
        "id": "warmup",
        "kind": "warmup",
        "query": "How does Piragi answer questions?",
        "relevant_sources": ["citations.md"],
    }
    warmups = [measure(warmup, condition) for condition in ("without_openjev", "with_openjev")]
    versions = {}
    for package in ("piragi", "semif-phase1", "torch", "transformers", "mlx", "mlx-lm", "lancedb"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "fixture_description": fixture["description"],
        "config": kb.config.model_dump(),
        "versions": versions,
        "platform": platform.platform(),
        "seed": args.seed,
        "top_k": args.top_k,
        "repeats": args.repeats,
        "setup_seconds": setup_seconds,
        "chunks": kb.count(),
        "warmups": warmups,
        "decision_model": next(iter(warmups[1]["decisions"][0]["judgments"].values()))["model"],
        "skip_generation": args.skip_generation,
        "strategy": "frozen hybrid candidates -> batch relevance -> retained-context answerability",
        "timing": (
            "retrieval measured once per query and added equally to both stages; "
            "warm decision and generation measured per run"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    queries = list(fixture["queries"])
    random.Random(args.seed).shuffle(queries)
    rows = []
    with (output / "runs.jsonl").open("w") as stream:
        for repeat in range(args.repeats):
            for index, query in enumerate(queries):
                order = ["without_openjev", "with_openjev"]
                if (index + repeat) % 2:
                    order.reverse()
                for condition in order:
                    row = {**measure(query, condition), "repeat": repeat}
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    (output / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
                    print(
                        f"{len(rows)}/{2 * args.repeats * len(queries)} "
                        f"{query['id']} {condition}: {row['seconds']:.2f}s "
                        f"recall={row['recall']} abstained={row['abstained']}",
                        flush=True,
                    )
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=Path(__file__).with_name("openjev_cases.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answer-model", default="llama3.2")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--revision", help="Revision label for a supplied checkpoint")
    parser.add_argument("--backend", choices=["auto", "torch", "mlx"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Measure real evidence decisions without an answer model",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.top_k < 1 or args.repeats < 1 or args.candidate_k < 1:
        parser.error("top-k and repeats must be positive")
    run(args)


if __name__ == "__main__":
    main()
