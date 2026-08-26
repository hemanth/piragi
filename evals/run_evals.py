"""Eval harness comparing piragi retrieval modes on a small keyword-heavy corpus.

Modeled on the "RAG is simpler than you think" recipes: measures retrieval
hit-rate@5 and per-query latency for dense, hybrid, bm25_only (+ query
rewrite), on_the_fly, and hot_cold modes on the same corpus/queries.

Usage:
    python evals/run_evals.py
"""

import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# transformers' AutoTokenizer.from_pretrained calls huggingface_hub.model_info()
# unconditionally (even for a fully-cached local model) to check for a mistral
# regex patch. On networks that block/intercept huggingface.co this hangs or
# SSL-fails every Ragi() construction. Cached tokenizer files are already on
# disk, so short-circuit that one lookup for this eval run.
import huggingface_hub
huggingface_hub.model_info = lambda repo_id, *a, **k: SimpleNamespace(id=repo_id, tags=[])

from piragi import Ragi

EVAL_DIR = os.path.dirname(__file__)
CORPUS = os.path.join(EVAL_DIR, "corpus", "*.md")
QUERIES = json.load(open(os.path.join(EVAL_DIR, "queries.json")))

LLM_CONFIG = {"model": "gemma4:latest"}
HOT_SOURCES = {"auth.md", "billing.md"}


def tag_tier(chunks):
    for chunk in chunks:
        basename = os.path.basename(chunk.source)
        chunk.metadata["tier"] = "hot" if basename in HOT_SOURCES else "cold"
    return chunks


MODES = [
    {"name": "dense (baseline)", "config": {"llm": LLM_CONFIG}},
    {"name": "hybrid (bm25+vector)", "config": {"llm": LLM_CONFIG, "retrieval": {"use_hybrid_search": True}}},
    {"name": "bm25_only", "config": {"llm": LLM_CONFIG, "retrieval": {"mode": "bm25_only"}}},
    {"name": "bm25_only + query_rewrite", "config": {"llm": LLM_CONFIG, "retrieval": {"mode": "bm25_only", "use_query_rewrite": True}}},
    {"name": "on_the_fly", "config": {"llm": LLM_CONFIG, "retrieval": {"mode": "on_the_fly"}}},
    {"name": "hot_cold", "config": {"llm": LLM_CONFIG, "retrieval": {"mode": "hot_cold"}}, "hooks": {"post_chunk": tag_tier}},
]


def run_mode(mode):
    persist_dir = os.path.join(EVAL_DIR, f".piragi_eval_{mode['name'].split()[0]}")
    shutil.rmtree(persist_dir, ignore_errors=True)

    kwargs = {
        "config": {**mode["config"], "auto_update": {"enabled": False}},
        "persist_dir": persist_dir,
    }
    if "hooks" in mode:
        kwargs["hooks"] = mode["hooks"]

    kb = Ragi(**kwargs)

    t0 = time.perf_counter()
    kb.add(CORPUS)
    ingest_s = time.perf_counter() - t0

    hits = 0
    latencies = []
    for q in QUERIES:
        t0 = time.perf_counter()
        citations = kb.retrieve(q["query"], top_k=5)
        latencies.append(time.perf_counter() - t0)
        sources = {os.path.basename(c.source) for c in citations}
        if q["expected_source"] in sources:
            hits += 1

    shutil.rmtree(persist_dir, ignore_errors=True)

    return {
        "mode": mode["name"],
        "hit_rate": hits / len(QUERIES),
        "ingest_s": ingest_s,
        "avg_latency_ms": 1000 * sum(latencies) / len(latencies),
        "p95_latency_ms": 1000 * sorted(latencies)[int(0.95 * len(latencies)) - 1],
    }


def main():
    results = [run_mode(mode) for mode in MODES]

    print(f"\n{len(QUERIES)} queries over {len(list(__import__('glob').glob(CORPUS)))} docs\n")
    header = f"{'mode':<28} {'hit@5':>8} {'ingest_s':>10} {'avg_ms':>9} {'p95_ms':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['mode']:<28} {r['hit_rate']:>8.0%} {r['ingest_s']:>10.2f} {r['avg_latency_ms']:>9.1f} {r['p95_latency_ms']:>9.1f}")


if __name__ == "__main__":
    main()
