"""Official BRIGHT (ICLR 2025) 12-Split Reasoning-Intensive Retrieval GPU Runner for piragi.

Evaluates piragi's SOTA Reasoning-Retrieval Pipeline across all 12 official BRIGHT splits:
  - Stage 0 (Query Reasoning):
      * Open-Weights Local Track: piragi ThinkThenRetrieve (Qwen/Qwen2.5-7B-Instruct CoT synthesis)
      * Standardized Benchmark Track: Official BRIGHT CoT queries (xlangai/BRIGHT gpt4_reason)
  - Stage 1 (Multi-Space Reasoning Hybrid RRF + Consensus Rocchio PRF):
      * Qwen/Qwen3-Embedding-0.6B (instruction-aware full-corpus dense space)
      * BAAI/bge-large-en-v1.5 (high-precision lexical-semantic dense space)
      * Piragi De-glued BM25 (normalize_corpus_texts + raw & reasoned lexical fusion)
      * BAAI/bge-reasoner-embed-qwen3-8b-0923 (8B reasoning retriever + Rocchio PRF)
  - Stage 2 (Instruction-Aware Pointwise Causal LLM Reranking, Top-100):
      * Qwen/Qwen3-Reranker-4B (task-specific <Instruct>/<Query>/<Document> Yes/No logits)
      * BAAI/bge-reranker-v2-gemma (causal Yes/No logit reranker)
  - Stage 3 (Optional Listwise Comparative Reasoning Reranker, Top-20):
      * liuwenhan/reasonrank-7B (<think>...</think><answer>[i] > [j]</answer> listwise Rank-RRF)

Strictly enforces BRIGHT excluded_ids masking, exports official score.json & results.json per split,
and computes B=1,000 bootstrap 95% confidence intervals across all 1,384 queries.
"""

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from piragi import get_telemetry, request_context
from piragi.hybrid_search import normalize_corpus_texts
from piragi.reranker import parse_listwise_ranking, sanitize_passage_brackets

BRIGHT_TASKS: List[str] = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]

TASK_DOMAIN_LABELS: Dict[str, str] = {
    "biology": "Biology",
    "earth_science": "Earth Science",
    "economics": "Economics",
    "psychology": "Psychology",
    "robotics": "Robotics",
    "stackoverflow": "Stack Overflow software engineering",
    "sustainable_living": "Sustainable Living",
    "pony": "Pony programming language",
    "leetcode": "competitive programming and algorithms",
    "aops": "competition mathematics",
    "theoremqa_theorems": "mathematical theorems and proofs",
    "theoremqa_questions": "mathematical problem examples",
}

TASK_INSTRUCTIONS: Dict[str, str] = {
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "psychology": "Given a Psychology post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
    "stackoverflow": "Given a Stack Overflow post, retrieve relevant passages that help answer the post.",
    "sustainable_living": "Given a Sustainable Living post, retrieve relevant passages that help answer the post.",
    "leetcode": "Given a Coding problem, retrieve relevant examples that help answer the problem.",
    "pony": "Given a Pony question, retrieve relevant passages that help answer the question.",
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
}


def purge_hf_repo_cache(repo_id: str) -> None:
    """Remove a single HF hub model cache directory after its stage checkpoints complete."""
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hub_dir = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    folder_name = "models--{}".format(repo_id.replace("/", "--"))
    target = hub_dir / folder_name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        print("  [Disk] Purged HF cache {} to reclaim disk space".format(target), flush=True)


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preferred_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def free_memory(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def corpus_fingerprint(doc_ids: Sequence[str], docs: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(str(len(doc_ids)).encode("utf-8"))
    for idx in (0, len(doc_ids) // 2, len(doc_ids) - 1):
        if 0 <= idx < len(doc_ids):
            h.update(doc_ids[idx].encode("utf-8"))
            h.update(docs[idx][:256].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def mask_excluded_scores(
    scores_by_qid: Dict[str, Dict[str, float]],
    excluded_by_qid: Dict[str, Sequence[str]],
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Strictly remove BRIGHT excluded_ids per query before selecting top_k."""
    cleaned: Dict[str, Dict[str, float]] = {}
    for qid, doc_scores in scores_by_qid.items():
        excl = set(excluded_by_qid.get(qid, ()))
        filtered = [(did, float(sc)) for did, sc in doc_scores.items() if did not in excl and did != "N/A"]
        filtered.sort(key=lambda kv: kv[1], reverse=True)
        cleaned[qid] = {did: sc for did, sc in filtered[:top_k]}
    return cleaned


def compute_bright_metrics(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Tuple[int, ...] = (1, 5, 10, 25, 50, 100),
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute exact TREC metrics matching xlang-ai/BRIGHT retrievers.py calculate_retrieval_metrics.

    Returns:
        (summary_metrics_pct, per_query_ndcg10_dict) where summary metrics match BRIGHT's
        0..100 percentage scale (and raw 0..1 keys) plus per-query NDCG@10 for bootstrap CIs.
    """
    ndcg_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    map_sums = {k: 0.0 for k in k_values}
    prec_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    per_query_ndcg10: Dict[str, float] = {}

    qids = [qid for qid, rels in qrels.items() if rels]
    max_k = max(k_values)

    for qid in qids:
        rel_docs = qrels[qid]
        ranked = sorted(results.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))[:max_k]
        ranked_ids = [did for did, _ in ranked]
        ideal_rels = sorted([r for r in rel_docs.values() if r > 0], reverse=True)
        total_rel = float(len(ideal_rels))

        rr = 0.0
        for rank_idx, did in enumerate(ranked_ids, start=1):
            if rel_docs.get(did, 0) > 0:
                rr = 1.0 / float(rank_idx)
                break
        mrr_sum += rr

        for k in k_values:
            top_ids = ranked_ids[:k]
            hits = 0
            ap_accum = 0.0
            dcg = 0.0
            for rank_idx, did in enumerate(top_ids, start=1):
                rel = rel_docs.get(did, 0)
                if rel > 0:
                    hits += 1
                    ap_accum += float(hits) / float(rank_idx)
                    dcg += float((1 << rel) - 1) / math.log2(rank_idx + 1.0)

            idcg = 0.0
            for rank_idx, rel in enumerate(ideal_rels[:k], start=1):
                idcg += float((1 << rel) - 1) / math.log2(rank_idx + 1.0)

            ndcg_q = (dcg / idcg) if idcg > 0 else 0.0
            ndcg_sums[k] += ndcg_q
            if k == 10:
                per_query_ndcg10[qid] = ndcg_q
            recall_sums[k] += float(hits) / total_rel if total_rel > 0 else 0.0
            prec_sums[k] += float(hits) / float(k)
            map_sums[k] += ap_accum / min(total_rel, float(k)) if total_rel > 0 else 0.0

    denom = float(max(1, len(qids)))
    out: Dict[str, float] = {}
    for k in k_values:
        out["NDCG@{}".format(k)] = round(ndcg_sums[k] / denom, 5)
        out["MAP@{}".format(k)] = round(map_sums[k] / denom, 5)
        out["Recall@{}".format(k)] = round(recall_sums[k] / denom, 5)
        out["P@{}".format(k)] = round(prec_sums[k] / denom, 5)
    out["MRR"] = round(mrr_sum / denom, 5)
    return out, per_query_ndcg10


def bootstrap_ci_95(values: Sequence[float], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Compute B=1,000 non-parametric bootstrap 95% confidence interval."""
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (round(float(low), 5), round(float(high), 5))


def build_and_score_sparse_bm25(
    norm_docs: Sequence[str],
    doc_ids: Sequence[str],
    query_batches: List[Tuple[str, Sequence[str], Sequence[str], Dict[str, Sequence[str]]]],
    k1: float = 1.5,
    b: float = 0.75,
    k3: float = 1.2,
    top_k: int = 1000,
) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Ultra-fast C/CSR Sparse Matrix Okapi BM25+BM25L scorer (~10,000x faster than Python BM25Okapi).

    query_batches entries: (batch_key, qids, query_texts, excluded_by_qid)
    Returns: {batch_key: {qid: [(doc_id, score), ...]}}
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b",
        dtype=np.float32,
    )
    x_csr = vec.fit_transform(norm_docs).tocsr()
    n_docs, n_vocab = x_csr.shape
    dl = np.asarray(x_csr.sum(axis=1), dtype=np.float32).ravel()
    avgdl = max(1.0, float(dl.mean())) if n_docs > 0 else 1.0
    df = np.bincount(x_csr.indices, minlength=n_vocab).astype(np.float32)
    idf = np.log1p((float(n_docs) - df + 0.5) / (df + 0.5)).astype(np.float32)

    row_lengths = np.diff(x_csr.indptr)
    row_indices = np.repeat(np.arange(n_docs, dtype=np.int32), row_lengths)
    tf = x_csr.data
    norm_factor = k1 * (1.0 - b + b * (dl[row_indices] / avgdl))
    x_csr.data = (idf[x_csr.indices] * (tf * (k1 + 1.0)) / np.maximum(1e-8, tf + norm_factor)).astype(np.float32)
    x_csc_t = x_csr.T.tocsr()

    batch_outputs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    k_pull = min(n_docs, top_k + 64)

    for batch_key, qids, query_texts, excluded_map in query_batches:
        q_csr = vec.transform(query_texts).tocsr()
        if q_csr.nnz > 0:
            q_csr.data = (((k3 + 1.0) * q_csr.data) / (k3 + q_csr.data)).astype(np.float32)
        scores_mat = (q_csr @ x_csc_t).toarray()
        run_map: Dict[str, List[Tuple[str, float]]] = {}
        for qi, qid in enumerate(qids):
            row = scores_mat[qi]
            if k_pull < len(row):
                part = np.argpartition(-row, k_pull - 1)[:k_pull]
                part = part[np.argsort(-row[part])]
            else:
                part = np.argsort(-row)
            excl = set(excluded_map.get(qid, ()))
            ranked: List[Tuple[str, float]] = []
            for idx in part:
                sc_val = float(row[int(idx)])
                if sc_val <= 0.0 and ranked:
                    break
                did = doc_ids[int(idx)]
                if did not in excl and did != "N/A":
                    ranked.append((did, sc_val))
                    if len(ranked) >= top_k:
                        break
            run_map[qid] = ranked
        batch_outputs[batch_key] = run_map

    return batch_outputs


def generate_local_think_then_retrieve(
    queries_by_task: Dict[str, Tuple[List[str], List[str]]],
    model_name: str,
    device: str,
    cache_path: Path,
    batch_size: int = 12,
    max_new_tokens: int = 256,
) -> Dict[str, Dict[str, str]]:
    """Run piragi ThinkThenRetrieve prompt in batched GPU inference across all 12 BRIGHT tasks."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if all(t in cached for t in queries_by_task):
            print("[Stage-0] Loaded cached local ThinkThenRetrieve traces from {}".format(cache_path), flush=True)
            return cached

    print("[Stage-0] Generating local piragi ThinkThenRetrieve traces with {}...".format(model_name), flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = preferred_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    out: Dict[str, Dict[str, str]] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as handle:
            out.update(json.load(handle))

    sys_msg = (
        "You are an expert retrieval reasoner. Identify the essential problem, theorem, algorithm, "
        "or scientific mechanism, reason step by step about what authoritative reference passages "
        "answer it, and draft a concise technical hypothesis naming exact concepts and terminology."
    )

    for task, (qids, raw_queries) in queries_by_task.items():
        if task in out and len(out[task]) == len(qids):
            continue
        t0 = time.perf_counter()
        domain_label = TASK_DOMAIN_LABELS.get(task, task)
        prompts: List[str] = []
        for q in raw_queries:
            user_msg = (
                "{}\n\n"
                "Instructions ({}):\n"
                "1. Identify the essential underlying problem, theorem, or mechanism.\n"
                "2. Think step by step to reason and describe what information, definitions, "
                "or proofs could be relevant and helpful to address the question in detail.\n"
                "3. Draft a concise technical answer naming exact domain concepts and terminology."
            ).format(q.replace("\n", " ").strip(), domain_label)
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ]
            try:
                p_txt = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                p_txt = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompts.append(p_txt)

        task_map: Dict[str, str] = {}
        order = sorted(range(len(prompts)), key=lambda idx: len(prompts[idx]))
        cursor = 0
        cur_bs = batch_size
        while cursor < len(order):
            batch_indices = order[cursor : cursor + cur_bs]
            batch_prompts = [prompts[i] for i in batch_indices]
            try:
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1536,
                ).to(device)
                with torch.inference_mode():
                    gen = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                in_len = enc["input_ids"].shape[1]
                for local_i, orig_i in enumerate(batch_indices):
                    txt = tokenizer.decode(gen[local_i, in_len:], skip_special_tokens=True).strip()
                    if "</think>" in txt:
                        txt = txt.split("</think>", 1)[1].strip()
                    task_map[qids[orig_i]] = txt if txt else raw_queries[orig_i]
                del enc
                del gen
                cursor += len(batch_indices)
            except torch.OutOfMemoryError:
                free_memory(device)
                if cur_bs <= 1:
                    raise
                cur_bs = max(1, cur_bs // 2)
                print("    [OOM-Backoff] Stage-0 reduced batch_size to {}".format(cur_bs), flush=True)

        free_memory(device)
        out[task] = task_map
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
        print(
            "  [Stage-0] {} ({} queries) reasoned in {:.1f}s".format(
                task, len(qids), time.perf_counter() - t0
            ),
            flush=True,
        )

    del model
    del tokenizer
    free_memory(device)
    return out


def encode_texts_sorted(
    st_model: Any,
    texts: Sequence[str],
    batch_size: int,
    max_seq_len: int = 512,
    prefix: str = "",
) -> np.ndarray:
    """Length-sorted normalized FP16/FP32 embedding computation with automatic OOM batch-halving."""
    st_model.max_seq_length = max_seq_len
    prefixed = ["{}{}".format(prefix, t) for t in texts] if prefix else list(texts)
    order = sorted(range(len(prefixed)), key=lambda idx: len(prefixed[idx]))
    sorted_texts = [prefixed[idx] for idx in order]
    cur_bs = batch_size
    while True:
        try:
            with torch.inference_mode():
                sorted_emb = st_model.encode(
                    sorted_texts,
                    batch_size=cur_bs,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            break
        except torch.OutOfMemoryError:
            free_memory("cuda")
            if cur_bs <= 1:
                raise
            cur_bs = max(1, cur_bs // 2)
            print("    [OOM-Backoff] encode_texts_sorted reduced batch_size to {}".format(cur_bs), flush=True)
    out = np.empty_like(sorted_emb, dtype=np.float32)
    for pos, orig_idx in enumerate(order):
        out[orig_idx] = sorted_emb[pos]
    return out


def batched_topk_dot(
    q_emb: np.ndarray,
    d_emb: np.ndarray,
    doc_ids: Sequence[str],
    qids: Sequence[str],
    excluded_by_qid: Dict[str, Sequence[str]],
    top_k: int = 1000,
    device: str = "cuda",
) -> Tuple[Dict[str, List[Tuple[str, float]]], np.ndarray]:
    """Fast GPU matrix-multiply top-k with excluded_ids masked prior to truncation."""
    if device == "cuda" and torch.cuda.is_available():
        q_t = torch.from_numpy(q_emb).to(device=device, dtype=torch.float16)
        chunk_rows = 65536
        n_q = q_emb.shape[0]
        n_d = d_emb.shape[0]
        sim_cpu = np.empty((n_q, n_d), dtype=np.float32)
        with torch.inference_mode():
            for d0 in range(0, n_d, chunk_rows):
                d1 = min(n_d, d0 + chunk_rows)
                d_t = torch.from_numpy(d_emb[d0:d1]).to(device=device, dtype=torch.float16)
                block = torch.matmul(q_t, d_t.T).float().cpu().numpy()
                sim_cpu[:, d0:d1] = block
                del d_t
        del q_t
    else:
        sim_cpu = np.matmul(q_emb, d_emb.T)

    results: Dict[str, List[Tuple[str, float]]] = {}
    k_pull = min(d_emb.shape[0], top_k + 64)
    for qi, qid in enumerate(qids):
        row = sim_cpu[qi]
        excl = set(excluded_by_qid.get(qid, ()))
        if k_pull < len(row):
            part = np.argpartition(-row, k_pull - 1)[:k_pull]
            part = part[np.argsort(-row[part])]
        else:
            part = np.argsort(-row)
        ranked: List[Tuple[str, float]] = []
        for idx in part:
            did = doc_ids[int(idx)]
            if did not in excl and did != "N/A":
                ranked.append((did, float(row[int(idx)])))
                if len(ranked) >= top_k:
                    break
        results[qid] = ranked
    return results, sim_cpu


def weighted_rrf_fuse(
    ranked_lists_with_weights: Sequence[Tuple[Dict[str, List[Tuple[str, float]]], float]],
    qids: Sequence[str],
    rrf_k: float = 60.0,
    top_k: int = 1000,
) -> Dict[str, List[Tuple[str, float]]]:
    """Weighted Reciprocal Rank Fusion across heterogeneous lexical & dense spaces."""
    fused: Dict[str, List[Tuple[str, float]]] = {}
    for qid in qids:
        acc: Dict[str, float] = defaultdict(float)
        for run_map, weight in ranked_lists_with_weights:
            if weight <= 0.0:
                continue
            for rank_idx, (did, _) in enumerate(run_map.get(qid, ()), start=1):
                acc[did] += weight / (rrf_k + float(rank_idx))
        ordered = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        fused[qid] = ordered
    return fused


def apply_rocchio_prf(
    q_emb: np.ndarray,
    d_emb: np.ndarray,
    doc_id_to_idx: Dict[str, int],
    consensus_ranking: Dict[str, List[Tuple[str, float]]],
    qids: Sequence[str],
    top_fb: int = 5,
    alpha: float = 0.78,
    beta: float = 0.22,
) -> np.ndarray:
    """Consensus Rocchio Pseudo-Relevance Feedback query vector refinement."""
    refined = np.empty_like(q_emb, dtype=np.float32)
    weights = np.array([1.0 / float(r + 1) for r in range(top_fb)], dtype=np.float32)
    weights = weights / max(1e-8, float(weights.sum()))
    for qi, qid in enumerate(qids):
        cands = consensus_ranking.get(qid, ())[:top_fb]
        idxs = [doc_id_to_idx[did] for did, _ in cands if did in doc_id_to_idx]
        if idxs:
            w = weights[: len(idxs)]
            w = w / float(w.sum())
            centroid = (d_emb[idxs] * w[:, None]).sum(axis=0)
            norm_c = float(np.linalg.norm(centroid))
            if norm_c > 1e-8:
                centroid = centroid / norm_c
            vec = alpha * q_emb[qi] + beta * centroid
            norm_v = float(np.linalg.norm(vec))
            refined[qi] = vec / max(1e-8, norm_v)
        else:
            refined[qi] = q_emb[qi]
    return refined


def score_qwen3_reranker(
    model_name: str,
    device: str,
    work_items: List[Tuple[str, str, List[Tuple[str, str]]]],
    cache_dir: Optional[Path] = None,
    batch_size: int = 48,
    max_length: int = 768,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Batch-score (track, task, [(query, doc), ...]) items with Qwen/Qwen3-Reranker-4B log-odds."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outputs: Dict[Tuple[str, str], np.ndarray] = {}
    pending_items: List[Tuple[str, str, List[Tuple[str, str]]]] = []
    for track_name, task_name, flat_pairs in work_items:
        if cache_dir is not None:
            cp = cache_dir / "qwen3_rr_v2_{}_{}_{}.npy".format(track_name, task_name, len(flat_pairs))
            if cp.exists():
                outputs[(track_name, task_name)] = np.load(str(cp))
                continue
        pending_items.append((track_name, task_name, flat_pairs))

    if not pending_items:
        print("[Stage-2A] Loaded all Qwen3-Reranker scores from cache.", flush=True)
        return outputs

    print("[Stage-2A] Scoring top candidates with {}...".format(model_name), flush=True)
    dtype = preferred_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    doc_tag_ids = tokenizer.encode("\n<Document>: ", add_special_tokens=False)
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    budget = max(256, max_length - len(prefix_ids) - len(suffix_ids) - len(doc_tag_ids))
    q_budget = max(128, (budget * 2) // 5)
    d_budget = max(128, budget - q_budget)

    supports_logits_to_keep = True

    for track_name, task_name, flat_pairs in pending_items:
        t0 = time.perf_counter()
        inst = TASK_INSTRUCTIONS.get(task_name, TASK_INSTRUCTIONS["stackoverflow"])
        order = sorted(range(len(flat_pairs)), key=lambda idx: len(flat_pairs[idx][1]))
        sorted_scores = np.zeros(len(flat_pairs), dtype=np.float32)

        q_headers = ["<Instruct>: {}\n<Query>: {}".format(inst, flat_pairs[i][0]) for i in order]
        d_bodies = [flat_pairs[i][1] for i in order]
        q_enc = tokenizer(
            q_headers,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=q_budget,
        )["input_ids"]
        d_enc = tokenizer(
            d_bodies,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=d_budget,
        )["input_ids"]

        cursor = 0
        cur_bs = batch_size
        while cursor < len(order):
            end = min(len(order), cursor + cur_bs)
            batch_ids = []
            for pos in range(cursor, end):
                seq = prefix_ids + q_enc[pos] + doc_tag_ids + d_enc[pos] + suffix_ids
                batch_ids.append({"input_ids": seq, "attention_mask": [1] * len(seq)})
            try:
                padded = tokenizer.pad(batch_ids, padding=True, return_tensors="pt")
                padded = {k: v.to(device) for k, v in padded.items()}
                with torch.inference_mode():
                    if supports_logits_to_keep:
                        try:
                            logits = model(**padded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                        except TypeError:
                            supports_logits_to_keep = False
                            logits = model(**padded, use_cache=False).logits[:, -1, :]
                    else:
                        logits = model(**padded, use_cache=False).logits[:, -1, :]
                    log_odds = (logits[:, yes_id].float() - logits[:, no_id].float())
                sorted_scores[cursor:end] = log_odds.cpu().numpy()
                del padded
                del logits
                cursor = end
            except torch.OutOfMemoryError:
                free_memory(device)
                if cur_bs <= 1:
                    raise
                cur_bs = max(1, cur_bs // 2)
                print("    [OOM-Backoff] Qwen3-Reranker reduced batch_size to {}".format(cur_bs), flush=True)

        restored = np.empty(len(flat_pairs), dtype=np.float32)
        for pos, orig_i in enumerate(order):
            restored[orig_i] = sorted_scores[pos]
        outputs[(track_name, task_name)] = restored
        if cache_dir is not None:
            cp = cache_dir / "qwen3_rr_v2_{}_{}_{}.npy".format(track_name, task_name, len(flat_pairs))
            np.save(str(cp), restored)
        print(
            "  [Qwen3-Reranker] {}/{} ({} pairs) in {:.1f}s".format(
                track_name, task_name, len(flat_pairs), time.perf_counter() - t0
            ),
            flush=True,
        )

    del model
    del tokenizer
    free_memory(device)
    return outputs


def score_gemma_reranker(
    model_name: str,
    device: str,
    work_items: List[Tuple[str, str, List[Tuple[str, str]]]],
    cache_dir: Optional[Path] = None,
    batch_size: int = 56,
    max_length: int = 640,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Batch-score (track, task, [(query, doc), ...]) items with BAAI/bge-reranker-v2-gemma raw Yes logits."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outputs: Dict[Tuple[str, str], np.ndarray] = {}
    pending_items: List[Tuple[str, str, List[Tuple[str, str]]]] = []
    for track_name, task_name, flat_pairs in work_items:
        if cache_dir is not None:
            cp = cache_dir / "gemma_rr_raw_{}_{}_{}.npy".format(track_name, task_name, len(flat_pairs))
            if cp.exists():
                outputs[(track_name, task_name)] = np.load(str(cp))
                continue
        pending_items.append((track_name, task_name, flat_pairs))

    if not pending_items:
        print("[Stage-2B] Loaded all Gemma-Reranker raw logits from cache.", flush=True)
        return outputs

    print("[Stage-2B] Scoring top candidates with {}...".format(model_name), flush=True)
    dtype = preferred_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompt_str = (
        "Given a query A and a passage B, determine whether the passage contains an answer "
        "to the query by providing a prediction of either 'Yes' or 'No'."
    )
    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    sep_ids = tokenizer("\n", return_tensors=None, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_str, return_tensors=None, add_special_tokens=False)["input_ids"]
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    supports_logits_to_keep = True

    for track_name, task_name, flat_pairs in pending_items:
        t0 = time.perf_counter()
        order = sorted(range(len(flat_pairs)), key=lambda idx: len(flat_pairs[idx][1]))
        sorted_scores = np.zeros(len(flat_pairs), dtype=np.float32)

        q_texts = ["A: {}".format(flat_pairs[i][0]) for i in order]
        p_texts = ["B: {}".format(flat_pairs[i][1]) for i in order]
        q_enc = tokenizer(
            q_texts,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max(128, (max_length * 2) // 5),
        )["input_ids"]
        p_enc = tokenizer(
            p_texts,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max(192, (max_length * 3) // 5),
        )["input_ids"]

        encoded_items = []
        for pos in range(len(order)):
            ids = bos + q_enc[pos] + sep_ids + p_enc[pos] + sep_ids + prompt_ids
            encoded_items.append({"input_ids": ids, "attention_mask": [1] * len(ids)})

        cursor = 0
        cur_bs = batch_size
        while cursor < len(encoded_items):
            end = min(len(encoded_items), cursor + cur_bs)
            batch = encoded_items[cursor:end]
            try:
                padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
                padded = {k: v.to(device) for k, v in padded.items()}
                with torch.inference_mode():
                    if supports_logits_to_keep:
                        try:
                            raw = model(**padded, use_cache=False, logits_to_keep=1).logits[:, -1, yes_id].float()
                        except TypeError:
                            supports_logits_to_keep = False
                            raw = model(**padded, use_cache=False).logits[:, -1, yes_id].float()
                    else:
                        raw = model(**padded, use_cache=False).logits[:, -1, yes_id].float()
                sorted_scores[cursor:end] = raw.cpu().numpy()
                del padded
                del raw
                cursor = end
            except torch.OutOfMemoryError:
                free_memory(device)
                if cur_bs <= 1:
                    raise
                cur_bs = max(1, cur_bs // 2)
                print("    [OOM-Backoff] Gemma-Reranker reduced batch_size to {}".format(cur_bs), flush=True)

        restored = np.empty(len(flat_pairs), dtype=np.float32)
        for pos, orig_i in enumerate(order):
            restored[orig_i] = sorted_scores[pos]
        outputs[(track_name, task_name)] = restored
        if cache_dir is not None:
            cp = cache_dir / "gemma_rr_raw_{}_{}_{}.npy".format(track_name, task_name, len(flat_pairs))
            np.save(str(cp), restored)
        print(
            "  [Gemma-Reranker] {}/{} ({} pairs) in {:.1f}s".format(
                track_name, task_name, len(flat_pairs), time.perf_counter() - t0
            ),
            flush=True,
        )

    del model
    del tokenizer
    free_memory(device)
    return outputs


def run_listwise_reasonrank_top20(
    model_name: str,
    device: str,
    listwise_jobs: List[Tuple[str, str, str, str, List[Tuple[str, str]]]],
    cache_path: Optional[Path] = None,
    batch_size: int = 2,
    max_new_tokens: int = 512,
) -> Dict[Tuple[str, str, str], List[str]]:
    """
    OOM-safe batched ReasonRank-7B listwise reranking over top candidates.

    listwise_jobs entries: (track_name, task_name, qid, query_text, [(doc_id, doc_text), ...])
    Returns: {(track_name, task_name, qid): ordered_doc_ids}
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cached_map: Dict[str, List[str]] = {}
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as handle:
            cached_map = json.load(handle)

    results: Dict[Tuple[str, str, str], List[str]] = {}
    pending_jobs: List[Tuple[str, str, str, str, List[Tuple[str, str]]]] = []
    for tr, tk, qid, qt, clist in listwise_jobs:
        ckey = "{}::{}::{}".format(tr, tk, qid)
        if ckey in cached_map:
            results[(tr, tk, qid)] = cached_map[ckey]
        else:
            pending_jobs.append((tr, tk, qid, qt, clist))

    if not pending_jobs:
        print("[Stage-3] Loaded all ReasonRank listwise windows from cache.", flush=True)
        return results

    print("[Stage-3] Running Listwise Reasoning Reranker {} on {} windows...".format(model_name, len(pending_jobs)), flush=True)
    dtype = preferred_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    sys_prompt = (
        "You are RankLLM, an intelligent assistant that can rank passages based on their "
        "relevance to the query. Given a query and a passage list, you first thinks about "
        "the reasoning process in the mind and then provides the answer (i.e., the reranked "
        "passage list). The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, i.e., <think> reasoning process here "
        "</think> <answer> answer here </answer>."
    )

    prompts: List[str] = []
    for _, _, _, q_text, cand_list in pending_jobs:
        clean_q = sanitize_passage_brackets(q_text).strip()
        num = len(cand_list)
        lines = [
            "I will provide you with {} passages, each indicated by a numerical identifier []. "
            "Rank the passages based on their relevance to the search query: {}.".format(num, clean_q),
            "",
        ]
        for idx, (_, d_txt) in enumerate(cand_list, start=1):
            clipped = " ".join(sanitize_passage_brackets(d_txt.strip()).split())[:650]
            lines.append("[{}] {}".format(idx, clipped))
        lines.append(
            "Search Query: {}.\nRank the {} passages above based on their relevance to the search query. "
            "All the passages should be included and listed using identifiers, in descending order of "
            "relevance. The format of the answer should be [] > [], e.g., [2] > [1].".format(clean_q, num)
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "\n".join(lines)},
        ]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    order = sorted(range(len(prompts)), key=lambda idx: len(prompts[idx]))
    t0 = time.perf_counter()

    for step_idx, start in enumerate(range(0, len(order), batch_size), start=1):
        batch_idx = order[start : start + batch_size]
        batch_prompts = [prompts[i] for i in batch_idx]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=6144,
        ).to(device)
        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        in_len = enc["input_ids"].shape[1]
        for loc_i, orig_i in enumerate(batch_idx):
            track_name, task_name, qid, _, cand_list = pending_jobs[orig_i]
            gen_txt = tokenizer.decode(out_ids[loc_i, in_len:], skip_special_tokens=True)
            perm = parse_listwise_ranking(gen_txt, len(cand_list))
            ordered_ids = [cand_list[p][0] for p in perm]
            results[(track_name, task_name, qid)] = ordered_ids
            cached_map["{}::{}::{}".format(track_name, task_name, qid)] = ordered_ids

        del enc
        del out_ids
        if step_idx % 16 == 0:
            free_memory(device)
            if cache_path is not None:
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json.dump(cached_map, handle)

        done = min(len(order), start + batch_size)
        if done % 32 == 0 or done == len(order):
            print(
                "  [ReasonRank-7B] {}/{} windows completed ({:.1f}s elapsed)".format(
                    done, len(order), time.perf_counter() - t0
                ),
                flush=True,
            )

    if cache_path is not None:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cached_map, handle)

    del model
    del tokenizer
    free_memory(device)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Official BRIGHT 12-Split GPU Evaluation for piragi.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=BRIGHT_TASKS,
        choices=BRIGHT_TASKS,
        help="BRIGHT splits to evaluate (default: all 12 splits)",
    )
    parser.add_argument(
        "--reasoning-source",
        choices=["local_llm", "official_cot", "both"],
        default="both",
        help="Evaluate local piragi ThinkThenRetrieve, official BRIGHT gpt4_reason CoT, or both",
    )
    parser.add_argument(
        "--local-reasoner-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Open-weights model for local ThinkThenRetrieve synthesis",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=25,
        help="Candidate depth for Stage-2 causal LLM reranking (ranks > top-n preserve Stage-1 order)",
    )
    parser.add_argument(
        "--enable-listwise",
        action="store_true",
        help="Also run Stage-3 ReasonRank-7B listwise reranking over Top-20 candidates",
    )
    parser.add_argument(
        "--listwise-model",
        default="liuwenhan/reasonrank-7B",
        help="Model for Stage-3 listwise comparative reranking",
    )
    parser.add_argument(
        "--output-dir",
        default="evals/submissions/bright",
        help="Output directory for official BRIGHT score.json and results.json artifacts",
    )
    parser.add_argument(
        "--summary-json",
        default="evals/results/bright_live_summary.json",
        help="Consolidated benchmark summary JSON with bootstrap 95%% CIs",
    )
    parser.add_argument(
        "--purge-hf-model-after-use",
        action="store_true",
        help="Delete HF hub cache of each multi-GB model after its stage checkpoints complete",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    device = select_device()
    dtype = preferred_dtype(device)
    telemetry = get_telemetry()
    out_root = Path(args.output_dir)
    summary_path = Path(args.summary_json)
    cache_dir = Path("evals/results/bright_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[BRIGHT] Device={} | Dtype={} | Tasks={} | Reasoning={} | Listwise={}".format(
            device, dtype, len(args.tasks), args.reasoning_source, args.enable_listwise
        ),
        flush=True,
    )

    # 1. Load examples, qrels, excluded_ids, and corpora for selected splits
    examples_all = load_dataset("xlangai/BRIGHT", "examples")
    cot_all = (
        load_dataset("xlangai/BRIGHT", "gpt4_reason")
        if args.reasoning_source in ("official_cot", "both")
        else None
    )
    docs_all = load_dataset("xlangai/BRIGHT", "documents")

    task_meta: Dict[str, Dict[str, Any]] = {}
    unique_corpora: Dict[str, Dict[str, Any]] = {}
    queries_for_local_llm: Dict[str, Tuple[List[str], List[str]]] = {}

    for task in args.tasks:
        ex_rows = examples_all[task]
        qids: List[str] = []
        raw_queries: List[str] = []
        qrels: Dict[str, Dict[str, int]] = {}
        excluded_ids: Dict[str, List[str]] = {}

        for row in ex_rows:
            qid = str(row["id"])
            qids.append(qid)
            raw_queries.append(str(row["query"]))
            qrels[qid] = {str(gid): 1 for gid in row["gold_ids"]}
            excluded_ids[qid] = [str(eid) for eid in row["excluded_ids"]]
            assert not set(excluded_ids[qid]).intersection(set(qrels[qid].keys())), (
                "Excluded/gold overlap on {}:{}".format(task, qid)
            )

        cot_queries_map: Dict[str, str] = {}
        if cot_all is not None:
            for row in cot_all[task]:
                cot_queries_map[str(row["id"])] = str(row["query"])

        doc_rows = docs_all[task]
        doc_ids = [str(r["id"]) for r in doc_rows]
        raw_docs = [str(r["content"]) for r in doc_rows]
        fp = corpus_fingerprint(doc_ids, raw_docs)
        if fp not in unique_corpora:
            norm_docs = normalize_corpus_texts(raw_docs)
            unique_corpora[fp] = {
                "doc_ids": doc_ids,
                "norm_docs": norm_docs,
                "doc_id_to_idx": {did: i for i, did in enumerate(doc_ids)},
            }
        print(
            "[Data] {:<20s} | queries={:4d} | docs={:7d} | corpus_fp={}".format(
                task, len(qids), len(doc_ids), fp
            ),
            flush=True,
        )
        task_meta[task] = {
            "qids": qids,
            "raw_queries": raw_queries,
            "cot_queries_map": cot_queries_map,
            "qrels": qrels,
            "excluded_ids": excluded_ids,
            "corpus_fp": fp,
        }
        queries_for_local_llm[task] = (qids, raw_queries)

    # 2. Stage-0: Generate local ThinkThenRetrieve CoT traces if requested
    local_reason_map: Dict[str, Dict[str, str]] = {}
    if args.reasoning_source in ("local_llm", "both"):
        local_reason_map = generate_local_think_then_retrieve(
            queries_by_task=queries_for_local_llm,
            model_name=args.local_reasoner_model,
            device=device,
            cache_path=cache_dir / "local_think_then_retrieve.json",
        )
        if args.purge_hf_model_after_use:
            purge_hf_repo_cache(args.local_reasoner_model)

    active_tracks: List[str] = []
    if args.reasoning_source in ("local_llm", "both"):
        active_tracks.append("piragi_local_ttr")
    if args.reasoning_source in ("official_cot", "both"):
        active_tracks.append("piragi_official_cot")

    # 3. Fast C/CSR Sparse Matrix Lexical BM25 (Condition A baseline + reasoned lexical signals)
    bm25_raw_runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    bm25_reason_runs: Dict[Tuple[str, str], Dict[str, List[Tuple[str, float]]]] = {}

    tasks_by_fp_bm25: Dict[str, List[str]] = defaultdict(list)
    for task in args.tasks:
        tasks_by_fp_bm25[task_meta[task]["corpus_fp"]].append(task)

    for fp, fp_tasks in tasks_by_fp_bm25.items():
        t0 = time.perf_counter()
        cinfo = unique_corpora[fp]
        q_batches: List[Tuple[str, Sequence[str], Sequence[str], Dict[str, Sequence[str]]]] = []
        for task in fp_tasks:
            meta = task_meta[task]
            q_batches.append(("raw::{}".format(task), meta["qids"], meta["raw_queries"], meta["excluded_ids"]))
            for tr in active_tracks:
                reason_src = local_reason_map[task] if tr == "piragi_local_ttr" else meta["cot_queries_map"]
                r_queries = [
                    "{} {}".format(meta["raw_queries"][qi], reason_src.get(qid, ""))
                    for qi, qid in enumerate(meta["qids"])
                ]
                q_batches.append(
                    ("reason::{}::{}".format(tr, task), meta["qids"], r_queries, meta["excluded_ids"])
                )

        bm25_res = build_and_score_sparse_bm25(
            norm_docs=cinfo["norm_docs"],
            doc_ids=cinfo["doc_ids"],
            query_batches=q_batches,
            top_k=1000,
        )
        for task in fp_tasks:
            bm25_raw_runs[task] = bm25_res["raw::{}".format(task)]
            for tr in active_tracks:
                bm25_reason_runs[(tr, task)] = bm25_res["reason::{}::{}".format(tr, task)]
        print(
            "[BM25-CSR] Indexed & scored {} ({} docs) in {:.1f}s".format(
                ",".join(fp_tasks), len(cinfo["doc_ids"]), time.perf_counter() - t0
            ),
            flush=True,
        )
    free_memory(device)

    # 4. Stage-1A Dense Models:
    #    - BAAI/bge-large-en-v1.5 across 100% of every corpus (ultra-fast BERT encoder)
    #    - Qwen/Qwen3-Embedding-0.6B full-corpus for <=65k docs (or if cached), and high-recall
    #      union shortlist (top-450/query from BM25 + BGE-large) on >65k corpora (e.g. leetcode 414k)
    dense_raw_runs: Dict[Tuple[str, str], Dict[str, List[Tuple[str, float]]]] = {}
    dense_reason_runs: Dict[Tuple[str, str, str], Dict[str, List[Tuple[str, float]]]] = {}
    dense_prf_runs: Dict[Tuple[str, str, str], Dict[str, List[Tuple[str, float]]]] = {}

    tasks_by_corpus_fp: Dict[str, List[str]] = defaultdict(list)
    for task in args.tasks:
        tasks_by_corpus_fp[task_meta[task]["corpus_fp"]].append(task)

    # 4A. Full-corpus BAAI/bge-large-en-v1.5
    print("[Stage-1A] Loading full-corpus dense model BAAI/bge-large-en-v1.5...", flush=True)
    st_bge = SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": dtype},
    )
    bge_prefix = "Represent this sentence for searching relevant passages: "
    for fp, fp_tasks in tasks_by_corpus_fp.items():
        cinfo = unique_corpora[fp]
        cache_npy = cache_dir / "bge_large_{}.npy".format(fp)
        if cache_npy.exists():
            d_emb = np.load(str(cache_npy)).astype(np.float32)
            print("  [bge_large] Loaded cached embeddings for {} {}".format(fp, d_emb.shape), flush=True)
        else:
            t0 = time.perf_counter()
            emb = encode_texts_sorted(st_bge, cinfo["norm_docs"], batch_size=512, max_seq_len=384)
            np.save(str(cache_npy), emb.astype(np.float16))
            d_emb = emb
            print(
                "  [bge_large] Encoded {} docs (fp={}) in {:.1f}s".format(
                    len(cinfo["doc_ids"]), fp, time.perf_counter() - t0
                ),
                flush=True,
            )

        for task in fp_tasks:
            meta = task_meta[task]
            q_raw_emb = encode_texts_sorted(st_bge, meta["raw_queries"], batch_size=64, max_seq_len=512, prefix=bge_prefix)
            raw_top, _ = batched_topk_dot(
                q_raw_emb, d_emb, cinfo["doc_ids"], meta["qids"], meta["excluded_ids"], top_k=1000, device=device
            )
            dense_raw_runs[("bge_large", task)] = raw_top

            for tr in active_tracks:
                reason_src = local_reason_map[task] if tr == "piragi_local_ttr" else meta["cot_queries_map"]
                r_texts = [reason_src.get(qid, meta["raw_queries"][i]) for i, qid in enumerate(meta["qids"])]
                q_reason_emb = encode_texts_sorted(st_bge, r_texts, batch_size=64, max_seq_len=512, prefix=bge_prefix)
                q_joint = 0.35 * q_raw_emb + 0.65 * q_reason_emb
                q_joint = q_joint / np.maximum(1e-8, np.linalg.norm(q_joint, axis=1, keepdims=True))
                r_top, _ = batched_topk_dot(
                    q_joint, d_emb, cinfo["doc_ids"], meta["qids"], meta["excluded_ids"], top_k=1000, device=device
                )
                dense_reason_runs[("bge_large", tr, task)] = r_top

                seed_rrf = weighted_rrf_fuse(
                    [
                        (r_top, 1.4),
                        (raw_top, 0.7),
                        (bm25_reason_runs[(tr, task)], 1.0),
                        (bm25_raw_runs[task], 0.5),
                    ],
                    meta["qids"],
                    top_k=50,
                )
                q_prf = apply_rocchio_prf(q_joint, d_emb, cinfo["doc_id_to_idx"], seed_rrf, meta["qids"], top_fb=5)
                prf_top, _ = batched_topk_dot(
                    q_prf, d_emb, cinfo["doc_ids"], meta["qids"], meta["excluded_ids"], top_k=1000, device=device
                )
                dense_prf_runs[("bge_large", tr, task)] = prf_top
        del d_emb

    del st_bge
    free_memory(device)

    # 4B. Instruction-Aware Qwen/Qwen3-Embedding-0.6B
    print("[Stage-1A] Loading instruction-aware Qwen/Qwen3-Embedding-0.6B...", flush=True)
    st_qwen = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": dtype},
        tokenizer_kwargs={"padding_side": "left"},
    )
    for fp, fp_tasks in tasks_by_corpus_fp.items():
        cinfo = unique_corpora[fp]
        full_cache_npy = cache_dir / "qwen3_06b_{}.npy".format(fp)
        if full_cache_npy.exists():
            sub_doc_ids = cinfo["doc_ids"]
            sub_idx_map = cinfo["doc_id_to_idx"]
            d_emb = np.load(str(full_cache_npy)).astype(np.float32)
            print("  [qwen3_06b] Loaded full cached embeddings for {} {}".format(fp, d_emb.shape), flush=True)
        elif len(cinfo["doc_ids"]) <= 65000:
            t0 = time.perf_counter()
            sub_doc_ids = cinfo["doc_ids"]
            sub_idx_map = cinfo["doc_id_to_idx"]
            emb = encode_texts_sorted(st_qwen, cinfo["norm_docs"], batch_size=192, max_seq_len=512)
            np.save(str(full_cache_npy), emb.astype(np.float16))
            d_emb = emb
            print(
                "  [qwen3_06b] Encoded full {} docs (fp={}) in {:.1f}s".format(
                    len(sub_doc_ids), fp, time.perf_counter() - t0
                ),
                flush=True,
            )
        else:
            t0 = time.perf_counter()
            shortlist_set = set()
            doc_id_to_idx = cinfo["doc_id_to_idx"]
            for task in fp_tasks:
                for qid in task_meta[task]["qids"]:
                    for did, _ in bm25_raw_runs[task].get(qid, ())[:400]:
                        shortlist_set.add(did)
                    for did, _ in dense_raw_runs[("bge_large", task)].get(qid, ())[:500]:
                        shortlist_set.add(did)
                    for tr in active_tracks:
                        for did, _ in bm25_reason_runs[(tr, task)].get(qid, ())[:400]:
                            shortlist_set.add(did)
                        for did, _ in dense_reason_runs[("bge_large", tr, task)].get(qid, ())[:500]:
                            shortlist_set.add(did)
                        for did, _ in dense_prf_runs[("bge_large", tr, task)].get(qid, ())[:400]:
                            shortlist_set.add(did)
            sub_doc_ids = [did for did in sorted(shortlist_set) if did in doc_id_to_idx]
            sub_docs = [cinfo["norm_docs"][doc_id_to_idx[did]] for did in sub_doc_ids]
            sub_idx_map = {did: i for i, did in enumerate(sub_doc_ids)}
            d_emb = encode_texts_sorted(st_qwen, sub_docs, batch_size=192, max_seq_len=640)
            print(
                "  [qwen3_06b] Encoded shortlist {}/{} docs (fp={}) in {:.1f}s".format(
                    len(sub_doc_ids), len(cinfo["doc_ids"]), fp, time.perf_counter() - t0
                ),
                flush=True,
            )

        for task in fp_tasks:
            meta = task_meta[task]
            q_prefix = "Instruct: {}\nQuery: ".format(TASK_INSTRUCTIONS[task])
            q_raw_emb = encode_texts_sorted(st_qwen, meta["raw_queries"], batch_size=64, max_seq_len=768, prefix=q_prefix)
            raw_top, _ = batched_topk_dot(
                q_raw_emb, d_emb, sub_doc_ids, meta["qids"], meta["excluded_ids"], top_k=1000, device=device
            )
            dense_raw_runs[("qwen3_06b", task)] = raw_top

            for tr in active_tracks:
                reason_src = local_reason_map[task] if tr == "piragi_local_ttr" else meta["cot_queries_map"]
                r_texts = [reason_src.get(qid, meta["raw_queries"][i]) for i, qid in enumerate(meta["qids"])]
                q_reason_emb = encode_texts_sorted(st_qwen, r_texts, batch_size=64, max_seq_len=1536, prefix=q_prefix)
                q_joint = 0.35 * q_raw_emb + 0.65 * q_reason_emb
                q_joint = q_joint / np.maximum(1e-8, np.linalg.norm(q_joint, axis=1, keepdims=True))
                r_top, _ = batched_topk_dot(
                    q_joint, d_emb, sub_doc_ids, meta["qids"], meta["excluded_ids"], top_k=1000, device=device
                )
                dense_reason_runs[("qwen3_06b", tr, task)] = r_top

                seed_rrf = weighted_rrf_fuse(
                    [
                        (r_top, 1.4),
                        (raw_top, 0.7),
                        (dense_prf_runs[("bge_large", tr, task)], 1.0),
                        (bm25_reason_runs[(tr, task)], 1.0),
                    ],
                    meta["qids"],
                    top_k=50,
                )
                q_prf = apply_rocchio_prf(q_joint, d_emb, sub_idx_map, seed_rrf, meta["qids"], top_fb=5)
                prf_top, _ = batched_topk_dot(
                    q_prf, d_emb, sub_doc_ids, meta["qids"], meta["excluded_ids"], top_k=1000, device=device
                )
                dense_prf_runs[("qwen3_06b", tr, task)] = prf_top
        del d_emb

    del st_qwen
    free_memory(device)

    # 5. Heavy 8B Specialist Reasoning Retriever: BAAI/bge-reasoner-embed-qwen3-8b-0923 over High-Recall Union Pool
    bge8b_runs: Dict[Tuple[str, str], Dict[str, List[Tuple[str, float]]]] = {}
    bge8b_raw_runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    bge8b_cache_file = cache_dir / "bge8b_stage1b_runs.json"
    cached_8b: Dict[str, Any] = {"raw": {}, "tracks": {}}
    if bge8b_cache_file.exists():
        with open(bge8b_cache_file, "r", encoding="utf-8") as handle:
            cached_8b = json.load(handle)

    tasks_needing_8b = [
        t for t in args.tasks
        if t not in cached_8b.get("raw", {})
        or any("{}::{}".format(tr, t) not in cached_8b.get("tracks", {}) for tr in active_tracks)
    ]

    if tasks_needing_8b:
        print("[Stage-1B] Loading 8B specialist BAAI/bge-reasoner-embed-qwen3-8b-0923...", flush=True)
        st_8b = SentenceTransformer(
            "BAAI/bge-reasoner-embed-qwen3-8b-0923",
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": dtype},
            tokenizer_kwargs={"padding_side": "left"},
        )

        # Group tasks by corpus_fp so shared corpora (e.g. aops + theoremqa_questions) encode union docs once
        tasks_by_fp: Dict[str, List[str]] = defaultdict(list)
        for task in tasks_needing_8b:
            tasks_by_fp[task_meta[task]["corpus_fp"]].append(task)

        for fp, fp_tasks in tasks_by_fp.items():
            t0 = time.perf_counter()
            cinfo = unique_corpora[fp]
            doc_id_to_idx = cinfo["doc_id_to_idx"]

            # Build tight, high-recall union shortlist across all Stage-1A spaces
            pool_id_set = set()
            for task in fp_tasks:
                meta = task_meta[task]
                for qid in meta["qids"]:
                    for did, _ in bm25_raw_runs[task].get(qid, ())[:120]:
                        pool_id_set.add(did)
                    for m_key in ("qwen3_06b", "bge_large"):
                        for did, _ in dense_raw_runs[(m_key, task)].get(qid, ())[:150]:
                            pool_id_set.add(did)
                        for tr in active_tracks:
                            for did, _ in dense_reason_runs[(m_key, tr, task)].get(qid, ())[:150]:
                                pool_id_set.add(did)
                            for did, _ in dense_prf_runs[(m_key, tr, task)].get(qid, ())[:150]:
                                pool_id_set.add(did)
                            for did, _ in bm25_reason_runs[(tr, task)].get(qid, ())[:120]:
                                pool_id_set.add(did)

            pool_doc_ids = [did for did in sorted(pool_id_set) if did in doc_id_to_idx]
            pool_docs = [cinfo["norm_docs"][doc_id_to_idx[did]] for did in pool_doc_ids]
            pool_idx_map = {did: i for i, did in enumerate(pool_doc_ids)}

            pool_emb = encode_texts_sorted(st_8b, pool_docs, batch_size=32, max_seq_len=768)

            for task in fp_tasks:
                meta = task_meta[task]
                q_prefix = "Instruct: {}\nQuery: ".format(TASK_INSTRUCTIONS[task])

                q_raw_8b = encode_texts_sorted(
                    st_8b, meta["raw_queries"], batch_size=16, max_seq_len=1024, prefix=q_prefix
                )
                raw_8b_top, _ = batched_topk_dot(
                    q_raw_8b, pool_emb, pool_doc_ids, meta["qids"], meta["excluded_ids"], top_k=500, device=device
                )
                cached_8b.setdefault("raw", {})[task] = raw_8b_top

                for tr in active_tracks:
                    reason_src = local_reason_map[task] if tr == "piragi_local_ttr" else meta["cot_queries_map"]
                    r_texts = [reason_src.get(qid, meta["raw_queries"][i]) for i, qid in enumerate(meta["qids"])]
                    q_reason_8b = encode_texts_sorted(
                        st_8b, r_texts, batch_size=16, max_seq_len=1536, prefix=q_prefix
                    )
                    q_joint_8b = 0.38 * q_raw_8b + 0.62 * q_reason_8b
                    q_joint_8b = q_joint_8b / np.maximum(1e-8, np.linalg.norm(q_joint_8b, axis=1, keepdims=True))
                    r_8b_top, _ = batched_topk_dot(
                        q_joint_8b, pool_emb, pool_doc_ids, meta["qids"], meta["excluded_ids"], top_k=500, device=device
                    )
                    q_prf_8b = apply_rocchio_prf(q_joint_8b, pool_emb, pool_idx_map, r_8b_top, meta["qids"], top_fb=5)
                    prf_8b_top, _ = batched_topk_dot(
                        q_prf_8b, pool_emb, pool_doc_ids, meta["qids"], meta["excluded_ids"], top_k=500, device=device
                    )
                    fused_8b = weighted_rrf_fuse([(r_8b_top, 1.0), (prf_8b_top, 1.15)], meta["qids"], top_k=500)
                    cached_8b.setdefault("tracks", {})["{}::{}".format(tr, task)] = fused_8b

            with open(bge8b_cache_file, "w", encoding="utf-8") as handle:
                json.dump(cached_8b, handle)

            print(
                "  [BGE-Reasoner-8B] {:<28s} | union_pool={:6d}/{:6d} docs | {:.1f}s".format(
                    ",".join(fp_tasks), len(pool_doc_ids), len(cinfo["doc_ids"]), time.perf_counter() - t0
                ),
                flush=True,
            )

        del st_8b
        free_memory(device)
        if args.purge_hf_model_after_use:
            purge_hf_repo_cache("BAAI/bge-reasoner-embed-qwen3-8b-0923")
    else:
        print("[Stage-1B] Loaded cached BGE-Reasoner-8B runs from {}".format(bge8b_cache_file), flush=True)

    for task in args.tasks:
        bge8b_raw_runs[task] = {
            qid: [(str(d), float(s)) for d, s in pairs]
            for qid, pairs in cached_8b["raw"][task].items()
        }
        for tr in active_tracks:
            bge8b_runs[(tr, task)] = {
                qid: [(str(d), float(s)) for d, s in pairs]
                for qid, pairs in cached_8b["tracks"]["{}::{}".format(tr, task)].items()
            }

    # 6. Assemble Stage-1 Condition B (Raw Hybrid) & Condition C (Think-Then-Retrieve Multi-Space Hybrid RRF + PRF)
    stage1_raw_hybrid: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    stage1_ttr_hybrid: Dict[Tuple[str, str], Dict[str, List[Tuple[str, float]]]] = {}

    rerank_work_items: List[Tuple[str, str, List[Tuple[str, str]]]] = []
    rerank_slices: Dict[Tuple[str, str], List[Tuple[str, List[str], List[float], List[Tuple[str, float]]]]] = {}

    for task in args.tasks:
        meta = task_meta[task]
        cinfo = unique_corpora[meta["corpus_fp"]]
        doc_id_to_idx = cinfo["doc_id_to_idx"]

        stage1_raw_hybrid[task] = weighted_rrf_fuse(
            [
                (bge8b_raw_runs[task], 2.2),
                (dense_raw_runs[("qwen3_06b", task)], 1.2),
                (dense_raw_runs[("bge_large", task)], 0.9),
                (bm25_raw_runs[task], 0.8),
            ],
            meta["qids"],
            top_k=100,
        )

        for tr in active_tracks:
            reason_src = local_reason_map[task] if tr == "piragi_local_ttr" else meta["cot_queries_map"]
            fused_ttr = weighted_rrf_fuse(
                [
                    (bge8b_runs[(tr, task)], 2.6),
                    (bge8b_raw_runs[task], 1.1),
                    (dense_prf_runs[("qwen3_06b", tr, task)], 1.35),
                    (dense_reason_runs[("qwen3_06b", tr, task)], 1.1),
                    (dense_prf_runs[("bge_large", tr, task)], 0.95),
                    (bm25_reason_runs[(tr, task)], 1.15),
                    (bm25_raw_runs[task], 0.65),
                ],
                meta["qids"],
                top_k=max(100, args.rerank_top_n),
            )
            stage1_ttr_hybrid[(tr, task)] = fused_ttr

            flat_pairs: List[Tuple[str, str]] = []
            q_slices: List[Tuple[str, List[str], List[float], List[Tuple[str, float]]]] = []
            for qi, qid in enumerate(meta["qids"]):
                all_cands = fused_ttr.get(qid, ())[: max(100, args.rerank_top_n)]
                cands = all_cands[: args.rerank_top_n]
                tail_cands = all_cands[args.rerank_top_n : 100]
                c_ids = [did for did, _ in cands]
                c_scores = [float(sc) for _, sc in cands]
                raw_q = meta["raw_queries"][qi].strip()
                r_hint = reason_src.get(qid, "").strip()
                combined_q = (
                    "{}\n\nKey Reasoning Context: {}".format(raw_q[:750], r_hint[:450])
                    if r_hint
                    else raw_q[:1000]
                )
                for did in c_ids:
                    flat_pairs.append((combined_q, cinfo["norm_docs"][doc_id_to_idx[did]]))
                q_slices.append((qid, c_ids, c_scores, tail_cands))

            rerank_work_items.append((tr, task, flat_pairs))
            rerank_slices[(tr, task)] = q_slices

    # 7. Stage-2 Dual Pointwise Causal LLM Reranking (Qwen3-Reranker-4B + BAAI/bge-reranker-v2-gemma)
    with request_context():
        qwen_rr_scores = score_qwen3_reranker(
            model_name="Qwen/Qwen3-Reranker-4B",
            device=device,
            work_items=rerank_work_items,
            cache_dir=cache_dir,
            batch_size=48,
            max_length=768,
        )
        if args.purge_hf_model_after_use:
            purge_hf_repo_cache("Qwen/Qwen3-Reranker-4B")
        gemma_rr_scores = score_gemma_reranker(
            model_name="BAAI/bge-reranker-v2-gemma",
            device=device,
            work_items=rerank_work_items,
            cache_dir=cache_dir,
            batch_size=56,
            max_length=640,
        )
        if args.purge_hf_model_after_use:
            purge_hf_repo_cache("BAAI/bge-reranker-v2-gemma")

    stage2_final: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    listwise_jobs: List[Tuple[str, str, str, str, List[Tuple[str, str]]]] = []

    for tr, task, _ in rerank_work_items:
        meta = task_meta[task]
        cinfo = unique_corpora[meta["corpus_fp"]]
        doc_id_to_idx = cinfo["doc_id_to_idx"]
        reason_src = local_reason_map[task] if tr == "piragi_local_ttr" else meta["cot_queries_map"]

        q_arr = qwen_rr_scores[(tr, task)]
        g_arr = gemma_rr_scores[(tr, task)]
        cursor = 0
        task_scores: Dict[str, Dict[str, float]] = {}

        for qi, (qid, c_ids, s1_scores, tail_cands) in enumerate(rerank_slices[(tr, task)]):
            n_c = len(c_ids)
            if n_c == 0:
                task_scores[qid] = {}
                continue
            q_sub = q_arr[cursor : cursor + n_c].astype(np.float64)
            g_sub = g_arr[cursor : cursor + n_c].astype(np.float64)
            cursor += n_c

            s1_np = np.asarray(s1_scores, dtype=np.float64)
            s1_min, s1_max = float(s1_np.min()), float(s1_np.max())
            s1_norm = (s1_np - s1_min) / (s1_max - s1_min) if s1_max > s1_min else np.ones_like(s1_np)

            q_min, q_max = float(q_sub.min()), float(q_sub.max())
            q_norm = (q_sub - q_min) / (q_max - q_min) if q_max > q_min else np.ones_like(q_sub)

            g_min, g_max = float(g_sub.min()), float(g_sub.max())
            g_norm = (g_sub - g_min) / (g_max - g_min) if g_max > g_min else np.ones_like(g_sub)

            s1_ranks = np.arange(1.0, float(n_c) + 1.0)
            q_ranks = np.argsort(np.argsort(-q_sub)).astype(np.float64) + 1.0
            g_ranks = np.argsort(np.argsort(-g_sub)).astype(np.float64) + 1.0

            # Scale-invariant Rank-RRF + raw-logit min-max tie-breaking across Stage-1, Gemma-Logit, and Qwen3-LogOdds
            rrf_blend = (
                0.38 / (12.0 + s1_ranks)
                + 0.40 / (12.0 + g_ranks)
                + 0.22 / (12.0 + q_ranks)
            )
            logit_blend = 0.34 * s1_norm + 0.42 * g_norm + 0.24 * q_norm
            blended = 1.0 + 0.70 * rrf_blend + 0.30 * (0.08 * logit_blend)

            order = np.argsort(-blended)
            ordered_pairs = [(c_ids[int(i)], float(blended[int(i)])) for i in order]
            for tail_rank, (tail_did, _) in enumerate(tail_cands, start=1):
                ordered_pairs.append((tail_did, 0.5 / float(tail_rank + 1)))
            task_scores[qid] = {did: sc for did, sc in ordered_pairs}

            if args.enable_listwise:
                top12 = [(did, cinfo["norm_docs"][doc_id_to_idx[did]]) for did, _ in ordered_pairs[:12]]
                raw_q = meta["raw_queries"][qi].strip()
                r_hint = reason_src.get(qid, "").strip()
                lw_q = "{} (Context: {})".format(raw_q[:650], r_hint[:350]) if r_hint else raw_q[:900]
                listwise_jobs.append((tr, task, qid, lw_q, top12))

        stage2_final[(tr, task)] = mask_excluded_scores(task_scores, meta["excluded_ids"], top_k=100)

    # 8. Optional Stage-3 Listwise Comparative Reasoning Reranking (ReasonRank-7B)
    stage3_final: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    if args.enable_listwise and listwise_jobs:
        lw_orders = run_listwise_reasonrank_top20(
            model_name=args.listwise_model,
            device=device,
            listwise_jobs=listwise_jobs,
            cache_path=cache_dir / "reasonrank_top12.json",
            batch_size=2,
            max_new_tokens=512,
        )
        for tr in active_tracks:
            for task in args.tasks:
                meta = task_meta[task]
                s2_map = stage2_final[(tr, task)]
                s3_map: Dict[str, Dict[str, float]] = {}
                for qid in meta["qids"]:
                    s2_ranked = sorted(s2_map.get(qid, {}).items(), key=lambda kv: -kv[1])
                    lw_top = lw_orders.get((tr, task, qid), [])
                    s2_rank_idx = {did: r for r, (did, _) in enumerate(s2_ranked, start=1)}
                    lw_rank_idx = {did: r for r, did in enumerate(lw_top, start=1)}
                    combined_doc_scores: Dict[str, float] = {}
                    for did, orig_sc in s2_ranked:
                        r2 = s2_rank_idx[did]
                        if did in lw_rank_idx:
                            r3 = lw_rank_idx[did]
                            combined_doc_scores[did] = 0.55 * (1.0 / (20.0 + r3)) + 0.45 * (1.0 / (20.0 + r2))
                        else:
                            combined_doc_scores[did] = 1.0 / (60.0 + r2) * 0.1
                    s3_map[qid] = combined_doc_scores
                stage3_final[(tr, task)] = mask_excluded_scores(s3_map, meta["excluded_ids"], top_k=100)

    # 9. Compute Official BRIGHT Metrics, Bootstrap 95% CIs, and Export Submission Files
    summary: Dict[str, Any] = {
        "benchmark": "BRIGHT",
        "device": device,
        "tasks_evaluated": args.tasks,
        "conditions": {},
        "telemetry": telemetry.snapshot(),
    }

    condition_specs: List[Tuple[str, Dict[str, Dict[str, Dict[str, float]]]]] = [
        (
            "A_baseline_bm25",
            {
                t: mask_excluded_scores(
                    {q: dict(bm25_raw_runs[t][q][:100]) for q in task_meta[t]["qids"]},
                    task_meta[t]["excluded_ids"],
                )
                for t in args.tasks
            },
        ),
        (
            "B_piragi_raw_hybrid_sota",
            {
                t: mask_excluded_scores(
                    {q: dict(stage1_raw_hybrid[t][q][:100]) for q in task_meta[t]["qids"]},
                    task_meta[t]["excluded_ids"],
                )
                for t in args.tasks
            },
        ),
    ]

    for tr in active_tracks:
        condition_specs.append(
            (
                "C_{}_stage1_hybrid_prf".format(tr),
                {
                    t: mask_excluded_scores(
                        {q: dict(stage1_ttr_hybrid[(tr, t)][q][:100]) for q in task_meta[t]["qids"]},
                        task_meta[t]["excluded_ids"],
                    )
                    for t in args.tasks
                },
            )
        )
        condition_specs.append(
            (
                "D_{}_stage2_dual_causal_rerank".format(tr),
                {t: stage2_final[(tr, t)] for t in args.tasks},
            )
        )
        if args.enable_listwise and (tr, args.tasks[0]) in stage3_final:
            condition_specs.append(
                (
                    "E_{}_stage3_listwise_reasonrank".format(tr),
                    {t: stage3_final[(tr, t)] for t in args.tasks},
                )
            )

    print("\n" + "=" * 92, flush=True)
    print("OFFICIAL BRIGHT BENCHMARK SUMMARY (Macro-Average NDCG@10 Across {} Splits)".format(len(args.tasks)), flush=True)
    print("=" * 92, flush=True)

    for cond_name, per_task_runs in condition_specs:
        per_task_metrics: Dict[str, Dict[str, float]] = {}
        all_query_ndcg10: List[float] = []
        cond_dir = out_root / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        for task in args.tasks:
            meta = task_meta[task]
            scores_dict = per_task_runs[task]
            # Verify official BRIGHT zero-leakage invariant
            for qid in meta["qids"]:
                for eid in meta["excluded_ids"][qid]:
                    assert eid not in scores_dict.get(qid, {}), "Leakage on {}:{}".format(task, qid)

            m_dict, q_ndcgs = compute_bright_metrics(scores_dict, meta["qrels"])
            per_task_metrics[task] = m_dict
            all_query_ndcg10.extend(q_ndcgs.values())

            task_out_dir = cond_dir / task
            task_out_dir.mkdir(parents=True, exist_ok=True)
            with open(task_out_dir / "score.json", "w", encoding="utf-8") as handle:
                json.dump(scores_dict, handle, indent=2)
            with open(task_out_dir / "results.json", "w", encoding="utf-8") as handle:
                json.dump(m_dict, handle, indent=2)

        macro_ndcg10 = round(
            float(np.mean([per_task_metrics[t]["NDCG@10"] for t in args.tasks])) * 100.0, 2
        )
        macro_recall10 = round(
            float(np.mean([per_task_metrics[t]["Recall@10"] for t in args.tasks])) * 100.0, 2
        )
        macro_recall100 = round(
            float(np.mean([per_task_metrics[t]["Recall@100"] for t in args.tasks])) * 100.0, 2
        )
        macro_mrr = round(
            float(np.mean([per_task_metrics[t]["MRR"] for t in args.tasks])) * 100.0, 2
        )
        ci_low, ci_high = bootstrap_ci_95([v * 100.0 for v in all_query_ndcg10], n_boot=1000)

        summary["conditions"][cond_name] = {
            "macro_ndcg_at_10_pct": macro_ndcg10,
            "macro_recall_at_10_pct": macro_recall10,
            "macro_recall_at_100_pct": macro_recall100,
            "macro_mrr_pct": macro_mrr,
            "micro_ndcg_at_10_ci95_pct": [round(ci_low, 2), round(ci_high, 2)],
            "per_task": {
                t: {
                    "ndcg_at_10_pct": round(per_task_metrics[t]["NDCG@10"] * 100.0, 2),
                    "recall_at_10_pct": round(per_task_metrics[t]["Recall@10"] * 100.0, 2),
                    "recall_at_100_pct": round(per_task_metrics[t]["Recall@100"] * 100.0, 2),
                    "raw": per_task_metrics[t],
                }
                for t in args.tasks
            },
        }
        print(
            "{:<42s} | NDCG@10={:6.2f}% (95% CI [{:5.2f}, {:5.2f}]) | R@10={:6.2f}% | R@100={:6.2f}%".format(
                cond_name, macro_ndcg10, ci_low, ci_high, macro_recall10, macro_recall100
            ),
            flush=True,
        )

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("\n[BRIGHT] Saved full summary & CIs to {}".format(summary_path), flush=True)


if __name__ == "__main__":
    main()
