"""Official AIR-Bench 24.05 Long-Doc GPU evaluation and live Hugging Face leaderboard runner for piragi.

Evaluates piragi's SOTA Multi-Space Hybrid RRF:
  - Stage-1: Alibaba-NLP/gte-Qwen2-1.5B-instruct + BAAI/bge-large-en-v1.5 +
             mixedbread-ai/mxbai-embed-large-v1 + intfloat/e5-large-v2 +
             Piragi De-glued BM25 + Consensus Rocchio Pseudo-Relevance Feedback (PRF)
  - Stage-2: Length-Sorted LLM & Cross-Encoder Reranking on GCP NVIDIA L4 GPU
             (BAAI/bge-reranker-v2-gemma + BAAI/bge-reranker-v2-m3, top-150, max_length=512)
across Official Dev (4 subtasks) and Official Test (11 subtasks), and submits verified archives live.
"""

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import zipfile

from huggingface_hub import hf_hub_download
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        def _compat_get_usable_length(self: Any, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
            return int(self.get_seq_length(layer_idx or 0))

        setattr(DynamicCache, "get_usable_length", _compat_get_usable_length)
except Exception:
    pass

from piragi import HybridSearcher, get_telemetry, request_context
from piragi.hybrid_search import normalize_corpus_texts

BENCHMARK_VERSION = "AIR-Bench_24.05"
RETRIEVAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QWEN_QUERY_INSTRUCT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
GEMMA_RERANK_PROMPT = (
    "Given a query A and a passage B, determine whether the passage contains an answer "
    "to the query by providing a prediction of either 'Yes' or 'No'."
)

DEV_TASKS: List[Tuple[str, str]] = [
    ("healthcare", "pubmed_100K-200K_3"),
    ("arxiv", "llama2"),
    ("law", "lex_files_300K-400K"),
    ("book", "a-brief-history-of-time_stephen-hawking"),
]

TEST_TASKS: List[Tuple[str, str]] = [
    ("healthcare", "pubmed_100K-200K_1"),
    ("healthcare", "pubmed_100K-200K_2"),
    ("healthcare", "pubmed_40K-50K_5-merged"),
    ("healthcare", "pubmed_30K-40K_10-merged"),
    ("arxiv", "gpt3"),
    ("arxiv", "gemini"),
    ("arxiv", "llm-survey"),
    ("law", "lex_files_400K-500K"),
    ("law", "lex_files_500K-600K"),
    ("law", "lex_files_600K-700K"),
    ("book", "origin-of-species_darwin"),
]


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _preferred_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _free_device_memory(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_subtask_data(
    domain: str, dataset_name: str, split: str
) -> Tuple[Dict[str, str], Dict[str, str], Optional[Dict[str, Dict[str, int]]]]:
    """Download/load official AIR-Bench 24.05 corpus, queries, and (for dev) qrels."""
    repo_id = "AIR-Bench/long-doc_{}_en".format(domain)
    corpus_path = hf_hub_download(
        repo_id=repo_id,
        filename="{}/{}/corpus.jsonl".format(BENCHMARK_VERSION, dataset_name),
        repo_type="dataset",
    )
    queries_path = hf_hub_download(
        repo_id=repo_id,
        filename="{}/{}/{}_queries.jsonl".format(BENCHMARK_VERSION, dataset_name, split),
        repo_type="dataset",
    )

    corpus = {row["id"]: row["text"] for row in _load_jsonl(corpus_path)}
    queries = {row["id"]: row["text"] for row in _load_jsonl(queries_path)}

    qrels: Optional[Dict[str, Dict[str, int]]] = None
    if split == "dev":
        qrels_repo = "AIR-Bench/qrels-long-doc_{}_en-dev".format(domain)
        qrels_path = hf_hub_download(
            repo_id=qrels_repo,
            filename="{}/{}/dev_qrels.jsonl".format(BENCHMARK_VERSION, dataset_name),
            repo_type="dataset",
        )
        qrels_map: Dict[str, Dict[str, int]] = defaultdict(dict)
        for row in _load_jsonl(qrels_path):
            rel = int(row["relevance"])
            if rel > 0:
                qrels_map[row["qid"]][row["docid"]] = rel
        qrels = dict(qrels_map)

    return corpus, queries, qrels


def compute_ir_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int, ...] = (1, 3, 5, 10, 50, 100, 1000),
) -> Dict[str, float]:
    """Compute exact TREC Recall@k, nDCG@k, and MRR@k matching AIR-Bench Evaluator."""
    recall_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    mrr_sums = {k: 0.0 for k in k_values}
    valid_q = 0

    max_k = max(k_values)
    for qid, rel_docs in qrels.items():
        if not rel_docs:
            continue
        valid_q += 1
        ranked = sorted(results.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)[:max_k]
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        ideal_rels = sorted(rel_docs.values(), reverse=True)
        total_rel = float(sum(1 for r in rel_docs.values() if r > 0))

        for k in k_values:
            top_ids = ranked_doc_ids[:k]
            hits = sum(1 for did in top_ids if rel_docs.get(did, 0) > 0)
            recall_sums[k] += float(hits) / total_rel

            dcg = 0.0
            for rank_idx, did in enumerate(top_ids, start=1):
                rel = rel_docs.get(did, 0)
                if rel > 0:
                    dcg += (float((1 << rel) - 1)) / math.log2(rank_idx + 1.0)

            idcg = 0.0
            for rank_idx, rel in enumerate(ideal_rels[:k], start=1):
                idcg += (float((1 << rel) - 1)) / math.log2(rank_idx + 1.0)
            ndcg_sums[k] += (dcg / idcg) if idcg > 0 else 0.0

            rr = 0.0
            for rank_idx, did in enumerate(top_ids, start=1):
                if rel_docs.get(did, 0) > 0:
                    rr = 1.0 / float(rank_idx)
                    break
            mrr_sums[k] += rr

    denom = float(max(1, valid_q))
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics["recall_at_{}".format(k)] = round(recall_sums[k] / denom, 5)
        metrics["ndcg_at_{}".format(k)] = round(ndcg_sums[k] / denom, 5)
        metrics["mrr_at_{}".format(k)] = round(mrr_sums[k] / denom, 5)
    return metrics


def _score_causal_llm_reranker(
    model_name: str,
    device: str,
    pairs_by_subtask: List[Tuple[str, str, List[Tuple[str, str]]]],
    max_length: int = 512,
    batch_size: int = 48,
    telemetry: Any = None,
) -> List[np.ndarray]:
    """Score (query, passage) pairs with causal LLM reranker (e.g. BAAI/bge-reranker-v2-gemma)."""
    dtype = _preferred_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    yes_loc = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    sep_ids = tokenizer("\n", return_tensors=None, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(GEMMA_RERANK_PROMPT, return_tensors=None, add_special_tokens=False)["input_ids"]
    bos_list = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    out_arrays: List[np.ndarray] = []
    supports_logits_to_keep = True

    for domain, ds_name, flat_pairs in pairs_by_subtask:
        t0 = time.perf_counter()
        length_order = sorted(range(len(flat_pairs)), key=lambda idx: len(flat_pairs[idx][1]))
        sorted_pairs = [flat_pairs[idx] for idx in length_order]

        # Pre-tokenize sorted pairs once on CPU
        encoded_items: List[Dict[str, Any]] = []
        for q_str, p_str in sorted_pairs:
            q_ids = tokenizer(
                "A: {}".format(q_str),
                return_tensors=None,
                add_special_tokens=False,
                max_length=(max_length * 3) // 4,
                truncation=True,
            )["input_ids"]
            p_ids = tokenizer(
                "B: {}".format(p_str),
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )["input_ids"]
            budget = max(64, max_length - len(bos_list) - len(q_ids) - len(sep_ids))
            merged_ids = bos_list + q_ids + sep_ids + p_ids[:budget] + sep_ids + prompt_ids
            encoded_items.append({
                "input_ids": merged_ids,
                "attention_mask": [1] * len(merged_ids),
            })

        sorted_scores = np.empty(len(sorted_pairs), dtype=np.float64)
        span_ctx = (
            telemetry.span("rerank.cross_encoder", operation="rerank", provider="llm_reranker", candidateCount=len(flat_pairs))
            if telemetry is not None
            else request_context(entry_point="cli_eval")
        )
        with span_ctx, torch.inference_mode():
            for start in range(0, len(encoded_items), batch_size):
                batch_slice = encoded_items[start : start + batch_size]
                padded = tokenizer.pad(
                    batch_slice,
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                padded = {k: v.to(device, non_blocking=True) for k, v in padded.items()}
                if supports_logits_to_keep:
                    try:
                        logits = model(**padded, logits_to_keep=1, use_cache=False).logits[:, -1, yes_loc]
                    except TypeError:
                        supports_logits_to_keep = False
                        logits = model(**padded, use_cache=False).logits[:, -1, yes_loc]
                else:
                    logits = model(**padded, use_cache=False).logits[:, -1, yes_loc]
                sorted_scores[start : start + len(batch_slice)] = logits.float().cpu().numpy()

        restored = np.empty(len(flat_pairs), dtype=np.float64)
        for s_pos, orig_idx in enumerate(length_order):
            restored[orig_idx] = sorted_scores[s_pos]
        out_arrays.append(restored)
        print(
            "     [{}/{}] {} pairs scored by {} in {:.1f}s".format(
                domain, ds_name, len(flat_pairs), model_name, time.perf_counter() - t0
            )
        )

    del model
    _free_device_memory(device)
    return out_arrays


class PiragiAirBenchEngine:
    """GPU-accelerated Model-Outer piragi retrieval + reranking engine for SOTA Long-Doc IR."""

    def __init__(
        self,
        dense_models: List[str],
        reranker_models: List[str],
        device: Optional[str] = None,
        rerank_top_n: int = 150,
        max_length: int = 512,
        rrf_k: int = 60,
        bm25_weight: float = 0.90,
        prf_alpha: float = 0.82,
        prf_top_m: int = 3,
    ):
        self.device = device or _select_device()
        self.dense_model_names = dense_models
        self.reranker_model_names = reranker_models
        self.rerank_top_n = rerank_top_n
        self.max_length = max_length
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.prf_alpha = prf_alpha
        self.prf_top_m = prf_top_m
        self.telemetry = get_telemetry()

    def _encode_queries(self, m_name: str, model: SentenceTransformer, query_texts: List[str], batch_size: int) -> np.ndarray:
        lower_name = m_name.lower()
        if "qwen2" in lower_name:
            if getattr(model, "prompts", None) and "query" in model.prompts:
                return model.encode(
                    query_texts,
                    prompt_name="query",
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)
            prefixed = [QWEN_QUERY_INSTRUCT + q for q in query_texts]
        elif "bge-" in lower_name or "mxbai-embed" in lower_name or "arctic-embed" in lower_name:
            prefixed = [RETRIEVAL_QUERY_PREFIX + q for q in query_texts]
        elif "e5-" in lower_name:
            prefixed = ["query: " + q for q in query_texts]
        else:
            prefixed = query_texts
        return model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def _encode_passages(self, m_name: str, model: SentenceTransformer, passage_texts: List[str], batch_size: int) -> np.ndarray:
        lower_name = m_name.lower()
        if "e5-" in lower_name:
            inputs = ["passage: " + p for p in passage_texts]
        else:
            inputs = passage_texts
        return model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def _compute_rrf_pool(
        self,
        sim_list: List[np.ndarray],
        dense_weights: List[float],
        bm25_pos_orders: List[np.ndarray],
        num_docs: int,
        out_k: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        top_indices_all: List[np.ndarray] = []
        top_scores_all: List[np.ndarray] = []
        cand_k = min(num_docs, max(out_k, self.rerank_top_n))
        num_queries = len(bm25_pos_orders)

        for qi in range(num_queries):
            rrf = np.zeros(num_docs, dtype=np.float64)
            for m_idx, sim_mat in enumerate(sim_list):
                w = dense_weights[m_idx]
                row = sim_mat[qi]
                order = np.argsort(-row)
                ranks = np.empty(num_docs, dtype=np.float64)
                ranks[order] = np.arange(1.0, num_docs + 1.0)
                rrf += w / (float(self.rrf_k) + ranks)
                rrf += 0.002 * w * np.clip(row.astype(np.float64), 0.0, 1.0)

            pos_order = bm25_pos_orders[qi]
            if len(pos_order) > 0:
                bm25_ranks = np.arange(1.0, len(pos_order) + 1.0)
                rrf[pos_order] += self.bm25_weight / (float(self.rrf_k) + bm25_ranks)

            top_idx = np.argsort(-rrf)[:cand_k]
            top_sc = rrf[top_idx]
            top_indices_all.append(top_idx)
            top_scores_all.append(top_sc)

        return top_indices_all, top_scores_all

    def evaluate_task_batch(
        self,
        tasks: List[Tuple[str, str]],
        split: str,
        top_k_output: int = 1000,
        use_prf: bool = True,
        fusion_weight: float = 0.15,
        reranker_weights: Optional[List[float]] = None,
        collect_ablations: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run Model-Outer, Subtask-Inner execution across all tasks on GPU."""
        t_batch_start = time.perf_counter()
        subtasks: List[Dict[str, Any]] = []
        dtype = _preferred_dtype(self.device)

        print("\n[Step 1/4] Loading corpora & running Piragi PDF/EPUB de-gluing + BM25 indexing ({} subtasks)...".format(len(tasks)))
        for domain, dataset_name in tasks:
            corpus, queries, qrels = load_subtask_data(domain, dataset_name, split=split)
            doc_ids = list(corpus.keys())
            raw_doc_texts = [corpus[did] for did in doc_ids]
            norm_doc_texts = normalize_corpus_texts(raw_doc_texts)
            q_ids = list(queries.keys())
            raw_q_texts = [queries[qid] for qid in q_ids]
            norm_q_texts = [q.translate({0xFB01: "fi", 0xFB02: "fl", 0x00A0: " "}) for q in raw_q_texts]

            with self.telemetry.span("search.hybrid", operation="hybrid_fusion", provider="bm25", chunkCount=len(doc_ids)):
                hs = HybridSearcher(use_rrf=True, rrf_k=self.rrf_k, telemetry=self.telemetry)
                tokenized_docs = hs._tokenize_corpus_texts(norm_doc_texts)
                bm25 = BM25Okapi(tokenized_docs)
                if getattr(bm25, "average_idf", 0.0) > 0:
                    floor_idf = bm25.epsilon * bm25.average_idf
                    for tok, val in list(bm25.idf.items()):
                        if val <= 0.0:
                            bm25.idf[tok] = floor_idf

                bm25_pos_orders: List[np.ndarray] = []
                for q_txt in norm_q_texts:
                    q_toks = hs._tokenize(q_txt)
                    b_scores = bm25.get_scores(q_toks)
                    pos = np.nonzero(b_scores > 0)[0]
                    if len(pos) > 0:
                        bm25_pos_orders.append(pos[np.argsort(-b_scores[pos])])
                    else:
                        bm25_pos_orders.append(np.empty(0, dtype=np.int64))

            subtasks.append({
                "domain": domain,
                "dataset_name": dataset_name,
                "doc_ids": doc_ids,
                "norm_doc_texts": norm_doc_texts,
                "q_ids": q_ids,
                "norm_q_texts": norm_q_texts,
                "qrels": qrels,
                "bm25_pos_orders": bm25_pos_orders,
                "d_embs": [],
                "q_embs": [],
                "sim_s1a": [],
                "ce_logits_list": [],
            })

        print("\n[Step 2/4] Encoding dense bi-encoders sequentially on {} ({})...".format(self.device, dtype))
        dense_weights = [1.35 if "qwen2" in m.lower() else 1.0 for m in self.dense_model_names]
        for m_idx, m_name in enumerate(self.dense_model_names):
            t_m0 = time.perf_counter()
            print("  -> Bi-Encoder [{}/{}]: {}".format(m_idx + 1, len(self.dense_model_names), m_name))
            is_llm_emb = "qwen2" in m_name.lower()
            enc_batch = 32 if is_llm_emb else 128
            st_model = SentenceTransformer(
                m_name,
                device=self.device,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": dtype},
            )
            st_model.max_seq_length = min(getattr(st_model, "max_seq_length", 512), self.max_length)

            for st_item in subtasks:
                with self.telemetry.span(
                    "embed.chunks",
                    operation="embed",
                    provider="local",
                    chunkCount=len(st_item["doc_ids"]),
                ):
                    d_emb = self._encode_passages(m_name, st_model, st_item["norm_doc_texts"], batch_size=enc_batch)
                    q_emb = self._encode_queries(m_name, st_model, st_item["norm_q_texts"], batch_size=enc_batch)
                sim_mat = np.matmul(q_emb, d_emb.T)
                st_item["d_embs"].append(d_emb)
                st_item["q_embs"].append(q_emb)
                st_item["sim_s1a"].append(sim_mat)

            del st_model
            _free_device_memory(self.device)
            print("     completed in {:.1f}s".format(time.perf_counter() - t_m0))

        print("\n[Step 3/4] Computing Multi-Space Hybrid RRF + Consensus Rocchio PRF...")
        prf_rank_weights = np.array(
            [1.0 / float(r + 1) for r in range(self.prf_top_m)],
            dtype=np.float32,
        )
        prf_rank_weights = prf_rank_weights / float(np.sum(prf_rank_weights))

        for st_item in subtasks:
            num_docs = len(st_item["doc_ids"])
            out_k = min(top_k_output, num_docs)
            s1a_idx, s1a_sc = self._compute_rrf_pool(
                sim_list=st_item["sim_s1a"],
                dense_weights=dense_weights,
                bm25_pos_orders=st_item["bm25_pos_orders"],
                num_docs=num_docs,
                out_k=out_k,
            )
            st_item["s1a_top_indices"] = s1a_idx
            st_item["s1a_top_scores"] = s1a_sc

            sim_prf_list: List[np.ndarray] = []
            top_m = min(self.prf_top_m, num_docs)
            consensus_top_m = np.stack([row[:top_m] for row in s1a_idx], axis=0)

            for m_idx in range(len(self.dense_model_names)):
                d_emb = st_item["d_embs"][m_idx]
                q_emb = st_item["q_embs"][m_idx]
                top_doc_vecs = d_emb[consensus_top_m]
                centroid = np.tensordot(top_doc_vecs, prf_rank_weights[:top_m], axes=([1], [0]))
                c_norms = np.linalg.norm(centroid, axis=1, keepdims=True)
                centroid = np.divide(centroid, np.maximum(c_norms, 1e-8))
                q_prf = self.prf_alpha * q_emb + (1.0 - self.prf_alpha) * centroid
                q_norms = np.linalg.norm(q_prf, axis=1, keepdims=True)
                q_prf = np.divide(q_prf, np.maximum(q_norms, 1e-8))
                sim_prf_list.append(np.matmul(q_prf, d_emb.T))

            st_item["d_embs"].clear()
            st_item["q_embs"].clear()
            st_item["sim_s1a"].clear()

            s1b_idx, s1b_sc = self._compute_rrf_pool(
                sim_list=sim_prf_list,
                dense_weights=dense_weights,
                bm25_pos_orders=st_item["bm25_pos_orders"],
                num_docs=num_docs,
                out_k=out_k,
            )
            st_item["s1b_top_indices"] = s1b_idx
            st_item["s1b_top_scores"] = s1b_sc

            active_idx = s1b_idx if use_prf else s1a_idx
            active_sc = s1b_sc if use_prf else s1a_sc
            st_item["stage1_top_indices"] = active_idx
            st_item["stage1_top_scores"] = active_sc

        gc.collect()

        print("\n[Step 4/4] Running Length-Sorted GPU Reranking (top-{}, max_len={})...".format(
            self.rerank_top_n, self.max_length
        ))
        for r_idx, r_name in enumerate(self.reranker_model_names):
            t_r0 = time.perf_counter()
            print("  -> Reranker [{}/{}]: {}".format(r_idx + 1, len(self.reranker_model_names), r_name))

            pairs_by_subtask: List[Tuple[str, str, List[Tuple[str, str]]]] = []
            for st_item in subtasks:
                num_docs = len(st_item["doc_ids"])
                rerank_n = min(self.rerank_top_n, num_docs)
                q_texts = st_item["norm_q_texts"]
                d_texts = st_item["norm_doc_texts"]
                s1_indices = st_item["stage1_top_indices"]
                flat_pairs = [
                    (q_str, d_texts[int(didx)])
                    for qi, q_str in enumerate(q_texts)
                    for didx in s1_indices[qi][:rerank_n]
                ]
                pairs_by_subtask.append((st_item["domain"], st_item["dataset_name"], flat_pairs))

            if "gemma" in r_name.lower():
                scored_arrays = _score_causal_llm_reranker(
                    model_name=r_name,
                    device=self.device,
                    pairs_by_subtask=pairs_by_subtask,
                    max_length=self.max_length,
                    batch_size=48 if self.device == "cuda" else 16,
                    telemetry=self.telemetry,
                )
                for st_item, raw_logits in zip(subtasks, scored_arrays):
                    rerank_n = min(self.rerank_top_n, len(st_item["doc_ids"]))
                    st_item["ce_logits_list"].append(raw_logits.reshape(len(st_item["q_ids"]), rerank_n))
            else:
                ce = CrossEncoder(
                    r_name,
                    device=self.device,
                    max_length=self.max_length,
                    trust_remote_code=True,
                    model_kwargs={"torch_dtype": dtype},
                )
                ce_batch_size = 128 if self.device == "cuda" else 64
                for st_item, (domain, ds_name, flat_pairs) in zip(subtasks, pairs_by_subtask):
                    t_sub0 = time.perf_counter()
                    with self.telemetry.span(
                        "rerank.cross_encoder",
                        operation="rerank",
                        provider="cross_encoder",
                        candidateCount=len(flat_pairs),
                    ):
                        length_order = sorted(range(len(flat_pairs)), key=lambda idx: len(flat_pairs[idx][1]))
                        sorted_pairs = [flat_pairs[idx] for idx in length_order]
                        sorted_logits = np.asarray(
                            ce.predict(sorted_pairs, batch_size=ce_batch_size, show_progress_bar=False),
                            dtype=np.float64,
                        )
                        raw_logits = np.empty(len(flat_pairs), dtype=np.float64)
                        for s_pos, orig_idx in enumerate(length_order):
                            raw_logits[orig_idx] = sorted_logits[s_pos]

                    rerank_n = min(self.rerank_top_n, len(st_item["doc_ids"]))
                    st_item["ce_logits_list"].append(raw_logits.reshape(len(st_item["q_ids"]), rerank_n))
                    print(
                        "     [{}/{}] {} pairs scored in {:.1f}s".format(
                            domain, ds_name, len(flat_pairs), time.perf_counter() - t_sub0
                        )
                    )
                del ce
                _free_device_memory(self.device)

            print("     completed {} in {:.1f}s".format(r_name, time.perf_counter() - t_r0))

        if reranker_weights is None:
            reranker_weights = [
                1.35 if "gemma" in r.lower() else (1.0 if "v2-m3" in r.lower() else 0.45)
                for r in self.reranker_model_names
            ]

        outputs: List[Dict[str, Any]] = []
        for st_item in subtasks:
            doc_ids = st_item["doc_ids"]
            q_ids = st_item["q_ids"]
            num_docs = len(doc_ids)
            out_k = min(top_k_output, num_docs)
            rerank_n = min(self.rerank_top_n, num_docs)

            def _to_result_dict(idx_list: List[np.ndarray], sc_list: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
                res: Dict[str, Dict[str, float]] = {}
                for qi, qid in enumerate(q_ids):
                    res[qid] = {
                        doc_ids[int(di)]: round(float(sc), 6)
                        for di, sc in zip(idx_list[qi][:out_k], sc_list[qi][:out_k])
                    }
                return res

            stage1_results = _to_result_dict(st_item["stage1_top_indices"], st_item["stage1_top_scores"])

            def _build_fused(
                s1_weight: float,
                ce_w_vec: List[float],
                active_ce_indices: Optional[List[int]] = None,
            ) -> Dict[str, Dict[str, float]]:
                if active_ce_indices is None:
                    active_ce_indices = list(range(len(st_item["ce_logits_list"])))
                fused_out: Dict[str, Dict[str, float]] = {}
                w_sum = sum(ce_w_vec[i] for i in active_ce_indices) or 1.0

                for qi, qid in enumerate(q_ids):
                    top_idx = st_item["stage1_top_indices"][qi]
                    s1_scores = st_item["stage1_top_scores"][qi]

                    s1_head = s1_scores[:rerank_n]
                    s1_min, s1_max = float(np.min(s1_head)), float(np.max(s1_head))
                    s1_norm = (s1_head - s1_min) / (s1_max - s1_min) if s1_max > s1_min else np.ones_like(s1_head)

                    ce_blend = np.zeros(rerank_n, dtype=np.float64)
                    for c_i in active_ce_indices:
                        logits = np.clip(st_item["ce_logits_list"][c_i][qi], -50.0, 50.0)
                        probs = 1.0 / (1.0 + np.exp(-logits))
                        p_min, p_max = float(np.min(probs)), float(np.max(probs))
                        p_norm = (probs - p_min) / (p_max - p_min) if p_max > p_min else probs

                        l_std = float(np.std(logits))
                        if l_std > 1e-6:
                            z_norm = (logits - float(np.mean(logits))) / l_std
                            z_min, z_max = float(np.min(z_norm)), float(np.max(z_norm))
                            z_scaled = (z_norm - z_min) / (z_max - z_min) if z_max > z_min else p_norm
                            ce_unit = 0.50 * p_norm + 0.50 * z_scaled
                        else:
                            ce_unit = p_norm
                        ce_blend += (ce_w_vec[c_i] / w_sum) * ce_unit

                    combined_head = (1.0 - s1_weight) * ce_blend + s1_weight * s1_norm
                    head_order = np.argsort(-combined_head)
                    q_map: Dict[str, float] = {}
                    for pos in head_order:
                        doc_id = doc_ids[int(top_idx[pos])]
                        q_map[doc_id] = round(2.0 + float(combined_head[pos]), 6)

                    tail_idx = top_idx[rerank_n:out_k]
                    tail_sc = s1_scores[rerank_n:out_k]
                    for didx, sc in zip(tail_idx, tail_sc):
                        q_map[doc_ids[int(didx)]] = round(float(sc), 6)
                    fused_out[qid] = q_map
                return fused_out

            per_reranker_results: Dict[int, Dict[str, Dict[str, float]]] = {}
            for c_idx in range(len(st_item["ce_logits_list"])):
                per_reranker_results[c_idx] = _build_fused(
                    fusion_weight, [1.0 for _ in st_item["ce_logits_list"]], active_ce_indices=[c_idx]
                )

            item_out: Dict[str, Any] = {
                "domain": st_item["domain"],
                "dataset_name": st_item["dataset_name"],
                "num_docs": num_docs,
                "num_queries": len(q_ids),
                "qrels": st_item["qrels"],
                "stage1": stage1_results,
                "reranked": _build_fused(fusion_weight, reranker_weights),
                "per_reranker": per_reranker_results,
            }
            if collect_ablations:
                item_out["stage1_no_prf"] = _to_result_dict(st_item["s1a_top_indices"], st_item["s1a_top_scores"])
                item_out["stage1_prf"] = _to_result_dict(st_item["s1b_top_indices"], st_item["s1b_top_scores"])
                item_out["per_ce_sweeps"] = {
                    c_idx: {
                        w: _build_fused(w, [1.0 for _ in st_item["ce_logits_list"]], active_ce_indices=[c_idx])
                        for w in (0.0, 0.08, 0.12, 0.15, 0.20)
                    }
                    for c_idx in range(len(st_item["ce_logits_list"]))
                }
                if len(self.reranker_model_names) > 1:
                    candidate_blends: Dict[str, List[float]] = {
                        "full_blend": list(reranker_weights),
                    }
                    if len(self.reranker_model_names) >= 3:
                        candidate_blends["gemma_m3_only"] = [1.45, 1.0, 0.0]
                        candidate_blends["gemma_heavy"] = [1.80, 1.0, 0.25]
                    item_out["blend_sweeps"] = {
                        b_name: {
                            "weights": b_vec,
                            "sweep": {
                                w: _build_fused(w, b_vec)
                                for w in (0.0, 0.08, 0.12, 0.15, 0.20)
                            },
                        }
                        for b_name, b_vec in candidate_blends.items()
                    }
            outputs.append(item_out)

        print("\nCompleted batch ({} subtasks) in {:.1f}s".format(len(tasks), time.perf_counter() - t_batch_start))
        return outputs


def run_dev_benchmark(engine: PiragiAirBenchEngine, fusion_weight: float) -> Dict[str, Any]:
    """Evaluate piragi across all 4 official AIR-Bench_24.05 Long-Doc dev datasets."""
    print("\n=== Official AIR-Bench_24.05 Long-Doc Dev Evaluation ===")
    batch_out = engine.evaluate_task_batch(
        tasks=DEV_TASKS,
        split="dev",
        top_k_output=1000,
        use_prf=True,
        fusion_weight=fusion_weight,
        collect_ablations=True,
    )

    s1_noprf_all: List[Dict[str, float]] = []
    s1_prf_all: List[Dict[str, float]] = []
    rr_all: List[Dict[str, float]] = []
    per_ce_sweeps: Dict[int, Dict[float, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    blend_sweeps: Dict[str, Dict[float, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    blend_weight_map: Dict[str, List[float]] = {}

    for item in batch_out:
        qrels = item["qrels"]
        assert qrels is not None
        m_noprf = compute_ir_metrics(qrels, item["stage1_no_prf"])
        m_prf = compute_ir_metrics(qrels, item["stage1_prf"])
        m_rr = compute_ir_metrics(qrels, item["reranked"])
        s1_noprf_all.append(m_noprf)
        s1_prf_all.append(m_prf)
        rr_all.append(m_rr)

        for c_idx, w_map in item["per_ce_sweeps"].items():
            for w, res_w in w_map.items():
                per_ce_sweeps[c_idx][w].append(compute_ir_metrics(qrels, res_w))
        for b_name, b_info in item.get("blend_sweeps", {}).items():
            blend_weight_map[b_name] = b_info["weights"]
            for w, res_w in b_info["sweep"].items():
                blend_sweeps[b_name][w].append(compute_ir_metrics(qrels, res_w))

        print(
            "[{}/{}] Stage1(NoPRF->PRF) R@10={:.2f}%->{:.2f}% R@100={:.2f}%->{:.2f}% | Ensemble R@10={:.2f}% nDCG@10={:.2f}% MRR@10={:.2f}%".format(
                item["domain"],
                item["dataset_name"],
                m_noprf["recall_at_10"] * 100.0,
                m_prf["recall_at_10"] * 100.0,
                m_noprf["recall_at_100"] * 100.0,
                m_prf["recall_at_100"] * 100.0,
                m_rr["recall_at_10"] * 100.0,
                m_rr["ndcg_at_10"] * 100.0,
                m_rr["mrr_at_10"] * 100.0,
            )
        )

    avg_noprf_r10 = float(np.mean([m["recall_at_10"] for m in s1_noprf_all])) * 100.0
    avg_noprf_r100 = float(np.mean([m["recall_at_100"] for m in s1_noprf_all])) * 100.0
    avg_prf_r10 = float(np.mean([m["recall_at_10"] for m in s1_prf_all])) * 100.0
    avg_prf_ndcg10 = float(np.mean([m["ndcg_at_10"] for m in s1_prf_all])) * 100.0
    avg_prf_r100 = float(np.mean([m["recall_at_100"] for m in s1_prf_all])) * 100.0
    avg_rr_r10 = float(np.mean([m["recall_at_10"] for m in rr_all])) * 100.0
    avg_rr_ndcg10 = float(np.mean([m["ndcg_at_10"] for m in rr_all])) * 100.0
    avg_rr_mrr10 = float(np.mean([m["mrr_at_10"] for m in rr_all])) * 100.0

    prf_improved = (avg_prf_r10 + avg_prf_r100) >= (avg_noprf_r10 + avg_noprf_r100)

    print("\n--- Dev Macro-Average Summary (4 Official Domains) ---")
    print("Stage-1 Hybrid RRF (No PRF): Recall@10 = {:.2f}% | Recall@100 = {:.2f}%".format(avg_noprf_r10, avg_noprf_r100))
    print("Stage-1 Hybrid RRF (+ PRF) : Recall@10 = {:.2f}% | nDCG@10 = {:.2f}% | Recall@100 = {:.2f}%".format(
        avg_prf_r10, avg_prf_ndcg10, avg_prf_r100
    ))
    print("Piragi + Ensemble Reranker : Recall@10 = {:.2f}% | nDCG@10 = {:.2f}% | MRR@10 = {:.2f}%".format(
        avg_rr_r10, avg_rr_ndcg10, avg_rr_mrr10
    ))

    num_r = len(engine.reranker_model_names)
    default_r_weights = [
        1.35 if "gemma" in r.lower() else (1.0 if "v2-m3" in r.lower() else 0.45)
        for r in engine.reranker_model_names
    ]
    best_w = fusion_weight
    best_r_weights = list(default_r_weights)
    best_config_name = "default_ensemble"
    best_score = -1.0
    best_r10 = avg_rr_r10
    best_ndcg10 = avg_rr_ndcg10
    best_mrr10 = avg_rr_mrr10

    for c_idx, w_map in sorted(per_ce_sweeps.items()):
        r_label = engine.reranker_model_names[c_idx]
        one_hot = [1.0 if i == c_idx else 0.0 for i in range(num_r)]
        print("\n--- Individual Reranker [{}] Sweep (Dev Macro-Average) ---".format(r_label))
        for w in sorted(w_map.keys()):
            r10 = float(np.mean([m["recall_at_10"] for m in w_map[w]])) * 100.0
            ndcg10 = float(np.mean([m["ndcg_at_10"] for m in w_map[w]])) * 100.0
            mrr10 = float(np.mean([m["mrr_at_10"] for m in w_map[w]])) * 100.0
            print("  fusion_weight={:.2f} -> Recall@10={:.2f}% | nDCG@10={:.2f}% | MRR@10={:.2f}%".format(
                w, r10, ndcg10, mrr10
            ))
            if (r10 * 1.5 + ndcg10) > best_score:
                best_score = r10 * 1.5 + ndcg10
                best_w = w
                best_r_weights = one_hot
                best_config_name = "single:{}".format(r_label)
                best_r10, best_ndcg10, best_mrr10 = r10, ndcg10, mrr10

    for b_name, w_map in sorted(blend_sweeps.items()):
        print("\n--- Reranker Blend [{}] Sweep (Dev Macro-Average) ---".format(b_name))
        for w in sorted(w_map.keys()):
            r10 = float(np.mean([m["recall_at_10"] for m in w_map[w]])) * 100.0
            ndcg10 = float(np.mean([m["ndcg_at_10"] for m in w_map[w]])) * 100.0
            mrr10 = float(np.mean([m["mrr_at_10"] for m in w_map[w]])) * 100.0
            print("  [{}] fusion_weight={:.2f} -> Recall@10={:.2f}% | nDCG@10={:.2f}% | MRR@10={:.2f}%".format(
                b_name, w, r10, ndcg10, mrr10
            ))
            if (r10 * 1.5 + ndcg10) > best_score:
                best_score = r10 * 1.5 + ndcg10
                best_w = w
                best_r_weights = list(blend_weight_map[b_name])
                best_config_name = b_name
                best_r10, best_ndcg10, best_mrr10 = r10, ndcg10, mrr10

    print("\nSelected Best Dev Config: {} (fusion_weight={:.2f}, weights={}) -> R@10={:.2f}% | nDCG@10={:.2f}% | MRR@10={:.2f}%".format(
        best_config_name, best_w, best_r_weights, best_r10, best_ndcg10, best_mrr10
    ))

    return {
        "stage1_noprf_recall_at_10": round(avg_noprf_r10, 3),
        "stage1_prf_recall_at_10": round(avg_prf_r10, 3),
        "stage1_prf_ndcg_at_10": round(avg_prf_ndcg10, 3),
        "stage1_prf_recall_at_100": round(avg_prf_r100, 3),
        "reranked_recall_at_10": round(best_r10, 3),
        "reranked_ndcg_at_10": round(best_ndcg10, 3),
        "reranked_mrr_at_10": round(best_mrr10, 3),
        "best_config_name": best_config_name,
        "best_fusion_weight": best_w,
        "best_reranker_weights": best_r_weights,
        "use_prf": prf_improved,
    }


def _write_and_verify_zip(zip_path: Path, entries: Dict[str, str], expected_count: int) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, content in sorted(entries.items()):
            zf.writestr(arcname, content)
    _verify_airbench_zip(zip_path, expected_count=expected_count)


def generate_test_submission_zips(
    engine: PiragiAirBenchEngine,
    output_dir: Path,
    model_name: str,
    reranker_label: str,
    fusion_weight: float,
    use_prf: bool = True,
    reranker_weights: Optional[List[float]] = None,
) -> Dict[str, Path]:
    """Run all 11 official AIR-Bench_24.05 Long-Doc test splits and write verified .zip archives."""
    print("\n=== Generating Official AIR-Bench_24.05 Long-Doc Test Predictions (11 Subtasks) ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_out = engine.evaluate_task_batch(
        tasks=TEST_TASKS,
        split="test",
        top_k_output=1000,
        use_prf=use_prf,
        fusion_weight=fusion_weight,
        reranker_weights=reranker_weights,
        collect_ablations=False,
    )

    safe_label = reranker_label.replace("/", "_")
    ensemble_zip = output_dir / "piragi_airbench_2405_longdoc_{}.zip".format(safe_label)
    norerank_zip = output_dir / "piragi_airbench_2405_longdoc_noreranker_sota.zip"

    ensemble_entries: Dict[str, str] = {}
    norerank_entries: Dict[str, str] = {}
    individual_entries: Dict[int, Dict[str, str]] = defaultdict(dict)

    for item in batch_out:
        domain = item["domain"]
        dataset_name = item["dataset_name"]
        arcname = "long-doc/{}/en_{}_test.json".format(domain, dataset_name)

        ensemble_entries[arcname] = json.dumps({
            "model_name": model_name,
            "reranker_name": reranker_label,
            "benchmark_version": BENCHMARK_VERSION,
            "task_type": "long-doc",
            "domain": domain,
            "language": "en",
            "dataset_name": dataset_name,
            "split": "test",
            "search_results": item["reranked"],
        }, ensure_ascii=False)

        norerank_entries[arcname] = json.dumps({
            "model_name": model_name,
            "reranker_name": "NoReranker",
            "benchmark_version": BENCHMARK_VERSION,
            "task_type": "long-doc",
            "domain": domain,
            "language": "en",
            "dataset_name": dataset_name,
            "split": "test",
            "search_results": item["stage1"],
        }, ensure_ascii=False)

        for c_idx, c_res in item["per_reranker"].items():
            c_short = engine.reranker_model_names[c_idx].split("/")[-1]
            individual_entries[c_idx][arcname] = json.dumps({
                "model_name": model_name,
                "reranker_name": c_short,
                "benchmark_version": BENCHMARK_VERSION,
                "task_type": "long-doc",
                "domain": domain,
                "language": "en",
                "dataset_name": dataset_name,
                "split": "test",
                "search_results": c_res,
            }, ensure_ascii=False)

    written: Dict[str, Path] = {}
    _write_and_verify_zip(ensemble_zip, ensemble_entries, len(TEST_TASKS))
    written["primary"] = ensemble_zip

    _write_and_verify_zip(norerank_zip, norerank_entries, len(TEST_TASKS))
    written["noreranker"] = norerank_zip

    for c_idx, c_entries in individual_entries.items():
        c_short = engine.reranker_model_names[c_idx].split("/")[-1]
        c_zip = output_dir / "piragi_airbench_2405_longdoc_single_{}.zip".format(c_short)
        _write_and_verify_zip(c_zip, c_entries, len(TEST_TASKS))
        written["single_{}".format(c_short)] = c_zip

    return written


def _verify_airbench_zip(zip_path: Path, expected_count: int) -> None:
    """Strictly verify archive layout against AIR-Bench Evaluator.evaluate_results rules."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert len(names) == expected_count, "Expected {} files, got {}".format(expected_count, len(names))
        for name in names:
            assert not name.endswith(".DS_Store") and "__MACOSX" not in name
            parts = name.split("/")
            assert len(parts) == 3 and parts[0] == "long-doc", "Invalid path inside zip: {}".format(name)
            data = json.loads(zf.read(name).decode("utf-8"))
            assert data["benchmark_version"] == BENCHMARK_VERSION
            assert data["task_type"] == "long-doc"
            assert data["domain"] == parts[1]
            assert data["language"] == "en"
            assert data["split"] == "test"
            assert len(data["search_results"]) > 0

    md5_hex = hashlib.md5(zip_path.read_bytes()).hexdigest()
    size_mb = zip_path.stat().st_size / (1024.0 * 1024.0)
    print("Verified zip:", str(zip_path), "({:.2f} MB, md5={})".format(size_mb, md5_hex))


def submit_to_airbench_space(
    zip_path: Path,
    model_name: str,
    model_url: str,
    reranker_name: str,
    reranker_url: str,
) -> str:
    """Upload validated .zip and invoke submit_results on https://huggingface.co/spaces/AIR-Bench/leaderboard."""
    import httpx

    base_url = "https://air-bench-leaderboard.hf.space"
    token = None
    try:
        from huggingface_hub import get_token
        token = get_token()
    except Exception:
        token = os.environ.get("HF_TOKEN")

    headers = {}
    if token:
        headers["Authorization"] = "Bearer {}".format(token)

    with httpx.Client(timeout=180.0, follow_redirects=True, headers=headers) as client:
        info_resp = client.get("{}/info".format(base_url))
        api_prefix = ""
        if info_resp.status_code != 200:
            info_resp2 = client.get("{}/gradio_api/info".format(base_url))
            if info_resp2.status_code == 200:
                api_prefix = "/gradio_api"

        upload_url = "{}{}/upload".format(base_url, api_prefix)
        print("Uploading archive to:", upload_url)
        with open(zip_path, "rb") as f_handle:
            up_resp = client.post(
                upload_url,
                files={"files": (zip_path.name, f_handle, "application/zip")},
            )
        up_resp.raise_for_status()
        uploaded_paths = up_resp.json()
        remote_path = uploaded_paths[0]
        print("Uploaded remote path:", remote_path)

        file_data = {
            "path": remote_path,
            "orig_name": zip_path.name,
            "size": zip_path.stat().st_size,
            "mime_type": "application/zip",
            "meta": {"_type": "gradio.FileData"},
        }

        cfg_resp = client.get("{}/config".format(base_url))
        cfg_resp.raise_for_status()
        cfg = cfg_resp.json()
        upload_fn_index = None
        submit_fn_index = None
        for dep_idx, dep in enumerate(cfg.get("dependencies", [])):
            api_name = dep.get("api_name")
            if api_name == "upload_file":
                upload_fn_index = dep.get("id") if dep.get("id") is not None else dep_idx
            elif api_name == "submit_results":
                submit_fn_index = dep.get("id") if dep.get("id") is not None else dep_idx
        if submit_fn_index is None:
            raise RuntimeError("Could not locate submit_results dependency in Gradio config")

        def _call_gradio_fn(fn_idx: int, data_args: List[Any]) -> Any:
            session_hash = hashlib.md5(os.urandom(16)).hexdigest()[:11]
            join_url = "{}{}/queue/join".format(base_url, api_prefix)
            payload = {
                "data": data_args,
                "fn_index": fn_idx,
                "session_hash": session_hash,
            }
            join_resp = client.post(join_url, json=payload)
            join_resp.raise_for_status()
            data_url = "{}{}/queue/data?session_hash={}".format(base_url, api_prefix, session_hash)
            with client.stream("GET", data_url) as stream:
                for raw_line in stream.iter_lines():
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    evt = json.loads(raw_line[5:].strip())
                    msg = evt.get("msg")
                    if msg == "process_completed":
                        if not evt.get("success", True):
                            raise RuntimeError("Gradio call failed: {}".format(evt))
                        output_data = evt.get("output", {}).get("data", [None])
                        return output_data[0]
                    if msg == "queue_full":
                        raise RuntimeError("AIR-Bench Gradio queue is full")
            raise RuntimeError("Gradio SSE stream ended before process_completed")

        if upload_fn_index is not None:
            file_output_val = _call_gradio_fn(upload_fn_index, [file_data])
            if isinstance(file_output_val, dict) and file_output_val.get("path"):
                file_data = file_output_val

        result_msg = _call_gradio_fn(
            submit_fn_index,
            [
                file_data,
                model_name,
                model_url,
                reranker_name,
                reranker_url,
                BENCHMARK_VERSION,
                False,
            ],
        )
        return str(result_msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run piragi GPU SOTA on official AIR-Bench 24.05 Long-Doc")
    parser.add_argument("--mode", choices=["dev", "test", "both"], default="both")
    parser.add_argument(
        "--dense-models",
        nargs="+",
        default=[
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "BAAI/bge-large-en-v1.5",
            "mixedbread-ai/mxbai-embed-large-v1",
            "intfloat/e5-large-v2",
        ],
    )
    parser.add_argument(
        "--rerankers",
        nargs="+",
        default=[
            "BAAI/bge-reranker-v2-gemma",
            "BAAI/bge-reranker-v2-m3",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ],
    )
    parser.add_argument("--reranker-label", default="bge-reranker-v2-gemma")
    parser.add_argument("--reranker-url", default="https://huggingface.co/BAAI/bge-reranker-v2-gemma")
    parser.add_argument("--rerank-top-n", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fusion-weight", type=float, default=0.12)
    parser.add_argument("--model-name", default="piragi")
    parser.add_argument("--model-url", default="https://github.com/hemanth/piragi")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--submit-noreranker", action="store_true")
    args = parser.parse_args()

    with request_context(entry_point="cli_eval", preserve_existing=False):
        engine = PiragiAirBenchEngine(
            dense_models=args.dense_models,
            reranker_models=args.rerankers,
            rerank_top_n=args.rerank_top_n,
            max_length=args.max_length,
        )

        chosen_weight = args.fusion_weight
        chosen_prf = True
        chosen_reranker_weights: Optional[List[float]] = None
        summary: Dict[str, Any] = {}
        if args.mode in ("dev", "both"):
            dev_res = run_dev_benchmark(engine, fusion_weight=chosen_weight)
            chosen_weight = dev_res["best_fusion_weight"]
            chosen_prf = dev_res["use_prf"]
            chosen_reranker_weights = dev_res.get("best_reranker_weights")
            summary["dev"] = dev_res

        if args.mode in ("test", "both"):
            out_dir = Path("evals/submissions/airbench_2405")
            written_zips = generate_test_submission_zips(
                engine=engine,
                output_dir=out_dir,
                model_name=args.model_name,
                reranker_label=args.reranker_label,
                fusion_weight=chosen_weight,
                use_prf=chosen_prf,
                reranker_weights=chosen_reranker_weights,
            )
            summary["zips"] = {k: str(v) for k, v in written_zips.items()}

            if args.submit:
                msg_rr = submit_to_airbench_space(
                    zip_path=written_zips["primary"],
                    model_name=args.model_name,
                    model_url=args.model_url,
                    reranker_name=args.reranker_label,
                    reranker_url=args.reranker_url,
                )
                print("LIVE SUBMISSION RESULT (Primary):", msg_rr)
                summary["submission_primary"] = msg_rr

                # Also submit pure single bge-reranker-v2-m3 if available
                m3_key = "single_bge-reranker-v2-m3"
                if m3_key in written_zips and args.reranker_label != "bge-reranker-v2-m3":
                    time.sleep(3.0)
                    msg_m3 = submit_to_airbench_space(
                        zip_path=written_zips[m3_key],
                        model_name=args.model_name,
                        model_url=args.model_url,
                        reranker_name="bge-reranker-v2-m3",
                        reranker_url="https://huggingface.co/BAAI/bge-reranker-v2-m3",
                    )
                    print("LIVE SUBMISSION RESULT (bge-reranker-v2-m3):", msg_m3)
                    summary["submission_bge_reranker_v2_m3"] = msg_m3

            if args.submit_noreranker:
                if args.submit:
                    time.sleep(3.0)
                msg_nr = submit_to_airbench_space(
                    zip_path=written_zips["noreranker"],
                    model_name=args.model_name,
                    model_url=args.model_url,
                    reranker_name="NoReranker",
                    reranker_url="",
                )
                print("LIVE SUBMISSION RESULT (NoReranker):", msg_nr)
                summary["submission_noreranker"] = msg_nr

        tel_snap = get_telemetry().snapshot()
        summary["telemetry_red"] = tel_snap["red"]
        out_report = Path("evals/results/airbench_2405_live_summary.json")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nSaved summary report to:", str(out_report))


if __name__ == "__main__":
    main()
