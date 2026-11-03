"""Cross-encoder reranking for improved retrieval accuracy."""

import logging
from typing import List, Optional, Tuple

from .types import Citation

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers cross-encoder models.

    Cross-encoders jointly encode (query, passage) pairs and provide much more
    accurate relevance scores than bi-encoder similarities, at the cost of
    being slower (can't pre-compute passage embeddings).

    Best used as a second-stage reranker: retrieve top-N with bi-encoder,
    then rerank to top-K with cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        trust_remote_code: bool = False,
        max_length: Optional[int] = None,
        fusion_weight: float = 0.0,
        telemetry: Optional[object] = None,
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Cross-encoder model to use. Good options:
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, good quality)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (slower, better quality)
                - "BAAI/bge-reranker-base" (good multilingual support)
                - "BAAI/bge-reranker-large" (best quality, slowest)
                - "Alibaba-NLP/gte-multilingual-reranker-base" (requires trust_remote_code=True)
            device: Device to run on ('cuda', 'cpu', 'mps', or None for auto-detect)
            batch_size: Batch size for scoring
            trust_remote_code: Whether to trust remote code for custom models
                (required for some models like Alibaba-NLP/gte-multilingual-reranker-base)
            max_length: Optional tokenizer max_length cap to reduce padding latency
            fusion_weight: Optional weight (0.0-1.0) blending first-stage retrieval rank/score
                with cross-encoder score for consensus stability
            telemetry: Optional TelemetryCollector instance
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device
        self._trust_remote_code = trust_remote_code
        self.max_length = max_length
        self.fusion_weight = fusion_weight
        self._telemetry = telemetry

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                ce_kwargs = {
                    "device": self._device,
                    "trust_remote_code": self._trust_remote_code,
                }
                if self.max_length is not None:
                    ce_kwargs["max_length"] = self.max_length

                self._model = CrossEncoder(
                    self.model_name,
                    **ce_kwargs,
                )
                
                # Fix for models without pad_token_id (e.g., Qwen3 rerankers)
                # Batch inference fails if pad_token_id is not defined
                # Fallback to eos_token_id when available (fixes #14)
                try:
                    model_cfg = getattr(self._model, "model", None)
                    if model_cfg and hasattr(model_cfg, "config"):
                        cfg = model_cfg.config
                        if getattr(cfg, "pad_token_id", None) is None:
                            eos = getattr(cfg, "eos_token_id", None)
                            if eos is not None:
                                cfg.pad_token_id = eos
                                logger.debug("Missing pad_token_id detected; set to eos_token_id={}".format(eos))
                            else:
                                logger.warning("Neither pad_token_id nor eos_token_id present in model config")
                except Exception as e:
                    logger.warning("Failed to adjust pad_token_id on CrossEncoder: {}".format(e))
                
                logger.info("Loaded cross-encoder model: {}".format(self.model_name))
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        citations: List[Citation],
        top_k: Optional[int] = None,
        fusion_weight: Optional[float] = None,
    ) -> List[Citation]:
        """
        Rerank citations using length-sorted cross-encoder scoring and optional stage fusion.

        Args:
            query: The search query
            citations: List of citations to rerank
            top_k: Number of top results to return (None = return all)
            fusion_weight: Optional override for first-stage score fusion weight (0.0-1.0)

        Returns:
            Reranked citations with updated scores
        """
        if not citations:
            return citations

        if len(citations) == 1:
            return citations

        from .telemetry import get_telemetry
        tel = self._telemetry if self._telemetry is not None else get_telemetry()

        with tel.span(
            "rerank.cross_encoder",
            operation="rerank",
            provider="cross_encoder",
            candidateCount=len(citations),
            topK=top_k if top_k is not None else len(citations),
        ):
            model = self._load_model()

            # Prepare query-passage pairs
            pairs = [(query, citation.chunk) for citation in citations]

            import numpy as np

            # Length-sort across multi-batch inputs to minimize sequence padding waste on GPU/MPS/CPU
            if len(pairs) > self.batch_size:
                order = sorted(range(len(pairs)), key=lambda idx: len(pairs[idx][1]))
                sorted_pairs = [pairs[idx] for idx in order]
                sorted_raw = np.asarray(model.predict(sorted_pairs, batch_size=self.batch_size), dtype=np.float64)
                scores = np.empty(len(pairs), dtype=np.float64)
                for pos, orig_idx in enumerate(order):
                    scores[orig_idx] = sorted_raw[pos]
            else:
                scores = np.asarray(model.predict(pairs, batch_size=self.batch_size), dtype=np.float64)

            # Normalize scores to 0-1 range using numerically stable sigmoid
            clipped = np.clip(scores, -50.0, 50.0)
            normalized_scores = 1.0 / (1.0 + np.exp(-clipped))

            blend = self.fusion_weight if fusion_weight is None else float(fusion_weight)
            if blend > 0.0:
                # Min-max normalize first-stage scores or fall back to reciprocal rank
                stage1_raw = np.array([float(c.score) for c in citations], dtype=np.float64)
                s_min = float(np.min(stage1_raw))
                s_max = float(np.max(stage1_raw))
                if s_max > s_min:
                    stage1_norm = (stage1_raw - s_min) / (s_max - s_min)
                else:
                    ranks = np.arange(1.0, len(citations) + 1.0)
                    stage1_norm = 1.0 / (60.0 + ranks)
                    stage1_norm = stage1_norm / float(np.max(stage1_norm))
                normalized_scores = (1.0 - blend) * normalized_scores + blend * stage1_norm

            # Create new citations with updated scores
            scored_citations = []
            for citation, score in zip(citations, normalized_scores):
                new_citation = Citation(
                    source=citation.source,
                    chunk=citation.chunk,
                    score=float(score),
                    metadata=citation.metadata,
                )
                scored_citations.append(new_citation)

            # Sort by score descending
            scored_citations.sort(key=lambda c: c.score, reverse=True)

            # Return top_k if specified
            if top_k is not None:
                return scored_citations[:top_k]

            return scored_citations

    def score_pair(self, query: str, passage: str) -> float:
        """
        Score a single query-passage pair.

        Args:
            query: The search query
            passage: The passage to score

        Returns:
            Relevance score (higher = more relevant)
        """
        model = self._load_model()
        score = model.predict([(query, passage)])[0]

        # Normalize to 0-1
        import numpy as np
        return float(1 / (1 + np.exp(-score)))


class TFIDFReranker:
    """
    TF-IDF based reranker for fast keyword-aware reranking.

    Useful as a lightweight alternative to cross-encoder when speed is critical.
    Combines vector similarity with TF-IDF keyword matching.
    """

    def __init__(self, vector_weight: float = 0.6, tfidf_weight: float = 0.4):
        """
        Initialize TF-IDF reranker.

        Args:
            vector_weight: Weight for original vector similarity score (0-1)
            tfidf_weight: Weight for TF-IDF score (0-1)
        """
        if not (0 <= vector_weight <= 1) or not (0 <= tfidf_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        self.vector_weight = vector_weight
        self.tfidf_weight = tfidf_weight
        self._vectorizer = None
        self._fitted = False

    def _get_vectorizer(self):
        """Get or create TF-IDF vectorizer."""
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),  # Unigrams and bigrams
                max_features=10000,
            )
        return self._vectorizer

    def rerank(
        self,
        query: str,
        citations: List[Citation],
        top_k: Optional[int] = None,
    ) -> List[Citation]:
        """
        Rerank citations using TF-IDF scoring combined with vector scores.

        Args:
            query: The search query
            citations: List of citations to rerank
            top_k: Number of top results to return (None = return all)

        Returns:
            Reranked citations with combined scores
        """
        if not citations:
            return citations

        if len(citations) == 1:
            return citations

        vectorizer = self._get_vectorizer()

        # Fit vectorizer on all chunks
        chunks = [c.chunk for c in citations]
        try:
            tfidf_matrix = vectorizer.fit_transform(chunks)
            query_vector = vectorizer.transform([query])

            # Compute cosine similarity between query and each chunk
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

        except Exception as e:
            logger.warning(f"TF-IDF scoring failed: {e}, using vector scores only")
            tfidf_scores = [0.0] * len(citations)

        # Combine scores
        scored_citations = []
        for citation, tfidf_score in zip(citations, tfidf_scores):
            combined_score = (
                self.vector_weight * citation.score +
                self.tfidf_weight * float(tfidf_score)
            )
            new_citation = Citation(
                source=citation.source,
                chunk=citation.chunk,
                score=combined_score,
                metadata=citation.metadata,
            )
            scored_citations.append(new_citation)

        # Sort by combined score
        scored_citations.sort(key=lambda c: c.score, reverse=True)

        if top_k is not None:
            return scored_citations[:top_k]

        return scored_citations


class HybridReranker:
    """
    Hybrid reranker that combines multiple reranking strategies.

    Uses TF-IDF for initial reranking, then optionally applies
    cross-encoder for final reranking of top candidates.
    """

    def __init__(
        self,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_top_n: int = 20,
        tfidf_vector_weight: float = 0.6,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize hybrid reranker.

        Args:
            use_cross_encoder: Whether to use cross-encoder as final stage
            cross_encoder_model: Cross-encoder model name
            cross_encoder_top_n: Number of candidates to pass to cross-encoder
            tfidf_vector_weight: Weight for vector vs TF-IDF in first stage
            device: Device for cross-encoder
            trust_remote_code: Whether to trust remote code for custom models
        """
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_top_n = cross_encoder_top_n

        # Initialize TF-IDF reranker
        self.tfidf_reranker = TFIDFReranker(
            vector_weight=tfidf_vector_weight,
            tfidf_weight=1.0 - tfidf_vector_weight,
        )

        # Initialize cross-encoder (lazy loaded)
        self._cross_encoder = None
        if use_cross_encoder:
            self._cross_encoder = CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=device,
                trust_remote_code=trust_remote_code,
            )

    def rerank(
        self,
        query: str,
        citations: List[Citation],
        top_k: Optional[int] = None,
    ) -> List[Citation]:
        """
        Rerank using hybrid approach.

        Args:
            query: The search query
            citations: List of citations to rerank
            top_k: Number of top results to return

        Returns:
            Reranked citations
        """
        if not citations:
            return citations

        # Stage 1: TF-IDF + vector reranking
        stage1_results = self.tfidf_reranker.rerank(
            query,
            citations,
            top_k=self.cross_encoder_top_n if self.use_cross_encoder else top_k,
        )

        # Stage 2: Cross-encoder reranking (if enabled)
        if self.use_cross_encoder and self._cross_encoder:
            final_results = self._cross_encoder.rerank(
                query,
                stage1_results,
                top_k=top_k,
            )
            return final_results

        return stage1_results[:top_k] if top_k else stage1_results


def sanitize_passage_brackets(text: str) -> str:
    """Replace numeric bracket references like [12] with (12) so listwise IDs stay unambiguous."""
    import re

    return re.sub(r"\[(\d+)\]", r"(\1)", text or "")


def parse_listwise_ranking(raw_output: str, num_candidates: int) -> List[int]:
    """
    Parse RankGPT / ReasonRank listwise output into a complete 0-indexed permutation.

    Handles `<think>...</think><answer>[3] > [1] > [2]</answer>`, `<answer>...</answer>`,
    ties (`[3] > [1] = [2]`), out-of-range IDs, duplicates, and missing IDs (appended in
    original relative order so no candidate is ever dropped).
    """
    import re

    if num_candidates <= 0:
        return []

    text = raw_output or ""
    ans_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if ans_match:
        target = ans_match.group(1)
    elif "</think>" in text:
        target = text.split("</think>", 1)[1]
    else:
        target = text

    found_one_based = [int(m) for m in re.findall(r"\[(\d+)\]", target)]
    if not found_one_based:
        found_one_based = [int(m) for m in re.findall(r"\b(\d+)\b", target)]

    seen = set()
    ordered: List[int] = []
    for idx_1 in found_one_based:
        idx_0 = idx_1 - 1
        if 0 <= idx_0 < num_candidates and idx_0 not in seen:
            seen.add(idx_0)
            ordered.append(idx_0)

    for idx_0 in range(num_candidates):
        if idx_0 not in seen:
            ordered.append(idx_0)

    return ordered


class CausalLLMReranker:
    """
    Instruction-aware causal LLM Yes/No logit reranker (Qwen3-Reranker & BGE-Gemma).

    Scores (instruction, query, document) in a single forward pass without autoregressive
    decoding overhead, using length-sorted batching and optional Stage-1 score blending.
    """

    DEFAULT_INSTRUCTION = (
        "Given a reasoning-intensive search query, retrieve relevant passages, theorems, "
        "or documentation that help answer the query"
    )
    GEMMA_PROMPT = (
        "Given a query A and a passage B, determine whether the passage contains an answer "
        "to the query by providing a prediction of either 'Yes' or 'No'."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        instruction: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 1024,
        trust_remote_code: bool = True,
        fusion_weight: float = 0.0,
        telemetry: Optional[object] = None,
    ) -> None:
        self.model_name = model_name
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self._device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._trust_remote_code = trust_remote_code
        self.fusion_weight = fusion_weight
        self._telemetry = telemetry
        self._tokenizer = None
        self._model = None

    def _is_qwen3_style(self) -> bool:
        lower = self.model_name.lower()
        return "qwen3-reranker" in lower or "qwen3_reranker" in lower

    def format_qwen3_pair(self, query: str, doc: str, instruction: Optional[str] = None) -> str:
        inst = instruction or self.instruction
        return "<Instruct>: {}\n<Query>: {}\n<Document>: {}".format(inst, query, doc)

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dev = self._resolve_device()
            if dev == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif dev in ("cuda", "mps"):
                dtype = torch.float16
            else:
                dtype = torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=self._trust_remote_code,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=self._trust_remote_code,
            ).to(dev)
            self._model.eval()
        return self._tokenizer, self._model

    def _score_pairs(self, query: str, passages: List[str], instruction: Optional[str] = None) -> List[float]:
        import numpy as np
        import torch

        tokenizer, model = self._load_model()
        dev = next(model.parameters()).device

        order = sorted(range(len(passages)), key=lambda idx: len(passages[idx]))
        sorted_passages = [passages[idx] for idx in order]
        sorted_scores = np.zeros(len(passages), dtype=np.float64)

        if self._is_qwen3_style():
            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on "
                "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
                "or \"no\".<|im_end|>\n<|im_start|>user\n"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            yes_id = tokenizer.convert_tokens_to_ids("yes")
            no_id = tokenizer.convert_tokens_to_ids("no")
            budget = max(64, self.max_length - len(prefix_ids) - len(suffix_ids))

            for start in range(0, len(sorted_passages), self.batch_size):
                batch_docs = sorted_passages[start : start + self.batch_size]
                formatted = [self.format_qwen3_pair(query, d, instruction=instruction) for d in batch_docs]
                enc = tokenizer(
                    formatted,
                    padding=False,
                    truncation="longest_first",
                    return_attention_mask=False,
                    max_length=budget,
                )
                enc["input_ids"] = [prefix_ids + ids + suffix_ids for ids in enc["input_ids"]]
                padded = tokenizer.pad(enc, padding=True, return_tensors="pt")
                padded = {k: v.to(dev) for k, v in padded.items()}
                with torch.inference_mode():
                    logits = model(**padded, use_cache=False).logits[:, -1, :]
                    pair_logits = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
                    probs = torch.nn.functional.log_softmax(pair_logits.float(), dim=1)[:, 1].exp()
                sorted_scores[start : start + len(batch_docs)] = probs.cpu().numpy()
        else:
            yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
            sep_ids = tokenizer("\n", return_tensors=None, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(self.GEMMA_PROMPT, return_tensors=None, add_special_tokens=False)["input_ids"]
            bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
            q_ids = tokenizer(
                "A: {}".format(query),
                return_tensors=None,
                add_special_tokens=False,
                max_length=(self.max_length * 3) // 4,
                truncation=True,
            )["input_ids"]

            for start in range(0, len(sorted_passages), self.batch_size):
                batch_docs = sorted_passages[start : start + self.batch_size]
                items = []
                for doc in batch_docs:
                    p_ids = tokenizer(
                        "B: {}".format(doc),
                        return_tensors=None,
                        add_special_tokens=False,
                        max_length=self.max_length,
                        truncation=True,
                    )["input_ids"]
                    ids = bos + q_ids + sep_ids + p_ids + sep_ids + prompt_ids
                    items.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
                padded = tokenizer.pad(items, padding=True, return_tensors="pt")
                padded = {k: v.to(dev) for k, v in padded.items()}
                with torch.inference_mode():
                    raw = model(**padded, use_cache=False).logits[:, -1, yes_id].float()
                    probs = torch.sigmoid(torch.clamp(raw, -50.0, 50.0))
                sorted_scores[start : start + len(batch_docs)] = probs.cpu().numpy()

        restored = np.empty(len(passages), dtype=np.float64)
        for pos, orig_idx in enumerate(order):
            restored[orig_idx] = sorted_scores[pos]
        return restored.tolist()

    def rerank(
        self,
        query: str,
        citations: List[Citation],
        top_k: Optional[int] = None,
        fusion_weight: Optional[float] = None,
        instruction: Optional[str] = None,
    ) -> List[Citation]:
        import numpy as np
        from .telemetry import get_telemetry

        if not citations or len(citations) == 1:
            return citations[:top_k] if top_k is not None else citations

        tel = self._telemetry if self._telemetry is not None else get_telemetry()
        with tel.span(
            "rerank.causal_llm",
            operation="rerank",
            provider="causal_llm",
            candidateCount=len(citations),
            topK=top_k if top_k is not None else len(citations),
        ):
            probs = np.asarray(
                self._score_pairs(query, [c.chunk for c in citations], instruction=instruction),
                dtype=np.float64,
            )
            blend = self.fusion_weight if fusion_weight is None else float(fusion_weight)
            if blend > 0.0:
                stage1 = np.asarray([float(c.score) for c in citations], dtype=np.float64)
                s_min, s_max = float(np.min(stage1)), float(np.max(stage1))
                if s_max > s_min:
                    s_norm = (stage1 - s_min) / (s_max - s_min)
                else:
                    ranks = np.arange(1.0, len(citations) + 1.0)
                    s_norm = (1.0 / (60.0 + ranks)) / (1.0 / 61.0)
                probs = (1.0 - blend) * probs + blend * s_norm

            scored = [
                Citation(
                    source=c.source,
                    chunk=c.chunk,
                    score=float(sc),
                    metadata=c.metadata,
                )
                for c, sc in zip(citations, probs)
            ]
            scored.sort(key=lambda c: c.score, reverse=True)
            return scored[:top_k] if top_k is not None else scored


class ListwiseReasoningReranker:
    """
    Sliding-window listwise reasoning reranker (ReasonRank / RankLLM compatible).

    Jointly compares windows of candidate passages using explicit step-by-step
    `<think>...</think><answer>[i] > [j] ...</answer>` comparative reasoning,
    progressing bottom-up so deep relevant passages bubble into top-K.
    """

    SYSTEM_PROMPT_REASONING = (
        "You are RankLLM, an intelligent assistant that can rank passages based on their "
        "relevance to the query. Given a query and a passage list, you first thinks about "
        "the reasoning process in the mind and then provides the answer (i.e., the reranked "
        "passage list). The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, i.e., <think> reasoning process here "
        "</think> <answer> answer here </answer>."
    )

    def __init__(
        self,
        llm_client=None,
        model_name: Optional[str] = None,
        window_size: int = 20,
        step_size: int = 10,
        max_passage_chars: int = 1200,
        max_tokens: int = 1536,
        fusion_weight: float = 0.0,
        device: Optional[str] = None,
        telemetry: Optional[object] = None,
    ) -> None:
        self.llm_client = llm_client
        self.model_name = model_name
        self.window_size = max(2, int(window_size))
        self.step_size = max(1, int(step_size))
        self.max_passage_chars = max_passage_chars
        self.max_tokens = max_tokens
        self.fusion_weight = fusion_weight
        self._device = device
        self._telemetry = telemetry
        self._hf_tokenizer = None
        self._hf_model = None

    def build_messages(self, query: str, window_passages: List[str]) -> List[dict]:
        """Construct exact ReasonRank / RankGPT_reasoning chat messages."""
        clean_q = sanitize_passage_brackets(query).strip()
        num = len(window_passages)
        lines = [
            "I will provide you with {} passages, each indicated by a numerical identifier []. "
            "Rank the passages based on their relevance to the search query: {}.".format(num, clean_q),
            "",
        ]
        for idx, passage in enumerate(window_passages, start=1):
            clipped = sanitize_passage_brackets(passage.strip())[: self.max_passage_chars]
            clipped = " ".join(clipped.split())
            lines.append("[{}] {}".format(idx, clipped))

        lines.append(
            "Search Query: {}.\nRank the {} passages above based on their relevance to the search query. "
            "All the passages should be included and listed using identifiers, in descending order of "
            "relevance. The format of the answer should be [] > [], e.g., [2] > [1].".format(clean_q, num)
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT_REASONING},
            {"role": "user", "content": "\n".join(lines)},
        ]

    def _generate_window_ranking(self, messages: List[dict]) -> str:
        if self.llm_client is not None:
            resp = self.llm_client.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        if not self.model_name:
            raise ValueError("Either llm_client or model_name must be provided to ListwiseReasoningReranker")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._hf_model is None:
            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self._hf_tokenizer.pad_token is None:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(dev)
            self._hf_model.eval()

        prompt = self._hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        dev = next(self._hf_model.parameters()).device
        enc = self._hf_tokenizer([prompt], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out_ids = self._hf_model.generate(
                **enc,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self._hf_tokenizer.pad_token_id,
            )
        gen_tokens = out_ids[0, enc["input_ids"].shape[1] :]
        return self._hf_tokenizer.decode(gen_tokens, skip_special_tokens=True)

    def _sliding_windows(self, n: int) -> List[Tuple[int, int]]:
        """Compute bottom-up [start, end) windows covering n candidates."""
        if n <= self.window_size:
            return [(0, n)]
        windows: List[Tuple[int, int]] = []
        end = n
        while end > 0:
            start = max(0, end - self.window_size)
            windows.append((start, end))
            if start == 0:
                break
            end -= self.step_size
        return windows

    def rerank(
        self,
        query: str,
        citations: List[Citation],
        top_k: Optional[int] = None,
        fusion_weight: Optional[float] = None,
    ) -> List[Citation]:
        from .telemetry import get_telemetry

        if not citations or len(citations) == 1:
            return citations[:top_k] if top_k is not None else citations

        tel = self._telemetry if self._telemetry is not None else get_telemetry()
        with tel.span(
            "rerank.listwise",
            operation="rerank",
            provider="listwise_llm",
            candidateCount=len(citations),
            topK=top_k if top_k is not None else len(citations),
        ):
            working = list(citations)
            for start, end in self._sliding_windows(len(working)):
                sub = working[start:end]
                if len(sub) <= 1:
                    continue
                try:
                    msgs = self.build_messages(query, [c.chunk for c in sub])
                    raw_text = self._generate_window_ranking(msgs)
                    perm = parse_listwise_ranking(raw_text, len(sub))
                    working[start:end] = [sub[i] for i in perm]
                except Exception as exc:
                    logger.warning("Listwise window [{}:{}] failed: {}".format(start, end, exc))

            blend = self.fusion_weight if fusion_weight is None else float(fusion_weight)
            orig_rank_map = {(c.source, c.chunk): r for r, c in enumerate(citations, start=1)}

            n_total = float(len(working))
            scored: List[Citation] = []
            for new_rank, c in enumerate(working, start=1):
                listwise_score = 1.0 - (float(new_rank - 1) / max(1.0, n_total))
                if blend > 0.0:
                    orig_rank = orig_rank_map.get((c.source, c.chunk), new_rank)
                    rrf_new = 1.0 / (60.0 + float(new_rank))
                    rrf_old = 1.0 / (60.0 + float(orig_rank))
                    final_score = (1.0 - blend) * rrf_new + blend * rrf_old
                else:
                    final_score = listwise_score
                scored.append(
                    Citation(
                        source=c.source,
                        chunk=c.chunk,
                        score=float(final_score),
                        metadata=c.metadata,
                    )
                )

            scored.sort(key=lambda c: c.score, reverse=True)
            return scored[:top_k] if top_k is not None else scored
