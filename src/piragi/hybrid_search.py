"""Hybrid search combining vector similarity with BM25 keyword matching."""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from rank_bm25 import BM25Okapi

from .telemetry import get_telemetry
from .types import Citation

logger = logging.getLogger(__name__)

_CAMEL_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])|([A-Z]+)([A-Z][a-z])|([a-zA-Z])([0-9])|([0-9])([a-zA-Z])")
_SAFE_MISSING_SPACE_CAMEL_RE = re.compile(
    r"([a-z]{4,}|\b(?:of|to|in|on|at|by|as|is|or|an|be|it|we|a))([A-Z][a-zA-Z0-9]*)"
)
_ACRONYM_LOWER_BOUNDARY_RE = re.compile(r"([A-Z]{2,})([a-z]{3,})")
_PUNCT_GLUE_RE = re.compile(r",([a-zA-Z]{2,})")
_HYPHEN_LINEWRAP_RE = re.compile(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})")
_ALPHA_RUN_RE = re.compile(r"[A-Za-z]{12,}")
_WORD_TOKEN_RE = re.compile(r"\b\w+\b")

_LIGATURE_TABLE = str.maketrans({
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00a0": " ",
})

# Common English connector words that get glued in PDF/EPUB extraction without whitespace
_GLUE_CONNECTORS = frozenset({
    "the", "and", "for", "are", "with", "from", "that", "this", "into", "over",
    "under", "between", "through", "about", "after", "before", "their", "there",
    "which", "where", "while", "when", "what", "were", "have", "been", "will",
    "would", "could", "should", "other", "these", "those", "such", "than", "then",
    "some", "more", "most", "only", "also", "used", "using", "based", "data",
    "model", "models", "system", "power", "usage", "time", "high", "large",
    "not", "can", "may", "one", "two", "all", "any", "our", "out", "new", "how",
    "of", "in", "to", "is", "on", "as", "at", "by", "or", "an", "be", "it", "its",
    "we", "so", "if", "no", "up", "do", "a",
})

# Grammar/prose anchor words confirming a glued run is natural-language prose rather than an IUPAC chemical compound
_PROSE_GLUE_ANCHORS = frozenset({
    "the", "and", "for", "are", "with", "from", "that", "this", "into", "over",
    "under", "between", "through", "about", "after", "before", "their", "there",
    "which", "where", "while", "when", "what", "were", "have", "been", "will",
    "would", "could", "should", "other", "these", "those", "such", "than", "then",
    "some", "more", "most", "only", "also", "used", "using", "based", "not",
    "can", "may", "our", "how", "of", "to", "is", "on", "as", "at", "by",
    "we", "if", "its", "further", "present", "discuss", "however", "because",
})


def _segment_glued_alpha_exact(token: str, vocab: Set[str]) -> Optional[List[str]]:
    """Segment an unseen glued alphabetic prose token (>= 12 chars) into clean words."""
    lower = token.lower()
    n = len(lower)
    if n < 12 or not lower.isalpha() or lower in vocab:
        return None

    # Allow skipping rare out-of-vocab words on long runs (n >= 18) with a small penalty,
    # preserving those skipped character spans intact as words in the output.
    allow_gaps = n >= 18
    neg_inf = -10**9
    best_score = [neg_inf] * (n + 1)
    best_score[0] = 0
    back_ptr: List[Optional[Tuple[int, Optional[str]]]] = [None] * (n + 1)

    for i in range(1, n + 1):
        if allow_gaps and best_score[i - 1] > neg_inf:
            prev_step = back_ptr[i - 1]
            in_gap = prev_step is not None and prev_step[1] is None
            gap_cost = 1 if in_gap else 12
            best_score[i] = best_score[i - 1] - gap_cost
            back_ptr[i] = (i - 1, None)

        max_w = min(i, 15)
        for length in range(1, max_w + 1):
            sub = lower[i - length : i]
            if length == 1:
                prev_step = back_ptr[i - 1]
                prev_was_gap = prev_step is not None and prev_step[1] is None
                is_valid = (sub == "a") and not prev_was_gap
            elif length <= 3:
                is_valid = sub in _GLUE_CONNECTORS
            else:
                is_valid = sub in vocab or sub in _GLUE_CONNECTORS
            if not is_valid:
                continue
            prev = best_score[i - length]
            if prev <= neg_inf:
                continue
            if length >= 4:
                bonus = length * length * 2
            elif length >= 2:
                bonus = length * 4
            else:
                bonus = 1
            cand = prev + bonus
            if cand > best_score[i]:
                best_score[i] = cand
                back_ptr[i] = (i - length, token[i - length : i])

    if best_score[n] <= neg_inf:
        return None

    matched_words: List[str] = []
    tagged_rev: List[Tuple[str, bool]] = []  # (text, is_gap)
    gap_chars: List[str] = []
    matched_char_count = 0
    cur = n
    while cur > 0:
        step = back_ptr[cur]
        if step is None:
            return None
        prev_idx, surface = step
        if surface is None:
            gap_chars.append(token[prev_idx])
        else:
            if gap_chars:
                gap_chars.reverse()
                tagged_rev.append(("".join(gap_chars), True))
                gap_chars = []
            tagged_rev.append((surface, False))
            matched_words.append(surface.lower())
            matched_char_count += len(surface)
        cur = prev_idx

    if gap_chars:
        gap_chars.reverse()
        tagged_rev.append(("".join(gap_chars), True))

    tagged_rev.reverse()
    # Merge stray 1-char 'a' adjacent to an out-of-vocab gap span (e.g. 'a' + 'ctu' + 'a' + 'l' -> 'actual')
    merged: List[Tuple[str, bool]] = []
    for seg_txt, is_gap in tagged_rev:
        if merged and (
            (merged[-1][1] and (is_gap or len(seg_txt) == 1))
            or (len(merged[-1][0]) == 1 and is_gap)
        ):
            prev_txt, _ = merged[-1]
            merged[-1] = (prev_txt + seg_txt, True)
        else:
            merged.append((seg_txt, is_gap))
    segments_rev = [seg_txt for seg_txt, _ in merged]
    if len(segments_rev) < 2:
        return None
    # Unconditionally require at least one natural-language prose anchor so IUPAC/scientific terms stay intact
    if not any(w in _PROSE_GLUE_ANCHORS for w in matched_words):
        return None
    if matched_char_count < int(0.60 * n):
        return None
    if sum(len(w) for w in matched_words if len(w) >= 4) < int(0.35 * n):
        return None
    return segments_rev


def normalize_corpus_texts(texts: List[str]) -> List[str]:
    """Repair PDF/EPUB ligatures, line-wrap hyphens, and glued missing-space runs for neural & lexical indexing."""
    if not texts:
        return []

    precleaned = [t.translate(_LIGATURE_TABLE) for t in texts]

    # Build corpus vocabulary of clean standalone alphabetic words (freq >= 2 if >= 12 chars)
    counts: Dict[str, int] = defaultdict(int)
    for txt in precleaned:
        for tok in _WORD_TOKEN_RE.findall(txt.lower()):
            if tok.isalpha() and 4 <= len(tok) <= 22:
                counts[tok] += 1

    vocab: Set[str] = {
        tok for tok, cnt in counts.items() if (len(tok) <= 11 and cnt >= 1) or cnt >= 2
    }

    def _fix_hyphen(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        joined = (left + right).lower()
        if joined in vocab:
            return left + right
        return "{}-{}".format(left, right)

    def _fix_alpha_run(match: re.Match) -> str:
        tok = match.group(0)
        parts = _segment_glued_alpha_exact(tok, vocab)
        if parts is not None:
            return " ".join(parts)
        return tok

    normalized: List[str] = []
    for txt in precleaned:
        s = _HYPHEN_LINEWRAP_RE.sub(_fix_hyphen, txt)
        s = _PUNCT_GLUE_RE.sub(r", \1", s)
        # Apply twice to handle consecutive CamelCase glue like 'fromtheThermalDesign'
        s = _SAFE_MISSING_SPACE_CAMEL_RE.sub(r"\1 \2", s)
        s = _SAFE_MISSING_SPACE_CAMEL_RE.sub(r"\1 \2", s)
        s = _ACRONYM_LOWER_BOUNDARY_RE.sub(r"\1 \2", s)
        if vocab:
            s = _ALPHA_RUN_RE.sub(_fix_alpha_run, s)
        normalized.append(s)
    return normalized


def _deglue_long_token(token: str, vocab: Set[str]) -> List[str]:
    """Segment abnormally long glued PDF/EPUB tokens (>= 14 chars) using corpus vocabulary DP."""
    n = len(token)
    if n < 14 or not token.isalpha() or token in vocab:
        return []

    # Dynamic programming: maximize character coverage by valid words
    best_score = [0] * (n + 1)
    back_ptr: List[Optional[Tuple[int, Optional[str]]]] = [None] * (n + 1)

    for i in range(1, n + 1):
        # Option 1: skip character i-1 (unmatched char)
        best_score[i] = best_score[i - 1]
        back_ptr[i] = (i - 1, None)

        max_w = min(i, 14)
        for length in range(2, max_w + 1):
            sub = token[i - length : i]
            is_valid = (length >= 4 and sub in vocab) or (sub in _GLUE_CONNECTORS)
            if is_valid:
                # Favor longer real words over tiny connector fragments
                bonus = length * length if length >= 4 else length
                cand = best_score[i - length] + bonus
                if cand > best_score[i]:
                    best_score[i] = cand
                    back_ptr[i] = (i - length, sub)

    parts: List[str] = []
    matched_chars = 0
    cur = n
    while cur > 0 and back_ptr[cur] is not None:
        prev_idx, word = back_ptr[cur]  # type: ignore[misc]
        if word is not None:
            matched_chars += len(word)
            if len(word) > 2 and word not in ("the", "and", "for", "are", "with", "from", "that", "this"):
                parts.append(word)
        cur = prev_idx

    # Only emit segments when at least 70% of the glued token resolved cleanly into >= 2 words
    if matched_chars >= int(0.7 * n) and len(parts) >= 2:
        parts.reverse()
        return parts
    return []


class HybridSearcher:
    """
    Hybrid search combining dense vector retrieval with sparse BM25.

    This implements the fusion approach where results from both retrievers
    are combined using Reciprocal Rank Fusion (RRF) or weighted scoring.
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        use_rrf: bool = True,
        rrf_k: int = 60,
        telemetry: Any = None,
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_weight: Weight for vector similarity scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            use_rrf: Use Reciprocal Rank Fusion instead of weighted scoring
            rrf_k: RRF constant (typically 60)
            telemetry: Optional TelemetryCollector instance
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        self._telemetry = telemetry

        self._bm25: Optional[BM25Okapi] = None
        self._chunk_texts: List[str] = []
        self._chunk_to_idx: Dict[str, int] = {}
        self._vocab: Set[str] = set()
        # Corpus BM25 cache: corpus_fingerprint -> BM25Okapi
        self._corpus_cache: Dict[Any, BM25Okapi] = {}
        self._max_cache_size: int = 16

    def _get_telemetry(self):
        return self._telemetry if self._telemetry is not None else get_telemetry()

    def _compute_corpus_key(self, corpus: List[Citation]) -> Optional[Tuple[int, int]]:
        """Compute an efficient, collision-resistant hash fingerprint of the corpus."""
        if not corpus:
            return None
        return (len(corpus), hash(tuple((c.source, hash(c.chunk)) for c in corpus)))

    def _tokenize_corpus_texts(self, texts: List[str]) -> List[List[str]]:
        """Tokenize corpus texts and recover words inside glued PDF/EPUB runs."""
        raw_tokenized = [self._tokenize(t) for t in texts]
        vocab: Set[str] = set()
        for toks in raw_tokenized:
            for tok in toks:
                if 4 <= len(tok) <= 13 and tok.isalpha():
                    vocab.add(tok)
        self._vocab = vocab
        if not vocab:
            return raw_tokenized

        enriched: List[List[str]] = []
        for toks in raw_tokenized:
            has_long = any(len(tok) >= 14 for tok in toks)
            if not has_long:
                enriched.append(toks)
                continue
            row = list(toks)
            for tok in toks:
                if len(tok) >= 14:
                    row.extend(_deglue_long_token(tok, vocab))
            enriched.append(row)
        return enriched

    def _get_or_create_bm25(self, corpus: List[Citation]) -> Optional[BM25Okapi]:
        """Retrieve cached BM25Okapi for the corpus or tokenize and build a new instance."""
        if not corpus:
            return None

        tel = self._get_telemetry()
        corpus_key = self._compute_corpus_key(corpus)
        if corpus_key is not None and corpus_key in self._corpus_cache:
            tel.record_bm25_cache(
                cache_size=len(self._corpus_cache),
                max_cache_size=self._max_cache_size,
                hit=True,
            )
            return self._corpus_cache[corpus_key]

        tokenized = self._tokenize_corpus_texts([c.chunk for c in corpus])
        if not tokenized or not any(tokenized):
            tel.record_bm25_cache(
                cache_size=len(self._corpus_cache),
                max_cache_size=self._max_cache_size,
                hit=False,
            )
            return None

        bm25 = BM25Okapi(tokenized)
        if getattr(bm25, "average_idf", 0.0) > 0:
            floor_idf = bm25.epsilon * bm25.average_idf
            for tok, val in list(bm25.idf.items()):
                if val <= 0.0:
                    bm25.idf[tok] = floor_idf

        evicted = False
        if corpus_key is not None:
            if len(self._corpus_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._corpus_cache))
                del self._corpus_cache[oldest_key]
                evicted = True
            self._corpus_cache[corpus_key] = bm25

        tel.record_bm25_cache(
            cache_size=len(self._corpus_cache),
            max_cache_size=self._max_cache_size,
            hit=False,
            evicted=evicted,
        )
        if evicted:
            tel.log(
                "info",
                "bm25_cache_evicted",
                operation="hybrid_fusion",
                provider="bm25",
                cacheSize=len(self._corpus_cache),
                maxCacheSize=self._max_cache_size,
            )
        return bm25

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text with lowercasing plus CamelCase/alphanumeric boundary recovery."""
        base_tokens = [t for t in _WORD_TOKEN_RE.findall(text.lower()) if len(t) > 1]
        # Recover CamelCase or letter-digit sub-tokens (e.g. 'aGPUis' -> 'gpu', 'llama2' -> 'llama')
        split_text = _CAMEL_BOUNDARY_RE.sub(
            lambda m: "{} {}".format(
                m.group(1) or m.group(3) or m.group(5) or m.group(7),
                m.group(2) or m.group(4) or m.group(6) or m.group(8),
            ),
            text,
        )
        if split_text != text:
            seen = set(base_tokens)
            for extra in _WORD_TOKEN_RE.findall(split_text.lower()):
                if len(extra) > 1 and extra not in seen:
                    base_tokens.append(extra)
                    seen.add(extra)
        if self._vocab:
            for tok in list(base_tokens):
                if len(tok) >= 14:
                    base_tokens.extend(_deglue_long_token(tok, self._vocab))
        return base_tokens

    def index_chunks(self, chunks: List[str]) -> None:
        """
        Index chunks for BM25 search.

        Args:
            chunks: List of chunk text strings
        """
        if not chunks:
            self._chunk_texts = []
            self._chunk_to_idx = {}
            self._bm25 = None
            return

        tokenized_corpus = self._tokenize_corpus_texts(chunks)
        if not tokenized_corpus or not any(tokenized_corpus):
            self._chunk_texts = list(chunks)
            self._chunk_to_idx = {text: i for i, text in enumerate(chunks)}
            self._chunk_to_idx.update({text[:200]: i for i, text in enumerate(chunks)})
            self._bm25 = None
            return

        self._chunk_texts = list(chunks)
        self._chunk_to_idx = {text: i for i, text in enumerate(chunks)}
        self._chunk_to_idx.update({text[:200]: i for i, text in enumerate(chunks)})
        self._bm25 = BM25Okapi(tokenized_corpus)
        if getattr(self._bm25, "average_idf", 0.0) > 0:
            floor_idf = self._bm25.epsilon * self._bm25.average_idf
            for tok, val in list(self._bm25.idf.items()):
                if val <= 0.0:
                    self._bm25.idf[tok] = floor_idf
        logger.info("Indexed {} chunks for BM25 search".format(len(chunks)))

    def search(
        self,
        query: str,
        vector_citations: List[Citation],
        top_k: int = 10,
    ) -> List[Citation]:
        """
        Perform hybrid search combining vector results with BM25.

        Args:
            query: Search query
            vector_citations: Citations from vector search (already retrieved)
            top_k: Number of results to return

        Returns:
            Combined and reranked citations
        """
        if not vector_citations:
            return []

        if self._bm25 is None:
            # BM25 not indexed, fall back to vector only
            logger.warning("BM25 not indexed, using vector search only")
            return vector_citations[:top_k]

        # Get BM25 scores for all indexed chunks
        query_tokens = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)

        # Create mapping of (source, chunk) to vector citation
        vector_scores: Dict[Tuple[str, str], float] = {}
        citation_map: Dict[Tuple[str, str], Citation] = {}

        for citation in vector_citations:
            key = (citation.source, citation.chunk)
            vector_scores[key] = citation.score
            citation_map[key] = citation

        if self.use_rrf:
            # Reciprocal Rank Fusion
            combined_scores = self._rrf_fusion(
                vector_citations,
                bm25_scores,
                citation_map,
            )
        else:
            # Weighted score combination
            combined_scores = self._weighted_fusion(
                vector_citations,
                bm25_scores,
                citation_map,
            )

        # Sort by combined score and return top_k
        sorted_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for key, score in sorted_items[:top_k]:
            if key in citation_map:
                citation = citation_map[key]
                results.append(
                    Citation(
                        source=citation.source,
                        chunk=citation.chunk,
                        score=score,
                        metadata=citation.metadata,
                    )
                )

        return results

    def search_corpus(self, query, vector_citations, corpus, top_k=20):
        """Fuse independent BM25 and dense lists, including keyword-only hits.

        Reuses cached BM25Okapi index when corpus has not changed, eliminating
        repeated tokenization overhead across queries.
        """
        if top_k <= 0:
            return []
        if not corpus:
            return vector_citations[:top_k]

        bm25 = self._get_or_create_bm25(corpus)
        if bm25 is None:
            return vector_citations[:top_k]

        raw = bm25.get_scores(self._tokenize(query))
        keyword = sorted(
            [(c, float(score)) for c, score in zip(corpus, raw) if score > 0],
            key=lambda pair: pair[1], reverse=True,
        )[:top_k]
        scores, citations = {}, {}
        vector_citations = sorted(vector_citations, key=lambda c: c.score, reverse=True)
        lists = [
            ([(c, c.score) for c in vector_citations[:top_k]], self.vector_weight),
            (keyword, self.bm25_weight),
        ]
        for items, weight in lists:
            maximum = max((score for _, score in items), default=1) or 1
            seen = set()
            for rank, (citation, score) in enumerate(items, 1):
                key = (citation.source, citation.chunk)
                if key in seen:
                    continue
                seen.add(key)
                citations[key] = citation
                contribution = weight / (self.rrf_k + rank) if self.use_rrf else weight * score / maximum
                scores[key] = scores.get(key, 0) + contribution
        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [citations[key].model_copy(update={"score": scores[key]}) for key in ordered]

    def invalidate_bm25_cache(self) -> None:
        """Invalidate and clear all cached BM25 indexes and tokenized corpora."""
        self._corpus_cache.clear()
        self._bm25 = None
        self._chunk_texts = []
        self._chunk_to_idx.clear()
        logger.debug("BM25 index cache invalidated")

    def _rrf_fusion(
        self,
        vector_citations: List[Citation],
        bm25_scores: List[float],
        citation_map: Dict[Any, Citation],
    ) -> Dict[Any, float]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) for each retriever i

        Args:
            vector_citations: Vector search results
            bm25_scores: BM25 scores for indexed chunks
            citation_map: Mapping of (source, chunk) or chunk key to citation

        Returns:
            Dict of citation key to RRF score
        """
        rrf_scores: Dict[Any, float] = defaultdict(float)

        # Vector ranks
        for rank, citation in enumerate(vector_citations, 1):
            if (citation.source, citation.chunk) in citation_map:
                key = (citation.source, citation.chunk)
            elif citation.chunk[:200] in citation_map:
                key = citation.chunk[:200]
            else:
                key = (citation.source, citation.chunk)
            rrf_scores[key] += self.vector_weight / (self.rrf_k + rank)

        # BM25 ranks - only rank items that had positive match score
        indexed_bm25 = [(i, s) for i, s in enumerate(bm25_scores) if s > 0]
        indexed_bm25.sort(key=lambda x: x[1], reverse=True)

        for rank, (idx, score) in enumerate(indexed_bm25, 1):
            if idx < len(self._chunk_texts):
                chunk_text = self._chunk_texts[idx]
                for key in citation_map:
                    matched = (isinstance(key, tuple) and key[1] == chunk_text) or (
                        isinstance(key, str) and (key == chunk_text or key == chunk_text[:200])
                    )
                    if matched:
                        rrf_scores[key] += self.bm25_weight / (self.rrf_k + rank)

        return dict(rrf_scores)

    def _weighted_fusion(
        self,
        vector_citations: List[Citation],
        bm25_scores: List[float],
        citation_map: Dict[Any, Citation],
    ) -> Dict[Any, float]:
        """
        Combine results using weighted score fusion.

        Args:
            vector_citations: Vector search results
            bm25_scores: BM25 scores for indexed chunks
            citation_map: Mapping of (source, chunk) or chunk key to citation

        Returns:
            Dict of citation key to combined score
        """
        combined: Dict[Any, float] = {}

        # Normalize BM25 scores to 0-1
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            normalized_bm25 = [s / max_bm25 for s in bm25_scores]
        else:
            normalized_bm25 = []

        # Combine scores
        for citation in vector_citations:
            if (citation.source, citation.chunk) in citation_map:
                key = (citation.source, citation.chunk)
            elif citation.chunk[:200] in citation_map:
                key = citation.chunk[:200]
            else:
                key = (citation.source, citation.chunk)
            vector_score = citation.score * self.vector_weight

            # Find BM25 score for this chunk
            bm25_score = 0.0
            if citation.chunk in self._chunk_to_idx:
                idx = self._chunk_to_idx[citation.chunk]
                if idx < len(normalized_bm25):
                    bm25_score = normalized_bm25[idx] * self.bm25_weight
            elif citation.chunk[:200] in self._chunk_to_idx:
                idx = self._chunk_to_idx[citation.chunk[:200]]
                if idx < len(normalized_bm25):
                    bm25_score = normalized_bm25[idx] * self.bm25_weight

            combined[key] = vector_score + bm25_score

        return combined


class BM25Index:
    """
    Standalone BM25 keyword index that returns full Citations on its own.

    Unlike HybridSearcher (which reranks an existing set of vector results),
    this indexes chunks directly and answers queries without any embeddings -
    the "full-text search first" recipe: no vector store, no chunking-for-
    embedding concerns, sub-10ms lookups.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[Citation] = []

    def _tokenize(self, text: str) -> List[str]:
        import re
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if len(t) > 1]

    def index_chunks(self, chunks: List[Citation]) -> None:
        """
        Index chunks as Citations for standalone BM25 search.

        Args:
            chunks: Citations to index (source/chunk/metadata used, score ignored)
        """
        self._chunks.extend(chunks)
        tokenized_corpus = [self._tokenize(c.chunk) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        logger.info(f"BM25-only index now holds {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int = 10) -> List[Citation]:
        """Search the index and return top_k Citations with BM25 scores."""
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            source_chunk = self._chunks[idx]
            results.append(
                Citation(
                    source=source_chunk.source,
                    chunk=source_chunk.chunk,
                    score=scores[idx] / max_score,
                    metadata=source_chunk.metadata,
                )
            )
        return results

    def count(self) -> int:
        return len(self._chunks)

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source. Returns number deleted."""
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.source != source]
        deleted = before - len(self._chunks)
        if deleted:
            tokenized_corpus = [self._tokenize(c.chunk) for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        return deleted


def create_hybrid_searcher(
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    use_rrf: bool = True,
) -> HybridSearcher:
    """
    Factory function to create a hybrid searcher.

    Args:
        vector_weight: Weight for vector similarity (0-1)
        bm25_weight: Weight for BM25 (0-1)
        use_rrf: Use Reciprocal Rank Fusion

    Returns:
        Configured HybridSearcher
    """
    return HybridSearcher(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        use_rrf=use_rrf,
    )
