"""Retrieval and answer generation using OpenAI-compatible APIs."""

import logging
import os
import re
from typing import List, Optional, Set, Tuple

from openai import OpenAI
import openai

from .types import Answer, Citation
from .retry import retry_with_backoff

logger = logging.getLogger(__name__)


def reorder_u_shaped(citations: List[Citation]) -> List[Citation]:
    """Reorder citations into a U-shaped attention layout [c0, c2, c4, ..., c3, c1].

    Mitigates 'Lost in the Middle' attention degradation by placing highest-relevance
    citations at the beginning and end of the context window.

    Args:
        citations: List of citations sorted descending by relevance.

    Returns:
        Reordered list of citations.
    """
    if len(citations) <= 2:
        return list(citations)
    return citations[0::2] + citations[1::2][::-1]


def parse_citation_references(text: str, max_citations: int) -> List[int]:
    """Parse 1-indexed citation references from LLM-generated text.

    Extracts citation numbers in order of first appearance, bounded by [1, max_citations].

    Args:
        text: Generated response text
        max_citations: Total number of candidate citations in prompt context

    Returns:
        List of unique 1-indexed citation integers in order of first appearance.
    """
    if not text or max_citations <= 0:
        return []

    matches_with_pos: List[Tuple[int, int]] = []

    # 1. Bracketed expressions: [1], [2], [1, 2], [1-3], [^1], [Source 1], [Sources 1, 2]
    bracket_pattern = re.compile(
        r"\[\^?(?:Sources?|Ref|Reference)?\s*([0-9\s,\-\–\—and]+)\]",
        re.IGNORECASE,
    )
    for match in bracket_pattern.finditer(text):
        content = match.group(1)
        pos = match.start()
        # Range check: e.g. 1-3
        range_match = re.match(r"^\s*(\d+)\s*[\-\–\—]\s*(\d+)\s*$", content)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start <= end and (end - start) <= 20:
                for idx in range(start, end + 1):
                    matches_with_pos.append((pos, idx))
            continue
        # Split by commas or whitespace
        parts = re.split(r"[\s,]+(?:and\s+)?", content)
        for part in parts:
            part = part.strip()
            if part.isdigit():
                matches_with_pos.append((pos, int(part)))

    # 2. Textual expressions: "Source 1", "Source 2", "Sources 1 and 2"
    text_pattern = re.compile(r"\bSources?\s+([0-9\s,\-\–\—and]+)\b", re.IGNORECASE)
    for match in text_pattern.finditer(text):
        pos = match.start()
        # Avoid double-matching if preceded by [
        if pos > 0 and text[pos - 1] == "[":
            continue
        content = match.group(1)
        range_match = re.match(r"^\s*(\d+)\s*[\-\–\—]\s*(\d+)\s*$", content)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start <= end and (end - start) <= 20:
                for idx in range(start, end + 1):
                    matches_with_pos.append((pos, idx))
            continue
        parts = re.split(r"[\s,]+(?:and\s+)?", content)
        for part in parts:
            part = part.strip()
            if part.isdigit():
                matches_with_pos.append((pos, int(part)))

    # Sort matches by appearance in text
    matches_with_pos.sort(key=lambda x: x[0])

    referenced: List[int] = []
    seen: Set[int] = set()
    for _, idx in matches_with_pos:
        if 1 <= idx <= max_citations and idx not in seen:
            seen.add(idx)
            referenced.append(idx)

    return referenced


def prune_citations(
    text: str,
    citations: List[Citation],
) -> Tuple[List[Citation], List[Citation]]:
    """Prune unreferenced citations based on references parsed from text.

    Args:
        text: Generated answer text containing inline citations.
        citations: List of candidate Citation objects provided in context (1-indexed).

    Returns:
        Tuple of (active_citations, unreferenced_citations):
        - active_citations: Citations referenced in text (or all citations if none were cited).
        - unreferenced_citations: Citations that were provided in context but not cited.
    """
    if not citations:
        return [], []

    ref_indices = parse_citation_references(text, max_citations=len(citations))
    if not ref_indices:
        # Fall back cleanly to returning all candidate citations
        return list(citations), []

    ref_set = set(ref_indices)
    active: List[Citation] = []
    unreferenced: List[Citation] = []

    for idx, c in enumerate(citations, 1):
        c_copy = c.model_copy(deep=True)
        c_copy.metadata["citation_index"] = idx
        if idx in ref_set:
            active.append(c_copy)
        else:
            unreferenced.append(c_copy)

    return active, unreferenced


class Retriever:
    """Generate answers from retrieved chunks using OpenAI-compatible LLM APIs."""

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        enable_reranking: bool = True,
        enable_query_expansion: bool = True,
        llm_client = None,
        context_ordering: str = "u_shaped",
        prune_unreferenced_citations: bool = True,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            model: Model name to use (default: llama3.2 for Ollama)
            api_key: API key (optional for local models like Ollama)
            base_url: Base URL for OpenAI-compatible API (e.g., http://localhost:11434/v1 for Ollama)
            temperature: Sampling temperature (0.0-1.0, default: 0.1)
            enable_reranking: Enable reranking of results (default: True)
            enable_query_expansion: Enable query expansion for better retrieval (default: True)
            llm_client: Optional pre-configured LLM client
            context_ordering: Layout of citations in prompt context ("u_shaped" or "descending")
            prune_unreferenced_citations: Prune unreferenced candidate citations from answer
        """
        self.enable_reranking = enable_reranking
        self.enable_query_expansion = enable_query_expansion
        self.context_ordering = context_ordering
        self.prune_unreferenced_citations = prune_unreferenced_citations

        if llm_client is not None:
            self.llm_client = llm_client
            self.model = llm_client.model
            self.temperature = llm_client.temperature
            self.client = llm_client.client
        else:
            self.model = model
            self.temperature = temperature
            
            # Default to Ollama if no base_url provided
            if base_url is None:
                base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    
            # API key is optional for local models
            if api_key is None:
                api_key = os.getenv("LLM_API_KEY", "not-needed")
            
            from .llm_client import LLMClient
            self.llm_client = LLMClient(model=model, api_key=api_key, base_url=base_url, temperature=temperature)
            self.client = self.llm_client.client

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query into multiple variations for better retrieval.

        Args:
            query: Original query

        Returns:
            List of query variations including original
        """
        if not self.enable_query_expansion:
            return [query]

        try:
            expansion_prompt = f"""Given this question: "{query}"

Generate 2 alternative phrasings that preserve the same meaning but use different words.
Return only the alternatives, one per line, without numbering or explanation."""

            response = self.llm_client.complete(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rephrases questions."},
                    {"role": "user", "content": expansion_prompt},
                ],
                temperature=0.7,
                max_tokens=100,
            )

            alternatives = response.choices[0].message.content or ""
            variations = [query] + [line.strip() for line in alternatives.split('\n') if line.strip()]
            return variations[:3]  # Original + 2 alternatives

        except Exception as e:
            logger.debug(f"Query expansion failed: {e}")
            return [query]

    def rerank_citations(self, query: str, citations: List[Citation]) -> List[Citation]:
        """
        Rerank citations by relevance to query using simple scoring.

        Args:
            query: User's question
            citations: List of citations to rerank

        Returns:
            Reranked citations
        """
        if not self.enable_reranking or len(citations) <= 1:
            return citations

        try:
            from .reranker import TFIDFReranker
            reranker = TFIDFReranker()
            return reranker.rerank(query, citations)
        except ImportError:
            # Simple keyword-based reranking fallback
            query_terms = set(query.lower().split())

            def relevance_score(citation: Citation) -> float:
                chunk_terms = set(citation.chunk.lower().split())
                # Combine original vector score with keyword overlap
                keyword_overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
                # Weight: 70% vector similarity, 30% keyword overlap
                return 0.7 * citation.score + 0.3 * keyword_overlap

            # Rerank by combined score
            reranked = sorted(citations, key=relevance_score, reverse=True)
            return reranked

    def generate_answer(
        self,
        query: str,
        citations: List[Citation],
        system_prompt: Optional[str] = None,
        reorder: Optional[bool] = None,
    ) -> Answer:
        """
        Generate an answer from retrieved citations.

        Args:
            query: User's question
            citations: Retrieved citations
            system_prompt: Optional custom system prompt
            reorder: Optional override for context reordering. If None, uses self.context_ordering.

        Returns:
            Answer with citations
        """
        if not citations:
            return Answer(
                text="I couldn't find any relevant information to answer your question.",
                citations=[],
                query=query,
                unreferenced_citations=[],
            )

        # Rerank citations if enabled
        if self.enable_reranking:
            citations = self.rerank_citations(query, citations)

        # Apply U-shaped attention context ordering if enabled
        should_reorder = (
            reorder if reorder is not None else (getattr(self, "context_ordering", "u_shaped") == "u_shaped")
        )
        if should_reorder:
            ordered_citations = reorder_u_shaped(citations)
        else:
            ordered_citations = list(citations)

        # Build context from ordered citations
        context = self._build_context(ordered_citations)

        # Generate answer
        answer_text = self._generate_with_llm(query, context, system_prompt)

        # Attribution pruning
        if self.prune_unreferenced_citations:
            active_citations, unreferenced = prune_citations(answer_text, ordered_citations)
        else:
            active_citations = ordered_citations
            unreferenced = []

        return Answer(
            text=answer_text,
            citations=active_citations,
            query=query,
            unreferenced_citations=unreferenced,
        )

    def _build_context(self, citations: List[Citation]) -> str:
        """Build context string from citations."""
        context_parts = []

        for i, citation in enumerate(citations, 1):
            source_info = f"Source {i} ({citation.source}):"
            context_parts.append(f"{source_info}\n{citation.chunk}\n")

        return "\n".join(context_parts)

    def _generate_with_llm(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate answer using OpenAI-compatible API."""
        if system_prompt is None:
            system_prompt = (
                "You are an expert assistant. Answer questions based ONLY on the provided context sources. "
                "Your answer must be grounded in the given sources - do not add information from outside knowledge. "
                "Always cite sources using bracketed numbers like [1], [2] or 'Source 1', 'Source 2' corresponding to the source number. "
                "If the context lacks information to fully answer the question, explain what you found and what's missing. "
                "Be specific, detailed, and accurate in your responses."
            )

        user_prompt = f"""Context from documents:

{context}

Question: {query}

Please answer the question based on the context provided above. Cite your sources."""

        try:
            response = self.llm_client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {e}")
